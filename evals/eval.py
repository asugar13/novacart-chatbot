"""
evals/eval.py  -  NovaCart RAG Evaluation Harness

Runs the full chatbot pipeline against a hand-authored dataset across the 6
retrieval-toggle configurations (vector / BM25 / hybrid x rerank on/off), uses
a local Qwen model as LLM-as-judge to classify each answer as TP/FP/FN/TN,
and writes a single Markdown report with confusion matrices, per-config F1 /
precision / recall / accuracy, retrieval P@k / R@k, and a per-failure analysis.

Run:
    python -m evals.eval                              # full sweep
    python -m evals.eval --limit 3 --config vector    # smoke test
    python -m evals.eval --no-judge                   # retrieval metrics only
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import ollama
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import (  # noqa: E402
    extract_shipment_id,
    format_context,
    is_relevant,
    lookup_shipment,
    rerank,
    retrieve,
    retrieve_context,
    OFF_TOPIC_RESPONSE,
    RELEVANCE_THRESHOLD,
    RERANK_CANDIDATES,
    RERANK_TOP_N,
    SYSTEM_PROMPT,
)

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset.yaml"
REPORTS_DIR = ROOT / "reports"
RAW_RESULTS = REPORTS_DIR / "raw_results.jsonl"
REPORT_MD = REPORTS_DIR / "results.md"
JUDGE_CACHE = REPORTS_DIR / "judge_cache.json"

ANSWER_MODEL = os.environ.get("EVAL_ANSWER_MODEL", "qwen2.5:7b")
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "qwen2.5:7b")

CONFIGS = [
    {"name": "vector",           "mode": "vector", "rerank": False},
    {"name": "vector+rerank",    "mode": "vector", "rerank": True},
    {"name": "bm25",             "mode": "bm25",   "rerank": False},
    {"name": "bm25+rerank",      "mode": "bm25",   "rerank": True},
    {"name": "hybrid",           "mode": "hybrid", "rerank": False},
    {"name": "hybrid+rerank",    "mode": "hybrid", "rerank": True},
]

VALID_LABELS = {"TP", "FP", "FN", "TN"}


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class PipelineOutput:
    answer: str
    refused: bool
    retrieved_sources: list
    shipment_hit: bool
    latency_sec: float


@dataclass
class Result:
    case_id: str
    config: str
    category: str
    subcategory: str
    question: str
    in_scope: bool
    gold_facts: list
    gold_source: str
    bot_answer: str
    refused: bool
    retrieved_sources: list
    shipment_hit: bool
    latency_sec: float
    judge_label: str = ""
    judge_reason: str = ""


# ─── Pipeline runner ─────────────────────────────────────────────────────────

def run_pipeline(question: str, config: dict) -> PipelineOutput:
    """Mirror app.py's pipeline flow without Streamlit."""
    start = time.time()

    shipment_id = extract_shipment_id(question)
    shipment_rec = lookup_shipment(shipment_id) if shipment_id else None
    shipment_hit = shipment_rec is not None

    retrieval_k = RERANK_CANDIDATES if config["rerank"] else 5
    chunks = retrieve_context(question, mode=config["mode"], k=retrieval_k)

    relevance_chunks = (
        chunks if config["mode"] == "vector" else retrieve(question, k=5)
    )

    if not is_relevant(question, relevance_chunks, threshold=RELEVANCE_THRESHOLD):
        return PipelineOutput(
            answer=OFF_TOPIC_RESPONSE,
            refused=True,
            retrieved_sources=list({c["source"] for c in chunks}),
            shipment_hit=shipment_hit,
            latency_sec=time.time() - start,
        )

    if config["rerank"]:
        chunks = rerank(question, chunks, top_n=RERANK_TOP_N)

    context_parts = []
    if shipment_rec:
        ship_block = (
            f"SHIPMENT RECORD (exact match for {shipment_id}):\n"
            + "\n".join(f"  {k}: {v}" for k, v in shipment_rec.items())
        )
        context_parts.append(ship_block)
    context_parts.append(format_context(chunks))
    full_context = "\n\n".join(context_parts)

    rag_prompt = (
        f"Use the following knowledge base excerpts to answer the customer question.\n\n"
        f"CONTEXT:\n{full_context}\n\n"
        f"CUSTOMER QUESTION: {question}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": rag_prompt},
    ]

    response = ollama.chat(model=ANSWER_MODEL, messages=messages, stream=False)
    answer = response["message"]["content"]

    retrieved_sources = list({c["source"] for c in chunks})
    if shipment_rec:
        retrieved_sources.insert(0, "shipments")
    retrieved_sources = list(dict.fromkeys(retrieved_sources))

    return PipelineOutput(
        answer=answer,
        refused=False,
        retrieved_sources=retrieved_sources,
        shipment_hit=shipment_hit,
        latency_sec=time.time() - start,
    )


# ─── Judge ───────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an evaluator for a customer support chatbot. Classify the BOT ANSWER into exactly ONE label.

QUESTION: {question}
GOLD FACTS (key info the answer should contain, if any): {gold_facts}
IN-SCOPE: {in_scope}   (true = the bot SHOULD answer; false = the bot SHOULD refuse)
BOT ANSWER: {bot_answer}

Labels:
- TP: in_scope is true AND the bot answer is correct (mentions / matches the gold facts)
- FN: in_scope is true BUT the bot refused, said "I don't know", or gave wrong / missing info
- TN: in_scope is false AND the bot correctly refused or declined to answer
- FP: in_scope is false BUT the bot still produced an answer instead of refusing

Reply with ONE LINE of JSON only: {{"label": "TP|FP|FN|TN", "reason": "<one short sentence>"}}"""


def _judge_cache_key(question: str, bot_answer: str) -> str:
    return hashlib.sha1(f"{question}||{bot_answer}".encode()).hexdigest()


def _load_judge_cache() -> dict:
    if JUDGE_CACHE.exists():
        return json.loads(JUDGE_CACHE.read_text())
    return {}


def _save_judge_cache(cache: dict) -> None:
    JUDGE_CACHE.write_text(json.dumps(cache, indent=2))


def judge(
    question: str,
    bot_answer: str,
    gold_facts: list,
    in_scope: bool,
    cache: dict,
) -> tuple[str, str]:
    key = _judge_cache_key(question, bot_answer)
    if key in cache:
        return cache[key]["label"], cache[key]["reason"]

    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_facts=", ".join(gold_facts) if gold_facts else "(none — question is unanswerable)",
        in_scope="true" if in_scope else "false",
        bot_answer=bot_answer,
    )

    response = ollama.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        options={"temperature": 0.0},
    )
    raw = response["message"]["content"].strip()

    label, reason = _parse_judge(raw)
    cache[key] = {"label": label, "reason": reason}
    return label, reason


def _parse_judge(raw: str) -> tuple[str, str]:
    # Try strict JSON first
    try:
        obj = json.loads(raw)
        label = obj.get("label", "").upper()
        reason = obj.get("reason", "")
        if label in VALID_LABELS:
            return label, reason
    except json.JSONDecodeError:
        pass

    # Fallback: regex pluck
    m = re.search(r'"label"\s*:\s*"(TP|FP|FN|TN)"', raw)
    label = m.group(1) if m else ""
    m2 = re.search(r'"reason"\s*:\s*"([^"]*)"', raw)
    reason = m2.group(1) if m2 else raw[:200]

    if label not in VALID_LABELS:
        # Last-ditch: look for bare TP/FP/FN/TN in the output
        m3 = re.search(r'\b(TP|FP|FN|TN)\b', raw)
        label = m3.group(1) if m3 else "FN"  # default to FN on parse failure
        reason = f"[parse fallback] {reason}"

    return label, reason


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(results: list) -> dict:
    """
    Binary classifier framing:
      positive class = "the bot correctly handled this question"
                       (i.e. judge label is TP for in-scope or TN for OOS)
      Predicted positive = the bot's actual handling was correct
      Predicted negative = the bot's handling was wrong (FP or FN)
      True positive class in this binary mapping:
          y_true = 1 if in_scope else 0 ... too ambiguous.

    Cleaner framing for the assignment: treat each case label directly as a
    confusion-matrix entry. Then derive:
        precision = TP / (TP + FP)
        recall    = TP / (TP + FN)
        F1        = 2PR / (P+R)
        accuracy  = (TP + TN) / N
    We also report sklearn metrics on the binary "answered correctly" view
    (1 = TP or TN, 0 = FP or FN) to confirm.
    """
    labels = [r.judge_label for r in results]
    n = len(labels)
    tp = labels.count("TP")
    fp = labels.count("FP")
    fn = labels.count("FN")
    tn = labels.count("TN")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / n if n else 0.0

    # sklearn cross-check on "answered correctly" binary
    y_true = [1 if r.in_scope else 0 for r in results]
    y_pred = [
        1 if r.judge_label == "TP" else 0 if r.judge_label == "TN"
        else (0 if r.in_scope else 1)  # FN keeps y_pred=0; FP keeps y_pred=1
        for r in results
    ]
    sk_acc = accuracy_score(y_true, y_pred) if n else 0.0
    sk_prec = precision_score(y_true, y_pred, zero_division=0) if n else 0.0
    sk_rec = recall_score(y_true, y_pred, zero_division=0) if n else 0.0
    sk_f1 = f1_score(y_true, y_pred, zero_division=0) if n else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist() if n else [[0, 0], [0, 0]]

    return {
        "n": n,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "sklearn": {
            "accuracy": sk_acc,
            "precision": sk_prec,
            "recall": sk_rec,
            "f1": sk_f1,
            "confusion_matrix": cm,
        },
    }


def compute_retrieval_metrics(results: list) -> dict:
    """Coarse source-level P@k / R@k. Each case has at most one gold source."""
    in_scope = [r for r in results if r.in_scope and r.gold_source != "none"]
    if not in_scope:
        return {"n": 0, "precision_at_k": 0.0, "recall_at_k": 0.0, "shipment_hit_rate": 0.0}

    precisions = []
    recalls = []
    for r in in_scope:
        srcs = r.retrieved_sources
        if r.gold_source in srcs:
            precisions.append(1.0 / max(len(srcs), 1))
            recalls.append(1.0)
        else:
            precisions.append(0.0)
            recalls.append(0.0)

    ship_cases = [r for r in results if r.category == "shipment-lookup" and r.in_scope]
    ship_hit = sum(1 for r in ship_cases if r.shipment_hit) / len(ship_cases) if ship_cases else 0.0

    return {
        "n": len(in_scope),
        "precision_at_k": sum(precisions) / len(precisions),
        "recall_at_k": sum(recalls) / len(recalls),
        "shipment_hit_rate": ship_hit,
    }


# ─── Reporter ────────────────────────────────────────────────────────────────

def write_report(all_results: list) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    by_config: dict[str, list] = {}
    for r in all_results:
        by_config.setdefault(r.config, []).append(r)

    lines = []
    lines.append("# NovaCart RAG Evaluation Report\n")
    lines.append(f"- Answer model: `{ANSWER_MODEL}`")
    lines.append(f"- Judge model:  `{JUDGE_MODEL}`")
    lines.append(f"- Total runs: {len(all_results)} ({len(by_config)} configs x {len(all_results)//max(len(by_config),1)} cases)")
    lines.append("")

    # 1. Per-config overall metrics
    lines.append("## 1. Overall metrics by config\n")
    lines.append("| Config | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy | Avg latency (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cfg_name, rs in by_config.items():
        m = compute_metrics(rs)
        avg_lat = sum(r.latency_sec for r in rs) / len(rs) if rs else 0
        lines.append(
            f"| {cfg_name} | {m['TP']} | {m['FP']} | {m['FN']} | {m['TN']} "
            f"| {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['accuracy']:.3f} | {avg_lat:.1f} |"
        )
    lines.append("")

    # 2. Per-category breakdown (F1 only, easy to scan)
    lines.append("## 2. F1 by category and config\n")
    cats = ["shipment-lookup", "generic", "out-of-scope"]
    header = "| Config | " + " | ".join(cats) + " |"
    sep = "|---|" + "|".join(["---:"] * len(cats)) + "|"
    lines.append(header)
    lines.append(sep)
    for cfg_name, rs in by_config.items():
        row = [cfg_name]
        for cat in cats:
            cat_rs = [r for r in rs if r.category == cat]
            row.append(f"{compute_metrics(cat_rs)['f1']:.3f}" if cat_rs else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # 3. Retrieval metrics
    lines.append("## 3. Retrieval quality\n")
    lines.append("| Config | P@k | R@k | Shipment hit rate |")
    lines.append("|---|---:|---:|---:|")
    for cfg_name, rs in by_config.items():
        m = compute_retrieval_metrics(rs)
        lines.append(
            f"| {cfg_name} | {m['precision_at_k']:.3f} | {m['recall_at_k']:.3f} | {m['shipment_hit_rate']:.3f} |"
        )
    lines.append("")
    lines.append("> P@k is the fraction of retrieved sources that match the gold source (coarse — source-label match, not chunk-level). Shipment hit rate is the share of in-scope shipment cases where the regex extracted a valid ID that hit `shipments.json`.\n")

    # 4. Failures
    lines.append("## 4. Failure cases (FP and FN)\n")
    failures = [r for r in all_results if r.judge_label in ("FP", "FN")]
    if not failures:
        lines.append("_None — all cases were classified TP or TN._")
    else:
        lines.append(f"_{len(failures)} failure cases across all configs._\n")
        for r in failures:
            lines.append(f"### `{r.case_id}` · `{r.config}` · **{r.judge_label}**")
            lines.append(f"- **Q:** {r.question}")
            lines.append(f"- **Gold facts:** {', '.join(r.gold_facts) if r.gold_facts else '(none, refusal expected)'}")
            lines.append(f"- **Bot answer:** {r.bot_answer[:300]}{'...' if len(r.bot_answer) > 300 else ''}")
            lines.append(f"- **Judge reason:** {r.judge_reason}")
            lines.append("")

    REPORT_MD.write_text("\n".join(lines))
    print(f"\nReport written to {REPORT_MD}")


# ─── Main loop ───────────────────────────────────────────────────────────────

def load_dataset() -> list:
    return yaml.safe_load(DATASET_PATH.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of cases (for smoke tests)")
    parser.add_argument("--config", type=str, default=None,
                        help="Run only one config by name (e.g. 'vector')")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip the judge step (retrieval-only run)")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore raw_results.jsonl and re-run everything")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(exist_ok=True)
    dataset = load_dataset()
    if args.limit:
        dataset = dataset[:args.limit]

    configs = CONFIGS
    if args.config:
        configs = [c for c in CONFIGS if c["name"] == args.config]
        if not configs:
            sys.exit(f"Unknown config: {args.config}")

    # Resume: skip (case_id, config) pairs already in raw_results.jsonl
    done = set()
    existing_rows = []
    if RAW_RESULTS.exists() and not args.fresh:
        for line in RAW_RESULTS.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done.add((row["case_id"], row["config"]))
            existing_rows.append(row)
        print(f"Resuming — {len(done)} (case, config) pairs already complete.")
    else:
        if RAW_RESULTS.exists():
            RAW_RESULTS.unlink()

    judge_cache = _load_judge_cache()

    todo = [
        (case, cfg) for cfg in configs for case in dataset
        if (case["id"], cfg["name"]) not in done
    ]
    print(f"Running {len(todo)} (case, config) combinations "
          f"[{len(dataset)} cases x {len(configs)} configs, {len(done)} already done]")

    with RAW_RESULTS.open("a") as fh:
        for case, cfg in tqdm(todo, desc="eval"):
            try:
                out = run_pipeline(case["question"], cfg)
            except Exception as e:
                print(f"\nPipeline error on {case['id']} / {cfg['name']}: {e}")
                continue

            label, reason = "", ""
            if not args.no_judge:
                try:
                    label, reason = judge(
                        case["question"],
                        out.answer,
                        case.get("gold_facts", []),
                        case["in_scope"],
                        judge_cache,
                    )
                except Exception as e:
                    print(f"\nJudge error on {case['id']} / {cfg['name']}: {e}")
                    label, reason = "FN", f"[judge error] {e}"

            result = Result(
                case_id=case["id"],
                config=cfg["name"],
                category=case["category"],
                subcategory=case.get("subcategory", ""),
                question=case["question"],
                in_scope=case["in_scope"],
                gold_facts=case.get("gold_facts", []),
                gold_source=case.get("gold_source", "none"),
                bot_answer=out.answer,
                refused=out.refused,
                retrieved_sources=out.retrieved_sources,
                shipment_hit=out.shipment_hit,
                latency_sec=out.latency_sec,
                judge_label=label,
                judge_reason=reason,
            )
            fh.write(json.dumps(asdict(result)) + "\n")
            fh.flush()
            _save_judge_cache(judge_cache)

    # Read everything back and report
    all_rows = []
    for line in RAW_RESULTS.read_text().splitlines():
        if not line.strip():
            continue
        all_rows.append(json.loads(line))

    all_results = [Result(**row) for row in all_rows]
    write_report(all_results)


if __name__ == "__main__":
    main()
