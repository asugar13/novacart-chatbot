"""
evals/eval_escalation.py  -  NovaCart Escalation (human-handoff) Eval Harness

Runs the escalation detector against a hand-authored, labelled dataset across the
four detection modes (hybrid / rules / emotion / qwen) and scores the binary
"should the bot offer a human handoff?" decision.

Unlike the RAG eval, the detector returns a structured `should_offer` boolean, so
predictions are compared to the gold label DIRECTLY — there is no LLM-as-judge.
Positive class = the bot SHOULD escalate:

    TP  gold escalate = true  AND offered handoff
    FN  gold escalate = true  BUT did not offer      (missed escalation — costly)
    FP  gold escalate = false BUT offered handoff     (false alarm)
    TN  gold escalate = false AND did not offer

The Markdown report covers per-mode precision/recall/F1/accuracy, a recall+F1
breakdown by category, detector attribution (which sub-detector fired), latency,
and a per-failure analysis.

Run:
    python -m evals.eval_escalation                          # all four modes
    python -m evals.eval_escalation --mode rules --limit 5   # smoke test, no Ollama
    python -m evals.eval_escalation --fresh                  # ignore cached results
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

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

from escalation import (  # noqa: E402
    DETECTION_MODES,
    assess_escalation,
    get_emotion_scores,
)

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "escalation_dataset.yaml"
REPORTS_DIR = ROOT / "reports"
RAW_RESULTS = REPORTS_DIR / "escalation_raw_results.jsonl"
REPORT_MD = REPORTS_DIR / "escalation_results.md"

MODEL = os.environ.get("EVAL_ESCALATION_MODEL", "qwen2.5:7b")

MODES = list(DETECTION_MODES)  # ["hybrid", "rules", "emotion", "qwen"]

VALID_LABELS = {"TP", "FP", "FN", "TN"}

# Order categories appear in the report (positives first, then negatives).
CATEGORY_ORDER = [
    "explicit-request",
    "strong-frustration",
    "high-emotion",
    "trend-frustration",
    "contextual-frustration",
    "calm-negative",
    "normal-query",
    "polite",
]


# ─── Data class ──────────────────────────────────────────────────────────────

@dataclass
class Result:
    case_id: str
    mode: str
    category: str
    message: str
    gold_escalate: bool
    predicted_offer: bool
    label: str
    detector: str
    level: str
    confidence: float
    streak: int
    reason: str
    latency_sec: float
    model_error: str = ""


def to_label(gold: bool, pred: bool) -> str:
    """Confusion-matrix entry for one prediction (positive class = escalate)."""
    if gold and pred:
        return "TP"
    if gold and not pred:
        return "FN"
    if not gold and pred:
        return "FP"
    return "TN"


# ─── Runner ──────────────────────────────────────────────────────────────────

def run_case(case: dict, mode: str) -> Result:
    """Assess one case under one mode and package the result."""
    message = case["message"]
    context = case.get("context", []) or []
    previous_streak = int(case.get("previous_streak", 0) or 0)
    gold = bool(case["gold_escalate"])

    start = time.time()
    assessment = assess_escalation(
        message,
        context,
        mode=mode,
        model=MODEL,
        previous_streak=previous_streak,
    )
    latency = time.time() - start

    pred = bool(assessment.should_offer)
    return Result(
        case_id=case["id"],
        mode=mode,
        category=case["category"],
        message=message,
        gold_escalate=gold,
        predicted_offer=pred,
        label=to_label(gold, pred),
        detector=assessment.detector,
        level=assessment.level,
        confidence=round(float(assessment.confidence), 3),
        streak=assessment.streak,
        reason=assessment.reason,
        latency_sec=latency,
        model_error=assessment.model_error or "",
    )


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(results: list) -> dict:
    """
    Confusion-matrix-first metrics on the escalate-vs-not binary.
        precision = TP / (TP + FP)
        recall    = TP / (TP + FN)
        F1        = 2PR / (P+R)
        accuracy  = (TP + TN) / N
    sklearn metrics on the same binary (1 = escalate) are reported as a cross-check.
    """
    labels = [r.label for r in results]
    n = len(labels)
    tp = labels.count("TP")
    fp = labels.count("FP")
    fn = labels.count("FN")
    tn = labels.count("TN")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / n if n else 0.0

    # sklearn cross-check on the escalate-vs-not binary (1 = should/did escalate)
    y_true = [1 if r.gold_escalate else 0 for r in results]
    y_pred = [1 if r.predicted_offer else 0 for r in results]
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


# ─── Reporter ────────────────────────────────────────────────────────────────

def write_report(all_results: list, warmup_ok: bool = True) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    by_mode: dict[str, list] = {}
    for r in all_results:
        by_mode.setdefault(r.mode, []).append(r)
    # Keep the canonical mode order where present.
    ordered_modes = [m for m in MODES if m in by_mode] + [
        m for m in by_mode if m not in MODES
    ]

    cats = [c for c in CATEGORY_ORDER if any(r.category == c for r in all_results)]
    cats += sorted({r.category for r in all_results} - set(cats))

    lines = []
    lines.append("# NovaCart Escalation Evaluation Report\n")
    lines.append(f"- Qwen model (qwen/hybrid modes): `{MODEL}`")
    n_cases = len(all_results) // max(len(by_mode), 1)
    lines.append(f"- Total runs: {len(all_results)} ({len(by_mode)} modes x {n_cases} cases)")
    lines.append(
        "- Positive class = the bot **should escalate** (offer a human handoff). "
        "**FN = missed escalation (costly); FP = false alarm (annoyance).**"
    )
    if warmup_ok:
        lines.append(
            "- Latency excludes the one-time emotion-model load (warmed up before timing)."
        )
    else:
        lines.append(
            "- ⚠️ Emotion-model warm-up failed, so the first emotion/hybrid case's "
            "latency includes the one-time model load — treat those averages with care."
        )
    lines.append("")

    # 1. Overall metrics by mode
    lines.append("## 1. Overall metrics by mode\n")
    lines.append("| Mode | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy | Avg latency (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mode in ordered_modes:
        rs = by_mode[mode]
        m = compute_metrics(rs)
        avg_lat = sum(r.latency_sec for r in rs) / len(rs) if rs else 0
        lines.append(
            f"| {DETECTION_MODES.get(mode, mode)} | {m['TP']} | {m['FP']} | {m['FN']} | {m['TN']} "
            f"| {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['accuracy']:.3f} | {avg_lat:.3f} |"
        )
    lines.append("")

    # 2. Recall by category and mode (where the modes diverge)
    lines.append("## 2. Recall by category and mode\n")
    lines.append(
        "> Recall on the **should-escalate** categories (how many true escalations each "
        "mode catches) and, for the negative categories, the **TN rate** = correctly NOT "
        "escalating. `—` means the category has no cases of that polarity.\n"
    )
    header = "| Mode | " + " | ".join(cats) + " |"
    sep = "|---|" + "|".join(["---:"] * len(cats)) + "|"
    lines.append(header)
    lines.append(sep)
    for mode in ordered_modes:
        rs = by_mode[mode]
        row = [DETECTION_MODES.get(mode, mode)]
        for cat in cats:
            cat_rs = [r for r in rs if r.category == cat]
            if not cat_rs:
                row.append("—")
                continue
            positives = [r for r in cat_rs if r.gold_escalate]
            if positives:  # recall = caught / should-have-caught
                caught = sum(1 for r in positives if r.predicted_offer)
                row.append(f"{caught / len(positives):.2f}")
            else:  # TN rate = correctly did not escalate
                correct = sum(1 for r in cat_rs if not r.predicted_offer)
                row.append(f"{correct / len(cat_rs):.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # 3. Detector attribution — which sub-detector fired on the cases each mode escalated
    lines.append("## 3. Detector attribution (on escalated cases)\n")
    lines.append(
        "> Of the cases each mode chose to escalate, which sub-detector fired. The regex "
        "**rules** pre-step runs in every mode; this shows how much it carries vs. what the "
        "emotion model / trend / Qwen add.\n"
    )
    detectors = ["rules", "emotion", "trend", "qwen"]
    lines.append("| Mode | escalated | " + " | ".join(detectors) + " | other |")
    lines.append("|---|---:|" + "|".join(["---:"] * (len(detectors) + 1)) + "|")
    for mode in ordered_modes:
        offered = [r for r in by_mode[mode] if r.predicted_offer]
        counts = {d: sum(1 for r in offered if r.detector == d) for d in detectors}
        other = len(offered) - sum(counts.values())
        row = [DETECTION_MODES.get(mode, mode), str(len(offered))]
        row += [str(counts[d]) for d in detectors]
        row.append(str(other))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # 4. Failure cases (FP and FN)
    lines.append("## 4. Failure cases (FP and FN)\n")
    failures = [r for r in all_results if r.label in ("FP", "FN")]
    if not failures:
        lines.append("_None — all cases were classified TP or TN._")
    else:
        lines.append(f"_{len(failures)} failure cases across all modes._\n")
        # Group by mode for readability.
        for mode in ordered_modes:
            mode_fail = [r for r in failures if r.mode == mode]
            if not mode_fail:
                continue
            lines.append(f"### Mode: {DETECTION_MODES.get(mode, mode)}\n")
            for r in mode_fail:
                kind = "missed escalation" if r.label == "FN" else "false alarm"
                lines.append(f"- **`{r.case_id}`** ({r.category}) · **{r.label}** ({kind})")
                lines.append(f"  - **Message:** {r.message}")
                lines.append(
                    f"  - gold_escalate=`{r.gold_escalate}`, offered=`{r.predicted_offer}`, "
                    f"detector=`{r.detector}`, level=`{r.level}`, confidence=`{r.confidence}`"
                )
                lines.append(f"  - **Reason:** {r.reason}")
                if r.model_error:
                    lines.append(f"  - **Model error:** {r.model_error}")
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
    parser.add_argument("--mode", type=str, default=None,
                        help="Run only one mode (hybrid|rules|emotion|qwen)")
    parser.add_argument("--model", type=str, default=None,
                        help="Ollama model for qwen/hybrid modes (default qwen2.5:7b)")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore escalation_raw_results.jsonl and re-run everything")
    args = parser.parse_args()

    global MODEL
    if args.model:
        MODEL = args.model

    REPORTS_DIR.mkdir(exist_ok=True)
    dataset = load_dataset()
    if args.limit:
        dataset = dataset[:args.limit]

    modes = MODES
    if args.mode:
        if args.mode not in DETECTION_MODES:
            sys.exit(f"Unknown mode: {args.mode} (choose from {', '.join(MODES)})")
        modes = [args.mode]

    # Resume: skip (case_id, mode) pairs already in the JSONL.
    done = set()
    if RAW_RESULTS.exists() and not args.fresh:
        for line in RAW_RESULTS.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done.add((row["case_id"], row["mode"]))
        print(f"Resuming — {len(done)} (case, mode) pairs already complete.")
    elif RAW_RESULTS.exists():
        RAW_RESULTS.unlink()

    todo = [
        (case, mode) for mode in modes for case in dataset
        if (case["id"], mode) not in done
    ]
    print(f"Running {len(todo)} (case, mode) combinations "
          f"[{len(dataset)} cases x {len(modes)} modes, {len(done)} already done]")

    # Warm up the emotion model once so per-case latency reflects steady-state
    # inference, not the one-time DistilRoBERTa download/load. Track whether it
    # actually succeeded so the report's latency disclaimer stays honest.
    warmup_ok = True
    if any(m in {"emotion", "hybrid"} for m in modes):
        warmup_ok = False
        try:
            warmup_ok = bool(get_emotion_scores("warm up"))
        except Exception as e:
            print(f"Emotion-model warm-up failed (will surface per case): {e}")
        if not warmup_ok:
            print("Warning: emotion-model warm-up returned no scores; the first "
                  "emotion/hybrid case latency may include the model load.")

    with RAW_RESULTS.open("a") as fh:
        for case, mode in tqdm(todo, desc="escalation-eval"):
            try:
                result = run_case(case, mode)
            except Exception as e:
                print(f"\nError on {case['id']} / {mode}: {e}")
                continue
            fh.write(json.dumps(asdict(result)) + "\n")
            fh.flush()

    # Read everything back and report.
    all_rows = []
    for line in RAW_RESULTS.read_text().splitlines():
        if not line.strip():
            continue
        all_rows.append(json.loads(line))

    all_results = [Result(**row) for row in all_rows]
    write_report(all_results, warmup_ok=warmup_ok)


if __name__ == "__main__":
    main()
