"""
rag.py  –  NovaCart RAG Retrieval Module
Handles semantic search over the ChromaDB collection and structured shipment lookup.
"""

import os
import re
import json
import math
from collections import Counter

import chromadb
from chromadb.utils import embedding_functions

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR  = os.path.join(BASE_DIR, "chroma_db")
COLLECTION  = "novacart_knowledge"
EMBED_MODEL = "all-MiniLM-L6-v2"
SHIPS_JSON  = os.path.join(BASE_DIR, "shipments.json")
RERANK_CANDIDATES = 10
RERANK_TOP_N = 3

_client = None
_col    = None
_ships  = None
_cross_encoder = None
_bm25_index = None


def _get_collection():
    global _client, _col
    if _col is None:
        ef      = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        _col    = _client.get_collection(COLLECTION, embedding_function=ef)
    return _col


def _get_shipments() -> dict:
    global _ships
    if _ships is None:
        if os.path.exists(SHIPS_JSON):
            with open(SHIPS_JSON) as f:
                _ships = json.load(f)
        else:
            _ships = {}
    return _ships


def extract_shipment_id(text: str) -> str | None:
    """Pull NVC########  from user message if present."""
    m = re.search(r"\bNVC\d{8}\b", text.upper())
    return m.group(0) if m else None


def lookup_shipment(shipment_id: str) -> dict | None:
    return _get_shipments().get(shipment_id.upper())


def retrieve(query: str, k: int = 5) -> list[dict]:
    """Semantic search; returns list of {text, source, distance}."""
    col = _get_collection()
    results = col.query(
        query_texts = [query],
        n_results   = k,
        include     = ["documents", "metadatas", "distances"],
    )
    out = []
    for chunk_id, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        out.append({"id": chunk_id, "text": doc, "source": meta["source"], "distance": dist})
    return out


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _get_bm25_index() -> dict:
    """Build a lightweight BM25 index from the ChromaDB documents."""
    global _bm25_index
    if _bm25_index is not None:
        return _bm25_index

    col = _get_collection()
    results = col.get(include=["documents", "metadatas"])
    docs = results["documents"]
    metas = results["metadatas"]
    ids = results["ids"]

    tokenized_docs = [_tokenize(doc) for doc in docs]
    term_counts = [Counter(tokens) for tokens in tokenized_docs]
    doc_freq = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))

    total_docs = len(docs)
    avg_doc_len = (
        sum(len(tokens) for tokens in tokenized_docs) / total_docs
        if total_docs
        else 0
    )
    idf = {
        term: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in doc_freq.items()
    }

    _bm25_index = {
        "docs": docs,
        "metas": metas,
        "ids": ids,
        "term_counts": term_counts,
        "doc_lengths": [len(tokens) for tokens in tokenized_docs],
        "avg_doc_len": avg_doc_len,
        "idf": idf,
    }
    return _bm25_index


def retrieve_bm25(query: str, k: int = 5) -> list[dict]:
    """Lexical BM25 search; returns list of {text, source, bm25_score}."""
    index = _get_bm25_index()
    query_terms = _tokenize(query)
    if not query_terms or not index["docs"]:
        return []

    k1 = 1.5
    b = 0.75
    scores = []
    avg_doc_len = index["avg_doc_len"] or 1

    for i, counts in enumerate(index["term_counts"]):
        score = 0.0
        doc_len = index["doc_lengths"][i] or 1
        for term in query_terms:
            freq = counts.get(term, 0)
            if not freq:
                continue
            denom = freq + k1 * (1 - b + b * doc_len / avg_doc_len)
            score += index["idf"].get(term, 0.0) * (freq * (k1 + 1) / denom)
        if score > 0:
            scores.append((score, i))

    ranked = sorted(scores, reverse=True)[:k]
    return [
        {
            "id": index["ids"][i],
            "text": index["docs"][i],
            "source": index["metas"][i]["source"],
            "bm25_score": float(score),
        }
        for score, i in ranked
    ]


def _chunk_key(chunk: dict) -> str:
    return chunk.get("id") or f"{chunk.get('source', '')}:{hash(chunk.get('text', ''))}"


def reciprocal_rank_fusion(
    vector_chunks: list[dict],
    bm25_chunks: list[dict],
    rrf_k: int = 60,
) -> list[dict]:
    """Merge vector and BM25 rankings using reciprocal rank fusion."""
    scores = {}
    chunks_by_key = {}

    for result_list in (vector_chunks, bm25_chunks):
        for rank, chunk in enumerate(result_list, start=1):
            key = _chunk_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1 / (rrf_k + rank)
            if key not in chunks_by_key:
                chunks_by_key[key] = chunk.copy()
            else:
                chunks_by_key[key].update(chunk)

    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    fused = []
    for key in ranked_keys:
        item = chunks_by_key[key].copy()
        item["rrf_score"] = scores[key]
        fused.append(item)
    return fused


def retrieve_hybrid(query: str, k: int = 5) -> list[dict]:
    """Combine vector and BM25 retrieval with reciprocal rank fusion."""
    vector_chunks = retrieve(query, k=k)
    bm25_chunks = retrieve_bm25(query, k=k)
    return reciprocal_rank_fusion(vector_chunks, bm25_chunks)[:k]


def retrieve_context(query: str, mode: str = "vector", k: int = 5) -> list[dict]:
    """Retrieve context chunks with the selected first-stage method."""
    if mode == "vector":
        return retrieve(query, k=k)
    if mode == "bm25":
        return retrieve_bm25(query, k=k)
    if mode == "hybrid":
        return retrieve_hybrid(query, k=k)
    raise ValueError(f"Unknown retrieval mode: {mode}")


def get_cross_encoder():
    """Lazy-load the cross-encoder used for optional second-stage reranking."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers.cross_encoder import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def rerank(query: str, chunks: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """Rerank retrieved chunks by scoring each query/chunk pair jointly."""
    if not chunks:
        return []

    ce = get_cross_encoder()
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = ce.predict(pairs)

    ranked = []
    for score, chunk in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True):
        item = chunk.copy()
        item["rerank_score"] = float(score)
        ranked.append(item)
    return ranked[:top_n]


SOURCE_LABELS = {
    "company_profile": "Company Profile",
    "qa_knowledge":    "Customer Q&A",
    "hr_policies":     "HR Policies",
    "shipments":       "Shipment Database",
}

SYSTEM_PROMPT = """You are Nova, the friendly and knowledgeable customer support assistant for NovaCart Marketplace, a digital marketplace operating across the Middle East.

Your role is to help customers with questions about:
- Order tracking and shipment status
- Delivery options, timelines, and rescheduling
- Returns, refunds, and exchanges
- Payment methods and NovaWallet
- NovaPlus membership
- Seller marketplace policies
- Product categories and purchasing policies
- Customer account and checkout questions
- NovaCart employee HR policies (leave, attendance, payroll, conduct, and workplace procedures)

Guidelines:
- Be friendly, concise, and helpful.
- Base your answers ONLY on the context provided below.
- When a shipment ID is involved, use the exact status data provided.
- If you are unsure about something, say so honestly and suggest the customer contact live support.
- Do NOT answer questions unrelated to NovaCart products, services, or policies.
- Never reveal that you are powered by an AI language model unless directly asked.
- Do not invent shipment statuses, refund timelines, or policies not in the context.
"""

OFF_TOPIC_RESPONSE = """I appreciate you reaching out! I'm Nova, NovaCart's support assistant, and I'm specifically here to help you with topics related to NovaCart Marketplace, such as:

- 📦 **Order tracking and shipment status**
- 🚚 **Delivery options and timelines**
- 🔄 **Returns, refunds, and exchanges**
- 💳 **Payment methods and NovaWallet**
- 🌟 **NovaPlus membership**
- 🛒 **Products, sellers, and policies**

I'm not able to help with topics outside of NovaCart. Please feel free to ask me anything about your orders or our services!"""

# Topics the chatbot is allowed to answer about
ALLOWED_TOPICS = [
    "shipment", "order", "delivery", "track", "status", "return",
    "refund", "exchange", "payment", "novacart", "novaplus", "seller",
    "warranty", "cancel", "address", "wallet", "courier", "invoice",
    "product", "grocery", "fashion", "electronics", "membership",
    "coupon", "cash on delivery", "account", "support", "nvc",
    "hr", "employee", "leave", "vacation", "sick leave", "parental",
    "payroll", "compensation", "attendance", "remote work", "work from home",
    "probation", "performance review", "disciplinary", "grievance",
]

RELEVANCE_THRESHOLD = 1.2   # ChromaDB cosine distance; lower = stricter


def is_relevant(
    query: str,
    chunks: list[dict],
    threshold: float = RELEVANCE_THRESHOLD,
) -> bool:
    """Return True if the query looks on-topic."""
    q_lower = query.lower()
    # Fast keyword check
    if any(kw in q_lower for kw in ALLOWED_TOPICS):
        return True
    # Fall back to embedding distance
    if chunks and chunks[0]["distance"] < threshold:
        return True
    return False


def format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        label = SOURCE_LABELS.get(c["source"], c["source"])
        parts.append(f"[{label}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)
