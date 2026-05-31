"""
rag.py  –  NovaCart RAG Retrieval Module
Handles semantic search over the ChromaDB collection and structured shipment lookup.
"""

import os
import re
import json

import chromadb
from chromadb.utils import embedding_functions

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR  = os.path.join(BASE_DIR, "chroma_db")
COLLECTION  = "novacart_knowledge"
EMBED_MODEL = "all-MiniLM-L6-v2"
SHIPS_JSON  = os.path.join(BASE_DIR, "shipments.json")

_client = None
_col    = None
_ships  = None


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
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        out.append({"text": doc, "source": meta["source"], "distance": dist})
    return out


SOURCE_LABELS = {
    "company_profile": "Company Profile",
    "qa_knowledge":    "Customer Q&A",
    "shipments":       "Shipment Database",
}

# Topics the chatbot is allowed to answer about
ALLOWED_TOPICS = [
    "shipment", "order", "delivery", "track", "status", "return",
    "refund", "exchange", "payment", "novacart", "novaplus", "seller",
    "warranty", "cancel", "address", "wallet", "courier", "invoice",
    "product", "grocery", "fashion", "electronics", "membership",
    "coupon", "cash on delivery", "account", "support", "nvc",
]

RELEVANCE_THRESHOLD = 1.2   # ChromaDB cosine distance; lower = more similar


def is_relevant(query: str, chunks: list[dict]) -> bool:
    """Return True if the query looks on-topic."""
    q_lower = query.lower()
    # Fast keyword check
    if any(kw in q_lower for kw in ALLOWED_TOPICS):
        return True
    # Fall back to embedding distance
    if chunks and chunks[0]["distance"] < RELEVANCE_THRESHOLD:
        return True
    return False


def format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        label = SOURCE_LABELS.get(c["source"], c["source"])
        parts.append(f"[{label}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)
