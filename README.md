# NovaCart Customer Support Chatbot

A RAG-powered customer support chatbot for NovaCart Marketplace, built with Streamlit + Qwen (via Ollama) + ChromaDB.

---

## Project Structure

```
novacart_chatbot/
├── app.py              ← Streamlit chatbot UI
├── ingest.py           ← Document ingestion & indexing pipeline
├── rag.py              ← Retrieval logic (semantic search + shipment lookup)
├── requirements.txt    ← Python dependencies
├── data/               ← Put the 3 source documents here
│   ├── NovaCart_Company_Profile.docx
│   ├── NovaCart_Customer_QA_Knowledge_Base.docx
│   └── NovaCart_Shipment_Status_Database.xlsx
├── chroma_db/          ← Auto-created by ingest.py
└── shipments.json      ← Auto-created by ingest.py (fast exact lookup)
```

---

## Setup

### 1. Prerequisites

- **Python 3.10+**
- **Ollama** running locally with at least one Qwen model pulled:
  ```bash
  ollama pull qwen2.5:7b
  # optional extras:
  ollama pull qwen2.5:3b
  ollama pull qwen2.5:14b
  ```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> The first run of `ingest.py` will download the `all-MiniLM-L6-v2` embedding model (~90 MB) from HuggingFace automatically.

### 3. Place documents in `data/`

Copy the three NovaCart files into the `data/` folder (already done if you received this project pre-packaged).

### 4. Run ingestion (one-time setup)

```bash
python ingest.py
```

This will:
- Parse all 3 documents
- Chunk and embed the text
- Store everything in `chroma_db/`
- Export `shipments.json` for fast ID lookups

### 5. Launch the chatbot

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Features

| Feature | Details |
|---|---|
| **Model selector** | Switch between `qwen2.5:3b`, `qwen2.5:7b`, `qwen2.5:14b` in the sidebar |
| **Shipment ID lookup** | Detects `NVC########` pattern, does exact match before semantic search |
| **Semantic RAG** | ChromaDB + `all-MiniLM-L6-v2` embeddings retrieve the most relevant chunks |
| **Relevance gate** | Off-topic questions get a polite redirect instead of a hallucinated answer |
| **Source badges** | Each answer shows which document(s) it came from |
| **Conversation memory** | Last 6 turns kept in context for follow-up questions |
| **Clear chat** | One-click reset from the sidebar |

---

## How It Works

```
User message
     │
     ├─► Extract NVC######## ID? → Exact shipment lookup (shipments.json)
     │
     ├─► Semantic retrieval from ChromaDB (top-5 chunks)
     │
     ├─► Relevance check (keyword + embedding distance)
     │        │
     │        ├─ Off-topic → polite redirect message
     │        │
     │        └─ On-topic → build RAG prompt with context
     │
     └─► Stream Qwen response via Ollama
```

---

## Customisation

- **CHUNK_SIZE / CHUNK_OVERLAP** in `ingest.py` — adjust for longer/shorter contexts
- **RELEVANCE_THRESHOLD** in `rag.py` — raise to be more permissive, lower to be stricter
- **SYSTEM_PROMPT** in `app.py` — tune the chatbot's persona and instructions
- **k=5** in `retrieve()` calls — change number of retrieved chunks
