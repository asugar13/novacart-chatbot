# NovaCart Customer Support Chatbot

A RAG-powered customer support chatbot for NovaCart Marketplace, built with Streamlit + Qwen (via Ollama) + ChromaDB.

---

## Project Structure

```
novacart_chatbot/
├── app.py              ← Streamlit chatbot UI
├── escalation.py       ← Frustration detection + simulated human handoff
├── ingest.py           ← Document ingestion & indexing pipeline
├── rag.py              ← Retrieval logic (semantic search + shipment lookup)
├── requirements.txt    ← Python dependencies
├── data/               ← Put the source documents here
│   ├── NovaCart_Company_Profile.docx
│   ├── NovaCart_Customer_QA_Knowledge_Base.docx
│   ├── NovaCart_HR_Policies.docx
│   └── NovaCart_Shipment_Status_Database.xlsx
├── chroma_db/          ← Auto-created by ingest.py
├── shipments.json      ← Auto-created by ingest.py (fast exact lookup)
├── evals/              ← LLM-as-judge evaluation harness
│   ├── eval.py
│   └── dataset.yaml
└── tests/
    └── test_escalation.py
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
>
> The first customer message analysed by the Emotion or Hybrid handoff detector
> will also download `emotion-english-distilroberta-base`. It is cached locally
> for later runs.

### 3. Place documents in `data/`

Copy the NovaCart files into the `data/` folder (already done if you received this project pre-packaged).

### 4. Run ingestion (one-time setup)

```bash
python ingest.py
```

This will:
- Parse all available source documents
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
| **Shipment ID lookup** | Detects `NVC########` pattern — exact match against shipment database before semantic search |
| **Order ID lookup** | Detects `ORD-YYYY-######` pattern — reverse-indexes the shipment database to resolve Order IDs to shipment records |
| **Semantic RAG** | ChromaDB + `all-MiniLM-L6-v2` embeddings retrieve the most relevant knowledge base chunks |
| **BM25 retrieval** | Keyword-based lexical search as an alternative to or complement of vector search |
| **Hybrid retrieval** | Combines vector and BM25 results via Reciprocal Rank Fusion (RRF) |
| **Cross-encoder reranking** | Optional second-stage reranking with `ms-marco-MiniLM-L-6-v2` — retrieves more candidates, keeps the best |
| **Stateful conversation memory** | Tracks active shipment IDs, Order IDs, and topic thread across turns; injects them into follow-up queries |
| **Entity-aware query augmentation** | Two-tier cascade: (1) fast entity injection when session context resolves references; (2) LLM rewrite fallback only when needed |
| **HR policy support** | Retrieval over employee HR policies — leave, attendance, payroll, conduct |
| **Relevance gate** | Off-topic questions get a polite redirect; threshold is configurable in the sidebar |
| **Source badges** | Each answer shows which document(s) it was sourced from |
| **Suggested follow-up questions** | After each answer, 3 clickable chips appear — template-based for known topics, LLM-generated otherwise |
| **Frustration-aware handoff** | Detects explicit human requests, strong anger/disgust, and repeated frustration; offers a support-agent button |
| **Hybrid escalation detector** | Uses rules first, a local DistilRoBERTa emotion model second, and Qwen only for borderline or fallback cases |
| **Simulated support ticket** | Packages the handoff reason, active shipment/order, and recent user messages without claiming a real ticket was submitted |
| **Clear chat** | One-click reset clears history, session memory, and suggestion state |
| **Session memory panel** | Sidebar expander showing active shipment, tracked IDs, topic thread, and which retrieval tier fired last |
| **LLM-as-judge evaluation** | `evals/eval.py` scores the pipeline on a YAML dataset using automated faithfulness and relevance metrics |

---

## How It Works

```
User message
     │
     ├─► Detect human request / frustration
     │       ├─ Rules → explicit request or strong language
     │       ├─ Emotion model → anger / disgust scores
     │       └─ Borderline → Qwen structured classification
     │
     ├─► Extract NVC######## Shipment ID? → Exact lookup in shipments.json
     │       └─ Not found? Extract ORD-YYYY-###### Order ID → Reverse index lookup
     │
     ├─► Augment query with session memory (entity injection or LLM rewrite)
     │
     ├─► Retrieve context chunks (Vector / BM25 / Hybrid)
     │       └─ Optional: cross-encoder reranking
     │
     ├─► Relevance gate (keyword + embedding distance check)
     │        │
     │        ├─ Off-topic → polite redirect message
     │        │
     │        └─ On-topic → build RAG prompt with context + shipment record
     │
     ├─► Stream Qwen response via Ollama
     │
     ├─► Show source badges
     │
     ├─► High frustration? → Offer simulated human-support handoff
     │
     └─► Otherwise generate suggested follow-up question chips
             └─ Update session memory (shipment IDs, topics)
```

---

## Sidebar Controls

| Control | Description |
|---|---|
| **Model** | Select the Qwen model size (3b / 7b / 14b) |
| **Shipment regex search** | Toggle exact ID lookup before RAG (default: on) |
| **Retrieval method** | Vector, BM25, or Hybrid |
| **Reranking** | Enable cross-encoder second-stage reranking |
| **Relevance threshold** | Slider to tune off-topic filtering strictness |
| **Human handoff detection** | Enable or disable frustration and escalation-intent detection |
| **Handoff detector** | Compare Hybrid, Rules only, Emotion model, or Qwen classifier |
| **Session Memory** | Expander showing tracked entities and last retrieval tier |
| **Escalation Monitor** | Shows the latest signal, frustration streak, detector, and prepared ticket |

---

## Evaluation

The `evals/` folder contains a standalone evaluation harness:

```bash
python evals/eval.py
```

- Loads test cases from `evals/dataset.yaml`
- Runs each question through the full RAG pipeline
- Scores answers using an LLM-as-judge prompt (faithfulness, relevance, correctness)
- Outputs a summary report

---

## Customisation

- **CHUNK_SIZE / CHUNK_OVERLAP** in `ingest.py` — adjust for longer/shorter contexts
- **RELEVANCE_THRESHOLD** in `rag.py` — raise to be more permissive, lower to be stricter
- **SYSTEM_PROMPT** in `rag.py` — tune the chatbot's persona and instructions
- **k=5** in `retrieve()` calls — change number of retrieved chunks
- **SUGGESTION_TEMPLATES** in `app.py` — edit the template-based follow-up chips per topic
- **Threshold constants** in `escalation.py` — tune anger, disgust, and medium-signal thresholds

---

## Human Handoff Behaviour

The handoff feature is intentionally based on **customer frustration**, not
generic negative sentiment. For example, “My parcel is delayed” describes a
negative situation but does not automatically trigger escalation.

The default Hybrid mode follows this cascade:

1. Explicit requests such as “connect me to an agent” trigger immediately.
2. Strong frustration phrases use a fast rules path.
3. A local emotion model checks anger and disgust.
4. Two consecutive moderate-frustration turns trigger a handoff offer.
5. Qwen handles borderline cases or acts as fallback if the emotion model is unavailable.

The support button creates an in-session demonstration payload only. Connecting
it to Zendesk, Salesforce, email, or another real ticketing system would require
an external API integration.
