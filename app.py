"""
app.py  –  NovaCart Customer Support Chatbot
Run with:  streamlit run app.py
Requires:  python ingest.py  to have been run first.
"""

import os
import random
import ollama
import streamlit as st

from rag import (
    retrieve,
    retrieve_context,
    rerank,
    is_relevant,
    format_context,
    extract_shipment_id,
    lookup_shipment,
    augment_query,
    update_conversation_memory,
    SOURCE_LABELS,
    SYSTEM_PROMPT,
    OFF_TOPIC_RESPONSE,
    RELEVANCE_THRESHOLD,
    RERANK_CANDIDATES,
    RERANK_TOP_N,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NovaCart Support",
    page_icon="🛒",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Brand header */
.brand-header {
    background: linear-gradient(135deg, #0052cc 0%, #00a3e0 100%);
    color: white;
    padding: 18px 24px;
    border-radius: 12px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-title { font-size: 1.6rem; font-weight: 700; margin: 0; }
.brand-sub   { font-size: 0.9rem; opacity: 0.85; margin: 0; }

/* Source badge */
.src-badge {
    display: inline-block;
    background: #e8f4fd;
    color: #0052cc;
    border: 1px solid #b3d7f5;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.72rem;
    margin: 2px 3px 2px 0;
    font-weight: 600;
}

/* Shipment card */
.ship-card {
    background: #f0faf5;
    border-left: 4px solid #00875a;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
}
.ship-status {
    font-size: 1.0rem;
    font-weight: 700;
    color: #00875a;
}

/* Off-topic notice */
.offtrack {
    background: #fff8e1;
    border-left: 4px solid #f4a900;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
AVAILABLE_MODELS = ["qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b"]
RETRIEVAL_METHODS = {
    "vector": "Vector",
    "bm25": "BM25",
    "hybrid": "Hybrid",
}

SUGGESTION_TEMPLATES: dict[str, list[str]] = {
    "shipment": [
        "What do I do if my shipment is delayed?",
        "How do I reschedule my delivery?",
        "Can I change my delivery address?",
    ],
    "delivery": [
        "What are NovaCart's delivery time windows?",
        "What happens if I miss my delivery?",
        "Can I request express delivery?",
    ],
    "return": [
        "How long does a refund take to process?",
        "Can I exchange an item instead of returning it?",
        "Which items are not eligible for return?",
    ],
    "payment": [
        "How do I add money to my NovaWallet?",
        "Is cash on delivery available in my area?",
        "How long does a card refund take?",
    ],
    "membership": [
        "What benefits does NovaPlus include?",
        "How do I cancel my NovaPlus subscription?",
        "Can I share my NovaPlus membership?",
    ],
    "hr": [
        "How many days of annual leave do I get?",
        "What is the remote work policy?",
        "How do I submit a leave request?",
    ],
}


def _rewrite_query(query: str, history: list[dict], model: str) -> str:
    """LLM fallback: rewrite a referential query as a standalone question."""
    recent_text = "\n".join(
        f"{h['role'].upper()}: {h['content'][:300]}"
        for h in history[-4:]
    )
    prompt = (
        f"Conversation so far:\n{recent_text}\n\n"
        f"Latest message: \"{query}\"\n\n"
        f"Rewrite the latest message as a complete standalone question in one sentence, "
        f"resolving any pronouns or vague references using the conversation above. "
        f"Output only the rewritten question, nothing else."
    )
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        options={"temperature": 0.0},
    )
    return resp["message"]["content"].strip()


def _generate_suggestions_llm(answer: str, model: str) -> list[str]:
    """Ask the LLM to suggest 3 follow-up questions when no template matches."""
    prompt = (
        f"A NovaCart customer support bot just gave this answer:\n{answer[:500]}\n\n"
        f"Generate exactly 3 short follow-up questions a customer might ask next, "
        f"related to NovaCart services. Each on a separate line. No numbering or bullets."
    )
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        options={"temperature": 0.3},
    )
    lines = [l.strip() for l in resp["message"]["content"].splitlines() if l.strip()]
    return lines[:3]


def _get_suggestions(topics: list[str], answer: str, model: str) -> list[str]:
    """Return 3 suggestion chips: template-based when topic is known, LLM otherwise."""
    for topic in topics:
        if topic in SUGGESTION_TEMPLATES:
            return random.sample(SUGGESTION_TEMPLATES[topic], min(3, len(SUGGESTION_TEMPLATES[topic])))
    try:
        return _generate_suggestions_llm(answer, model)
    except Exception:
        return []


def _render_suggestions_ui(suggestions: list[str], key_prefix: str = "s") -> None:
    """Render suggestion chips inside a chat message block."""
    if not suggestions:
        return
    st.markdown(
        "<p style='font-size:0.8rem;color:#888;margin:10px 0 4px;'>💬 <b>You might also ask:</b></p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for i, sug in enumerate(suggestions[:3]):
        with cols[i]:
            if st.button(sug, use_container_width=True, key=f"{key_prefix}_{hash(sug)}"):
                st.session_state.pending_query = sug
                st.session_state.suggestions = []
                st.rerun()


# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []        # [{role, content}]
if "model" not in st.session_state:
    st.session_state.model = AVAILABLE_MODELS[1]   # default 7b
if "shipment_regex_enabled" not in st.session_state:
    st.session_state.shipment_regex_enabled = True
if "relevance_threshold" not in st.session_state:
    st.session_state.relevance_threshold = RELEVANCE_THRESHOLD
if "reranking_enabled" not in st.session_state:
    st.session_state.reranking_enabled = False
if "retrieval_method" not in st.session_state:
    st.session_state.retrieval_method = "vector"
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = {"shipment_ids": [], "active_shipment": None, "topics": []}
if "last_retrieval_tier" not in st.session_state:
    st.session_state.last_retrieval_tier = "passthrough"
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shopping-cart.png", width=60)
    st.title("NovaCart Support")
    st.caption("Powered by Qwen + RAG")

    st.divider()

    st.subheader("Model")
    selected_model = st.selectbox(
        "Qwen model",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(st.session_state.model),
        label_visibility="collapsed",
    )
    st.session_state.model = selected_model

    with st.expander("Settings"):
        st.session_state.shipment_regex_enabled = st.toggle(
            "Shipment regex search",
            value=st.session_state.shipment_regex_enabled,
            help="Detect NVC######## IDs and run an exact shipment lookup before RAG.",
        )
        st.caption("When on, messages containing an NVC######## ID use exact shipment lookup before RAG.")
        st.session_state.retrieval_method = st.radio(
            "Retrieval method",
            options=list(RETRIEVAL_METHODS.keys()),
            index=list(RETRIEVAL_METHODS.keys()).index(st.session_state.retrieval_method),
            format_func=lambda mode: RETRIEVAL_METHODS[mode],
            horizontal=True,
            help="Choose how candidate chunks are retrieved before optional reranking.",
        )
        st.caption("Vector uses embeddings; BM25 uses keyword overlap; hybrid combines both.")
        st.session_state.reranking_enabled = st.toggle(
            f"Reranking (top-{RERANK_CANDIDATES} -> top-{RERANK_TOP_N})",
            value=st.session_state.reranking_enabled,
            help="Retrieve more chunks, score them with a cross-encoder, and pass only the best ones to Qwen.",
        )
        st.caption("Uses a cross-encoder to reorder retrieved chunks before answering.")
        st.session_state.relevance_threshold = st.slider(
            "Relevance threshold",
            min_value=0.1,
            max_value=2.0,
            value=float(st.session_state.relevance_threshold),
            step=0.05,
            help="Lower values make off-topic filtering stricter; higher values make it more permissive.",
        )
        st.caption("Lower means stricter off-topic filtering; higher means more permissive.")

    with st.expander("Session Memory"):
        mem = st.session_state.conversation_memory
        tier_labels = {
            "passthrough":      "⬜ Passthrough",
            "entity_injection": "🟢 Entity injection",
            "llm_rewrite":      "🟡 LLM rewrite",
        }
        tier = st.session_state.last_retrieval_tier
        st.caption(f"Last retrieval: **{tier_labels.get(tier, tier)}**")
        if mem["shipment_ids"]:
            st.caption(f"**Shipments:** {', '.join(mem['shipment_ids'])}")
        if mem["active_shipment"]:
            s = mem["active_shipment"]
            st.caption(
                f"**Active:** {s.get('Shipment ID', '—')} · "
                f"{s.get('Status', '—')} · "
                f"{s.get('Destination City', '—')}"
            )
        if mem["topics"]:
            st.caption(f"**Topics:** {', '.join(mem['topics'])}")
        if not mem["shipment_ids"] and not mem["topics"]:
            st.caption("No entities tracked yet.")

    st.divider()
    st.subheader("Topics I can help with")
    topics = [
        "📦 Order tracking",
        "🚚 Delivery & shipping",
        "🔄 Returns & refunds",
        "💳 Payments",
        "🌟 NovaPlus membership",
        "🛒 Products & sellers",
        "👤 Account & checkout",
    ]
    for t in topics:
        st.markdown(f"- {t}")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.conversation_memory = {"shipment_ids": [], "active_shipment": None, "topics": []}
        st.session_state.last_retrieval_tier = "passthrough"
        st.session_state.suggestions = []
        st.rerun()

    st.caption(f"Model: `{st.session_state.model}`")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="brand-header">
  <div>
    <p class="brand-title">🛒 NovaCart Support</p>
    <p class="brand-sub">Hi! I'm Nova — ask me about your orders, deliveries, returns, and more.</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Chat history display ──────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.history):
    with st.chat_message(msg["role"], avatar="🛒" if msg["role"] == "assistant" else "🧑"):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if (
            msg["role"] == "assistant"
            and i == len(st.session_state.history) - 1
            and st.session_state.suggestions
        ):
            _render_suggestions_ui(st.session_state.suggestions, key_prefix="h")


# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask me about your order, delivery, return policy...")

if not user_input and "pending_query" in st.session_state:
    user_input = st.session_state.pending_query
    del st.session_state.pending_query

if user_input:
    user_input = user_input.strip()
    st.session_state.suggestions = []

    # Display user message
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant", avatar="🛒"):
        response_placeholder = st.empty()
        sources_placeholder  = st.empty()

        # ── Step 1: Shipment ID exact lookup ─────────────────────────────────
        shipment_id  = None
        shipment_rec = None
        if st.session_state.shipment_regex_enabled:
            shipment_id = extract_shipment_id(user_input)
        if shipment_id:
            shipment_rec = lookup_shipment(shipment_id)

        # ── Step 1.5: Augment query with session memory ───────────────────────
        retrieval_query, tier = augment_query(user_input, st.session_state.conversation_memory)
        if tier == "needs_llm_rewrite" and len(st.session_state.history) > 1:
            try:
                retrieval_query = _rewrite_query(
                    user_input,
                    st.session_state.history[:-1],
                    st.session_state.model,
                )
                tier = "llm_rewrite"
            except Exception:
                retrieval_query = user_input
                tier = "passthrough"
        st.session_state.last_retrieval_tier = tier

        # ── Step 2: Context retrieval ─────────────────────────────────────────
        retrieval_k = RERANK_CANDIDATES if st.session_state.reranking_enabled else 5
        chunks = retrieve_context(
            retrieval_query,
            mode=st.session_state.retrieval_method,
            k=retrieval_k,
        )

        # The relevance threshold is based on vector distance, so keep the
        # off-topic gate anchored to vector retrieval across all retrieval modes.
        relevance_chunks = (
            chunks
            if st.session_state.retrieval_method == "vector"
            else retrieve(retrieval_query, k=5)
        )

        # ── Step 3: Relevance gate ────────────────────────────────────────────
        if not is_relevant(
            retrieval_query,
            relevance_chunks,
            threshold=st.session_state.relevance_threshold,
        ):
            response_placeholder.markdown(
                f'<div class="offtrack">{OFF_TOPIC_RESPONSE}</div>',
                unsafe_allow_html=True,
            )
            st.session_state.history.append(
                {"role": "assistant", "content": OFF_TOPIC_RESPONSE}
            )

        else:
            # ── Step 4: Optional reranking and context build ──────────────────
            if st.session_state.reranking_enabled:
                with st.spinner("Reranking retrieved chunks..."):
                    chunks = rerank(user_input, chunks, top_n=RERANK_TOP_N)

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
                f"CUSTOMER QUESTION: {user_input}"
            )

            # ── Step 5: Build message history for Ollama ─────────────────────
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            # Include last 6 turns for context
            # this could be parametrised or improved with a smarter selection strategy (e.g. include all turns since last shipment mention)
            for h in st.session_state.history[:-1][-6:]:
                messages.append({"role": h["role"], "content": h["content"]})
            messages.append({"role": "user", "content": rag_prompt})

            # ── Step 6: Stream response ───────────────────────────────────────
            full_response = ""
            try:
                stream = ollama.chat(
                    model    = st.session_state.model,
                    messages = messages,
                    stream   = True,
                )
                for chunk in stream:
                    token = chunk["message"]["content"]
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

            except Exception as e:
                err_msg = f"Sorry, I encountered an error connecting to the model: `{e}`\n\nPlease make sure Ollama is running and `{st.session_state.model}` is pulled."
                response_placeholder.warning(err_msg)
                full_response = err_msg

            # ── Step 7: Source badges ─────────────────────────────────────────
            used_sources = list({c["source"] for c in chunks})
            if shipment_rec:
                used_sources.insert(0, "shipments")
            used_sources = list(dict.fromkeys(used_sources))   # deduplicate, preserve order

            if used_sources:
                badges = " ".join(
                    f'<span class="src-badge">📄 {SOURCE_LABELS.get(s, s)}</span>'
                    for s in used_sources
                )
                sources_placeholder.markdown(
                    f"<div style='margin-top:4px;'>Sources: {badges}</div>",
                    unsafe_allow_html=True,
                )

            # ── Step 8: Store clean assistant reply ───────────────────────────
            st.session_state.history.append(
                {"role": "assistant", "content": full_response}
            )

            # ── Step 10: Suggestion chips ─────────────────────────────────────
            if not full_response.startswith("Sorry, I encountered an error"):
                suggestions = _get_suggestions(
                    st.session_state.conversation_memory["topics"],
                    full_response,
                    st.session_state.model,
                )
                st.session_state.suggestions = suggestions
                _render_suggestions_ui(suggestions, key_prefix="g")

        # ── Step 9: Update conversation memory ───────────────────────────────
        update_conversation_memory(
            st.session_state.conversation_memory,
            user_input,
            shipment_rec,
        )
