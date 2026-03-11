"""Streamlit Chat UI for the Secure RAG System."""

import os

import requests
import streamlit as st

RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")

# Model options: (display_label, provider, model_id)
MODEL_OPTIONS = {
    "Gemini 2.5 Flash": ("gemini", "gemini-2.5-flash"),
    "OpenAI GPT-4o": ("openai", "gpt-4o"),
    "Anthropic Sonnet 4.5": ("anthropic", "claude-sonnet-4-5-20250929"),
    "Anthropic Haiku 4.5": ("anthropic", "claude-haiku-4-5-20251001"),
}


def get_headers() -> dict:
    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def check_health() -> dict | None:
    try:
        resp = requests.get(f"{RAG_API_URL}/api/v1/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except requests.ConnectionError:
        pass
    return None


def query_rag(question: str) -> dict | None:
    try:
        resp = requests.post(
            f"{RAG_API_URL}/api/v1/query",
            json={"question": question},
            headers=get_headers(),
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"answer": f"Error: {resp.status_code} - {resp.text}", "was_blocked": True}
    except requests.ConnectionError:
        return {"answer": "Could not connect to the RAG API. Is it running?", "was_blocked": True}


def upload_document(file) -> dict | None:
    try:
        resp = requests.post(
            f"{RAG_API_URL}/api/v1/documents/upload",
            files={"file": (file.name, file.getvalue(), "application/pdf")},
            headers=get_headers(),
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"message": f"Upload failed: {resp.text}"}
    except requests.ConnectionError:
        return {"message": "Could not connect to the RAG API."}


def switch_model(provider: str, model: str, api_key: str | None = None) -> dict | None:
    try:
        payload = {"provider": provider, "model": model}
        if api_key:
            payload["api_key"] = api_key
        resp = requests.post(
            f"{RAG_API_URL}/api/v1/switch-model",
            json=payload,
            headers=get_headers(),
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"message": resp.json().get("detail", resp.text), "error": True}
    except requests.ConnectionError:
        return {"message": "Could not connect to the RAG API.", "error": True}


def get_metrics() -> dict | None:
    try:
        resp = requests.get(
            f"{RAG_API_URL}/api/v1/metrics",
            headers=get_headers(),
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.ConnectionError:
        pass
    return None


# ============================================================================
# Streamlit App
# ============================================================================

st.set_page_config(
    page_title="Secure RAG System",
    page_icon="🔒",
    layout="wide",
)

st.title("Secure RAG System")
st.caption("Production-ready RAG with guardrails, prompt injection defense, and evaluation")

# --- Sidebar ---
with st.sidebar:
    st.header("System Status")

    health = check_health()
    if health:
        st.success("API: Connected")
        st.metric("Documents Indexed", health.get("document_count", 0))
        # Show current active model
        active_provider = health.get("llm_provider", "unknown")
        st.text(f"Active LLM: {active_provider}")
    else:
        st.error("API: Not connected")
        st.info(f"Expecting API at: {RAG_API_URL}")

    st.divider()

    # --- Model Selector ---
    st.header("Chat Model")

    selected_model_label = st.selectbox(
        "Select Model",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
        help="Choose which LLM to use for answering questions",
    )

    custom_api_key = st.text_input(
        "API Key (optional)",
        type="password",
        placeholder="Leave empty to use server default",
        help="Provide your own API key to test with a different account. If empty, the key from the server's .env is used.",
    )

    if st.button("Switch Model", use_container_width=True):
        provider, model_id = MODEL_OPTIONS[selected_model_label]
        with st.spinner(f"Switching to {selected_model_label}..."):
            result = switch_model(provider, model_id, custom_api_key or None)
        if result and not result.get("error"):
            st.success(f"Now using: {selected_model_label}")
            st.rerun()
        else:
            st.error(result.get("message", "Failed to switch model") if result else "Connection failed")

    st.divider()

    # --- Document Upload ---
    st.header("Upload Document")
    st.caption(
        "Upload a PDF to add it to the knowledge base. "
        "The file is split into chunks, embedded, and indexed "
        "into the vector store so the RAG system can retrieve "
        "relevant passages when answering questions."
    )
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file and st.button("Index Document"):
        with st.spinner("Uploading and indexing..."):
            result = upload_document(uploaded_file)
            if result:
                st.success(result.get("message", "Done"))
                st.json({
                    "pages": result.get("pages_loaded"),
                    "chunks": result.get("chunks_created"),
                    "indexed": result.get("chunks_indexed"),
                })

    st.divider()

    # Metrics
    st.header("Metrics")
    metrics = get_metrics()
    if metrics:
        col1, col2 = st.columns(2)
        col1.metric("Total Queries", metrics.get("total_queries", 0))
        col2.metric("Blocked", metrics.get("queries_blocked", 0))
        if metrics.get("avg_faithfulness") is not None:
            st.metric("Avg Faithfulness", f"{metrics['avg_faithfulness']:.2f}")
    else:
        st.text("No metrics available")

    st.divider()
    st.markdown("[Phoenix Traces](http://localhost:6006)")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]

            # Faithfulness indicator
            score = meta.get("faithfulness_score", -1)
            if score >= 0:
                color = "green" if score >= 0.8 else ("orange" if score >= 0.5 else "red")
                st.markdown(f"**Faithfulness:** :{color}[{score:.2f}]")

            # Sources
            if meta.get("sources"):
                with st.expander("Sources"):
                    for source in meta["sources"]:
                        st.text(source)

            # Security metadata
            if meta.get("guardrails_triggered") or meta.get("defenses_triggered"):
                with st.expander("Security Details"):
                    if meta.get("guardrails_triggered"):
                        st.text(f"Guardrails: {', '.join(meta['guardrails_triggered'])}")
                    if meta.get("defenses_triggered"):
                        st.text(f"Defenses: {', '.join(meta['defenses_triggered'])}")
                    if meta.get("error_codes"):
                        st.text(f"Error codes: {', '.join(meta['error_codes'])}")

# Chat input
if prompt := st.chat_input("Ask a question about Nova Scotia driving rules..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            result = query_rag(prompt)

        if result:
            st.markdown(result["answer"])

            metadata = {
                "sources": result.get("sources", []),
                "faithfulness_score": result.get("faithfulness_score", -1),
                "guardrails_triggered": result.get("guardrails_triggered", []),
                "defenses_triggered": result.get("defenses_triggered", []),
                "error_codes": result.get("error_codes", []),
                "was_blocked": result.get("was_blocked", False),
            }

            score = metadata["faithfulness_score"]
            if score >= 0:
                color = "green" if score >= 0.8 else ("orange" if score >= 0.5 else "red")
                st.markdown(f"**Faithfulness:** :{color}[{score:.2f}]")

            if metadata["sources"]:
                with st.expander("Sources"):
                    for source in metadata["sources"]:
                        st.text(source)

            if metadata["guardrails_triggered"] or metadata["defenses_triggered"]:
                with st.expander("Security Details"):
                    if metadata["guardrails_triggered"]:
                        st.text(f"Guardrails: {', '.join(metadata['guardrails_triggered'])}")
                    if metadata["defenses_triggered"]:
                        st.text(f"Defenses: {', '.join(metadata['defenses_triggered'])}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "metadata": metadata,
            })
        else:
            error_msg = "Failed to get a response from the API."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
