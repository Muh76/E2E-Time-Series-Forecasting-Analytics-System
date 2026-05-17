"""
RAG Documentation Chat — ask questions about the forecasting system.

Calls POST /api/v1/chat/query (vector search over docs + optional OpenAI reply).
The index must be built first: python scripts/generate_rag_index.py
"""

import requests
import streamlit as st
from components.api import chat_query, describe_request_error
from components.ui import render_error, render_warning

_INDEX_NOT_BUILT_HINT = (
    "The RAG index has not been built yet. "
    "Run `python scripts/generate_rag_index.py` from the project root, "
    "then restart the API."
)

_EXAMPLE_QUESTIONS = [
    "What features does the LightGBM model use?",
    "How does recursive multi-step forecasting work?",
    "What API endpoints are available?",
    "How is data leakage prevented during training?",
    "What does the ETL pipeline do?",
    "What metrics are used to evaluate the model?",
]


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Retrieved sources ({len(sources)})"):
        for s in sources:
            score_pct = f"{s.get('score', 0) * 100:.1f}%"
            st.markdown(f"- **{s.get('header', '')}** — `{s.get('source', '')}` _(similarity: {score_pct})_")


def main() -> None:
    st.title("RAG Documentation Chat")
    st.markdown(
        "Ask anything about the forecasting system — architecture, API, features, training, "
        "ETL, or model behaviour. Answers are grounded in the project documentation."
    )
    render_warning("This assistant explains the system; it does not generate forecasts.")
    st.markdown("---")

    # Example questions as quick-fill buttons
    st.markdown("**Example questions**")
    cols = st.columns(3)
    chosen_example: str | None = None
    for i, q in enumerate(_EXAMPLE_QUESTIONS):
        if cols[i % 3].button(q, key=f"example_{i}"):
            chosen_example = q

    st.markdown("---")

    # Chat history stored in session state
    if "rag_history" not in st.session_state:
        st.session_state["rag_history"] = []

    # User input — pre-fill from example button if one was clicked
    default_val = chosen_example or ""
    message = st.text_area(
        "Your question",
        value=default_val,
        placeholder="e.g. How does the recursive forecasting work?",
        height=90,
        key="rag_input",
    )

    col_send, col_clear = st.columns([1, 5])
    send = col_send.button("Ask", type="primary")
    if col_clear.button("Clear history"):
        st.session_state["rag_history"] = []
        st.rerun()

    if send:
        query = message.strip()
        if not query:
            render_warning("Please enter a question before clicking Ask.")
        else:
            with st.spinner("Searching documentation…"):
                try:
                    result = chat_query(query)
                except requests.exceptions.HTTPError as exc:
                    if exc.response is not None and exc.response.status_code == 503:
                        render_error(_INDEX_NOT_BUILT_HINT)
                    else:
                        render_error(describe_request_error(exc))
                    result = None
                except requests.RequestException as exc:
                    render_error(describe_request_error(exc))
                    result = None

            if result:
                st.session_state["rag_history"].append(
                    {
                        "question": query,
                        "reply": result.get("reply", ""),
                        "sources": result.get("sources", []),
                        "generated_at": result.get("generated_at", ""),
                    }
                )

    # Render conversation history (newest first)
    for entry in reversed(st.session_state["rag_history"]):
        st.markdown("---")
        st.markdown(f"**You:** {entry['question']}")
        st.markdown("**Assistant:**")
        st.markdown(entry["reply"])
        _render_sources(entry["sources"])
        if entry.get("generated_at"):
            st.caption(f"Generated at {entry['generated_at']}")

    if not st.session_state["rag_history"]:
        st.info("Ask a question above to get started.")


main()
