"""
Insight Copilot — AI-assisted explanations from monitoring and drift results.
No forecasting logic; no raw data sent to LLM. Uses mock when API unavailable.
"""

import sys
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent.parent
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

import streamlit as st

from components.api import copilot_explain, get_monitoring_summary


def main() -> None:
    st.title("Insight Copilot")

    st.warning("⚠️ Copilot explains results — it does not generate predictions.")

    st.markdown("---")
    query = st.text_input(
        "Enter your question",
        placeholder="e.g. Why did the forecast increase? What does the current MAE indicate?",
        key="copilot_query",
    )

    if st.button("Submit", type="primary", key="copilot_submit"):
        if not query or not query.strip():
            st.warning("Enter a question to continue.")
        else:
            with st.spinner("Retrieving explanation..."):
                monitoring_summary = get_monitoring_summary()
                # Context: summary data only (performance, drift) — no raw time series
                context = {
                    "monitoring_summary": monitoring_summary,
                    "performance": monitoring_summary.get("performance") or {},
                    "drift": monitoring_summary.get("drift") or {},
                }
                result = copilot_explain(query=query.strip(), context=context)

            explanation = result.get("explanation", "")
            sources = result.get("sources", [])
            generated_at = result.get("generated_at", "")

            st.markdown("---")
            st.subheader("Response")
            with st.container():
                st.markdown(explanation)

            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.write(f"- {s.get('type', '')}: {s.get('note', s)}")

            if generated_at:
                st.caption(f"Generated at {generated_at}")


main()
