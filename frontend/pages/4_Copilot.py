"""
Copilot page — AI-assisted explanations and insights.
"""

import sys
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent.parent
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

import streamlit as st

from components.api import copilot_explain, get_api_base_url, get_metrics, get_monitoring_summary


def main() -> None:
    st.title("Copilot")
    st.markdown("AI-assisted explanations for forecasts and model behavior.")

    st.info("**Note:** Copilot explains forecasts and metrics using precomputed data — it does not predict.")

    api_base = get_api_base_url()
    st.caption(f"API: `{api_base}`")

    st.markdown("---")
    st.subheader("Ask a question")

    query = st.text_input(
        "Your question",
        placeholder="e.g. Why did the forecast increase? What does the current MAE indicate?",
        key="copilot_query",
    )

    if st.button("Submit", type="primary", key="copilot_submit"):
        if not query or not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Getting explanation..."):
                monitoring_summary = get_monitoring_summary()
                metrics = get_metrics()

                context = {
                    "include_metrics": True,
                    "monitoring_summary": monitoring_summary,
                    "metrics": metrics,
                }

                result = copilot_explain(
                    query=query.strip(),
                    context=context,
                    options={"max_tokens": 512, "format": "plain"},
                )

            explanation = result.get("explanation", "")
            sources = result.get("sources", [])
            generated_at = result.get("generated_at", "")

            st.markdown("---")
            st.subheader("Response")
            st.markdown(explanation)

            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.write(f"- {s.get('type', '')}: {s}")

            if generated_at:
                st.caption(f"Generated at {generated_at}")


if __name__ == "__main__":
    main()
