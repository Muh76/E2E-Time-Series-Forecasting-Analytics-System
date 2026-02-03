"""
Insight Copilot — AI-assisted explanations from monitoring and drift results.
No forecasting logic; no raw data sent to LLM. Uses mock when API unavailable.
"""

import streamlit as st

from components.api import copilot_explain, get_monitoring_summary
from components.ui import LOADING_COPIOT_MESSAGE, render_warning


def main() -> None:
    st.title("Insight Copilot")

    render_warning("Copilot explains results; it does not generate predictions.")

    st.markdown("---")

    # Check for alert context from Monitoring page redirect
    alert_context = st.session_state.pop("copilot_alert_context", None)
    alert_type = st.session_state.pop("copilot_alert_type", None)

    if alert_context is not None and alert_type:
        query = f"Explain the {alert_type} alert"
        context = {
            "monitoring_summary": alert_context,
            "performance": alert_context.get("performance") or {},
            "drift": alert_context.get("drift") or {},
        }
        with st.spinner(LOADING_COPIOT_MESSAGE):
            result = copilot_explain(query=query, context=context)

        st.text_input(
            "Enter your question",
            value=query,
            key="copilot_query",
            disabled=True,
        )

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
        return

    # Normal flow: user enters query manually
    query = st.text_input(
        "Enter your question",
        placeholder="e.g. Why did the forecast increase? What does the current MAE indicate?",
        key="copilot_query",
    )

    if st.button("Submit", type="primary", key="copilot_submit"):
        if not query or not query.strip():
            render_warning("Enter a question to continue.")
        else:
            with st.spinner(LOADING_COPIOT_MESSAGE):
                monitoring_summary = get_monitoring_summary()
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
