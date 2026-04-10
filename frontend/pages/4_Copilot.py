"""
Insight Copilot — rule-based explanations from monitoring and drift context.
Uses backend POST /api/v1/copilot/explain.
"""

import requests
import streamlit as st
from components.api import copilot_explain, describe_request_error, get_monitoring_summary
from components.ui import LOADING_COPIOT_MESSAGE, render_error, render_warning, with_loading


def _summary_only(summary: dict) -> dict:
    """Build summary-level context for LLM; exclude raw time-series data."""
    return {k: v for k, v in summary.items() if k != "rolling_series"}


def _optional_session_forecast() -> list | None:
    fc = st.session_state.get("fc_forecast") or {}
    f = fc.get("forecasts")
    return f if isinstance(f, list) and f else None


def _render_copilot_result(result: dict) -> None:
    answer = result.get("answer") or ""
    reasoning = result.get("reasoning") or ""
    explanation = result.get("explanation") or ""
    sources = result.get("sources", [])
    generated_at = result.get("generated_at", "")

    st.markdown("---")
    st.subheader("Response")
    if answer:
        st.markdown("**Answer**")
        st.markdown(answer)
    elif explanation:
        st.markdown(explanation)
    conf = result.get("confidence")
    if conf is not None:
        st.caption(f"Confidence: {float(conf):.2f}")
    if reasoning:
        with st.expander("Reasoning (signals used)"):
            st.markdown(reasoning)
    if sources:
        with st.expander("Sources"):
            for s in sources:
                st.write(f"- {s.get('type', '')}: {s.get('note', s.get('title', s))}")
    if generated_at:
        st.caption(f"Generated at {generated_at}")


def main() -> None:
    st.title("Insight Copilot")

    render_warning("Copilot explains results; it does not generate predictions.")

    st.markdown("---")

    # Check for alert context from Monitoring page redirect
    alert_context = st.session_state.pop("alert_context", None)

    if alert_context is not None:
        alerts = alert_context.get("alerts") or []
        alert_types = [a.get("alert_type", "") for a in alerts if a.get("alert_type")]
        if not alert_types:
            alert_types = ["alert"]
        alert_label = " and ".join(alert_types) if len(alert_types) > 1 else alert_types[0]
        query = f"Explain why the {alert_label} alert was triggered and what it means."

        monitoring_summary = _summary_only(alert_context.get("monitoring_summary") or {})
        context = {
            "monitoring_summary": monitoring_summary,
            "performance": monitoring_summary.get("performance") or {},
            "drift": monitoring_summary.get("drift") or {},
            "alerts": alerts,
            "overall_status": monitoring_summary.get("overall_status"),
            "recent_activity": monitoring_summary.get("recent_activity") or {},
            "alert_context": {
                "alerts": alerts,
                "timestamp": alert_context.get("timestamp"),
            },
        }
        of = _optional_session_forecast()
        if of is not None:
            context["forecast"] = of
        try:
            result = with_loading(copilot_explain, query=query, context=context, message=LOADING_COPIOT_MESSAGE)
        except requests.RequestException as exc:
            render_error(describe_request_error(exc))
            return

        st.text_input(
            "Enter your question",
            value=query,
            key="copilot_query",
            disabled=True,
        )

        _render_copilot_result(result)
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

            def _fetch_and_explain():
                monitoring_summary = _summary_only(get_monitoring_summary())
                context = {
                    "monitoring_summary": monitoring_summary,
                    "performance": monitoring_summary.get("performance") or {},
                    "drift": monitoring_summary.get("drift") or {},
                    "overall_status": monitoring_summary.get("overall_status"),
                    "recent_activity": monitoring_summary.get("recent_activity") or {},
                }
                of = _optional_session_forecast()
                if of is not None:
                    context["forecast"] = of
                return copilot_explain(query=query.strip(), context=context)

            try:
                result = with_loading(_fetch_and_explain, message=LOADING_COPIOT_MESSAGE)
            except requests.RequestException as exc:
                render_error(describe_request_error(exc))
                return

            _render_copilot_result(result)


main()
