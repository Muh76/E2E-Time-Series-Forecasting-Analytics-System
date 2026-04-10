"""
Insight Copilot — explanations from monitoring context via POST /api/v1/copilot/explain.

Uses OpenAI when ``OPENAI_API_KEY`` is set (natural language from forecast + metrics + drift);
otherwise rule-based fallbacks.
"""

import requests
import streamlit as st
from components.api import (
    copilot_explain,
    describe_request_error,
    get_forecast_evaluation_metrics,
    get_monitoring_summary,
    metrics_has_recorded_forecast,
)
from components.ui import LOADING_COPIOT_MESSAGE, render_error, render_warning, with_loading


def _summary_only(summary: dict) -> dict:
    """Build summary-level context for LLM; exclude raw time-series data."""
    return {k: v for k, v in summary.items() if k != "rolling_series"}


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
    gen = result.get("generator") or ""
    if gen == "openai":
        st.caption("Powered by OpenAI (forecast, metrics, and drift in context).")
    elif gen == "rules":
        st.caption("Rule-based explanation (set `OPENAI_API_KEY` in API `.env` for natural-language mode).")
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

    # Normal flow: require a server-recorded forecast so Copilot uses the latest run.
    try:
        metrics_snapshot = get_forecast_evaluation_metrics()
        has_recorded_forecast = metrics_has_recorded_forecast(metrics_snapshot)
    except requests.RequestException as exc:
        metrics_snapshot = None
        has_recorded_forecast = False
        render_error(describe_request_error(exc))

    if not has_recorded_forecast and metrics_snapshot is not None:
        st.info(
            "**Run a forecast first.** Open **Store Forecast**, generate a forecast, then return here. "
            "The API records the latest series on each run so Copilot answers match current results."
        )

    query = st.text_input(
        "Enter your question",
        placeholder="e.g. Why did the forecast increase? What does the current MAE indicate?",
        key="copilot_query",
        disabled=not has_recorded_forecast,
    )

    submit = st.button("Submit", type="primary", key="copilot_submit", disabled=not has_recorded_forecast)
    if not has_recorded_forecast:
        st.caption("Submit is disabled until a forecast has been recorded by the API.")

    if submit:
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
                return copilot_explain(query=query.strip(), context=context)

            try:
                result = with_loading(_fetch_and_explain, message=LOADING_COPIOT_MESSAGE)
            except requests.RequestException as exc:
                render_error(describe_request_error(exc))
                return

            _render_copilot_result(result)


main()
