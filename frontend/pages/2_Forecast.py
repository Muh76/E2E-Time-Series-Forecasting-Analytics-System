"""
Store-level Forecast — generate forecast, evaluation metrics, and copilot insights from the API.
"""

import requests
import streamlit as st
from components.api import (
    copilot_forecast_insights,
    describe_request_error,
    forecast_store,
    get_forecast_evaluation_metrics,
    metrics_response_current,
    parse_api_error,
)
from components.metrics import format_float, format_mape
from components.ui import render_error, render_warning


def _render_error_list(errors: list[dict[str, str]]) -> None:
    for err in errors:
        field = err.get("field", "unknown")
        msg = err.get("message", "")
        label = f"**{field}**: {msg}" if field != "unknown" else msg
        render_error(label)


def _render_evaluation_metrics(metrics: dict) -> None:
    st.subheader("Evaluation metrics")
    status = metrics.get("status", "")
    if status == "ok":
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", format_float(metrics.get("mae")))
        c2.metric("RMSE", format_float(metrics.get("rmse")))
        c3.metric("MAPE", format_mape(metrics.get("mape")))
        n = metrics.get("n_samples")
        if n is not None:
            st.caption(f"Based on {n} overlapping forecast dates with ground truth.")
        drift = metrics.get("drift")
        if drift and isinstance(drift, dict):
            st.caption(
                f"Drift (distribution): score {format_float(drift.get('drift_score'))} · "
                f"status {drift.get('status', '—')}"
            )
    else:
        msg = metrics.get("message") or "Metrics are not available for this forecast."
        reason = metrics.get("reason")
        if reason:
            st.caption(f"Reason: `{reason}`")
        render_warning(msg)
        st.caption("This is normal when forecast dates do not overlap processed actuals in the dataset.")


def _render_copilot(copilot: dict) -> None:
    st.subheader("Copilot insights")
    st.markdown(copilot.get("summary", ""))
    st.markdown("---")
    st.markdown(copilot.get("insights", ""))
    conf = copilot.get("confidence")
    if conf is not None:
        st.caption(f"Rule-based confidence: {float(conf):.2f}")


def main() -> None:
    st.title("Store Forecast")

    col1, col2 = st.columns(2)
    with col1:
        store_id = st.number_input("Store ID", min_value=1, value=1, step=1, key="fc_store")
    with col2:
        horizon = st.slider("Horizon (days)", min_value=1, max_value=60, value=7, key="fc_horizon")

    run = st.button("Generate Forecast", disabled=st.session_state.get("fc_loading", False))

    if run:
        st.session_state["fc_loading"] = True
        st.session_state.pop("fc_forecast", None)
        st.session_state.pop("fc_metrics", None)
        st.session_state.pop("fc_copilot", None)
        st.session_state.pop("fc_errors", None)

        errors: dict[str, list[dict[str, str]] | str] = {}
        sid = int(store_id)

        try:
            with st.spinner("Running forecast…"):
                fc_json = forecast_store(sid, horizon)
            st.session_state["fc_forecast"] = fc_json
        except requests.HTTPError as exc:
            errors["forecast"] = parse_api_error(exc)
        except requests.RequestException as exc:
            errors["forecast"] = describe_request_error(exc)
        except Exception as exc:
            errors["forecast"] = str(exc)

        fc_json = st.session_state.get("fc_forecast")
        forecasts = (fc_json or {}).get("forecasts", [])

        if fc_json is not None:
            try:
                with st.spinner("Computing metrics…"):
                    metrics = get_forecast_evaluation_metrics(sid)
                st.session_state["fc_metrics"] = metrics
            except requests.HTTPError as exc:
                errors["metrics"] = parse_api_error(exc)
            except requests.RequestException as exc:
                errors["metrics"] = describe_request_error(exc)
            except Exception as exc:
                errors["metrics"] = str(exc)

            metrics_for_copilot = metrics_response_current(st.session_state.get("fc_metrics") or {})
            try:
                with st.spinner("Generating insights…"):
                    copilot = copilot_forecast_insights(forecasts, metrics_for_copilot)
                st.session_state["fc_copilot"] = copilot
            except requests.HTTPError as exc:
                errors["copilot"] = parse_api_error(exc)
            except requests.RequestException as exc:
                errors["copilot"] = describe_request_error(exc)
            except Exception as exc:
                errors["copilot"] = str(exc)

        if errors:
            st.session_state["fc_errors"] = errors
        else:
            st.session_state.pop("fc_errors", None)
        st.session_state["fc_loading"] = False

    errs = st.session_state.get("fc_errors") or {}
    if errs.get("forecast"):
        fe = errs["forecast"]
        if isinstance(fe, list):
            _render_error_list(fe)
        else:
            render_error(str(fe))
    elif errs.get("metrics") or errs.get("copilot"):
        if errs.get("metrics"):
            me = errs["metrics"]
            render_error("; ".join(e["message"] for e in me) if isinstance(me, list) else str(me))
        if errs.get("copilot"):
            ce = errs["copilot"]
            render_error("; ".join(e["message"] for e in ce) if isinstance(ce, list) else str(ce))

    fc_json = st.session_state.get("fc_forecast")
    if not fc_json:
        return

    forecasts = fc_json.get("forecasts", [])
    st.success(
        f"Forecast for store **{fc_json.get('store_id', '')}** " f"(horizon {fc_json.get('horizon', '')} steps)."
    )

    if forecasts:
        dates = [f["date"] for f in forecasts]
        values = [f["forecast"] for f in forecasts]
        conf_low = [f.get("confidence_low") for f in forecasts]
        conf_high = [f.get("confidence_high") for f in forecasts]

        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#3b82f6"),
            )
        )
        if conf_low and conf_high and conf_low[0] is not None and conf_high[0] is not None:
            fig.add_trace(
                go.Scatter(
                    x=dates + dates[::-1],
                    y=conf_high + conf_low[::-1],
                    fill="toself",
                    fillcolor="rgba(59,130,246,0.12)",
                    line=dict(width=0),
                    name="95% CI",
                    showlegend=True,
                )
            )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Forecast",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=30, b=60, l=60, r=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast data")
        st.dataframe(
            [
                {
                    "Date": f["date"],
                    "Forecast": round(f["forecast"], 2),
                    "Low": f.get("confidence_low"),
                    "High": f.get("confidence_high"),
                }
                for f in forecasts
            ],
            use_container_width=True,
            hide_index=True,
        )

    metrics = st.session_state.get("fc_metrics")
    if metrics is not None:
        _render_evaluation_metrics(metrics_response_current(metrics))

    copilot = st.session_state.get("fc_copilot")
    if copilot is not None:
        _render_copilot(copilot)


main()
