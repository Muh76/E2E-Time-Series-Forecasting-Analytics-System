"""
Store-level Forecast — generate and visualise multi-step forecasts.
"""

import requests
import streamlit as st

from components.api import forecast_store, parse_api_error
from components.ui import render_error


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
        st.session_state.pop("fc_result", None)
        st.session_state.pop("fc_error", None)

        with st.spinner("Running forecast…"):
            try:
                result = forecast_store(int(store_id), horizon)
                st.session_state["fc_result"] = result
            except requests.HTTPError as exc:
                st.session_state["fc_error"] = parse_api_error(exc)
            except requests.ConnectionError:
                st.session_state["fc_error"] = [{"field": "connection", "message": "Backend is unreachable."}]
            finally:
                st.session_state["fc_loading"] = False

    if st.session_state.get("fc_error"):
        for err in st.session_state["fc_error"]:
            field = err["field"]
            msg = err["message"]
            label = f"**{field}**: {msg}" if field != "unknown" else msg
            render_error(label)

    result = st.session_state.get("fc_result")
    if result:
        forecasts = result.get("forecasts", [])

        if forecasts:
            dates = [f["date"] for f in forecasts]
            values = [f["forecast"] for f in forecasts]
            conf_low = [f.get("confidence_low") for f in forecasts]
            conf_high = [f.get("confidence_high") for f in forecasts]

            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=values, mode="lines+markers", name="Forecast", line=dict(color="#3b82f6")))
            if conf_low[0] is not None and conf_high[0] is not None:
                fig.add_trace(go.Scatter(
                    x=dates + dates[::-1],
                    y=conf_high + conf_low[::-1],
                    fill="toself",
                    fillcolor="rgba(59,130,246,0.12)",
                    line=dict(width=0),
                    name="95% CI",
                    showlegend=True,
                ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Forecast",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=30, b=60, l=60, r=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Forecast Data")
            st.dataframe(
                [{"Date": f["date"], "Forecast": round(f["forecast"], 2),
                  "Low": f.get("confidence_low"), "High": f.get("confidence_high")}
                 for f in forecasts],
                use_container_width=True,
                hide_index=True,
            )


main()
