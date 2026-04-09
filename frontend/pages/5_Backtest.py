"""
Store-level Backtest — rolling-origin evaluation with per-split metrics.
"""

import requests
import streamlit as st

from components.api import backtest_store, parse_api_error
from components.ui import render_error


def main() -> None:
    st.title("Rolling-Origin Backtest")

    col1, col2, col3 = st.columns(3)
    with col1:
        store_id = st.number_input("Store ID", min_value=1, value=1, step=1, key="bt_store")
    with col2:
        horizon = st.slider("Horizon (days)", min_value=1, max_value=60, value=7, key="bt_horizon")
    with col3:
        n_splits = st.slider("Splits", min_value=1, max_value=20, value=3, key="bt_splits")

    run = st.button("Run Backtest", disabled=st.session_state.get("bt_loading", False))

    if run:
        st.session_state["bt_loading"] = True
        st.session_state.pop("bt_result", None)
        st.session_state.pop("bt_error", None)

        with st.spinner("Running backtest…"):
            try:
                result = backtest_store(int(store_id), horizon, n_splits)
                st.session_state["bt_result"] = result
            except requests.HTTPError as exc:
                st.session_state["bt_error"] = parse_api_error(exc)
            except requests.ConnectionError:
                st.session_state["bt_error"] = [{"field": "connection", "message": "Backend is unreachable."}]
            finally:
                st.session_state["bt_loading"] = False

    if st.session_state.get("bt_error"):
        for err in st.session_state["bt_error"]:
            field = err["field"]
            msg = err["message"]
            label = f"**{field}**: {msg}" if field != "unknown" else msg
            render_error(label)

    result = st.session_state.get("bt_result")
    if result:
        avg = result.get("average", {})
        st.subheader("Average Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{avg.get('rmse', 0):.2f}")
        c2.metric("MAE", f"{avg.get('mae', 0):.2f}")
        c3.metric("MAPE", f"{avg.get('mape', 0):.1f}%")

        splits = result.get("splits", [])
        if splits:
            st.subheader("Per-Split Results")
            st.dataframe(
                [{
                    "Split": s["split"],
                    "Cutoff": s["cutoff_date"],
                    "Horizon": s["horizon"],
                    "RMSE": round(s["rmse"], 2),
                    "MAE": round(s["mae"], 2),
                    "MAPE": f"{s['mape']:.1f}%",
                } for s in splits],
                use_container_width=True,
                hide_index=True,
            )

            import plotly.graph_objects as go

            cutoffs = [s["cutoff_date"] for s in splits]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=cutoffs, y=[s["rmse"] for s in splits], name="RMSE", marker_color="#3b82f6"))
            fig.add_trace(go.Bar(x=cutoffs, y=[s["mae"] for s in splits], name="MAE", marker_color="#10b981"))
            fig.update_layout(
                barmode="group",
                xaxis_title="Cutoff Date",
                yaxis_title="Error",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=30, b=60, l=60, r=40),
            )
            st.plotly_chart(fig, use_container_width=True)


main()
