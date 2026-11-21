"""Market Risk Engine dashboard.

Run with::

    streamlit run src/riskengine/dashboard/app.py

Reads the pipeline's outputs (``data/processed/`` or ``$RISKENGINE_OUTPUT_DIR``).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from riskengine.conventions import SQRT_TIME_CAVEAT
from riskengine.dashboard import charts
from riskengine.dashboard.data import Outputs, OutputsMissing, load_outputs, output_dir
from riskengine.risk.registry import METHOD_LABELS

st.set_page_config(page_title="Market Risk Engine", page_icon="📉", layout="wide")


@st.cache_data(show_spinner="Loading pipeline outputs...")
def _load(directory: str) -> Outputs:
    return load_outputs(directory)


def _usd(x: float) -> str:
    return f"${x:,.0f}"


def header(o: Outputs) -> None:
    m = o.metadata
    st.title("Market Risk Engine")
    st.caption(
        f"{m['portfolio']['name']} · notional {_usd(o.notional)} · data {m['data_start']} to "
        f"{m['data_end']} ({m['trading_days']:,} trading days) · source: {m['source']} · "
        f"window {m['risk']['window']} days"
    )
    if o.synthetic:
        st.warning(
            "**SYNTHETIC DATA.** Every number on this page comes from simulated prices and is "
            "for demonstrating the engine only. These are not real market results.",
            icon="⚠️",
        )


def overview_tab(o: Outputs) -> None:
    est = o.estimates
    hs = est[(est["method"] == "historical") & (est["horizon_days"] == 1)].set_index("alpha")
    st.subheader("Headline risk (historical simulation, 1-day)")
    cols = st.columns(4)
    for col, alpha in zip(cols[:2], (0.95, 0.99), strict=True):
        if alpha in hs.index:
            col.metric(f"VaR {alpha:.0%}", _usd(hs.loc[alpha, "var_usd"]),
                       f"{hs.loc[alpha, 'var_pct']:.2%} of notional", delta_color="off")
    for col, alpha in zip(cols[2:], (0.95, 0.99), strict=True):
        if alpha in hs.index:
            col.metric(f"ES {alpha:.0%}", _usd(hs.loc[alpha, "es_usd"]),
                       f"{hs.loc[alpha, 'es_pct']:.2%} of notional", delta_color="off")
    st.caption(
        "VaR and ES are shown as positive loss amounts: the loss not expected to be exceeded "
        "on 95% / 99% of days (VaR), and the average loss on the days that do exceed it (ES)."
    )

    left, right = st.columns([3, 2])
    left.plotly_chart(charts.pnl_history(o.returns), width="stretch")
    m = o.metadata["portfolio"]
    portfolio = pd.DataFrame(
        {
            "asset class": pd.Series(m["asset_classes"]),
            "weight": o.weights,
            "position ($)": o.weights * o.notional,
        }
    )
    right.markdown("**Portfolio**")
    right.dataframe(
        portfolio.style.format({"weight": "{:.1%}", "position ($)": "${:,.0f}"}),
        width="stretch", height=460,
    )


def methods_tab(o: Outputs) -> None:
    est = o.estimates
    c1, c2 = st.columns(2)
    alphas = sorted(est["alpha"].unique())
    horizons = sorted(est["horizon_days"].unique())
    alpha = c1.radio("Confidence level", alphas, index=len(alphas) - 1,
                     format_func=lambda a: f"{a:.0%}", horizontal=True)
    horizon = c2.radio("Horizon", horizons, format_func=lambda h: f"{h}-day", horizontal=True)
    st.plotly_chart(charts.method_comparison(est, alpha, horizon), width="stretch")
    if horizon > 1:
        st.info(SQRT_TIME_CAVEAT)
    table = est[(est["alpha"] == alpha) & (est["horizon_days"] == horizon)]
    table = table.set_index("label")[["var_usd", "es_usd", "var_pct", "es_pct"]]
    st.dataframe(
        table.style.format({"var_usd": "${:,.0f}", "es_usd": "${:,.0f}",
                            "var_pct": "{:.2%}", "es_pct": "{:.2%}"}),
        width="stretch",
    )
    st.caption(
        "Historical simulation uses past losses as they happened; the parametric models assume "
        "a normal or fat-tailed Student-t distribution; EWMA filtered HS rescales history to "
        "today's volatility; Monte Carlo simulates from a normal or t-copula model."
    )


ZONE_BADGE = {"green": ("green", "Green zone"), "yellow": ("orange", "Yellow zone"),
              "red": ("red", "Red zone")}


def backtest_tab(o: Outputs) -> None:
    s = o.backtest_summary
    alpha = o.metadata["risk"]["backtest_alpha"]
    first = s.iloc[0]
    st.markdown(
        f"Rolling out-of-sample **1-day {alpha:.0%} VaR**, {int(first['days']):,} days "
        f"({first['start']} to {first['end']}). Each forecast uses only data before its day."
    )
    methods = [m for m in METHOD_LABELS if m in s.index]
    method = st.selectbox("Method", methods, format_func=METHOD_LABELS.get,
                          index=methods.index("ewma_fhs") if "ewma_fhs" in methods else 0)
    row = s.loc[method]
    color, label = ZONE_BADGE[row["traffic_light"]]
    c = st.columns(4)
    c[0].metric("Exceptions", int(row["exceptions"]),
                f"{row['expected_exceptions']:.1f} expected", delta_color="off")
    c[1].metric("Kupiec p-value", f"{row['kupiec_p']:.3f}")
    c[2].metric("Christoffersen p-value", f"{row['christoffersen_p']:.3f}")
    with c[3]:
        st.markdown("**Traffic light (last 250 days)**")
        st.badge(f"{label}: {int(row['traffic_light_exceptions'])} exceptions", color=color)
    st.plotly_chart(charts.backtest_chart(o.forecasts, method), width="stretch")

    st.markdown("**All methods**")
    table = s[["exceptions", "expected_exceptions", "kupiec_p", "christoffersen_p", "cc_p",
               "traffic_light", "traffic_light_exceptions"]].rename(index=METHOD_LABELS)
    table["traffic_light"] = table["traffic_light"].str.capitalize()
    st.dataframe(
        table.style.format({"expected_exceptions": "{:.1f}", "kupiec_p": "{:.3f}",
                            "christoffersen_p": "{:.3f}", "cc_p": "{:.3f}"}),
        width="stretch",
    )
    st.caption(
        "p-values below 0.05 reject the model at 5%. Kupiec tests the exception rate, "
        "Christoffersen tests whether exceptions cluster, conditional coverage tests both. "
        "Traffic light zones follow the Basel convention: green 0-4, yellow 5-9, red 10+ "
        "exceptions in 250 days at 99%."
    )


def main() -> None:
    try:
        o = _load(str(output_dir()))
    except OutputsMissing as exc:
        st.title("Market Risk Engine")
        st.error(str(exc))
        st.stop()
        return

    header(o)
    tabs = st.tabs(["Overview", "Method comparison", "Backtest"])
    with tabs[0]:
        overview_tab(o)
    with tabs[1]:
        methods_tab(o)
    with tabs[2]:
        backtest_tab(o)


main()
