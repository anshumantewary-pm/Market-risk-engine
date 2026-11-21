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
from riskengine.dashboard.whatif import normalise, recompute, risk_config_from_metadata
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


def attribution_tab(o: Outputs) -> None:
    a = o.attribution
    alpha = max(o.metadata["risk"]["confidence_levels"])
    top = a["concentration"].idxmax()
    st.markdown(
        f"Which positions drive the portfolio's **{alpha:.0%} 1-day VaR**. Component VaR "
        "splits parametric VaR across positions so the pieces add up to the total; "
        f"component ES does the same for historical ES. Largest concentration: **{top}** "
        f"carries {a.loc[top, 'risk_share']:.1%} of VaR on {a.loc[top, 'weight_share']:.1%} "
        "of capital."
    )
    st.plotly_chart(charts.attribution_chart(a), width="stretch")
    cols = ["asset_class", "weight", "risk_share", "es_share", "concentration",
            "component_var_usd", "component_es_usd"]
    st.dataframe(
        a[[c for c in cols if c in a.columns]].style.format({
            "weight": "{:.1%}", "risk_share": "{:.1%}", "es_share": "{:.1%}",
            "concentration": "{:+.1%}", "component_var_usd": "${:,.0f}",
            "component_es_usd": "${:,.0f}",
        }),
        width="stretch",
    )
    st.caption("A negative component means the position hedges the rest of the portfolio.")


def stress_tab(o: Outputs) -> None:
    st_df = o.stress
    st.markdown(
        "**Historical** scenarios replay each asset's actual move over the window, computed "
        "from the loaded prices, on today's portfolio. **Illustrative** scenarios apply "
        "shock sizes chosen as assumptions in `config/stress_scenarios.yaml`; they are not "
        "historical facts."
    )
    if (st_df["status"] == "run").any():
        st.plotly_chart(charts.stress_chart(st_df), width="stretch")
    table = st_df[["name", "kind", "status", "pnl_usd", "pnl_pct", "worst_day_usd",
                   "start", "end", "reason"]].set_index("name")
    st.dataframe(
        table.style.format({"pnl_usd": "${:,.0f}", "pnl_pct": "{:.1%}",
                            "worst_day_usd": "${:,.0f}"}, na_rep="-"),
        width="stretch",
    )
    skipped = st_df[st_df["status"] == "skipped"]
    for _, r in skipped.iterrows():
        st.caption(f"Skipped {r['name']}: {r['reason']}")


def whatif_tab(o: Outputs) -> None:
    st.markdown(
        "Change the weights and the engine recomputes 1-day VaR, ES and attribution on the "
        "latest window, using the same functions as the pipeline. Weights are rescaled to "
        "sum to 100% (long-only)."
    )
    base = o.weights
    cols = st.columns(4)
    raw = {}
    for i, t in enumerate(base.index):
        raw[t] = cols[i % 4].number_input(
            f"{t} (%)", min_value=0.0, max_value=100.0, value=round(base[t] * 100, 1),
            step=1.0, key=f"w_{t}",
        )
    try:
        new_w = normalise(raw)
    except ValueError as exc:
        st.error(str(exc))
        return
    cfg = risk_config_from_metadata(o.metadata)
    est_new, attrib_new = recompute(o.returns, new_w, o.notional, cfg)
    est_old, _ = recompute(o.returns, base, o.notional, cfg)
    alpha = max(cfg.confidence_levels)
    old = est_old[est_old["alpha"] == alpha].set_index("label")
    new = est_new[est_new["alpha"] == alpha].set_index("label")
    compare = pd.DataFrame({
        "current VaR": old["var_usd"], "what-if VaR": new["var_usd"],
        "change": new["var_usd"] - old["var_usd"],
        "current ES": old["es_usd"], "what-if ES": new["es_usd"],
    })
    st.markdown(f"**{alpha:.0%} 1-day VaR and ES**")
    st.dataframe(compare.style.format("${:,.0f}"), width="stretch")
    st.plotly_chart(charts.attribution_chart(attrib_new), width="stretch")


def main() -> None:
    try:
        o = _load(str(output_dir()))
    except OutputsMissing as exc:
        st.title("Market Risk Engine")
        st.error(str(exc))
        st.stop()
        return

    header(o)
    tabs = st.tabs(
        ["Overview", "Method comparison", "Backtest", "Attribution", "Stress tests", "What-if"]
    )
    with tabs[0]:
        overview_tab(o)
    with tabs[1]:
        methods_tab(o)
    with tabs[2]:
        backtest_tab(o)
    with tabs[3]:
        attribution_tab(o)
    with tabs[4]:
        stress_tab(o)
    with tabs[5]:
        whatif_tab(o)


main()
