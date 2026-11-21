"""Market Risk Engine dashboard.

Run with::

    streamlit run src/riskengine/dashboard/app.py

Reads the pipeline's outputs (``data/processed/`` or ``$RISKENGINE_OUTPUT_DIR``).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from riskengine.dashboard import charts
from riskengine.dashboard.data import Outputs, OutputsMissing, load_outputs, output_dir

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


def main() -> None:
    try:
        o = _load(str(output_dir()))
    except OutputsMissing as exc:
        st.title("Market Risk Engine")
        st.error(str(exc))
        st.stop()
        return

    header(o)
    tabs = st.tabs(["Overview"])
    with tabs[0]:
        overview_tab(o)


main()
