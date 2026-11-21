"""Plotly figures for the dashboard. Pure functions: tables in, figures out.

Colour rules: methods and series take categorical colours in a fixed order (a method keeps
its colour on every chart); status colours are reserved for exceptions and traffic lights
and always come with a text label.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from riskengine.risk.registry import METHOD_LABELS

# Categorical palette, fixed order (validated for colour-vision deficiency, adjacent pairs).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
METHOD_COLORS = {m: SERIES[i] for i, m in enumerate(METHOD_LABELS)}
STATUS = {"green": "#0ca30c", "yellow": "#fab219", "red": "#d03b3b"}
CRITICAL = "#d03b3b"

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def _layout(fig: go.Figure, title: str, height: int = 380, **kw) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 15, "color": INK}},
        height=height,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": "Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif",
              "color": INK_2, "size": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0, "title": None},
        hoverlabel={"bgcolor": "white", "font": {"color": INK}},
        bargap=0.25,
        **kw,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont={"color": MUTED})
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont={"color": MUTED})
    return fig


def pnl_history(returns: pd.DataFrame) -> go.Figure:
    """Cumulative portfolio P&L in dollars (sum of daily linear P&L)."""
    cum = returns["pnl"].cumsum()
    fig = go.Figure(
        go.Scatter(
            x=cum.index, y=cum.values, mode="lines", line={"color": SERIES[0], "width": 2},
            name="Cumulative P&L", hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    _layout(fig, "Cumulative portfolio P&L ($)", showlegend=False, hovermode="x unified")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


def method_comparison(estimates: pd.DataFrame, alpha: float, horizon: int) -> go.Figure:
    """VaR and ES in dollars for every method at one confidence level and horizon."""
    sub = estimates[(estimates["alpha"] == alpha) & (estimates["horizon_days"] == horizon)]
    sub = sub.set_index("method").reindex([m for m in METHOD_LABELS if m in set(sub["method"])])
    labels = [METHOD_LABELS[m] for m in sub.index]
    fig = go.Figure()
    for col, name, color in (("var_usd", "VaR", SERIES[0]), ("es_usd", "Expected Shortfall",
                                                              SERIES[1])):
        fig.add_bar(
            y=labels, x=sub[col], name=name, orientation="h", marker={"color": color},
            hovertemplate="%{y}<br>" + name + ": $%{x:,.0f}<extra></extra>",
        )
    _layout(fig, f"{alpha:.0%} {horizon}-day VaR and ES by method ($)", height=360,
            barmode="group")
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(autorange="reversed", gridcolor=SURFACE)
    return fig


def backtest_chart(forecasts: pd.DataFrame, method: str) -> go.Figure:
    """Realised daily loss against the 1-day VaR forecast, with exceptions marked."""
    f = forecasts[forecasts["method"] == method].sort_values("date")
    exc = f[f["exception"]]
    fig = go.Figure()
    fig.add_scatter(
        x=f["date"], y=f["realised_loss"] * 100, mode="markers", name="Realised loss",
        marker={"color": AXIS, "size": 3},
        hovertemplate="%{x|%d %b %Y}<br>Loss %{y:.2f}%<extra></extra>",
    )
    fig.add_scatter(
        x=f["date"], y=f["var"] * 100, mode="lines", name=f"VaR ({METHOD_LABELS.get(method)})",
        line={"color": METHOD_COLORS.get(method, SERIES[0]), "width": 2},
        hovertemplate="%{x|%d %b %Y}<br>VaR %{y:.2f}%<extra></extra>",
    )
    fig.add_scatter(
        x=exc["date"], y=exc["realised_loss"] * 100, mode="markers",
        name=f"Exception: loss > VaR ({len(exc)})",
        marker={"color": CRITICAL, "size": 8, "symbol": "x", "line": {"width": 0}},
        hovertemplate="%{x|%d %b %Y}<br>Exception: loss %{y:.2f}%<extra></extra>",
    )
    _layout(fig, "Out-of-sample backtest: daily loss vs 1-day VaR (% of notional)", height=420)
    fig.update_yaxes(ticksuffix="%")
    return fig
