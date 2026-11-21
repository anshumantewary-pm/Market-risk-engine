"""Replay named historical windows on today's portfolio.

Each asset's cumulative move over the window is measured from the loaded prices (from the
last close before the window to the last close inside it) and applied to today's weights
and notional. This is exact for linear positions held unchanged through the window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskengine.config import HistoricalWindow
from riskengine.stress.result import StressResult


def replay_window(
    prices: pd.DataFrame, weights: pd.Series, notional: float, window: HistoricalWindow
) -> StressResult:
    w = weights.astype(float)
    px = prices[list(w.index)]
    before = px.loc[px.index < window.start]
    inside = px.loc[window.start : window.end]

    def skipped(reason: str) -> StressResult:
        return StressResult(window.name, "historical", "skipped", reason,
                            start=window.start, end=window.end)

    if before.empty:
        return skipped(f"data starts {px.index[0].date()}, after the window starts")
    if px.index[-1] < window.end:
        return skipped(f"data ends {px.index[-1].date()}, before the window ends")
    if len(inside) < 2:
        return skipped("fewer than two trading days of data inside the window")

    base = before.iloc[-1]
    cum_return = inside.iloc[-1] / base - 1
    by_position = notional * w * cum_return

    path = pd.concat([before.iloc[[-1]], inside])
    daily_pnl = notional * (np.log(path).diff().iloc[1:] @ w)
    return StressResult(
        name=window.name,
        kind="historical",
        status="run",
        pnl_usd=float(by_position.sum()),
        pnl_pct=float(by_position.sum() / notional),
        by_position_usd={k: float(v) for k, v in by_position.items()},
        start=inside.index[0],
        end=inside.index[-1],
        worst_day_usd=float(daily_pnl.min()),
        worst_day=daily_pnl.idxmin(),
    )


def replay_all(
    prices: pd.DataFrame, weights: pd.Series, notional: float, windows
) -> list[StressResult]:
    return [replay_window(prices, weights, notional, w) for w in windows]
