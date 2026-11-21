"""Apply illustrative factor shocks to today's portfolio.

Shock sizes come from ``config/stress_scenarios.yaml`` and are assumptions, not history, so
every result is labelled "illustrative".
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from riskengine.config import FactorShock
from riskengine.stress.result import StressResult

ILLUSTRATIVE = "illustrative: shock sizes are assumptions, not historical moves"


def position_shocks(
    tickers: list[str], asset_classes: Mapping[str, str], scenario: FactorShock
) -> pd.Series:
    """Shock per ticker: a ticker entry wins, else its asset class, else zero."""
    return pd.Series(
        {
            t: scenario.shocks.get(t, scenario.shocks.get(asset_classes.get(t, ""), 0.0))
            for t in tickers
        },
        dtype=float,
    )


def apply_shock(
    weights: pd.Series,
    asset_classes: Mapping[str, str],
    notional: float,
    scenario: FactorShock,
) -> StressResult:
    w = weights.astype(float)
    shocks = position_shocks(list(w.index), asset_classes, scenario)
    by_position = notional * w * shocks
    return StressResult(
        name=scenario.name,
        kind="hypothetical",
        status="run",
        reason=ILLUSTRATIVE,
        pnl_usd=float(by_position.sum()),
        pnl_pct=float(by_position.sum() / notional),
        by_position_usd={k: float(v) for k, v in by_position.items()},
    )


def apply_all(
    weights: pd.Series, asset_classes: Mapping[str, str], notional: float, scenarios
) -> list[StressResult]:
    return [apply_shock(weights, asset_classes, notional, s) for s in scenarios]
