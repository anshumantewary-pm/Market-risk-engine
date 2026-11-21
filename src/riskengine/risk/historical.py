"""Historical simulation: the empirical distribution of the window's portfolio losses."""

from __future__ import annotations

import pandas as pd

from riskengine.risk.base import VarEs, empirical_var_es, prepare


class HistoricalSimulation:
    """VaR = α-quantile of past portfolio losses; ES = mean of losses at or beyond VaR.

    No distributional assumption, but every past day gets equal weight, so the estimate
    reacts slowly when volatility changes.
    """

    name = "historical"

    def estimate(self, returns: pd.DataFrame, weights: pd.Series, alpha: float) -> VarEs:
        r, w = prepare(returns, weights, alpha)
        losses = -(r @ w)
        var, es = empirical_var_es(losses, alpha)
        tail = int((losses >= var).sum())
        return VarEs(var, es, {"observations": len(losses), "tail_observations": tail})
