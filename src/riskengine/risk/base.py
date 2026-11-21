"""Shared types and helpers for every VaR/ES estimator.

Every estimator is a pure function of a window of asset returns and the portfolio weights.
The caller decides which rows are in the window; that is what keeps "no look-ahead" a
property of a single place in the code (``riskengine.backtest.rolling``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from riskengine.conventions import check_var_es, validate_alpha

MIN_OBSERVATIONS = 30


@dataclass(frozen=True)
class VarEs:
    """1-day VaR and ES as positive fractions of notional, plus model diagnostics."""

    var: float
    es: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_var_es(self.var, self.es)


@runtime_checkable
class Estimator(Protocol):
    """Anything with a ``name`` and an ``estimate`` method is an estimator."""

    name: str

    def estimate(self, returns: pd.DataFrame, weights: pd.Series, alpha: float) -> VarEs:
        """Return 1-day VaR/ES using only the rows in ``returns``."""
        ...


def prepare(
    returns: pd.DataFrame, weights: pd.Series, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Validate inputs and return (asset return matrix T x N, weight vector N) as arrays."""
    validate_alpha(alpha)
    missing = [t for t in weights.index if t not in returns.columns]
    if missing:
        raise ValueError(f"no returns for weighted tickers: {missing}")
    r = returns[list(weights.index)].to_numpy(dtype=float)
    if len(r) < MIN_OBSERVATIONS:
        raise ValueError(f"need at least {MIN_OBSERVATIONS} observations, got {len(r)}")
    if not np.isfinite(r).all():
        raise ValueError("returns contain NaN or infinite values")
    return r, weights.to_numpy(dtype=float)


def empirical_var_es(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    """VaR = α-quantile of losses (linear interpolation); ES = mean of losses >= VaR."""
    losses = np.asarray(losses, dtype=float)
    var = float(np.quantile(losses, alpha))
    es = float(losses[losses >= var].mean())
    return var, es
