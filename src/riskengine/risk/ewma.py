"""EWMA (RiskMetrics) volatility and filtered historical simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LAMBDA = 0.94
SEED_DAYS = 30


def ewma_variance(r: np.ndarray, lam: float = DEFAULT_LAMBDA, seed_days: int = SEED_DAYS):
    """RiskMetrics variance for each day in the window and for the next day.

    ``σ²_t = λ·σ²_{t−1} + (1−λ)·r²_{t−1}``, seeded with the sample variance of the first
    ``seed_days`` rows, so nothing outside the window is used.

    Returns ``(var_in_window, var_next)``: an array shaped like ``r`` with the variance known
    *before* each day's return, and the forecast variance for the day after the window.
    """
    r = np.asarray(r, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lambda must be in (0, 1), got {lam}")
    seed = r[: max(2, min(seed_days, len(r)))].var(axis=0, ddof=1)
    var = np.empty_like(r)
    var[0] = seed
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    var_next = lam * var[-1] + (1 - lam) * r[-1] ** 2
    return var, var_next


def ewma_volatility(returns: pd.DataFrame, lam: float = DEFAULT_LAMBDA) -> pd.DataFrame:
    """EWMA volatility per asset and day (the vol known before that day's return)."""
    var, _ = ewma_variance(returns.to_numpy(dtype=float), lam)
    return pd.DataFrame(np.sqrt(var), index=returns.index, columns=returns.columns)
