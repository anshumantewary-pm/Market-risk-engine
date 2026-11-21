"""Rolling out-of-sample VaR forecasts.

This is the only place in the engine that slices data by time. The forecast for day t is
fitted on rows ``t - window`` to ``t - 1`` and then compared with the loss realised on
day t, so no forecast ever sees the day it is predicting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskengine.risk.base import Estimator


def rolling_forecasts(
    returns: pd.DataFrame,
    weights: pd.Series,
    estimator: Estimator,
    alpha: float,
    window: int = 500,
    last_n: int | None = None,
) -> pd.DataFrame:
    """Forecast 1-day VaR/ES for every day after the first ``window`` days.

    Parameters
    ----------
    last_n:
        Only forecast the final ``last_n`` days (useful to keep run time down).

    Returns
    -------
    DataFrame indexed by forecast date with columns ``var``, ``es``, ``realised_loss`` and
    ``exception`` (``realised_loss > var``). All values are fractions of notional.
    """
    if len(returns) <= window:
        raise ValueError(f"need more than {window} days of returns, have {len(returns)}")
    w = weights.astype(float)
    realised = -(returns[list(w.index)].to_numpy(dtype=float) @ w.to_numpy())

    first = window if last_n is None else max(window, len(returns) - last_n)
    rows = []
    for t in range(first, len(returns)):
        hist = returns.iloc[t - window : t]  # rows t-window .. t-1; day t is excluded
        res = estimator.estimate(hist, w, alpha)
        rows.append((res.var, res.es, realised[t]))

    out = pd.DataFrame(rows, index=returns.index[first:], columns=["var", "es", "realised_loss"])
    out["exception"] = out["realised_loss"] > out["var"]
    out.index.name = "date"
    return out


def stack_forecasts(by_method: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine per-method forecast frames into one long table with a ``method`` column."""
    frames = [f.assign(method=name).reset_index() for name, f in by_method.items()]
    out = pd.concat(frames, ignore_index=True)
    return out[["date", "method", "var", "es", "realised_loss", "exception"]].astype(
        {"exception": bool, "var": np.float64}
    )
