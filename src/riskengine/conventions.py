"""Conventions shared by every layer of the engine.

These rules are defined once here so that every estimator, the backtest, attribution
and the dashboard agree on signs, units and horizons.

* Asset returns are daily **log returns**, ``r_t = ln(P_t / P_{t-1})``.
* The portfolio return is **linear** in asset returns, ``r_p = sum_i w_i * r_i``, and
  portfolio P&L in dollars is ``notional * r_p``.
* ``loss = -P&L``. VaR and ES are reported as **loss amounts**: a positive number means
  the model expects to lose money at that confidence level.
* The forecast *for* day t uses data up to and including day t-1 only.
* 10-day figures are 1-day figures scaled by the square root of time, which assumes
  independent, identically distributed daily returns.
"""

from __future__ import annotations

import math

import numpy as np

TRADING_DAYS_PER_YEAR = 252
DEFAULT_CONFIDENCE_LEVELS: tuple[float, ...] = (0.95, 0.99)
DEFAULT_HORIZONS: tuple[int, ...] = (1, 10)
DEFAULT_WINDOW = 500

SQRT_TIME_CAVEAT = (
    "10-day figures use square-root-of-time scaling, which assumes independent, "
    "identically distributed daily returns. Volatility clustering makes this an "
    "approximation."
)


def to_loss(pnl):
    """Convert P&L (or returns) into losses: ``loss = -pnl``.

    Works on scalars, NumPy arrays and pandas objects alike.
    """
    return -pnl


def validate_alpha(alpha: float) -> float:
    """Check a confidence level is a probability strictly between 0.5 and 1."""
    if not isinstance(alpha, (int, float)) or not 0.5 < float(alpha) < 1.0:
        raise ValueError(f"confidence level must be in (0.5, 1), got {alpha!r}")
    return float(alpha)


def check_var_es(var: float, es: float, tol: float = 1e-12) -> tuple[float, float]:
    """Validate a (VaR, ES) pair and return it as floats.

    ES is the average loss beyond VaR, so it can never be smaller than VaR.
    """
    var, es = float(var), float(es)
    if not (math.isfinite(var) and math.isfinite(es)):
        raise ValueError(f"VaR and ES must be finite, got VaR={var}, ES={es}")
    if es < var - tol * max(1.0, abs(var)):
        raise ValueError(f"ES ({es}) must be at least VaR ({var})")
    return var, es


def scale_horizon(value, horizon_days: int):
    """Scale a 1-day risk figure to ``horizon_days`` using the square root of time."""
    if int(horizon_days) != horizon_days or horizon_days < 1:
        raise ValueError(f"horizon must be a positive whole number of days, got {horizon_days!r}")
    return value * np.sqrt(horizon_days)
