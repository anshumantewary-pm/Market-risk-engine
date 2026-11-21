"""Parametric (variance-covariance) VaR and ES."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from riskengine.risk.base import VarEs, prepare


def portfolio_moments(r: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Portfolio mean ``wᵀμ`` and volatility ``√(wᵀΣw)`` from sample moments."""
    mu = float(r.mean(axis=0) @ w)
    cov = np.cov(r, rowvar=False, ddof=1)
    sigma = float(np.sqrt(w @ np.atleast_2d(cov) @ w))
    return mu, sigma


def normal_var_es(mu: float, sigma: float, alpha: float) -> tuple[float, float]:
    """Closed form for a normal P&L: VaR = z·σ − μ, ES = σ·φ(z)/(1−α) − μ."""
    z = stats.norm.ppf(alpha)
    return z * sigma - mu, sigma * stats.norm.pdf(z) / (1 - alpha) - mu


class ParametricNormal:
    """Assumes portfolio returns are normal with the window's mean and covariance."""

    name = "normal"

    def estimate(self, returns: pd.DataFrame, weights: pd.Series, alpha: float) -> VarEs:
        r, w = prepare(returns, weights, alpha)
        mu, sigma = portfolio_moments(r, w)
        var, es = normal_var_es(mu, sigma, alpha)
        return VarEs(var, es, {"mu": mu, "sigma": sigma})
