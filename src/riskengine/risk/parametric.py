"""Parametric (variance-covariance) VaR and ES."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

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


NU_BOUNDS = (2.1, 200.0)


def fit_t_dof(x: np.ndarray, bounds: tuple[float, float] = NU_BOUNDS) -> float:
    """Fit Student-t degrees of freedom ν by profile maximum likelihood.

    The data are standardised with their sample mean and standard deviation, and the scale
    is tied to ν so the fitted distribution keeps the sample variance. Only ν is searched,
    which is fast enough to refit every day in a rolling backtest.
    """
    z = (x - x.mean()) / x.std(ddof=1)

    def neg_loglik(nu: float) -> float:
        s = np.sqrt((nu - 2) / nu)
        return -float(np.sum(stats.t.logpdf(z / s, nu) - np.log(s)))

    res = optimize.minimize_scalar(neg_loglik, bounds=bounds, method="bounded")
    return float(res.x)


def student_t_var_es(mu: float, sigma: float, nu: float, alpha: float) -> tuple[float, float]:
    """Closed-form VaR/ES for a Student-t with mean μ and standard deviation σ.

    With scale ``s = σ·√((ν−2)/ν)`` and ``q = t_ν⁻¹(α)``::

        VaR = s·q − μ
        ES  = s · f_ν(q)/(1−α) · (ν + q²)/(ν − 1) − μ
    """
    s = sigma * np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(alpha, nu)
    var = s * q - mu
    es = s * stats.t.pdf(q, nu) / (1 - alpha) * (nu + q**2) / (nu - 1) - mu
    return float(var), float(es)


class ParametricStudentT:
    """Like the normal model, but with fat tails: ν is fitted to the window's portfolio returns."""

    name = "student_t"

    def estimate(self, returns: pd.DataFrame, weights: pd.Series, alpha: float) -> VarEs:
        r, w = prepare(returns, weights, alpha)
        rp = r @ w
        mu, sigma = float(rp.mean()), float(rp.std(ddof=1))
        nu = fit_t_dof(rp)
        var, es = student_t_var_es(mu, sigma, nu, alpha)
        return VarEs(var, es, {"mu": mu, "sigma": sigma, "nu": nu})
