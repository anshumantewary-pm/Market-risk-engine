"""Marginal and component VaR (parametric) and component ES (historical)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from riskengine.conventions import validate_alpha


def parametric_components(
    window_returns: pd.DataFrame, weights: pd.Series, alpha: float
) -> pd.DataFrame:
    """Euler decomposition of zero-mean parametric VaR, ``z_α · σ_p``.

    * marginal VaR_i  = z_α · (Σw)_i / σ_p   (VaR change per unit of extra weight)
    * component VaR_i = w_i · marginal VaR_i

    Because σ_p is homogeneous of degree one in the weights, the components add up exactly
    to the total. All figures are fractions of notional.
    """
    validate_alpha(alpha)
    w = weights.astype(float)
    r = window_returns[list(w.index)].to_numpy(dtype=float)
    cov = np.atleast_2d(np.cov(r, rowvar=False, ddof=1))
    wv = w.to_numpy()
    sigma_p = float(np.sqrt(wv @ cov @ wv))
    z = stats.norm.ppf(alpha)
    marginal = z * (cov @ wv) / sigma_p
    component = wv * marginal
    total = z * sigma_p
    return pd.DataFrame(
        {
            "weight": wv,
            "marginal_var": marginal,
            "component_var": component,
            "risk_share": component / total,
        },
        index=w.index,
    )
