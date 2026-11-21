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


def historical_component_es(
    window_returns: pd.DataFrame, weights: pd.Series, alpha: float
) -> pd.Series:
    """Each position's average loss on the window's tail days.

    Tail days are those whose portfolio loss is at or beyond historical VaR (the same
    definition as ``HistoricalSimulation``). Position i's loss on day t is ``−w_i·r_it``,
    so the contributions add up exactly to historical ES.
    """
    validate_alpha(alpha)
    w = weights.astype(float)
    r = window_returns[list(w.index)].to_numpy(dtype=float)
    position_losses = -(r * w.to_numpy())
    portfolio_losses = position_losses.sum(axis=1)
    var = np.quantile(portfolio_losses, alpha)
    tail = portfolio_losses >= var
    return pd.Series(position_losses[tail].mean(axis=0), index=w.index, name="component_es")


def attribution_table(
    window_returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float,
    notional: float,
    asset_classes: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Per-position risk table for the dashboard and the pipeline output.

    ``concentration`` = risk share − weight share: positive means the position carries more
    of the portfolio's risk than its share of capital.
    """
    table = parametric_components(window_returns, weights, alpha)
    ces = historical_component_es(window_returns, weights, alpha)
    table["component_es"] = ces
    table["es_share"] = ces / ces.sum()
    gross = table["weight"].abs().sum()
    table["weight_share"] = table["weight"].abs() / gross
    table["concentration"] = table["risk_share"] - table["weight_share"]
    table["component_var_usd"] = table["component_var"] * notional
    table["component_es_usd"] = table["component_es"] * notional
    if asset_classes is not None:
        table.insert(0, "asset_class", [asset_classes.get(t, "other") for t in table.index])
    table.index.name = "ticker"
    return table
