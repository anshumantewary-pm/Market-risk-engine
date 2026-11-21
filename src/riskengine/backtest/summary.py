"""One row per method: exceptions, test p-values and the traffic-light zone."""

from __future__ import annotations

import pandas as pd

from riskengine.backtest.tests import (
    christoffersen_independence,
    conditional_coverage,
    kupiec_pof,
    traffic_light,
)


def summarise(forecasts: pd.DataFrame, alpha: float, traffic_days: int = 250) -> dict:
    """Summarise one method's rolling forecasts (output of ``rolling_forecasts``)."""
    e = forecasts["exception"].to_numpy(dtype=bool)
    pof = kupiec_pof(e, alpha)
    ind = christoffersen_independence(e)
    cc = conditional_coverage(e, alpha)
    tl = traffic_light(e, alpha, traffic_days)
    return {
        "start": forecasts.index[0],
        "end": forecasts.index[-1],
        "days": len(e),
        "exceptions": int(e.sum()),
        "expected_exceptions": len(e) * (1 - alpha),
        "exception_rate": float(e.mean()),
        "kupiec_lr": pof.statistic,
        "kupiec_p": pof.p_value,
        "christoffersen_lr": ind.statistic,
        "christoffersen_p": ind.p_value,
        "cc_lr": cc.statistic,
        "cc_p": cc.p_value,
        "traffic_light": tl.zone,
        "traffic_light_exceptions": tl.exceptions,
        "traffic_light_days": tl.days,
        "avg_var": float(forecasts["var"].mean()),
        "avg_es": float(forecasts["es"].mean()),
    }


def summary_table(
    by_method: dict[str, pd.DataFrame], alpha: float, traffic_days: int = 250
) -> pd.DataFrame:
    """Summaries for every method, indexed by method name."""
    rows = {name: summarise(f, alpha, traffic_days) for name, f in by_method.items()}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "method"
    return out
