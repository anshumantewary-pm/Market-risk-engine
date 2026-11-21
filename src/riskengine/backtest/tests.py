"""Statistical tests for VaR exception series.

All tests take a boolean series of exceptions (``True`` = loss exceeded VaR that day).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import special, stats


@dataclass(frozen=True)
class TestResult:
    """Likelihood-ratio statistic, its p-value and whether the model is rejected at 5%."""

    __test__ = False  # not a pytest test class

    name: str
    statistic: float
    p_value: float
    dof: int
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def reject_at_5pct(self) -> bool:
        return self.p_value < 0.05


def _as_bool_array(exceptions) -> np.ndarray:
    x = np.asarray(exceptions, dtype=bool)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("exceptions must be a non-empty 1-D sequence")
    return x


def kupiec_pof(exceptions, alpha: float) -> TestResult:
    """Kupiec proportion-of-failures test: is the exception rate equal to 1 − α?

    ``LR = −2 ln[(1−p)^{T−x} p^x] + 2 ln[(1−x/T)^{T−x} (x/T)^x]``  ~ χ²(1)

    with p = 1 − α, x exceptions in T days, and 0·ln 0 taken as 0.
    """
    e = _as_bool_array(exceptions)
    t, x = len(e), int(e.sum())
    p = 1 - alpha
    rate = x / t
    log_null = special.xlogy(t - x, 1 - p) + special.xlogy(x, p)
    log_alt = special.xlogy(t - x, 1 - rate) + special.xlogy(x, rate)
    lr = max(0.0, -2 * (log_null - log_alt))
    return TestResult(
        "kupiec_pof", lr, float(stats.chi2.sf(lr, 1)), 1,
        {"observations": t, "exceptions": x, "expected": t * p, "rate": rate},
    )


def transition_counts(exceptions) -> dict[str, int]:
    """Counts of consecutive-day pairs: n_ij = days in state i followed by state j."""
    e = _as_bool_array(exceptions).astype(int)
    prev, curr = e[:-1], e[1:]
    return {
        f"n{i}{j}": int(np.sum((prev == i) & (curr == j))) for i in (0, 1) for j in (0, 1)
    }


def christoffersen_independence(exceptions) -> TestResult:
    """Christoffersen test: is today's exception independent of yesterday's?

    Compares a first-order Markov chain (π01 = P(exception | none yesterday),
    π11 = P(exception | exception yesterday)) against a single rate π. Clustered
    exceptions, typical of a model that reacts too slowly to volatility, give π11 ≫ π01.
    ~ χ²(1).
    """
    c = transition_counts(exceptions)
    n00, n01, n10, n11 = c["n00"], c["n01"], c["n10"], c["n11"]
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    log_null = special.xlogy(n00 + n10, 1 - pi) + special.xlogy(n01 + n11, pi)
    log_alt = (
        special.xlogy(n00, 1 - pi01) + special.xlogy(n01, pi01)
        + special.xlogy(n10, 1 - pi11) + special.xlogy(n11, pi11)
    )
    lr = max(0.0, -2 * (log_null - log_alt))
    return TestResult(
        "christoffersen_independence", lr, float(stats.chi2.sf(lr, 1)), 1,
        {**c, "pi01": pi01, "pi11": pi11},
    )


def conditional_coverage(exceptions, alpha: float) -> TestResult:
    """Joint test of the right rate and independence: LR_cc = LR_pof + LR_ind ~ χ²(2)."""
    pof = kupiec_pof(exceptions, alpha)
    ind = christoffersen_independence(exceptions)
    lr = pof.statistic + ind.statistic
    return TestResult(
        "conditional_coverage", lr, float(stats.chi2.sf(lr, 2)), 2,
        {"lr_pof": pof.statistic, "lr_ind": ind.statistic},
    )


GREEN_CUTOFF = 0.95      # cumulative binomial probability where the yellow zone starts
YELLOW_CUTOFF = 0.9999   # ... and where the red zone starts


@dataclass(frozen=True)
class TrafficLight:
    zone: str            # "green", "yellow" or "red"
    exceptions: int
    days: int
    green_max: int       # most exceptions still in the green zone
    yellow_max: int      # most exceptions still in the yellow zone


def traffic_light(exceptions, alpha: float = 0.99, days: int = 250) -> TrafficLight:
    """Basel-style traffic light on the most recent ``days`` observations.

    Zones come from the cumulative binomial probability of seeing that many exceptions if
    the model were right: green below 95%, yellow below 99.99%, red above. For 250 days at
    99% this gives the familiar green 0–4, yellow 5–9, red 10+.
    """
    e = _as_bool_array(exceptions)[-days:]
    n, x = len(e), int(e.sum())
    cdf = stats.binom.cdf(np.arange(n + 1), n, 1 - alpha)
    green_max = int(np.searchsorted(cdf, GREEN_CUTOFF, side="left")) - 1
    yellow_max = int(np.searchsorted(cdf, YELLOW_CUTOFF, side="left")) - 1
    zone = "green" if x <= green_max else "yellow" if x <= yellow_max else "red"
    return TrafficLight(zone, x, n, green_max, yellow_max)
