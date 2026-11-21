"""Common result type for every stress scenario."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass(frozen=True)
class StressResult:
    name: str
    kind: str                     # "historical" or "hypothetical"
    status: str                   # "run" or "skipped"
    reason: str = ""              # why a scenario was skipped, or a label such as "illustrative"
    pnl_usd: float = float("nan")
    pnl_pct: float = float("nan")
    by_position_usd: dict[str, float] = field(default_factory=dict)
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    worst_day_usd: float = float("nan")
    worst_day: pd.Timestamp | None = None


def stress_frame(results: list[StressResult]) -> pd.DataFrame:
    """One row per scenario; position P&L spread into ``pnl_<ticker>`` columns."""
    rows = []
    for r in results:
        row = asdict(r)
        positions = row.pop("by_position_usd")
        row.update({f"pnl_{t}": v for t, v in positions.items()})
        rows.append(row)
    return pd.DataFrame(rows)
