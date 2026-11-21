"""End-to-end risk run.

Usage::

    python -m riskengine.pipeline            # real data in data/raw/prices.csv
    python -m riskengine.pipeline --sample   # generate synthetic prices first

Steps: load config -> prices -> returns -> headline VaR/ES -> rolling backtest ->
attribution -> stress -> write ``data/processed/``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from riskengine import __version__
from riskengine.attribution.components import attribution_table
from riskengine.backtest.rolling import rolling_forecasts, stack_forecasts
from riskengine.backtest.summary import summary_table
from riskengine.config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_STRESS_PATH,
    Config,
    load_config,
    load_stress_config,
)
from riskengine.data.load import DEFAULT_PRICES_PATH, LoadReport, load_prices, save_prices_csv
from riskengine.data.returns import log_returns
from riskengine.data.synthetic import generate_prices
from riskengine.risk.registry import build_estimators, estimate_all, estimates_frame, latest_window
from riskengine.stress.historical import replay_all
from riskengine.stress.hypothetical import apply_all
from riskengine.stress.result import stress_frame

DEFAULT_OUT_DIR = Path("data/processed")


@dataclass(frozen=True)
class RunSettings:
    config_path: Path = DEFAULT_CONFIG_PATH
    stress_path: Path = DEFAULT_STRESS_PATH
    prices_path: Path = DEFAULT_PRICES_PATH
    out_dir: Path = DEFAULT_OUT_DIR
    sample: bool = False
    sample_start: str = "2005-01-03"
    sample_end: str = "2024-12-31"
    backtest_days: int | None = None   # None = every day after the first window
    backtest_sims: int = 5_000         # Monte Carlo simulations per backtest forecast
    verbose: bool = True


def parse_args(argv: list[str] | None = None) -> RunSettings:
    p = argparse.ArgumentParser(description="Run the market risk pipeline.")
    p.add_argument("--sample", action="store_true",
                   help="generate SYNTHETIC prices instead of using downloaded data")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--stress-config", default=str(DEFAULT_STRESS_PATH))
    p.add_argument("--prices", default=str(DEFAULT_PRICES_PATH))
    p.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--sample-start", default=RunSettings.sample_start)
    p.add_argument("--sample-end", default=RunSettings.sample_end)
    p.add_argument("--backtest-days", type=int, default=None,
                   help="only backtest the most recent N days (default: all)")
    p.add_argument("--backtest-sims", type=int, default=RunSettings.backtest_sims)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    return RunSettings(
        config_path=Path(a.config), stress_path=Path(a.stress_config),
        prices_path=Path(a.prices), out_dir=Path(a.out), sample=a.sample,
        sample_start=a.sample_start, sample_end=a.sample_end,
        backtest_days=a.backtest_days, backtest_sims=a.backtest_sims, verbose=not a.quiet,
    )


def _log(settings: RunSettings, msg: str) -> None:
    if settings.verbose:
        print(msg, flush=True)


@dataclass(frozen=True)
class MarketData:
    prices: pd.DataFrame
    returns: pd.DataFrame
    report: LoadReport
    source: str        # "synthetic" or "stooq"
    synthetic: bool


def load_market_data(settings: RunSettings, cfg: Config) -> MarketData:
    """Step 1: get prices (generating synthetic ones if asked), validate, compute returns."""
    tickers = cfg.portfolio.tickers
    if settings.sample:
        _log(settings, "Generating SYNTHETIC prices (not real market data)...")
        synthetic = generate_prices(
            tickers, settings.sample_start, settings.sample_end, seed=cfg.risk.seed
        )
        save_prices_csv(synthetic, settings.prices_path)
    prices, report = load_prices(settings.prices_path, tickers)
    returns = log_returns(prices)
    _log(settings, f"Prices: {report.n_rows} days, {report.start.date()} to {report.end.date()}"
         f" ({report.trimmed_rows} trimmed, {report.dropped_rows} dropped rows)")
    source = "synthetic" if settings.sample else "stooq"
    return MarketData(prices, returns, report, source, settings.sample)


def current_estimates(settings: RunSettings, cfg: Config, data: MarketData) -> pd.DataFrame:
    """Step 2: every method x confidence level x horizon on the latest window."""
    window = latest_window(data.returns, cfg.risk.window)
    estimates = estimate_all(
        window, cfg.portfolio.weight_series(), cfg.portfolio.notional,
        build_estimators(cfg.risk), cfg.risk.confidence_levels, cfg.risk.horizons,
    )
    table = estimates_frame(estimates)
    _log(settings, f"Estimates: {len(table)} rows for window ending {window.index[-1].date()}")
    return table


def backtest(
    settings: RunSettings, cfg: Config, data: MarketData
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 3: rolling out-of-sample 1-day VaR for every method, then the statistical tests."""
    alpha = cfg.risk.backtest_alpha
    estimators = build_estimators(cfg.risk, n_sims=settings.backtest_sims)
    weights = cfg.portfolio.weight_series()
    by_method = {}
    for name, est in estimators.items():
        t0 = time.perf_counter()
        by_method[name] = rolling_forecasts(
            data.returns, weights, est, alpha, cfg.risk.window, settings.backtest_days
        )
        f = by_method[name]
        _log(settings, f"Backtest {name:<12} {len(f)} days, {int(f['exception'].sum())} "
             f"exceptions ({time.perf_counter() - t0:.1f}s)")
    summary = summary_table(by_method, alpha, cfg.risk.traffic_light_days)
    return stack_forecasts(by_method), summary


def attribution(cfg: Config, data: MarketData) -> pd.DataFrame:
    """Step 4: which positions drive risk, at the highest configured confidence level."""
    window = latest_window(data.returns, cfg.risk.window)
    return attribution_table(
        window, cfg.portfolio.weight_series(), max(cfg.risk.confidence_levels),
        cfg.portfolio.notional, dict(cfg.portfolio.asset_classes),
    )


def stress(settings: RunSettings, cfg: Config, data: MarketData) -> pd.DataFrame:
    """Step 5: historical replay (from loaded prices) and illustrative factor shocks."""
    scenarios = load_stress_config(settings.stress_path)
    p = cfg.portfolio
    results = replay_all(data.prices, p.weight_series(), p.notional, scenarios.historical)
    results += apply_all(p.weight_series(), p.asset_classes, p.notional, scenarios.hypothetical)
    for r in results:
        if r.status == "skipped":
            _log(settings, f"Stress {r.name!r} skipped: {r.reason}")
    return stress_frame(results)


OUTPUT_FILES = (
    "returns.csv",
    "estimates.csv",
    "backtest_forecasts.csv",
    "backtest_summary.csv",
    "attribution.csv",
    "stress.csv",
)


def write_outputs(out_dir: Path, cfg: Config, data: MarketData, results: dict) -> list[Path]:
    """Step 6: write every table to ``out_dir`` as CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = cfg.portfolio.weight_series()
    returns = data.returns.copy()
    returns["portfolio_return"] = returns[list(weights.index)] @ weights
    returns["pnl"] = returns["portfolio_return"] * cfg.portfolio.notional
    tables = {
        "returns.csv": (returns, True),
        "estimates.csv": (results["estimates"], False),
        "backtest_forecasts.csv": (results["backtest_forecasts"], False),
        "backtest_summary.csv": (results["backtest_summary"], True),
        "attribution.csv": (results["attribution"], True),
        "stress.csv": (results["stress"], False),
    }
    paths = []
    for name in OUTPUT_FILES:
        frame, keep_index = tables[name]
        path = out_dir / name
        frame.to_csv(path, index=keep_index, date_format="%Y-%m-%d")
        paths.append(path)
    return paths


SYNTHETIC_WARNING = (
    "SYNTHETIC DATA: these results come from simulated prices and are for demonstrating "
    "the engine only. They are not real market results."
)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def run_metadata(settings: RunSettings, cfg: Config, data: MarketData) -> dict:
    """What produced these outputs: data source, synthetic flag, dates, config, versions."""
    p = cfg.portfolio
    meta = {
        "source": data.source,
        "synthetic": data.synthetic,
        "warning": SYNTHETIC_WARNING if data.synthetic else None,
        "data_start": str(data.report.start.date()),
        "data_end": str(data.report.end.date()),
        "trading_days": data.report.n_rows,
        "trimmed_rows": data.report.trimmed_rows,
        "dropped_rows": data.report.dropped_rows,
        "portfolio": {
            "name": p.name, "notional": p.notional, "weights": dict(p.weights),
            "asset_classes": dict(p.asset_classes),
        },
        "risk": asdict(cfg.risk),
        "backtest_days": settings.backtest_days,
        "backtest_sims": settings.backtest_sims,
        "package_version": __version__,
        "git_commit": _git_commit(),
        "run_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if data.synthetic:
        meta["sample_range"] = [settings.sample_start, settings.sample_end]
    return meta


def run(settings: RunSettings) -> dict:
    """Run every step and return a summary of what was produced."""
    cfg = load_config(settings.config_path)
    data = load_market_data(settings, cfg)
    estimates = current_estimates(settings, cfg, data)
    forecasts, bt_summary = backtest(settings, cfg, data)
    attrib = attribution(cfg, data)
    stress_results = stress(settings, cfg, data)
    results = {
        "source": data.source, "days": data.report.n_rows, "estimates": estimates,
        "backtest_forecasts": forecasts, "backtest_summary": bt_summary,
        "attribution": attrib, "stress": stress_results,
    }
    results["paths"] = write_outputs(settings.out_dir, cfg, data, results)
    meta = run_metadata(settings, cfg, data)
    meta_path = settings.out_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    results["paths"].append(meta_path)
    results["metadata"] = meta
    if data.synthetic:
        _log(settings, SYNTHETIC_WARNING)
    _log(settings, f"Wrote {len(results['paths'])} files to {settings.out_dir}")
    return results


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
