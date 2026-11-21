"""End-to-end pipeline runs on short synthetic histories."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from riskengine.data.load import DataError
from riskengine.pipeline import OUTPUT_FILES, RunSettings, main, parse_args, run


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    tmp = tmp_path_factory.mktemp("run")
    raw = yaml.safe_load((root / "config" / "portfolio.yaml").read_text())
    raw["risk"].update(window=120, mc_simulations=2000)
    cfg_path = tmp / "portfolio.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    settings = RunSettings(
        config_path=cfg_path,
        stress_path=root / "config" / "stress_scenarios.yaml",
        prices_path=tmp / "raw" / "prices.csv",
        out_dir=tmp / "processed",
        sample=True,
        sample_start="2019-01-01",
        sample_end="2021-06-30",
        backtest_days=40,
        backtest_sims=1000,
        verbose=False,
    )
    return settings, run(settings)


def test_every_output_file_is_written(small_run):
    settings, results = small_run
    for name in (*OUTPUT_FILES, "run_metadata.json"):
        assert (settings.out_dir / name).exists(), name
    assert settings.prices_path.exists()


def test_estimates_table_obeys_conventions(small_run):
    settings, _ = small_run
    est = pd.read_csv(settings.out_dir / "estimates.csv")
    assert len(est) == 6 * 2 * 2
    assert (est["var_pct"] > 0).all() and (est["es_pct"] >= est["var_pct"]).all()
    np.testing.assert_allclose(est["var_usd"], est["var_pct"] * 10_000_000)


def test_backtest_outputs_cover_requested_days(small_run):
    settings, _ = small_run
    fc = pd.read_csv(settings.out_dir / "backtest_forecasts.csv", parse_dates=["date"])
    assert fc.groupby("method").size().eq(40).all()
    summary = pd.read_csv(settings.out_dir / "backtest_summary.csv", index_col="method")
    assert set(summary["traffic_light"]) <= {"green", "yellow", "red"}
    assert summary["kupiec_p"].between(0, 1).all()


def test_attribution_components_sum_to_one(small_run):
    settings, _ = small_run
    att = pd.read_csv(settings.out_dir / "attribution.csv", index_col="ticker")
    assert len(att) == 12
    assert att["risk_share"].sum() == pytest.approx(1.0)


def test_stress_windows_outside_the_data_are_skipped(small_run):
    settings, _ = small_run
    st = pd.read_csv(settings.out_dir / "stress.csv")
    status = dict(zip(st["name"], st["status"], strict=True))
    assert status["2008 global financial crisis"] == "skipped"
    assert status["2022 rate shock"] == "skipped"
    assert status["COVID-19 crash"] == "run"
    assert (st.loc[st["kind"] == "hypothetical", "status"] == "run").all()


def test_metadata_flags_synthetic_data(small_run):
    settings, _ = small_run
    meta = json.loads((settings.out_dir / "run_metadata.json").read_text())
    assert meta["synthetic"] is True and meta["source"] == "synthetic"
    assert "SYNTHETIC" in meta["warning"]
    assert meta["data_start"] >= "2019-01-01"
    assert meta["risk"]["window"] == 120
    assert sum(meta["portfolio"]["weights"].values()) == pytest.approx(1.0)


def test_real_mode_without_prices_explains_what_to_do(tmp_path, repo_root):
    settings = RunSettings(
        config_path=repo_root / "config" / "portfolio.yaml",
        prices_path=tmp_path / "missing.csv",
        out_dir=tmp_path / "out",
        verbose=False,
    )
    with pytest.raises(DataError, match="riskengine.download"):
        run(settings)


def test_parse_args_defaults_and_flags():
    s = parse_args(["--sample", "--backtest-days", "30", "--quiet"])
    assert s.sample and s.backtest_days == 30 and not s.verbose
    assert parse_args([]).sample is False


def test_main_returns_zero(small_run, tmp_path):
    settings, _ = small_run
    code = main([
        "--sample", "--quiet", "--config", str(settings.config_path),
        "--prices", str(tmp_path / "p.csv"), "--out", str(tmp_path / "o"),
        "--sample-start", "2019-01-01", "--sample-end", "2020-06-30",
        "--backtest-days", "5", "--backtest-sims", "1000",
    ])
    assert code == 0
