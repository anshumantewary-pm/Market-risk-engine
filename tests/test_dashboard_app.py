"""Streamlit AppTest smoke tests: the app runs, every tab renders, the banner shows."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from riskengine.dashboard.data import OUTPUT_DIR_ENV

APP = str(Path(__file__).resolve().parents[1] / "src" / "riskengine" / "dashboard" / "app.py")
TABS = ["Overview", "Method comparison", "Backtest", "Attribution", "Stress tests", "What-if"]


@pytest.fixture
def app(small_outputs, monkeypatch, repo_root):
    monkeypatch.setenv(OUTPUT_DIR_ENV, str(small_outputs))
    monkeypatch.chdir(repo_root)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_app_runs_without_errors_and_shows_all_tabs(app):
    assert not app.exception, [e.value for e in app.exception]
    assert [t.label for t in app.tabs] == TABS


def test_synthetic_banner_is_shown(app):
    assert any("SYNTHETIC DATA" in w.value for w in app.warning)


def test_headline_metrics_are_positive_dollar_losses(app):
    labels = [m.label for m in app.metric]
    assert {"VaR 95%", "VaR 99%", "ES 95%", "ES 99%"} <= set(labels)
    for m in app.metric:
        if m.label.startswith(("VaR", "ES")):
            assert m.value.startswith("$") and not m.value.startswith("$-")


def test_switching_controls_rerenders_cleanly(app):
    app.radio[1].set_value(10).run()  # 10-day horizon on the method comparison tab
    assert not app.exception
    assert any("square-root-of-time" in i.value for i in app.info)
    app.selectbox[0].set_value("normal").run()
    assert not app.exception


def test_whatif_edit_recomputes(app):
    app.number_input(key="w_SPY").set_value(80.0).run()
    assert not app.exception


def test_missing_outputs_show_an_error_not_a_crash(tmp_path, monkeypatch, repo_root):
    monkeypatch.setenv(OUTPUT_DIR_ENV, str(tmp_path))
    monkeypatch.chdir(repo_root)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("make sample" in e.value for e in at.error)
