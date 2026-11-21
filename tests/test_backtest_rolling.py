import numpy as np
import pandas as pd
import pytest

from riskengine.backtest.rolling import rolling_forecasts, stack_forecasts
from riskengine.risk.base import VarEs
from riskengine.risk.historical import HistoricalSimulation
from riskengine.risk.parametric import ParametricNormal


@pytest.fixture
def data():
    rng = np.random.default_rng(21)
    idx = pd.bdate_range("2020-01-01", periods=160)
    df = pd.DataFrame(rng.normal(0, 0.01, (160, 2)), index=idx, columns=["A", "B"])
    return df, pd.Series({"A": 0.5, "B": 0.5})


def test_no_look_ahead_changing_the_future_does_not_change_past_forecasts(data):
    df, w = data
    base = rolling_forecasts(df, w, ParametricNormal(), 0.99, window=60)
    t = 100
    shocked = df.copy()
    shocked.iloc[t:] *= 25  # wild changes from day t onwards
    after = rolling_forecasts(shocked, w, ParametricNormal(), 0.99, window=60)
    day_t = df.index[t]
    # Every forecast up to and including day t must be identical...
    pd.testing.assert_series_equal(base.loc[:day_t, "var"], after.loc[:day_t, "var"])
    # ...while the realised loss on day t and forecasts after it do change.
    assert base.loc[day_t, "realised_loss"] != after.loc[day_t, "realised_loss"]
    assert base["var"].iloc[-1] != after["var"].iloc[-1]


class _Spy:
    """Records which dates each forecast was fitted on."""

    name = "spy"

    def __init__(self):
        self.windows = []

    def estimate(self, returns, weights, alpha):
        self.windows.append((returns.index[0], returns.index[-1], len(returns)))
        return VarEs(0.01, 0.02)


def test_each_forecast_uses_exactly_the_previous_window_days(data):
    df, w = data
    spy = _Spy()
    out = rolling_forecasts(df, w, spy, 0.99, window=60)
    assert len(out) == len(df) - 60
    first_start, first_end, n = spy.windows[0]
    assert (first_start, first_end, n) == (df.index[0], df.index[59], 60)
    assert out.index[0] == df.index[60]
    assert all(end < day for (_, end, _), day in zip(spy.windows, out.index, strict=True))


def test_realised_loss_and_exception_flag(data):
    df, w = data
    out = rolling_forecasts(df, w, HistoricalSimulation(), 0.95, window=60)
    expected = -(df.loc[out.index] @ w)
    np.testing.assert_allclose(out["realised_loss"], expected)
    assert (out["exception"] == (out["realised_loss"] > out["var"])).all()


def test_last_n_limits_the_forecast_period(data):
    df, w = data
    out = rolling_forecasts(df, w, HistoricalSimulation(), 0.99, window=60, last_n=20)
    assert len(out) == 20 and out.index[-1] == df.index[-1]


def test_too_little_data_raises(data):
    df, w = data
    with pytest.raises(ValueError):
        rolling_forecasts(df.iloc[:60], w, HistoricalSimulation(), 0.99, window=60)


def test_stack_forecasts_long_format(data):
    df, w = data
    a = rolling_forecasts(df, w, HistoricalSimulation(), 0.99, window=60, last_n=5)
    long = stack_forecasts({"historical": a, "copy": a})
    assert len(long) == 10
    assert list(long.columns) == ["date", "method", "var", "es", "realised_loss", "exception"]
