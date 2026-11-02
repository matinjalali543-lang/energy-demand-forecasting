"""Every metric against a value worked out by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
import metrics as M

ACTUAL = [100.0, 200.0, 400.0]
FORECAST = [110.0, 190.0, 300.0]  # errors +10, -10, -100


def test_point_metrics_by_hand():
    assert M.mae(ACTUAL, FORECAST) == pytest.approx(40.0)
    assert M.rmse(ACTUAL, FORECAST) == pytest.approx(np.sqrt(10_200 / 3))
    assert M.bias(ACTUAL, FORECAST) == pytest.approx(-100 / 3)
    # (10/100 + 10/200 + 100/400) / 3 = 0.4 / 3
    assert M.mape(ACTUAL, FORECAST) == pytest.approx(40 / 3)


def test_missing_values_are_skipped_not_counted_as_errors():
    assert M.mae([100.0, np.nan, 400.0], [110.0, 5.0, np.nan]) == pytest.approx(10.0)


def test_mape_refuses_non_positive_actuals():
    with pytest.raises(ValueError, match="strictly positive"):
        M.mape([0.0, 100.0], [1.0, 100.0])


def test_mase_divides_by_the_given_scale():
    assert M.mase(ACTUAL, FORECAST, scale=20.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        M.mase(ACTUAL, FORECAST, scale=0.0)


def test_seasonal_naive_scale_is_the_week_on_week_mae():
    """Week one all 0, week two all 10: each week-two hour misses by 10."""
    index = pd.date_range("2025-01-06", periods=336, freq="h", tz="UTC")
    demand = pd.Series(np.repeat([0.0, 10.0], 168), index=index)
    assert M.seasonal_naive_scale(demand) == pytest.approx(10.0)


@pytest.mark.parametrize("forecast, expected", [(8.0, 1.8), (12.0, 0.2), (10.0, 0.0)])
def test_pinball_loss_at_p90(forecast, expected):
    """Under-forecasting a P90 costs 0.9 per unit; over-forecasting costs 0.1."""
    assert M.pinball_loss([10.0], [forecast], q=0.9) == pytest.approx(expected)


def test_coverage_counts_actuals_inside_the_band():
    actual = [1.0, 2.0, 3.0, 4.0]
    lower = [0.0, 0.0, 0.0, 5.0]
    upper = [2.0, 1.0, 5.0, 6.0]  # in, out, in, out
    assert M.coverage(actual, lower, upper) == pytest.approx(50.0)


def test_daily_peak_errors_measure_height_and_timing():
    """Actual peaks at 100 MW at 15:00; forecast peaks at 90 MW at 17:00."""
    local = pd.date_range("2025-07-16", periods=24, freq="h", tz=config.LOCAL_TZ)
    actual = np.full(24, 50.0)
    forecast = np.full(24, 50.0)
    actual[15], forecast[17] = 100.0, 90.0
    results = pd.DataFrame(
        {"target_time": local.tz_convert("UTC"), "actual": actual, "forecast": forecast}
    )
    out = M.daily_peak_errors(results)
    assert out["peak_mae_mw"] == pytest.approx(10.0)
    assert out["peak_mape_%"] == pytest.approx(10.0)
    assert out["peak_timing_h"] == pytest.approx(2.0)


def test_breakdown_by_peak_splits_on_local_hours():
    local = pd.date_range("2025-07-16", periods=24, freq="h", tz=config.LOCAL_TZ)
    actual = np.full(24, 100.0)
    forecast = np.where(np.isin(np.arange(24), M.PEAK_HOURS), 110.0, 100.0)
    results = pd.DataFrame(
        {"target_time": local.tz_convert("UTC"), "actual": actual, "forecast": forecast}
    )
    table = M.breakdown(results, "peak")
    assert table.loc["peak 14:00-19:59", "MAPE_%"] == pytest.approx(10.0)
    assert table.loc["off-peak", "MAPE_%"] == pytest.approx(0.0)
    assert table["hours"].sum() == 24


def test_day_types_put_holidays_before_weekends():
    stamps = pd.DatetimeIndex(
        [pd.Timestamp(f"{day} 12:00", tz=config.LOCAL_TZ) for day in
         ("2025-12-25", "2025-12-27", "2025-11-11", "2025-10-04", "2025-10-08")]
    ).tz_convert("UTC")
    assert list(M.day_type(stamps)) == [
        "major holiday",  # Christmas Day
        "day next to a major holiday",  # a Saturday, but the 26th is major in Texas
        "minor holiday",  # Veterans Day
        "weekend",
        "ordinary weekday",
    ]
