"""The rolling-origin engine: what is trained on, what is scored, and when.

Feature-level leakage is covered in test_leakage.py. These tests cover the
other way a backtest cheats: training on labels from after the forecast was
issued.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import backtest
import config
from features import target_index
from models import Model, SeasonalNaive
from synthetic import make_synthetic_frame

START = pd.Timestamp("2024-06-01", tz=config.LOCAL_TZ)
END = pd.Timestamp("2024-07-15", tz=config.LOCAL_TZ)


class Spy(Model):
    """Records what it was trained on, and forecasts last week's value."""

    name = "spy"
    log: list = []

    def fit(self, X, y):
        Spy.log.append({"train_max": X.index.max(), "rows": len(X)})
        return self

    def predict(self, X):
        Spy.log[-1]["predicted"] = X.index
        return pd.DataFrame({"forecast": X["demand_lag_168h"].to_numpy()}, index=X.index)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_synthetic_frame(start="2024-03-01", days=140)


@pytest.fixture(scope="module")
def results(frame) -> pd.DataFrame:
    Spy.log = []
    return backtest.rolling_origin(frame, START, END, models=lambda f: [SeasonalNaive(), Spy()])


def test_no_training_label_is_at_or_after_any_issue_time_it_forecasts(results):
    issue_of = results[results["model"] == "spy"].set_index("target_time")["issue_time"]
    assert len(Spy.log) == 2  # June and the first half of July
    for entry in Spy.log:
        first_issue = issue_of.loc[entry["predicted"]].min()
        assert entry["train_max"] < first_issue


def test_every_target_day_is_forecast_exactly_once_per_model(results):
    days = pd.date_range(START.tz_localize(None), END.tz_localize(None), freq="D",
                         inclusive="left")
    expected = sum(len(target_index(day.tz_localize(config.LOCAL_TZ)
                                    - pd.Timedelta(days=1) + pd.Timedelta(hours=10)))
                   for day in days)
    for _, group in results.groupby("model"):
        assert len(group) == expected
        assert group["target_time"].is_unique


def test_models_retrain_monthly_and_training_grows(results):
    cutoffs = sorted(results["trained_until"].unique())
    assert len(cutoffs) == 2
    assert Spy.log[1]["rows"] > Spy.log[0]["rows"]


def test_issue_is_ten_local_the_day_before(results):
    local_issue = pd.DatetimeIndex(results["issue_time"]).tz_convert(config.LOCAL_TZ)
    local_target = pd.DatetimeIndex(results["target_time"]).tz_convert(config.LOCAL_TZ)
    assert (local_issue.hour == config.ISSUE_HOUR_LOCAL).all()
    assert (local_target.normalize() - local_issue.normalize() == pd.Timedelta(days=1)).all()


def test_a_history_gap_leaves_rows_unforecast_rather_than_imputed(frame):
    """A missing week of demand blanks the lag-168 feature for the week after."""
    gappy = frame.copy()
    gap = (gappy.index >= pd.Timestamp("2024-06-03", tz="UTC")) & (
        gappy.index < pd.Timestamp("2024-06-06", tz="UTC"))
    gappy.loc[gap, "demand_mw"] = np.nan
    out = backtest.rolling_origin(gappy, START, END, models=lambda f: [SeasonalNaive()])
    assert out["forecast"].isna().sum() > 0
    assert out["forecast"].notna().sum() > 0.8 * len(out)


def test_weather_free_models_run_once_across_arms(frame):
    """They are identical in both arms, so only the observed arm runs them."""
    observed = backtest.rolling_origin(frame, START, END, weather="observed",
                                       models=lambda f: [SeasonalNaive()])
    forecast = backtest.rolling_origin(frame, START, END, weather="forecast",
                                       models=lambda f: [SeasonalNaive()])
    assert set(observed["arm"]) == {backtest.NO_WEATHER_ARM}
    assert forecast.empty


class CalibrationSpy(Spy):
    """Records the calibration window the backtest hands it."""

    name = "calibration spy"
    wants_calibration = True
    windows: list = []

    def fit(self, X, y, calibration=None):
        X_cal, _ = calibration
        CalibrationSpy.windows.append({"index": X_cal.index, "temp": X_cal["temp_c"]})
        return super().fit(X, y)


@pytest.mark.parametrize("arm", ["observed", "forecast"])
def test_calibration_window_is_recent_known_and_in_the_arms_weather(frame, arm):
    CalibrationSpy.windows = []
    out = backtest.rolling_origin(frame, START, END, weather=arm,
                                  models=lambda f: [CalibrationSpy()])
    cutoffs = sorted(out["trained_until"].unique())
    assert len(CalibrationSpy.windows) == len(cutoffs)
    column = "temp_c" if arm == "observed" else None
    for window, cutoff in zip(CalibrationSpy.windows, cutoffs, strict=True):
        index = window["index"]
        assert index.max() < cutoff  # every label already observed
        assert index.min() >= cutoff - pd.Timedelta(days=config.CALIBRATION_DAYS)
        assert len(index) > 24 * (config.CALIBRATION_DAYS - 3)
        if column:  # observed arm: measured temperature
            np.testing.assert_allclose(window["temp"], frame[column].reindex(index))
        else:  # forecast arm: never the measured value
            assert not np.allclose(window["temp"], frame["temp_c"].reindex(index))


def test_backtest_is_deterministic(frame):
    first = backtest.rolling_origin(frame, START, END, models=lambda f: [SeasonalNaive()])
    second = backtest.rolling_origin(frame, START, END, models=lambda f: [SeasonalNaive()])
    pd.testing.assert_frame_equal(first, second)


def test_a_weather_outage_blanks_only_the_models_that_read_weather(frame):
    """A model is skipped only for hours missing an input it actually reads."""

    class ReadsEverything(Model):
        name = "reads every input"

        def predict(self, X):
            return pd.DataFrame({"forecast": 1000.0 * X["temp_c"].to_numpy()}, index=X.index)

    outage = frame.copy()
    hole = (outage.index >= pd.Timestamp("2024-06-10 12:00", tz="UTC")) & (
        outage.index < pd.Timestamp("2024-06-10 18:00", tz="UTC"))
    outage.loc[hole, "temp_c"] = np.nan
    out = backtest.rolling_origin(outage, START, END,
                                  models=lambda f: [SeasonalNaive(), ReadsEverything()])
    in_hole = out["target_time"].isin(outage.index[hole])
    naive = out[in_hole & (out["model"] == SeasonalNaive.name)]
    reader = out[in_hole & (out["model"] == ReadsEverything.name)]
    assert len(naive) == 6 and naive["forecast"].notna().all()
    assert len(reader) == 6 and reader["forecast"].isna().all()
