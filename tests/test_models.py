"""Model behaviour on synthetic data: each does what its docstring claims."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import metrics as M
from features import FeatureSpec, build_training_set, daily_issue_times
from models import ErcotPublished, SeasonalNaive, VanillaRegression
from synthetic import make_synthetic_frame


@pytest.fixture(scope="module")
def year_and_a_half():
    """Train on the first full year, test on the months after it."""
    frame = make_synthetic_frame(start="2023-01-01", days=540)
    spec = FeatureSpec()
    issues = daily_issue_times(frame, spec)
    split = pd.Timestamp("2024-01-01", tz="UTC")
    train = build_training_set(frame, issues[issues < split - pd.Timedelta(days=2)], spec)
    test = build_training_set(frame, issues[issues >= split], spec)
    return frame, train, test


def test_seasonal_naive_is_last_weeks_value(year_and_a_half):
    _, _, (X, _) = year_and_a_half
    np.testing.assert_array_equal(SeasonalNaive().predict(X)["forecast"], X["demand_lag_168h"])


def test_vanilla_regression_beats_seasonal_naive_on_weather_driven_demand(year_and_a_half):
    """The synthetic demand is driven by temperature and the calendar, which is
    exactly what the vanilla model captures and the naive rule cannot.

    The margin asked for is modest because this is one fit on one year, tested
    for six months without retraining: a trend fitted on a single year is
    weakly identified and drifts. The monthly-retrained backtest does better.
    """
    _, (X_train, y_train), (X_test, y_test) = year_and_a_half
    vanilla = VanillaRegression().fit(X_train, y_train).predict(X_test)["forecast"]
    naive = SeasonalNaive().predict(X_test)["forecast"]
    assert M.mape(y_test, vanilla) < 0.8 * M.mape(y_test, naive)


def test_vanilla_regression_refuses_a_month_it_never_saw(year_and_a_half):
    _, (X_train, y_train), (X_test, _) = year_and_a_half
    january_only = X_train["month"] == 1
    model = VanillaRegression().fit(X_train[january_only], y_train[january_only])
    with pytest.raises(ValueError, match="full year"):
        model.predict(X_test[X_test["month"] == 2])


def test_vanilla_regression_is_deterministic(year_and_a_half):
    _, (X_train, y_train), (X_test, _) = year_and_a_half
    first = VanillaRegression().fit(X_train, y_train).predict(X_test)
    second = VanillaRegression().fit(X_train, y_train).predict(X_test)
    pd.testing.assert_frame_equal(first, second)


def test_ercot_benchmark_replays_its_column_at_the_target_hour(year_and_a_half):
    frame, _, (X_test, _) = year_and_a_half
    out = ErcotPublished(frame).predict(X_test)["forecast"]
    np.testing.assert_array_equal(out, frame["ercot_da_forecast_mw"].reindex(X_test.index))


def test_weather_free_models_say_so():
    assert not SeasonalNaive.uses_weather
    assert not ErcotPublished.uses_weather
    assert VanillaRegression.uses_weather
