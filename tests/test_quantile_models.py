"""LightGBM quantile models: ordering, calibration, growth and reproducibility.

Smaller ensembles than the real backtest, so the suite stays fast; the
behaviour under test does not depend on the tree count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import metrics as M
import models as Mo
from features import FeatureSpec, build_training_set, daily_issue_times
from synthetic import make_synthetic_frame

FAST = {"n_estimators": 120}


def _split(frame: pd.DataFrame, split: str, test_days: int = 60):
    spec = FeatureSpec()
    issues = daily_issue_times(frame, spec)
    cut = pd.Timestamp(split, tz="UTC")
    train = build_training_set(frame, issues[issues < cut - pd.Timedelta(days=2)], spec)
    test = build_training_set(
        frame, issues[(issues >= cut) & (issues < cut + pd.Timedelta(days=test_days))], spec
    )
    return train, test


@pytest.fixture(scope="module")
def data():
    return _split(make_synthetic_frame(start="2023-01-01", days=500), "2024-03-01")


@pytest.fixture(scope="module")
def fitted(data):
    (X, y), _ = data
    return Mo.LightGBMQuantile(FAST).fit(X, y)


def test_quantiles_are_ordered_and_the_point_forecast_is_the_median(data, fitted):
    _, (X_test, _) = data
    out = fitted.predict(X_test)
    assert list(out.columns) == ["forecast", "p10", "p50", "p90"]
    assert (out["p10"] <= out["p50"]).all() and (out["p50"] <= out["p90"]).all()
    np.testing.assert_array_equal(out["forecast"], out["p50"])


def test_lightgbm_is_reproducible(data, fitted):
    (X, y), (X_test, _) = data
    again = Mo.LightGBMQuantile(FAST).fit(X, y)
    pd.testing.assert_frame_equal(fitted.predict(X_test), again.predict(X_test))


def test_relative_target_keeps_up_with_growth_that_breaks_an_mw_model():
    """Demand grows 5% a year, so the test period peaks above anything in
    training. Trees fitted on MW cannot forecast above their training range;
    the relative model can, because its output scales with the recent level."""
    frame = make_synthetic_frame(start="2023-01-01", days=1000)
    years = np.asarray((frame.index - frame.index[0]) / pd.Timedelta(days=365.25))
    frame["demand_mw"] *= 1 + 0.05 * years
    (X, y), (X_test, y_test) = _split(frame, "2025-06-01", test_days=90)
    assert y_test.max() > y.max()  # the premise of the test

    relative = Mo.LightGBMQuantile(FAST).fit(X, y).predict(X_test)["forecast"]
    absolute = Mo.LightGBMAbsolute(FAST).fit(X, y).predict(X_test)["forecast"]
    assert M.mape(y_test, relative) < 0.7 * M.mape(y_test, absolute)


def _calibration_split(X, y, days: int = 28):
    start = X.index.max() - pd.Timedelta(days=days)
    window = X.index > start
    return (X, y), (X.loc[window], y.loc[window])


def test_conformal_band_reaches_its_target_on_the_calibration_window(data):
    (X, y), _ = data
    (X_fit, y_fit), (X_cal, y_cal) = _calibration_split(X, y)
    model = Mo.LightGBMConformal(FAST).fit(X_fit, y_fit, calibration=(X_cal, y_cal))
    band = model.predict(X_cal)
    # By construction of the conformal quantile, at least 80% of the
    # calibration hours fall inside the adjusted band.
    assert M.coverage(y_cal, band["p10"], band["p90"]) >= 80.0


def test_conformal_widens_a_band_that_is_too_narrow(data):
    """Noisy temperatures in the calibration window (as weather forecasts are)
    make the raw band overconfident there, so the margin must be positive."""
    (X, y), _ = data
    (X_fit, y_fit), (X_cal, y_cal) = _calibration_split(X, y)
    noisy = X_cal.copy()
    rng = np.random.default_rng(0)
    for column in ("temp_c", "temp_lag_3h", "temp_day_max", "temp_day_mean"):
        noisy[column] = noisy[column] + rng.normal(0, 4.0, len(noisy))
    clean = Mo.LightGBMConformal(FAST).fit(X_fit, y_fit, calibration=(X_cal, y_cal))
    blurred = Mo.LightGBMConformal(FAST).fit(X_fit, y_fit, calibration=(noisy, y_cal))
    assert blurred.margin_ > clean.margin_


def test_conformal_trains_only_on_data_before_its_calibration_window(data, monkeypatch):
    (X, y), _ = data
    (X_fit, y_fit), (X_cal, y_cal) = _calibration_split(X, y)
    seen = {}
    original = Mo.LightGBMQuantile._fit_quantiles

    def spy(self, X_train, y_train):
        seen["max"] = X_train.index.max()
        return original(self, X_train, y_train)

    monkeypatch.setattr(Mo.LightGBMQuantile, "_fit_quantiles", spy)
    Mo.LightGBMConformal(FAST).fit(X_fit, y_fit, calibration=(X_cal, y_cal))
    assert seen["max"] < X_cal.index.min()


def test_conformal_refuses_to_fit_without_a_calibration_window(data):
    (X, y), _ = data
    with pytest.raises(ValueError, match="calibration window"):
        Mo.LightGBMConformal(FAST).fit(X, y)


def test_relative_inputs_drop_the_level_and_rescale_demand_columns(data):
    (X, _), _ = data
    rel = Mo.to_relative(X)
    assert Mo.LEVEL not in rel.columns
    np.testing.assert_allclose(rel["demand_lag_168h"], X["demand_lag_168h"] / X[Mo.LEVEL])
    pd.testing.assert_series_equal(rel["temp_c"], X["temp_c"])
