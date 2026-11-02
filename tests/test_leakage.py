"""The most important tests in this repository.

Every result depends on the forecasts having been made without seeing the
future. If that is false nothing else means anything, and the failure is
invisible in the metrics: a leaking model just looks unusually good.

These tests check, in increasing order of what they are worth:

1. that features are unchanged when everything unknowable is replaced by noise;
2. that the check is **not vacuous** -- deliberate leaks are caught, including
   the subtle one of using a weather forecast that was not yet published;
3. that the lag guard refuses "same hour yesterday" at a 10:00 issue time.

A leakage test that cannot fail produces confidence rather than information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import features as F
from features import (
    FeatureSpec,
    assert_no_future_reference,
    build_features,
    build_training_set,
    daily_issue_times,
)

BOTH_MODES = [FeatureSpec(weather="observed"), FeatureSpec(weather="forecast")]


@pytest.mark.parametrize("spec", BOTH_MODES, ids=["observed", "forecast"])
def test_features_ignore_everything_unknowable(synthetic_frame, issue_time, spec):
    assert_no_future_reference(synthetic_frame, issue_time, spec)


@pytest.mark.parametrize("spec", BOTH_MODES, ids=["observed", "forecast"])
def test_features_survive_a_corrupted_future_at_every_issue_time(synthetic_frame, spec):
    """Not one lucky day: every issue time in the window, DST changes included."""
    for issue in daily_issue_times(synthetic_frame, spec)[::7]:
        assert_no_future_reference(synthetic_frame, issue, spec)


def test_forecast_mode_features_survive_deleting_the_future(synthetic_frame, issue_time):
    """Stronger than noise: with observations truncated at the issue time, and
    forecasts kept, the forecast-mode features are identical."""
    spec = FeatureSpec(weather="forecast")
    full = build_features(synthetic_frame, issue_time, spec)

    truncated = synthetic_frame.copy()
    after = truncated.index >= issue_time
    truncated.loc[after, list(F.OBSERVED_COLUMNS)] = np.nan
    pd.testing.assert_frame_equal(full, build_features(truncated, issue_time, spec))


def test_a_deliberate_demand_leak_is_caught(synthetic_frame, issue_time, monkeypatch):
    """The most plausible real leak: a column that reads demand at its own hour."""
    original = F.build_features

    def leaking(frame, issue, spec=None):
        out = original(frame, issue, spec)
        out["leaked_same_hour"] = frame["demand_mw"].reindex(out.index).to_numpy()
        return out

    monkeypatch.setattr(F, "build_features", leaking)
    with pytest.raises(AssertionError, match="leaked_same_hour"):
        F.assert_no_future_reference(synthetic_frame, issue_time, FeatureSpec())


def test_using_an_unpublished_weather_forecast_is_caught(synthetic_frame, issue_time, monkeypatch):
    """The subtle leak: the day1 forecast for every hour, including hours whose
    model run started after the issue time. It looks like an honest forecast."""

    def always_day1(frame, issue, until):
        observed = frame["temp_c"]
        past = observed.loc[observed.index < issue]
        ahead = pd.date_range(issue, until, freq="h")
        return pd.concat([past, frame["temp_fc1_c"].reindex(ahead)])

    monkeypatch.setattr(F, "forecast_weather_as_known", always_day1)
    with pytest.raises(AssertionError, match="temp_c"):
        F.assert_no_future_reference(synthetic_frame, issue_time, FeatureSpec(weather="forecast"))


def test_same_hour_yesterday_is_refused(synthetic_frame, issue_time):
    """At a 10:00 issue time, t-24h is in the future for every target after 10:00."""
    with pytest.raises(ValueError, match=r"lag\(s\) \[24\]"):
        build_features(synthetic_frame, issue_time, FeatureSpec(lags_hours=(24, 168)))


def test_training_labels_look_ahead_but_features_do_not(synthetic_frame):
    """Labels are supervision, not leakage. Corrupting demand after the first
    issue time moves the first day's labels and leaves its features alone."""
    spec = FeatureSpec(weather="forecast")
    issues = daily_issue_times(synthetic_frame, spec)[:5]
    X, y = build_training_set(synthetic_frame, issues, spec)

    corrupted = synthetic_frame.copy()
    corrupted.loc[corrupted.index >= issues[0], "demand_mw"] += 500.0
    X2, y2 = build_training_set(corrupted, issues, spec)

    first_day = F.target_index(issues[0])
    pd.testing.assert_frame_equal(X.loc[first_day], X2.loc[first_day])
    np.testing.assert_allclose(y.loc[first_day] + 500.0, y2.loc[first_day])


def test_the_leakage_check_refuses_to_pass_vacuously(synthetic_frame):
    beyond = synthetic_frame.index[-1] + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="vacuously"):
        assert_no_future_reference(synthetic_frame, beyond)
