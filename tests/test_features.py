"""Shapes, timing, calendar and weather-selection behaviour of the features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
import features as F
from features import FeatureSpec, build_features, daily_issue_times, issue_time_for, target_index


def test_issue_time_is_ten_local_and_targets_cover_the_next_local_day():
    issue = issue_time_for("2025-07-15")
    assert issue.tz_convert(config.LOCAL_TZ).hour == config.ISSUE_HOUR_LOCAL
    targets = target_index(issue).tz_convert(config.LOCAL_TZ)
    assert len(targets) == 24
    assert targets[0] == pd.Timestamp("2025-07-16 00:00", tz=config.LOCAL_TZ)
    assert targets[-1] == pd.Timestamp("2025-07-16 23:00", tz=config.LOCAL_TZ)


@pytest.mark.parametrize("day, hours", [("2025-03-08", 23), ("2025-11-01", 25)])
def test_dst_days_have_23_or_25_target_hours(day, hours):
    """The forecast covers the local day the market trades, not 24 UTC hours."""
    assert len(target_index(issue_time_for(day))) == hours


def test_leads_run_from_14_to_37_hours():
    leads = build_features_for_july()["lead_hours"]
    assert leads.min() == 14 and leads.max() == 37


def build_features_for_july(spec: FeatureSpec | None = None) -> pd.DataFrame:
    from synthetic import make_synthetic_frame

    frame = make_synthetic_frame(start="2025-06-01", days=60)
    return build_features(frame, issue_time_for("2025-07-15"), spec)


def test_feature_columns_match_the_description(synthetic_frame, issue_time):
    out = build_features(synthetic_frame, issue_time)
    assert list(out.columns) == list(F.describe_features().index)
    assert out.notna().all().all()


def test_window_statistics_are_constant_across_the_day(synthetic_frame, issue_time):
    """They describe the moment of issue. If one varied by target hour, it would
    have been recomputed per target -- exactly how a window statistic starts leaking."""
    out = build_features(synthetic_frame, issue_time)
    for column in out.columns:
        if column.startswith("demand_last_"):
            assert out[column].nunique() == 1, column


def test_lag_168h_is_the_same_hour_last_week(synthetic_frame, issue_time):
    out = build_features(synthetic_frame, issue_time)
    week_ago = out.index - pd.Timedelta(hours=168)
    np.testing.assert_array_equal(
        out["demand_lag_168h"], synthetic_frame["demand_mw"].reindex(week_ago)
    )


def test_forecast_mode_uses_day1_early_and_day2_late(synthetic_frame, issue_time):
    """Day1 up to an 18-hour lead, day2 beyond: the freshest run surely published.
    Each is bias-corrected by its own recent error before use."""
    out = build_features(synthetic_frame, issue_time, FeatureSpec(weather="forecast"))
    limit = 24 - config.FORECAST_RUN_LATENCY_HOURS
    early = out["lead_hours"] <= limit
    assert early.any() and (~early).any()
    for rows, column in ((early, "temp_fc1_c"), (~early, "temp_fc2_c")):
        expected = (synthetic_frame[column].reindex(out.index[rows])
                    - F.recent_bias(synthetic_frame, column, issue_time))
        np.testing.assert_allclose(out.loc[rows, "temp_c"], expected)


def test_observed_mode_uses_measured_temperature(synthetic_frame, issue_time):
    out = build_features(synthetic_frame, issue_time, FeatureSpec(weather="observed"))
    np.testing.assert_allclose(out["temp_c"], synthetic_frame["temp_c"].reindex(out.index))


def test_degree_hours_split_at_the_balance_point(synthetic_frame, issue_time):
    out = build_features(synthetic_frame, issue_time)
    assert ((out["cooling_degree_c"] > 0) & (out["heating_degree_c"] > 0)).sum() == 0
    np.testing.assert_allclose(
        out["cooling_degree_c"] - out["heating_degree_c"], out["temp_c"] - F.BALANCE_TEMP_C
    )


def _calendar_of(day: str) -> pd.DataFrame:
    """Calendar features of local `day`, via the issue time the day before."""
    issue = issue_time_for(pd.Timestamp(day) - pd.Timedelta(days=1))
    return F.calendar_features(target_index(issue))


def test_christmas_and_its_neighbours_are_flagged():
    assert _calendar_of("2025-12-23")["is_day_before_major_holiday"].all()  # Eve is major
    assert _calendar_of("2025-12-25")["is_major_holiday"].all()
    december_27 = _calendar_of("2025-12-27")
    assert not december_27["is_major_holiday"].any()
    assert december_27["is_day_after_major_holiday"].all()  # the 26th is major in Texas


def test_minor_holidays_are_kept_apart_from_major_ones():
    calendar = _calendar_of("2025-11-11")  # Veterans Day (Texas has no Columbus Day)
    assert calendar["is_minor_holiday"].all()
    assert not calendar["is_major_holiday"].any()


def test_issue_times_leave_room_for_the_longest_lookback(synthetic_frame):
    spec = FeatureSpec()
    issues = daily_issue_times(synthetic_frame, spec)
    assert issues[0] - synthetic_frame.index[0] >= pd.Timedelta(hours=spec.max_lookback_hours)
    assert target_index(issues[-1])[-1] <= synthetic_frame.index[-1]


def test_forecast_bias_is_removed_using_only_the_past(synthetic_frame, issue_time):
    """A forecast that runs a steady 2 °C warm is corrected back to the honest one."""
    spec = FeatureSpec(weather="forecast")
    honest = build_features(synthetic_frame, issue_time, spec)
    warm = synthetic_frame.copy()
    warm[["temp_fc1_c", "temp_fc2_c"]] += 2.0
    corrected = build_features(warm, issue_time, spec)
    np.testing.assert_allclose(corrected["temp_c"], honest["temp_c"], atol=1e-9)


def test_too_little_history_blanks_rather_than_guesses_the_correction(synthetic_frame,
                                                                      issue_time):
    gappy = synthetic_frame.copy()
    gappy.loc[gappy.index < issue_time, "temp_fc1_c"] = np.nan
    out = build_features(gappy, issue_time, FeatureSpec(weather="forecast"))
    early = out["lead_hours"] <= 24 - config.FORECAST_RUN_LATENCY_HOURS
    assert out.loc[early, "temp_c"].isna().all()
