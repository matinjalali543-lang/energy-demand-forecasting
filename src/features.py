"""Feature construction, built so that leakage is structurally impossible.

A forecast is issued at 10:00 Central on day D for every hour of day D+1. Every
feature must be computable from what was known at 10:00 on D, and nothing else.
That is the whole contract, and three design choices defend it.

**One function, both paths.** `build_features` serves training and backtesting
alike, so there is no separate scoring path that could quietly drift.

**Slice before you look.** Demand features are computed from `past`, the rows
strictly before the issue time, and from nothing else. A later edit cannot
reach the future because the future is not in scope.

**Weather as it was known.** Weather is the one input that legitimately
describes the future, so it gets its own function, `weather_as_known`, with two
explicit modes:

* ``"observed"`` uses measured temperature for the target hours too: a perfect
  weather forecast. It is an *upper bound on accuracy*, labelled as such
  everywhere it is reported.
* ``"forecast"`` uses measured temperature up to the issue time and, after it,
  the archived weather forecast that had genuinely been published by then.

Lags are taken relative to the **target** hour. A lag L at target t reads the
demand at t - L, which is only known at the issue time if t - L is before it.
The latest target is 37-38 hours after the issue, so `lag_24h` -- "same hour
yesterday", the most common load-forecasting feature -- would read the future
for every target after 10:00. `FeatureSpec.check_leads` refuses it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Literal

import holidays
import numpy as np
import pandas as pd

import config

HOUR = pd.Timedelta(hours=1)

# 65 °F, the US degree-day convention. Below it buildings heat, above it they cool.
BALANCE_TEMP_C = 18.3

# The holidays on which Texas genuinely stops working. The others in the
# `holidays` calendar (Columbus Day, Texas Independence Day, ...) are kept as a
# separate, weaker flag: lumping them together would teach the model that a
# holiday is a mild event, which is false for Christmas and true for most of them.
MAJOR_HOLIDAYS = frozenset(
    {
        "New Year's Day",
        "Memorial Day",
        "Independence Day",
        "Labor Day",
        "Thanksgiving Day",
        "Friday After Thanksgiving",
        "Christmas Eve",
        "Christmas Day",
        "Day After Christmas",
    }
)

WeatherMode = Literal["observed", "forecast"]


@dataclass(frozen=True)
class FeatureSpec:
    """Which features to build. Defaults are the project's choices."""

    # Hours before the *target*. Every entry must exceed the longest lead time.
    lags_hours: tuple[int, ...] = (48, 168)
    # Windows, in hours, ending at the issue time. They describe "what demand
    # was doing when the forecast was made", so they are constant across the day.
    windows_hours: tuple[int, ...] = (24, 168, 672)
    weather: WeatherMode = "observed"

    def __post_init__(self) -> None:
        if not self.lags_hours or any(lag <= 0 for lag in self.lags_hours):
            raise ValueError(f"lags_hours must be positive and non-empty, got {self.lags_hours}")
        if any(window <= 0 for window in self.windows_hours):
            raise ValueError(f"windows_hours must be positive, got {self.windows_hours}")
        if self.weather not in ("observed", "forecast"):
            raise ValueError(f"weather must be 'observed' or 'forecast', got {self.weather!r}")

    @property
    def max_lookback_hours(self) -> int:
        """How much history must exist before the first issue time."""
        return max((*self.lags_hours, *self.windows_hours))

    def check_leads(self, max_lead_hours: float) -> None:
        """Refuse any lag that would read demand at or after the issue time."""
        offenders = [lag for lag in self.lags_hours if lag <= max_lead_hours]
        if offenders:
            raise ValueError(
                f"lag(s) {offenders} h are not longer than the {max_lead_hours:.0f} h lead "
                f"to the last target hour. A lag of L at a target t hours after the issue "
                "reads demand at issue + t - L, which has not been observed yet when "
                "L <= t. Use lags longer than the lead, e.g. 48 and 168."
            )


# --- Timing ------------------------------------------------------------------


def issue_time_for(day: pd.Timestamp | str) -> pd.Timestamp:
    """The issue time on local calendar `day`, in UTC."""
    local_day = pd.Timestamp(day)
    local_day = (
        local_day.tz_localize(config.LOCAL_TZ)
        if local_day.tz is None
        else local_day.tz_convert(config.LOCAL_TZ)
    )
    return (local_day.normalize() + pd.Timedelta(hours=config.ISSUE_HOUR_LOCAL)).tz_convert("UTC")


def target_index(issue_time: pd.Timestamp) -> pd.DatetimeIndex:
    """Every hour of the local day after the issue, in UTC.

    Usually 24 hours, but 23 on the spring DST change and 25 in the autumn:
    the forecast covers the local day the market trades, not 24 fixed UTC hours.
    """
    local = pd.Timestamp(issue_time).tz_convert(config.LOCAL_TZ)
    day_start = (local.normalize() + pd.Timedelta(days=1)).tz_localize(None)
    start = day_start.tz_localize(config.LOCAL_TZ)
    end = (day_start + pd.Timedelta(days=1)).tz_localize(config.LOCAL_TZ)
    return pd.date_range(
        start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left",
        name="timestamp_utc",
    )


def daily_issue_times(
    frame: pd.DataFrame,
    spec: FeatureSpec | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """One issue time per local day, wherever the history and the target day fit.

    `start`/`end` restrict the *target* days to [start, end).
    """
    spec = spec or FeatureSpec()
    earliest = frame.index[0] + HOUR * spec.max_lookback_hours
    first_day = frame.index[0].tz_convert(config.LOCAL_TZ).normalize().tz_localize(None)
    last_day = frame.index[-1].tz_convert(config.LOCAL_TZ).normalize().tz_localize(None)
    times = []
    for day in pd.date_range(first_day, last_day, freq="D"):
        issue = issue_time_for(day)
        targets = target_index(issue)
        if issue < earliest or targets[-1] > frame.index[-1]:
            continue
        if start is not None and targets[0] < start:
            continue
        if end is not None and targets[0] >= end:
            continue
        times.append(issue)
    return pd.DatetimeIndex(times, name="issue_time")


# --- Weather -------------------------------------------------------------------


def forecast_weather_as_known(
    frame: pd.DataFrame, issue_time: pd.Timestamp, until: pd.Timestamp
) -> pd.Series:
    """Temperature as a forecaster at `issue_time` could have known it, up to `until`.

    Before the issue time: measured temperature. After it: the archived
    forecast from the freshest model run that had *finished* by the issue time.

    Open-Meteo's `previous_day1` value for hour t comes from a run started 24 to
    47 hours before t; `previous_day2` from one started 48 to 71 hours before.
    With runs taking up to `FORECAST_RUN_LATENCY_HOURS` to publish, a day1
    value was certainly available at the issue time only if
    ``t - 24 h + latency <= issue``, i.e. a lead of at most 18 hours. Beyond
    that, day2 is used, which is safe out to a 42-hour lead. This is stricter
    than necessary for some runs, so it understates accuracy slightly.

    Each forecast is then **bias-corrected** against the airport readings the
    models were trained on: its mean error over the `WEATHER_BIAS_DAYS` days
    before the issue time is subtracted. Grid-model forecasts and airport
    thermometers differ systematically -- in the backtest year the day1
    forecast ran 0.7 °C cool and the day2 forecast 1.0 °C warm -- and a model
    trained on one but fed the other would read that offset as weather. Only
    hours before the issue time enter the correction, so it is leakage-safe;
    an operator would do the same.
    """
    latency = config.FORECAST_RUN_LATENCY_HOURS
    observed = frame["temp_c"]
    past = observed.loc[observed.index < issue_time]

    ahead = pd.date_range(issue_time, until, freq="h")
    lead = np.asarray((ahead - issue_time) / HOUR)
    if lead.size and lead.max() > 48 - latency:
        raise ValueError(
            f"a lead of {lead.max():.0f} h exceeds what the day2 forecast can cover "
            f"safely ({48 - latency} h)"
        )
    day1 = frame["temp_fc1_c"].reindex(ahead).to_numpy() - recent_bias(
        frame, "temp_fc1_c", issue_time)
    day2 = frame["temp_fc2_c"].reindex(ahead).to_numpy() - recent_bias(
        frame, "temp_fc2_c", issue_time)
    future = pd.Series(np.where(lead <= 24 - latency, day1, day2), index=ahead)
    return pd.concat([past, future])


def recent_bias(frame: pd.DataFrame, column: str, issue_time: pd.Timestamp) -> float:
    """Mean of forecast minus measured temperature over the days before `issue_time`.

    NaN when fewer than a week of paired hours exist, which blanks the
    forecast-weather features rather than correcting them with a guess.
    """
    window = frame.loc[
        (frame.index >= issue_time - pd.Timedelta(days=config.WEATHER_BIAS_DAYS))
        & (frame.index < issue_time)
    ]
    error = (window[column] - window["temp_c"]).dropna()
    return float(error.mean()) if len(error) >= 24 * 7 else float("nan")


def weather_as_known(
    frame: pd.DataFrame, issue_time: pd.Timestamp, until: pd.Timestamp, mode: WeatherMode
) -> pd.Series:
    """The temperature series the features may use, per `FeatureSpec.weather`."""
    if mode == "observed":
        # Deliberately not leakage-safe: a perfect weather forecast. Reported
        # as an upper bound, never as achievable accuracy.
        return frame["temp_c"].loc[:until]
    return forecast_weather_as_known(frame, issue_time, until)


# --- Calendar ----------------------------------------------------------------------


@cache
def _holidays_for(year: int) -> dict:
    calendar = holidays.country_holidays(
        config.HOLIDAY_COUNTRY, subdiv=config.HOLIDAY_SUBDIV, years=year
    )
    return {day: name.replace(" (observed)", "") for day, name in calendar.items()}


def _holiday_kind(day) -> str | None:
    name = _holidays_for(day.year).get(day)
    if name is None:
        return None
    # Two holidays on one date arrive as one "A; B" name.
    return "major" if MAJOR_HOLIDAYS.intersection(name.split("; ")) else "minor"


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features of the target hours, in local time.

    Not leakage: the calendar is known arbitrarily far ahead. Local, because
    people follow the local clock.
    """
    local = index.tz_convert(config.LOCAL_TZ)
    dates = local.date
    kinds = [_holiday_kind(d) for d in dates]
    before = [_holiday_kind(d + pd.Timedelta(days=1)) == "major" for d in dates]
    after = [_holiday_kind(d - pd.Timedelta(days=1)) == "major" for d in dates]
    return pd.DataFrame(
        {
            "hour": local.hour,
            "dayofweek": local.day_of_week,
            "month": local.month,
            "dayofyear": local.day_of_year,
            "is_weekend": np.asarray(local.day_of_week >= 5, dtype=int),
            "is_major_holiday": [int(k == "major") for k in kinds],
            "is_minor_holiday": [int(k == "minor") for k in kinds],
            "is_day_before_major_holiday": np.asarray(before, dtype=int),
            "is_day_after_major_holiday": np.asarray(after, dtype=int),
        },
        index=index,
    )


# --- Assembly ------------------------------------------------------------------------


def build_features(
    frame: pd.DataFrame, issue_time: pd.Timestamp, spec: FeatureSpec | None = None
) -> pd.DataFrame:
    """Features for every target hour of the day after `issue_time`.

    Indexed by the target timestamps (UTC). Rows whose history is incomplete
    come back with NaN, which training drops and scoring refuses rather than
    quietly imputing.
    """
    spec = spec or FeatureSpec()
    issue_time = pd.Timestamp(issue_time)
    if issue_time.tz is None:
        raise ValueError("issue_time must be timezone-aware")
    targets = target_index(issue_time)
    lead = np.asarray((targets - issue_time) / HOUR)
    spec.check_leads(lead.max())

    # The leakage barrier. Demand features below see only `past`.
    past = frame.loc[frame.index < issue_time, "demand_mw"]
    if past.empty:
        raise ValueError(f"no history before issue_time {issue_time}")

    columns: dict[str, np.ndarray] = {"lead_hours": lead}
    for lag in spec.lags_hours:
        # reindex, not .loc: an unobserved hour becomes NaN instead of raising.
        columns[f"demand_lag_{lag}h"] = past.reindex(targets - HOUR * lag).to_numpy()
    for window in spec.windows_hours:
        recent = past.loc[past.index >= issue_time - HOUR * window]
        for name, value in (("mean", recent.mean()), ("max", recent.max()), ("min", recent.min())):
            columns[f"demand_last_{window}h_{name}"] = np.full(len(targets), float(value))
    columns["demand_last_observed"] = np.full(len(targets), float(past.iloc[-1]))

    # Weather, from the series a forecaster at issue time could have had.
    weather = weather_as_known(frame, issue_time, targets[-1], spec.weather)
    temp = weather.reindex(targets).to_numpy()
    columns["temp_c"] = temp
    columns["cooling_degree_c"] = np.clip(temp - BALANCE_TEMP_C, 0, None)
    columns["heating_degree_c"] = np.clip(BALANCE_TEMP_C - temp, 0, None)
    # Buildings lag the weather: an evening after a hot afternoon still cools.
    columns["temp_lag_3h"] = weather.reindex(targets - 3 * HOUR).to_numpy()
    columns["temp_prev_24h_mean"] = np.array(
        [weather.loc[t - 24 * HOUR : t - HOUR].mean() for t in targets]
    )
    columns["temp_day_max"] = np.full(len(targets), np.nanmax(temp) if np.isfinite(temp).any()
                                      else np.nan)
    columns["temp_day_mean"] = np.full(len(targets), np.nanmean(temp) if np.isfinite(temp).any()
                                       else np.nan)

    features = pd.DataFrame(columns, index=targets)
    return features.join(calendar_features(targets))


def build_training_set(
    frame: pd.DataFrame, issue_times: Sequence[pd.Timestamp], spec: FeatureSpec | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Stack `build_features` over many issue times into one design matrix.

    Labels are read at the target hours: the supervision signal, and the only
    place the future is touched. Each target hour belongs to exactly one issue
    time, so the index stays unique. Rows with any missing feature or label are
    dropped, so nothing is trained on silently imputed values.
    """
    spec = spec or FeatureSpec()
    if len(issue_times) == 0:
        raise ValueError("issue_times is empty, so there is nothing to train on")
    blocks = [build_features(frame, issue, spec) for issue in issue_times]
    X = pd.concat(blocks)
    y = frame["demand_mw"].reindex(X.index)
    keep = X.notna().all(axis=1) & y.notna()
    return X.loc[keep], y.loc[keep]


# --- Checks ------------------------------------------------------------------------

# Columns that describe the world at their own timestamp. Corrupting them after
# the issue time must not change any feature.
OBSERVED_COLUMNS = ("demand_mw", "ercot_da_forecast_mw", "temp_c", "rh_pct")


def corrupt_unknowable(
    frame: pd.DataFrame, issue_time: pd.Timestamp, spec: FeatureSpec, rng: np.random.Generator
) -> pd.DataFrame:
    """A copy of `frame` with everything not knowable at `issue_time` replaced by noise.

    Observed columns are unknowable from the issue time on -- except measured
    temperature in "observed" weather mode, which is the declared oracle.
    Archived forecasts are unknowable where their run may not have finished:
    day1 beyond an 18-hour lead, day2 beyond 42.
    """
    corrupted = frame.copy()
    after = corrupted.index >= issue_time
    for column in OBSERVED_COLUMNS:
        if column == "temp_c" and spec.weather == "observed":
            continue
        if column in corrupted:
            corrupted.loc[after, column] = rng.normal(1e6, 1e5, int(after.sum()))
    latency = config.FORECAST_RUN_LATENCY_HOURS
    lead = (corrupted.index - issue_time) / HOUR
    for column, limit in (("temp_fc1_c", 24 - latency), ("temp_fc2_c", 48 - latency)):
        if column in corrupted:
            stale = np.asarray(lead > limit)
            corrupted.loc[stale, column] = rng.normal(1e6, 1e5, int(stale.sum()))
    return corrupted


def assert_no_future_reference(
    frame: pd.DataFrame,
    issue_time: pd.Timestamp,
    spec: FeatureSpec | None = None,
    rng: np.random.Generator | None = None,
) -> None:
    """Prove empirically that features use only what was known at `issue_time`.

    Builds features on the real frame and on one whose unknowable values are
    noise, and requires them to be identical. A feature that fails this leaks,
    however reasonable it looked.
    """
    spec = spec or FeatureSpec()
    rng = rng or np.random.default_rng(config.SEED)
    if not (frame.index >= issue_time).any():
        raise ValueError(f"nothing at or after {issue_time} to corrupt; the check would pass "
                         "vacuously")
    honest = build_features(frame, issue_time, spec)
    tampered = build_features(corrupt_unknowable(frame, issue_time, spec, rng), issue_time, spec)
    if not honest.equals(tampered):
        differing = [c for c in honest.columns if not honest[c].equals(tampered[c])]
        raise AssertionError(
            f"features changed when unknowable data was corrupted, so they leak. "
            f"Offending column(s): {differing}"
        )


def describe_features(spec: FeatureSpec | None = None) -> pd.DataFrame:
    """One row per feature: where its information comes from."""
    spec = spec or FeatureSpec()
    weather_source = (
        "measured weather at the target hour (perfect-forecast upper bound)"
        if spec.weather == "observed"
        else "archived weather forecast published before the issue time"
    )
    rows = [("lead_hours", "hours from issue time to target")]
    rows += [(f"demand_lag_{lag}h", "demand at target minus lag, always before issue")
             for lag in spec.lags_hours]
    rows += [(f"demand_last_{w}h_{s}", "window statistic ending at issue time")
             for w in spec.windows_hours for s in ("mean", "max", "min")]
    rows += [("demand_last_observed", "final demand observation before issue")]
    rows += [(name, weather_source) for name in (
        "temp_c", "cooling_degree_c", "heating_degree_c", "temp_lag_3h",
        "temp_prev_24h_mean", "temp_day_max", "temp_day_mean")]
    rows += [(name, "calendar (known arbitrarily far ahead)") for name in (
        "hour", "dayofweek", "month", "dayofyear", "is_weekend", "is_major_holiday",
        "is_minor_holiday", "is_day_before_major_holiday", "is_day_after_major_holiday")]
    return pd.DataFrame(rows, columns=["feature", "information source"]).set_index("feature")
