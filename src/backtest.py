"""Rolling-origin backtest: the core of the project's credibility.

    python src/backtest.py               # real data -> results/*.csv
    python src/backtest.py --synthetic   # synthetic data, prints only, writes nothing

For each month of the backtest window, every model is retrained on all data
known before that month's first forecast, then forecasts each day of the month
from its 10:00 issue time. Nothing is ever scored on data its model trained on,
and `rolling_origin` asserts it rather than trusting the arithmetic.

Two weather arms, reported side by side:

* **observed** -- measured weather at the target hours, i.e. a perfect weather
  forecast. An upper bound on accuracy, labelled as such.
* **forecast** -- the archived weather forecast published before each issue
  time. What the model would really have scored. Models are *trained* on
  measured weather in both arms (the history a real operator has) and only
  *fed* forecasts when predicting, which is how this is done in practice.

Models that ignore weather (seasonal naive, ERCOT's published forecast) give
the same numbers in both arms, so they run once.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from itertools import pairwise

import numpy as np
import pandas as pd

import config
import metrics
from features import FeatureSpec, build_features, daily_issue_times
from models import (
    ErcotPublished,
    LightGBMAbsolute,
    LightGBMConformal,
    LightGBMQuantile,
    Model,
    SeasonalNaive,
    VanillaRegression,
)

# Weather arm label for models that ignore weather. Not "n/a": pandas reads that
# back from CSV as missing.
NO_WEATHER_ARM = "not used"


def default_models(frame: pd.DataFrame) -> list[Model]:
    """The comparison set: baselines, learned models, an ablation, the benchmark."""
    return [
        SeasonalNaive(),
        VanillaRegression(),
        LightGBMQuantile(),
        LightGBMConformal(),
        LightGBMAbsolute(),
        ErcotPublished(frame),
    ]


def feature_table(
    frame: pd.DataFrame, issue_times: Sequence[pd.Timestamp], spec: FeatureSpec
) -> tuple[pd.DataFrame, pd.Series]:
    """Features for every target hour of every issue time, NaN rows kept.

    Returns the features and, aligned with them, the issue time each row was
    built at. Rows are dropped later -- from training if incomplete, from
    scoring if the model cannot forecast them -- so each step can count them.
    """
    blocks = [build_features(frame, issue, spec) for issue in issue_times]
    X = pd.concat(blocks)
    issued = pd.Series(
        np.repeat(np.asarray(issue_times, dtype="datetime64[ns]"), [len(b) for b in blocks]),
        index=X.index, name="issue_time",
    ).dt.tz_localize("UTC")
    return X, issued


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Local month boundaries from `start` to `end`, as UTC timestamps."""
    local = pd.date_range(
        start.tz_convert(config.LOCAL_TZ).tz_localize(None),
        end.tz_convert(config.LOCAL_TZ).tz_localize(None),
        freq="MS",
    )
    bounds = [b.tz_localize(config.LOCAL_TZ).tz_convert("UTC") for b in local]
    if not bounds or bounds[0] > start:
        bounds.insert(0, start.tz_convert("UTC"))
    if bounds[-1] < end:
        bounds.append(end.tz_convert("UTC"))
    return bounds


def rolling_origin(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    models: Callable[[pd.DataFrame], list[Model]] = default_models,
    weather: str = "observed",
) -> pd.DataFrame:
    """Backtest every model over target days in [start, end), retraining monthly.

    Returns tidy rows: target_time, issue_time, trained_until, model, arm,
    actual, forecast (and p10/p50/p90 for probabilistic models).
    """
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    train_spec = FeatureSpec(weather="observed")
    test_spec = FeatureSpec(weather=weather)

    all_issues = daily_issue_times(frame, train_spec, end=end)
    test_issues = daily_issue_times(frame, test_spec, start=start, end=end)
    if len(test_issues) == 0:
        raise ValueError(f"no complete target days between {start} and {end}")
    # Features in the arm's weather mode start a calibration window early, so
    # the first month's conformal model has recent forecasts to calibrate on.
    calibration = pd.Timedelta(days=config.CALIBRATION_DAYS)
    arm_issues = daily_issue_times(frame, test_spec, start=start - calibration, end=end)

    # Built once and sliced per month: features do not depend on the cutoff.
    X_train_all, _ = feature_table(frame, all_issues[all_issues < test_issues[-1]], train_spec)
    X_arm_all, arm_issued = feature_table(frame, arm_issues, test_spec)
    y = frame["demand_mw"]

    rows = []
    for month_start, month_end in pairwise(_month_starts(start, end)):
        in_month = (X_arm_all.index >= month_start) & (X_arm_all.index < month_end)
        if not in_month.any():
            continue
        X_test = X_arm_all.loc[in_month]
        cutoff = arm_issued.loc[in_month].min()

        # The calibration window: the last few weeks of target hours already
        # observed at the cutoff, with features as the arm would have had them.
        y_arm = y.reindex(X_arm_all.index)
        in_window = (
            (X_arm_all.index >= cutoff - calibration)
            & (X_arm_all.index < cutoff)
            & X_arm_all.notna().all(axis=1).to_numpy()
            & y_arm.notna().to_numpy()
        )
        calibration_set = (X_arm_all.loc[in_window], y_arm.loc[in_window])

        # Train on every target hour that had been observed by the month's
        # first issue time, and nothing else.
        y_train = y.reindex(X_train_all.index)
        usable = (
            (X_train_all.index < cutoff)
            & X_train_all.notna().all(axis=1).to_numpy()
            & y_train.notna().to_numpy()
        )
        X_train, y_train = X_train_all.loc[usable], y_train.loc[usable]
        if X_train.index.max() >= cutoff:  # the time-split leakage guard
            raise AssertionError(f"training label at {X_train.index.max()} >= issue {cutoff}")

        for model in models(frame):
            if not model.uses_weather and weather != "observed":
                continue  # identical in both arms; run once
            if model.wants_calibration:
                model.fit(X_train, y_train, calibration=calibration_set)
            else:
                model.fit(X_train, y_train)
            # A model cannot forecast a row missing a feature it reads (a
            # history gap); those rows come back NaN and are counted, not imputed.
            needed = list(X_test.columns if model.required is None else model.required)
            complete = X_test[needed].notna().all(axis=1)
            predicted = pd.DataFrame(index=X_test.index, columns=["forecast"], dtype=float)
            if complete.any():
                out = model.predict(X_test.loc[complete])
                predicted = predicted.reindex(columns=out.columns)
                predicted.loc[complete, out.columns] = out.to_numpy()
            block = predicted.assign(
                target_time=X_test.index,
                issue_time=arm_issued.loc[in_month].to_numpy(),
                trained_until=cutoff,
                model=model.name,
                arm=weather if model.uses_weather else NO_WEATHER_ARM,
                actual=y.reindex(X_test.index).to_numpy(),
            )
            rows.append(block.reset_index(drop=True))

    columns = ["target_time", "issue_time", "trained_until", "model", "arm", "actual", "forecast"]
    if not rows:  # e.g. only weather-free models, in the forecast arm
        return pd.DataFrame(columns=columns)
    out = pd.concat(rows, ignore_index=True)
    extra = [c for c in out.columns if c not in columns]
    return out[columns + extra]


BREAKDOWNS = ("season", "peak", "hour", "temp_bin", "month", "day_type")


def summary_table(results: pd.DataFrame, scale: float) -> pd.DataFrame:
    """One row per model and weather arm.

    `skill_vs_naive_%` is the reduction in MAE against seasonal naive,
    measured on exactly the hours both were scored on.
    """
    naive = results[results["model"] == SeasonalNaive.name].set_index("target_time")
    rows = []
    for (model, arm), group in results.groupby(["model", "arm"], sort=False):
        row = {"model": model, "weather": arm, **metrics.summarise(group, scale)}
        if not naive.empty:
            both = group.set_index("target_time")[["actual", "forecast"]].join(
                naive["forecast"].rename("naive"), how="inner").dropna()
            row["skill_vs_naive_%"] = 100 * (
                1 - metrics.mae(both["actual"], both["forecast"])
                / metrics.mae(both["actual"], both["naive"]))
        rows.append(row)
    return pd.DataFrame(rows)


def tables_from(results: pd.DataFrame, frame: pd.DataFrame,
                start: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """The summary and every breakdown, from the hourly predictions."""
    scale = metrics.seasonal_naive_scale(frame.loc[frame.index < start, "demand_mw"])
    tables = {"predictions": results, "summary": summary_table(results, scale)}
    for by in BREAKDOWNS:
        parts = []
        for (model, arm), group in results.groupby(["model", "arm"], sort=False):
            part = metrics.breakdown(group, by, temp=frame["temp_c"])
            parts.append(part.assign(model=model, weather=arm).reset_index(names=by))
        tables[f"breakdown_{by}"] = pd.concat(parts, ignore_index=True)
    tables["summary"].attrs["mase_scale_mw"] = scale
    return tables


def run(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Both weather arms, the summary and the breakdowns."""
    arms = ["observed"]
    forecast_cols = frame.loc[frame.index >= start, ["temp_fc1_c", "temp_fc2_c"]]
    if forecast_cols.notna().any().all():
        arms.append("forecast")
    else:
        print("WARNING: no archived weather forecasts in the backtest window; "
              "running the observed-weather arm only.")

    results = pd.concat([rolling_origin(frame, start, end, weather=arm) for arm in arms],
                        ignore_index=True)
    return tables_from(results, frame, start)


def load_predictions() -> pd.DataFrame:
    """The hourly predictions written by the last full run."""
    out = pd.read_csv(config.PROCESSED_DIR / "predictions.csv")
    for column in ("target_time", "issue_time", "trained_until"):
        out[column] = pd.to_datetime(out[column], utc=True)
    return out


def _print_summary(summary: pd.DataFrame) -> None:
    columns = ["model", "weather", "MAPE_%", "MASE", "skill_vs_naive_%", "peak_mape_%",
               "coverage_p10_p90_%", "hours_scored"]
    shown = summary[[c for c in columns if c in summary.columns]]
    rounding = {"MAPE_%": 2, "MASE": 3, "skill_vs_naive_%": 1, "peak_mape_%": 2,
                "coverage_p10_p90_%": 1}
    with pd.option_context("display.width", 160, "display.max_colwidth", 36):
        print(shown.round(rounding).to_string(index=False, na_rep="-"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rolling-origin backtest.")
    parser.add_argument("--synthetic", action="store_true",
                        help="run on seeded synthetic data; prints only, writes nothing")
    parser.add_argument("--from-predictions", action="store_true",
                        help="rebuild tables and figures from the last run's predictions, "
                             "without refitting any model")
    args = parser.parse_args(argv)

    if args.synthetic:
        from synthetic import make_synthetic_frame

        # Two years, so every calendar month is in training before it is tested.
        frame = make_synthetic_frame(start="2023-01-01", days=730)
        start = pd.Timestamp("2024-10-01", tz=config.LOCAL_TZ)
        end = pd.Timestamp("2025-01-01", tz=config.LOCAL_TZ)
        tables = run(frame, start, end)
        print(f"Synthetic data, {start:%Y-%m-%d} to {end:%Y-%m-%d}. "
              "These numbers measure the generator's noise, not Texas.\n")
        _print_summary(tables["summary"])
        return 0

    if not config.PROCESSED_FILE.exists():
        print(f"{config.PROCESSED_FILE} not found. Run `python src/data_download.py` first.")
        return 1
    frame = pd.read_csv(config.PROCESSED_FILE, index_col=0, parse_dates=True)
    if args.from_predictions:
        tables = tables_from(load_predictions(), frame, config.BACKTEST_START)
    else:
        tables = run(frame, config.BACKTEST_START, config.BACKTEST_END)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        # Hourly predictions are large and fully reproducible: kept out of git.
        target = (config.PROCESSED_DIR if name == "predictions" else config.RESULTS_DIR)
        table.to_csv(target / f"{name}.csv", index=False, float_format="%.4f")
    print(f"MASE scale (seasonal-naive MAE before the backtest): "
          f"{tables['summary'].attrs['mase_scale_mw']:,.0f} MW\n")
    _print_summary(tables["summary"])
    print(f"\nwrote {config.RESULTS_DIR}")

    import plots  # matplotlib only when figures are wanted

    for path in plots.render_all(tables["predictions"], frame, config.FIGURES_DIR):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
