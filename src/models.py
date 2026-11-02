"""Forecasting models, simplest first.

Every model sees the same feature table from `features.build_features`, so
none of them can see past the leakage barrier, and every model is scored on
the same rows by the same backtest. A model's only job is `fit(X, y)` and
`predict(X)`; `predict` returns a frame with a ``forecast`` column, plus
``p10``/``p50``/``p90`` if the model is probabilistic.

The baselines are first-class, not footnotes. A model that cannot beat "same
hour last week" has not earned its complexity, and the README says so if it
happens.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import lightgbm as lgb
import numpy as np
import pandas as pd

import config

# Origin for the trend term. Any fixed date works; this one predates the data.
TREND_ORIGIN = pd.Timestamp("2021-01-01", tz="UTC")


class Model(ABC):
    name: ClassVar[str]
    # False for models that ignore weather: they give identical forecasts in
    # both weather arms, so the backtest runs them once.
    uses_weather: ClassVar[bool] = True
    # True for models whose fit() takes a `calibration=(X_cal, y_cal)` window:
    # the most recent labelled days, with features in the arm's weather mode.
    wants_calibration: ClassVar[bool] = False
    # The feature columns the model actually reads; None means all of them. A
    # row missing one of these cannot be forecast; a row missing anything else
    # can, so a station outage does not blank a model that ignores weather.
    required: ClassVar[tuple[str, ...] | None] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Model:
        return self

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame: ...


class SeasonalNaive(Model):
    """Same hour last week. The benchmark everything else must beat.

    A strong baseline for load, not a straw man: it carries the full weekly
    shape at zero parameters. Its weakness is weather -- a heat wave arriving
    this week is invisible to it -- which is exactly what the other models add.
    """

    name = "Seasonal naive (same hour last week)"
    uses_weather = False
    required = ("demand_lag_168h",)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"forecast": X["demand_lag_168h"].to_numpy()}, index=X.index)


def _one_hot(codes: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros((len(codes), size))
    out[np.arange(len(codes)), codes.astype(int)] = 1.0
    return out


class VanillaRegression(Model):
    """The GEFCom2012 "vanilla" benchmark (Hong, Pinson & Fan, 2014).

    Ordinary least squares on

        trend + month + weekday x hour
              + hour x (T, T², T³) + month x (T, T², T³)

    where T is temperature at the target hour. It is the standard reference
    model of the load-forecasting literature: everything it knows comes from
    weather and the calendar, with no recent demand at all. Major holidays are
    coded as Sundays, the usual treatment in that literature.

    Temperature is centred and scaled before the polynomial so the cubic
    terms stay numerically well-conditioned.
    """

    name = "Vanilla regression (GEFCom benchmark)"

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.months_seen_: set[int] = set()

    @staticmethod
    def design(X: pd.DataFrame) -> np.ndarray:
        hour = X["hour"].to_numpy()
        month = X["month"].to_numpy() - 1
        weekday = np.where(X["is_major_holiday"].to_numpy() == 1, 6, X["dayofweek"].to_numpy())
        t = (X["temp_c"].to_numpy() - 20.0) / 10.0
        trend = np.asarray((X.index - TREND_ORIGIN) / pd.Timedelta(days=365.25))

        hours = _one_hot(hour, 24)
        months = _one_hot(month, 12)
        blocks = [
            np.ones((len(X), 1)),
            trend[:, None],
            months[:, 1:],  # first level dropped: the intercept absorbs it
            _one_hot(weekday * 24 + hour, 168)[:, 1:],
        ]
        for power in (1, 2, 3):
            blocks.append(hours * (t**power)[:, None])
            blocks.append(months * (t**power)[:, None])
        return np.hstack(blocks)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> VanillaRegression:
        # lstsq, not an explicit inverse: a weekday x hour cell missing from a
        # short training window leaves the design rank-deficient, and lstsq
        # returns the minimum-norm solution instead of failing.
        self.coef_, *_ = np.linalg.lstsq(self.design(X), y.to_numpy(), rcond=None)
        self.months_seen_ = set(X["month"].unique().tolist())
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.coef_ is None:
            raise RuntimeError(f"{self.name} has not been fitted")
        # A month absent from training has all-zero month terms, so the model
        # would silently forecast it with untrained coefficients. Refuse.
        unseen = sorted(set(X["month"].unique().tolist()) - self.months_seen_)
        if unseen:
            raise ValueError(
                f"{self.name} was not trained on month(s) {unseen}; it needs at least "
                "a full year of training data"
            )
        return pd.DataFrame({"forecast": self.design(X) @ self.coef_}, index=X.index)


class ErcotPublished(Model):
    """ERCOT's own day-ahead forecast, replayed from EIA-930 (type `DF`).

    An external benchmark, not a model: a professional forecast of the same
    quantity, made with weather vendors and far more information than this
    project has. It reads its column at the *target* hour, which for any other
    input would be leakage; it is legitimate only because ERCOT publishes the
    forecast the day before. EIA-930 does not record the exact publication
    time, so the comparison is "same day-ahead horizon", not "same minute".
    """

    name = "ERCOT published day-ahead forecast"
    uses_weather = False
    required = ()

    def __init__(self, frame: pd.DataFrame) -> None:
        self.published = frame["ercot_da_forecast_mw"]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"forecast": self.published.reindex(X.index).to_numpy()},
                            index=X.index)


# --- Gradient boosting ---------------------------------------------------------------

# Fixed before any backtest was run and never tuned on it: tuning on the test
# year would make its score optimistic. Moderate depth, row and column
# subsampling, and enough trees at a slow rate to be stable.
LIGHTGBM_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": config.SEED,
    # Bit-identical results on rerun, at a small cost in speed.
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}

# The level every demand input and the target are divided by: mean demand
# over the four weeks before the issue time. Known at the issue time, so
# dividing by it is not leakage. Four weeks rather than one, because a single
# week's mean carries that week's weather into the target (a mild week before
# a heat wave makes the ratio jump for reasons the model cannot see). On
# synthetic data with 5%/year growth, the 4-week level scored 1.78% MAPE, the
# 1-week level 1.94%, and a model of demand in MW 3.92% -- see docs/decisions.md.
LEVEL = "demand_last_672h_mean"


def to_relative(X: pd.DataFrame) -> pd.DataFrame:
    """Express every demand input as a multiple of the last four weeks' mean demand.

    Trees cannot predict outside the range they were trained on: a model fitted
    on MW tops out near the highest demand in its training data. Texas set new
    records in 2026, about 6 GW above anything before. Predicting demand
    *relative* to the recent level lets the forecast scale with growth, while
    the trees still learn the shape of the day and the response to heat.
    """
    out = X.copy()
    level = X[LEVEL]
    for column in X.columns:
        if column.startswith("demand_") and column != LEVEL:
            out[column] = X[column] / level
    return out.drop(columns=LEVEL)


class LightGBMQuantile(Model):
    """LightGBM quantile regression for P10, P50 and P90.

    One model per quantile, each minimising pinball loss. The point forecast is
    the P50. Crossed quantiles -- possible because the three are fitted
    separately -- are sorted, so P10 <= P50 <= P90 on every row.
    """

    name = "LightGBM, P10/P50/P90"
    quantiles: ClassVar[tuple[float, ...]] = (0.1, 0.5, 0.9)
    relative: ClassVar[bool] = True

    def __init__(self, params: dict | None = None) -> None:
        self.params = {**LIGHTGBM_PARAMS, **(params or {})}
        self.models_: dict[float, lgb.LGBMRegressor] = {}
        self.columns_: list[str] = []

    def _inputs(self, X: pd.DataFrame) -> pd.DataFrame:
        return to_relative(X) if self.relative else X

    def _fit_quantiles(self, X: pd.DataFrame, y: pd.Series) -> None:
        inputs = self._inputs(X)
        target = y / X[LEVEL] if self.relative else y
        self.columns_ = list(inputs.columns)
        self.models_ = {
            q: lgb.LGBMRegressor(objective="quantile", alpha=q, **self.params).fit(
                inputs, target
            )
            for q in self.quantiles
        }

    def _raw_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        """Sorted quantile predictions, in the model's target units."""
        if not self.models_:
            raise RuntimeError(f"{self.name} has not been fitted")
        inputs = self._inputs(X)[self.columns_]
        raw = np.column_stack([self.models_[q].predict(inputs) for q in self.quantiles])
        return np.sort(raw, axis=1)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LightGBMQuantile:
        self._fit_quantiles(X, y)
        return self

    def _to_frame(self, X: pd.DataFrame, raw: np.ndarray) -> pd.DataFrame:
        scale = X[LEVEL].to_numpy()[:, None] if self.relative else 1.0
        mw = raw * scale
        out = pd.DataFrame(mw, index=X.index,
                           columns=[f"p{round(q * 100)}" for q in self.quantiles])
        if "p50" in out:
            out.insert(0, "forecast", out["p50"])
        else:
            out.insert(0, "forecast", mw[:, len(self.quantiles) // 2])
        return out

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._to_frame(X, self._raw_quantiles(X))


class LightGBMConformal(LightGBMQuantile):
    """The same model, with its P10-P90 band calibrated on recent forecasts.

    Conformalized quantile regression (Romano, Patterson & Candès, 2019). The
    quantile models are fitted on everything *before* a calibration window --
    the last `CALIBRATION_DAYS` days before the issue time -- and scored on it.
    Each calibration hour gives a conformity score: how far the actual fell
    outside the band (negative if inside). The band is then widened, or
    narrowed, by the score's (1 - alpha) quantile, so that about 80% of
    calibration hours would have fallen inside.

    Why it matters here: the quantile models learn from *measured* weather, so
    they have never seen a weather forecast miss. Fed real forecasts, their
    band is too narrow. The calibration window uses features in the arm's own
    weather mode, so in the forecast arm the correction includes weather
    forecast error. Scores are in relative units, so the width scales with the
    level of demand rather than being one MW figure for July and January alike.

    The coverage guarantee of conformal prediction assumes exchangeable data,
    which hourly load is not; the correction is approximate, and the backtest
    measures how well it actually does.
    """

    name = "LightGBM, P10/P50/P90 + conformal"
    wants_calibration = True
    target_coverage: ClassVar[float] = 0.8

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.margin_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, calibration=None) -> LightGBMConformal:
        if calibration is None:
            raise ValueError(f"{self.name} needs a calibration window")
        X_cal, y_cal = calibration
        if len(X_cal) == 0:
            raise ValueError("the calibration window is empty")
        # Fit only on what precedes the calibration window, so its scores are
        # genuinely out of sample.
        before = X.index < X_cal.index.min()
        self._fit_quantiles(X.loc[before], y.loc[before])

        raw = self._raw_quantiles(X_cal)
        actual = (y_cal / X_cal[LEVEL]).to_numpy()
        lower, upper = raw[:, 0], raw[:, -1]
        scores = np.maximum(lower - actual, actual - upper)
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * self.target_coverage) / n)
        self.margin_ = float(np.quantile(scores, level, method="higher"))
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.margin_ is None:
            raise RuntimeError(f"{self.name} has not been fitted")
        raw = self._raw_quantiles(X)
        raw[:, 0] -= self.margin_
        raw[:, -1] += self.margin_
        return self._to_frame(X, raw)


class LightGBMAbsolute(LightGBMQuantile):
    """Ablation: the median model trained on demand in MW, not relative demand.

    Exists only to measure what predicting relative demand is worth. Expected
    to fail precisely when demand exceeds anything in its training data.
    """

    name = "LightGBM, MW target (ablation)"
    quantiles = (0.5,)
    relative = False
