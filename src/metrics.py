"""Forecast accuracy, in the forms a client and a reviewer each need.

Point metrics:

* **MAPE** -- the number utilities quote. Meaningful here because system demand
  never comes near zero (minimum ~30 GW).
* **MAE / RMSE** -- in MW, the unit an operator plans in. RMSE punishes the
  occasional large miss, which is what actually costs money.
* **MASE** -- MAE divided by the MAE that "same hour last week" scores on the
  *training* history. Below 1 means better than the naive rule. The scale is
  fixed before the backtest starts, so it is identical for every model.
* **Peak errors** -- each day's highest hour, in magnitude and in timing. The
  daily peak sets capacity and price, so an average error can hide what matters.

Probabilistic metrics:

* **Pinball loss** -- the proper score for a quantile: low only if the quantile
  is both well placed and sharp.
* **Coverage** -- how often the P10-P90 band contains the actual. Should be
  about 80%; much higher means the band is too wide to be useful, lower means
  it is overconfident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

# Local hours treated as the system peak. Texas demand peaks in the late
# afternoon on hot days; prices and scarcity concentrate here.
PEAK_HOURS = tuple(range(14, 20))

SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}


def _clean(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"actual has shape {actual.shape} but predicted has {predicted.shape}")
    keep = np.isfinite(actual) & np.isfinite(predicted)
    return actual[keep], predicted[keep]


def mae(actual, predicted) -> float:
    a, p = _clean(actual, predicted)
    return float(np.mean(np.abs(a - p))) if a.size else float("nan")


def rmse(actual, predicted) -> float:
    a, p = _clean(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2))) if a.size else float("nan")


def bias(actual, predicted) -> float:
    """Mean of forecast minus actual. Positive means the forecast runs high."""
    a, p = _clean(actual, predicted)
    return float(np.mean(p - a)) if a.size else float("nan")


def mape(actual, predicted) -> float:
    """Mean absolute percentage error, in percent."""
    a, p = _clean(actual, predicted)
    if not a.size:
        return float("nan")
    if np.any(a <= 0):
        raise ValueError("MAPE needs strictly positive actuals")
    return float(np.mean(np.abs(a - p) / a) * 100)


def seasonal_naive_scale(demand: pd.Series, season_hours: int = 168) -> float:
    """In-sample MAE of "same hour last week": the MASE denominator."""
    lagged = demand.shift(freq=pd.Timedelta(hours=season_hours)).reindex(demand.index)
    return mae(demand, lagged)


def mase(actual, predicted, scale: float) -> float:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"MASE scale must be positive, got {scale}")
    return mae(actual, predicted) / scale


def pinball_loss(actual, quantile_forecast, q: float) -> float:
    """Mean pinball (quantile) loss for quantile level `q` in (0, 1)."""
    if not 0 < q < 1:
        raise ValueError(f"q must be in (0, 1), got {q}")
    a, p = _clean(actual, quantile_forecast)
    if not a.size:
        return float("nan")
    diff = a - p
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(actual, lower, upper) -> float:
    """Share of actuals inside [lower, upper], in percent."""
    a = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    keep = np.isfinite(a) & np.isfinite(lo) & np.isfinite(hi)
    if not keep.any():
        return float("nan")
    return float(np.mean((a[keep] >= lo[keep]) & (a[keep] <= hi[keep])) * 100)


def daily_peak_errors(results: pd.DataFrame) -> dict[str, float]:
    """Error in each local day's peak: how high, and when.

    `results` has a UTC `target_time` column plus `actual` and `forecast`.
    Peak magnitude error compares the two daily maxima wherever they fall;
    timing error is the gap in hours between the hours they fall in.
    """
    work = results.dropna(subset=["actual", "forecast"]).copy()
    if work.empty:
        return {"peak_mape_%": float("nan"), "peak_mae_mw": float("nan"),
                "peak_timing_h": float("nan")}
    local = pd.DatetimeIndex(work["target_time"]).tz_convert(config.LOCAL_TZ)
    work["day"] = local.date
    work["hour"] = local.hour
    rows = []
    for _, day in work.groupby("day"):
        actual_peak = day.loc[day["actual"].idxmax()]
        forecast_peak = day.loc[day["forecast"].idxmax()]
        rows.append((actual_peak["actual"], forecast_peak["forecast"],
                     abs(actual_peak["hour"] - forecast_peak["hour"])))
    a, p, timing = (np.array(column, dtype=float) for column in zip(*rows, strict=True))
    return {
        "peak_mape_%": float(np.mean(np.abs(p - a) / a) * 100),
        "peak_mae_mw": float(np.mean(np.abs(p - a))),
        "peak_timing_h": float(np.mean(timing)),
    }


def summarise(results: pd.DataFrame, scale: float) -> dict[str, float]:
    """Headline metrics for one model's backtest rows."""
    a, p = results["actual"], results["forecast"]
    local_hour = pd.DatetimeIndex(results["target_time"]).tz_convert(config.LOCAL_TZ).hour
    in_peak = np.asarray(local_hour.isin(PEAK_HOURS))
    summary = {
        "MAPE_%": mape(a, p),
        "MAE_mw": mae(a, p),
        "RMSE_mw": rmse(a, p),
        "bias_mw": bias(a, p),
        "MASE": mase(a, p, scale),
        "MAPE_peak_hours_%": mape(a[in_peak], p[in_peak]),
        **daily_peak_errors(results),
        "hours_scored": int((np.isfinite(a) & np.isfinite(p)).sum()),
    }
    if {"p10", "p90"}.issubset(results.columns):
        summary["coverage_p10_p90_%"] = coverage(a, results["p10"], results["p90"])
        summary["pinball_mw"] = float(np.mean([
            pinball_loss(a, results[f"p{int(q * 100)}"], q)
            for q in (0.1, 0.5, 0.9) if f"p{int(q * 100)}" in results
        ]))
    return summary


def day_type(index: pd.DatetimeIndex) -> np.ndarray:
    """Label each hour: major holiday, next to one, minor holiday, weekend or weekday.

    Checked in that order, so a holiday on a Saturday counts as a holiday.
    """
    from features import calendar_features  # local import: metrics needs it only here

    cal = calendar_features(index)
    return np.select(
        [
            cal["is_major_holiday"] == 1,
            (cal["is_day_before_major_holiday"] == 1) | (cal["is_day_after_major_holiday"] == 1),
            cal["is_minor_holiday"] == 1,
            cal["is_weekend"] == 1,
        ],
        ["major holiday", "day next to a major holiday", "minor holiday", "weekend"],
        default="ordinary weekday",
    )


def breakdown(results: pd.DataFrame, by: str, temp: pd.Series | None = None) -> pd.DataFrame:
    """MAPE and MAE per group: "season", "peak", "hour", "temp_bin", "month" or "day_type".

    `temp` (observed temperature by UTC hour) is needed for "temp_bin". Bins
    use measured temperature in every weather arm, so the groups are identical
    across arms and comparable.
    """
    work = results.dropna(subset=["actual", "forecast"]).copy()
    local = pd.DatetimeIndex(work["target_time"]).tz_convert(config.LOCAL_TZ)
    if by == "season":
        work["group"] = [SEASONS[m] for m in local.month]
    elif by == "peak":
        work["group"] = np.where(local.hour.isin(PEAK_HOURS), "peak 14:00-19:59", "off-peak")
    elif by == "hour":
        work["group"] = local.hour
    elif by == "temp_bin":
        if temp is None:
            raise ValueError("breakdown by temp_bin needs the observed temperature series")
        t = temp.reindex(pd.DatetimeIndex(work["target_time"])).to_numpy()
        edges = [-np.inf, 0, 10, 20, 25, 30, 35, np.inf]
        labels = ["< 0", "0-10", "10-20", "20-25", "25-30", "30-35", "≥ 35"]
        work["group"] = pd.cut(t, edges, labels=labels, right=False)
    elif by == "month":
        work["group"] = local.strftime("%Y-%m")
    elif by == "day_type":
        work["group"] = day_type(pd.DatetimeIndex(work["target_time"]))
    else:
        raise ValueError(f"unknown breakdown {by!r}")
    grouped = work.groupby("group", observed=True)
    return pd.DataFrame({
        "MAPE_%": grouped.apply(lambda g: mape(g["actual"], g["forecast"]), include_groups=False),
        "MAE_mw": grouped.apply(lambda g: mae(g["actual"], g["forecast"]), include_groups=False),
        "bias_mw": grouped.apply(lambda g: bias(g["actual"], g["forecast"]), include_groups=False),
        "hours": grouped.size(),
    })
