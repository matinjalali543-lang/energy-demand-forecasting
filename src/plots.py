"""Figures for the README and the Upwork portfolio.

Every figure shares one small house style, so the set reads as a series: a
near-white surface, recessive grid, thin marks, colors assigned by role in a
fixed order (actual demand is always slot 1 blue, a forecast always slot 2
orange), and text in ink colors rather than series colors. One y-axis per
panel, always; two measures of different scale get two panels.

    python src/plots.py --probe data/processed/probe_2025-07.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are written to files; no display needed

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import metrics  # noqa: E402
from backtest import NO_WEATHER_ARM  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
# Categorical slots in their validated order. Roles, not ranks: the same
# quantity keeps the same color in every figure.
ACTUAL = "#2a78d6"  # slot 1, blue: measured demand
FORECAST = "#eb6834"  # slot 2, orange: this project's model (LightGBM)
THIRD = "#1baf7a"  # slot 3, aqua: ERCOT's published forecast
FOURTH = "#eda100"  # slot 4, yellow: vanilla regression
FIFTH = "#e87ba4"  # slot 5, magenta: the MW-target ablation
REFERENCE = "#a3a19b"  # recessive grey: seasonal naive, a reference not a contender

# Long model names from backtest.py -> legend label, end-of-line label, fixed color.
MODELS = {
    "LightGBM, P10/P50/P90 + conformal": ("LightGBM (this project)", "LightGBM", FORECAST),
    "ERCOT published day-ahead forecast": ("ERCOT's own forecast", "ERCOT", THIRD),
    "Vanilla regression (GEFCom benchmark)": ("Vanilla regression", "Vanilla", FOURTH),
    "Seasonal naive (same hour last week)": ("Seasonal naive", "Naive", REFERENCE),
}
ABLATION = "LightGBM, MW target (ablation)"
RAW_QUANTILES = "LightGBM, P10/P50/P90"
MAIN = "LightGBM, P10/P50/P90 + conformal"
SOURCE = ("Backtest Oct 2025 – Sep 2026, ERCOT. Forecasts issued 10:00 Central the day before, "
          "with weather forecasts as published.")


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK,
            "lines.linewidth": 1.6,
            "font.size": 10,
        }
    )


def _local(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out.index = pd.DatetimeIndex(out.index).tz_convert(config.LOCAL_TZ)
    return out


def sanity_check(table: pd.DataFrame, path: Path) -> Path:
    """Phase 1 check: does the data look like Texas load?

    Left: demand against population-weighted temperature, which should rise
    steeply with heat. Right: the hottest week, actual demand against ERCOT's
    own published day-ahead forecast, which should track it closely; if the
    two were an hour apart, the timestamp convention would be wrong.
    """
    apply_style()
    local = _local(table)
    gw = local[["demand_mw", "ercot_da_forecast_mw"]] / 1000

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.6]}
    )

    left.scatter(local["temp_c"], gw["demand_mw"], s=9, color=ACTUAL, alpha=0.55, linewidths=0)
    left.set_title("Demand rises with heat")
    left.set_xlabel("Population-weighted temperature (°C)")
    left.set_ylabel("ERCOT demand (GW)")

    # The seven days with the highest mean temperature, starting at midnight.
    daily_temp = local["temp_c"].resample("D").mean()
    rolling = daily_temp.rolling(7).mean().dropna()
    week_end = rolling.idxmax() + pd.Timedelta(days=1)
    week = gw.loc[week_end - pd.Timedelta(days=7) : week_end - pd.Timedelta(hours=1)]

    right.plot(week.index, week["demand_mw"], color=ACTUAL, label="Actual demand")
    right.plot(week.index, week["ercot_da_forecast_mw"], color=FORECAST,
               label="ERCOT day-ahead forecast", linestyle=(0, (4, 2)))
    right.set_title(f"Hottest week, {week.index[0]:%d %b} – {week.index[-1]:%d %b %Y}")
    right.set_ylabel("GW")
    right.legend(loc="upper left", ncols=2)
    right.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d", tz=config.LOCAL_TZ))

    fig.text(0.01, 0.005, "Source: EIA-930 (ERCO), NOAA GHCNh. Times are US Central.",
             color=INK_SECONDARY, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --- Backtest figures ------------------------------------------------------------------


def _honest(predictions: pd.DataFrame) -> pd.DataFrame:
    """Forecast-weather rows, plus the weather-free models (identical in both arms)."""
    return predictions[predictions["arm"].isin(["forecast", NO_WEATHER_ARM])]


def _local(times) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(times).tz_convert(config.LOCAL_TZ)


def _finish(fig, path: Path, note: str = SOURCE) -> Path:
    fig.text(0.01, 0.012, note, color=INK_SECONDARY, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _label_line_ends(ax, items: list[tuple[float, str]], x: float) -> None:
    """Direct labels at the right end of lines, nudged apart so they never overlap."""
    low, high = ax.get_ylim()
    gap = 0.045 * (high - low)
    placed: list[float] = []
    for y, text in sorted(items):
        y = max(y, placed[-1] + gap) if placed else y
        placed.append(y)
        ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_SECONDARY, annotation_clip=False)


def _mape_by(rows: pd.DataFrame, key) -> pd.Series:
    """MAPE per value of `key`, an array aligned with `rows`."""
    work = rows.assign(group=list(key))
    return work.groupby("group").apply(
        lambda g: metrics.mape(g["actual"], g["forecast"]), include_groups=False)


def _record_week(predictions: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Seven local days centred on the highest demand of the backtest, as UTC bounds."""
    main = _honest(predictions)
    main = main[main["model"] == MAIN]
    peak = _local([main.loc[main["actual"].idxmax(), "target_time"]])[0].normalize()
    start, end = peak - pd.Timedelta(days=3), peak + pd.Timedelta(days=4)
    return start.tz_convert("UTC"), end.tz_convert("UTC")


def _week_rows(predictions, model, start, end) -> pd.DataFrame:
    rows = _honest(predictions)
    rows = rows[(rows["model"] == model) & (rows["target_time"] >= start)
                & (rows["target_time"] < end)]
    return rows.sort_values("target_time")


def _day_axis(ax) -> None:
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d %b", tz=config.LOCAL_TZ))
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(tz=config.LOCAL_TZ))


def record_week_band(predictions: pd.DataFrame, path: Path) -> Path:
    """README chart 1: the record-demand week, forecast with its P10-P90 band."""
    apply_style()
    start, end = _record_week(predictions)
    rows = _week_rows(predictions, MAIN, start, end)
    x = _local(rows["target_time"])
    inside = metrics.coverage(rows["actual"], rows["p10"], rows["p90"])

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.fill_between(x, rows["p10"] / 1e3, rows["p90"] / 1e3, color=FORECAST, alpha=0.16,
                    linewidth=0, label="P10–P90 band")
    ax.plot(x, rows["forecast"] / 1e3, color=FORECAST, label="Forecast (P50)")
    ax.plot(x, rows["actual"] / 1e3, color=ACTUAL, label="Actual demand")
    ax.set_title("Texas record-demand week: day-ahead forecast vs actual")
    ax.set_ylabel("ERCOT demand (GW)")
    _day_axis(ax)
    ax.legend(loc="upper left", ncols=3)
    ax.text(0.995, 0.03, f"Actual inside the band in {inside:.0f}% of this week's hours",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=INK_SECONDARY)
    return _finish(fig, path)


def record_week_ablation(predictions: pd.DataFrame, frame: pd.DataFrame, path: Path) -> Path:
    """Why the model predicts relative demand: the MW model hits its training ceiling."""
    apply_style()
    start, end = _record_week(predictions)
    main = _week_rows(predictions, MAIN, start, end)
    mw = _week_rows(predictions, ABLATION, start, end)
    ceiling = frame.loc[frame.index < config.BACKTEST_START, "demand_mw"].max() / 1e3

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axhline(ceiling, color=INK_SECONDARY, linewidth=0.9, linestyle=(0, (2, 2)))
    ax.plot(_local(main["target_time"]), main["actual"] / 1e3, color=ACTUAL,
            label="Actual demand")
    ax.plot(_local(main["target_time"]), main["forecast"] / 1e3, color=FORECAST,
            label="LightGBM, relative target (used)")
    ax.plot(_local(mw["target_time"]), mw["forecast"] / 1e3, color=FIFTH,
            label="LightGBM, MW target (ablation)")
    ax.text(_local(main["target_time"])[2], ceiling + 0.25,
            f"Highest demand in all training data: {ceiling:.1f} GW",
            fontsize=8.5, color=INK_SECONDARY, va="bottom")
    ax.set_title("Why the model predicts demand relative to its recent level")
    ax.set_ylabel("ERCOT demand (GW)")
    low, high = ax.get_ylim()
    ax.set_ylim(low - 0.18 * (high - low), high)
    _day_axis(ax)
    ax.legend(loc="lower left", ncols=3)
    return _finish(fig, path)


def _model_lines(ax, series: dict[str, pd.Series], x_label_at: float, fmt="{:.1f}%") -> None:
    """One line per model in fixed colors, naive dashed underneath, ends labelled."""
    labels = []
    for model, (label, short, color) in reversed(MODELS.items()):  # naive drawn first
        if model not in series:
            continue
        values = series[model]
        dashed = model.startswith("Seasonal")
        ax.plot(values.index, values.to_numpy(), color=color, label=label,
                linestyle=(0, (4, 2)) if dashed else "-", marker=None if dashed else "o",
                markersize=4, zorder=2 if dashed else 3)
        labels.append((float(values.iloc[-1]), f"{short} {fmt.format(values.iloc[-1])}"))
    _label_line_ends(ax, labels, x_label_at)
    handles, names = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], names[::-1], loc="upper left", ncols=4)


def error_by_hour(predictions: pd.DataFrame, path: Path) -> Path:
    """README chart 2: error through the day, every model on the same hours."""
    apply_style()
    rows = _honest(predictions)
    series = {}
    for model in MODELS:
        group = rows[rows["model"] == model]
        series[model] = _mape_by(group, _local(group["target_time"]).hour)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axvspan(13.5, 19.5, color=GRID, alpha=0.5, linewidth=0)
    ax.text(16.5, 0.02, "system peak", transform=ax.get_xaxis_transform(), ha="center",
            va="bottom", fontsize=8.5, color=INK_SECONDARY)
    ax.set_ylim(0, max(s.max() for s in series.values()) * 1.18)
    _model_lines(ax, series, x_label_at=23)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 26.5)
    ax.set_xlabel("Hour of day (Central time)")
    ax.set_ylabel("Mean absolute % error")
    ax.set_title("Forecast error through the day")
    return _finish(fig, path)


def error_by_month(predictions: pd.DataFrame, path: Path) -> Path:
    apply_style()
    rows = _honest(predictions)
    series = {}
    for model in MODELS:
        group = rows[rows["model"] == model]
        by_month = _mape_by(group, _local(group["target_time"]).strftime("%Y-%m"))
        by_month.index = pd.to_datetime(by_month.index + "-15")
        series[model] = by_month

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_ylim(0, max(s.max() for s in series.values()) * 1.18)
    last = max(s.index[-1] for s in series.values())
    _model_lines(ax, series, x_label_at=matplotlib.dates.date2num(last))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    ax.set_xlim(right=matplotlib.dates.date2num(last + pd.Timedelta(days=55)))
    ax.set_ylabel("Mean absolute % error")
    ax.set_title("Forecast error month by month")
    return _finish(fig, path)


def error_by_temperature(predictions: pd.DataFrame, frame: pd.DataFrame, path: Path) -> Path:
    apply_style()
    rows = _honest(predictions)
    models = list(MODELS)
    tables = {m: metrics.breakdown(rows[rows["model"] == m], "temp_bin", temp=frame["temp_c"])
              for m in models}
    bins = tables[MAIN].index.astype(str).tolist()
    hours = tables[MAIN]["hours"]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        label, _, color = MODELS[model]
        values = tables[model]["MAPE_%"].reindex(tables[MAIN].index)
        ax.bar([b + (i - 1.5) * width for b in range(len(bins))], values, width=width,
               color=color, label=label, edgecolor=SURFACE, linewidth=1.5)
    ax.set_xticks(range(len(bins)),
                  [f"{b} °C\n{h:,} h" for b, h in zip(bins, hours, strict=True)])
    ax.set_xlabel("Measured temperature at the target hour (population-weighted)")
    ax.set_ylabel("Mean absolute % error")
    ax.set_title("Forecast error by temperature")
    ax.legend(loc="upper left", ncols=4)
    ax.grid(axis="x", visible=False)
    return _finish(fig, path)


def coverage_by_month(predictions: pd.DataFrame, path: Path) -> Path:
    """How often the P10-P90 band held the actual, before and after calibration."""
    apply_style()
    rows = _honest(predictions)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axhspan(75, 85, color=GRID, alpha=0.6, linewidth=0)
    ax.text(0.995, 86, "target 75–85%", transform=ax.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=8.5, color=INK_SECONDARY)
    for model, label, color, style in (
        (RAW_QUANTILES, "Raw quantile band", REFERENCE, (0, (4, 2))),
        (MAIN, "After conformal calibration", FORECAST, "-"),
    ):
        group = rows[rows["model"] == model]
        months = _local(group["target_time"]).strftime("%Y-%m")
        cov = group.groupby(months).apply(
            lambda g: metrics.coverage(g["actual"], g["p10"], g["p90"]), include_groups=False)
        cov.index = pd.to_datetime(cov.index + "-15")
        ax.plot(cov.index, cov.to_numpy(), color=color, linestyle=style, marker="o",
                markersize=4, label=label)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    ax.set_ylabel("Hours inside P10–P90 (%)")
    ax.set_title("Does the P10–P90 band hold 80% of actual hours?")
    ax.legend(loc="lower right", ncols=2)
    return _finish(fig, path)


def render_all(predictions: pd.DataFrame, frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Every backtest figure. The first two are the README's."""
    return [
        record_week_band(predictions, out_dir / "record_week_forecast.png"),
        error_by_hour(predictions, out_dir / "error_by_hour.png"),
        error_by_month(predictions, out_dir / "error_by_month.png"),
        error_by_temperature(predictions, frame, out_dir / "error_by_temperature.png"),
        coverage_by_month(predictions, out_dir / "coverage_by_month.png"),
        record_week_ablation(predictions, frame, out_dir / "relative_target_ablation.png"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render figures.")
    parser.add_argument("--probe", type=Path, help="processed probe CSV for the sanity check")
    args = parser.parse_args(argv)
    if args.probe:
        table = pd.read_csv(args.probe, index_col=0, parse_dates=True)
        out = sanity_check(table, config.FIGURES_DIR / f"sanity_check_{args.probe.stem[6:]}.png")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
