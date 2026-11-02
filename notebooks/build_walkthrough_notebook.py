"""Generate and execute notebooks/forecasting_walkthrough.ipynb.

The notebook holds no logic of its own: it narrates over `src/`, so it cannot
drift from what the pipeline does. It needs the downloaded data and a finished
backtest (`python src/data_download.py && python src/backtest.py`), and the
`notebooks` extra:

    pip install -e ".[notebooks]"
    python notebooks/build_walkthrough_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).parent
TARGET = HERE / "forecasting_walkthrough.ipynb"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        "# Day-ahead demand forecasting for ERCOT: a walkthrough\n"
        "\n"
        "Every morning at 10:00 Central, forecast each hour of tomorrow's Texas demand as a\n"
        "P10/P50/P90 range. This notebook walks through the parts that make the result\n"
        "trustworthy, then the results themselves:\n"
        "\n"
        "1. the data, and what it looks like;\n"
        "2. the leakage barrier, checked on real data;\n"
        "3. weather *as it was forecast*, not as it happened;\n"
        "4. one forecast, end to end, for the record-demand day;\n"
        "5. the twelve-month backtest, and where the model fails.\n"
        "\n"
        "Every number comes from the code in `src/`; nothing here is computed differently.",
    ),
    (
        "code",
        "import sys\n"
        "\n"
        "sys.path.insert(0, '../src')\n"
        "\n"
        "import pandas as pd\n"
        "from IPython.display import Image, display\n"
        "\n"
        "import backtest\n"
        "import config\n"
        "import features as F\n"
        "import metrics as M\n"
        "import models\n"
        "\n"
        "pd.set_option('display.width', 140)\n"
        "frame = pd.read_csv(config.PROCESSED_FILE, index_col=0, parse_dates=True)\n"
        "first, last = frame.index[0], frame.index[-1]\n"
        "print(f'{len(frame):,} hours, {first:%Y-%m-%d} to {last:%Y-%m-%d} UTC')\n"
        "frame.describe().round(1)",
    ),
    (
        "markdown",
        "## 1. The data\n"
        "\n"
        "Demand comes from EIA-930, measured weather from five airport stations (NOAA GHCNh)\n"
        "weighted by city population, and archived GFS weather forecasts from Open-Meteo. The\n"
        "forecast columns are filled only for the backtest year and its lead-in, because\n"
        "that is the only place they are used. Demand climbs steeply with heat, which is\n"
        "most of what there is to predict:",
    ),
    (
        "code",
        "annual_peak = frame['demand_mw'].groupby(frame.index.year).max() / 1e3\n"
        "print('Annual peak demand (GW):', annual_peak.round(1).to_dict())\n"
        "display(Image(filename='../reports/figures/sanity_check_2025-07.png'))",
    ),
    (
        "markdown",
        "The 2026 peak, 91.1 GW, is about 6 GW above anything before it. A model has to be able\n"
        "to forecast demand it has never seen, which shapes the design in section 4.\n"
        "\n"
        "## 2. The leakage barrier\n"
        "\n"
        "Every feature for a forecast issued at 10:00 on day D is built from data strictly\n"
        "before that moment. The table says where each feature's information comes from:",
    ),
    (
        "code",
        "F.describe_features(F.FeatureSpec(weather='forecast'))",
    ),
    (
        "markdown",
        "The check that matters: replace everything that was not knowable at the issue time\n"
        "with noise, and require the features to come out identical. Here it is on the real\n"
        "data, for the record-demand day, in both weather modes:",
    ),
    (
        "code",
        "issue = F.issue_time_for('2026-07-21')  # forecasts 22 July 2026, the record day\n"
        "for mode in ('observed', 'forecast'):\n"
        "    F.assert_no_future_reference(frame, issue, F.FeatureSpec(weather=mode))\n"
        "    print(f'{mode:>8} weather: features unchanged when the future is noise')\n"
        "\n"
        "try:\n"
        "    F.build_features(frame, issue, F.FeatureSpec(lags_hours=(24, 168)))\n"
        "except ValueError as error:\n"
        "    print('\\nRefused, as it should be:', str(error).split('. ')[0])",
    ),
    (
        "markdown",
        "\"Same hour yesterday\" is the most common feature in load forecasting, and at a 10:00\n"
        "issue time it reads the future for every target hour after 10:00. The guard refuses it.\n"
        "\n"
        "## 3. Weather as it was forecast\n"
        "\n"
        "The honest arm uses GFS forecasts archived as they were published, never the weather\n"
        "that later happened. Forecast and airport readings disagree systematically, so each\n"
        "forecast is corrected by its own mean error over the 28 days before each issue time:",
    ),
    (
        "code",
        "window = frame.index >= config.BACKTEST_START\n"
        "rows = []\n"
        "for column, lead in (('temp_fc1_c', 'day 1 (lead up to 18 h)'),\n"
        "                     ('temp_fc2_c', 'day 2 (lead over 18 h)')):\n"
        "    error = (frame[column] - frame['temp_c'])[window]\n"
        "    rows.append({'forecast': lead, 'mean error (°C)': error.mean(),\n"
        "                 'MAE (°C)': error.abs().mean()})\n"
        "pd.DataFrame(rows).set_index('forecast').round(2)",
    ),
    (
        "markdown",
        "## 4. One forecast, end to end\n"
        "\n"
        "Train the deliverable model on everything known at 10:00 on 21 July 2026 and forecast\n"
        "22 July, the day Texas demand hit its record. LightGBM predicts demand *relative to its\n"
        "4-week level*, so it can go above anything in its training data. The conformal step\n"
        "calibrates its P10-P90 band on the previous 28 days of forecast-weather errors.",
    ),
    (
        "code",
        "train_spec = F.FeatureSpec(weather='observed')\n"
        "test_spec = F.FeatureSpec(weather='forecast')\n"
        "history = F.daily_issue_times(frame, train_spec)\n"
        "# Issued two or more days earlier: their whole target day was observed by 10:00 today.\n"
        "X, y = F.build_training_set(frame, history[history <= issue - pd.Timedelta(days=2)],\n"
        "                            train_spec)\n"
        "recent = F.daily_issue_times(frame, test_spec,\n"
        "                             start=issue - pd.Timedelta(days=config.CALIBRATION_DAYS),\n"
        "                             end=issue - pd.Timedelta(hours=14))\n"
        "calibration = F.build_training_set(frame, recent, test_spec)\n"
        "\n"
        "model = models.LightGBMConformal().fit(X, y, calibration=calibration)\n"
        "forecast = model.predict(F.build_features(frame, issue, test_spec))\n"
        "forecast['actual'] = frame['demand_mw'].reindex(forecast.index)\n"
        "peak = forecast['actual'].idxmax()\n"
        "print(f'Trained on {len(X):,} hours; calibrated on {len(calibration[0]):,}.')\n"
        "print(f\"Peak hour {peak.tz_convert(config.LOCAL_TZ):%H:%M} CDT: actual \"\n"
        "      f\"{forecast.loc[peak, 'actual'] / 1e3:.1f} GW, forecast \"\n"
        "      f\"{forecast.loc[peak, 'forecast'] / 1e3:.1f} GW, \"\n"
        "      f\"P10-P90 {forecast.loc[peak, 'p10'] / 1e3:.1f}\"\n"
        "      f\"-{forecast.loc[peak, 'p90'] / 1e3:.1f} GW\")\n"
        "print(f\"MAPE over the day: {M.mape(forecast['actual'], forecast['forecast']):.2f}%\")",
    ),
    (
        "markdown",
        "On the hottest day on record the forecast fell short at the peak and the actual broke\n"
        "above the band. It is a hard day, and the backtest below includes it.\n"
        "\n"
        "## 5. Twelve months of forecasts\n"
        "\n"
        "`python src/backtest.py` repeats section 4 for every day from October 2025 to\n"
        "September 2026, retraining monthly, for every model. These are its results (the honest\n"
        "arm, with weather forecasts as published):",
    ),
    (
        "code",
        "tables = backtest.tables_from(backtest.load_predictions(), frame, config.BACKTEST_START)\n"
        "summary = tables['summary']\n"
        "honest = summary[summary['weather'].isin(['forecast', backtest.NO_WEATHER_ARM])]\n"
        "honest.set_index('model')[['MAPE_%', 'MASE', 'skill_vs_naive_%', 'peak_mape_%',\n"
        "                           'coverage_p10_p90_%', 'hours_scored']].round(2)",
    ),
    (
        "code",
        "for name in ('record_week_forecast', 'error_by_hour', 'error_by_month',\n"
        "             'coverage_by_month', 'relative_target_ablation'):\n"
        "    display(Image(filename=f'../reports/figures/{name}.png'))",
    ),
    (
        "markdown",
        "## Where it fails\n"
        "\n"
        "Error by kind of day and by temperature, for the deliverable model and ERCOT. Holidays\n"
        "are not the weak spot; freezing hours are.",
    ),
    (
        "code",
        "keep = ['LightGBM, P10/P50/P90 + conformal', 'ERCOT published day-ahead forecast',\n"
        "        'Vanilla regression (GEFCom benchmark)']\n"
        "for by in ('day_type', 'temp_bin'):\n"
        "    table = tables[f'breakdown_{by}']\n"
        "    table = table[table['model'].isin(keep)\n"
        "                  & table['weather'].isin(['forecast', backtest.NO_WEATHER_ARM])]\n"
        "    display(table.pivot(index=by, columns='model', values='MAPE_%').round(2))",
    ),
    (
        "markdown",
        "## What this adds up to\n"
        "\n"
        "- The model cuts the seasonal-naive error by 59%, to 2.9% MAPE.\n"
        "- It beats the classical regression benchmark only narrowly.\n"
        "- It does not beat ERCOT's own forecast (2.0%).\n"
        "- Predicting relative demand is what kept it accurate through a record summer.\n"
        "- The calibrated band holds 78% of hours over the year, with real month-to-month swings.\n"
        "\n"
        "Limitations and the full analysis: `README.md` and `docs/analysis.md`.",
    ),
]


def build() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(source) if kind == "markdown" else nbf.v4.new_code_cell(source)
        for kind, source in CELLS
    ]
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    return notebook


def main() -> None:
    notebook = build()
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(
        notebook, {"metadata": {"path": str(HERE)}}
    )
    nbf.write(notebook, TARGET)
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
