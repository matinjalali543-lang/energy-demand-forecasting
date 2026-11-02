# Project brief: Electricity Demand Forecasting (Upwork portfolio piece)

> Planning brief for Claude Code. Start in plan mode: propose the repo structure and phase plan below, confirm data access, then build phase by phase.

## 1. Purpose

A public, reproducible Python project showing that I can forecast electricity demand properly: honest backtesting, clear baselines, weather features, and results a client can trust.

It goes on my **Energy Data Scientist** Upwork profile as evidence for jobs like "load forecasting", "energy demand prediction" and "forecast + optimization". It should build on my published research in energy demand forecasting. Where my paper used a specific method, dataset or region, adapt that here.

**Audience:** a non-academic client (utility analyst, energy startup, facility manager) skimming the README for 60 seconds, and a technical reviewer reading the notebook.

## 2. Scope

**In scope**
- Hourly electricity demand for one region/balancing authority, 3–5 years of history.
- Day-ahead (24–48 h) forecast as the main task; a short note on week-ahead.
- Weather (temperature at minimum) and calendar features.
- 3 model tiers compared on the same rolling backtest.
- Probabilistic output (quantiles), because clients plan on ranges, not a single line.

**Out of scope (for v1)**
- Deep learning beyond one optional comparison.
- Real-time deployment, APIs, dashboards (maybe later as a small Streamlit demo).

## 3. Data (public, free)

Pick one primary source; confirm access before building.

| Data | Source | Notes |
|---|---|---|
| Hourly demand | **EIA-930** hourly demand by balancing authority (EIA Open Data API, free key) | US regions, e.g. ERCO (Texas) — doubles as useful ERCOT context |
| Hourly demand (alternative) | ERCOT public hourly load archives, or ENTSO-E Transparency (free token) for a European country | Use if EIA access is awkward |
| Weather | **Open-Meteo historical weather API** (free, no key) | Hourly temperature for 2–5 representative cities, population-weighted |
| Holidays | `holidays` Python package | Country/state holidays |

Requirements:
- Cache raw downloads to `data/raw/` (git-ignored), with a script that re-downloads them. Never commit large data.
- Document every source, licence/terms, and download date in `data/README.md`.

## 4. Method

### Baselines (must beat these)
1. **Seasonal naive**: same hour last week.
2. **Classical**: SARIMAX or ETS, or a regression on temperature with hour-of-week dummies.

### Main model
3. **Gradient boosting** (LightGBM or XGBoost) with:
   - calendar: hour, day-of-week, month, holiday flags, day-before/after-holiday
   - weather: temperature, heating/cooling degree hours, lagged temperature, optional humidity
   - lags: demand at t-24, t-48, t-168 (only lags available at forecast time; no leakage)

### Probabilistic
- Quantile regression (LightGBM quantile objective) for P10/P50/P90, or conformal intervals.
- Report interval coverage (does P10–P90 contain about 80% of actuals?).

### Backtesting (the core of credibility)
- **Rolling-origin** evaluation over at least 12 months, retraining weekly or monthly.
- Use **forecasted-like** weather honestly: either note that actual weather was used (an upper bound on accuracy) or simulate forecast error. State the choice clearly.
- Metrics: MAPE, RMSE, MAE, **MASE** (vs seasonal naive), peak-hour error, and pinball loss for quantiles.
- Break results down by season and by peak vs off-peak hours.

## 5. Deliverables

```
energy-demand-forecasting/
├── README.md              # 1-screen summary: problem, data, results table, 2 charts
├── data/README.md         # sources, licences, download dates
├── src/
│   ├── data_download.py   # EIA + Open-Meteo, cached
│   ├── features.py
│   ├── models.py          # baselines, GBM, quantile
│   └── backtest.py        # rolling-origin engine
├── notebooks/
│   └── forecasting_walkthrough.ipynb
├── reports/figures/       # PNGs used in README and Upwork portfolio
├── tests/                 # leakage check, feature shapes, metric functions
├── requirements.txt (or pyproject.toml)
└── LICENSE (MIT)
```

**README must show:**
- One results table: each model × MAPE / MASE / peak error / P10–P90 coverage.
- Chart 1: forecast vs actual for a hot summer week (with P10–P90 band).
- Chart 2: error by hour of day, or by temperature bin.
- "What this means for a client" in 3 bullets (e.g. how much better than naive, where it fails: holidays, heat waves).
- Honest limitations section.

**Upwork portfolio item (separate):**
- Title, e.g. "Day-ahead electricity demand forecast: X% better than seasonal naive"
- 2–3 images from `reports/figures/`, 3-sentence description, link to the GitHub repo.

## 6. Phases for Claude Code

1. **Plan + data access**: confirm the source and region; write `data_download.py`; save one month of data; plot a sanity check.
2. **Full data + features**: 3–5 years; feature pipeline with a unit test proving no future leakage.
3. **Baselines + backtest engine**: seasonal naive and classical; rolling-origin framework; metrics.
4. **GBM + quantiles**: main model; probabilistic intervals; coverage check.
5. **Analysis + figures**: error breakdowns; README charts.
6. **Polish**: README, notebook narrative, tests passing, reproducible from a clean environment.

## 7. Acceptance criteria

- [ ] `python src/data_download.py && python src/backtest.py` reproduces the README numbers from scratch.
- [ ] GBM beats seasonal naive on MASE across the full backtest, or the README honestly says where it doesn't.
- [ ] A test fails if any feature uses information after the forecast issue time.
- [ ] P10–P90 coverage reported; within roughly 75–85%, or the README explains why not.
- [ ] README readable in 60 seconds by a non-technical client.
- [ ] No data or API keys committed.

## 8. Notes for Claude Code

- Ask before downloading more than ~200 MB.
- Keep the code plain and well commented; this is a showcase, so prefer clarity over cleverness.
- Use fixed random seeds; pin package versions.
- Don't overstate results anywhere. If the model only marginally beats a baseline, say so.

---

## Overrides and decisions made during planning

Recorded so the brief above stays as written and every departure from it is visible.

| Brief says | Decision | Why |
|---|---|---|
| Build on my published research | **Not linked in v1** | Owner's call; a "Related research" line can be added later. |
| Lags at t-24, t-48, t-168 | **t-48 and t-168 only**, plus window statistics ending at the issue time | Forecasts are issued at **10:00 Central on day D for D+1**, matching ERCOT day-ahead market timing. At that moment "same hour yesterday" has not happened yet for hours 10:00–23:00, so t-24 would be leakage. The feature guard refuses it, and a test proves it. |
| Classical baseline: SARIMAX, ETS, or temperature regression | **GEFCom "vanilla" regression** (Hong/Tao): trend, month, hour×weekday, hour×temperature polynomial | The well-known load-forecasting benchmark, and fast enough to refit for every rolling origin. SARIMAX across a year of origins is slow and adds little. |
| Use forecasted-like weather honestly | **Two arms**: observed weather (an upper bound) and real archived GFS day-ahead forecasts (Open-Meteo Previous Runs API) | The archive exists for this window, so no forecast error needs to be simulated. |
| One region, 3–5 years | **ERCOT, 2021-07-01 → 2026-09-30**; backtest **2025-10-01 → 2026-09-30**, monthly retraining | Starts after Winter Storm Uri, whose load shedding corrupts recorded demand. |
| Main model: LightGBM or XGBoost | **LightGBM**, point plus quantile objective (P10/P50/P90) | Split-conformal calibration is the fallback if raw quantile coverage falls outside 75–85%. |
| (not in brief) | **ERCOT's own published day-ahead forecast** (EIA-930 `DF`) added as an external benchmark | A client's real question is "can you beat what the grid operator already publishes?" It scored 2.2% MAPE in July 2025, a demanding bar. |
| Layout `src/*.py` | Kept flat, as written | So that `python src/data_download.py && python src/backtest.py` works literally. Pytest uses `pythonpath = ["src"]`. |
| Main model: LightGBM with lags | LightGBM **predicts demand relative to its 4-week mean**; an MW-target LightGBM is kept as an ablation | Trees cannot forecast above their training range, and Texas set a new record (91.1 GW) in the backtest year. Evidence in `docs/decisions.md` D3. |
| Quantile regression *or* conformal intervals | **Both**, reported side by side | Raw quantile bands are too narrow once fed real weather forecasts; conformal calibration on recent forecast-weather days corrects that. `docs/decisions.md` D4. |
| Weather: Open-Meteo historical weather API | **Measured weather from NOAA GHCNh airport stations**; Open-Meteo kept for archived *forecasts* only | Open-Meteo's archive was overloaded for a full day. NOAA readings are real thermometer measurements and public domain. The station-versus-grid offset this creates (−0.7 °C day1, +1.0 °C day2) is removed by a leakage-safe trailing bias correction (`docs/decisions.md` D7). |
