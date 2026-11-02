# Data

Nothing in this folder except this file is committed. Rebuild everything with:

```bash
python src/data_download.py
```

Raw responses are cached in `data/raw/` (resumable; never re-fetched once present) and
assembled into `data/processed/ercot_hourly.csv`. The full window is a few MB.

## Sources

| Data | Source | Access | Terms |
|---|---|---|---|
| Hourly demand (`D`) and ERCOT's day-ahead demand forecast (`DF`), respondent `ERCO` | [EIA-930 Hourly Electric Grid Monitor](https://www.eia.gov/electricity/gridmonitor/), via the [EIA Open Data API v2](https://www.eia.gov/opendata/) route `electricity/rto/region-data` | Free API key (`EIA_API_KEY` in `.env`) | US government data, public domain; EIA asks for attribution |
| Measured hourly temperature and relative humidity at five airports | [NOAA GHCNh](https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly) (Global Historical Climatology Network, hourly), yearly Parquet files per station | No key | US government data, public domain |
| Day-ahead temperature *forecasts* as issued | [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api), `temperature_2m_previous_day1` and `_day2`, model pinned to `gfs_seamless` (NOAA GFS) | No key | CC BY 4.0, attribution to Open-Meteo and NOAA |
| Public holidays | [`holidays`](https://pypi.org/project/holidays/) package, `US` / subdivision `TX` | pip | MIT |

**Download dates:** EIA-930 and NOAA GHCNh, full window, 2026-11-02. Open-Meteo forecasts:
pending, because the service returned "service overloaded" for all of 2026-11-02.

**Why NOAA and not Open-Meteo for measured weather.** The first version used Open-Meteo's
ERA5 archive for both measured weather and forecasts. Its servers were overloaded for a whole
day, so measured weather moved to NOAA's airport stations. These are actual thermometer
readings rather than a reanalysis grid, and public domain. NOAA's older ISD archive has no
2026 files (NOAA replaced it with GHCNh in 2025), so GHCNh is used.

## What the demand data looks like (2021-07-01 → 2026-09-30)

- 46,032 hours. **One** missing demand hour (2021-11-07 05:00 UTC), no non-positive values,
  and no hour-to-hour jump above 8 GW.
- ERCOT's day-ahead forecast is missing for 120 hours, all in the backtest year
  (Dec 2025, Aug–Oct 2026). Those hours are left out when that benchmark is scored.
- Annual peaks: 73.5 GW (2021, from July), 79.8 (2022), 85.4 (2023), 85.5 (2024), 83.6 (2025),
  **91.1 GW (2026-07-22 17:00 CDT)**. The 2026 peak is genuine: demand ramps smoothly into it,
  ERCOT's own forecast was 88–89 GW, and 188 hours across late July and early August 2026
  exceeded 86 GW. Summer 2026 is therefore **outside the range of all training data**, which
  matters for tree models (they cannot predict above what they have seen).

## Processed table: `ercot_hourly.csv`

One row per hour, indexed by **hour-beginning UTC** (`timestamp_utc`).

| Column | Unit | Meaning |
|---|---|---|
| `demand_mw` | MW | ERCOT system demand, the hour's average (EIA reports MWh per hour) |
| `ercot_da_forecast_mw` | MW | ERCOT's own day-ahead forecast for that hour. **An external benchmark, never a model input.** |
| `temp_c` | °C | Measured airport temperature, population-weighted over five cities |
| `rh_pct` | % | Measured relative humidity, same weighting |
| `temp_fc1_c` | °C | Temperature as forecast by a GFS run started 24–47 h before the hour, same weighting |
| `temp_fc2_c` | °C | The same from a run started 48–71 h before the hour. Both forecast columns are filled from 58 days before the backtest start: a 28-day conformal calibration window, a 28-day bias-correction window before it, and margin |

Which forecast column a feature may use depends on the lead time: see
`forecast_weather_as_known` in `src/features.py`. In short, for later target hours a
day1 run may not have been *published* by the 10:00 issue time, so day2 is used there.

## Choices worth knowing about

- **Hour-ending → hour-beginning.** EIA-930 labels each hour by when it ends; weather is
  labelled by the moment it was measured. EIA timestamps are shifted back one hour so demand
  for 14:00–15:00 shares a row with the 14:00 temperature.
- **Stations:** Houston Intercontinental (`USW00012960`), San Antonio International
  (`USW00012921`), Dallas Love Field (`USW00013960`), Austin–Bergstrom (`USW00013904`) and
  Fort Worth Meacham (`USW00013961`).
- **From readings to hours.** Airports report at about :53 past each hour, plus synoptic and
  special reports in between. Each reading is snapped to the nearest whole hour (UTC) and an
  hour's valid readings are averaged. Readings flagged by NOAA quality control, or outside
  −30…50 °C, are dropped. Gaps of up to 3 hours are interpolated; longer gaps stay missing.
- **Stations versus forecast grid.** The models train on airport readings, but the forecast
  arm feeds them GFS grid-cell forecasts. Over the backtest year the day1 forecast ran
  **0.70 °C cooler** than the airports and the day2 forecast **1.04 °C warmer** (day2 minus
  day1: +0.9 °C in spring, up to +2.6 °C in December). There was no time shift: both line up
  best at zero lag. Each forecast is therefore bias-corrected by its own mean error over the
  28 days before each issue time, which cut its MAE against the airports from 1.02 to
  0.82 °C (day1) and from 1.48 to 1.24 °C (day2). See `docs/decisions.md` D7.
- **Five cities, weighted by 2020 Census city population:** Houston 33%, San Antonio 21%,
  Dallas 19%, Austin 14%, Fort Worth 13%. El Paso is outside the ERCOT grid and excluded.
  An hour with any city missing is left blank rather than averaged over fewer cities.
- **The window starts 2021-07-01, after Winter Storm Uri.** In February 2021 ERCOT shed up to
  ~20 GW of load, so recorded demand understates true demand exactly when it was coldest.
- **Non-positive demand values are blanked, not imputed.** The backtest skips what it cannot
  score and reports how many hours that was.
- **Forecast weather has gaps.** In a spot check, GFS previous-day values were missing for
  all of 2024-01-15 but present across the backtest year. Coverage over the full backtest is
  reported once the full download runs.
