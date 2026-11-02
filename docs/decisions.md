# Design decisions

Each entry: what was decided, what it was decided against, and the evidence. Decisions are
made on synthetic data or on data *before* the backtest window, never by looking at backtest
scores, so the reported results are not tuned to the test year.

## D1. Issue at 10:00 Central for the next local day; no `lag_24h`

ERCOT's day-ahead market closes at 10:00 the day before delivery, so that is when a
day-ahead forecast is useful. From 10:00, the target hours are 14–37 hours ahead, so
"same hour yesterday" has not happened yet for every target after 10:00. Using it is the
most common leak in published load-forecasting notebooks. `FeatureSpec.check_leads`
refuses any lag that does not exceed the longest lead, and `tests/test_leakage.py`
checks that refusal.

## D2. Weather forecasts as they were published, not as they were later archived

Open-Meteo's `previous_day1` value for hour *t* comes from a model run started 24–47 h
before *t*. For the later target hours, that run can start *after* the 10:00 issue time.
`features.forecast_weather_as_known` uses day1 only up to an 18-hour lead (24 h minus a
6-hour publication latency) and the older day2 run beyond that. This is stricter than
strictly necessary, so it slightly understates accuracy. A test plants the tempting
version, day1 everywhere, and requires the leakage check to catch it.

## D3. LightGBM predicts demand relative to its 4-week mean

Trees cannot predict outside the range of their training targets. ERCOT demand grows
several percent a year and peaked at 91.1 GW in July 2026, about 6 GW above any earlier
year. So the main model predicts demand ÷ (mean demand over the 4 weeks before the
issue time) and multiplies back. Its demand inputs are scaled the same way.

Evidence: LightGBM median model on seeded synthetic data, trained to 2025-05-30 and
tested on the following 90 days.

| Synthetic scenario | MW target | ÷ 1-week mean | ÷ 4-week mean |
|---|---:|---:|---:|
| No load growth | **1.34%** | 1.92% | 1.77% |
| 5% / year growth (test peak above training max) | 3.92% | 1.94% | **1.78%** |

The MW model is best when the future looks like the past and fails badly when it does
not. Of the two relative versions, the 4-week mean wins in both scenarios: a 1-week mean
carries that week's weather into the target. The MW model stays in the backtest as an
ablation, so the real-data cost or benefit of this choice is reported rather than assumed.
`tests/test_quantile_models.py` encodes the growth result.

## D4. Conformal calibration of the P10–P90 band, on recent forecast-weather days

Quantile models trained on measured weather have never seen a weather forecast miss, so
fed real forecasts their bands are too narrow. In the synthetic backtest
(`python src/backtest.py --synthetic`) the raw LightGBM band covered **59.2%** of actuals
with measured weather and **45.2%** with forecast weather, against a target of 80%. After
conformalized quantile regression (Romano et al., 2019), calibrated on the 28 days before
each issue time with features in the arm's own weather mode, coverage was **82.3%** and
**84.3%**, with no loss of point accuracy (P50 MAPE 3.06% in the forecast arm either way).

Both variants are reported. Conformal guarantees assume exchangeable data, which hourly load
is not, so the correction is approximate and the backtest measures how well it works.

## D5. Hyperparameters fixed in advance

`models.LIGHTGBM_PARAMS` were set once, before any backtest, and are not tuned. Tuning on
the backtest year would make its score optimistic. Determinism is on, so reruns are
bit-identical.

## D6. Baselines

- **Seasonal naive** (same hour last week): the MASE denominator, scaled on the history
  before the backtest.
- **Vanilla regression**: the GEFCom2012 benchmark (Hong, Pinson & Fan, 2014), chosen over
  SARIMAX because it is the literature's reference load model and refits in about a second.
  It refuses to forecast a calendar month absent from its training data, instead of silently
  using untrained coefficients.
- **ERCOT's published day-ahead forecast**: external benchmark, never an input.

## D7. Bias-correct the archived weather forecasts against the airport readings

Measured weather comes from NOAA airport stations (Open-Meteo's archive was overloaded for a
day, see `data/README.md`); forecasts come from GFS via Open-Meteo. Over the backtest year
the two disagree systematically:

| Backtest year, vs airport readings | Mean error | MAE raw | MAE corrected |
|---|---:|---:|---:|
| Day1 forecast (lead ≤ 18 h) | −0.70 °C | 1.02 °C | 0.82 °C |
| Day2 forecast (lead > 18 h) | +1.04 °C | 1.48 °C | 1.24 °C |

A time shift was ruled out: both columns fit best at zero lag. The day1/day2 gap is largest
in winter (+2.6 °C in December), which suggests the two lead times come from different
underlying models in Open-Meteo's "GFS seamless" blend; that is a guess, not something the
data can confirm.

Models trained on airport readings would read an uncorrected offset as weather (a degree too
warm in July means too much cooling load). So `features.recent_bias` subtracts each
forecast's mean error over the `WEATHER_BIAS_DAYS` (28) days *before* the issue time. Only
past hours enter, so it is leakage-safe, and it is what an operator's forecast
post-processing would do. With fewer than seven days of paired history the forecast features
are left blank rather than corrected with a guess. Tests check that a steady +2 °C offset is
removed exactly, and that too little history blanks rather than guesses.
