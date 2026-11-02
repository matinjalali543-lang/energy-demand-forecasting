# What the backtest shows

Twelve months of day-ahead forecasts for ERCOT (October 2025 → September 2026), issued at
10:00 Central the day before, retrained monthly, scored hour by hour. Every number below
comes from `results/*.csv` or `data/processed/predictions.csv`, produced by
`python src/backtest.py`. The figures are in `reports/figures/`.

Unless stated otherwise, numbers are for the **honest arm**: weather inputs are the archived
GFS forecasts that had been published by the issue time, bias-corrected using only earlier
data.

**The deliverable is "LightGBM + conformal band"**: its P50 is the point forecast and its
calibrated P10–P90 is the range a client would plan on. The plain "LightGBM, P50" row is the
same model trained on 28 more days, without the band calibration. It is shown because the
ablation and the weather comparison are cleanest between models trained identically.

## Headline

| Model | MAPE | MASE | MAE vs naive | Daily-peak error | P10–P90 coverage |
|---|---:|---:|---:|---:|---:|
| Seasonal naive (same hour last week) | 7.42% | 0.923 | — | 8.55% | — |
| Vanilla regression (GEFCom benchmark) | 2.97% | 0.391 | −57.6% | 3.68% | — |
| **LightGBM, P50** | **2.89%** | **0.376** | **−59.3%** | **3.27%** | 43.9% raw |
| LightGBM + conformal band | 2.94% | 0.382 | −58.6% | 3.27% | **78.4%** |
| LightGBM, MW target (ablation) | 3.70% | 0.481 | −47.9% | 4.17% | — |
| ERCOT's own published forecast | 2.00% | 0.250 | −73.0% | 2.02% | — |

"MAE vs naive" is the change in mean absolute error against seasonal naive, on exactly the
hours both were scored on. MASE divides by the naive rule's error on the training history,
so it is comparable across models but not across datasets.

## Findings

**1. The model cuts the naive error by 59%.** Typical hourly error falls from 7.4% to 2.9%
(2.94% MAPE for the deliverable, 58.6% lower MAE than naive on the same hours).
That is the gap between planning on last week and planning on a weather-driven forecast.

**2. It beats the classical benchmark, but only narrowly.** 2.94% (2.89% without the band
calibration) against the vanilla regression's 2.97% is a small gain. LightGBM wins clearly in summer and in
the hottest hours, and ties or loses in several mild months. A well-specified regression on
temperature and calendar captures most of what is predictable here.

**3. It does not beat ERCOT.** ERCOT's published forecast scores 2.00%, better than the deliverable
in every month except November 2025 (2.34% against 2.52%). Even with *perfect* weather, LightGBM
scores 2.29%, so the gap is not only weather data: the grid operator has more inputs (load
zones, large-customer schedules, commercial weather) and a team. The fair claim is
"within about a percentage point of the grid operator, from public data only".

**4. Weather forecast error costs about 0.6 points.** The same LightGBM scores 2.29% with
measured weather and 2.89% with forecasts as published. The measured-weather number is an
upper bound and is never quoted as achievable accuracy.

**5. Predicting relative demand paid off in the record summer.** Summer 2026 set a new ERCOT
record (91.1 GW on 22 July), and 174 hours exceeded 86 GW, above anything in the training
data (85.5 GW). In those hours LightGBM scored 1.58% MAPE; the MW-target version scored 3.27%
and ran 1.9 GW low on average through July and August, because trees cannot predict above
their training range (`relative_target_ablation.png`). Over the whole year the relative
target cut MAPE from 3.70% to 2.89%.

**6. The calibrated band works on average, not every month.** Raw quantile bands held only
43.9% of actual hours: the models learned from measured weather and had never seen a weather
forecast miss. Conformal calibration on the previous 28 days lifted coverage to **78.4%**,
inside the 75–85% target, at no cost to point accuracy. Month by month it ranged from 51.8%
to 93.6%, and only 4 of 12 months landed inside 75–85% (`coverage_by_month.png`).

## Where it fails

- **Cold snaps.** The 80 hours below 0 °C have 6.25% MAPE (ERCOT: 3.83%). Freezes are rare in
  the training years by design (Winter Storm Uri is excluded), and heating load in Texas is
  driven by a few extreme days.
- **The record day itself.** On 22 July 2026 demand peaked at 91.1 GW. The forecast peaked at
  86.1 GW, and the actual went above the P90 (89.7 GW at that hour). Relative targeting lets
  the model exceed its training range, but not by enough on the most extreme day.
- **Turning points in the season.** In September 2026, demand fell after the record August.
  The model ran 1.7 GW high on average that month (3.4% MAPE), and its band, calibrated on
  August's errors, held only 51.8% of hours. The 4-week demand level that makes the model
  robust to growth also makes it slow to follow a seasonal drop. October 2025 shows the same
  pattern.
- **The afternoon peak.** Errors are highest from 14:00 to 19:59 (3.43% against 2.77%
  off-peak), the hours that set prices and capacity needs. ERCOT's error is nearly flat
  through the day (2.03% against 1.99%).

**Not a weak spot: holidays.** On major holidays LightGBM scored 2.74%, better than ordinary
weekdays (3.02%). The separate major and minor holiday flags appear to work. That is only
240 hours (ten days), so it is not strong evidence either way.

## Week-ahead: a note, not a result

The day-ahead pipeline does not stretch to seven days: from an issue time six or seven days
ahead, even "same hour last week" has not happened yet. What this data does support:

- **Naive degrades slowly.** "Same hour two weeks earlier" scores 7.93% MAPE, against 7.42% for
  one week.
- **The regression is limited only by the weather forecast.** The vanilla regression uses no
  recent demand, so with perfect weather it scores 2.42% at any horizon. Its real week-ahead
  accuracy is set by how good 6–7-day temperature forecasts are, and forecast error grows
  quickly with lead: 1.02 °C at day 1, 1.48 °C at day 2 (raw, against the airports).

A credible week-ahead number needs archived 6–7-day forecasts (Open-Meteo's previous-runs API
offers longer leads). That is future work rather than a guess presented as a result.
