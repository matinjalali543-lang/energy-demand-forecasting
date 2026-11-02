# Upwork portfolio item

Ready to paste. Every number is from `results/summary.csv`.

## Title

Day-ahead electricity demand forecast for Texas: 59% lower error than seasonal naive

## Description (3 sentences)

I built a day-ahead hourly demand forecast for the Texas grid (ERCOT) that issues a P10/P50/P90
range every morning at 10:00, using only public data and weather forecasts as they were
actually published. Over a full year it never saw, it cut the error of a "same hour last week"
baseline by 59% (2.9% MAPE), and its calibrated P10–P90 band held 78% of actual hours. The
repository is equally clear about the limits (ERCOT's own forecast is still more accurate, at
2.0%) and includes leakage tests proving that no result used information from the future.

## Images (in this order)

1. `reports/figures/record_week_forecast.png`: the record-demand week, with the forecast,
   P10–P90 band and actual demand. The strongest single image.
2. `reports/figures/error_by_hour.png`: every model's error through the day, against
   seasonal naive and ERCOT.
3. `reports/figures/relative_target_ablation.png`: the design choice that kept the model
   accurate through a record summer.

## Link

https://github.com/matinjalali543-lang/energy-demand-forecasting

## Skills to tag

Time series forecasting · Load forecasting · LightGBM · Quantile regression ·
Conformal prediction · Python · pandas · Energy analytics · Backtesting
