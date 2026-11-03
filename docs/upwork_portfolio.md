# Upwork portfolio item

Ready to paste into Upwork's portfolio form. Field limits are checked: the title is at most 70
characters, the description at most 600, and there are at most 5 skills. Every number is from
`results/summary.csv`.

## Title

Day-ahead electricity demand forecast for Texas: 59% less error

## Your role

Sole developer: data pipeline, forecasting models, backtesting, uncertainty and reporting

## Description

Day-ahead hourly demand forecast for the Texas grid (ERCOT), issued at 10:00 each morning as a
P10/P50/P90 range, using only public data and weather forecasts as they were actually published.
Over a full year it never saw, it cut the error of a "same hour last week" baseline by 59% (2.9%
MAPE), and its calibrated P10–P90 band held 78% of actual hours. ERCOT's own forecast is still
more accurate (2.0%), and the repo says so. Leakage tests prove no result used future data.

## Skills (5)

1. Python
2. Machine Learning
3. Time Series Forecasting
4. LightGBM
5. Data Science

## Images (in this order)

1. `reports/figures/record_week_forecast.png`: the record-demand week, with the forecast,
   P10–P90 band and actual demand. The strongest single image.
2. `reports/figures/error_by_hour.png`: every model's error through the day, against
   seasonal naive and ERCOT.
3. `reports/figures/relative_target_ablation.png`: the design choice that kept the model
   accurate through a record summer.

## Link

https://github.com/matinjalali543-lang/energy-demand-forecasting
