"""Seeded synthetic data with the same columns as the real table.

For tests only. Every reported number comes from real data; on synthetic data
a model's accuracy would measure the noise this generator injected, which says
nothing about Texas.

The shape is deliberately Texas-like -- a strong cooling response, a weaker
heating response, a weekday/weekend gap, a daily cycle -- so that tests can
check that a model *learns something* without being tied to real values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def make_synthetic_frame(
    start: str = "2024-01-01", days: int = 120, seed: int = config.SEED
) -> pd.DataFrame:
    """An hourly frame indexed by hour-beginning UTC, like `ercot_hourly.csv`."""
    rng = np.random.default_rng(seed)
    local_start = pd.Timestamp(start, tz=config.LOCAL_TZ)
    index = pd.date_range(
        local_start.tz_convert("UTC"), periods=24 * days, freq="h", name="timestamp_utc"
    )
    local = index.tz_convert(config.LOCAL_TZ)
    hour = local.hour.to_numpy()
    doy = np.asarray(local.day_of_year)
    n = len(index)

    # Temperature: a seasonal cycle peaking in late July, a daily cycle peaking
    # mid-afternoon, and AR(1) weather noise so that days differ.
    noise = np.zeros(n)
    shocks = rng.normal(0, 0.6, n)
    for i in range(1, n):
        noise[i] = 0.97 * noise[i - 1] + shocks[i]
    temp = (
        20.0
        + 9.0 * np.sin(2 * np.pi * (doy - 115) / 365)
        + 5.0 * np.sin(2 * np.pi * (hour - 9) / 24)
        + noise
    )

    cooling = np.clip(temp - 18.3, 0, None)
    heating = np.clip(18.3 - temp, 0, None)
    weekend = np.asarray(local.day_of_week >= 5)
    demand = (
        42_000
        + 1_500 * cooling
        + 800 * heating
        + 6_000 * np.sin(2 * np.pi * (hour - 11) / 24)
        - 3_500 * weekend
        + rng.normal(0, 800, n)
    )

    return pd.DataFrame(
        {
            "demand_mw": demand,
            "ercot_da_forecast_mw": demand + rng.normal(0, 1_200, n),
            "temp_c": temp,
            "rh_pct": np.clip(60 - 1.5 * (temp - 20) + rng.normal(0, 5, n), 5, 100),
            # Forecasts are the truth plus error that grows with lead time.
            "temp_fc1_c": temp + rng.normal(0, 1.0, n),
            "temp_fc2_c": temp + rng.normal(0, 1.6, n),
        },
        index=index,
    )
