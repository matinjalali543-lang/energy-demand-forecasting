"""Every fixed choice in the project, in one place.

Nothing here is tuned. These are the decisions a reviewer should be able to
find and argue with without reading the code: which region, which window, when
a forecast is issued, which cities stand in for "Texas weather" and how much
each one counts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "reports" / "figures"

PROCESSED_FILE = PROCESSED_DIR / "ercot_hourly.csv"

# --- Region ------------------------------------------------------------------

# EIA-930 respondent code for the ERCOT balancing authority.
RESPONDENT = "ERCO"
# ERCOT runs on Central time. Calendar features use local time, because people
# follow the local clock: a model given UTC hours would have to learn the DST
# shift as noise.
LOCAL_TZ = "America/Chicago"
# Country and subdivision for the `holidays` package.
HOLIDAY_COUNTRY = "US"
HOLIDAY_SUBDIV = "TX"

# --- Window ------------------------------------------------------------------

# Starts after Winter Storm Uri (February 2021). During Uri ERCOT shed up to
# ~20 GW of load, so *recorded* demand is far below what customers wanted; a
# model trained on it would learn that extreme cold lowers demand. Starting in
# July 2021 keeps that out of training and testing alike.
DATA_START = pd.Timestamp("2021-07-01", tz=LOCAL_TZ)
# Exclusive. The last full month before this project was built.
DATA_END = pd.Timestamp("2026-10-01", tz=LOCAL_TZ)

# Rolling-origin backtest: twelve months, every forecast scored out of sample.
BACKTEST_START = pd.Timestamp("2025-10-01", tz=LOCAL_TZ)
BACKTEST_END = DATA_END

# --- Forecast timing -----------------------------------------------------------

# Issue at 10:00 local on day D, forecasting D+1 00:00-23:00. That mirrors the
# ERCOT day-ahead market, whose offers close at 10:00 the day before delivery.
# It also means "same hour yesterday" (t-24h) is *not yet observed* for most
# target hours -- see features.py.
ISSUE_HOUR_LOCAL = 10
HORIZON_HOURS = 24

# How long after a weather model run starts before its forecast can be used.
# NOAA GFS output is complete roughly 4-5 hours after initialisation; 6 is a
# safety margin. features.py uses it to decide which archived run was
# genuinely available at the issue time.
FORECAST_RUN_LATENCY_HOURS = 6

# Days of recent, labelled forecasts used to calibrate the conformal P10-P90
# band. Four weeks: long enough for ~670 scores, short enough to track the season.
CALIBRATION_DAYS = 28

# Days of past forecast-versus-measured temperature used to bias-correct the
# archived weather forecasts before each issue time (features.recent_bias).
WEATHER_BIAS_DAYS = 28

# --- Weather -----------------------------------------------------------------


@dataclass(frozen=True)
class City:
    name: str
    latitude: float
    longitude: float
    # 2020 US Census city population, used only as a weight.
    population: int
    # NOAA GHCNh station ID of the city's main airport weather station.
    station: str


# The five largest cities inside the ERCOT footprint (El Paso is outside it).
# A population-weighted temperature is the standard simple proxy for "the
# weather the load sees": people, not land area, drive demand.
CITIES: tuple[City, ...] = (
    City("Houston", 29.7604, -95.3698, 2_304_580, "USW00012960"),  # Intercontinental
    City("San Antonio", 29.4241, -98.4936, 1_434_625, "USW00012921"),  # International
    City("Dallas", 32.7767, -96.7970, 1_304_379, "USW00013960"),  # Love Field
    City("Austin", 30.2672, -97.7431, 961_855, "USW00013904"),  # Bergstrom
    City("Fort Worth", 32.7555, -97.3308, 918_915, "USW00013961"),  # Meacham
)


def city_weights() -> dict[str, float]:
    """Population shares, summing to 1."""
    total = sum(city.population for city in CITIES)
    return {city.name: city.population / total for city in CITIES}


# --- Reproducibility -----------------------------------------------------------

SEED = 42


def eia_api_key() -> str | None:
    """The EIA key from the environment or a git-ignored `.env`, never from code.

    Returns None when absent; the caller decides whether that is fatal.
    """
    key = os.environ.get("EIA_API_KEY", "").strip()
    if key:
        return key
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "EIA_API_KEY" and value.strip():
                return value.strip()
    return None
