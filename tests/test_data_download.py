"""The data conventions that would corrupt every result if they were wrong.

No network: the EIA reader is fed a hand-written cache file, and the weighting
is checked on toy frames.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import config
import data_download as dd


def test_city_weights_sum_to_one_and_favour_houston():
    weights = config.city_weights()
    assert sum(weights.values()) == pytest.approx(1.0)
    assert max(weights, key=weights.get) == "Houston"


def test_eia_hour_ending_becomes_hour_beginning(tmp_path, monkeypatch):
    """EIA's "T15" is the hour 14:00-15:00 UTC, so it must land on the 14:00 row."""
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    month = pd.Timestamp("2025-07-01")
    rows = [
        {"period": "2025-07-01T15", "type": "D", "value": "60000"},
        {"period": "2025-07-01T15", "type": "DF", "value": "61000"},
    ]
    cache = tmp_path / "eia" / f"{config.RESPONDENT}_{month:%Y-%m}.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps(rows), encoding="utf-8")

    start = pd.Timestamp("2025-07-01 00:00", tz="UTC")
    frame = dd.eia_frame(start, start + pd.Timedelta(days=1), api_key="unused")

    stamp = pd.Timestamp("2025-07-01 14:00", tz="UTC")
    assert list(frame.index) == [stamp]
    assert frame.loc[stamp, "demand_mw"] == 60000
    assert frame.loc[stamp, "ercot_da_forecast_mw"] == 61000


def test_population_weighted_blanks_hours_with_a_missing_city():
    """A partial average would shift the series silently; a gap is honest."""
    index = pd.date_range("2025-07-01", periods=2, freq="h", tz="UTC")
    frames = {
        city.name: pd.DataFrame({"temperature_2m": [30.0, 30.0]}, index=index)
        for city in config.CITIES
    }
    frames["Dallas"].iloc[1, 0] = np.nan

    weighted = dd.population_weighted(frames, "temperature_2m")

    assert weighted.iloc[0] == pytest.approx(30.0)
    assert np.isnan(weighted.iloc[1])


def test_utc_months_cover_a_local_month_that_straddles_utc():
    """July in Central time runs into 1 August UTC, so both months are fetched."""
    start = pd.Timestamp("2025-07-01", tz=config.LOCAL_TZ)
    months = dd._utc_months(start, start + pd.offsets.MonthBegin(1))
    assert [f"{m:%Y-%m}" for m in months] == ["2025-07", "2025-08"]


def _readings(rows):
    """NOAA-shaped readings: (UTC time, temperature, quality code)."""
    return pd.DataFrame(
        {
            "DATE": [pd.Timestamp(t) for t, _, _ in rows],
            "temperature": [v for _, v, _ in rows],
            "temperature_Quality_Code": [q for _, _, q in rows],
            "relative_humidity": ["50"] * len(rows),  # text, as in NOAA's Parquet files
            "relative_humidity_Quality_Code": [None] * len(rows),
        }
    )


def test_noaa_readings_snap_to_the_nearest_hour_and_average():
    """The routine report at :53 belongs to the next hour, alongside the :00 synoptic."""
    hourly = dd.station_hourly(_readings([
        ("2025-07-01 13:53", 30.0, None),
        ("2025-07-01 14:00", 31.0, "1"),
        ("2025-07-01 14:53", 33.0, 1.0),  # float-coded, as Parquet stores it
    ]))
    assert hourly.loc[pd.Timestamp("2025-07-01 14:00", tz="UTC"), "temp_c"] == pytest.approx(30.5)
    assert hourly.loc[pd.Timestamp("2025-07-01 15:00", tz="UTC"), "temp_c"] == pytest.approx(33.0)


def test_noaa_readings_flagged_by_quality_control_are_dropped():
    hourly = dd.station_hourly(_readings([
        ("2025-07-01 13:00", 30.0, None),
        ("2025-07-01 14:00", 99.0, "L"),  # failed a NOAA check
        ("2025-07-01 14:10", 31.0, None),
        ("2025-07-01 15:00", 32.0, None),
    ]))
    assert hourly.loc[pd.Timestamp("2025-07-01 14:00", tz="UTC"), "temp_c"] == pytest.approx(31.0)


def test_short_gaps_are_interpolated_and_long_ones_left_missing():
    hourly = dd.station_hourly(_readings(
        [("2025-07-01 00:00", 20.0, None), ("2025-07-01 03:00", 23.0, None),  # 2-hour gap
         ("2025-07-01 09:00", 29.0, None)]  # 5-hour gap
    ))
    assert hourly.loc[pd.Timestamp("2025-07-01 01:00", tz="UTC"), "temp_c"] == pytest.approx(21.0)
    assert hourly.loc["2025-07-01 04:00":"2025-07-01 08:00", "temp_c"].isna().all()


def test_a_truncated_parquet_download_is_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        dd._check_parquet(b"PAR1 ...cut off")
    assert dd._check_parquet(b"PAR1...PAR1") == b"PAR1...PAR1"


def test_humidity_stored_as_text_is_read_as_numbers():
    hourly = dd.station_hourly(_readings([("2025-07-01 14:00", 30.0, None)]))
    assert hourly["rh_pct"].iloc[0] == pytest.approx(50.0)
