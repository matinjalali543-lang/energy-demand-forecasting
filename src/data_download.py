"""Download and cache every input, then build one hourly table.

    python src/data_download.py                 # the full window in config.py
    python src/data_download.py --month 2025-07 # one month, for a quick check

Three sources, all free:

* **EIA-930** (EIA Open Data API v2): hourly ERCOT demand (`D`) and ERCOT's own
  published day-ahead demand forecast (`DF`). Needs a free key in `.env`; falls
  back to the public, heavily rate-limited `DEMO_KEY` with a warning.
* **NOAA GHCNh** (Global Historical Climatology Network, hourly): *measured*
  temperature and humidity at the main airport of five Texas cities. Used for
  training, and for the "observed weather" backtest arm, which is an upper
  bound on accuracy. Public domain, no key.
* **Open-Meteo previous-runs API**: what the weather model *forecast* one and
  two days before each hour. Used only in the backtest window, for the honest
  arm.

Raw responses are cached under `data/raw/` and never re-fetched once present,
so a rerun after a dropped connection resumes where it stopped. Nothing under
`data/` is committed.

Timestamps: EIA-930 labels each hour by when it *ends* (UTC). Everything here is
shifted to hour-*beginning* UTC so that the demand for 14:00-15:00 sits on the
same row as the 14:00 temperature.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import config

EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
EIA_PAGE = 5000  # the API's maximum rows per request
EIA_TYPES = {"D": "demand_mw", "DF": "ercot_da_forecast_mw"}

NOAA_URL = (
    "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/"
    "by-year/{year}/parquet/GHCNh_{station}_{year}.parquet"
)
NOAA_COLUMNS = ["DATE", "temperature", "temperature_Quality_Code",
                "relative_humidity", "relative_humidity_Quality_Code"]
# A blank quality code means the reading passed NOAA's checks; the legacy ISD
# codes 0, 1, 4, 5 and 9 mean the same. Anything else is flagged as suspect.
NOAA_GOOD_QUALITY = {"", "0", "1", "4", "5", "9"}
# Hours with no valid reading are filled by linear interpolation only when the
# gap is this short; a longer gap stays missing.
MAX_INTERPOLATED_HOURS = 3

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_MODEL = "gfs_seamless"
# The value for hour t as forecast by a run started 24-47 h (day1) or 48-71 h
# (day2) before t. Both are needed: for the later target hours, a day1 run may
# have started *after* our 10:00 issue time, so features.py switches to day2
# there. See `features.forecast_weather_as_known`.
PREVIOUS_RUNS_VARIABLES = {
    "temperature_2m_previous_day1": "temp_fc1_c",
    "temperature_2m_previous_day2": "temp_fc2_c",
}

USER_AGENT = "energy-demand-forecasting/0.1 (portfolio project)"


# --- HTTP ----------------------------------------------------------------------


def fetch(url: str, params: dict | list | None = None, *, parse=json.loads,
          retries: int = 6):
    """GET `url`, backing off on rate limits, overload and dropped connections.

    Open-Meteo answers 503 "service overloaded" at busy times and EIA answers
    429 when the key's hourly quota is spent; both are worth waiting out. A
    400 means the request itself is wrong and is raised immediately. `parse`
    turns the body into the result (JSON by default); if it raises ValueError
    the body was truncated or empty, and the request is retried.
    """
    query = f"?{urllib.parse.urlencode(params, doseq=True)}" if params else ""
    request = urllib.request.Request(f"{url}{query}", headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return parse(response.read())
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")[:300]
            if error.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = 10 * 2**attempt
                print(f"  HTTP {error.code} ({body.strip()}); retrying in {wait}s")
                time.sleep(wait)
                continue
            # Never echo the URL: for EIA it contains the API key.
            raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from None
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError,
                TimeoutError, ValueError) as error:
            # All seen from Open-Meteo under load: dropped connections, and
            # "200 OK" with an empty body. Transient, so treated like a 503.
            if attempt < retries - 1:
                wait = 10 * 2**attempt
                print(f"  {type(error).__name__}; retrying in {wait}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("unreachable")


def _cached(path: Path, download) -> dict | list:
    """Return the cached JSON at `path`, downloading it first if absent."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    payload = download()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name first so an interrupted run never leaves a
    # truncated file that would later be trusted as a complete cache entry.
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(payload), encoding="utf-8")
    partial.replace(path)
    return payload


# --- EIA-930 -----------------------------------------------------------------------


def _utc_months(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """First instant of every UTC calendar month overlapping [start, end)."""
    first = start.tz_convert("UTC").tz_localize(None).to_period("M").to_timestamp()
    last = (end.tz_convert("UTC") - pd.Timedelta(hours=1)).tz_localize(None)
    return list(pd.date_range(first, last, freq="MS"))


def download_eia_month(month: pd.Timestamp, api_key: str) -> list[dict]:
    """All D and DF rows for one UTC month, following the API's pagination."""
    path = config.RAW_DIR / "eia" / f"{config.RESPONDENT}_{month:%Y-%m}.json"

    def download() -> list[dict]:
        # EIA periods are hour-ending, so the hours that *begin* in this month
        # are labelled 01:00 on the 1st through 00:00 on the 1st of the next.
        start = month + pd.Timedelta(hours=1)
        end = month + pd.offsets.MonthBegin(1)
        rows: list[dict] = []
        while True:
            params = [
                ("api_key", api_key),
                ("frequency", "hourly"),
                ("data[0]", "value"),
                ("facets[respondent][]", config.RESPONDENT),
                *[("facets[type][]", code) for code in EIA_TYPES],
                ("start", f"{start:%Y-%m-%dT%H}"),
                ("end", f"{end:%Y-%m-%dT%H}"),
                ("sort[0][column]", "period"),
                ("sort[0][direction]", "asc"),
                ("offset", len(rows)),
                ("length", EIA_PAGE),
            ]
            page = fetch(EIA_URL, params)["response"]
            rows += page["data"]
            if len(rows) >= int(page["total"]) or not page["data"]:
                return rows

    print(f"EIA {config.RESPONDENT} {month:%Y-%m}" + ("" if path.exists() else " (downloading)"))
    return _cached(path, download)


def eia_frame(start: pd.Timestamp, end: pd.Timestamp, api_key: str) -> pd.DataFrame:
    """Demand and ERCOT's day-ahead forecast, hour-beginning UTC, in MW."""
    rows: list[dict] = []
    for month in _utc_months(start, end):
        rows += download_eia_month(month, api_key)
    raw = pd.DataFrame(rows)
    # "megawatthours" over one hour is the hour's average MW.
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw["timestamp"] = pd.to_datetime(raw["period"], format="%Y-%m-%dT%H", utc=True)
    raw["timestamp"] -= pd.Timedelta(hours=1)  # hour-ending -> hour-beginning
    wide = raw.pivot_table(index="timestamp", columns="type", values="value", aggfunc="first")
    return wide.rename(columns=EIA_TYPES).reindex(columns=list(EIA_TYPES.values()))


# --- NOAA GHCNh ---------------------------------------------------------------------


def download_station_year(station: str, year: int) -> pd.DataFrame:
    """One station-year of NOAA readings, only the columns used, cached as Parquet."""
    path = config.RAW_DIR / "noaa" / f"{station}_{year}.parquet"
    if not path.exists():
        print(f"NOAA {station} {year} (downloading)")
        payload = fetch(NOAA_URL.format(station=station, year=year), parse=_check_parquet)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        partial.write_bytes(payload)
        partial.replace(path)
    return pd.read_parquet(path, columns=NOAA_COLUMNS)


def _check_parquet(body: bytes) -> bytes:
    """Reject a truncated download: every Parquet file ends with the magic `PAR1`."""
    if not body.endswith(b"PAR1"):
        raise ValueError("incomplete Parquet file")
    return body


def _good(values: pd.Series, codes: pd.Series, low: float, high: float) -> pd.Series:
    """`values` where they passed quality control and are physically plausible."""
    # NOAA's Parquet files store some variables (relative humidity) as text.
    values = pd.to_numeric(values, errors="coerce")
    code = codes.astype("string").fillna("").str.replace(r"\.0$", "", regex=True)
    ok = code.isin(NOAA_GOOD_QUALITY) & values.between(low, high)
    return values.where(ok)


def station_hourly(readings: pd.DataFrame) -> pd.DataFrame:
    """Hourly temperature and humidity from one station's raw readings.

    Airports report routinely at about :53 past each hour, plus synoptic and
    special reports in between. Each reading is assigned to the nearest whole
    hour (UTC) and the valid readings in an hour are averaged. Gaps of up to
    `MAX_INTERPOLATED_HOURS` are filled linearly; longer gaps stay missing.
    """
    stamp = pd.to_datetime(readings["DATE"], utc=True).dt.round("h")
    hourly = pd.DataFrame({
        "temp_c": _good(readings["temperature"], readings["temperature_Quality_Code"],
                        -30, 50),
        "rh_pct": _good(readings["relative_humidity"],
                        readings["relative_humidity_Quality_Code"], 0, 100),
    }).set_axis(pd.DatetimeIndex(stamp), axis=0).groupby(level=0).mean()
    full = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    return _fill_short_gaps(hourly.reindex(full), MAX_INTERPOLATED_HOURS)


def _fill_short_gaps(frame: pd.DataFrame, max_hours: int) -> pd.DataFrame:
    """Interpolate runs of missing hours no longer than `max_hours`; leave the rest.

    Not pandas' `interpolate(limit=...)`, which fills the first `limit` hours of
    *every* gap -- so half of a ten-hour outage would be invented.
    """
    out = frame.copy()
    interpolated = frame.interpolate(method="time", limit_area="inside")
    for column in frame.columns:
        missing = frame[column].isna()
        run = (missing != missing.shift()).cumsum()
        run_length = missing.groupby(run).transform("sum")
        short = missing & (run_length <= max_hours)
        out.loc[short, column] = interpolated.loc[short, column]
    return out


def observed_weather(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Hourly measured weather for each city, from its airport station."""
    first = (start.tz_convert("UTC") - pd.Timedelta(days=1)).year
    last = (end.tz_convert("UTC") + pd.Timedelta(days=1)).year
    frames = {}
    for city in config.CITIES:
        readings = pd.concat(
            [download_station_year(city.station, year) for year in range(first, last + 1)],
            ignore_index=True,
        )
        frames[city.name] = station_hourly(readings)
    return frames


# --- Open-Meteo ------------------------------------------------------------------


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _years(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[str, str]]:
    """(first_day, last_day) pairs per calendar year covering [start, end), in UTC dates.

    A day of margin on either side, because the local window straddles UTC
    midnight; the surplus is trimmed when the table is assembled.
    """
    first = (start.tz_convert("UTC") - pd.Timedelta(days=1)).date()
    last = (end.tz_convert("UTC") + pd.Timedelta(days=1)).date()
    spans = []
    for year in range(first.year, last.year + 1):
        lo = max(first, pd.Timestamp(f"{year}-01-01").date())
        hi = min(last, pd.Timestamp(f"{year}-12-31").date())
        spans.append((lo.isoformat(), hi.isoformat()))
    return spans


def _hourly_series(payload: dict) -> pd.DataFrame:
    hourly = payload["hourly"]
    index = pd.to_datetime(hourly.pop("time"), utc=True)
    return pd.DataFrame(hourly, index=index).astype(float)


def weather_frame(
    url: str, model: str, variables: tuple[str, ...], kind: str,
    start: pd.Timestamp, end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """One hourly frame per city, cached per city and year."""
    frames: dict[str, pd.DataFrame] = {}
    for city in config.CITIES:
        parts = []
        for first_day, last_day in _years(start, end):
            name = f"{kind}_{_slug(city.name)}_{first_day}_{last_day}.json"
            path = config.RAW_DIR / "weather" / name
            params = {
                "latitude": city.latitude,
                "longitude": city.longitude,
                "start_date": first_day,
                "end_date": last_day,
                "hourly": ",".join(variables),
                "models": model,
                "timezone": "GMT",
                "temperature_unit": "celsius",
            }
            status = "" if path.exists() else " (downloading)"
            print(f"{kind} {city.name} {first_day}..{last_day}{status}")
            parts.append(_hourly_series(_cached(path, lambda p=params: fetch(url, p))))
            time.sleep(0.5)  # be a polite guest on a free API
        frames[city.name] = pd.concat(parts).sort_index()
        frames[city.name] = frames[city.name][~frames[city.name].index.duplicated()]
    return frames


def population_weighted(frames: dict[str, pd.DataFrame], column: str) -> pd.Series:
    """Population-weighted average of `column` across cities.

    If any city is missing an hour, the hour is NaN rather than averaged over
    the cities that remain: a silently changing set of cities would shift the
    series by tenths of a degree with no trace.
    """
    weights = config.city_weights()
    stacked = pd.DataFrame({name: frame[column] for name, frame in frames.items()})
    weighted = sum(stacked[name] * weight for name, weight in weights.items())
    return weighted.where(stacked.notna().all(axis=1))


# --- Assembly ------------------------------------------------------------------------


def build_table(
    start: pd.Timestamp, end: pd.Timestamp, *, forecast_weather_from: pd.Timestamp | None
) -> pd.DataFrame:
    """The project's single input table: one row per UTC hour in [start, end)."""
    api_key = config.eia_api_key()
    if api_key is None:
        print(
            "WARNING: no EIA_API_KEY in the environment or .env; using the public "
            "DEMO_KEY. It allows only a few dozen requests per hour -- fine for "
            "one month, not for the full history. See .env.example."
        )
        api_key = "DEMO_KEY"

    index = pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"),
                          freq="h", inclusive="left", name="timestamp_utc")
    table = eia_frame(start, end, api_key).reindex(index)

    actual = observed_weather(start, end)
    table["temp_c"] = population_weighted(actual, "temp_c").reindex(index)
    table["rh_pct"] = population_weighted(actual, "rh_pct").reindex(index)

    for column in PREVIOUS_RUNS_VARIABLES.values():
        table[column] = np.nan
    if forecast_weather_from is not None and forecast_weather_from < end:
        # Reach back a calibration window before the backtest start (the
        # conformal model calibrates on recent forecasts), then a bias-correction
        # window before that (each forecast is corrected by its recent error),
        # plus two days because each issue time's weather window spans the day.
        margin = pd.Timedelta(days=config.CALIBRATION_DAYS + config.WEATHER_BIAS_DAYS + 2)
        fc_start = max(start, forecast_weather_from - margin)
        try:
            forecast = weather_frame(
                PREVIOUS_RUNS_URL, PREVIOUS_RUNS_MODEL, tuple(PREVIOUS_RUNS_VARIABLES),
                "forecast_runs", fc_start, end,
            )
        except (RuntimeError, OSError) as error:
            # Forecast weather feeds only the second backtest arm, so an outage
            # on that one API should not block everything else. Whatever was
            # fetched is cached; rerunning later completes it.
            print(
                f"\nWARNING: forecast weather unavailable ({error}).\n"
                "Forecast temperature columns are left empty; rerun this script later.\n"
            )
        else:
            inside = index >= fc_start
            for variable, column in PREVIOUS_RUNS_VARIABLES.items():
                fc = population_weighted(forecast, variable).reindex(index)
                table.loc[inside, column] = fc[inside]

    # A non-positive system demand is a reporting error, not a measurement.
    # Blank it rather than impute: the backtest skips what it cannot score.
    bad = table["demand_mw"] <= 0
    table.loc[bad, "demand_mw"] = np.nan
    return table


def report(table: pd.DataFrame) -> None:
    """What a reviewer would ask first: how much is missing, and what range."""
    print(f"\n{len(table):,} hours, {table.index[0]} .. {table.index[-1]} (UTC, hour-beginning)")
    for column in table.columns:
        series = table[column]
        missing = int(series.isna().sum())
        if series.notna().any():
            print(f"  {column:22s} missing {missing:6,d}  "
                  f"min {series.min():10,.1f}  max {series.max():10,.1f}")
        else:
            print(f"  {column:22s} missing {missing:6,d}  (no data)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--month", help="download only this local month, e.g. 2025-07")
    args = parser.parse_args(argv)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if args.month:
        start = pd.Timestamp(args.month, tz=config.LOCAL_TZ)
        end = start + pd.offsets.MonthBegin(1)
        # For a probe, fetch forecast weather too, so its coverage is checked.
        table = build_table(start, end, forecast_weather_from=start)
        out = config.PROCESSED_DIR / f"probe_{args.month}.csv"
    else:
        table = build_table(config.DATA_START, config.DATA_END,
                            forecast_weather_from=config.BACKTEST_START)
        out = config.PROCESSED_FILE

    report(table)
    table.to_csv(out, float_format="%.3f")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
