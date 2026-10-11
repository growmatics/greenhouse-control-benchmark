"""Fetch real weather (Open-Meteo) and optional day-ahead prices (ENTSO-E).

Open-Meteo's archive API is free and keyless. ENTSO-E day-ahead prices need a
free API token and the optional ``entsoe`` package. The JSON-to-frame parser and
the year-range parser are pure and unit-tested; the network calls are
integration-only (excluded from coverage).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

import pandas as pd

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_OPEN_METEO_HOURLY = [
    "shortwave_radiation",  # W/m2
    "temperature_2m",  # degC
    "relative_humidity_2m",  # %
    "wind_speed_10m",  # m/s (with wind_speed_unit=ms)
]
_OM_RENAME = {
    "shortwave_radiation": "radiation",
    "temperature_2m": "air_temp",
    "relative_humidity_2m": "rh",
    "wind_speed_10m": "wind_speed",
}


def parse_years(spec: str) -> list[int]:
    """Parse ``"2019-2023"`` or ``"2020,2022"`` into a list of years."""
    spec = spec.strip()
    if "-" in spec:
        start, end = (int(x) for x in spec.split("-", 1))
        if end < start:
            raise ValueError(f"year range end {end} precedes start {start}")
        return list(range(start, end + 1))
    return [int(y) for y in spec.split(",") if y.strip()]


def open_meteo_hourly_to_raw(hourly: dict[str, Any]) -> pd.DataFrame:
    """Convert an Open-Meteo ``hourly`` block to a raw weather frame (pure)."""
    missing = ({"time", *_OPEN_METEO_HOURLY}) - set(hourly)
    if missing:
        raise ValueError(f"open-meteo response missing fields: {sorted(missing)}")
    index = pd.to_datetime(hourly["time"])
    data = {new: hourly[old] for old, new in _OM_RENAME.items()}
    return pd.DataFrame(data, index=index)


def fetch_open_meteo(
    lat: float, lon: float, year: int, timezone: str = "UTC"
) -> pd.DataFrame:  # pragma: no cover - network
    """Fetch one calendar year of hourly weather from the Open-Meteo archive."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": ",".join(_OPEN_METEO_HOURLY),
        "wind_speed_unit": "ms",
        "timezone": timezone,
    }
    url = f"{OPEN_METEO_ARCHIVE_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.load(response)
    return open_meteo_hourly_to_raw(payload["hourly"])


def fetch_entsoe_prices(
    token: str, zone: str, year: int, tz: str = "Europe/Amsterdam"
) -> pd.Series:  # pragma: no cover - network + token
    """Day-ahead electricity prices [EUR/MWh] for one year, UTC-naive hourly."""
    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=token)
    start = pd.Timestamp(f"{year}-01-01", tz=tz)
    end = pd.Timestamp(f"{year + 1}-01-01", tz=tz)
    series = client.query_day_ahead_prices(zone, start=start, end=end)
    series = series.tz_convert("UTC").tz_localize(None)
    series.name = "elec_price"
    return series


def fetch_year_raw(
    lat: float,
    lon: float,
    year: int,
    *,
    timezone: str = "UTC",
    entsoe_token: str | None = None,
    entsoe_zone: str | None = None,
) -> pd.DataFrame:  # pragma: no cover - network
    """Fetch one year of real weather, plus prices if an ENTSO-E token is given."""
    raw = fetch_open_meteo(lat, lon, year, timezone=timezone)
    if entsoe_token and entsoe_zone:
        prices = fetch_entsoe_prices(entsoe_token, entsoe_zone, year)
        aligned = prices.reindex(raw.index.union(prices.index)).interpolate(method="time")
        raw["elec_price"] = aligned.reindex(raw.index).to_numpy()
    return raw
