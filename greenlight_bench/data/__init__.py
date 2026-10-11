"""Weather-data ingestion for GreenLight-Gym2.

Turns a raw hourly / sub-hourly weather (and optional price) table into a
GreenLight-Gym-compatible per-year CSV at 5-minute resolution, with the exact
column schema the simulator's ``load_weather_data`` reads.
"""

from greenlight_bench.data.fetch import (
    fetch_open_meteo,
    fetch_year_raw,
    open_meteo_hourly_to_raw,
    parse_years,
)
from greenlight_bench.data.weather import (
    GLG2_WEATHER_COLUMNS,
    build_weather_year,
    estimate_sky_temperature,
    write_weather_csv,
)

__all__ = [
    "GLG2_WEATHER_COLUMNS",
    "build_weather_year",
    "estimate_sky_temperature",
    "fetch_open_meteo",
    "fetch_year_raw",
    "open_meteo_hourly_to_raw",
    "parse_years",
    "write_weather_csv",
]
