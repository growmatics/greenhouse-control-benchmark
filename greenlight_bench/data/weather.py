"""Build a GreenLight-Gym2 weather CSV from a raw weather/price table.

Schema truth (from ``gl_gym.environments.utils.load_weather_data``): the loader
reads, by column name, ``time`` (seconds since Jan 1), ``global radiation`` [W/m2],
``air temperature`` [degC], ``RH`` [%], ``wind speed`` [m/s], and ``sky temperature``
[degC]. It ignores ``CO2 concentration`` (forced to 400 ppm internally), ``day
number``, and the unused ``??`` column, and it PCHIP-resamples to the solver step,
so source resolution is flexible. We still emit the full real-file layout for
drop-in compatibility, plus an ``electricity_price`` column for the (not-yet-wired)
energy-price extension.

    from greenlight_bench.data import build_weather_year, write_weather_csv
    df = build_weather_year(raw, 2026)          # raw: DatetimeIndex, 5 columns
    write_weather_csv(df, "gl_gym/environments/weather/Custom_Location/2026.csv")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Exact header order of the bundled GreenLight-Gym weather files. The loader
# matches by NAME, so order is cosmetic, but we keep it for drop-in parity.
GLG2_WEATHER_COLUMNS: list[str] = [
    "time",
    "global radiation",
    "wind speed",
    "air temperature",
    "sky temperature",
    "??",
    "CO2 concentration",
    "day number",
    "RH",
]

# Columns the loader actually consumes (used for validation).
REQUIRED_BY_LOADER: frozenset[str] = frozenset(
    {"time", "global radiation", "wind speed", "air temperature", "sky temperature", "RH"}
)

# Mapping from raw input names to GreenLight names.
_RAW_TO_GLG2 = {
    "radiation": "global radiation",
    "wind_speed": "wind speed",
    "air_temp": "air temperature",
    "rh": "RH",
}

_STEP_SECONDS = 300  # 5-minute target grid
_SECONDS_PER_DAY = 86_400
_CO2_BASELINE_PPM = 400.0
_KELVIN = 273.15


def estimate_sky_temperature(air_temp_c: NDArray[np.float64]) -> NDArray[np.float64]:
    """Clear-sky effective sky temperature [degC] from air temperature (Swinbank).

    ``T_sky = 0.0552 * T_air_K**1.5``. A documented approximation: it ignores
    cloud cover and humidity, so it is a floor for clear nights. Replace with a
    longwave/cloud model when that data is available. Zeroing sky temperature (as
    a naive pipeline might) is physically wrong, the loader feeds it to the FIR
    radiative-exchange terms.
    """
    tair_k = np.asarray(air_temp_c, dtype=float) + _KELVIN
    tsky_k = 0.0552 * tair_k**1.5
    return tsky_k - _KELVIN


def build_weather_year(
    raw: pd.DataFrame,
    year: int,
    *,
    include_price: bool = True,
) -> pd.DataFrame:
    """Resample a raw weather table to a full-year 5-minute GreenLight CSV.

    Args:
        raw: indexed by a (tz-naive) ``DatetimeIndex`` at any resolution, with
            columns ``radiation`` [W/m2], ``wind_speed`` [m/s], ``air_temp`` [degC],
            ``rh`` [%], and optionally ``elec_price`` [EUR/MWh].
        year: target calendar year.
        include_price: emit an ``electricity_price`` column (inert in stock GLG2).

    Returns:
        A DataFrame with the GreenLight schema, one row per 300 s of ``year``.
    """
    missing = {"radiation", "wind_speed", "air_temp", "rh"} - set(raw.columns)
    if missing:
        raise ValueError(f"raw is missing required columns: {sorted(missing)}")
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise TypeError("raw must be indexed by a DatetimeIndex")

    year_start = pd.Timestamp(f"{year}-01-01")
    grid = pd.date_range(
        year_start, f"{year + 1}-01-01", freq=f"{_STEP_SECONDS}s", inclusive="left"
    )

    src = raw.sort_index()
    src = src[~src.index.duplicated(keep="first")]
    # time-based interpolation onto the union, then take the grid; fill the edges.
    resampled = (
        src.reindex(src.index.union(grid)).interpolate(method="time").reindex(grid).ffill().bfill()
    )

    seconds = (grid - year_start).total_seconds().to_numpy().astype(float)
    out = pd.DataFrame(
        {
            "time": seconds,
            "global radiation": resampled["radiation"].to_numpy(),
            "wind speed": resampled["wind_speed"].to_numpy(),
            "air temperature": resampled["air_temp"].to_numpy(),
            "sky temperature": estimate_sky_temperature(resampled["air_temp"].to_numpy()),
            "??": 0.0,
            "CO2 concentration": _CO2_BASELINE_PPM,
            "day number": np.floor(seconds / _SECONDS_PER_DAY),
            "RH": resampled["rh"].to_numpy(),
        }
    )
    if include_price and "elec_price" in resampled.columns:
        out["electricity_price"] = resampled["elec_price"].to_numpy()
    return out


def write_weather_csv(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Write a built weather frame to ``out_path`` (creating parent dirs)."""
    if not REQUIRED_BY_LOADER.issubset(df.columns):
        raise ValueError(
            f"frame is missing loader-required columns: {REQUIRED_BY_LOADER - set(df.columns)}"
        )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
