"""Tests for the weather ingestion pipeline (pure; no simulator)."""

import numpy as np
import pandas as pd
import pytest

from greenlight_bench.data.weather import (
    GLG2_WEATHER_COLUMNS,
    build_weather_year,
    estimate_sky_temperature,
)


def _synthetic_hourly(year: int) -> pd.DataFrame:
    idx = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="1h", inclusive="left")
    phase = np.linspace(0.0, 2.0 * np.pi * 365, len(idx))
    return pd.DataFrame(
        {
            "radiation": np.clip(np.sin(phase) * 400.0, 0.0, None),
            "wind_speed": np.full(len(idx), 3.0),
            "air_temp": 10.0 + 5.0 * np.sin(phase),
            "rh": np.full(len(idx), 75.0),
            "elec_price": 50.0 + 20.0 * np.cos(phase),
        },
        index=idx,
    )


def test_builds_full_year_5min_grid() -> None:
    df = build_weather_year(_synthetic_hourly(2026), 2026)
    assert len(df) == 365 * 24 * 12  # 2026 is not a leap year
    assert df["time"].iloc[1] - df["time"].iloc[0] == 300.0
    assert df["time"].iloc[0] == 0.0
    assert list(df.columns)[: len(GLG2_WEATHER_COLUMNS)] == GLG2_WEATHER_COLUMNS
    assert "electricity_price" in df.columns
    assert df["CO2 concentration"].eq(400.0).all()
    assert df["day number"].max() == 364.0


def test_sky_temperature_is_not_zero_and_below_air() -> None:
    air = np.array([-5.0, 0.0, 20.0])
    sky = estimate_sky_temperature(air)
    assert np.all(np.isfinite(sky))
    assert np.all(sky != 0.0)
    assert np.all(sky < air)  # a clear sky is colder than the air


def test_missing_columns_rejected() -> None:
    bad = pd.DataFrame(
        {"radiation": [1.0]}, index=pd.date_range("2026-01-01", periods=1, freq="1h")
    )
    with pytest.raises(ValueError):
        build_weather_year(bad, 2026)


def test_requires_datetime_index() -> None:
    bad = pd.DataFrame({"radiation": [1.0], "wind_speed": [1.0], "air_temp": [1.0], "rh": [1.0]})
    with pytest.raises(TypeError):
        build_weather_year(bad, 2026)
