"""Tests for the pure fetch helpers (no network)."""

import numpy as np
import pytest

from greenlight_bench.data.fetch import open_meteo_hourly_to_raw, parse_years


def test_parse_years_range_and_list() -> None:
    assert parse_years("2019-2023") == [2019, 2020, 2021, 2022, 2023]
    assert parse_years("2020,2022") == [2020, 2022]
    assert parse_years("2021") == [2021]


def test_parse_years_rejects_reversed_range() -> None:
    with pytest.raises(ValueError):
        parse_years("2023-2019")


def test_open_meteo_parser_maps_and_indexes() -> None:
    hourly = {
        "time": ["2023-01-01T00:00", "2023-01-01T01:00"],
        "shortwave_radiation": [0.0, 12.0],
        "temperature_2m": [3.0, 3.5],
        "relative_humidity_2m": [80.0, 82.0],
        "wind_speed_10m": [4.0, 4.5],
    }
    raw = open_meteo_hourly_to_raw(hourly)
    assert list(raw.columns) == ["radiation", "air_temp", "rh", "wind_speed"]
    assert len(raw) == 2
    assert raw.index[0].year == 2023
    assert np.allclose(raw["radiation"].to_numpy(), [0.0, 12.0])


def test_open_meteo_parser_rejects_missing_fields() -> None:
    with pytest.raises(ValueError):
        open_meteo_hourly_to_raw({"time": ["2023-01-01T00:00"], "temperature_2m": [3.0]})
