"""Tests for the pure TomatoEnv info parser (no simulator needed)."""

import numpy as np
import pytest

from greenlight_bench.envs.glg2 import VIOLATION_FIELDS, season_result_from_infos
from greenlight_bench.metrics.safety import safety_profile_from_exceedance


def _info(revenue: float, variable: float, fixed: float, **violations: float) -> dict:
    base = {
        "revenue": revenue,
        "variable_costs": variable,
        "fixed_costs": fixed,
        "temp_violation": 0.0,
        "co2_violation": 0.0,
        "rh_violation": 0.0,
        "lamp_violation": 0.0,
    }
    base.update(violations)
    return base


def test_net_economic_value_is_revenue_minus_variable_costs() -> None:
    # net excludes fixed costs (a control-independent offset), per the benchmark
    # objective and GLG2's own GreenhouseReward.profit.
    infos = [
        _info(10.0, 2.0, 1.0),
        _info(5.0, 1.0, 1.0),
    ]
    result = season_result_from_infos(infos)
    assert result.net_economic_value == pytest.approx(10.0 + 5.0 - 2.0 - 1.0)


def test_exceedances_parsed_per_class() -> None:
    infos = [
        _info(0.0, 0.0, 0.0, temp_violation=1.5, rh_violation=0.0),
        _info(0.0, 0.0, 0.0, temp_violation=0.0, rh_violation=2.0),
    ]
    result = season_result_from_infos(infos)
    assert set(result.exceedances) == set(VIOLATION_FIELDS)
    assert np.allclose(result.exceedances["temperature"], [1.5, 0.0])
    assert np.allclose(result.exceedances["humidity"], [0.0, 2.0])
    # feed one class straight into the safety profile
    prof = safety_profile_from_exceedance([result.exceedances["temperature"]], step_hours=0.25)
    assert prof.incidence == 1.0
    assert prof.worst_excursion == 1.5


def test_empty_infos_rejected() -> None:
    with pytest.raises(ValueError):
        season_result_from_infos([])


def test_missing_info_key_raises_not_defaults() -> None:
    # a dropped/renamed gl_gym key must fail loudly, not silently report 0 (would be a
    # plausible-but-wrong metric)
    incomplete = {
        "variable_costs": 1.0,
        "temp_violation": 0.0,
        "co2_violation": 0.0,
        "rh_violation": 0.0,
        "lamp_violation": 0.0,
    }  # no "revenue"
    with pytest.raises(KeyError, match="missing expected keys"):
        season_result_from_infos([incomplete])
