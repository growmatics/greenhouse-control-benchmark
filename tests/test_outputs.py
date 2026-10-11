"""Unit tests for the symbolic GreenLight output functions.

These recompute the reference (numpy) formulas inline and check the symbolic
CasADi functions match, at sample points and on a CasADi symbol. This pins the
math to the simulator's equations without needing gl_gym (an integration dep);
``examples/validate_mpc.py`` additionally checks against the live env.
"""

import numpy as np
import pytest

from greenlight_bench.models.outputs import (
    I_CO2_AIR,
    I_TAIR,
    I_VPAIR,
    co2_dens_to_ppm,
    outputs_from_state,
    sat_vp,
    vapor_pres_to_rh,
)


def _ref_sat_vp(temp: float) -> float:
    return 610.78 * np.exp(17.2694 * temp / (temp + 238.3))


def _ref_co2_ppm(temp: float, dens_kg_m3: float) -> float:
    return 1e6 * 8.3144598 * (temp + 273.15) * dens_kg_m3 / (101325 * 44.01e-3)


@pytest.mark.parametrize("temp", [-5.0, 0.0, 12.0, 20.0, 34.0])
def test_sat_vp_matches_reference(temp: float) -> None:
    assert float(sat_vp(temp)) == pytest.approx(_ref_sat_vp(temp), rel=1e-9)


@pytest.mark.parametrize(("temp", "dens"), [(15.0, 8e-4), (20.0, 1.2e-3), (25.0, 2.7e-3)])
def test_co2_dens_to_ppm_matches_reference(temp: float, dens: float) -> None:
    assert float(co2_dens_to_ppm(temp, dens)) == pytest.approx(_ref_co2_ppm(temp, dens), rel=1e-9)


def test_vapor_pres_to_rh_matches_reference() -> None:
    temp, vp = 20.0, 1500.0
    assert float(vapor_pres_to_rh(temp, vp)) == pytest.approx(
        100.0 * vp / _ref_sat_vp(temp), rel=1e-9
    )


def test_outputs_from_state_uses_correct_indices_and_units() -> None:
    # CO2 state is mg/m^3 (converted to kg/m^3 before the ppm conversion).
    x = np.zeros(28)
    x[I_TAIR] = 20.0
    x[I_CO2_AIR] = 1200.0  # mg/m^3
    x[I_VPAIR] = 1500.0  # Pa
    co2_ppm, temp, rh = outputs_from_state(x)
    assert float(temp) == 20.0
    assert float(co2_ppm) == pytest.approx(_ref_co2_ppm(20.0, 1200.0 * 1e-6), rel=1e-9)
    assert float(rh) == pytest.approx(100.0 * 1500.0 / _ref_sat_vp(20.0), rel=1e-9)


def test_functions_accept_casadi_symbols() -> None:
    ca = pytest.importorskip("casadi")
    t = ca.SX.sym("t")
    v = ca.SX.sym("v")
    # building an expression must not raise, and evaluating it matches the reference
    f = ca.Function("f", [t, v], [vapor_pres_to_rh(t, v)])
    assert float(f(20.0, 1500.0)) == pytest.approx(100.0 * 1500.0 / _ref_sat_vp(20.0), rel=1e-9)
