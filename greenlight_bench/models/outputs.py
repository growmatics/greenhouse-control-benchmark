"""Symbolic GreenLight output functions (CO2 ppm, RH, saturation vapour pressure).

Ported from ``gl_gym.environments.utils`` (numpy) so the same equations the
simulator uses to score constraint violations can be embedded, symbolically, in
the MPC objective. Every function works on plain floats/numpy *and* on CasADi
symbols (only ``exp`` and arithmetic are used). The tests pin these against the
numpy originals, so the version we optimise is the simulator's own math.

Source: Katzin GreenLight; ``gl_gym/environments/utils.py`` (``satVp``,
``co2dens2ppm``, ``vaporPres2rh``) and the indoor-climate observation map in
``gl_gym/environments/observations.py`` (states ``[0, 2, 15]`` -> CO2/temp/RH).
"""

from __future__ import annotations

from typing import Any

# GreenLight state indices for the indoor-climate outputs (from observations.py:74).
I_CO2_AIR = 0  # CO2 air density [mg m^-3]
I_TAIR = 2  # air temperature [degC]
I_VPAIR = 15  # air vapour pressure [Pa]

# Physical constants (ideal-gas CO2 conversion), verbatim from gl_gym utils.
_R = 8.3144598  # molar gas constant [J mol^-1 K^-1]
_C2K = 273.15  # Celsius -> Kelvin
_M_CO2 = 44.01e-3  # molar mass of CO2 [kg mol^-1]
_P_ATM = 101325  # pressure, 1 atm [Pa]

# Saturation-vapour-pressure coefficients (Magnus form), verbatim from satVp.
_SVP_A = 610.78
_SVP_B = 17.2694
_SVP_C = 238.3
# Pole-safe denominator floor: the Magnus form diverges at temp = -_SVP_C, and the MPC's
# economic penalty evaluates satVp on unbounded interval-end states during IPOPT line
# search, which can cross the pole and NaN. eps << the physical denominator (~200-320), so
# the value is unchanged to < 1e-9 at real temperatures.
_SVP_POLE_EPS = 1e-4


def _svp_denom(temp: Any) -> Any:
    """Pole-safe denominator sqrt((temp + C)^2 + eps^2): equals (temp + C) at physical
    temperatures, stays strictly positive everywhere (a differentiable domain extension)."""
    import casadi as ca

    return ca.sqrt((temp + _SVP_C) * (temp + _SVP_C) + _SVP_POLE_EPS * _SVP_POLE_EPS)


# mg -> kg, for the CO2 air-density state which is stored in mg m^-3.
_MG_TO_KG = 1e-6


def sat_vp(temp: Any) -> Any:
    """Saturation vapour pressure [Pa] at air temperature ``temp`` [degC].

    Uses a pole-safe denominator so the function is finite for *all* ``temp`` (the MPC
    optimiser probes unphysical temperatures); at physical temperatures it equals the
    Magnus form to < 1e-9.
    """
    import casadi as ca

    return _SVP_A * ca.exp(_SVP_B * temp / _svp_denom(temp))


def co2_dens_to_ppm(temp: Any, dens_kg_m3: Any) -> Any:
    """CO2 density [kg m^-3] -> molar concentration [ppm] at temperature ``temp`` [degC]."""
    return 1e6 * _R * (temp + _C2K) * dens_kg_m3 / (_P_ATM * _M_CO2)


def vapor_pres_to_rh(temp: Any, vapor_pres: Any) -> Any:
    """Vapour pressure [Pa] -> relative humidity [%].

    Unclipped, unlike the numpy ``vaporPres2rh`` (which clips to [0, 100]): the MPC
    objective needs a smooth function, and in the operating range the values agree.
    The RH *upper*-bound penalty is unaffected by the missing 100 % clip.
    """
    return 100.0 * vapor_pres / sat_vp(temp)


def outputs_from_state(x: Any) -> tuple[Any, Any, Any]:
    """Map a state vector to the three penalised climate outputs.

    Returns ``(co2_ppm, temp_c, rh_percent)`` from states ``[0, 2, 15]`` exactly as
    ``IndoorClimateObservations.compute_obs`` does (CO2 state is mg m^-3, converted
    to kg m^-3 before the ppm conversion). Works for numpy or CasADi ``x``.
    """
    temp = x[I_TAIR]
    co2_ppm = co2_dens_to_ppm(temp, x[I_CO2_AIR] * _MG_TO_KG)
    rh = vapor_pres_to_rh(temp, x[I_VPAIR])
    return co2_ppm, temp, rh
