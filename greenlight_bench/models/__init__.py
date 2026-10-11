"""Reusable GreenLight model pieces shared by the physics controllers (arms 3, 4).

``outputs`` holds the symbolic (CasADi) output functions (CO2 ppm, RH, saturation
vapour pressure) and the state->output map, ported from ``gl_gym`` and validated
against its numpy originals in the tests. Keeping them here (not inside a single
controller) is the single home for this physics: arm 3 and arm 4 import the same
functions, and the package ``__init__`` re-exports them as the public inventory.
"""

from greenlight_bench.models.outputs import (
    I_CO2_AIR,
    I_TAIR,
    I_VPAIR,
    co2_dens_to_ppm,
    outputs_from_state,
    sat_vp,
    vapor_pres_to_rh,
)

__all__ = [
    "I_CO2_AIR",
    "I_TAIR",
    "I_VPAIR",
    "co2_dens_to_ppm",
    "outputs_from_state",
    "sat_vp",
    "vapor_pres_to_rh",
]
