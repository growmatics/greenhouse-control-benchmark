"""Unit tests for the isolated MPC arm's pure pieces (no simulator).

The collocation math, the scaled state box, and the inverse-crime boundary are
tested here. The full closed-loop solve is an integration test validated
separately (examples/validate_mpc.py), not run in CI.
"""

import numpy as np
import pytest

from greenlight_bench.controllers.base import GroundTruthParameters, NominalParameters
from greenlight_bench.controllers.mpc import (
    _SINGULAR_IDX,
    STATE_SCALE,
    PhysicsOnlyMPC,
    _domain_safe,
    _state_box,
    radau_coefficients,
)


def test_radau_end_weights_partition_unity() -> None:
    coll, end = radau_coefficients(3)
    assert coll.shape == (4, 4)
    assert end.shape == (4,)
    # Lagrange basis sums to 1 everywhere, so the end-interpolation weights sum to 1.
    assert end.sum() == pytest.approx(1.0)


def test_state_box_is_loose_conditioning_only() -> None:
    lb, ub = _state_box(28)
    assert lb.shape == (28,)
    assert ub.shape == (28,)
    assert (ub > lb).all()
    # The box is now loose (numerical conditioning only): domain safety moved to
    # _domain_safe, and a tight physical bound would make the OCP infeasible (IPOPT
    # never violates a variable bound). Temperature is not tightly bounded, and the
    # divisor states are not floored by the box.
    assert lb[2] < -1.0 and ub[2] > 6.0  # air temperature not tightly boxed
    assert lb[0] < 0.0  # CO2 no longer floored by the box (regularised instead)


def test_domain_safe_floors_divisor_states_and_is_identity_elsewhere() -> None:
    z = np.full(28, -5.0)  # every state pushed negative (an infeasible IPOPT trial point)
    safe = np.asarray(_domain_safe(z, 28)).flatten()
    for i in range(28):
        if i in _SINGULAR_IDX:
            assert safe[i] > 0.0  # divisor states floored strictly positive -> ODE finite
        else:
            assert safe[i] == pytest.approx(-5.0)  # other states untouched
    # for a large positive value the floor is a no-op (identity to O(eps^2))
    zbig = np.full(28, 3.0)
    assert np.asarray(_domain_safe(zbig, 28)).flatten()[0] == pytest.approx(3.0, rel=1e-4)


def test_state_scale_covers_all_states() -> None:
    assert STATE_SCALE.shape == (28,)
    assert (STATE_SCALE > 0).all()


def test_mpc_refuses_ground_truth_parameters() -> None:
    # the inverse-crime boundary: a physics controller may not hold the sim's truth
    with pytest.raises(TypeError, match="inverse-crime"):
        PhysicsOnlyMPC(GroundTruthParameters({"0": 1.0}))  # type: ignore[arg-type]


def test_mpc_constructs_with_nominal_parameters() -> None:
    mpc = PhysicsOnlyMPC(NominalParameters({str(i): 0.0 for i in range(5)}), horizon=2)
    assert mpc.horizon == 2
    assert np.asarray(mpc._params_vec).shape == (5,)


def test_solve_stats_initializes_and_resets() -> None:
    # the per-solve reliability tally (clean/salvage/hold) starts empty and reset() clears it
    mpc = PhysicsOnlyMPC(NominalParameters({str(i): 0.0 for i in range(5)}), horizon=2)
    assert mpc.solve_stats == {"clean": 0, "salvage": 0, "hold": 0}
    mpc.solve_stats["clean"] = 7
    mpc.solve_stats["hold"] = 2
    mpc.reset()
    assert mpc.solve_stats == {"clean": 0, "salvage": 0, "hold": 0}


def test_mpc_rejects_unknown_objective() -> None:
    with pytest.raises(ValueError, match="objective must be"):
        PhysicsOnlyMPC(NominalParameters({"0": 0.0}), objective="bogus")


def test_economic_objective_requires_its_config() -> None:
    # objective='economic' without economic=/climate_bounds= is a construction error
    with pytest.raises(ValueError, match="requires economic"):
        PhysicsOnlyMPC(NominalParameters({"0": 0.0}), objective="economic")


def test_collocation_integrates_a_linear_ode() -> None:
    # Validation layer 1: the Radau coefficients must integrate a known ODE.
    # For dx/dt = a*x on [0, dt], one Radau-3 step must match exp(a*dt)*x0.
    coll, end = radau_coefficients(3)
    a, dt, x0, degree = -0.5, 1.0, 1.0, 3
    # collocation eqs (point j): sum_{r=0..K} C[r,j]*X_r = dt*a*X_j, with X_0 = x0
    mat = np.zeros((degree, degree))
    rhs = np.zeros(degree)
    for j in range(1, degree + 1):
        for r in range(1, degree + 1):
            mat[j - 1, r - 1] = coll[r, j] - (dt * a if r == j else 0.0)
        rhs[j - 1] = -coll[0, j] * x0
    x_coll = np.linalg.solve(mat, rhs)
    x_end = end[0] * x0 + sum(end[r] * x_coll[r - 1] for r in range(1, degree + 1))
    assert x_end == pytest.approx(np.exp(a * dt) * x0, abs=1e-3)
