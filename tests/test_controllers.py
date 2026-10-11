"""Tests for the Controller interface and the inverse-crime boundary."""

import numpy as np
import pytest

from greenlight_bench.controllers import (
    Controller,
    GroundTruthParameters,
    NominalParameters,
    PhysicsController,
)


class _ConstantController:
    """A trivial controller used to check the protocol is satisfiable."""

    def reset(self, seed: int) -> None:
        self._seed = seed

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(6, dtype=float)


def test_constant_controller_satisfies_protocol() -> None:
    ctrl = _ConstantController()
    assert isinstance(ctrl, Controller)
    ctrl.reset(0)
    assert ctrl.act(np.zeros(22)).shape == (6,)


def test_physics_controller_accepts_nominal_params() -> None:
    ctrl = PhysicsController(NominalParameters({"floor_area": 1.0}))
    assert ctrl.params.values["floor_area"] == 1.0


def test_physics_controller_refuses_ground_truth() -> None:
    with pytest.raises(TypeError, match="inverse-crime"):
        PhysicsController(GroundTruthParameters({"floor_area": 1.0}))  # type: ignore[arg-type]
