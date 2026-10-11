"""Tests for the Controller protocol, ControlContext, and the inverse-crime boundary."""

import numpy as np
import pytest

from greenlight_bench.controllers.base import (
    ControlContext,
    Controller,
    GroundTruthParameters,
    NominalParameters,
    PhysicsController,
)


def _ctx() -> ControlContext:
    return ControlContext(
        obs=np.zeros(22),
        state=np.zeros(28),
        forecast=np.zeros((4, 10)),
        timestep=0,
    )


def test_control_context_holds_the_per_arm_inputs() -> None:
    ctx = _ctx()
    assert ctx.obs.shape == (22,)
    assert ctx.state.shape == (28,)
    assert ctx.forecast.shape == (4, 10)
    assert ctx.timestep == 0
    assert ctx.env is None  # optional; only the incumbent needs it


def test_a_class_with_reset_and_act_satisfies_the_protocol() -> None:
    class Dummy:
        def reset(self, seed: int) -> None:
            pass

        def act(self, ctx: ControlContext) -> np.ndarray:
            return np.zeros(6)

    d = Dummy()
    assert isinstance(d, Controller)
    assert d.act(_ctx()).shape == (6,)


def test_a_class_missing_act_is_not_a_controller() -> None:
    class NoAct:
        def reset(self, seed: int) -> None:
            pass

    assert not isinstance(NoAct(), Controller)


def test_physics_controller_refuses_ground_truth_parameters() -> None:
    with pytest.raises(TypeError, match="inverse-crime"):
        PhysicsController(GroundTruthParameters({"0": 1.0}))  # type: ignore[arg-type]


def test_physics_controller_accepts_nominal_parameters() -> None:
    c = PhysicsController(NominalParameters({"0": 1.0}))
    assert c.params.values == {"0": 1.0}
