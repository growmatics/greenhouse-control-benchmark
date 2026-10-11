"""Unit tests for arm 4 (physics-informed RL-in-MPC) pieces that need no simulator.

The rollout and scoring need gl_gym (integration, covered by examples). Here we test
the trajectory-selection decision and the wiring with lightweight fakes.
"""

import numpy as np

from greenlight_bench.controllers.base import NominalParameters
from greenlight_bench.controllers.rl_mpc import PhysicsInformedMPC


class _FakeMPC:
    horizon = 4

    def __init__(self) -> None:
        self.did_reset = False

    def reset(self, seed: int) -> None:
        self.did_reset = True


class _FakeEnv:
    def __init__(self) -> None:
        self.did_reset = False

    def reset(self, seed: int = 0):
        self.did_reset = True
        return None, {}


def test_control_kind_is_absolute() -> None:
    assert PhysicsInformedMPC.control_kind == "absolute"


def test_constructs_and_reset_propagates_to_mpc() -> None:
    mpc = _FakeMPC()
    arm4 = PhysicsInformedMPC(
        NominalParameters({"0": 1.0}), policy=lambda o: np.zeros(6), rollout_env=_FakeEnv(), mpc=mpc
    )
    assert arm4.horizon == 4
    assert arm4.rl_selection_rate == 0.0  # no steps yet
    arm4.reset(0)
    assert mpc.did_reset


def test_selection_rate_reflects_counts() -> None:
    arm4 = PhysicsInformedMPC(
        NominalParameters({"0": 1.0}),
        policy=lambda o: np.zeros(6),
        rollout_env=_FakeEnv(),
        mpc=_FakeMPC(),
    )
    arm4._n_steps = 10
    arm4._n_selected_rl = 3
    assert arm4.rl_selection_rate == 0.3
