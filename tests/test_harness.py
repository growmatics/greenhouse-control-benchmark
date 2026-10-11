"""Regression tests for the season-runner harness's control-convention dispatch.

The bug these guard against: GLG2's ``env.step`` treats the action as an *incremental*
rate change (``u <- u + a*delta_u_max``, for RL), while ``env.step_raw_control`` sets the
control *absolutely* (for the physics MPC and the rule-based incumbent). Feeding an
absolute control through ``env.step`` silently applies tiny increments -- the plant then
executes different controls than the controller planned. ``run_season_controller`` must
dispatch on ``controller.control_kind`` so the control reaches the plant unchanged.
"""

import numpy as np

from greenlight_bench.controllers.base import ControlContext
from greenlight_bench.envs.glg2 import run_season_controller


class _FakeEnv:
    """Minimal env recording which step method received which control."""

    def __init__(self, n_steps: int = 3) -> None:
        self.n_steps = n_steps
        self.timestep = 0
        self.x = np.zeros(28)
        self.weather_data = np.zeros((n_steps + 30, 10))
        self.raw_calls: list[np.ndarray] = []
        self.gym_calls: list[np.ndarray] = []

    def reset(self, seed: int = 0):
        self.timestep = 0
        return np.zeros(22), {}

    def _info(self):
        return {
            "revenue": 1.0,
            "variable_costs": 0.2,
            "fixed_costs": 0.1,
            "temp_violation": 0.0,
            "co2_violation": 0.0,
            "rh_violation": 0.0,
            "lamp_violation": 0.0,
        }

    def _advance(self):
        self.timestep += 1
        return np.zeros(22), 0.0, self.timestep >= self.n_steps, False, self._info()

    def step_raw_control(self, control):
        self.raw_calls.append(np.asarray(control, dtype=float))
        return self._advance()

    def step(self, action):
        self.gym_calls.append(np.asarray(action, dtype=float))
        return self._advance()


class _Controller:
    def __init__(self, kind: str) -> None:
        self.control_kind = kind
        self.output = np.array([0.5, 0.3, 0.0, 0.7, 0.0, 0.1])

    def reset(self, seed: int) -> None:
        pass

    def act(self, ctx: ControlContext) -> np.ndarray:
        return self.output


def test_absolute_controller_uses_step_raw_control_unchanged() -> None:
    env = _FakeEnv()
    ctrl = _Controller("absolute")
    run_season_controller(env, ctrl, seed=0)
    assert len(env.raw_calls) == env.n_steps  # absolute controls went to step_raw_control
    assert len(env.gym_calls) == 0  # never the incremental gym step
    # the control reached the plant unchanged (not throttled to an increment)
    np.testing.assert_allclose(env.raw_calls[0], ctrl.output)


def test_incremental_controller_uses_gym_step() -> None:
    env = _FakeEnv()
    ctrl = _Controller("incremental")
    run_season_controller(env, ctrl, seed=0)
    assert len(env.gym_calls) == env.n_steps  # RL-style actions go to the gym step
    assert len(env.raw_calls) == 0


def test_default_control_kind_is_absolute() -> None:
    # a controller without control_kind is treated as absolute (the physics-arm default)
    env = _FakeEnv()

    class _NoKind:
        def reset(self, seed: int) -> None:
            pass

        def act(self, ctx: ControlContext) -> np.ndarray:
            return np.zeros(6)

    run_season_controller(env, _NoKind(), seed=0)
    assert len(env.raw_calls) == env.n_steps
    assert len(env.gym_calls) == 0
