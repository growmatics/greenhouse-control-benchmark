"""Unit tests for arm 2 (black-box RL controller) that need no simulator."""

import numpy as np

from greenlight_bench.controllers.base import ControlContext, Controller
from greenlight_bench.controllers.rl import RLController


def _ctx(obs: list[float]) -> ControlContext:
    return ControlContext(
        obs=np.array(obs), state=np.zeros(28), forecast=np.zeros((4, 10)), timestep=0
    )


def test_rl_controller_is_incremental_and_forwards_the_observation() -> None:
    # the policy sees only the observation and returns an incremental rate action
    seen: list[np.ndarray] = []

    def policy(obs: np.ndarray) -> np.ndarray:
        seen.append(obs)
        return np.full(6, 0.25)

    ctrl = RLController(policy)
    assert isinstance(ctrl, Controller)  # satisfies the shared interface
    assert ctrl.control_kind == "incremental"  # routes through env.step, not step_raw_control
    action = ctrl.act(_ctx([1.0, 2.0, 3.0]))
    assert np.allclose(action, 0.25)
    assert np.allclose(seen[0], [1.0, 2.0, 3.0])  # forwarded obs, not state
