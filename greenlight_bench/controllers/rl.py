"""Arm 2: the black-box RL controller (a trained policy through the shared harness).

The pure data-driven comparator: a policy trained on the native environment reward that
sees only the observation, no model. Wrapping it as a ``Controller`` (rather than running
it as a bare policy) makes it substitutable in ``run_season_controller`` with the other
three arms -- same online-cost timing, same ``SeasonResult`` -- so the four-way comparison
is apples-to-apples.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from greenlight_bench.controllers.base import ControlContext

Policy = Callable[[NDArray[np.float64]], NDArray[np.float64]]  # obs -> incremental action


class RLController:
    """Arm 2: a trained RL policy as a benchmark ``Controller``.

    Emits *incremental* rate actions in [-1, 1] (the gym convention), so the harness routes
    it through ``env.step`` (``u <- clip(u + a * delta_u_max)``), NOT ``step_raw_control``.
    Getting this convention wrong silently feeds the plant different controls than the
    policy chose (the bug that corrupted the early L5 numbers); ``control_kind`` makes it
    explicit and testable.
    """

    control_kind = "incremental"

    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    def reset(self, seed: int = 0) -> None:
        # a saved deterministic policy carries no per-season state to reset
        pass

    def act(self, ctx: ControlContext) -> NDArray[np.float64]:
        return np.asarray(self.policy(np.asarray(ctx.obs, dtype=float)), dtype=float)
