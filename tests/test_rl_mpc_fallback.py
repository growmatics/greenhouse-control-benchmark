"""Arm-4 advisor fallback: an advisory-path failure degrades the step, never the season.

The mock MPC ENCODES the real solve() contract (terminal_weight > 0 refuses a missing
terminal_target): the first production fallback (2026-07-17, season 2022/m0.10 s1)
failed precisely because the original test's mock accepted the call the real MPC
rejects. The fallback must therefore pass the CURRENT state as the terminal target.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from greenlight_bench.controllers.base import ControlContext
from greenlight_bench.controllers.rl_mpc import PhysicsInformedMPC

_N_CONTROLS = 6


def _contract_mpc() -> MagicMock:
    mpc = MagicMock()
    mpc.horizon = 4

    def solve(x, w, terminal_target=None, state_guess=None):
        if terminal_target is None:  # the real guard in PhysicsOnlyMPC.solve
            raise ValueError("terminal_weight > 0 requires a terminal_target in solve()")
        return np.array([0.5] * _N_CONTROLS)

    mpc.solve.side_effect = solve
    return mpc


def _bare_controller(mpc: MagicMock) -> PhysicsInformedMPC:
    c = PhysicsInformedMPC.__new__(PhysicsInformedMPC)
    c.policy, c.rollout_env, c.mpc, c.horizon = MagicMock(), MagicMock(), mpc, 4
    c._n_selected_rl, c._n_steps = 0, 0
    c.advisor_stats = {"guided": 0, "fallback": 0}
    return c


def test_advisor_failure_falls_back_to_plain_mpc_with_current_state_target() -> None:
    mpc = _contract_mpc()
    c = _bare_controller(mpc)
    ctx = MagicMock(spec=ControlContext)
    ctx.state = np.linspace(0.1, 2.8, 28)
    ctx.forecast = None
    ctx.env = MagicMock()
    with patch(
        "greenlight_bench.controllers.rl_mpc.rollout_policy",
        side_effect=RuntimeError("CV_CONV_FAILURE"),
    ):
        u = c.act(ctx)
    assert c.advisor_stats == {"guided": 0, "fallback": 1}
    assert np.allclose(u, 0.5)
    _args, kwargs = mpc.solve.call_args
    assert np.allclose(kwargs["terminal_target"], ctx.state)  # stay-near-here, no advice
