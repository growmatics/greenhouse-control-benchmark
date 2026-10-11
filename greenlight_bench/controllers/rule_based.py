"""Arm 1: the rule-based setpoint incumbent (the deployment baseline).

Wraps GreenLight-Gym2's shipped ``RuleBasedController`` (arXiv:2410.05336,
[vanlaatum2024greenlight]): a commercial-style Het-Nieuwe-Telen /
temperature-integration controller with day/night temperature setpoints, CO2
dosing, RH-driven ventilation and mechanical dehumidification, and thermal-screen
and lamp scheduling. This is the bar every learned method must beat.

The incumbent's tuned parameters are its **competence standard** (so the four-way
comparison is fair, not a strawman, per ``method/design.md``): they are injected,
not invented, from GLG2's shipped ``configs/agents/rule_based.yml``. The harness
records the exact parameter set with each run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from greenlight_bench.controllers.base import ControlContext


class RuleBasedIncumbent:
    """Adapter making GLG2's ``RuleBasedController`` a benchmark ``Controller``.

    Args:
        params: the rule-based controller's tuned parameters (the competence
            standard), e.g. loaded from GLG2's ``rule_based.yml``.
    """

    control_kind = "absolute"  # outputs absolute controls -> harness uses step_raw_control

    def __init__(self, params: dict[str, Any]) -> None:  # pragma: no cover - needs gl_gym
        from gl_gym.environments.baseline import RuleBasedController

        self._params = dict(params)
        self._rb = RuleBasedController(**self._params)

    @property
    def params(self) -> dict[str, Any]:
        """The tuned competence-standard parameters, for the run record."""
        return dict(self._params)

    def reset(self, seed: int = 0) -> None:
        """No internal state to reset; kept for the ``Controller`` protocol."""

    def act(self, ctx: ControlContext) -> NDArray[np.float64]:  # pragma: no cover - integration
        """Return the 6 controls for the current step.

        The incumbent needs the full state, the current weather row, and the live
        env (hour-of-day, radiation sums, screen state), so it reads ``ctx.state``,
        ``ctx.env`` and the disturbance at ``ctx.timestep``.
        """
        d = ctx.env.weather_data[ctx.timestep]
        return np.asarray(self._rb.predict(ctx.state, d, ctx.env), dtype=float)
