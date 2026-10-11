"""The ``Controller`` protocol and the inverse-crime boundary (contract rows 15-19).

Every comparator (rule-based incumbent, black-box RL, physics-only MPC,
physics-informed) implements ``Controller``, so they are substitutable in the
harness. ``PhysicsController`` structurally refuses the simulator's ground-truth
parameters: the inverse-crime defence is an assertion, not a comment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ControlContext:
    """Everything a controller might need at one step; each arm uses a subset.

    Bundling these keeps all four arms substitutable behind ``Controller.act``
    even though they need different inputs: the RL arm uses ``obs``, the physics
    arms use the full ``state`` and ``forecast``, and the rule-based incumbent
    also needs the live ``env`` (hour-of-day, radiation sums, screen state). A
    controller reads only the fields it needs and ignores the rest.
    """

    obs: NDArray[np.float64]  # environment observation (lossy; 22-dim) -> RL arm
    state: NDArray[np.float64]  # full simulator state (28-dim) -> physics arms
    forecast: NDArray[np.float64]  # weather disturbances over the horizon -> physics arms
    timestep: int  # index into the season
    env: Any = None  # live env, for controllers that need its internal context


@runtime_checkable
class Controller(Protocol):
    """Uniform control interface. Implementations must be seedable and pure per step."""

    def reset(self, seed: int) -> None:
        """Reset internal state deterministically from ``seed``."""
        ...

    def act(self, ctx: ControlContext) -> NDArray[np.float64]:
        """Map a per-step context to an action in the environment's action space."""
        ...


@dataclass(frozen=True)
class GroundTruthParameters:
    """The simulator's TRUE parameters. Physics controllers must never receive these."""

    values: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class NominalParameters:
    """The nominal model a physics controller is allowed to use."""

    values: dict[str, float] = field(default_factory=dict)


class PhysicsController:
    """Base for the physics-only and physics-informed arms.

    Enforces the inverse-crime boundary: a physics controller may hold only a
    ``NominalParameters`` model, never ``GroundTruthParameters``. Constructing one
    with the simulator's true parameters raises ``TypeError``.
    """

    def __init__(self, params: NominalParameters) -> None:
        if isinstance(params, GroundTruthParameters):
            raise TypeError(
                "inverse-crime violation: a physics controller was given the "
                "simulator's ground-truth parameters; pass NominalParameters"
            )
        if not isinstance(params, NominalParameters):
            raise TypeError("PhysicsController requires NominalParameters")
        self._params = params

    @property
    def params(self) -> NominalParameters:
        return self._params
