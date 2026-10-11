"""Controller interface and the physics-arm boundary.

All four comparators implement one ``Controller`` protocol so the harness runs
any of them unchanged. Physics controllers enforce the inverse-crime boundary:
they may hold only a nominal model, never the simulator's ground truth.
"""

from greenlight_bench.controllers.base import (
    ControlContext,
    Controller,
    GroundTruthParameters,
    NominalParameters,
    PhysicsController,
)
from greenlight_bench.controllers.rl_mpc import PhysicsInformedMPC, load_ppo_policy
from greenlight_bench.controllers.rule_based import RuleBasedIncumbent

__all__ = [
    "ControlContext",
    "Controller",
    "GroundTruthParameters",
    "NominalParameters",
    "PhysicsController",
    "PhysicsInformedMPC",
    "RuleBasedIncumbent",
    "load_ppo_policy",
]
