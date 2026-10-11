"""The unified benchmark runner: run every arm on ONE shared config, comparably.

``run_benchmark`` is the top-level orchestrator, distinct from
``run_season_controller`` (one arm, one season). It runs the full grid
(arms x conditions x seeds) through that primitive on a *single* environment
configuration, so the four arms are finally cross-comparable, and it aggregates the
per-arm net economic value, the season-level feasibility gate, and per-class safety
profiles. Every result carries a **run record** (config version, git commit, grid,
timestamp) so a reported number traces to one named run -- the audit-trail rule.

This is the public-facing benchmark: at preprint it is the centrepiece of the spun-out
open-source repo ("run our benchmark"). Keep it config-driven and deterministic.

Naming, to keep the levels distinct:
- ``run_season_controller`` -> one arm, one season (the primitive, in ``envs/glg2.py``).
- ``run_benchmark`` -> all arms x conditions x seeds -> ``BenchmarkResult`` + run record.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from greenlight_bench.config import GateConfig
from greenlight_bench.envs.glg2 import SeasonResult, run_season_controller
from greenlight_bench.metrics.gate import passes_soft_gate
from greenlight_bench.metrics.pareto import hypervolume_2d, normalize_minmax, pareto_front
from greenlight_bench.metrics.safety import SafetyProfile, safety_profile_from_exceedance

# A benchmark condition: a label plus the env overrides that define it (weather year,
# construction params, model-mismatch level, ...). ``make_env`` maps it to a live env.
ArmFactory = Callable[[Any, "Condition", int], Any]  # (env, condition, seed) -> Controller

_MIN_PARETO_ARMS = 2  # need at least the incumbent plus one comparator for a front


def hardware_stamp() -> dict[str, Any]:
    """Machine identity for the run record, so the online-cost axis is interpretable.

    ``online_cost`` is wall-clock time (``time.perf_counter``), so it is **relative**:
    comparable across arms on the *same* machine (the Pareto ordering is valid), but the
    absolute ms/step is not portable across hardware. Stamping the machine makes the
    numbers reproducible in context. A fully hardware-independent cost proxy (IPOPT
    iteration count / NLP evaluations per step) is a possible future refinement.
    """
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }


@dataclass(frozen=True)
class Condition:
    """One benchmark condition (a point on the generalisation / mismatch grid).

    ``env_overrides`` configure the *plant* (weather year, construction params). ``mismatch``
    is the model-mismatch magnitude for the *physics arms*: their nominal model is the plant's
    true parameters perturbed by this much (0 = the inverse-crime reference). The arm factory
    reads it from the condition.
    """

    label: str
    env_overrides: dict[str, Any] = field(default_factory=dict)
    mismatch: float = 0.0


@dataclass(frozen=True)
class BenchmarkResult:
    """Every season's result across the grid, plus the run record and gate settings.

    Keyed by ``(arm, condition_label, seed)``. Aggregations (net value, gate, safety)
    are derived on demand so the raw per-season results stay the source of truth.
    """

    seasons: dict[tuple[str, str, int], SeasonResult]
    record: dict[str, Any]  # config version, git commit, grid, timestamp -- traceability
    gate: GateConfig
    soft_classes: tuple[str, ...]  # exceedance classes gated as soft crop-health
    step_hours: float
    # Seasons that CRASHED (per arm). A season the controller could not complete is a
    # deployment failure, not a missing observation: it counts as violating every soft
    # class in the gate (never silently excluded, which would bias toward the arm that
    # crashes on its hardest seasons). Net-value stats are over completed seasons only,
    # reported next to these counts.
    failed: dict[str, int] = field(default_factory=dict)

    def arms(self) -> list[str]:
        return sorted({arm for (arm, _c, _s) in self.seasons} | set(self.failed))

    def _arm_seasons(self, arm: str) -> list[SeasonResult]:
        return [r for (a, _c, _s), r in self.seasons.items() if a == arm]

    def net_value_summary(self, arm: str) -> tuple[float, float]:
        """Mean and standard deviation of net economic value over the arm's COMPLETED
        seasons (a crashed season has no honest number; report it via ``failed`` and the
        gate instead of inventing one)."""
        vals = np.array([r.net_economic_value for r in self._arm_seasons(arm)])
        if vals.size == 0:
            return float("nan"), float("nan")
        return float(vals.mean()), float(vals.std())

    def gate_outcome(self, arm: str) -> dict[str, bool]:
        """Per soft class: does the arm clear the season-level feasibility gate?

        A season counts as violating a class if that class is exceeded at any step; the
        Clopper-Pearson upper bound on the violating-season rate must be <= epsilon.
        A season that CRASHED counts as violating every class (conservative: an
        un-runnable controller cannot claim feasibility on the season it failed).
        """
        seasons = self._arm_seasons(arm)
        failed = self.failed.get(arm, 0)
        n = len(seasons) + failed
        outcome: dict[str, bool] = {}
        for cls in self.soft_classes:
            k = failed + sum(1 for r in seasons if r.exceedances.get(cls, np.zeros(1)).max() > 0.0)
            outcome[cls] = passes_soft_gate(k, n, self.gate.epsilon, self.gate.delta)
        return outcome

    def gate_underpowered(self, arm: str) -> bool:
        """True when the arm's season count (completed + failed) is below the
        pre-registered ``n_seasons_min`` floor, so its gate outcome is off-protocol
        and must be reported as such (never silently)."""
        n = len(self._arm_seasons(arm)) + self.failed.get(arm, 0)
        return n < self.gate.n_seasons_min

    def safety(self, arm: str, cls: str) -> SafetyProfile:
        """Pooled per-class safety profile across the arm's seasons."""
        traces = [r.exceedances[cls] for r in self._arm_seasons(arm) if cls in r.exceedances]
        return safety_profile_from_exceedance(traces, step_hours=self.step_hours)

    def online_cost(self, arm: str) -> float:
        """Mean online (deployment) decision time per step [s] over the arm's seasons.

        Wall-clock, hence hardware-dependent: valid as a *relative* Pareto axis on one
        machine, but interpret the absolute value only with ``record`` hardware stamp
        (see :func:`hardware_stamp`).
        """
        return float(np.mean([r.online_cost_seconds for r in self._arm_seasons(arm)]))

    def solve_success(self, arm: str) -> float | None:
        """Mean MPC solve-success over the arm's seasons (deployment reliability).

        Fraction of NLP solves that converged or salvaged a usable iterate. ``None`` for
        solver-free arms (rule-based, RL), whose seasons carry no ``solve_success``.
        """
        vals = [r.solve_success for r in self._arm_seasons(arm) if r.solve_success is not None]
        return float(np.mean(vals)) if vals else None

    def pareto(
        self, incumbent: str = "incumbent", gate_passers_only: bool = True
    ) -> dict[str, Any]:
        """The (net value, online cost) Pareto comparison over gate-passing arms.

        Maximises net economic value and minimises online compute; min-max normalises each
        axis. The hypervolume is measured against the anti-ideal corner (worst on each axis),
        the only valid reference: the incumbent is typically cost-optimal and so cannot anchor
        a hypervolume (see ``metrics/pareto.py``). ``hypervolume_vs_incumbent`` reports the
        improvement over the deployed status quo, HV(front) - HV(incumbent), both against the
        anti-ideal reference (contract rows 8/17). The hypervolume mixes value and cost on one
        normalised scale, so treat it as a diagnostic, not a ranking input. The incumbent is
        always included even if it fails the gate. Returns the considered arms, the
        non-dominated set, and the two hypervolumes.
        """
        arms = [
            a for a in self.arms() if not gate_passers_only or all(self.gate_outcome(a).values())
        ]
        if incumbent in self.arms() and incumbent not in arms:
            arms = [*arms, incumbent]
        arms = sorted(arms)
        if len(arms) < _MIN_PARETO_ARMS or incumbent not in arms:
            return {
                "arms": arms,
                "non_dominated": arms,
                "hypervolume": 0.0,
                "hypervolume_vs_incumbent": 0.0,
                "reference": "anti_ideal",
            }
        # maximisation on both axes: net value up, online cost down (negated)
        points = np.array([[self.net_value_summary(a)[0], -self.online_cost(a)] for a in arms])
        norm, _lo, _hi = normalize_minmax(points)
        non_dominated = [arms[i] for i in pareto_front(norm)]
        # A valid hypervolume reference is the anti-ideal corner (worst on each axis); in
        # min-max-normalised coordinates that is the origin. The incumbent cannot anchor it:
        # it is typically cost-optimal, so nothing dominates it on cost and HV collapses to 0.
        reference = np.zeros(points.shape[1])
        inc = arms.index(incumbent)
        hv_front = float(hypervolume_2d(norm, reference))
        hv_incumbent = float(hypervolume_2d(norm[inc : inc + 1], reference))
        return {
            "arms": arms,
            "non_dominated": non_dominated,
            "hypervolume": hv_front,
            "hypervolume_vs_incumbent": hv_front - hv_incumbent,
            "reference": "anti_ideal",
        }


def run_benchmark(
    arm_factories: dict[str, ArmFactory],
    conditions: Sequence[Condition],
    seeds: Sequence[int],
    make_env: Callable[[Condition], Any],
    *,
    record: dict[str, Any],
    gate: GateConfig,
    soft_classes: tuple[str, ...] = ("humidity",),
    step_hours: float = 0.25,
) -> BenchmarkResult:  # pragma: no cover - needs a live env
    """Run every arm on the same grid of conditions and seeds; return a ``BenchmarkResult``.

    Args:
        arm_factories: ``name -> (env, condition, seed) -> Controller``. Each arm builds
            itself from the shared env (physics arms pull the nominal model from it).
        conditions: the generalisation / mismatch grid points.
        seeds: seeds per condition (multiple, per the reporting floor).
        make_env: builds a fresh env for a condition (one env per arm-condition-seed run).
        record: run metadata (config version, git commit, timestamp) for the audit trail.
    """
    seasons: dict[tuple[str, str, int], SeasonResult] = {}
    for arm_name, factory in arm_factories.items():
        for condition in conditions:
            for seed in seeds:
                env = make_env(condition)
                controller = factory(env, condition, seed)
                seasons[(arm_name, condition.label, seed)] = run_season_controller(
                    env, controller, seed
                )
    return BenchmarkResult(
        seasons=seasons,
        record=dict(record),
        gate=gate,
        soft_classes=soft_classes,
        step_hours=step_hours,
    )
