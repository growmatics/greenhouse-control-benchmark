"""Data-efficiency and generalisation estimators (contract rows 5, 6, 10, 11).

Data efficiency is read per method class from a performance-vs-budget curve;
generalisation is the drop from training to held-out net value. Results are
reported as distributions across seeds (bootstrap CIs), never single-run means.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# The simulator pins numpy < 2.0 (no np.trapezoid); fall back to np.trapz there.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

_MIN_CURVE_POINTS = 2  # a curve needs at least two budgets to integrate


def data_efficiency_auc(budgets: Sequence[float], performance: Sequence[float]) -> float:
    """Budget-normalized area under the performance-vs-budget curve (trapezoid).

    Higher means more performance per unit budget across the swept range.
    """
    b = np.asarray(budgets, dtype=float)
    p = np.asarray(performance, dtype=float)
    if b.shape != p.shape or b.ndim != 1:
        raise ValueError("budgets and performance must be matching 1-D sequences")
    if len(b) < _MIN_CURVE_POINTS:
        raise ValueError("need at least two budget points")
    order = np.argsort(b)
    b, p = b[order], p[order]
    span = b[-1] - b[0]
    if span <= 0:
        raise ValueError("budgets must span a positive range")
    return float(_trapezoid(p, b) / span)


def bootstrap_ci(
    samples: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    statistic: Callable[[NDArray[np.float64]], float] = np.mean,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and ``(1 - alpha)`` bootstrap CI for a statistic.

    Returns ``(estimate, lower, upper)``. Seeded for reproducibility.
    """
    s = np.asarray(samples, dtype=float)
    if s.ndim != 1 or len(s) == 0:
        raise ValueError("samples must be a non-empty 1-D sequence")
    rng = np.random.default_rng(seed)
    boots = np.array([statistic(rng.choice(s, size=len(s), replace=True)) for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(statistic(s)), float(lo), float(hi)


def generalisation_gap(train_value: float, heldout_value: float) -> float:
    """Generalisation gap in net economic value: ``train - held-out``."""
    return float(train_value) - float(heldout_value)


@dataclass(frozen=True)
class DataEfficiencyCurve:
    """One learned arm's held-out performance vs training-data budget (the headline plot).

    Reported *per method class*: RL-steps and identification-samples are incommensurable,
    so curves are never placed on a joint axis (contract row 5). Built from a budget sweep
    such as ``examples/ppo_sweep.py``.
    """

    arm: str
    budgets: tuple[float, ...]
    net_values: tuple[float, ...]  # held-out net economic value at each budget

    def auc(self) -> float:
        """Budget-normalized area under the curve (performance per unit budget)."""
        return data_efficiency_auc(self.budgets, self.net_values)

    @classmethod
    def from_points(cls, arm: str, points: Sequence[tuple[float, float]]) -> DataEfficiencyCurve:
        """Build from ``[(budget, net_value), ...]`` (e.g. ``ppo_sweep``'s JSON output)."""
        budgets = tuple(float(b) for b, _v in points)
        net_values = tuple(float(v) for _b, v in points)
        return cls(arm=arm, budgets=budgets, net_values=net_values)
