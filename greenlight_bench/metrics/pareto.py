"""Pareto dominance and 2-D hypervolume (contract rows 8, 17).

Both objectives are treated as "higher is better", so pass online cost through a
higher-is-better transform (e.g. negative solve time) before calling these.

The hypervolume reference must be the *anti-ideal* corner (the worst value on each
axis), because a valid reference has to be dominated by every point it measures.
The deployed incumbent cannot serve as the reference: a rule-based incumbent is
typically the cheapest arm to run, so it is Pareto-optimal on the cost axis and no
other arm dominates it there, which collapses the hypervolume to zero. Measure
"improvement over the status quo" instead as HV(front) - HV(incumbent alone), both
against the anti-ideal reference (see ``benchmark.py``).

Caveat: the bi-objective hypervolume mixes net value and online cost on one
normalised scale, so its value depends on the cost transform and the axis
normalisation; treat it as a diagnostic, not a ranking metric.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_MATRIX_NDIM = 2  # points are stored as a 2-D array (rows = points)
_N_OBJECTIVES = 2  # this module implements the bi-objective (net value, cost) case


def dominates(a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
    """True if ``a`` Pareto-dominates ``b`` (maximization on every axis)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(np.all(a >= b) and np.any(a > b))


def pareto_front(points: NDArray[np.float64]) -> NDArray[np.intp]:
    """Indices of the non-dominated points (maximization)."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != _MATRIX_NDIM:
        raise ValueError("points must be a 2-D array")
    keep = [
        i
        for i, p in enumerate(pts)
        if not any(dominates(q, p) for j, q in enumerate(pts) if j != i)
    ]
    return np.asarray(sorted(keep), dtype=np.intp)


def hypervolume_2d(points: NDArray[np.float64], reference: NDArray[np.float64]) -> float:
    """Dominated hypervolume above-right of ``reference`` (2-D maximization).

    Points not dominating the reference contribute nothing.
    """
    ref = np.asarray(reference, dtype=float)
    pts = np.asarray(points, dtype=float)
    if pts.ndim != _MATRIX_NDIM or pts.shape[1] != _N_OBJECTIVES or ref.shape != (_N_OBJECTIVES,):
        raise ValueError("need 2-D points with 2 columns and a length-2 reference")
    pts = pts[np.all(pts >= ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    front = pts[pareto_front(pts)]
    front = front[np.argsort(front[:, 1])]  # y ascending => x descending on a front
    hv = 0.0
    y_prev = ref[1]
    for x, y in front:
        hv += (x - ref[0]) * (y - y_prev)
        y_prev = y
    return float(hv)


def normalize_minmax(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Min-max normalize each axis to ``[0, 1]``. Returns ``(normalized, lo, hi)``.

    A constant axis maps to zeros (documented, not a divide-by-zero).
    """
    pts = np.asarray(points, dtype=float)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    normalized = np.where(hi > lo, (pts - lo) / span, 0.0)
    return normalized, lo, hi
