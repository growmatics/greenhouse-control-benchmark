"""Tests for Pareto dominance and 2-D hypervolume."""

import numpy as np

from greenlight_bench.metrics.pareto import (
    dominates,
    hypervolume_2d,
    normalize_minmax,
    pareto_front,
)


def test_dominance() -> None:
    assert dominates([2.0, 2.0], [1.0, 1.0]) is True
    assert dominates([2.0, 1.0], [1.0, 2.0]) is False
    assert dominates([1.0, 1.0], [1.0, 1.0]) is False  # equal is not dominating


def test_pareto_front_drops_dominated() -> None:
    pts = np.array([[2.0, 1.0], [1.0, 2.0], [1.0, 1.0]])
    front = pareto_front(pts)
    assert list(front) == [0, 1]


def test_hypervolume_known_value() -> None:
    # Union of (0,0)-(2,1) and (0,0)-(1,2) minus overlap (0,0)-(1,1) = 2 + 2 - 1 = 3.
    hv = hypervolume_2d(np.array([[2.0, 1.0], [1.0, 2.0]]), reference=np.array([0.0, 0.0]))
    assert hv == 3.0


def test_hypervolume_ignores_points_below_reference() -> None:
    hv = hypervolume_2d(np.array([[0.5, 0.5]]), reference=np.array([1.0, 1.0]))
    assert hv == 0.0


def test_normalize_minmax_constant_axis_is_zero() -> None:
    pts = np.array([[1.0, 5.0], [3.0, 5.0]])
    normalized, lo, hi = normalize_minmax(pts)
    assert np.allclose(normalized[:, 0], [0.0, 1.0])
    assert np.allclose(normalized[:, 1], [0.0, 0.0])  # constant axis -> zeros
    assert np.allclose(lo, [1.0, 5.0])
    assert np.allclose(hi, [3.0, 5.0])
