"""Tests for the data-efficiency and generalisation estimators."""

import numpy as np
import pytest

from greenlight_bench.metrics.efficiency import (
    DataEfficiencyCurve,
    bootstrap_ci,
    data_efficiency_auc,
    generalisation_gap,
)


def test_auc_of_a_straight_line() -> None:
    # Line from (0,0) to (1,1): area 0.5, normalized by span 1 -> 0.5.
    assert data_efficiency_auc([0.0, 1.0], [0.0, 1.0]) == pytest.approx(0.5)


def test_auc_is_order_independent() -> None:
    a = data_efficiency_auc([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    b = data_efficiency_auc([2.0, 0.0, 1.0], [3.0, 1.0, 2.0])
    assert a == pytest.approx(b)


def test_auc_needs_two_points() -> None:
    with pytest.raises(ValueError):
        data_efficiency_auc([1.0], [1.0])


def test_generalisation_gap() -> None:
    assert generalisation_gap(10.0, 7.0) == 3.0


def test_bootstrap_ci_is_seeded_and_brackets_the_mean() -> None:
    samples = list(np.arange(0.0, 100.0))
    est, lo, hi = bootstrap_ci(samples, n_boot=2000, seed=0)
    assert est == pytest.approx(np.mean(samples))
    assert lo <= est <= hi
    # Deterministic under a fixed seed.
    est2, lo2, hi2 = bootstrap_ci(samples, n_boot=2000, seed=0)
    assert (est, lo, hi) == (est2, lo2, hi2)


def test_data_efficiency_curve_from_ppo_sweep_points() -> None:
    # the shape of ppo_sweep's JSON output: [(budget, held-out net), ...]
    curve = DataEfficiencyCurve.from_points("black_box_rl", [(50000, -16.1), (100000, -16.8)])
    assert curve.arm == "black_box_rl"
    assert curve.budgets == (50000.0, 100000.0)
    assert curve.net_values == (-16.1, -16.8)
    # AUC is the budget-normalized mean here (two points): (-16.1 + -16.8)/2
    assert curve.auc() == pytest.approx((-16.1 + -16.8) / 2.0)
