"""Tests for the benchmark statistical helpers (pure functions, synthetic data)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from stats import _cliffs_delta, holm, paired_net


def test_cliffs_delta_bounds() -> None:
    assert _cliffs_delta(np.array([1.0, 2.0, 3.0])) == 1.0  # all positive
    assert _cliffs_delta(np.array([-1.0, -2.0])) == -1.0  # all negative
    assert _cliffs_delta(np.array([1.0, -1.0])) == 0.0  # balanced


def test_holm_is_monotone_and_bounded() -> None:
    adj = holm([0.01, 0.04, 0.03])
    assert all(0.0 <= a <= 1.0 for a in adj)
    # Holm never makes a p smaller than raw * (fewest remaining); smallest raw gets x m
    assert adj[0] >= 0.01 * 3 - 1e-9


def test_paired_net_matches_conditions_and_signs() -> None:
    # arm a beats arm b by +1 on every shared (year, mismatch, seed)
    rows = []
    for yr in ("2022", "2023"):
        for seed in range(3):
            rows.append({"arm": "a", "label": f"{yr}/m0.0", "seed": seed, "net": 5.0})
            rows.append({"arm": "b", "label": f"{yr}/m0.0", "seed": seed, "net": 4.0})
    res = paired_net(rows, "a", "b")
    assert res is not None
    assert res["n"] == 6
    assert res["median_diff"] == 1.0
    assert res["cliffs_delta"] == 1.0  # a wins every pair
    assert res["p_raw"] < 0.05


def test_paired_net_returns_none_when_too_few_pairs() -> None:
    rows = [
        {"arm": "a", "label": "2022/m0.0", "seed": 0, "net": 1.0},
        {"arm": "b", "label": "2022/m0.0", "seed": 0, "net": 0.0},
    ]
    assert paired_net(rows, "a", "b") is None  # only one matched pair
