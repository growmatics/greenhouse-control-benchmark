"""Tests for the season-level feasibility gate."""

import pytest

from greenlight_bench.metrics.gate import (
    clopper_pearson_upper,
    min_seasons,
    passes_hard_gate,
    passes_soft_gate,
)


def test_min_seasons_matches_rule_of_three() -> None:
    # eps = delta = 0.05  ->  ceil(ln 0.05 / ln 0.95) = 59
    assert min_seasons(0.05, 0.05) == 59


def test_clopper_pearson_zero_violations_boundary() -> None:
    # At n = 59, zero violations clears eps = 0.05; at n = 58 it does not.
    assert clopper_pearson_upper(0, 59, 0.05) <= 0.05
    assert clopper_pearson_upper(0, 58, 0.05) > 0.05


def test_clopper_pearson_all_violations_is_one() -> None:
    assert clopper_pearson_upper(5, 5, 0.05) == 1.0


def test_soft_gate_pass_and_fail() -> None:
    assert passes_soft_gate(0, 59, 0.05, 0.05) is True
    assert passes_soft_gate(0, 58, 0.05, 0.05) is False
    assert passes_soft_gate(10, 60, 0.05, 0.05) is False  # 10/60 way above 5%


def test_hard_gate_is_zero_tolerance() -> None:
    assert passes_hard_gate(0) is True
    assert passes_hard_gate(1) is False


@pytest.mark.parametrize(
    ("k", "n", "delta"),
    [(-1, 10, 0.05), (11, 10, 0.05), (0, 10, 0.0), (0, 10, 1.0)],
)
def test_clopper_pearson_rejects_bad_input(k: int, n: int, delta: float) -> None:
    with pytest.raises(ValueError):
        clopper_pearson_upper(k, n, delta)
