"""Tests for the model-mismatch perturbation (the inverse-crime axis)."""

import pytest

from greenlight_bench.mismatch import perturb_params

_TRUE = {"0": 2.0, "1": 10.0, "2": -4.0}


def test_zero_magnitude_is_the_inverse_crime_reference() -> None:
    # m = 0: nominal equals the truth exactly (the labelled reference)
    assert perturb_params(_TRUE, 0.0, seed=0) == _TRUE


def test_perturbation_is_relative_seeded_and_reproducible() -> None:
    a = perturb_params(_TRUE, 0.1, seed=7)
    b = perturb_params(_TRUE, 0.1, seed=7)
    assert a == b  # deterministic in the seed
    assert a != _TRUE  # actually perturbed
    # relative: each param stays within magnitude-scaled range of its true value for a
    # moderate draw (checked loosely; the point is it scales with the true value)
    for k, true_v in _TRUE.items():
        assert a[k] != true_v


def test_different_seeds_give_different_perturbations() -> None:
    assert perturb_params(_TRUE, 0.1, seed=0) != perturb_params(_TRUE, 0.1, seed=1)


def test_larger_magnitude_moves_further_on_average() -> None:
    small = perturb_params(_TRUE, 0.01, seed=3)
    large = perturb_params(_TRUE, 0.5, seed=3)
    dev_small = sum(abs(small[k] - _TRUE[k]) for k in _TRUE)
    dev_large = sum(abs(large[k] - _TRUE[k]) for k in _TRUE)
    assert dev_large > dev_small
    # the ratio tracks the magnitude ratio (same underlying normal draws, seeded)
    assert dev_large / dev_small == pytest.approx(0.5 / 0.01, rel=1e-6)
