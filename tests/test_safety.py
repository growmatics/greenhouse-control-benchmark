"""Tests for the multi-measure safety profile."""

import numpy as np
import pytest

from greenlight_bench.metrics.safety import safety_profile


def test_profile_on_a_known_two_season_case() -> None:
    # Bound [0, 10], step = 1 h. Season A breaches at 2 steps by 1 and 2;
    # season B never breaches.
    season_a = np.array([11.0, 12.0, 5.0, 5.0])
    season_b = np.array([1.0, 2.0, 3.0, 4.0])
    prof = safety_profile([season_a, season_b], lower=0.0, upper=10.0, step_hours=1.0)

    assert prof.incidence == 0.5  # one of two seasons violates
    assert prof.season_violation_prob == 0.5
    assert prof.mean_duration_h == 1.0  # (2 h + 0 h) / 2
    assert prof.degree_hours == 1.5  # ((1 + 2) + 0) / 2
    assert prof.worst_excursion == 2.0


def test_profile_all_clean_is_zero() -> None:
    prof = safety_profile([np.zeros(5)], lower=-1.0, upper=1.0, step_hours=0.25)
    assert prof.incidence == 0.0
    assert prof.degree_hours == 0.0
    assert prof.worst_excursion == 0.0


@pytest.mark.parametrize(
    ("lower", "upper", "step"),
    [(1.0, 0.0, 1.0), (0.0, 1.0, 0.0)],
)
def test_profile_rejects_bad_bounds(lower: float, upper: float, step: float) -> None:
    with pytest.raises(ValueError):
        safety_profile([np.zeros(3)], lower=lower, upper=upper, step_hours=step)


def test_profile_needs_a_season() -> None:
    with pytest.raises(ValueError):
        safety_profile([], lower=0.0, upper=1.0, step_hours=1.0)
