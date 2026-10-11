"""Season-level feasibility gate (conformance contract rows 7, 7b).

The trial unit is the season, not the timestep. Hard equipment/safety classes
are zero-tolerance. Soft crop-health classes pass if the one-sided ``1 - delta``
Clopper-Pearson upper bound on the season-level violation rate is ``<= epsilon``.
The values of ``epsilon``, ``delta`` and the minimum season count live in
``configs/benchmark.yaml``; this module only implements the maths.
"""

from __future__ import annotations

import math

from scipy.stats import beta


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """One-sided ``1 - delta`` Clopper-Pearson upper bound on a Bernoulli rate.

    Args:
        k: number of seasons with a violation (``0 <= k <= n``).
        n: number of seasons evaluated.
        delta: significance level in ``(0, 1)``; confidence is ``1 - delta``.

    Returns:
        The upper confidence bound on the true violation rate, in ``[0, 1]``.
    """
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if k == n:
        return 1.0
    return float(beta.ppf(1.0 - delta, k + 1, n - k))


def passes_soft_gate(k: int, n: int, epsilon: float, delta: float) -> bool:
    """True if a soft crop-health class clears the statistical gate."""
    if not 0.0 < epsilon < 1.0:
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
    return clopper_pearson_upper(k, n, delta) <= epsilon


def passes_hard_gate(violation_seasons: int) -> bool:
    """Hard equipment/safety classes are zero-tolerance: any violation fails."""
    if violation_seasons < 0:
        raise ValueError("violation_seasons must be non-negative")
    return violation_seasons == 0


def min_seasons(epsilon: float, delta: float) -> int:
    """Fewest seasons for a clean (zero-violation) pass at ``(epsilon, delta)``.

    Solves ``(1 - epsilon) ** n <= delta`` for the smallest integer ``n`` (the
    "rule of three", ``n ~ 3 / epsilon``).
    """
    if not 0.0 < epsilon < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("epsilon and delta must be in (0, 1)")
    return math.ceil(math.log(delta) / math.log(1.0 - epsilon))
