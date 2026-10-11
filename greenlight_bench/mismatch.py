"""Model-mismatch perturbation for the inverse-crime defence (contract rows 15-19).

The physics arms (3, 4) never receive the simulator's ground-truth parameters. Under a
mismatch magnitude ``m``, a physics arm's NOMINAL parameters are the plant's true
parameters perturbed by ``m``; the plant is run with the true parameters. At ``m = 0``
the nominal equals the truth -- the labelled inverse-crime reference, an *upper bound* on
the physics advantage, not the honest reading (``m > 0`` is). Every benchmark result is
read as a function of ``m``; the Pareto front is reported per mismatch level.

This is a simple relative Gaussian perturbation of the parameter vector; it maps onto
GreenLight's parameter-sampling API (relative mode) and can be narrowed to a named subset
of physically-meaningful parameters when the axis is finalised.
"""

from __future__ import annotations

import numpy as np


def perturb_params(true_params: dict[str, float], magnitude: float, seed: int) -> dict[str, float]:
    """Nominal params = true * (1 + magnitude * N(0, 1)), seeded and reproducible.

    ``magnitude == 0`` returns the true params unchanged (the inverse-crime reference).
    The perturbation is deterministic in ``seed`` so a mismatch level is reproducible.
    """
    if magnitude == 0.0:
        return dict(true_params)
    rng = np.random.default_rng(seed)
    return {
        key: float(value) * (1.0 + magnitude * float(rng.standard_normal()))
        for key, value in true_params.items()
    }
