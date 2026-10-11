"""Four-way physics-informed control benchmark on GreenLight-Gym2 (the benchmark paper).

Public API is intentionally small: the config loader, the Controller interface,
and the pure metric functions. Constants live in ``configs/benchmark.yaml`` (the
single source of truth); nothing here hardcodes a benchmark constant.
"""

__version__ = "0.0.0"
