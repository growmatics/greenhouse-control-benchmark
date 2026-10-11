"""Run one TomatoEnv season and compute the benchmark metrics on real output.

Requires ``gl_gym`` installed. Demonstrates the pipeline end to end:
environment -> net economic value + per-class safety profile. Not a benchmark
result, a plumbing check on a single random-policy season.

    # config/weather live in the source checkout, not the wheel:
    GLG2_CONFIG_DIR=.../gl_gym/configs/envs \
    GLG2_WEATHER_DIR=.../gl_gym/environments/weather \
    python examples/run_season_demo.py
"""

from __future__ import annotations

import os

import numpy as np

from greenlight_bench.envs.glg2 import make_tomato_env, run_season
from greenlight_bench.metrics.safety import safety_profile_from_exceedance

STEP_HOURS = 900.0 / 3600.0  # 900 s timestep


def main() -> None:
    env = make_tomato_env(
        config_dir=os.environ.get("GLG2_CONFIG_DIR"),
        weather_dir=os.environ.get("GLG2_WEATHER_DIR"),
    )

    def random_policy(_obs: np.ndarray) -> np.ndarray:
        return np.asarray(env.action_space.sample(), dtype=float)

    result = run_season(env, random_policy, seed=0)

    print(f"net economic value (season): {result.net_economic_value:,.2f}")
    for cls, exc in result.exceedances.items():
        prof = safety_profile_from_exceedance([exc], step_hours=STEP_HOURS)
        print(
            f"  {cls:>11}: incidence={prof.incidence:.0f} "
            f"duration={prof.mean_duration_h:7.1f} h "
            f"degree-hours={prof.degree_hours:9.2f} "
            f"worst={prof.worst_excursion:.3f}"
        )


if __name__ == "__main__":
    main()
