"""Arm 4: run the physics-informed RL-in-MPC (trajectory selection) for one season.

Loads a saved guiding PPO policy (from ``ppo_sweep`` with ``PPO_SAVE_PATH``), builds the
economic MPC (arm 3) with an L1 terminal cost toward the RL rollout's terminal state, and
runs the trajectory-selection controller through the uniform harness. Reports net economic
value, the RL-selection rate (how often the RL rollout out-scored the MPC), and the
humidity safety profile.

    GLG2_CONFIG_DIR=.../configs/envs GLG2_WEATHER_DIR=.../bench_weather \
    PPO_SAVE_PATH=.../ppo_policy python examples/run_arm4.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from greenlight_bench.config import load_config
from greenlight_bench.controllers.base import NominalParameters
from greenlight_bench.controllers.mpc import PhysicsOnlyMPC
from greenlight_bench.controllers.rl_mpc import PhysicsInformedMPC, load_ppo_policy
from greenlight_bench.envs.glg2 import make_tomato_env, run_season_controller

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "benchmark.yaml"


def main() -> None:
    cfg = load_config(str(_CONFIG))
    config_dir = os.environ.get("GLG2_CONFIG_DIR")
    weather_dir = os.environ.get("GLG2_WEATHER_DIR")
    policy_path = os.environ["PPO_SAVE_PATH"]
    season_days = int(os.environ.get("BENCH_SEASON_DAYS", "3"))  # short by default for validation
    terminal_weight = float(os.environ.get("ARM4_TERMINAL_WEIGHT", "0.01"))
    location = cfg.generalisation.location
    year = cfg.generalisation.heldout_years[0]  # a held-out year

    def make_env() -> Any:
        return make_tomato_env(
            config_dir,
            weather_dir,
            overrides={
                "location": location,
                "start_train_year": year,
                "end_train_year": year,
                "start_train_day": 59,
                "end_train_day": 59,
                "season_length": season_days,
            },
        )

    plant = make_env()
    rollout = make_env()
    plant.reset(seed=0)
    params = NominalParameters(
        {str(i): float(v) for i, v in enumerate(np.array(plant.p).flatten())}
    )
    bounds = {
        "co2_ppm": tuple(cfg.raw["constraints"]["co2_ppm"]),
        "air_temp_c": tuple(cfg.raw["constraints"]["air_temp_c"]),
        "rh_percent": tuple(cfg.raw["constraints"]["rh_percent"]),
    }
    mpc = PhysicsOnlyMPC(
        params,
        horizon=4,
        dt=900.0,
        objective="economic",
        economic=cfg.economic,
        climate_bounds=bounds,
        terminal_weight=terminal_weight,
    )
    policy = load_ppo_policy(policy_path)
    arm4 = PhysicsInformedMPC(params, policy, rollout, mpc)

    result = run_season_controller(plant, arm4, seed=0)
    print(f"Arm 4 (RL-in-MPC) - {location} {year}, {season_days}-day season:")
    print(f"  net economic value = {result.net_economic_value:+.4f} EUR/m2")
    print(f"  online cost        = {result.online_cost_seconds * 1e3:.1f} ms/step")
    print(
        f"  RL-selection rate  = {arm4.rl_selection_rate:.2f} (fraction of steps RL out-scored MPC)"
    )
    st = mpc.solve_stats
    tot = sum(st.values()) or 1
    print(
        f"  MPC solve-success  = {(st['clean'] + st['salvage']) / tot:.2f} "
        f"(clean {st['clean']}, salvage {st['salvage']}, hold {st['hold']})"
    )
    rh = result.exceedances["humidity"]
    print(f"  humidity exceedance: steps>0 {int((rh > 0).sum())}, worst {rh.max():.2f}")


if __name__ == "__main__":
    main()
