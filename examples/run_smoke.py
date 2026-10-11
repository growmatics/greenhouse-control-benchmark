"""Run the unified benchmark: every available arm on ONE shared config, comparably.

This is the public "run our benchmark" entry point. It runs each arm across a grid of
conditions and seeds through ``run_benchmark`` and prints the per-arm net economic value,
the season-level feasibility gate, and the run record (config version + git commit +
grid) that makes every number traceable.

By default it runs a SMOKE grid (a short season, two weather years, one seed) so it
finishes quickly and proves the arms are cross-comparable. The full benchmark grid
(60-day seasons x the weather split x mismatch levels x multiple seeds) is the same call
with the full conditions/seeds and the economic MPC objective.

    GLG2_CONFIG_DIR=.../configs/envs GLG2_WEATHER_DIR=.../weather \
    BENCH_SEASON_DAYS=2 python examples/run_benchmark.py
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from greenlight_bench.benchmark import Condition, hardware_stamp, run_benchmark
from greenlight_bench.config import load_config
from greenlight_bench.controllers.base import NominalParameters
from greenlight_bench.controllers.mpc import PhysicsOnlyMPC
from greenlight_bench.controllers.rule_based import RuleBasedIncumbent
from greenlight_bench.envs.glg2 import default_glg2_paths, make_tomato_env
from greenlight_bench.mismatch import perturb_params

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "benchmark.yaml"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    cfg = load_config(str(_CONFIG))
    config_dir = os.environ.get("GLG2_CONFIG_DIR")
    weather_dir = os.environ.get("GLG2_WEATHER_DIR")
    config_path = Path(config_dir) if config_dir else default_glg2_paths()[0]
    season_days = int(os.environ.get("BENCH_SEASON_DAYS", "2"))  # SMOKE default

    def make_env(condition: Condition) -> Any:
        year = int(condition.env_overrides["year"])
        return make_tomato_env(
            config_dir,
            weather_dir,
            overrides={
                "location": "Amsterdam",
                "start_train_year": year,
                "end_train_year": year,
                "season_length": season_days,
            },
        )

    rb_params = yaml.safe_load((config_path.parent / "agents" / "rule_based.yml").read_text())[
        "TomatoEnv"
    ]

    def build_incumbent(env: Any, condition: Condition, seed: int) -> Any:
        return RuleBasedIncumbent(rb_params)

    def build_physics_mpc(env: Any, condition: Condition, seed: int) -> Any:
        # inverse-crime boundary: the physics arm gets a NOMINAL model = the plant's true
        # parameters perturbed by the condition's mismatch magnitude (0 = the reference).
        true_params = {str(i): float(v) for i, v in enumerate(np.array(env.p).flatten())}
        nominal = perturb_params(true_params, condition.mismatch, seed)
        return PhysicsOnlyMPC(NominalParameters(nominal), horizon=4, dt=900.0)

    arms = {"incumbent": build_incumbent, "physics_mpc": build_physics_mpc}
    # a small generalisation x mismatch grid for the smoke (m=0 reference + m=0.02)
    conditions = [
        Condition("2010/m0.00", {"year": 2010}, mismatch=0.0),
        Condition("2010/m0.02", {"year": 2010}, mismatch=0.02),
        Condition("2015/m0.00", {"year": 2015}, mismatch=0.0),
    ]
    record = {
        "config_version": cfg.testbed["release"],
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "season_days": season_days,
        "smoke": season_days < int(cfg.raw["episode"]["crop_days"]),  # < full 60-day season
        "hardware": hardware_stamp(),  # so the online-cost axis is interpretable/reproducible
    }

    result = run_benchmark(
        arms, conditions, seeds=[0], make_env=make_env, record=record, gate=cfg.gate
    )

    print("=== unified benchmark run ===")
    print(f"record: {result.record}")
    print(f"grid: {len(conditions)} conditions x 1 seed; arms: {result.arms()}")
    for arm in result.arms():
        mean, std = result.net_value_summary(arm)
        gate = result.gate_outcome(arm)
        rh = result.safety(arm, "humidity")
        ss = result.solve_success(arm)
        ss_str = f"{ss:.2f}" if ss is not None else "  - "
        print(
            f"  {arm:12s}  net {mean:+7.3f} +/- {std:5.3f} EUR/m2   "
            f"online {result.online_cost(arm) * 1e3:7.1f} ms/step   "
            f"gate(humidity)={gate['humidity']}   rh-incidence {rh.incidence:.2f}   "
            f"solve-success {ss_str}"
        )
    par = result.pareto(incumbent="incumbent", gate_passers_only=False)
    print(
        f"pareto (net value vs online cost): non-dominated {par['non_dominated']}, "
        f"hypervolume over incumbent {par['hypervolume']:.3f}"
    )


if __name__ == "__main__":
    main()
