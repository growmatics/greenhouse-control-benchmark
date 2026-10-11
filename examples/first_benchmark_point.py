"""First benchmark data point: rule-based incumbent vs a short-trained PPO.

Trains a small PPO on the pinned TRAIN weather years, then evaluates it and
gl_gym's rule-based incumbent (Het Nieuwe Telen setpoints) on the HELD-OUT
years, scoring each by net economic value and the per-class safety profile.

This is a plumbing / first-point demo, NOT a tuned result: the PPO budget is
tiny and held-out N is small, so the numbers show the machinery works, not who
wins. Requires gl_gym + stable-baselines3 + the fetched weather.

    GLG2_CONFIG_DIR=.../gl_gym/configs/envs \
    GLG2_AGENTS_DIR=.../gl_gym/configs/agents \
    GLG2_WEATHER_DIR=.../bench_weather \
    PPO_BUDGET=100000 python examples/first_benchmark_point.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from stable_baselines3 import PPO

from greenlight_bench.config import load_config
from greenlight_bench.envs.glg2 import SeasonResult, make_tomato_env, season_result_from_infos
from greenlight_bench.metrics.safety import safety_profile_from_exceedance

STEP_HOURS = 900.0 / 3600.0
PPO_BUDGET = int(os.environ.get("PPO_BUDGET", "100000"))


def _eval_env(config_dir: str, weather_dir: str, location: str, year: int) -> Any:
    return make_tomato_env(
        config_dir,
        weather_dir,
        overrides={"location": location, "training": False},
        spec_overrides={
            "eval_options": {"eval_days": [59], "eval_years": [year], "location": location}
        },
    )


def _run_incumbent(env: Any, controller: Any) -> SeasonResult:
    env.reset(seed=0)
    infos, done = [], False
    while not done:
        control = controller.predict(env.x, env.weather_data[env.timestep], env)
        _obs, _r, terminated, truncated, info = env.step_raw_control(control)
        infos.append(info)
        done = bool(terminated or truncated)
    return season_result_from_infos(infos)


def _run_policy(env: Any, model: PPO) -> SeasonResult:
    obs, _info = env.reset(seed=0)
    infos, done = [], False
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, _r, terminated, truncated, info = env.step(action)
        infos.append(info)
        done = bool(terminated or truncated)
    return season_result_from_infos(infos)


def _report(name: str, results: list[SeasonResult]) -> None:
    net = float(np.mean([r.net_economic_value for r in results]))
    print(
        f"\n{name}  |  held-out net economic value (mean over {len(results)} seasons): {net:,.1f}"
    )
    for cls in results[0].exceedances:
        prof = safety_profile_from_exceedance([r.exceedances[cls] for r in results], STEP_HOURS)
        print(
            f"   {cls:>11}: season-violation-prob {prof.season_violation_prob:.2f}"
            f"  degree-hours {prof.degree_hours:10.1f}  worst {prof.worst_excursion:.2f}"
        )


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "benchmark.yaml")
    location = cfg.generalisation.location
    train_years = cfg.generalisation.train_years
    heldout = cfg.generalisation.heldout_years
    config_dir = os.environ["GLG2_CONFIG_DIR"]
    weather_dir = os.environ["GLG2_WEATHER_DIR"]
    agents_dir = os.environ["GLG2_AGENTS_DIR"]

    from gl_gym.environments.baseline import RuleBasedController

    rb_params = yaml.safe_load(Path(agents_dir, "rule_based.yml").read_text())["TomatoEnv"]
    incumbent = RuleBasedController(**rb_params)

    train_env = make_tomato_env(
        config_dir,
        weather_dir,
        overrides={
            "location": location,
            "start_train_year": min(train_years),
            "end_train_year": max(train_years),
            "start_train_day": 59,
            "end_train_day": 59,
            "training": True,
        },
    )
    print(
        f"training PPO {PPO_BUDGET} steps on {location} {min(train_years)}-{max(train_years)} ..."
    )
    model = PPO(
        "MlpPolicy", train_env, n_steps=1024, batch_size=256, seed=0, device="cpu", verbose=0
    )
    model.learn(total_timesteps=PPO_BUDGET, progress_bar=False)

    inc = [
        _run_incumbent(_eval_env(config_dir, weather_dir, location, int(y)), incumbent)
        for y in heldout
    ]
    ppo = [
        _run_policy(_eval_env(config_dir, weather_dir, location, int(y)), model) for y in heldout
    ]

    print(f"\n=== First benchmark point (held-out years {list(heldout)}) ===")
    _report("rule-based incumbent", inc)
    _report(f"PPO ({PPO_BUDGET} steps, tiny)", ppo)


if __name__ == "__main__":
    main()
