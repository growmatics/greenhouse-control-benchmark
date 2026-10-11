"""PPO data-efficiency sweep (SUPERSEDED 2026-07-16 by ``train_policies.py``: this early driver
predates the recipe fix and hardcodes a pre-benchmark checkpoint ladder [50k..300k] that disagrees
with the canonical ``data_efficiency.ppo_budgets``; kept only as the contract-row-5 record).

Draws the first real points of the black-box arm's data-efficiency curve
(contract row 5). Trains continuously and evaluates the policy on the held-out
years at each budget, so N points cost one training run. Requires gl_gym +
stable-baselines3 + the fetched weather.

    GLG2_CONFIG_DIR=.../gl_gym/configs/envs GLG2_WEATHER_DIR=.../bench_weather \
    python examples/ppo_sweep.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from greenlight_bench.config import load_config
from greenlight_bench.envs.glg2 import make_tomato_env, run_season

CHECKPOINTS = [50_000, 100_000, 200_000, 300_000]


def _eval_env(config_dir: str, weather_dir: str, location: str, year: int) -> Any:
    return make_tomato_env(
        config_dir,
        weather_dir,
        overrides={"location": location, "training": False},
        spec_overrides={
            "eval_options": {"eval_days": [59], "eval_years": [year], "location": location}
        },
    )


def _held_out_net(
    model: PPO, config_dir: str, weather_dir: str, location: str, years: tuple[int, ...]
) -> float:
    def policy(obs: np.ndarray) -> np.ndarray:
        action, _state = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=float)

    nets = [
        run_season(
            _eval_env(config_dir, weather_dir, location, int(y)), policy, seed=0
        ).net_economic_value
        for y in years
    ]
    return float(np.mean(nets))


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "benchmark.yaml")
    location = cfg.generalisation.location
    train_years = cfg.generalisation.train_years
    heldout = cfg.generalisation.heldout_years
    config_dir = os.environ["GLG2_CONFIG_DIR"]
    weather_dir = os.environ["GLG2_WEATHER_DIR"]

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
    model = PPO(
        "MlpPolicy", train_env, n_steps=1024, batch_size=256, seed=0, device="cpu", verbose=0
    )

    curve: list[tuple[int, float]] = []
    trained = 0
    for budget in CHECKPOINTS:
        model.learn(total_timesteps=budget - trained, reset_num_timesteps=(trained == 0))
        trained = budget
        net = _held_out_net(model, config_dir, weather_dir, location, heldout)
        curve.append((budget, net))
        print(f"budget {budget:>7}: held-out net economic value {net:8.2f}", flush=True)

    print("PPO_DATA_EFFICIENCY_CURVE " + json.dumps(curve))

    # save the trained policy so arm 4 (RL-in-MPC) can use it as its guiding policy
    save_path = os.environ.get("PPO_SAVE_PATH")
    if save_path:
        model.save(save_path)
        print(f"saved PPO policy to {save_path}")


if __name__ == "__main__":
    main()
