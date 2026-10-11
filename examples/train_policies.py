"""benchmark step 1: train the PPO policies, mirroring the GLG2 reference recipe.

The first full run (``RUN-REPORT-mac-M4-2026-07-16``) produced a FLAT data-efficiency
curve: PPO with SB3 defaults on the raw env does not learn on GreenLight. The reference
pipeline (van Laatum's GLG2 ``ExperimentManager``, ``vanlaatum2026rlmpc``) differs in
two load-bearing ways this driver now mirrors:

1. **Normalization.** Training runs on ``SubprocVecEnv(n_envs)`` wrapped in
   ``VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10)``. GreenLight
   observations span orders of magnitude (ppm CO2 vs W/m2 vs degrees C), which breaks a
   raw-obs MLP. The statistics are saved next to every checkpoint and applied again at
   deployment (``load_ppo_policy``); evaluating the net on raw observations was the
   silent failure mode.
2. **Hyperparameters.** Bayes-optimised values from the pinned GLG2 checkout's
   ``gl_gym/configs/agents/ppo.yml`` (lr 2e-5, tuned gamma/GAE/entropy, large SiLU
   nets), read AT RUNTIME from ``$GLG2_CONFIG_DIR/../agents/ppo.yml`` so the provenance
   is the reference file itself (the same pattern ``run_benchmark`` uses for
   ``rule_based.yml``). Their VecMonitor/W&B logging is intentionally not mirrored.

For each seed this trains ONE continuous PPO run and saves, at every budget in
``data_efficiency.ppo_budgets`` (env steps): the policy (``ppo_s{seed}_b{budget}.zip``)
plus its VecNormalize statistics (``..._vecnorm.pkl``), then evaluates the saved PAIR on
the held-out years through ``load_ppo_policy`` (round-tripping the exact artifacts the
benchmark loads). SB3 collects whole rollouts, so a checkpoint may overshoot its nominal
budget by up to ``n_envs * n_steps``; the actual step count is recorded in the summary.

Outputs (see the run reports in results/benchmark/ for the storage contract):
- ``checkpoints/ppo_s{seed}_b{budget}.zip`` + ``..._vecnorm.pkl`` -- bulk storage.
- ``<out>/data_efficiency_s{seed}.json`` -- per-budget held-out net (KB, committed).

CPU-bound; each seed uses ``n_envs`` (reference: 8) worker processes, so shard
``--seeds`` across terminals only if cores allow ``shards x n_envs``.

    GLG2_CONFIG_DIR=.../gl_gym/configs/envs GLG2_WEATHER_DIR=.../bench_weather \
    python examples/train_policies.py --checkpoint-dir checkpoints --out results/benchmark
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from greenlight_bench.config import load_config
from greenlight_bench.controllers.rl_mpc import load_ppo_policy, vecnorm_stats_path
from greenlight_bench.envs.glg2 import make_tomato_env, run_season

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "benchmark.yaml"

# GLG2's reference VecNormalize settings (RL/experiment_manager.py, pinned commit).
_VEC_NORM_KWARGS = {"norm_obs": True, "norm_reward": True, "clip_obs": 10.0}


def _reference_hyperparams(config_dir: str) -> tuple[dict[str, Any], int]:
    """The GLG2 reference PPO hyperparameters, read from the pinned checkout.

    Returns ``(ppo_kwargs, n_envs)``. ``total_timesteps`` is dropped (our budget
    ladder governs run length); the string-valued ``policy_kwargs`` entries are
    translated to the objects SB3 expects, exactly as GLG2's own loader does.
    """
    import torch

    path = Path(config_dir).parent / "agents" / "ppo.yml"
    params = dict(yaml.safe_load(path.read_text())["TomatoEnv"])
    params.pop("total_timesteps")
    n_envs = int(params.pop("n_envs"))
    params.pop("policy")  # positional in our PPO(...) call

    activations = {"silu": torch.nn.SiLU, "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}
    optimizers = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop}
    pk = dict(params["policy_kwargs"])
    pk["activation_fn"] = activations[str(pk["activation_fn"]).lower()]
    pk["optimizer_class"] = optimizers[str(pk["optimizer_class"]).lower()]
    pk["log_std_init"] = _parse_log_std_init(pk["log_std_init"])
    params["policy_kwargs"] = pk
    params["learning_rate"] = float(params["learning_rate"])
    return params, n_envs


def _parse_log_std_init(value: Any) -> float:
    """GLG2's yaml writes ``log_std_init: np.log(1)`` as a string; evaluate that one
    known pattern (never a general eval) or accept a plain number."""
    if isinstance(value, int | float):
        return float(value)
    match = re.fullmatch(r"np\.log\(([0-9.]+)\)", str(value).strip())
    if match is None:
        raise ValueError(f"unrecognised log_std_init: {value!r}")
    return math.log(float(match.group(1)))


def _make_train_venv(
    config_dir: str,
    weather_dir: str,
    location: str,
    train_years: tuple[int, ...],
    n_envs: int,
    seed: int,
) -> Any:
    """``SubprocVecEnv(n_envs)`` + ``VecNormalize``, per the GLG2 reference pipeline."""
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    overrides = {
        "location": location,
        "start_train_year": min(train_years),
        "end_train_year": max(train_years),
        "start_train_day": 59,
        "end_train_day": 59,
        "training": True,
    }

    def _factory(rank: int) -> Any:
        def _init() -> Any:
            env = make_tomato_env(config_dir, weather_dir, overrides=dict(overrides))
            env.reset(seed=seed + rank)
            env.action_space.seed(seed + rank)
            return env

        return _init

    venv = SubprocVecEnv([_factory(rank) for rank in range(n_envs)])
    return VecNormalize(venv, **_VEC_NORM_KWARGS)


def _eval_env(config_dir: str, weather_dir: str, location: str, year: int) -> Any:
    return make_tomato_env(
        config_dir,
        weather_dir,
        overrides={"location": location, "training": False},
        spec_overrides={
            "eval_options": {"eval_days": [59], "eval_years": [year], "location": location}
        },
    )


def _heldout_net(
    model_path: Path, config_dir: str, weather_dir: str, location: str, years: tuple[int, ...]
) -> float:
    """Held-out net of a SAVED checkpoint pair, loaded exactly as the benchmark loads it
    (model + VecNormalize stats through ``load_ppo_policy``), so the curve measures the
    artifact we ship, not the in-memory model."""
    policy = load_ppo_policy(str(model_path))
    nets = [
        run_season(
            _eval_env(config_dir, weather_dir, location, int(y)), policy, seed=0
        ).net_economic_value
        for y in years
    ]
    return float(np.mean(nets))


def train_one_seed(
    seed: int,
    budgets: list[int],
    config_dir: str,
    weather_dir: str,
    location: str,
    train_years: tuple[int, ...],
    heldout: tuple[int, ...],
    checkpoint_dir: Path,
) -> dict[int, dict[str, float]]:
    """Train one PPO run per the reference recipe; save a policy + stats pair at each
    budget; return ``{budget: {heldout_net, actual_steps}}``."""
    from stable_baselines3 import PPO

    ppo_kwargs, n_envs = _reference_hyperparams(config_dir)
    venv = _make_train_venv(config_dir, weather_dir, location, train_years, n_envs, seed)
    try:
        model = PPO("MlpPolicy", venv, seed=seed, device="cpu", verbose=0, **ppo_kwargs)
        curve: dict[int, dict[str, float]] = {}
        for budget in budgets:
            remaining = budget - model.num_timesteps
            if remaining > 0:
                model.learn(
                    total_timesteps=remaining,
                    reset_num_timesteps=(model.num_timesteps == 0),
                )
            path = checkpoint_dir / f"ppo_s{seed}_b{budget}.zip"
            model.save(path)
            venv.save(str(vecnorm_stats_path(path)))
            net = _heldout_net(path, config_dir, weather_dir, location, heldout)
            curve[budget] = {"heldout_net": net, "actual_steps": int(model.num_timesteps)}
            print(
                f"  seed {seed} budget {budget:>8} (actual {model.num_timesteps:>8}): "
                f"held-out net {net:8.3f}  -> {path.name}",
                flush=True,
            )
        return curve
    finally:
        venv.close()  # SubprocVecEnv workers must not outlive the seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seeds", type=int, nargs="*", default=None, help="override seeds (default: config)"
    )
    ap.add_argument("--checkpoint-dir", default="checkpoints", help="gitignored policy dir (local)")
    ap.add_argument("--out", default="results/benchmark", help="committed summary dir")
    ap.add_argument("--budgets", type=int, nargs="*", default=None, help="override budget ladder")
    args = ap.parse_args()

    cfg = load_config(_CONFIG)
    config_dir = os.environ["GLG2_CONFIG_DIR"]
    weather_dir = os.environ["GLG2_WEATHER_DIR"]
    location = cfg.generalisation.location
    train_years = cfg.generalisation.train_years
    heldout = cfg.generalisation.heldout_years
    budgets = (
        args.budgets
        if args.budgets is not None
        else [int(b) for b in cfg.raw["data_efficiency"]["ppo_budgets"]]
    )
    seeds = (
        args.seeds
        if args.seeds is not None
        else list(range(int(cfg.raw["reporting"]["seeds_per_condition"])))
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"training {len(seeds)} seed(s) x {len(budgets)} budget(s); location {location}; "
        "recipe: GLG2 reference (VecNormalize + pinned ppo.yml)",
        flush=True,
    )
    for seed in seeds:
        curve = train_one_seed(
            seed, budgets, config_dir, weather_dir, location, train_years, heldout, checkpoint_dir
        )
        summary = out_dir / f"data_efficiency_s{seed}.json"
        summary.write_text(
            json.dumps(
                {"seed": seed, "location": location, "recipe": "glg2-reference", "curve": curve},
                indent=2,
            )
        )
        print(f"seed {seed}: wrote {summary}", flush=True)


if __name__ == "__main__":
    main()
