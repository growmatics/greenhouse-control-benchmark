"""benchmark (review round 1): train the SAC data-efficiency curve as a second black-box arm.

Peer review asked whether the data-efficiency crossover is a property of learned control
or only of PPO. SAC is the off-policy counterpart GreenLight-Gym2 also ships. This driver
mirrors ``train_policies.py`` (same VecNormalize pipeline, budget ladder, checkpoint+stats
pairs, held-out evaluation of the SAVED artifact) and builds SAC FAITHFULLY from the pinned
reference recipe ``gl_gym/configs/agents/sac.yml``.

RESULT HISTORY. As-shipped (2026-07-17, full ladder + two audits): faithful training from the
shipped recipe dips to ~-16 at 100-200k, RECOVERS, and plateaus at held-out net +1.62 at 2M
(seeds 0/1: +1.496 / +1.747); the earlier "diverges" reading was premature and is retracted.
The shipped v2 config is aggressive versus standard SAC (lr 7e-4 vs 3e-4, tau 0.0135 vs 0.005,
train_freq 50 + gradient_steps 10 vs 1+1, n_envs 8 vs 1: update-to-data ~0.025 vs ~1.0).
TUNED (2026-07-23, the current standing result): the equal-procedure random-search tune
(sealed prereg ``experiments/expectations-r2-sac-tuning.md``) lifts SAC to a 2M 5-seed mean
of +3.21 (winner t19; runner-up t36 +3.14), positive by ~100k with no valley, vs PPO's 5-seed
+3.50; all 4 sealed predictions confirmed (``experiments/outcomes-r2-sac-tuning.md``). This
driver trains either recipe: default = as-shipped; ``--config`` overlays a tuning winner.

Outputs: ``sac_s{seed}_b{budget}.zip`` + ``..._vecnorm.pkl`` (checkpoints, bulk storage),
and ``<out>/data_efficiency_sac_s{seed}.json`` (committed curve).

    GLG2_CONFIG_DIR=.../gl_gym/configs/envs GLG2_WEATHER_DIR=.../bench_weather \
    python examples/train_sac.py --checkpoint-dir checkpoints --out results/benchmark
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
from greenlight_bench.controllers.rl_mpc import vecnorm_stats_path
from greenlight_bench.envs.glg2 import make_tomato_env, run_season

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "benchmark.yaml"
# gamma is set per-run to the model gamma (the reference passes it to VecNormalize so the
# reward-return running std uses the right discount; omitting it defaults to 0.99).
_VEC_NORM_BASE = {"norm_obs": True, "norm_reward": True, "clip_obs": 10.0}


def _parse_log_std_init(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    match = re.fullmatch(r"np\.log\(([0-9.]+)\)", str(value).strip())
    if match is None:
        raise ValueError(f"unrecognised log_std_init: {value!r}")
    return math.log(float(match.group(1)))


def _reference_sac_hyperparams(config_dir: str) -> tuple[dict[str, Any], int]:
    """The GLG2 reference SAC hyperparameters from the pinned checkout's sac.yml.

    Returns ``(sac_kwargs, n_envs)``. Off-policy specifics (buffer, tau, train_freq,
    gradient_steps, action noise, ent_coef=auto) pass straight to SB3 SAC; the
    string-valued policy_kwargs and the action-noise block are translated to objects.
    """
    import torch
    from stable_baselines3.common.noise import NormalActionNoise

    path = Path(config_dir).parent / "agents" / "sac.yml"
    params = dict(yaml.safe_load(path.read_text())["TomatoEnv"])
    params.pop("total_timesteps")
    n_envs = int(params.pop("n_envs"))
    params.pop("policy")

    activations = {"silu": torch.nn.SiLU, "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}
    optimizers = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop}
    pk = dict(params["policy_kwargs"])
    pk["activation_fn"] = activations[str(pk["activation_fn"]).lower()]
    pk["optimizer_class"] = optimizers[str(pk["optimizer_class"]).lower()]
    # Keep log_std_init to match the reference config. NOTE (independent audit 2026-07-17):
    # for SAC with use_sde=False, log_std_init is INERT -- SB3's SAC derives log-std from a
    # state-dependent layer, so this value does not affect training. It is retained only for
    # faithfulness; it was NOT the cause of the divergence (an earlier docstring wrongly
    # claimed a "20x exploration cut" -- retracted). The substantive VecNormalize gamma fix
    # below is the one that genuinely mattered.
    pk["log_std_init"] = _parse_log_std_init(pk["log_std_init"])
    params["policy_kwargs"] = pk
    params["learning_rate"] = float(params["learning_rate"])

    noise = params.pop("action_noise", None)
    if noise and "normalactionnoise" in noise:
        sigma = float(noise["normalactionnoise"]["sigma"])
        params["action_noise"] = NormalActionNoise(mean=np.zeros(6), sigma=sigma * np.ones(6))
    # drop keys SB3 SAC does not take as constructor kwargs
    for k in ("replay_buffer_class", "replay_buffer_kwargs", "stats_window_size"):
        params.pop(k, None)
    return params, n_envs


def _make_train_venv(config_dir, weather_dir, location, train_years, n_envs, seed, gamma) -> Any:
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

    venv = SubprocVecEnv([_factory(r) for r in range(n_envs)])
    return VecNormalize(venv, gamma=gamma, **_VEC_NORM_BASE)


def _eval_env(config_dir, weather_dir, location, year) -> Any:
    return make_tomato_env(
        config_dir,
        weather_dir,
        overrides={"location": location, "training": False},
        spec_overrides={
            "eval_options": {"eval_days": [59], "eval_years": [year], "location": location}
        },
    )


def _heldout_net(model_path: Path, config_dir, weather_dir, location, years) -> float:
    from greenlight_bench.controllers.rl_mpc import load_ppo_policy  # loads any SB3 .zip + stats

    policy = load_ppo_policy(str(model_path))
    nets = [
        run_season(
            _eval_env(config_dir, weather_dir, location, int(y)), policy, seed=0
        ).net_economic_value
        for y in years
    ]
    return float(np.mean(nets))


def _apply_winner(
    sac_kwargs: dict[str, Any], n_envs: int, override_cfg: dict[str, Any] | None
) -> tuple[dict[str, Any], int]:
    """Overlay a tuning winner's SEARCHED params onto the frozen reference kwargs, so the
    confirmation run retrains the winning config bit-identically to how the trial was trained.

    This mirrors ``tune_sac_random._apply_overrides`` (the same four SAC kwargs) plus ``n_envs``,
    which sizes the vec env rather than being a SAC kwarg. Keep the two in sync. ``None`` leaves
    the reference recipe unchanged (train the shipped SAC).
    """
    if override_cfg is None:
        return sac_kwargs, n_envs
    out = dict(sac_kwargs)
    out["learning_rate"] = float(override_cfg["learning_rate"])
    out["tau"] = float(override_cfg["tau"])
    out["train_freq"] = int(override_cfg["train_freq"])
    out["gradient_steps"] = int(override_cfg["gradient_steps"])
    return out, int(override_cfg["n_envs"])


def _build_q_diagnostics_callback(every: int = 5000, batch_size: int = 256) -> Any:
    """Build a callback that logs the critic's Q magnitude and twin-critic disagreement, so SAC's
    overestimation behaviour is inspectable (the SAC-vs-TQC overestimation question we raised).

    SAC bootstraps from ``min(Q1, Q2)`` -- the clipped double-Q estimate -- to curb the
    overestimation bias of a single critic (``haarnoja2018sac``; the clipped-double-Q idea is
    TD3's). Every ``every`` steps after the buffer fills, we sample a batch exactly as ``SAC.train``
    does (through the VecNormalize env, so the critic sees training-scale inputs) and record: the
    mean bootstrapped value ``min(Q1,Q2)``, its spread, and the mean twin gap ``|Q1 - Q2|``. A
    widening twin gap or a runaway Q mean is the overestimation signature. Values are in
    normalised-reward units (VecNormalize), so read them as trends, not absolute EUR. Landed as
    ``diag/*`` columns in progress.csv on the next logger dump; cost is one critic forward pass per
    ``every`` steps (negligible against a 2M budget). Defined lazily to keep SB3 an optional import
    (mirrors the rest of this module and ``rl_mpc.py``).
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class _QDiagnosticsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self._every = every
            self._batch_size = batch_size

        def _on_step(self) -> bool:
            if self.num_timesteps % self._every != 0:
                return True
            import torch

            buf = self.model.replay_buffer
            if buf.size() * buf.n_envs < self._batch_size:
                return True
            data = buf.sample(self._batch_size, env=self.model.get_vec_normalize_env())
            with torch.no_grad():
                qs = self.model.critic(data.observations, data.actions)
                q_min = torch.min(torch.stack(qs), dim=0).values  # clipped double-Q value
                self.logger.record("diag/q_min_mean", float(q_min.mean()))
                self.logger.record("diag/q_min_std", float(q_min.std()))
                self.logger.record("diag/q_twin_gap_mean", float((qs[0] - qs[1]).abs().mean()))
            return True

    return _QDiagnosticsCallback()


def train_one_seed(
    seed,
    budgets,
    config_dir,
    weather_dir,
    location,
    train_years,
    heldout,
    ckpt,
    override_cfg=None,
    device="cpu",
) -> dict:
    from stable_baselines3 import SAC

    sac_kwargs, n_envs = _reference_sac_hyperparams(config_dir)
    sac_kwargs, n_envs = _apply_winner(sac_kwargs, n_envs, override_cfg)
    venv = _make_train_venv(
        config_dir, weather_dir, location, train_years, n_envs, seed, sac_kwargs["gamma"]
    )
    try:
        # device: "cpu" (default; the WSL run of record, bit-deterministic) | "mps" (Apple GPU) |
        # "cuda" | "auto". A GPU device changes float numerics, so a GPU run is a SELF-CONSISTENT
        # set on ONE device, never mixed with CPU seeds in the same benchmark column.
        model = SAC("MlpPolicy", venv, seed=seed, device=device, verbose=0, **sac_kwargs)
        # Per-seed CSV progress log so a multi-hour 2M run is INSPECTABLE mid-run (read
        # ``<ckpt>/curves/sac_s{seed}/progress.csv`` for time/total_timesteps + rollout
        # ep_rew_mean), mirroring the tuning's per-trial logging. Without this a 2M run is
        # silent until completion -- the monitoring gap that bit the r2 search.
        from stable_baselines3.common.logger import configure as _configure_logger

        model.set_logger(_configure_logger(str(ckpt / "curves" / f"sac_s{seed}"), ["csv"]))
        q_diag = _build_q_diagnostics_callback()
        curve: dict[int, dict[str, float]] = {}
        for budget in budgets:
            remaining = budget - model.num_timesteps
            if remaining > 0:
                model.learn(
                    total_timesteps=remaining,
                    reset_num_timesteps=(model.num_timesteps == 0),
                    callback=q_diag,
                )
            path = ckpt / f"sac_s{seed}_b{budget}.zip"
            model.save(path)
            venv.save(str(vecnorm_stats_path(path)))
            net = _heldout_net(path, config_dir, weather_dir, location, heldout)
            curve[budget] = {"heldout_net": net, "actual_steps": int(model.num_timesteps)}
            print(
                f"  sac seed {seed} budget {budget:>8} (actual {model.num_timesteps:>8}): "
                f"held-out net {net:8.3f}  -> {path.name}",
                flush=True,
            )
        return curve
    finally:
        venv.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--out", default="results/benchmark")
    ap.add_argument("--budgets", type=int, nargs="*", default=None)
    ap.add_argument(
        "--config",
        default=None,
        help="winner config JSON from `tune_sac_random.py --select-only` "
        "(sac_random_winner.json); overlays its searched params onto the shipped recipe for the "
        "confirmation run. Default None trains the shipped reference recipe.",
    )
    ap.add_argument(
        "--device",
        default="cpu",
        help="torch device for the SAC nets: cpu (default; the deterministic run of record) | "
        "mps (Apple GPU) | cuda | auto. The env sim always runs on CPU. A GPU changes float "
        "numerics: keep a confirmation on ONE device, recorded (never mix into a CPU column).",
    )
    args = ap.parse_args()

    override_cfg = None
    if args.config:
        raw = json.loads(Path(args.config).read_text())
        override_cfg = raw.get("config", raw)  # accept the winner json or a bare config dict

    cfg = load_config(_CONFIG)
    config_dir = os.environ["GLG2_CONFIG_DIR"]
    weather_dir = os.environ["GLG2_WEATHER_DIR"]
    location = cfg.generalisation.location
    train_years = cfg.generalisation.train_years
    heldout = cfg.generalisation.heldout_years
    budgets = args.budgets or [int(b) for b in cfg.raw["data_efficiency"]["ppo_budgets"]]
    seeds = (
        args.seeds
        if args.seeds is not None
        else list(range(int(cfg.raw["reporting"]["seeds_per_condition"])))
    )

    ckpt = Path(args.checkpoint_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    recipe = f"tuning winner ({args.config})" if override_cfg else "GLG2 sac.yml"
    print(
        f"training SAC: {len(seeds)} seed(s) x {len(budgets)} budget(s); recipe: {recipe}",
        flush=True,
    )
    for seed in seeds:
        curve = train_one_seed(
            seed,
            budgets,
            config_dir,
            weather_dir,
            location,
            train_years,
            heldout,
            ckpt,
            override_cfg,
            args.device,
        )
        summary = out_dir / f"data_efficiency_sac_s{seed}.json"
        summary.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "algo": "sac",
                    "device": args.device,
                    "location": location,
                    "curve": curve,
                },
                indent=2,
            )
        )
        print(f"sac seed {seed}: wrote {summary}", flush=True)


if __name__ == "__main__":
    main()
