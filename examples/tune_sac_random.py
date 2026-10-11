"""Round-2 tuning: equal-procedure random search for SAC hyperparameters.

Mirrors the procedure the GreenLight-Gym paper used for PPO (a random search;
van Laatum et al. 2024, arXiv 2410.05336, Sec 3.3.3), applied to SAC, which the
paper does not tune. Pre-registered and SEALED in
``experiments/expectations-r2-sac-tuning.md`` (tag ``prereg-r2-2026-07-17``,
content SHA-256 recorded in the run records).

Method note (recorded in the outcomes file, not a deviation from the science):
the paper ran its random search on the Weights & Biases platform; we run the
identical METHOD (uniform random sampling over the same-style space, one fixed
sampler seed) locally, for reproducibility and to avoid an external-service
dependency. The scientific content the prereg fixes (the search space, 40 trials,
the per-trial budget, the selection metric and split) is exactly reproduced here.

Protocol (frozen, from the sealed prereg):
  - 40 trials, sampler seed 0, per-trial 1,000,000 env steps, single training seed 0.
  - Search space: learning_rate log-uniform [1e-5, 1e-3], n_envs {1,2,4,8},
    train_freq {1,8,16,50}, gradient_steps {1,4,8,10}, tau log-uniform [0.005,0.02].
  - Frozen: everything else in the reference sac.yml (net arch, gamma, batch_size,
    buffer_size, learning_starts, ent_coef auto, action noise, use_sde False).
  - Train on weather years 2014-2019; select on validation years 2020-2021 (held
    out from tuning-training). The test set 2022-2025 is untouched here.
  - Winner = highest mean validation held-out net; tie-break lower n_envs then lower lr.

All 40 configs are sampled up front from the seeded RNG, so the per-trial run is
reproducible and the work can be split across processes (``--trial-start`` /
``--trial-count``) without changing which config each trial index gets. Each
finished trial is appended immediately to ``<out>/sac_random_trials.jsonl`` (a
crash never loses completed trials).

    GLG2_CONFIG_DIR=.../gl_gym/configs/envs GLG2_WEATHER_DIR=.../bench_weather \
    python examples/tune_sac_random.py --checkpoint-dir <ckpt>/tuning --out <out>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from greenlight_bench.config import load_config
from greenlight_bench.controllers.rl_mpc import vecnorm_stats_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sac import (  # reuse the exact training + eval pipeline
    _heldout_net,
    _make_train_venv,
    _reference_sac_hyperparams,
)

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "benchmark.yaml"

# --- SEALED protocol constants (expectations-r2-sac-tuning.md). Do not change here;
#     a change is a new pre-registration. -----------------------------------------
_SAMPLER_SEED = 0
_N_TRIALS = 40
_TRIAL_BUDGET = 1_000_000
_TRAIN_SEED = 0
_TUNE_TRAIN_YEARS = list(range(2014, 2020))  # 2014-2019 (tuning-training)
_TUNE_VAL_YEARS = [2020, 2021]  # validation (held out from tuning-training)
# Search space: (kind, spec). "loguniform" -> (lo, hi); "choice" -> tuple of options.
_SEARCH_SPACE: dict[str, tuple[str, Any]] = {
    "learning_rate": ("loguniform", (1e-5, 1e-3)),
    "n_envs": ("choice", (1, 2, 4, 8)),
    "train_freq": ("choice", (1, 8, 16, 50)),
    "gradient_steps": ("choice", (1, 4, 8, 10)),
    "tau": ("loguniform", (0.005, 0.02)),
}

# --- AMENDMENT 2026-07-18/19 (recorded in outcomes-r2-sac-tuning.md and the project decision record;
#     the sealed constants above are untouched). Three trials are EXCLUDED for
#     feasibility -- all >= 2M gradient updates / > 1 day wall clock, which exceeds
#     the WSL box's observed mean time between OOM reboots, so they cannot finish and
#     each crash loses their full partial run (SB3 checkpoints only at completion).
#     The exclusion is blind to their scores (none ever recorded a result):
#       t29 (5.0M updates, lr 4.0e-4, ~70 h), t21 (2.0M, lr 1.0e-4, ~28 h),
#       t25 (2.0M updates, lr 1.7e-5, ~28 h; excluded 2026-07-19).
#     Only t29 and t21 were REPLACED (indices 40-41), because their lr sat in/at the
#     viable band, so excluding them lost real coverage of the winning region; the
#     replacements are drawn from a fresh seeded stream over a narrowed viable-band
#     subspace (train_freq >= 8; update volume capped to the winners' range, ~4 h).
#     t25 is NOT replaced: at lr 1.7e-5 it is a near-certain ~-15 floor (confirmed by
#     the 250k-update monotone triple t15/t13/t24), so excluding it loses no
#     information and there is nothing to restore. Reporting: 40 uniform configs drawn,
#     3 excluded for feasibility, 37 uniform trials completed, + 2 labelled refinement
#     trials. NOT a clean 40-trial uniform draw. ----------------------------------
_EXCLUDED_TRIALS = frozenset({21, 25, 29})
_REFINE_SEED = 1  # fresh stream; seed 0 is the sealed draw
_N_REFINE = 2
_REFINE_SPACE: dict[str, tuple[str, Any]] = {
    "learning_rate": ("loguniform", (1.5e-4, 9e-4)),
    "n_envs": ("choice", (2, 4, 8)),
    "train_freq": ("choice", (8, 16, 50)),
    "gradient_steps": ("choice", (1, 4, 8, 10)),
    "tau": ("loguniform", (0.005, 0.02)),
}
_REFINE_UPDATE_BAND = (20_000, 160_000)  # gradient updates at the 1M-step budget


def sample_configs(n_trials: int = _N_TRIALS, seed: int = _SAMPLER_SEED) -> list[dict[str, Any]]:
    """The full, reproducible list of sampled configs (uniform random search).

    Sampled once from a seeded generator so trial ``i`` is the same config in any
    process; splitting the run into slices does not change the assignment.
    """
    rng = np.random.default_rng(seed)
    return [_sample_one(rng, _SEARCH_SPACE) for _ in range(n_trials)]


def _sample_one(rng: np.random.Generator, space: dict[str, tuple[str, Any]]) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for name, (kind, spec) in space.items():
        if kind == "loguniform":
            lo, hi = spec
            cfg[name] = float(10 ** rng.uniform(math.log10(lo), math.log10(hi)))
        elif kind == "choice":
            # int() so JSON stores plain ints, not numpy scalars
            cfg[name] = int(rng.choice(np.asarray(spec)))
        else:  # pragma: no cover - guarded by the frozen spaces above
            raise ValueError(f"unknown sampling kind {kind!r}")
    return cfg


def gradient_updates(cfg: dict[str, Any], budget: int = _TRIAL_BUDGET) -> int:
    """Gradient updates a config performs over ``budget`` env steps (SB3 SAC schedule)."""
    return budget // (int(cfg["train_freq"]) * int(cfg["n_envs"])) * int(cfg["gradient_steps"])


def sample_refinement_configs(
    n_trials: int = _N_REFINE, seed: int = _REFINE_SEED
) -> list[dict[str, Any]]:
    """The amendment's replacement trials (indices 40+): a seeded draw from the
    narrowed refinement subspace, rejection-sampled to the winners' gradient-update
    band so no replacement can recreate the multi-day trials being replaced."""
    rng = np.random.default_rng(seed)
    configs: list[dict[str, Any]] = []
    lo, hi = _REFINE_UPDATE_BAND
    while len(configs) < n_trials:
        cfg = _sample_one(rng, _REFINE_SPACE)
        if lo <= gradient_updates(cfg) <= hi:
            configs.append(cfg)
    return configs


def _apply_overrides(sac_kwargs: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Overlay a sampled config onto the frozen reference kwargs (searched keys only)."""
    out = dict(sac_kwargs)
    out["learning_rate"] = float(cfg["learning_rate"])
    out["tau"] = float(cfg["tau"])
    out["train_freq"] = int(cfg["train_freq"])
    out["gradient_steps"] = int(cfg["gradient_steps"])
    return out  # n_envs is handled separately (it sizes the vec env, not a SAC kwarg)


def _saved_num_timesteps(path: Path) -> int:
    """``num_timesteps`` recorded inside an SB3 checkpoint zip (no model load needed)."""
    import io
    import zipfile

    with zipfile.ZipFile(path) as zf, zf.open("data") as fh:
        return int(json.load(io.TextIOWrapper(fh))["num_timesteps"])


def score_checkpoint(
    idx: int,
    cfg: dict[str, Any],
    config_dir: str,
    weather_dir: str,
    location: str,
    path: Path,
) -> dict[str, Any]:
    """Validation-score a trained trial checkpoint (the post-training half of a trial)."""
    val_net = _heldout_net(path, config_dir, weather_dir, location, _TUNE_VAL_YEARS)
    return {
        "trial": idx,
        "config": cfg,
        "val_net": float(val_net),
        "actual_steps": _saved_num_timesteps(path),
        "checkpoint": path.name,
    }


def run_trial(
    idx: int,
    cfg: dict[str, Any],
    config_dir: str,
    weather_dir: str,
    location: str,
    ckpt: Path,
    budget: int = _TRIAL_BUDGET,
) -> dict[str, Any]:
    """Train one sampled config to ``budget`` env steps, score on validation years."""
    from stable_baselines3 import SAC

    ref_kwargs, _ref_n_envs = _reference_sac_hyperparams(config_dir)
    sac_kwargs = _apply_overrides(ref_kwargs, cfg)
    n_envs = int(cfg["n_envs"])
    venv = _make_train_venv(
        config_dir,
        weather_dir,
        location,
        _TUNE_TRAIN_YEARS,
        n_envs,
        _TRAIN_SEED,
        sac_kwargs["gamma"],
    )
    try:
        model = SAC("MlpPolicy", venv, seed=_TRAIN_SEED, device="cpu", verbose=0, **sac_kwargs)
        # Passive observability, not part of the sealed protocol (selection stays on the
        # final validation value): stream the training curve (ep_rew_mean vs steps) to a
        # per-trial CSV so multi-hour trials are inspectable and stability is on record.
        from stable_baselines3.common.logger import configure

        curve_dir = ckpt / "curves" / f"t{idx:02d}"
        curve_dir.mkdir(parents=True, exist_ok=True)
        model.set_logger(configure(str(curve_dir), ["csv"]))
        model.learn(total_timesteps=budget, reset_num_timesteps=True)
        path = ckpt / f"sac_tune_t{idx:02d}.zip"
        model.save(path)
        venv.save(str(vecnorm_stats_path(path)))
        return score_checkpoint(idx, cfg, config_dir, weather_dir, location, path)
    finally:
        venv.close()


def _selection_key(t: dict[str, Any]) -> tuple[float, int, float]:
    """The sealed selection rule: highest validation net, tie-break toward lower n_envs
    then lower learning_rate. A non-finite val_net (a diverged trial whose evaluation
    returned NaN/inf) sorts to the bottom so it can never rank."""
    v = float(t["val_net"])
    v = v if math.isfinite(v) else -math.inf
    return (v, -int(t["config"]["n_envs"]), -float(t["config"]["learning_rate"]))


def select_winner(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """The single winning trial by the sealed selection rule (see ``_selection_key``)."""
    return max(trials, key=_selection_key)


def select_ranked(trials: list[dict[str, Any]], n: int = 2) -> list[dict[str, Any]]:
    """The top-``n`` trials by the sealed selection rule, best first. Used by the
    confirmation run to retrieve the winner AND the runner-up (A/B/C decision B: winner
    x5 seeds + runner-up x1 seed)."""
    return sorted(trials, key=_selection_key, reverse=True)[:n]


def _read_trials(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_all_trials(out_dir: Path) -> list[dict[str, Any]]:
    """Merge every per-slice trials file; on a duplicate trial index the later record wins."""
    by_idx: dict[int, dict[str, Any]] = {}
    for path in sorted(out_dir.glob("sac_random_trials*.jsonl")):
        for row in _read_trials(path):
            by_idx[int(row["trial"])] = row
    return [by_idx[k] for k in sorted(by_idx)]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default="checkpoints/tuning")
    ap.add_argument("--out", default="results/benchmark/tuning")
    ap.add_argument("--trial-start", type=int, default=0, help="first trial index (inclusive)")
    ap.add_argument("--trial-count", type=int, default=_N_TRIALS, help="how many trials to run")
    ap.add_argument(
        "--trials",
        default=None,
        help="explicit comma-separated trial indices (overrides --trial-start/--trial-count; "
        "lets workers take cost-balanced, non-contiguous slices)",
    )
    ap.add_argument("--n-trials", type=int, default=_N_TRIALS, help="total sampled configs (smoke)")
    ap.add_argument(
        "--budget", type=int, default=None, help="override per-trial steps (smoke only)"
    )
    ap.add_argument("--select-only", action="store_true", help="report the winner from the JSONL")
    return ap.parse_args()


def main() -> None:  # noqa: PLR0915 -- linear CLI driver of the completed sealed search; splitting it would scatter the audit trail
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.select_only:
        trials = _read_all_trials(out_dir)
        if not trials:
            print("no trials recorded yet", flush=True)
            return
        ranked = select_ranked(trials, n=2)
        win = ranked[0]
        (out_dir / "sac_random_winner.json").write_text(json.dumps(win, indent=2))
        print(
            f"winner:    trial {win['trial']} val_net {win['val_net']:.3f} config {win['config']}"
        )
        if len(ranked) > 1:
            run = ranked[1]
            (out_dir / "sac_random_runnerup.json").write_text(json.dumps(run, indent=2))
            print(
                f"runner-up: trial {run['trial']} val_net {run['val_net']:.3f} "
                f"config {run['config']}"
            )
        return

    # sealed per-trial budget is 1,000,000; --budget overrides for smoke tests only
    budget = int(args.budget) if args.budget is not None else _TRIAL_BUDGET
    cfg = load_config(_CONFIG)
    config_dir = os.environ["GLG2_CONFIG_DIR"]
    weather_dir = os.environ["GLG2_WEATHER_DIR"]
    location = cfg.generalisation.location
    ckpt = Path(args.checkpoint_dir)
    ckpt.mkdir(parents=True, exist_ok=True)

    configs = sample_configs(n_trials=args.n_trials)
    if args.n_trials == _N_TRIALS:
        configs = configs + sample_refinement_configs()  # amendment trials 40-41
    if args.trials is not None:
        indices = [int(t) for t in args.trials.split(",")]
    else:
        stop = min(args.trial_start + args.trial_count, len(configs))
        indices = list(range(args.trial_start, stop))
    dropped = [i for i in indices if i in _EXCLUDED_TRIALS]
    if dropped:
        print(f"excluded by the 2026-07-18 amendment, skipping: {dropped}", flush=True)
        indices = [i for i in indices if i not in _EXCLUDED_TRIALS]
    # per-slice file (named by the slice's first index) so concurrent processes never
    # append to the same file; _read_all_trials merges the slices at selection time
    trials_path = out_dir / f"sac_random_trials_s{indices[0]:02d}.jsonl"

    # resume support: a relaunch after a crash/reboot skips recorded trials and
    # recovers trained-but-unscored checkpoints instead of retraining them
    recorded = {int(row["trial"]) for row in _read_all_trials(out_dir)}
    print(
        f"SAC random search: trials {indices} of {len(configs)}; "
        f"budget {budget}/trial; train {_TUNE_TRAIN_YEARS[0]}-{_TUNE_TRAIN_YEARS[-1]}, "
        f"validate {_TUNE_VAL_YEARS}; {len(recorded)} already recorded",
        flush=True,
    )
    for idx in indices:
        if idx in recorded:
            print(f"  trial {idx:02d}: already recorded, skipping", flush=True)
            continue
        path = ckpt / f"sac_tune_t{idx:02d}.zip"
        if path.exists() and budget == _TRIAL_BUDGET:
            # trained by a run that died before scoring: eval-only recovery, no retrain
            print(f"  trial {idx:02d}: checkpoint found, recovering (eval-only)", flush=True)
            result = score_checkpoint(idx, configs[idx], config_dir, weather_dir, location, path)
            result["recovered"] = True
        else:
            result = run_trial(idx, configs[idx], config_dir, weather_dir, location, ckpt, budget)
        if idx >= _N_TRIALS:  # amendment replacement trial: labelled in the record
            result["stage"] = "refinement"
        with trials_path.open("a") as fh:  # append immediately: a crash keeps finished trials
            fh.write(json.dumps(result) + "\n")
        print(
            f"  trial {idx:02d}: val_net {result['val_net']:8.3f}  {result['config']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
