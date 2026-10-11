"""The multi-arm benchmark over the full grid, parallel across CPU cores.

Runs every arm through the SAME harness (``run_season_controller``) on the shared grid
defined in ``configs/benchmark.yaml`` (held-out years x mismatch ladder x seeds), then
aggregates net economic value, the feasibility gate, the Pareto front, and MPC
solve-success. Each (arm, year, mismatch, seed) season is an independent task; since every
IPOPT solve is single-threaded, the ONLY way to use the Mac's cores is to parallelise
across seasons -- ``--workers N`` does exactly that (each worker builds its own env and
controller from plain args, so nothing un-picklable crosses the process boundary).

Arm-aware axes (avoid redundant compute; each arm's *independent* trials):
  incumbent        -- years            (deterministic: seed/mismatch do not vary it)
  black_box (RL)   -- years x seeds    (policy per seed; mismatch does not apply)
  physics_mpc      -- years x mismatch x seeds
  physics_informed -- years x mismatch x seeds

Policies (arms 2, 4) load ``checkpoints/ppo_s{seed}_b{budget}.zip`` from ``train_policies``
(the max budget by default). Outputs: a small committed JSON summary under ``--out``. The
committed summary reduces each season's per-class exceedance arrays to scalars (max excursion,
degree-hours, banded flag at ONE threshold), so a NEW banded/disease threshold cannot be scored
after the fact from it. Pass ``--trace-dir`` to also persist the full per-step exceedance arrays
(one JSON per season) to big-file storage; that makes any later gate re-calibration a re-analysis
instead of a re-run (lesson 2026-07-19). See the run reports in results/benchmark/.

    GLG2_CONFIG_DIR=.../gl_gym/configs/envs GLG2_WEATHER_DIR=.../bench_weather \
    python examples/run_benchmark.py --workers 8 --checkpoint-dir checkpoints \
        --out results/benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from greenlight_bench.benchmark import BenchmarkResult, hardware_stamp
from greenlight_bench.config import load_config
from greenlight_bench.envs.glg2 import SeasonResult

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "benchmark.yaml"
_ARMS = ("incumbent", "black_box", "black_box_sac", "physics_mpc", "physics_informed")
_MAX_MISSING_SHOWN = 10  # cap the missing-policy preflight list in the error message


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    """True when the working tree differs from HEAD (or its state is unknowable).

    The run record's ``git_commit`` is only honest provenance if the tree matches it;
    a run from a patched tree must say so (stamped in the record AND in the summary
    filename), never present itself as the clean commit.
    """
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
        return bool(out)
    except Exception:
        return True  # unknown tree state counts as dirty, never as clean


def _policy_path(
    checkpoint_dir: str, seed: int, budget: int, override: str | None, algo: str = "ppo"
) -> str:
    if override:
        return override
    return str(Path(checkpoint_dir) / f"{algo}_s{seed}_b{budget}.zip")


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    """Build one arm on one condition and run a season. Pure w.r.t. its plain-data ``task``
    (runs in a worker process; imports and builds everything fresh, nothing pickled)."""
    import numpy as _np

    from greenlight_bench.config import load_config as _load
    from greenlight_bench.controllers.base import NominalParameters
    from greenlight_bench.controllers.mpc import PhysicsOnlyMPC
    from greenlight_bench.controllers.rl import RLController
    from greenlight_bench.controllers.rl_mpc import PhysicsInformedMPC, load_ppo_policy
    from greenlight_bench.controllers.rule_based import RuleBasedIncumbent
    from greenlight_bench.envs.glg2 import make_tomato_env, run_season_controller
    from greenlight_bench.mismatch import perturb_params

    cfg = _load(task["config_path"])
    config_dir, weather_dir = task["config_dir"], task["weather_dir"]
    year, mismatch, seed, arm = task["year"], task["mismatch"], task["seed"], task["arm"]
    season_days = task["season_days"]

    def make_env() -> Any:
        return make_tomato_env(
            config_dir,
            weather_dir,
            overrides={
                "location": cfg.generalisation.location,
                "start_train_year": year,
                "end_train_year": year,
                "start_train_day": 59,
                "end_train_day": 59,
                "season_length": season_days,
            },
        )

    env = make_env()
    env.reset(seed=seed)
    bounds = {
        "co2_ppm": tuple(cfg.raw["constraints"]["co2_ppm"]),
        "air_temp_c": tuple(cfg.raw["constraints"]["air_temp_c"]),
        "rh_percent": tuple(cfg.raw["constraints"]["rh_percent"]),
    }
    if arm == "incumbent":
        rb = yaml.safe_load((Path(config_dir).parent / "agents" / "rule_based.yml").read_text())[
            "TomatoEnv"
        ]
        controller: Any = RuleBasedIncumbent(rb)
    elif arm in ("black_box", "black_box_sac"):
        controller = RLController(load_ppo_policy(task["policy_path"]))
    elif arm in ("physics_mpc", "physics_informed"):
        true = {str(i): float(v) for i, v in enumerate(_np.array(env.p).flatten())}
        nominal = NominalParameters(perturb_params(true, mismatch, seed))
        terminal_w = 0.01 if arm == "physics_informed" else 0.0
        mpc = PhysicsOnlyMPC(
            nominal,
            horizon=4,
            dt=900.0,
            objective="economic",
            economic=cfg.economic,
            climate_bounds=bounds,
            terminal_weight=terminal_w,
        )
        if arm == "physics_mpc":
            controller = mpc
        else:
            rollout = make_env()
            controller = PhysicsInformedMPC(
                nominal, load_ppo_policy(task["policy_path"]), rollout, mpc
            )
    else:
        raise ValueError(f"unknown arm {arm}")

    result = run_season_controller(env, controller, seed)
    advisor = getattr(controller, "advisor_stats", None)
    # rl_selection_rate is a @property (a float), not a method: read it, do not call it
    # (the previous callable()/sel() path captured None for every season -- audit 2026-07-17).
    sel = getattr(controller, "rl_selection_rate", None)
    return {
        "arm": arm,
        "label": task["label"],
        "seed": seed,
        "net_economic_value": result.net_economic_value,
        "exceedances": {k: v.tolist() for k, v in result.exceedances.items()},
        "online_cost_seconds": result.online_cost_seconds,
        "solve_success": result.solve_success,
        # arm 4's per-step advisor reliability + RL-selection rate (None for other arms).
        # rl_selection_rate vs mismatch is the "model-free-carried robustness" mechanism test.
        "advisor_fallback_steps": advisor["fallback"] if advisor else None,
        "rl_selection_rate": float(sel) if isinstance(sel, (int, float)) else None,
    }


def _run_task_safe(task: dict[str, Any]) -> dict[str, Any]:
    """Run one season, isolating its Python-level failure from the rest of the grid.

    A single season that raises (e.g. a CVODES ``CV_CONV_FAILURE`` deep in an arm-4
    rollout under an extreme state) must not abort the rest of the grid, so on failure we
    return an ``error`` row with the full traceback. A failed season is a DEPLOYMENT
    FAILURE, not a missing observation: it counts as a gate violation for its arm
    (``BenchmarkResult.failed``) and is excluded only from the net-value stats (a
    crashed season has no honest number). Note the isolation covers exceptions only;
    a worker process that dies outright still aborts the pool (``BrokenProcessPool``).
    """
    try:
        return _run_task(task)
    except Exception as exc:  # deliberately broad: isolate one worker's failure
        msg = f"{type(exc).__name__}: {exc}".splitlines()[0][:200]
        return {
            "arm": task["arm"],
            "label": task["label"],
            "seed": task["seed"],
            "error": msg,
            "traceback": traceback.format_exc(),
        }


def _banded_violation(trace: np.ndarray, step_hours: float, depth: float, min_hours: float) -> bool:
    """True if the exceedance trace stays above ``depth`` for more than ``min_hours``
    CONSECUTIVE hours (the secondary banded gate, the decision record of 2026-07-16)."""
    over = trace > depth
    run = best = 0
    for flag in over:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best * step_hours > min_hours


def _season_rows(
    result: BenchmarkResult,
    banded: dict[str, float],
    advisor_fb: dict[tuple[str, str, int], int | None],
    selection: dict[tuple[str, str, int], float | None] | None = None,
) -> list[dict[str, Any]]:
    """Per-season SCALAR rows (KB-scale, committed with the summary).

    The per-arm aggregates alone cannot support the pre-registered analyses (the
    mismatch-degradation curve, per-slice gates, bootstrap CIs over seasons), so the
    summary carries one scalar row per completed season: net value, online cost,
    solve-success, advisor fallbacks (arm 4), the per-class any-step-violation flag,
    and the banded-gate ingredients (max excursion, degree-hours, banded flag).
    Raw traces stay out.
    """
    rows = []
    for (arm, label, seed), r in sorted(result.seasons.items()):
        rows.append(
            {
                "arm": arm,
                "label": label,
                "seed": seed,
                "net": r.net_economic_value,
                "online_cost_seconds": r.online_cost_seconds,
                "solve_success": r.solve_success,
                "advisor_fallback_steps": advisor_fb.get((arm, label, seed)),
                "rl_selection_rate": (selection or {}).get((arm, label, seed)),
                "violated": {cls: bool(r.exceedances[cls].max() > 0.0) for cls in r.exceedances},
                "violated_banded": {
                    cls: _banded_violation(
                        r.exceedances[cls], result.step_hours, banded["depth"], banded["min_hours"]
                    )
                    for cls in r.exceedances
                },
                "max_excursion": {cls: float(r.exceedances[cls].max()) for cls in r.exceedances},
                "degree_hours": {
                    cls: float(r.exceedances[cls].sum() * result.step_hours)
                    for cls in r.exceedances
                },
            }
        )
    return rows


def _dump_trace(row: dict[str, Any], trace_dir: Path) -> None:
    """Persist one season's full per-class exceedance arrays for later gate
    re-calibration (a new banded/disease threshold can then be scored offline).

    Filename keys the season; content is the raw per-step trace. Error rows carry no
    ``exceedances`` and are skipped. Write to big-file storage, not git (the arrays are
    ~5760 steps x classes)."""
    if "exceedances" not in row:
        return
    name = f"{row['arm']}__{str(row['label']).replace('/', '-')}__s{row['seed']}.json"
    keep = {k: row[k] for k in ("arm", "label", "seed", "exceedances")}
    (trace_dir / name).write_text(json.dumps(keep, default=float))


def _run_all(
    tasks: list[dict[str, Any]], workers: int, progress_path: Path, trace_dir: Path | None = None
) -> list[dict[str, Any]]:
    """Run every season, streaming each completion to an append-only progress file
    (live progress via ``wc -l``, crash-resumability; decision of 2026-07-16).

    If ``trace_dir`` is set, each season's full per-class exceedance arrays are also
    written there (one JSON per season) so the banded/degree-hours gate can be
    RE-CALIBRATED later without re-running the grid. Point it at big-file storage
    (``$GROWMATICS_BIGFILES``), never git. The committed progress/summary stay trace-free."""
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)

    def _stream(row: dict[str, Any], done: int) -> None:
        if trace_dir is not None:
            _dump_trace(row, trace_dir)
        slim = {k: v for k, v in row.items() if k != "exceedances"}
        with progress_path.open("a") as fh:
            fh.write(json.dumps(slim, default=float) + "\n")
        print(
            f"  [{done}/{len(tasks)}] {row['arm']} {row['label']} s{row['seed']}"
            + (" ERROR" if "error" in row else ""),
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_run_task_safe, t) for t in tasks]
            for fut in as_completed(futures):
                rows.append(fut.result())
                _stream(rows[-1], len(rows))
    else:
        for t in tasks:
            rows.append(_run_task_safe(t))
            _stream(rows[-1], len(rows))
    return rows


def _summarize(
    result: BenchmarkResult,
    failed_by_arm: dict[str, int],
    banded: dict[str, float],
    advisor_fb: dict[tuple[str, str, int], int | None],
    selection: dict[tuple[str, str, int], float | None],
) -> dict[str, Any]:
    """Per-arm aggregates plus the Pareto fronts, printed and returned for the summary."""
    summary: dict[str, Any] = {"record": result.record, "arms": {}}
    summary["seasons"] = _season_rows(result, banded, advisor_fb, selection)
    print(f"=== benchmark results ({'SMOKE' if result.record['smoke'] else 'FULL'}) ===")
    for arm in result.arms():
        mean, std = result.net_value_summary(arm)
        gate = result.gate_outcome(arm)
        ss = result.solve_success(arm)
        n_completed = sum(1 for (a, _c, _s) in result.seasons if a == arm)
        underpowered = result.gate_underpowered(arm)
        summary["arms"][arm] = {
            "net_value_mean": mean,
            "net_value_std": std,
            "n_seasons": n_completed,
            "n_failed": failed_by_arm.get(arm, 0),
            "online_cost_seconds": result.online_cost(arm),
            "gate_humidity": gate["humidity"],
            "gate_underpowered": underpowered,
            "solve_success": ss,
        }
        ss_str = f"{ss:.3f}" if ss is not None else "  -  "
        online_ms = result.online_cost(arm) * 1e3
        power = " GATE-UNDERPOWERED" if underpowered else ""
        print(
            f"  {arm:17s} net {mean:+7.3f}+/-{std:5.3f}  online {online_ms:7.1f}ms  "
            f"gate(rh)={gate['humidity']}  solve-success {ss_str}{power}"
        )
    # The contract gates before ranking: the committed Pareto is over gate-passing arms
    # (plus the incumbent as reference). The ungated front stays as a labeled diagnostic.
    summary["pareto"] = {
        **result.pareto(incumbent="incumbent", gate_passers_only=True),
        "gated": True,
    }
    summary["pareto_ungated_diagnostic"] = {
        **result.pareto(incumbent="incumbent", gate_passers_only=False),
        "gated": False,
    }
    return summary


def _report_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the error rows (leaving ``rows`` untouched), printing what failed and why."""
    failures = [r for r in rows if "error" in r]
    if failures:
        print(
            f"WARNING: {len(failures)}/{len(rows)} seasons FAILED. Each counts as a gate "
            "violation for its arm and is excluded from net-value stats only:"
        )
        for r in failures:
            print(f"  {r['arm']:17s} {r['label']} seed={r['seed']}: {r['error']}")
    return failures


def _build_tasks(cfg: Any, common: dict[str, Any]) -> list[dict[str, Any]]:
    years = [int(y) for y in cfg.generalisation.heldout_years]
    mismatches = common.get("mismatch_levels") or [
        float(m) for m in cfg.raw["mismatch"]["sweep_magnitudes"]
    ]
    n_seeds = int(cfg.raw["reporting"]["seeds_per_condition"])
    seeds = list(range(n_seeds))
    # which budget's policy the data-driven arms use (default: the max, i.e. fully trained;
    # override to draw arm-2/4 at a specific budget, e.g. for arm 4's data-efficiency curve).
    max_budget = common.get("policy_budget") or int(cfg.raw["data_efficiency"]["ppo_budgets"][-1])
    tasks: list[dict[str, Any]] = []
    for arm in common["arms"]:
        # per-arm axes: only the physics arms sweep mismatch; only data-driven arms need seeds
        arm_mismatch = mismatches if arm in ("physics_mpc", "physics_informed") else [0.0]
        algo = "sac" if arm == "black_box_sac" else "ppo"
        arm_seeds = [0] if arm == "incumbent" else seeds
        for year in years:
            for m in arm_mismatch:
                for seed in arm_seeds:
                    tasks.append(
                        {
                            "arm": arm,
                            "year": year,
                            "mismatch": m,
                            "seed": seed,
                            "label": f"{year}/m{m:.2f}",
                            "season_days": common["season_days"],
                            "policy_path": _policy_path(
                                common["checkpoint_dir"],
                                seed,
                                max_budget,
                                common["policy_override"],
                                algo,
                            ),
                            "config_path": common["config_path"],
                            "config_dir": common["config_dir"],
                            "weather_dir": common["weather_dir"],
                        }
                    )
    return tasks


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (set to CPU cores)")
    ap.add_argument("--arms", nargs="*", default=list(_ARMS), choices=_ARMS)
    ap.add_argument(
        "--season-days", type=int, default=None, help="override (smoke); default full season"
    )
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument(
        "--policy-override", default=None, help="one policy for all arm-2/4 slots (smoke)"
    )
    ap.add_argument("--out", default="results/benchmark")
    ap.add_argument(
        "--trace-dir",
        default=None,
        help="if set, persist each season's full per-step exceedance arrays here so a new "
        "banded/disease gate can be re-scored WITHOUT re-running the grid. Point at "
        "$GROWMATICS_BIGFILES (big-file storage), never git.",
    )
    ap.add_argument("--limit", type=int, default=None, help="truncate the grid to N tasks (smoke)")
    ap.add_argument(
        "--policy-budget", type=int, default=None, help="budget of arm-2/4 policy (default max)"
    )
    ap.add_argument(
        "--mismatch-levels",
        type=float,
        nargs="*",
        default=None,
        help="override the physics-arm mismatch ladder (R1 finer/structural rerun); "
        "does NOT mutate the canonical config",
    )
    return ap.parse_args()


def main() -> None:
    launch_commit, launch_dirty = _git_commit(), _git_dirty()  # provenance at LAUNCH
    tag_hint = f"{launch_commit}{'-dirty' if launch_dirty else ''}"
    args = _parse_args()

    cfg = load_config(_CONFIG)
    season_days = args.season_days or int(cfg.raw["episode"]["crop_days"])
    common = {
        "arms": args.arms,
        "season_days": season_days,
        "checkpoint_dir": args.checkpoint_dir,
        "policy_override": args.policy_override,
        "policy_budget": args.policy_budget,
        "mismatch_levels": args.mismatch_levels,
        "config_path": str(_CONFIG),
        "config_dir": os.environ["GLG2_CONFIG_DIR"],
        "weather_dir": os.environ["GLG2_WEATHER_DIR"],
    }
    tasks = _build_tasks(cfg, common)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    # Preflight: fail fast (before spawning workers) if a policy the grid needs is missing,
    # with the exact filenames, rather than a cryptic FileNotFound deep in a worker hours in.
    if args.policy_override is None:
        needed = {
            t["policy_path"]
            for t in tasks
            if t["arm"] in ("black_box", "black_box_sac", "physics_informed")
        }
        missing = sorted(p for p in needed if not Path(p).exists())
        if missing:
            raise SystemExit(
                f"missing {len(missing)} policy file(s) -- run train_policies.py first:\n  "
                + "\n  ".join(missing[:_MAX_MISSING_SHOWN])
                + ("\n  ..." if len(missing) > _MAX_MISSING_SHOWN else "")
            )
    print(
        f"benchmark grid: {len(tasks)} seasons, arms {args.arms}, {args.workers} worker(s), "
        f"{season_days}-day seasons",
        flush=True,
    )

    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    rows = _run_all(
        tasks, args.workers, Path(args.out) / f"progress_{tag_hint}.jsonl", trace_dir
    )

    failures = _report_failures(rows)
    failed_by_arm: dict[str, int] = {}
    for r in failures:
        failed_by_arm[r["arm"]] = failed_by_arm.get(r["arm"], 0) + 1

    seasons: dict[tuple[str, str, int], SeasonResult] = {}
    for r in rows:
        if "error" in r:
            continue
        seasons[(r["arm"], r["label"], r["seed"])] = SeasonResult(
            net_economic_value=r["net_economic_value"],
            exceedances={k: np.array(v) for k, v in r["exceedances"].items()},
            online_cost_seconds=r["online_cost_seconds"],
            solve_success=r["solve_success"],
        )
    record = {
        "config_version": cfg.testbed["release"],
        # captured at LAUNCH (main() entry), not at write time: a commit made while a
        # long run is in flight must not relabel the code that actually ran.
        "git_commit": launch_commit,
        "git_dirty": launch_dirty,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "season_days": season_days,
        "smoke": season_days < int(cfg.raw["episode"]["crop_days"]),
        "grid": {"seasons": len(tasks), "arms": args.arms},
        "n_failed_seasons": len(failures),
        # First line of each failure only; the full tracebacks go to failures_<tag>.json.
        "failed_seasons": [{k: v for k, v in r.items() if k != "traceback"} for r in failures],
        "hardware": hardware_stamp(),
    }
    result = BenchmarkResult(
        seasons=seasons,
        record=record,
        gate=cfg.gate,
        soft_classes=("humidity",),
        step_hours=cfg.raw["episode"]["timestep_seconds"] / 3600.0,
        failed=failed_by_arm,
    )

    banded = {k: float(v) for k, v in cfg.raw["gate"]["banded"].items()}
    advisor_fb = {
        (r["arm"], r["label"], r["seed"]): r.get("advisor_fallback_steps")
        for r in rows
        if "error" not in r
    }
    selection = {
        (r["arm"], r["label"], r["seed"]): r.get("rl_selection_rate")
        for r in rows
        if "error" not in r
    }
    summary = _summarize(result, failed_by_arm, banded, advisor_fb, selection)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirty = "-dirty" if record["git_dirty"] else ""
    tag = "smoke" if record["smoke"] else f"{record['git_commit']}{dirty}"
    out_path = out_dir / f"benchmark_{tag}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"wrote committed summary -> {out_path}")
    if failures:
        fail_path = out_dir / f"failures_{tag}.json"
        fail_path.write_text(json.dumps(failures, indent=2))
        print(f"wrote {len(failures)} failure traceback(s) -> {fail_path}")
    (Path(args.out) / f"progress_{tag_hint}.jsonl").unlink(missing_ok=True)  # stream served


if __name__ == "__main__":
    main()
