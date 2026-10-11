#!/usr/bin/env python3
"""Re-run ONE benchmark grid cell (2023 / seed 0 / m=0) with per-step logging, for the
illustrative disturbance-response figure (the benchmark paper, Section 5).

This does not touch the benchmark grid or its numbers: it drives the *validated* per-season
path (``run_benchmark._run_task``) unchanged for four arms on a single (year, seed, mismatch)
cell, and captures two things per season that the committed summary deliberately omits:

  * the per-step RH-exceedance array ``v_t`` (class ``rh``) -- the continuous humidity signal
    whose season sum is the reported degree-hours; and
  * the aligned exogenous weather ``env.weather_data[:, 0]`` (global radiation, W/m^2) and
    ``[:, 1]`` (outdoor temperature, deg C), the disturbance the figure plots.

Raw per-step arrays stay out of the committed summary by repo convention; this cell's arrays
are the source data for one illustrative figure, written next to it and regenerable by re-running
this driver. Expectations for what the traces must show are pre-registered and sealed in
``experiments/expectations-disturbance-figure.md`` (committed before this ran).

Run (env vars as in the run reports in results/benchmark/):
    GLG2_CONFIG_DIR=<clone>/gl_gym/configs/envs GLG2_WEATHER_DIR=<bench_weather> \
    python examples/trace_disturbance_cell.py \
        --checkpoint-dir "<bigfiles>/checkpoints" \
        --out results/benchmark/disturbance-cell
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Reuse the benchmark's own per-season path verbatim (no reimplementation of arm/env wiring).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_benchmark as rp

from greenlight_bench.config import load_config
from greenlight_bench.envs.glg2 import make_tomato_env

YEAR, SEED, MISMATCH = 2023, 0, 0.0
# Fast, policy-loading arm first so any wiring/policy error surfaces in seconds, not after the
# ~25-min physics seasons; the two slow (physics) arms run last.
ARMS = ("black_box", "incumbent", "physics_informed", "physics_mpc")
STEPS = 5760  # 60-day season at 15-min steps (configs/benchmark.yaml)
STEP_HOURS = 0.25
_RADIATION_COL, _OUTDOOR_TEMP_COL = 0, 1  # gl_gym weather_data layout (utils.load_weather_data)


def _capture_weather(cfg: Any, config_dir: str, weather_dir: str) -> dict[str, list[float]]:
    """Build the 2023 season env once and read the aligned exogenous disturbance channels.

    weather_data[t] is exactly the row the plant integrates at step t, so slicing [:STEPS]
    gives a per-step series aligned 1:1 with the RH-exceedance trace, no CSV re-alignment.
    """
    env = make_tomato_env(
        config_dir,
        weather_dir,
        overrides={
            "location": cfg.generalisation.location,
            "start_train_year": YEAR,
            "end_train_year": YEAR,
            "start_train_day": 59,
            "end_train_day": 59,
            "season_length": 60,
        },
    )
    env.reset(seed=SEED)
    w = np.asarray(env.weather_data, dtype=float)[:STEPS]
    return {
        "radiation_w_m2": w[:, _RADIATION_COL].tolist(),
        "outdoor_temp_c": w[:, _OUTDOOR_TEMP_COL].tolist(),
    }


def _run_arm(arm: str, checkpoint_dir: str, config_dir: str, weather_dir: str) -> dict[str, Any]:
    """Drive one arm through run_benchmark._run_task on the fixed cell; keep the rh trace."""
    max_budget = 2_000_000  # fully trained policy (data_efficiency.ppo_budgets[-1])
    algo = "sac" if arm == "black_box_sac" else "ppo"
    task = {
        "arm": arm,
        "year": YEAR,
        "mismatch": MISMATCH,
        "seed": SEED,
        "label": f"{YEAR}/m{MISMATCH:.2f}",
        "season_days": 60,
        "policy_path": rp._policy_path(checkpoint_dir, SEED, max_budget, None, algo),
        "config_path": str(rp._CONFIG),
        "config_dir": config_dir,
        "weather_dir": weather_dir,
    }
    row = rp._run_task(task)
    if "error" in row:
        raise RuntimeError(f"{arm} season failed: {row['error']}")
    # RH class key is "humidity" (greenlight_bench.envs.glg2.VIOLATION_FIELDS); per-step value
    # is the RH-point excursion above the ceiling, so sum * step_hours is the reported deg-hours.
    rh = [float(v) for v in row["exceedances"]["humidity"]]
    return {
        "arm": arm,
        "net_economic_value": float(row["net_economic_value"]),
        "rh_exceedance": rh,
        "rh_degree_hours": float(np.sum(rh) * STEP_HOURS),  # sanity vs the reported season value
        "rl_selection_rate": row.get("rl_selection_rate"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--out", default="results/benchmark/disturbance-cell")
    args = ap.parse_args()

    import os

    config_dir, weather_dir = os.environ["GLG2_CONFIG_DIR"], os.environ["GLG2_WEATHER_DIR"]
    cfg = load_config(rp._CONFIG)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"cell_{YEAR}_s{SEED}_m{MISMATCH:.2f}.json"

    print(f"[cell] weather capture for {YEAR}...", flush=True)
    weather = _capture_weather(cfg, config_dir, weather_dir)

    result: dict[str, Any] = {
        "meta": {
            "year": YEAR,
            "seed": SEED,
            "mismatch": MISMATCH,
            "steps": STEPS,
            "step_hours": STEP_HOURS,
            "rh_ceiling": cfg.raw["constraints"]["rh_percent"][1],
            "note": "one illustrative grid cell; drives run_benchmark._run_task unchanged",
        },
        "weather": weather,
        "arms": {},
    }
    # Write partial results after each arm (checkpoint long jobs: the two physics arms are slow).
    for arm in ARMS:
        t0 = time.perf_counter()
        print(f"[cell] running {arm}...", flush=True)
        result["arms"][arm] = _run_arm(arm, args.checkpoint_dir, config_dir, weather_dir)
        dh = result["arms"][arm]["rh_degree_hours"]
        net = result["arms"][arm]["net_economic_value"]
        print(
            f"[cell] {arm}: net={net:+.2f} rh_degree_hours={dh:.0f} "
            f"({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
        out_path.write_text(json.dumps(result))
    print(f"[cell] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
