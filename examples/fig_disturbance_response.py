#!/usr/bin/env python3
"""Illustrative disturbance-response figure for the benchmark paper (Section 5).

Reads the single traced cell written by ``trace_disturbance_cell.py`` (2023 / seed 0 / m=0)
and draws two stacked panels over a window selected by the *pre-registered* rule
(``experiments/expectations-disturbance-figure.md``): the exogenous disturbance on top
(radiation + outdoor temperature) and each arm's per-step RH excursion below. It visualises,
as a time trace, the humidity split already reported as season degree-hours; it adds no new
number. Window is fixed on the weather signal ONLY, before any arm's response is inspected.

    python examples/fig_disturbance_response.py \
        --cell results/benchmark/disturbance-cell/cell_2023_s0_m0.json \
        --outdir results/benchmark/disturbance-cell

It also prints the evaluation metrics that decide whether the figure fits the sealed story
(within-window degree-hours per arm; the pre-front vs post-front gap that tests E4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

STEP_HOURS = 0.25
WINDOW_HOURS = 48
_DROP_SPAN_H = 6  # rolling span for the outdoor-temperature drop that selects the window
_DROP_THRESHOLD_C = 4.0  # a "front"; below this we fall back to the steepest radiation edge

# One consistent per-arm colour/label scheme; physics arms warm, learned arms cool.
_ARM_STYLE = {
    "physics_mpc": {"label": "physics-only NMPC", "color": "#d62728", "lw": 2.0, "z": 5},
    "incumbent": {"label": "rule-based incumbent", "color": "#8c564b", "lw": 1.4, "z": 3},
    "black_box": {"label": "black-box PPO", "color": "#1f77b4", "lw": 1.6, "z": 4},
    "physics_informed": {"label": "RL-in-MPC (hybrid)", "color": "#2ca02c", "lw": 1.6, "z": 4},
}
_PLOT_ORDER = ("physics_mpc", "incumbent", "black_box", "physics_informed")


def _provenance_footer(fig: Any, detail: str) -> None:
    fig.text(
        0.99,
        0.006,
        f"greenhouse-control-benchmark | GreenLight-Gym2 v0.2.1 | {detail}",
        ha="right",
        va="bottom",
        fontsize=6,
        color="#8a8a8a",
        family="serif",
    )


def _select_window(tout: np.ndarray, radiation: np.ndarray) -> tuple[int, int, str]:
    """Pre-registered window rule: 48 h centred on the sharpest 6 h outdoor-temperature drop;
    fall back to the steepest radiation night->day edge if no drop exceeds the front threshold.
    Fixed on the exogenous signal only. Returns (start_step, end_step, criterion)."""
    span = int(_DROP_SPAN_H / STEP_HOURS)
    half = int(WINDOW_HOURS / STEP_HOURS) // 2
    drop = tout[:-span] - tout[span:]  # positive where temperature falls over the span
    i_drop = int(np.argmax(drop))
    if drop[i_drop] >= _DROP_THRESHOLD_C:
        centre, criterion = i_drop + span // 2, f"outdoor-temp drop {drop[i_drop]:.1f}C/6h"
    else:
        rise = radiation[span:] - radiation[:-span]  # steepest radiation increase
        i_rise = int(np.argmax(rise))
        centre, criterion = i_rise + span // 2, f"radiation edge {rise[i_rise]:.0f}W/6h (no front)"
    start = max(0, centre - half)
    end = min(len(tout), start + 2 * half)
    return start, end, criterion


def _degree_hours(v: np.ndarray) -> float:
    return float(np.sum(v) * STEP_HOURS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cell = json.loads(Path(args.cell).read_text())
    radiation = np.asarray(cell["weather"]["radiation_w_m2"], dtype=float)
    tout = np.asarray(cell["weather"]["outdoor_temp_c"], dtype=float)
    arms = cell["arms"]

    s, e, criterion = _select_window(tout, radiation)
    hours = (np.arange(s, e) - s) * STEP_HOURS
    print(
        f"[fig] window steps [{s},{e}) = season hours [{s * STEP_HOURS:.0f},{e * STEP_HOURS:.0f}); "
        f"criterion: {criterion}"
    )

    # Evaluation metrics for the sealed gate (printed, not baked into the figure).
    mid = (e - s) // 2
    print("[fig] within-window RH degree-hours + pre/post-front split (tests E1-E4):")
    for arm in _PLOT_ORDER:
        v = np.asarray(arms[arm]["rh_exceedance"], dtype=float)[s:e]
        pre, post = _degree_hours(v[:mid]), _degree_hours(v[mid:])
        print(
            f"    {arm:18s} window_dh={_degree_hours(v):6.1f}  pre={pre:6.1f}  post={post:6.1f}  "
            f"season_dh={arms[arm]['rh_degree_hours']:.0f}"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axw, axr) = plt.subplots(2, 1, figsize=(6.2, 4.8), sharex=True, height_ratios=[1.0, 1.3])

    # Top: the disturbance both controller classes are exposed to.
    axw.plot(hours, radiation[s:e], color="#e6a817", lw=1.3, label="global radiation")
    axw.set_ylabel("radiation [W m$^{-2}$]", color="#b8860b")
    axw.tick_params(axis="y", labelcolor="#b8860b")
    axt = axw.twinx()
    axt.plot(hours, tout[s:e], color="#555555", lw=1.3, ls="--", label="outdoor temp")
    axt.set_ylabel("outdoor temp [$^\\circ$C]", color="#555555")
    axt.tick_params(axis="y", labelcolor="#555555")
    axw.set_title("A near-step disturbance every arm sees", fontsize=9, loc="left")

    # Bottom: each arm's RH excursion above the 85% ceiling through the same window.
    for arm in _PLOT_ORDER:
        st = _ARM_STYLE[arm]
        v = np.asarray(arms[arm]["rh_exceedance"], dtype=float)[s:e]
        axr.plot(hours, v, color=st["color"], lw=st["lw"], label=st["label"], zorder=st["z"])
    axr.axhline(0.0, color="black", lw=0.6, alpha=0.4)
    axr.set_ylabel("RH excursion above 85% [pts]")
    axr.set_xlabel("time within window [h]")
    axr.legend(fontsize=7, ncol=2, loc="upper left")
    axr.set_title(
        "Physics-only NMPC and the incumbent breach grossly; the learned arms track the bound",
        fontsize=9,
        loc="left",
    )

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _provenance_footer(fig, "cell 2023/seed0/m0, one representative season")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(outdir / f"fig-disturbance-response.{ext}")
    plt.close(fig)
    print(f"[fig] wrote {outdir / 'fig-disturbance-response.svg'} (+ pdf)")


if __name__ == "__main__":
    main()
