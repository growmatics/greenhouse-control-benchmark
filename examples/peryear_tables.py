"""Regenerate the per-year tables in the manuscript from the committed benchmark anchor.

Reproduces, from ``benchmark_r1-union.json`` (the R1 anchor), the two per-year tables:
  * ``tab:peryear``          -- net by weather year at m=0, all five arms (+ leave-one-year-out).
  * ``tab:peryear-mismatch`` -- net by weather year x mismatch level, the two physics arms.
Prints them as Markdown so a human can diff the manuscript's hand-set LaTeX against this output.
No simulation; reads only the committed per-season rows (the audit-trail rule).

    python examples/peryear_tables.py \
        --benchmark results/benchmark/benchmark_r1-union.json
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

YEARS = ["2022", "2023", "2024", "2025"]
ARMS = ["incumbent", "black_box", "black_box_sac", "physics_mpc", "physics_informed"]
LABELS = {
    "incumbent": "incumbent",
    "black_box": "PPO",
    "black_box_sac": "SAC",
    "physics_mpc": "NMPC",
    "physics_informed": "RL-in-MPC",
}


def _cell(vals: list[float]) -> str:
    return f"{statistics.mean(vals):+.2f}" if vals else "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    args = ap.parse_args()
    seasons = json.loads(Path(args.benchmark).read_text())["seasons"]

    def mismatch(s: dict) -> str:
        return s["label"].split("/m")[-1]

    def year(s: dict) -> str:
        return s["label"].split("/")[0]

    # tab:peryear -- m=0, all arms
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in seasons:
        if mismatch(s) == "0.00":
            by[s["arm"]][year(s)].append(s["net"])
    print("## tab:peryear (net by year, m=0)\n")
    print("| arm | " + " | ".join(YEARS) + " | mean | spread |")
    print("|" + "---|" * (len(YEARS) + 3))
    for a in ARMS:
        cells = [by[a][y] for y in YEARS]
        present = [statistics.mean(c) for c in cells if c]
        mean = f"{statistics.mean(present):+.2f}" if present else "-"
        spread = f"{max(present) - min(present):.1f}" if len(present) > 1 else "-"
        print(f"| {LABELS[a]} | " + " | ".join(_cell(c) for c in cells) + f" | {mean} | {spread} |")

    # leave-one-year-out on PPO
    ppo = {y: statistics.mean(by["black_box"][y]) for y in YEARS if by["black_box"][y]}
    print("\n## PPO leave-one-year-out (drop each year -> mean of the other three)\n")
    for drop in YEARS:
        keep = [ppo[y] for y in YEARS if y != drop and y in ppo]
        print(f"- drop {drop}: {statistics.mean(keep):+.2f}")

    # tab:peryear-mismatch -- physics arms x {0, 0.10, 0.20}
    cell = collections.defaultdict(list)
    for s in seasons:
        cell[(s["arm"], mismatch(s), year(s))].append(s["net"])
    print("\n## tab:peryear-mismatch (physics arms, net by year x mismatch)\n")
    print("| arm | m | " + " | ".join(YEARS) + " |")
    print("|" + "---|" * (len(YEARS) + 2))
    for a in ("physics_mpc", "physics_informed"):
        for m in ("0.00", "0.10", "0.20"):
            row = [_cell(cell[(a, m, y)]) for y in YEARS]
            print(f"| {LABELS[a]} | {m} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
