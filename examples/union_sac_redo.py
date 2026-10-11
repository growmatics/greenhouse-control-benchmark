"""Build the R1 anchor benchmark by unioning the main grid run with the SAC-arm redo.

Why this exists (decision 2026-07-23): the main R1 `run_benchmark` invocation ran the SAC
arm (`black_box_sac`) with STALE seeds-0/1 checkpoints (a `cp -n` no-clobber staging slip), so
its 20 SAC seasons are a mix of as-shipped and tuned and its arm mean is wrong (+2.56 vs the true
+3.21). The fix (option A) re-ran ONLY the SAC arm with the corrected checkpoints. This script
produces the clean anchor summary: every non-SAC season from the main run, plus the 20 SAC seasons
from the redo. All other arms are byte-identical to the main run (verified: the union changes only
the `black_box_sac` rows).

Reproducible: reads two committed JSON summaries, writes one. No simulation. The downstream
analysis (`analyze.py`, `stats.py`) recomputes every paper number/figure from the
`seasons` list of the OUTPUT of this script.

    python examples/union_sac_redo.py \
        --main    results/benchmark/benchmark_29d9eb6.json \
        --sac-redo results/benchmark/sac-redo/benchmark_24105d8-dirty.json \
        --out     results/benchmark/benchmark_r1-union.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ARM = "black_box_sac"
N_SAC = 20  # SAC is m=0 only: 4 held-out years x 5 seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--sac-redo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    main_blob = json.loads(Path(args.main).read_text())
    redo_blob = json.loads(Path(args.sac_redo).read_text())

    redo_sac = [s for s in redo_blob["seasons"] if s["arm"] == ARM]
    if len(redo_sac) != N_SAC:
        raise SystemExit(f"expected {N_SAC} redo SAC seasons, got {len(redo_sac)}")

    # swap: keep every non-SAC season from main, append the redo SAC seasons
    kept = [s for s in main_blob["seasons"] if s["arm"] != ARM]
    dropped = len(main_blob["seasons"]) - len(kept)
    if dropped != N_SAC:
        raise SystemExit(f"expected to drop {N_SAC} buggy SAC seasons, dropped {dropped}")
    union_seasons = kept + redo_sac

    # recompute ONLY the SAC arm summary from the redo seasons (its gate outcome is unchanged:
    # n=20 < the N>=60 floor so it stays gate_underpowered, and it saturates the humidity gate
    # like every arm; only net and online cost move).
    nets = [s["net"] for s in redo_sac]
    costs = [s["online_cost_seconds"] for s in redo_sac]
    sac = dict(main_blob["arms"][ARM])
    sac["net_value_mean"] = statistics.mean(nets)
    sac["net_value_std"] = statistics.pstdev(nets)
    sac["online_cost_seconds"] = statistics.mean(costs)
    main_blob["arms"][ARM] = sac

    # mirror the change into the pareto snapshot if it carries a per-arm SAC point
    for block in ("pareto", "pareto_ungated_diagnostic"):
        pts = main_blob.get(block)
        if isinstance(pts, list):
            for p in pts:
                if p.get("arm") == ARM:
                    p["net_value_mean"] = sac["net_value_mean"]
                    p["online_cost_seconds"] = sac["online_cost_seconds"]

    main_blob["seasons"] = union_seasons
    rec = main_blob["record"]
    rec["git_dirty"] = True
    rec["union_note"] = (
        f"R1 anchor: non-SAC seasons from main run {rec.get('git_commit')}; "
        f"{ARM} 20 seasons from the SAC-arm redo (corrected checkpoints, decision 2026-07-23)."
    )

    Path(args.out).write_text(json.dumps(main_blob, indent=2, default=float))
    print(
        f"wrote {args.out}: {len(union_seasons)} seasons; "
        f"{ARM} net {sac['net_value_mean']:+.3f} +/- {sac['net_value_std']:.3f}"
    )


if __name__ == "__main__":
    main()
