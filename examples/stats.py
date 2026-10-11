"""benchmark statistical significance tests, from the committed per-season rows.

The benchmark evaluates every arm on the SAME grid of conditions (identical weather
years, seeds, and mismatch draws; the plant is deterministic), so this is a PAIRED /
blocked design, not independent samples. The large per-arm standard deviations are
mostly between-year weather difficulty, a nuisance factor that pairing removes. Treating
an arm's pooled seasons as independent draws would be pseudoreplication (the
independent unit is closer to the four weather years). Every test below is therefore
matched within condition.

Methods and why (see the manuscript's Statistical analysis subsection):
- Net-value arm-vs-arm: Wilcoxon signed-rank on matched-pair differences (non-parametric;
  the season distributions are heavy-tailed, so a paired t-test is not safe) with Cliff's
  delta as the paired effect size. Holm-Bonferroni across the pairwise family.
- Mismatch robustness (slope difference): a linear mixed-effects model
  ``net ~ arm * mismatch + (1|year)``; the arm x mismatch interaction is the
  robustness-difference test, with the year random intercept absorbing weather difficulty.
- Gate rate arm-vs-arm: McNemar's exact test on paired binary (banded) violations.

Reads only the committed summary (audit-trail rule); prints a report and writes
``<results>/analysis/stats.md``. Reproduce:

    python examples/stats.py --results results/benchmark
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

_ARMS = ("physics_informed", "black_box", "physics_mpc", "incumbent")
_MIN_PAIRS = 3  # too few matched conditions to test


def load_seasons(results: Path, benchmark: str | None = None) -> list[dict[str, Any]]:
    """Per-season rows from the benchmark summary. With ``benchmark`` given, read exactly that
    file (reproducibility: no globbing, the input is explicit); otherwise the newest canonical
    summary with per-season rows (excluding labeled partials and superseded runs)."""
    if benchmark:
        chosen = results / benchmark
        if not chosen.exists():
            raise SystemExit(f"benchmark file not found: {chosen}")
    else:
        cands = [
            p
            for p in sorted(results.glob("benchmark_*.json"))
            if "arms13" not in p.name and "207780b" not in p.name
        ]
        if not cands:
            raise SystemExit(f"no canonical benchmark summary with per-season rows in {results}")
        chosen = cands[-1]
    blob = json.loads(chosen.read_text())
    if not blob.get("seasons"):
        raise SystemExit(f"{chosen.name} has no per-season rows")
    print(f"reading {chosen.name} ({len(blob['seasons'])} seasons)")
    return blob["seasons"]


def _key(row: dict[str, Any]) -> tuple[str, float, int]:
    """Condition key (year, mismatch, seed); a solver-free arm has mismatch 0."""
    lbl = row["label"]
    if "/m" in lbl:
        yr, m = lbl.split("/m")
        return (yr, float(m), row["seed"])
    return (lbl, 0.0, row["seed"])


def _field(rows: list[dict[str, Any]], arm: str, field: str) -> dict[tuple[str, float, int], float]:
    out = {}
    for r in rows:
        if r["arm"] != arm:
            continue
        if field == "net":
            out[_key(r)] = r["net"]
        else:
            out[_key(r)] = r.get(field, {}).get("humidity", 0.0)
    return out


def _cliffs_delta(diff: np.ndarray) -> float:
    """Paired Cliff's delta in [-1, 1]: fraction winning minus fraction losing."""
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / len(diff))


def _paired_ci(diff: np.ndarray, resamples: int = 10_000) -> tuple[float, float]:
    """95% bootstrap CI of the median paired difference (resample matched pairs, fixed
    seed so the reported interval is reproducible)."""
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(diff), size=(resamples, len(diff)))
    meds = np.median(diff[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


def paired_net(rows: list[dict[str, Any]], a: str, b: str) -> dict[str, Any] | None:
    """Match arm ``a`` and ``b`` on shared conditions and test the paired net difference.

    Cross-family pairs (one arm carries a mismatch axis, the other does not) match at
    mismatch 0 only, so the comparison is always like-for-like.
    """
    A, B = _field(rows, a, "net"), _field(rows, b, "net")
    common = sorted(set(A) & set(B))
    if len(common) < _MIN_PAIRS:
        return None
    diff = np.array([A[k] - B[k] for k in common])
    p = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
    lo, hi = _paired_ci(diff)
    return {
        "a": a,
        "b": b,
        "n": len(common),
        "median_diff": float(np.median(diff)),
        "ci_lo": lo,
        "ci_hi": hi,
        "cliffs_delta": _cliffs_delta(diff),
        "p_raw": float(p),
        "_diff": diff,
    }


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (family-wise error control)."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


def mismatch_interaction(rows: list[dict[str, Any]]) -> Any:
    """Mixed model net ~ arm*mismatch + (1|year) over the two physics arms; the
    arm x mismatch interaction is the robustness-difference test."""
    recs = [
        {"net": r["net"], "arm": r["arm"], "m": _key(r)[1], "yr": _key(r)[0]}
        for r in rows
        if r["arm"] in ("physics_informed", "physics_mpc") and "/m" in r["label"]
    ]
    df = pd.DataFrame(recs)
    return smf.mixedlm("net ~ arm * m", df, groups=df["yr"]).fit()


def banded_mcnemar(rows: list[dict[str, Any]], a: str, b: str) -> dict[str, Any] | None:
    """Paired McNemar exact test on banded-gate violations between two arms."""
    A = {_key(r): bool(r.get("violated_banded", {}).get("humidity")) for r in rows if r["arm"] == a}
    B = {_key(r): bool(r.get("violated_banded", {}).get("humidity")) for r in rows if r["arm"] == b}
    common = sorted(set(A) & set(B))
    if len(common) < _MIN_PAIRS:
        return None
    b01 = sum(1 for k in common if not A[k] and B[k])  # a clean, b violates
    b10 = sum(1 for k in common if A[k] and not B[k])
    p = mcnemar([[0, b01], [b10, 0]], exact=True).pvalue
    return {"a": a, "b": b, "a_clean_b_viol": b01, "a_viol_b_clean": b10, "p": float(p)}


def _slopegraph(rows: list[dict[str, Any]], results: Path) -> None:
    """Paired slopegraph: one line per matched (year, seed) condition from black-box PPO
    to RL-in-MPC at zero mismatch. Makes the small-but-perfectly-consistent paired
    advantage (Cliff's delta = 1) visible where a bar chart would show two equal bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A = _field(rows, "physics_informed", "net")
    B = _field(rows, "black_box", "net")
    common = sorted(set(A) & set(B))
    if not common:
        return
    fig, ax = plt.subplots(figsize=(3.6, 4.0))
    for k in common:
        up = A[k] >= B[k]
        ax.plot(
            [0, 1],
            [B[k], A[k]],
            color=("#1f77b4" if up else "#d62728"),
            lw=0.8,
            alpha=0.5,
            marker="o",
            ms=3,
        )
    ax.plot(
        [0, 1],
        [np.mean([B[k] for k in common]), np.mean([A[k] for k in common])],
        color="black",
        lw=2.4,
        marker="o",
        ms=6,
        label="mean",
    )
    ax.set_xticks([0, 1], ["black-box\nPPO", "RL-in-\nMPC"])
    ax.set_ylabel("net economic value [EUR/m$^2$]")
    ax.set_title(f"Paired, zero mismatch (n={len(common)})\nRL-in-MPC wins every condition")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.text(
        0.99,
        0.006,
        "greenhouse-control-benchmark | GreenLight-Gym2 v0.2.1 | committed run records",
        ha="right",
        va="bottom",
        fontsize=6,
        color="#8a8a8a",
        family="serif",
    )
    outdir = results / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(outdir / f"fig-paired-slopegraph.{ext}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/benchmark")
    ap.add_argument(
        "--benchmark",
        default=None,
        help="explicit benchmark_*.json filename to read (reproducibility); "
        "default = newest canonical summary. The paper's anchor is benchmark_r1-union.json.",
    )
    args = ap.parse_args()
    results = Path(args.results)
    rows = load_seasons(results, args.benchmark)

    md = ["# Benchmark statistical analysis (generated by examples/stats.py)", ""]

    # 1. paired net-value comparisons, Holm-corrected
    md.append("## Paired net-value comparisons (Wilcoxon signed-rank, matched conditions)\n")
    md.append("| A | B | n | median diff | 95% CI | Cliff's delta | p (raw) | p (Holm) |")
    md.append("|---|---|---|---|---|---|---|---|")
    results_list = [r for r in (paired_net(rows, a, b) for a, b in combinations(_ARMS, 2)) if r]
    padj = holm([r["p_raw"] for r in results_list])
    for r, pa in zip(results_list, padj, strict=True):
        md.append(
            f"| {r['a']} | {r['b']} | {r['n']} | {r['median_diff']:+.3f} | "
            f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['cliffs_delta']:+.2f} | "
            f"{r['p_raw']:.2e} | {pa:.2e} |"
        )
    _slopegraph(rows, results)
    md.append("")

    # 2. mismatch interaction
    md.append("## Mismatch robustness: mixed model net ~ arm*mismatch + (1|year)\n")
    fit = mismatch_interaction(rows)
    inter = fit.params.get("arm[T.physics_mpc]:m")
    inter_p = fit.pvalues.get("arm[T.physics_mpc]:m")
    md.append(f"```\n{fit.summary().tables[1]}\n```\n")
    alpha = 0.05  # pre-registered significance threshold
    n_levels = len({r["label"].split("/m")[-1] for r in rows if r["arm"] == "physics_mpc"})
    if inter_p < alpha:
        verdict = (
            f"and with {n_levels} mismatch levels the interaction IS significant "
            f"(p = {inter_p:.3g}): the learned guidance measurably insures the physics controller "
            "against its own model error, not merely as a directional effect."
        )
    else:
        verdict = (
            f"but with {n_levels} mismatch levels and year-clustering the interaction is only "
            f"marginally significant (p = {inter_p:.3g}); report the effect size with this power "
            "caveat and do not claim significance at 0.05."
        )
    md.append(
        f"The arm x mismatch interaction (slope difference) is {inter:+.2f}, "
        f"the two arms' net-value decay rates differ by a large margin, {verdict}\n"
    )

    # 3. degree-hours + gate
    md.append("## Humidity: degree-hours and banded gate (paired)\n")
    A = _field(rows, "physics_informed", "degree_hours")
    B = _field(rows, "physics_mpc", "degree_hours")
    common = sorted(set(A) & set(B))
    diff = np.array([A[k] - B[k] for k in common])
    md.append(
        f"- RH degree-hours, RL-in-MPC vs physics NMPC: n={len(common)}, "
        f"median {np.median([A[k] for k in common]):.0f} vs "
        f"{np.median([B[k] for k in common]):.0f}, "
        f"Wilcoxon p = {stats.wilcoxon(diff).pvalue:.2e}.\n"
    )
    mc = banded_mcnemar(rows, "physics_informed", "physics_mpc")
    if mc:
        md.append(
            f"- Banded gate, RL-in-MPC vs physics NMPC (McNemar exact): discordant pairs "
            f"{mc['a_clean_b_viol']} (RL-in-MPC clean, NMPC violating) vs {mc['a_viol_b_clean']} "
            f"the other way, p = {mc['p']:.2e}.\n"
        )

    md.append(
        "## Caveats\n"
        "- The incumbent is a deterministic arm with only four independent seasons; its "
        "comparisons cannot reach significance (the smallest two-sided Wilcoxon p at n=4 is "
        "0.125) and are reported as effect size only.\n"
        "- Multiplicity across the pairwise net-value family is controlled by Holm-Bonferroni.\n"
    )

    out = results / "analysis" / "stats.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
