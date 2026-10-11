"""benchmark analytics: turn the committed JSON summaries into the paper's numbers/figures.

Reads ONLY committed artifacts (the audit-trail rule: every quoted number traces to a
file in ``results/benchmark/``) and writes to ``<results>/analysis/``:

- ``analysis.md``            -- the stats tables (data-efficiency band, per-arm, mismatch)
- ``fig-data-efficiency.*``  -- held-out net vs training budget, per-seed + mean/CI band,
                                with zero-training reference lines once the benchmark
                                summary provides incumbent / physics-MPC nets
- ``fig-mismatch.*``         -- physics-arm net vs mismatch level (advantage decay),
                                requires a summary with per-season rows
- ``fig-pareto.*``           -- net vs online cost per arm, gated front highlighted

Each figure saves as SVG (the tracked source, per ``docs/figure-standards.md``) plus a
PDF for direct viewing. Degrades gracefully: sections whose inputs are missing are
skipped with a note, so it can run mid-campaign on partial data.

    python examples/analyze.py --results results/benchmark
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

_BOOTSTRAP_RESAMPLES = 10_000
_CI = (2.5, 97.5)  # percentile bootstrap, 95%
_RNG_SEED = 0  # bootstrap resampling seed: fixed so the tables are reproducible
_EPSILON = 0.05  # pre-registered gate threshold (canonical: configs/benchmark.yaml)


def _provenance_footer(fig: Any, detail: str) -> None:
    """The series signature: every figure carries the provenance that regenerates it."""
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


def _bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    """Percentile-bootstrap CI of the mean (honest but WIDE at n~5; report n alongside)."""
    rng = np.random.default_rng(_RNG_SEED)
    idx = rng.integers(0, len(values), size=(_BOOTSTRAP_RESAMPLES, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, _CI)
    return float(lo), float(hi)


def load_curves(results: Path) -> dict[int, dict[int, float]]:
    """{seed: {budget: heldout_net}} for the PPO arm, from ``data_efficiency_s*.json`` at the
    results root. Skips ``data_efficiency_sac_s*.json`` (the as-shipped SAC curves): the glob would
    otherwise match them and, sharing seed indices 0/1, silently overwrite PPO seeds 0/1."""
    curves: dict[int, dict[int, float]] = {}
    for path in sorted(results.glob("data_efficiency_s*.json")):
        if "sac" in path.name:
            continue
        blob = json.loads(path.read_text())
        curve: dict[int, float] = {}
        for budget, point in blob["curve"].items():
            net = point["heldout_net"] if isinstance(point, dict) else float(point)
            curve[int(budget)] = float(net)
        curves[int(blob["seed"])] = curve
    return curves


def load_sac_curves(results: Path) -> dict[int, dict[int, float]]:
    """{seed: {budget: heldout_net}} for the tuned SAC arm, from the confirmation run committed
    under ``confirm/t19/`` (the search-winner config, 5 seeds). Empty if absent (figure degrades
    to PPO-only)."""
    curves: dict[int, dict[int, float]] = {}
    for path in sorted(results.glob("confirm/t19/data_efficiency_sac_s*.json")):
        blob = json.loads(path.read_text())
        curve = {int(b): float(p["heldout_net"]) for b, p in blob["curve"].items()}
        curves[int(blob["seed"])] = curve
    return curves


def load_benchmark(results: Path, benchmark: str | None = None) -> list[dict[str, Any]]:
    """Benchmark summaries to analyse. With ``benchmark`` given, only that file (reproducibility:
    the paper's figures come from one explicit anchor); otherwise every summary present."""
    if benchmark:
        chosen = results / benchmark
        if not chosen.exists():
            raise SystemExit(f"benchmark file not found: {chosen}")
        paths = [chosen]
    else:
        paths = sorted(results.glob("benchmark_*.json"))
    out = []
    for path in paths:
        blob = json.loads(path.read_text())
        blob["_file"] = path.name
        out.append(blob)
    return out


def curve_table(curves: dict[int, dict[int, float]]) -> tuple[str, dict[int, dict[str, float]]]:
    budgets = sorted({b for c in curves.values() for b in c})
    rows = ["| budget | n seeds | mean net | sd | 95% CI (bootstrap) |", "|---|---|---|---|---|"]
    stats: dict[int, dict[str, float]] = {}
    for b in budgets:
        vals = np.array([c[b] for c in curves.values() if b in c])
        lo, hi = _bootstrap_ci(vals) if len(vals) > 1 else (float("nan"), float("nan"))
        stats[b] = {
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "lo": lo,
            "hi": hi,
            "n": len(vals),
        }
        rows.append(
            f"| {b:,} | {len(vals)} | {vals.mean():+.3f} | {stats[b]['sd']:.3f} | "
            f"[{lo:+.3f}, {hi:+.3f}] |"
        )
    return "\n".join(rows), stats


def fig_data_efficiency(
    curves: dict[int, dict[int, float]],
    stats: dict[int, dict[str, float]],
    refs: dict[str, float],
    outdir: Path,
    sac_curves: dict[int, dict[int, float]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for _seed, curve in sorted(curves.items()):
        b = sorted(curve)
        ax.plot(b, [curve[x] for x in b], lw=0.8, alpha=0.45, color="tab:blue")
    budgets = sorted(stats)
    ax.plot(
        budgets,
        [stats[b]["mean"] for b in budgets],
        lw=2.2,
        color="tab:blue",
        marker="o",
        ms=4,
        label="black-box PPO (mean of seeds)",
    )
    ax.fill_between(
        budgets,
        [stats[b]["lo"] for b in budgets],
        [stats[b]["hi"] for b in budgets],
        alpha=0.18,
        color="tab:blue",
        label="95% bootstrap CI",
    )
    if sac_curves:
        sb = sorted({b for c in sac_curves.values() for b in c})
        sm = [float(np.mean([c[b] for c in sac_curves.values() if b in c])) for b in sb]
        ax.plot(
            sb,
            sm,
            lw=2.0,
            color="tab:orange",
            marker="s",
            ms=4,
            ls="--",
            label="black-box SAC (mean of seeds)",
        )
    ref_styles = {
        "incumbent": ("tab:gray", "--"),
        "physics_mpc": ("tab:green", "-."),
        "physics_informed": ("tab:red", ":"),
    }
    xs = np.array(budgets, dtype=float)
    ys = np.array([stats[b]["mean"] for b in budgets])
    for arm, net in refs.items():
        color, ls = ref_styles.get(arm, ("black", "--"))
        ax.axhline(net, color=color, ls=ls, lw=1.4, label=f"{arm} (no training)")
        # H3 annotation: budget where the mean curve first crosses this reference
        above = np.nonzero(ys >= net)[0]
        if len(above) and above[0] > 0:
            i = above[0]
            frac = (net - ys[i - 1]) / (ys[i] - ys[i - 1])
            x_cross = float(np.exp(np.log(xs[i - 1]) + frac * (np.log(xs[i]) - np.log(xs[i - 1]))))
            ax.axvline(x_cross, color=color, ls=":", lw=0.9, alpha=0.6)
            ax.annotate(
                f"{x_cross / 1e3:.0f}k",
                (x_cross, net),
                textcoords="offset points",
                xytext=(3, -11),
                fontsize=7.5,
                color=color,
            )
    # H2 annotation: the zero-training head start vs PURE physics (physics_mpc), the only
    # genuinely training-free model-based arm, at the smallest budget.
    if "physics_mpc" in refs:
        best_ref = refs["physics_mpc"]
        ax.annotate(
            "",
            xy=(xs[0], best_ref),
            xytext=(xs[0], ys[0]),
            arrowprops={"arrowstyle": "<->", "color": "black", "lw": 1.0, "alpha": 0.7},
        )
        ax.annotate(
            f"physics head start\n{best_ref - ys[0]:+.1f} EUR/m$^2$",
            (xs[0], (best_ref + ys[0]) / 2),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize=7.5,
            alpha=0.85,
        )
    ax.set_xscale("log")
    ax.set_xlabel("training budget [env steps]")
    ax.set_ylabel("held-out net economic value [EUR/m$^2$]")
    ax.axhline(0.0, color="black", lw=0.6, alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _provenance_footer(fig, f"{len(curves)} seed curves, committed run records")
    for ext in ("svg", "pdf"):
        fig.savefig(outdir / f"fig-data-efficiency.{ext}")
    plt.close(fig)


def mismatch_table_and_fig(seasons: list[dict[str, Any]], outdir: Path) -> str | None:
    """Advantage-decay: per (arm, mismatch) net over the physics arms' season rows."""
    rows = [r for r in seasons if "/m" in r["label"]]
    if not rows:
        return None
    by: dict[tuple[str, float], list[float]] = {}
    for r in rows:
        m = float(r["label"].split("/m")[1])
        by.setdefault((r["arm"], m), []).append(float(r["net"]))
    arms = sorted({a for a, _m in by})
    levels = sorted({m for _a, m in by})
    md = [
        "| arm | " + " | ".join(f"m={m:g}" for m in levels) + " |",
        "|---|" + "---|" * len(levels),
    ]
    for arm in arms:
        cells = []
        for m in levels:
            vals = by.get((arm, m), [])
            cells.append(f"{np.mean(vals):+.3f} (n={len(vals)})" if vals else "-")
        md.append(f"| {arm} | " + " | ".join(cells) + " |")

    ss_by: dict[tuple[str, float], list[float]] = {}
    for r in rows:
        if r.get("solve_success") is not None:
            m = float(r["label"].split("/m")[1])
            ss_by.setdefault((r["arm"], m), []).append(float(r["solve_success"]))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.2, 4.6), sharex=True, height_ratios=[2.2, 1.0])
    for arm in arms:
        means = [np.mean(by[(arm, m)]) for m in levels if (arm, m) in by]
        ax.plot(levels[: len(means)], means, marker="o", label=arm)
        ss = [np.mean(ss_by[(arm, m)]) for m in levels if (arm, m) in ss_by]
        if ss:
            ax2.plot(levels[: len(ss)], ss, marker="s", ms=4, label=arm)
    ax.set_ylabel("net value [EUR/m$^2$]")
    ax.legend(fontsize=8)
    ax2.axhline(0.9, color="tab:red", ls="--", lw=1.0, alpha=0.7)
    ax2.set_ylabel("solve-success")
    ax2.set_ylim(0.8, 1.02)
    ax2.set_xlabel("model-mismatch magnitude")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _provenance_footer(fig, f"{len(rows)} seasons, committed run records")
    for ext in ("svg", "pdf"):
        fig.savefig(outdir / f"fig-mismatch.{ext}")
    plt.close(fig)
    return "\n".join(md)


def fig_gate(
    seasons: list[dict[str, Any]], failed: dict[str, int], epsilon: float, outdir: Path
) -> str | None:
    """H5 figure: per-arm violating-season rate vs the gate threshold, with CP bounds."""
    if not seasons:
        return None
    from greenlight_bench.metrics.gate import clopper_pearson_upper

    by_arm: dict[str, list[bool]] = {}
    for r in seasons:
        by_arm.setdefault(r["arm"], []).append(bool(r["violated"].get("humidity", False)))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    banded_by: dict[str, list[bool]] = {}
    adv_fb: dict[str, int] = {}
    for r in seasons:
        banded_by.setdefault(r["arm"], []).append(
            bool(r.get("violated_banded", {}).get("humidity", False))
        )
        if r.get("advisor_fallback_steps"):
            adv_fb[r["arm"]] = adv_fb.get(r["arm"], 0) + int(r["advisor_fallback_steps"])
    arms = sorted(by_arm)
    rates, uppers, md = (
        [],
        [],
        [
            "| arm | strict | rate | CP upper | banded | banded rate | advisor fb |",
            "|---|---|---|---|---|---|---|",
        ],
    )
    for arm in arms:
        k = sum(by_arm[arm]) + failed.get(arm, 0)
        n = len(by_arm[arm]) + failed.get(arm, 0)
        kb = sum(banded_by.get(arm, [])) + failed.get(arm, 0)
        rates.append(k / n)
        uppers.append(clopper_pearson_upper(k, n, 0.05))
        md.append(
            f"| {arm} | {k}/{n} | {k / n:.2f} | {uppers[-1]:.3f} | {kb}/{n} | {kb / n:.2f} | "
            f"{adv_fb.get(arm, 0)} |"
        )
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = np.arange(len(arms))
    banded_rates = [
        (sum(banded_by.get(a, [])) + failed.get(a, 0)) / max(1, len(by_arm[a]) + failed.get(a, 0))
        for a in arms
    ]
    ax.bar(x - 0.19, rates, width=0.36, color="tab:blue", alpha=0.75, label="strict (any step)")
    ax.bar(
        x + 0.19,
        banded_rates,
        width=0.36,
        color="tab:orange",
        alpha=0.8,
        label="banded (>2 pts for >1 h)",
    )
    ax.scatter(x - 0.19, uppers, marker="_", s=300, color="black", label="CP 95% upper (strict)")
    ax.axhline(epsilon, color="tab:red", ls="--", lw=1.4, label=f"gate threshold = {epsilon:g}")
    ax.set_xticks(x, arms, rotation=12, fontsize=8)
    ax.set_ylabel("fraction of seasons violating RH")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _provenance_footer(fig, f"{sum(len(v) for v in by_arm.values())} seasons, crash-inclusive")
    for ext in ("svg", "pdf"):
        fig.savefig(outdir / f"fig-gate.{ext}")
    plt.close(fig)
    return "\n".join(md)


def fig_pareto(arms: dict[str, Any], outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for arm, a in sorted(arms.items()):
        x = a["online_cost_seconds"] * 1e3
        y = a["net_value_mean"]
        passed = a.get("gate_humidity", False)
        ax.scatter(
            x,
            y,
            s=60,
            marker="o" if passed else "x",
            label=f"{arm}{'' if passed else ' (gate fail)'}",
        )
        ax.annotate(arm, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("online decision cost [ms/step]")
    ax.set_ylabel("net economic value [EUR/m$^2$]")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _provenance_footer(fig, "per-arm means, committed run records")
    for ext in ("svg", "pdf"):
        fig.savefig(outdir / f"fig-pareto.{ext}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/benchmark")
    ap.add_argument(
        "--benchmark",
        default=None,
        help="explicit benchmark_*.json filename (reproducibility); default = all present. "
        "The paper's anchor is benchmark_r1-union.json.",
    )
    args = ap.parse_args()
    results = Path(args.results)
    outdir = results / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    md: list[str] = ["# Benchmark analysis (generated by examples/analyze.py)", ""]

    curves = load_curves(results)
    md.append(f"## Data efficiency ({len(curves)} seed curves found)\n")
    if curves:
        table, stats = curve_table(curves)
        md.extend([table, ""])
    benchmarks = load_benchmark(results, args.benchmark)
    refs: dict[str, float] = {}
    for blob in benchmarks:
        md.append(f"## Benchmark summary: `{blob['_file']}`\n")
        rec = blob.get("record", {})
        md.append(
            f"- commit `{rec.get('git_commit')}` dirty={rec.get('git_dirty')} | "
            f"{rec.get('grid', {}).get('seasons')} seasons | failed {rec.get('n_failed_seasons')}\n"
        )
        md.append(
            "| arm | net mean | sd | n | gate(rh) | underpowered | online ms | solve-success |"
        )
        md.append("|---|---|---|---|---|---|---|---|")
        for arm, a in sorted(blob.get("arms", {}).items()):
            refs.setdefault(arm, a["net_value_mean"])
            md.append(
                f"| {arm} | {a['net_value_mean']:+.3f} | {a['net_value_std']:.3f} | "
                f"{a.get('n_seasons', '?')} | {a['gate_humidity']} | "
                f"{a.get('gate_underpowered', '?')} | {a['online_cost_seconds'] * 1e3:.2f} | "
                f"{a['solve_success']} |"
            )
        md.append("")
        seasons = blob.get("seasons", [])
        if seasons:
            failed_counts: dict[str, int] = {}
            for fr in rec.get("failed_seasons", []):
                failed_counts[fr["arm"]] = failed_counts.get(fr["arm"], 0) + 1
            gate_md = fig_gate(seasons, failed_counts, _EPSILON, outdir)
            if gate_md:
                md.extend(
                    ["### Gate detail (violating-season rates, crash-inclusive)\n", gate_md, ""]
                )
            mm = mismatch_table_and_fig(seasons, outdir)
            if mm:
                md.extend(["### Net by mismatch level (advantage decay)\n", mm, ""])
        else:
            md.append("(no per-season rows in this summary; gate/mismatch figures skipped)\n")
        if blob.get("arms"):
            fig_pareto(blob["arms"], outdir)

    refs.pop("black_box", None)  # PPO is the curve itself, not a reference line
    refs.pop("physics_informed", None)  # RL-in-MPC uses the trained policy: not a no-training line
    if curves:
        fig_data_efficiency(curves, stats, refs, outdir, load_sac_curves(results))
        md.append(
            "Figures written: fig-data-efficiency, fig-pareto/fig-mismatch when inputs exist.\n"
        )

    (outdir / "analysis.md").write_text("\n".join(md))
    print(f"wrote {outdir / 'analysis.md'} and figures ->", outdir)


if __name__ == "__main__":
    main()
