# Reproducing every table and figure in the manuscript

Every quantitative table and figure regenerates from **one committed anchor file** by pointing a
Python script at it. No simulation is needed to reproduce the paper's numbers — the slow benchmark
runs are already distilled into per-season JSON records.

## The anchor (the single source of truth)

`results/benchmark/benchmark_r1-union.json` — the R1 anchor, 284 per-season rows, 5 arms,
6-level mismatch ladder. Built by `union_sac_redo.py` from the main grid run plus the SAC-arm redo
(see the R1 run report). Its `seasons` list is the raw audit trail; every script
below recomputes from it.

The anchor's run record carries `git_dirty: true`: the union was assembled from the main grid run and
the SAC-arm redo, whose own record is `sac-redo/benchmark_24105d8-dirty.json` (the redo ran with the
corrected SAC checkpoints staged in the working tree; the R1 run report documents it). The main grid
run itself (`benchmark_29d9eb6.json`) is from a clean tree.

Run everything from the `code/` directory with the project venv Python (`$PY` below):

```bash
cd papers/001-physics-informed-control-benchmark/code
PY=~/growmatics-runtime/venv/bin/python          # any env with numpy/scipy/statsmodels/matplotlib
BM=results/benchmark/benchmark_r1-union.json
```

## Map: manuscript artifact → command → input

| Manuscript artifact | Reproduce with | Input file(s) | Output |
|---------------------|----------------|---------------|--------|
| **Table** data-efficiency (`tab:dataeff`) | `analyze.py --benchmark …` | `data_efficiency_s0..4.json` (+ anchor for the reference lines) | `analysis/analysis.md` |
| **Table** four-way / five-way net + gate + cost | `analyze.py --benchmark …` | anchor | `analysis/analysis.md` |
| **Table** per-year (`tab:peryear`) + LOYO | `peryear_tables.py --benchmark …` | anchor | stdout (Markdown) |
| **Table** per-year × mismatch (`tab:peryear-mismatch`) | `peryear_tables.py --benchmark …` | anchor | stdout (Markdown) |
| **Stats** paired comparisons, mismatch interaction p, gate tests | `stats.py --benchmark …` | anchor | stdout / `analysis/stats.md` |
| **Figure** `fig-data-efficiency` | `analyze.py --benchmark …` | `data_efficiency_s*.json` + anchor | `analysis/fig-data-efficiency.{svg,pdf}` |
| **Figure** `fig-mismatch` | `analyze.py --benchmark …` | anchor | `analysis/fig-mismatch.{svg,pdf}` |
| **Figure** `fig-paired-slopegraph` | `analyze.py --benchmark …` | anchor | `analysis/fig-paired-slopegraph.{svg,pdf}` |
| **Figure** `fig-gate` | `analyze.py --benchmark …` | anchor | `analysis/fig-gate.{svg,pdf}` |
| **Figure** `fig-pareto` | `analyze.py --benchmark …` | anchor | `analysis/fig-pareto.{svg,pdf}` |
| **The anchor itself** | `union_sac_redo.py` | `benchmark_29d9eb6.json` + `sac-redo/benchmark_*.json` | `benchmark_r1-union.json` |
| Guarantee ledger (`tab:ledger`), `fig-experimental-setup` | hand-authored (qualitative / schematic) | — | not data-derived |

## One command to regenerate all data-derived tables + figures

```bash
$PY examples/analyze.py --benchmark $BM      # all figures + analysis.md (tables)
$PY examples/stats.py   --benchmark $BM      # all significance tests + the interaction p
$PY examples/peryear_tables.py --benchmark $BM      # the per-year tables
# figures land in results/benchmark/analysis/ ; copy the paper's set into manuscript/:
cp results/benchmark/analysis/fig-{data-efficiency,mismatch,paired-slopegraph,gate,pareto}.{svg,pdf} manuscript/
```

## How the figures are made

`analyze.py` renders each figure with **matplotlib** and saves **SVG** (the tracked source,
per `docs/figure-standards.md`) plus **PDF** (what `main.tex` includes). `manuscript/build.sh` then
runs `pdflatex → bibtex → pdflatex ×2`. So a figure in the paper traces: anchor JSON →
`analyze.py` → `fig-*.svg/pdf` → `\includegraphics` in `main.tex`.

## Verifying the paper matches the data

The LaTeX tables are hand-set (LaTeX has no clean auto-table path here), so to check the paper against
the data, run the three scripts above and diff their output against the corresponding table in
`main.tex`. Every number should match to the printed precision. Each table caption in `main.tex`
names the script that regenerates it.
