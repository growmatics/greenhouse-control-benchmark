# How Much Does Physics Buy You? Benchmark code and run records

Code, committed run records and the analysis pipeline behind the paper

> **How Much Does Physics Buy You? A Physics-Informed vs Black-Box Benchmark for Greenhouse Climate Control.**
> Mohsen Davarynejad (Growmatics B.V.; Breda University of Applied Sciences), 2026. arXiv preprint, identifier to follow.

This repository is a **frozen artifact of record**: it holds exactly the code and the records that produced every number in the paper, and it will not grow. The maintained, open benchmark that grew out of this code is **[Phytotron](https://github.com/growmatics/phytotron)** (Growmatics). Phytotron reproduces this study's rule-based incumbent on all four held-out years and its physics-only NMPC season to six decimals (see its validation page); it does not yet ship a trained policy or the RL-in-MPC arm, so this repository remains the reference for those arms and for the paper's pre-registered gate. New benchmark work goes to Phytotron.

All results are simulated. Nothing here is a climate controller and nothing here must drive greenhouse equipment.

## What the paper measures

On GreenLight-Gym2 (the GreenLight tomato greenhouse model), five controllers run through one harness on an identical grid: four held-out Amsterdam weather years (2022 to 2025), a six-level controller-model mismatch ladder (0 to 20 %), five seeds, 60-day seasons; 284 seasons in all.

| Arm | Class | Where physics enters |
|---|---|---|
| rule-based incumbent | deployed practice | none (fixed setpoint schedule, the simulator's shipped controller) |
| black-box PPO, black-box SAC | model-free RL | none; pays with interaction data |
| physics-only economic NMPC | model-based | hard: direct collocation on the GreenLight ODE, nominal (perturbed) parameters |
| RL-in-MPC hybrid | physics-informed | the NMPC plus a learned long-horizon target from a PPO rollout, trajectory selection |

Scored on net economic value, a season-level humidity feasibility gate (Clopper-Pearson), online decision cost and solver reliability, with a paired statistical analysis. The paper's reading is deliberately plain: on this testbed black-box RL is the pragmatic winner at two orders of magnitude lower online cost than the hybrid; the physics-only NMPC is myopic and loses to the incumbent; the hybrid's one measured advantage is insensitivity to parametric model error, and that is carried by its model-free guide.

## Reproduce every number without running a simulation

The slow benchmark runs are distilled into per-season JSON records. One committed file is the anchor:

```
results/benchmark/benchmark_r1-union.json     # 284 seasons, 5 arms, 6 mismatch levels
```

Every table, figure and significance test regenerates from it:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt           # the frozen resolve of the run machine (Linux x86-64)
export OMP_NUM_THREADS=1
BM=results/benchmark/benchmark_r1-union.json
python examples/analyze.py        --benchmark $BM   # all figures + analysis.md (tables)
python examples/stats.py          --benchmark $BM   # all paired tests + the mismatch interaction
python examples/peryear_tables.py --benchmark $BM   # the per-year tables
```

The per-artifact map (which script and input regenerates which table or figure) is [`manuscript/REPRODUCE.md`](manuscript/REPRODUCE.md). The manuscript builds with `manuscript/build.sh` (pdflatex, bibtex, rsvg-convert).

## Re-run the benchmark

Needs the pinned simulator checkout (configs and weather are read from it) and weather files:

```bash
git clone --branch v0.2.1 https://github.com/BartvLaatum/GreenLight-Gym2   # commit 8bb5b28
export GLG2_CONFIG_DIR=<checkout>/gl_gym/configs
export WEATHER_DIR=<weather dir>          # python examples/fetch_weather.py builds it from Open-Meteo
export OMP_NUM_THREADS=1                   # required: NMPC seasons reproduce exactly only single-threaded
python examples/train_policies.py --checkpoint-dir checkpoints --out results/benchmark   # PPO, 5 seeds x 6 budgets
python examples/train_sac.py      --checkpoint-dir checkpoints --out results/benchmark   # SAC
python examples/run_benchmark.py  --workers 8 --checkpoint-dir checkpoints --out results/benchmark
```

The full grid is CPU-bound (one IPOPT solve per step for the two model-based arms, about 250 ms each) and took about a day on 20 workers; `examples/run_smoke.py` runs a short cross-arm smoke grid in minutes. The run records in `results/benchmark/` name the git commit, hardware, thread settings and grid of every run; `RUN-REPORT-wsl-r1-2026-07-24.md` is the report of the anchor run, and `union_sac_redo.py` documents how the SAC arm's redo was unioned into the anchor.

## Layout

```
greenlight_bench/      the package: harness, controllers (rule_based, rl, mpc, rl_mpc), metrics, gate, mismatch
configs/benchmark.yaml the single home of every benchmark constant (pins, gate, bounds, ladder, prices)
examples/              train_policies.py, train_sac.py, tune_sac_random.py, run_benchmark.py, run_smoke.py,
                       analyze.py, stats.py, peryear_tables.py, union_sac_redo.py, validate_mpc.py, l5_reproduce.py
tests/                 unit and harness tests (pytest; coverage floor 90 %)
results/benchmark/     committed run records (the anchor, the data-efficiency curves, the SAC tuning trials,
                       the train-year evaluation, the analysis outputs)
manuscript/            LaTeX source, figures (SVG and PDF), note-free bibliography, REPRODUCE.md
```

## Pins

GreenLight-Gym2 v0.2.1 (commit `8bb5b28`), CasADi 3.6.7 (bundles IPOPT and SUNDIALS/CVODES), Stable-Baselines3 2.1.0, sb3-contrib 2.1.0, Gymnasium 0.29.1, PyTorch 2.13.0 (CPU), NumPy 1.26.4, Python 3.12. `requirements.lock.txt` is the frozen resolve of the machine that ran the grid.

## Licence

AGPL-3.0-or-later, inherited from the GreenLight-Gym2 dependency. See [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff). Please cite the paper; the arXiv identifier is added on announcement.

## Acknowledgements

Built on GreenLight-Gym2 (van Laatum et al.) and the GreenLight model (Katzin et al., Wageningen University & Research), both released openly; weather from the Open-Meteo archive. The RL-in-MPC arm reimplements the trajectory-selection method of van Laatum et al. (2026, arXiv:2607.07365) from the paper.
