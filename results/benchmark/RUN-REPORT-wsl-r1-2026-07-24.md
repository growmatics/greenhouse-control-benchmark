# benchmark R1 run report (WSL, 2026-07-24)

The R1 re-anchor run: 5 arms incl. tuned SAC, the finer 6-level mismatch ladder. Anchor tag
`benchmark-run-r1-2026-07-24`, anchor file `benchmark_r1-union.json`.

## 1. Environment actuals
- Machine: WSL2 (`Linux 6.6.87.2-microsoft-standard-WSL2-x86_64`), 11 physical / 22 logical cores,
  ~20 GB RAM (`.wslconfig` cap). the run machine.
- Python: `~/growmatics-runtime/venv` (durable runtime, not `/tmp`).
- Simulator: GreenLight-Gym2 v0.2.1 @ commit `8bb5b28` (source checkout, not just the wheel).
- Weather: `$GLG2_WEATHER_DIR=~/growmatics-runtime/bench_weather`, Amsterdam Open-Meteo, all held-out
  years 2022--2025 present. `GROWMATICS_BIGFILES` = the bulk-storage root.
- `OMP_NUM_THREADS=1`, `--workers 20`, `nice -n 5`.

## 2. Installation headaches
- None new this run (the environment was already stood up for the v2 run and the SAC search). GLG2
  needs a source checkout (the wheel alone is insufficient); already in place.

## 3. Smoke + determinism
- SAC-arm smoke through `run_benchmark --arms black_box_sac --limit 2` passed before launch (+3.417).
- Determinism confirmed by exact reproduction of the v2 anchor: incumbent **-2.742**, PPO **+3.500**,
  NMPC m=0 **-3.06**, hybrid m=0 **+3.58** all matched `benchmark_b868bc9.json` to the decimal.

## 4. What actually ran
- Grid: arms {incumbent, black_box (PPO), black_box_sac (SAC), physics_mpc (NMPC),
  physics_informed (hybrid)}; held-out 2022--2025; mismatch `{0, 0.025, 0.05, 0.10, 0.15, 0.20}`
  (physics arms only; learned + incumbent at m=0); 5 seeds; PPO/SAC deploy the 2M policy. 284 seasons.
- Wall-clock: ~6.5--7 h (main run), 20 workers; the physics arms (NMPC + hybrid, 240 seasons at
  ~0.3--0.45 s/step) dominate; the learned/incumbent seasons are ~seconds each.
- Main summary: `benchmark_29d9eb6.json` (committed).

## 5. Failures and anomalies
- **0 failed seasons** in the main run (`n_failed_seasons=0`).
- **SAC-arm staging bug (found + fixed, the decision record of 2026-07-24).** The tuned-winner checkpoints were
  staged into the flat `checkpoints/` dir with `cp -n`, which silently kept the 2026-07-17 as-shipped
  `sac_s{0,1}` files; the main run's SAC arm was therefore a mix (bogus arm mean +2.56). Caught mid-run
  by comparing the live per-arm interim mean to the confirmation. Fixed by overwriting seeds 0--1 with
  the tuned t19 checkpoints and re-running ONLY the SAC arm (`sac-redo/benchmark_24105d8-dirty.json`,
  SAC +3.214); the two were unioned into `benchmark_r1-union.json` by `examples/union_sac_redo.py`.
  Positive side finding: R1 reproduced BOTH the as-shipped and tuned per-seed nets to the third decimal
  against their original independent eval pipelines --- a cross-pipeline determinism check.
- Solve-success: NMPC **0.999**, hybrid **0.995** (both above the 0.9 floor, at all mismatch levels).
- Gate: every arm saturates the strict any-step humidity gate (144/144-style), as pre-registered.
- Seed spread: PPO data-efficiency curve tight at 2M (SD 0.24); widest at the ~200k crossover.

## 6. Files produced
| File | What |
|------|------|
| `benchmark_29d9eb6.json` | main R1 run summary (SAC arm buggy) |
| `sac-redo/benchmark_24105d8-dirty.json` | SAC-arm redo (corrected checkpoints) |
| `benchmark_r1-union.json` | **the anchor**: union of the above (SAC +3.214) |
| `analysis/*` | figures + `analysis.md` regenerated from the anchor |
| `analysis/stats.md` (via `stats`) | interaction p = 4.5e-13, paired tests |

Reproduce any table/figure: `manuscript/REPRODUCE.md`.
