# Benchmark results (committed audit trail)

Small JSON summaries only, the committed tier of the two-tier storage contract
(the run reports in this directory). Every number the paper quotes must trace to a file here.

- `benchmark_<gitcommit>.json` -- one four-way benchmark run: per-arm net value (mean +/- spread),
  feasibility-gate outcome, online cost, MPC solve-success, the Pareto block, and the run record
  (git commit, hardware stamp, timestamp, grid). Written by `examples/run_benchmark.py`.
- `data_efficiency_s{seed}.json` -- one PPO seed's data-efficiency curve `{budget: heldout_net}`.
  Written by `examples/train_policies.py`.

**Not here (gitignored, local on the run machine):** PPO checkpoints (`checkpoints/*.zip`), raw
per-step traces, logs. They are reproducible from the committed config + seeds + the pinned
simulator, so they do not belong in git.
