# SAC confirmation results (tuned, 5 seeds each)

The two tuned SAC configs from the r2 random search, retrained to 2M over the budget ladder and
evaluated on held-out 2022-2025. Data-efficiency curve per seed in each subdir's
`data_efficiency_sac_s{0..4}.json` (`curve[budget].heldout_net`).

- `t19/` — **winner** (search-selected config), 5 seeds, trained on the **Mac M4 (MPS)**. 2M mean **+3.21**.
- `t36/` — **runner-up** (robustness check), 5 seeds, trained on the **WSL box (CPU)**. 2M mean **+3.14**.

Headline in `../../outcomes-r2-sac-tuning.md`. The `device` field in each JSON records the training
device. Cross-device check: t19 seeds 0-1 were also trained on WSL/CPU (logs); the CPU-vs-MPS gap at
2M is ~0.07-0.18, within seed noise (decision 2026-07-23), so the conclusion is device-invariant.
