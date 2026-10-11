# Crossover-budget derivation (PPO data-efficiency curve)

Derivation record for the crossover budgets quoted in the manuscript Discussion. Input: the
5-seed mean PPO held-out net per budget (`data_efficiency_s0..4.json`, v2 anchor); thresholds =
the deployed arms' 60-season mean nets from `benchmark_b868bc9.json` (NMPC -3.500, incumbent -2.742).

| budget | PPO 5-seed mean |
|---|---|
| 50000 | -15.862 |
| 100000 | -16.542 |
| 200000 | -8.951 |
| 400000 | -0.757 |
| 1000000 | +2.901 |
| 2000000 | +3.500 |

Method: the curve crosses each threshold between 200k and 400k. Interpolate between those two
points; report the **log-budget** interpolation (budgets are log-spaced, so this is the natural scale),
with linear interpolation shown as sensitivity.

- vs NMPC (-3.500): frac=0.665 -> log-interp **~317k**, linear ~333k
- vs incumbent (-2.742): frac=0.758 -> log-interp **~338k**, linear ~352k

Steps per season = 5760 -> the log-interp crossovers correspond to ~55 and ~59 simulated seasons.

Quoted in the manuscript as approximate values with this file as the committed derivation.
