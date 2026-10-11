"""Tests for the unified benchmark aggregation (no simulator; synthetic seasons)."""

import numpy as np
import pytest

from greenlight_bench.benchmark import BenchmarkResult, hardware_stamp
from greenlight_bench.config import GateConfig
from greenlight_bench.envs.glg2 import SeasonResult

_GATE = GateConfig(epsilon=0.05, delta=0.05, n_seasons_min=60, method="clopper_pearson")


def _season(
    net: float,
    humidity_trace: list[float],
    online_cost: float = 0.0,
    solve_success: float | None = None,
) -> SeasonResult:
    return SeasonResult(
        net_economic_value=net,
        exceedances={
            "humidity": np.array(humidity_trace),
            "temperature": np.zeros(len(humidity_trace)),
        },
        online_cost_seconds=online_cost,
        solve_success=solve_success,
    )


def _result(seasons: dict, failed: dict[str, int] | None = None) -> BenchmarkResult:
    return BenchmarkResult(
        seasons=seasons,
        record={"config_version": "test"},
        gate=_GATE,
        soft_classes=("humidity",),
        step_hours=0.25,
        failed=failed or {},
    )


def test_arms_and_net_value_summary() -> None:
    seasons = {
        ("physics_mpc", "2022", 0): _season(1.0, [0.0, 0.0]),
        ("physics_mpc", "2022", 1): _season(3.0, [0.0, 0.0]),
        ("incumbent", "2022", 0): _season(-2.0, [0.5, 0.0]),
    }
    res = _result(seasons)
    assert res.arms() == ["incumbent", "physics_mpc"]
    mean, std = res.net_value_summary("physics_mpc")
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(1.0)


def test_gate_outcome_clean_arm_passes_violating_arm_fails() -> None:
    # a clean arm: 0 violating seasons out of a handful -> passes; but with too few
    # seasons the Clopper-Pearson bound is loose, so use enough clean seasons.
    clean = {("mpc", "y", s): _season(1.0, [0.0, 0.0]) for s in range(80)}
    assert _result(clean).gate_outcome("mpc")["humidity"] is True
    # an arm that violates humidity every season fails outright
    dirty = {("rb", "y", s): _season(1.0, [1.0, 0.0]) for s in range(80)}
    assert _result(dirty).gate_outcome("rb")["humidity"] is False


def test_failed_seasons_count_as_gate_violations_not_exclusions() -> None:
    # 80 clean seasons pass the gate; the same 80 plus 10 CRASHED seasons must fail it
    # (a crashed season is a deployment failure, never a silently dropped observation).
    clean = {("mpc", "y", s): _season(1.0, [0.0, 0.0]) for s in range(80)}
    assert _result(clean).gate_outcome("mpc")["humidity"] is True
    res = _result(clean, failed={"mpc": 10})
    assert res.gate_outcome("mpc")["humidity"] is False
    # net-value stats stay over completed seasons (no invented numbers for crashes)
    mean, _std = res.net_value_summary("mpc")
    assert mean == pytest.approx(1.0)
    # an arm known only through failures still appears, with NaN net value
    only_failed = _result(clean, failed={"rl": 3})
    assert "rl" in only_failed.arms()
    nan_mean, _ = only_failed.net_value_summary("rl")
    assert np.isnan(nan_mean)


def test_gate_underpowered_flags_season_counts_below_the_floor() -> None:
    # 59 completed + 1 failed = 60 meets the pre-registered floor; 59 alone does not.
    fiftynine = {("mpc", "y", s): _season(1.0, [0.0]) for s in range(59)}
    assert _result(fiftynine, failed={"mpc": 1}).gate_underpowered("mpc") is False
    assert _result(fiftynine).gate_underpowered("mpc") is True


def test_safety_profile_pools_seasons() -> None:
    seasons = {
        ("mpc", "2022", 0): _season(1.0, [2.0, 0.0]),
        ("mpc", "2023", 0): _season(1.0, [0.0, 3.0]),
    }
    prof = _result(seasons).safety("mpc", "humidity")
    assert prof.worst_excursion == pytest.approx(3.0)


def test_pareto_over_clean_arms() -> None:
    # incumbent: low value, moderately costly. rl: high value AND cheaper -> dominates the
    # incumbent and mpc. mpc: same value as rl but most costly -> dominated by rl.
    clean = {}
    for s in range(80):
        clean[("incumbent", "y", s)] = _season(0.0, [0.0], online_cost=0.5)
        clean[("mpc", "y", s)] = _season(5.0, [0.0], online_cost=1.0)
        clean[("rl", "y", s)] = _season(5.0, [0.0], online_cost=0.01)
    par = _result(clean).pareto(incumbent="incumbent")
    assert set(par["arms"]) == {"incumbent", "mpc", "rl"}
    assert set(par["non_dominated"]) == {"rl"}  # rl dominates both (better value, cheaper)
    assert par["hypervolume"] > 0.0  # the front dominates the anti-ideal corner
    assert par["reference"] == "anti_ideal"
    assert par["hypervolume_vs_incumbent"] >= 0.0  # the front covers at least the incumbent


def test_online_cost_mean() -> None:
    seasons = {
        ("mpc", "y", 0): _season(1.0, [0.0], online_cost=0.8),
        ("mpc", "y", 1): _season(1.0, [0.0], online_cost=1.2),
    }
    assert _result(seasons).online_cost("mpc") == pytest.approx(1.0)


def test_solve_success_averages_mpc_arms_and_is_none_for_solver_free() -> None:
    seasons = {
        ("mpc", "y", 0): _season(1.0, [0.0], solve_success=0.90),
        ("mpc", "y", 1): _season(1.0, [0.0], solve_success=1.00),
        ("incumbent", "y", 0): _season(1.0, [0.0], solve_success=None),
    }
    res = _result(seasons)
    assert res.solve_success("mpc") == pytest.approx(0.95)
    assert res.solve_success("incumbent") is None  # solver-free arm carries no reliability


def test_hardware_stamp_has_the_reproducibility_fields() -> None:
    stamp = hardware_stamp()
    # the online-cost axis is hardware-dependent, so the record must carry these
    for key in ("platform", "processor", "cpu_count", "python"):
        assert key in stamp
