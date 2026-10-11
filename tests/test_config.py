"""Tests that the single-source config loads, validates, and self-checks."""

from pathlib import Path

import pytest

from greenlight_bench.config import (
    BenchmarkConfig,
    EconomicConfig,
    GateConfig,
    GeneralisationConfig,
    load_config,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "benchmark.yaml"

_OK_GATE = GateConfig(epsilon=0.05, delta=0.05, n_seasons_min=60, method="x")
_OK_GEN = GeneralisationConfig(location="X", train_years=(2019, 2020), heldout_years=(2021,))


def test_canonical_config_loads_and_validates() -> None:
    cfg = load_config(CONFIG_PATH)
    assert cfg.gate.epsilon == 0.05
    assert cfg.gate.delta == 0.05
    assert cfg.gate.n_seasons_min >= 59  # enough for a clean pass
    assert cfg.testbed["release"] == "v0.2.1"
    # the pinned weather-year split (most recent 12 complete years)
    assert cfg.generalisation.train_years == (2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021)
    assert cfg.generalisation.heldout_years == (2022, 2023, 2024, 2025)
    assert not set(cfg.generalisation.train_years) & set(cfg.generalisation.heldout_years)


def test_validate_rejects_too_few_seasons() -> None:
    bad = BenchmarkConfig(
        raw={"testbed": {}},
        gate=GateConfig(epsilon=0.05, delta=0.05, n_seasons_min=10, method="x"),
        generalisation=_OK_GEN,
    )
    with pytest.raises(ValueError):
        bad.validate()


def test_validate_rejects_out_of_range_epsilon() -> None:
    bad = BenchmarkConfig(
        raw={"testbed": {}},
        gate=GateConfig(epsilon=1.5, delta=0.05, n_seasons_min=60, method="x"),
        generalisation=_OK_GEN,
    )
    with pytest.raises(ValueError):
        bad.validate()


def test_validate_rejects_leaky_split() -> None:
    bad = BenchmarkConfig(
        raw={"testbed": {}},
        gate=_OK_GATE,
        generalisation=GeneralisationConfig(
            location="X", train_years=(2019, 2020), heldout_years=(2020, 2021)
        ),
    )
    with pytest.raises(ValueError, match="leaks"):
        bad.validate()


def test_canonical_config_loads_economic_block() -> None:
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg.economic, EconomicConfig)
    assert cfg.economic.penalty_weights == (5.0e-5, 5.0e-3, 7.0e-4)  # [CO2, temp, RH]
    assert cfg.economic.cost_input_indices == (0, 1, 4)  # heating, CO2, lamp


def test_l5_economic_applies_paper_overrides() -> None:
    cfg = load_config(CONFIG_PATH)
    ec = cfg.l5_economic()
    # the paper's fruit price is 20 EUR/kg{DW} = 1.2 EUR/kg{FW} / dmfm 0.06
    assert ec.fruit_price == 1.2
    assert ec.dmfm == 0.06
    assert ec.elec_price == 0.20
    assert ec.price_per_kg_dw == pytest.approx(20.0)
    # unchanged fields fall through from the base economic block
    assert ec.heating_price == cfg.economic.heating_price
