"""Loader for the single source of truth, ``configs/benchmark.yaml``.

Every benchmark constant is read from here; nothing in the package hardcodes one.
``BenchmarkConfig.validate`` enforces internal consistency, including that the
declared season count is enough for the chosen ``(epsilon, delta)`` gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from greenlight_bench.metrics.gate import min_seasons


@dataclass(frozen=True)
class GateConfig:
    epsilon: float
    delta: float
    n_seasons_min: int
    method: str


@dataclass(frozen=True)
class GeneralisationConfig:
    """The weather-year split: train on some years, hold out others."""

    location: str
    train_years: tuple[int, ...]
    heldout_years: tuple[int, ...]


@dataclass(frozen=True)
class EconomicConfig:
    """Economic-MPC objective constants (the GLG2 ``GreenhouseReward`` EPI form).

    Prices, the [CO2, temp, RH] penalty weights, and the GreenLight ``p[]`` / ``x[]``
    indices the cost and gain terms use. Injected into the MPC; never hardcoded.
    """

    fruit_price: float  # EUR/kg fresh weight
    dmfm: float  # dry-matter / fresh-matter ratio
    heating_price: float  # EUR/kWh
    co2_price: float  # EUR/kg
    elec_price: float  # EUR/kWh
    penalty_weights: tuple[float, float, float]  # order [CO2, temp, RH]
    param_indices: dict[str, int]  # GreenLight p[] indices for cost conversions
    fruit_state_index: int  # x[] index of cFruit dry matter
    cost_input_indices: tuple[int, int, int]  # u[] indices: heating, CO2, lamp

    @property
    def price_per_kg_dw(self) -> float:
        """Effective fruit price per kg dry weight (``fruit_price / dmfm``)."""
        return self.fruit_price / self.dmfm


def _economic_from(block: dict[str, Any]) -> EconomicConfig:
    return EconomicConfig(
        fruit_price=float(block["fruit_price_eur_per_kg_fw"]),
        dmfm=float(block["dmfm"]),
        heating_price=float(block["heating_price_eur_per_kwh"]),
        co2_price=float(block["co2_price_eur_per_kg"]),
        elec_price=float(block["elec_price_eur_per_kwh"]),
        penalty_weights=tuple(float(w) for w in block["penalty_weights"]),  # type: ignore[arg-type]
        param_indices={str(k): int(v) for k, v in block["param_indices"].items()},
        fruit_state_index=int(block["fruit_state_index"]),
        cost_input_indices=tuple(int(i) for i in block["cost_input_indices"]),  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class BenchmarkConfig:
    """Parsed benchmark constants. ``raw`` keeps the untyped rest of the YAML."""

    raw: dict[str, Any]
    gate: GateConfig
    generalisation: GeneralisationConfig
    economic: EconomicConfig | None = None

    @property
    def testbed(self) -> dict[str, Any]:
        return dict(self.raw["testbed"])

    def l5_economic(self) -> EconomicConfig:
        """Economic config for the L5 reproduction: ``economic`` with the paper overrides."""
        base = dict(self.raw["economic"])
        l5 = self.raw["l5_reproduction"]
        for key in ("fruit_price_eur_per_kg_fw", "dmfm", "elec_price_eur_per_kwh"):
            if key in l5:
                base[key] = l5[key]
        return _economic_from(base)

    def validate(self) -> None:
        """Raise ``ValueError`` if any constant is out of range or inconsistent."""
        if not 0.0 < self.gate.epsilon < 1.0:
            raise ValueError(f"epsilon must be in (0, 1), got {self.gate.epsilon}")
        if not 0.0 < self.gate.delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {self.gate.delta}")
        required = min_seasons(self.gate.epsilon, self.gate.delta)
        if self.gate.n_seasons_min < required:
            raise ValueError(
                f"n_seasons_min={self.gate.n_seasons_min} is below the {required} "
                f"seasons required for a clean pass at eps={self.gate.epsilon}, "
                f"delta={self.gate.delta}"
            )
        train = set(self.generalisation.train_years)
        heldout = set(self.generalisation.heldout_years)
        if not train or not heldout:
            raise ValueError("train and held-out weather-year lists must be non-empty")
        overlap = train & heldout
        if overlap:
            raise ValueError(
                f"weather-year split leaks: {sorted(overlap)} in both train and held-out"
            )


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate ``benchmark.yaml``."""
    data = yaml.safe_load(Path(path).read_text())
    gate_raw = data["gate"]
    gate = GateConfig(
        epsilon=float(gate_raw["epsilon"]),
        delta=float(gate_raw["delta"]),
        n_seasons_min=int(gate_raw["n_seasons_min"]),
        method=str(gate_raw["method"]),
    )
    gen_raw = data["generalisation"]
    generalisation = GeneralisationConfig(
        location=str(gen_raw["location"]),
        train_years=tuple(int(y) for y in gen_raw["weather_years_train"]),
        heldout_years=tuple(int(y) for y in gen_raw["weather_years_heldout"]),
    )
    economic = _economic_from(data["economic"]) if "economic" in data else None
    config = BenchmarkConfig(raw=data, gate=gate, generalisation=generalisation, economic=economic)
    config.validate()
    return config
