"""GreenLight-Gym2 TomatoEnv adapter.

``season_result_from_infos`` is pure (no simulator) and unit-tested; it parses
the TomatoEnv ``info`` dict into the benchmark's objective (net economic value)
and per-class exceedance series. ``make_tomato_env`` and ``run_season`` require
``gl_gym`` and are integration-only, so ``gl_gym`` is imported lazily.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Constraint class -> the TomatoEnv info field carrying its per-step violation.
VIOLATION_FIELDS: dict[str, str] = {
    "temperature": "temp_violation",
    "co2": "co2_violation",
    "humidity": "rh_violation",
    "lamp": "lamp_violation",
}


@dataclass(frozen=True)
class SeasonResult:
    """One season's benchmark outputs."""

    net_economic_value: float  # revenue minus VARIABLE (energy + resource) cost over the season
    exceedances: dict[str, NDArray[np.float64]]  # per class, per-step magnitude (>= 0)
    online_cost_seconds: float = 0.0  # mean controller decision time per step (Pareto axis)
    # Fraction of NLP solves that converged or salvaged a usable iterate (vs held the last
    # control): a deployment-reliability metric for the MPC arms; None for solver-free arms.
    solve_success: float | None = None


def season_result_from_infos(infos: Sequence[dict[str, Any]]) -> SeasonResult:
    """Parse per-step TomatoEnv info dicts into a ``SeasonResult`` (pure).

    Net economic value is revenue minus the VARIABLE (energy and resource) costs only,
    per the benchmark objective (``configs/benchmark.yaml``) and GLG2's own
    ``GreenhouseReward.profit`` (which likewise excludes fixed costs). Fixed costs are a
    control-independent constant offset, so they do not affect ranking or the Pareto
    front; excluding them makes the number the true operating profit.
    """
    if len(infos) == 0:
        raise ValueError("need at least one step of info")
    # Validate the expected keys are present and REFUSE to default them: a missing key
    # (e.g. a gl_gym version renamed it) must fail loudly, not silently report 0 revenue /
    # 0 violations -- that would be a plausible-but-wrong number, the exact failure the
    # no-invented-metrics guardrail exists to prevent.
    required = ("revenue", "variable_costs", *VIOLATION_FIELDS.values())
    missing = [k for k in required if k not in infos[0]]
    if missing:
        raise KeyError(
            f"TomatoEnv info dict is missing expected keys {missing}; the installed gl_gym "
            "version may have changed. Refusing to default them to 0 (would report "
            "plausible-but-wrong metrics)."
        )
    revenue = np.array([float(i["revenue"]) for i in infos])
    variable = np.array([float(i["variable_costs"]) for i in infos])
    net = float(revenue.sum() - variable.sum())
    exceedances = {
        cls: np.array([abs(float(i[field])) for i in infos])
        for cls, field in VIOLATION_FIELDS.items()
    }
    return SeasonResult(net_economic_value=net, exceedances=exceedances)


def default_glg2_paths() -> tuple[Path, Path]:  # pragma: no cover - needs gl_gym
    """Config and weather directories from the installed ``gl_gym`` package."""
    import gl_gym

    root = Path(gl_gym.__file__).resolve().parent
    return root / "configs" / "envs", root / "environments" / "weather"


def make_tomato_env(
    config_dir: str | Path | None = None,
    weather_dir: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    spec_overrides: dict[str, Any] | None = None,
) -> Any:  # pragma: no cover - needs gl_gym + config/weather data
    """Build a ``TomatoEnv``. Requires ``gl_gym`` installed.

    ``overrides`` patch the base env params (e.g. ``training``, ``start_train_year``);
    ``spec_overrides`` patch the TomatoEnv spec (e.g. ``eval_options``). Defaults to
    the installed package's config and weather directories.
    """
    from gl_gym.common.utils import load_env_params
    from gl_gym.environments.tomato_env import TomatoEnv

    default_config, default_weather = default_glg2_paths()
    config_dir = Path(config_dir) if config_dir is not None else default_config
    weather_dir = Path(weather_dir) if weather_dir is not None else default_weather

    base_params, env_params = load_env_params("TomatoEnv", str(config_dir))
    base_params["weather_data_dir"] = str(weather_dir)
    if overrides:
        base_params.update(overrides)
    if spec_overrides:
        env_params.update(spec_overrides)
    return TomatoEnv(base_env_params=base_params, **env_params)


def run_season(
    env: Any,
    policy: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    seed: int,
) -> SeasonResult:  # pragma: no cover - needs a live env
    """Run one full season under a plain ``obs -> action`` ``policy`` (e.g. an RL model)."""
    obs, _info = env.reset(seed=seed)
    infos: list[dict[str, Any]] = []
    done = False
    while not done:
        obs, _reward, terminated, truncated, info = env.step(policy(obs))
        infos.append(info)
        done = bool(terminated or truncated)
    return season_result_from_infos(infos)


def run_season_controller(
    env: Any,
    controller: Any,
    seed: int,
    forecast_steps: int = 24,
) -> SeasonResult:  # pragma: no cover - needs a live env
    """Run one full season driving a benchmark ``Controller`` via ``act(ControlContext)``.

    This is the uniform harness every arm runs through: each step it assembles a
    ``ControlContext`` (observation, full state, weather forecast, timestep, live env)
    and calls ``controller.act``. ``forecast_steps`` is the disturbance window put in the
    context (>= the largest controller horizon; the env carries a prediction buffer past
    the season end, so the slice is always full).
    """
    from greenlight_bench.controllers.base import ControlContext

    # Controllers output one of two control conventions, and the env has a step for
    # each: "absolute" controls (physics MPC, rule-based) go through step_raw_control,
    # which sets u directly; "incremental" rate actions in [-1,1] (RL) go through the
    # gym step, which does u <- clip(u + a*delta_u_max). Using the wrong one silently
    # feeds the plant different controls than the controller planned.
    incremental = getattr(controller, "control_kind", "absolute") == "incremental"
    step = env.step if incremental else env.step_raw_control

    import time
    from dataclasses import replace

    obs, _info = env.reset(seed=seed)
    controller.reset(seed)
    infos: list[dict[str, Any]] = []
    act_seconds = 0.0
    done = False
    while not done:
        t = int(env.timestep)
        ctx = ControlContext(
            obs=np.asarray(obs, dtype=float),
            state=np.asarray(env.x, dtype=float).flatten(),
            forecast=np.asarray(env.weather_data[t : t + forecast_steps], dtype=float),
            timestep=t,
            env=env,
        )
        a = time.perf_counter()
        action = controller.act(ctx)  # timed: this is the arm's online (deployment) cost
        act_seconds += time.perf_counter() - a
        obs, _reward, terminated, truncated, info = step(action)
        infos.append(info)
        done = bool(terminated or truncated)
    result = season_result_from_infos(infos)
    # Pull the MPC solve-reliability tally if this arm has one (arm 3 directly, arm 4 via
    # its inner ``.mpc``); solver-free arms (rule-based, RL) leave it None.
    stats = getattr(controller, "solve_stats", None) or getattr(
        getattr(controller, "mpc", None), "solve_stats", None
    )
    solve_success = (
        (stats["clean"] + stats["salvage"]) / total
        if stats is not None and (total := sum(stats.values())) > 0
        else None
    )
    return replace(
        result,
        online_cost_seconds=act_seconds / max(len(infos), 1),
        solve_success=solve_success,
    )
