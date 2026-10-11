"""Arm-3 (physics-only MPC) validation harness.

Runs the layered validation plan against the real environment. Layer 1 (the
collocation coefficients) lives in tests/test_mpc.py and runs in CI. This script
runs the integration layers:

  L2  prediction-model fidelity: drive the MPC's own model and the CVODES plant
      with the SAME controls and quantify divergence.
  L3  optimizer status: does the MPC solve, return valid controls, respect bounds.

L4 (behavioural) and L5 (reproduce the published numbers) are added once the
economic objective is tuned. Requires gl_gym + the fetched weather.

    GLG2_CONFIG_DIR=.../gl_gym/configs/envs GLG2_WEATHER_DIR=.../bench_weather \
    python examples/validate_mpc.py
"""

from __future__ import annotations

import os

import numpy as np

from greenlight_bench.controllers.base import NominalParameters
from greenlight_bench.controllers.mpc import PhysicsOnlyMPC
from greenlight_bench.envs.glg2 import make_tomato_env

_LABELS = {0: "co2Air", 2: "tAir", 15: "vpAir", 25: "cFruit"}
_BOUND_TOL = 1e-6


def main() -> None:
    env = make_tomato_env(
        os.environ["GLG2_CONFIG_DIR"],
        os.environ["GLG2_WEATHER_DIR"],
        overrides={
            "location": "Amsterdam_OM",
            "start_train_year": 2018,
            "end_train_year": 2018,
            "start_train_day": 59,
            "end_train_day": 59,
            "training": True,
        },
    )
    env.reset(seed=0)
    params = NominalParameters({str(i): float(v) for i, v in enumerate(np.array(env.p).flatten())})
    mpc = PhysicsOnlyMPC(params, horizon=3)

    horizon = 6
    control = np.array([0.3, 0.2, 1.0, 0.0, 0.0, 0.0])  # mild fixed control
    x0 = np.array(env.x).flatten()
    weather = [np.array(env.weather_data[env.timestep + k]) for k in range(horizon)]

    # L2: roll the plant and the MPC model from the same x0 under the same controls
    plant = [x0.copy()]
    xp = x0.copy()
    for _ in range(horizon):
        _, _, _, _, _ = env.step_raw_control(control)
        xp = np.array(env.x).flatten()
        plant.append(xp.copy())
    model = [x0.copy()]
    xm = x0.copy()
    for k in range(horizon):
        xm = mpc.model_step(xm, control, weather[k])
        model.append(xm.copy())

    print("=== L2 prediction-model fidelity (model vs CVODES plant, same controls) ===")
    for idx, name in _LABELS.items():
        pv = np.array([p[idx] for p in plant])
        mv = np.array([m[idx] for m in model])
        rel = np.abs(mv - pv) / (np.abs(pv) + 1e-9)
        print(f"  {name:>7}: max abs err {np.abs(mv - pv).max():.4g}  max rel err {rel.max():.2%}")

    # L3: optimizer status on a few steps
    print("\n=== L3 optimizer status (does the MPC solve, bounds respected) ===")
    env.reset(seed=0)
    mpc.reset()
    solved = 0
    for t in range(4):
        fc = np.array(env.weather_data[env.timestep : env.timestep + mpc.horizon + 1])
        try:
            u = mpc.solve(np.array(env.x).flatten(), fc)
            ok = bool(np.all(u >= -_BOUND_TOL) and np.all(u <= 1 + _BOUND_TOL))
            solved += ok
            env.step_raw_control(u)
            print(f"  step {t}: solved, controls in bounds={ok}")
        except Exception as exc:  # report, do not crash the harness
            print(f"  step {t}: SOLVE FAILED ({type(exc).__name__}) - objective tuning still WIP")
            break
    print(f"L3: {solved}/4 steps solved with valid controls")


if __name__ == "__main__":
    main()
