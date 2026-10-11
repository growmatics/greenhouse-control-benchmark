"""Arm 4: physics-informed RL-in-MPC (van Laatum trajectory selection).

The method under test. At each step it (1) rolls the guiding RL policy over the
horizon on a copy of the current state, (2) solves the physics economic MPC (arm 3),
and (3) applies the first control of whichever trajectory scores higher on the
finite-horizon planning objective -- the *trajectory selection* that lets the
controller fall back to the RL rollout when the short-horizon MPC does not improve on
it (van Laatum et al. [vanlaatum2026rlmpc]).

Both trajectories are scored on the SAME model by re-rolling their control sequences on
a dedicated ``rollout_env`` (a second TomatoEnv, state-copied from the plant), which
reproduces the plant dynamics and the policy's exact observation bit-for-bit -- so no
observation has to be reconstructed by hand.

Reused, not reinvented (novelty is the four-way comparison, not the controller; see
the paper). Reimplemented from the paper: van Laatum's
own code (``GL-Gym-MPC``) is private, so this follows the published formulation (validated
by the L5 penalty reproduction). Terminal ingredients (an L1 terminal cost toward the RL
rollout's terminal state, van Laatum Eqs 17-18) that let the MPC itself see the RL's
longer-horizon economics are in place, as is a state warm-start of the MPC from the RL
rollout's forward trajectory (it raises MPC solve-success from 91% to 95% on the held-out
closed loop and speeds each solve). The pieces: the rollout, the economic MPC with the
terminal cost, the rollout state warm-start, and the trajectory-selection mechanism.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from greenlight_bench.controllers.base import ControlContext, NominalParameters, PhysicsController
from greenlight_bench.controllers.mpc import PhysicsOnlyMPC

Policy = Callable[[NDArray[np.float64]], NDArray[np.float64]]  # obs -> incremental action


def vecnorm_stats_path(model_path: str | Path) -> Path:
    """The VecNormalize-statistics file that belongs to a policy checkpoint.

    One naming convention, used by the trainer when saving and here when loading, so
    the pair can never drift apart silently.
    """
    p = Path(model_path)
    return p.with_name(p.stem + "_vecnorm.pkl")


def load_ppo_policy(path: str) -> Policy:  # pragma: no cover - needs sb3 + a saved model
    """Load a saved Stable-Baselines3 policy as a deterministic ``obs -> action`` callable.

    Handles both black-box arms: PPO (``ppo_*``, ``train_policies.py``) and SAC
    (``sac_*``, ``train_sac.py``); the algorithm is chosen from the checkpoint filename
    (both are SB3 with the same ``predict`` interface, so the name is kept for
    backward compatibility). Training follows the GLG2 reference recipe (VecNormalize
    over observations, ``vanlaatum2026rlmpc``), so the policy net expects NORMALIZED
    observations: when the checkpoint has a sibling stats file (see
    :func:`vecnorm_stats_path`), the returned callable applies ``normalize_obs`` before
    the net. Deploying the net on raw observations was the silent failure mode this
    guards against. A checkpoint without stats is used raw.
    """
    import pickle

    from stable_baselines3 import PPO, SAC

    algo = SAC if Path(path).name.startswith("sac") else PPO
    model = algo.load(path)
    stats_path = vecnorm_stats_path(path)
    vecnorm = None
    if stats_path.exists():
        with stats_path.open("rb") as fh:
            vecnorm = pickle.load(fh)  # a VecNormalize; normalize_obs needs no live venv

    def policy(obs: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.asarray(obs, dtype=np.float64)
        if vecnorm is not None:
            x = vecnorm.normalize_obs(x)
        action, _state = model.predict(x, deterministic=True)
        return np.asarray(action, dtype=float)

    return policy


def _copy_state_into(dst: Any, src: Any) -> None:  # pragma: no cover - needs gl_gym
    """Copy a TomatoEnv's full internal state (so ``dst`` continues ``src``'s episode)."""
    dst.x = np.copy(src.x)
    dst.x_prev = np.copy(src.x_prev)
    dst.u = np.copy(src.u)
    dst.timestep = int(src.timestep)
    dst.day_of_year = src.day_of_year
    dst.hour_of_day = src.hour_of_day
    dst.terminated = False


def rollout_policy(  # pragma: no cover - needs gl_gym + a policy
    rollout_env: Any, plant_env: Any, policy: Policy, horizon: int
) -> tuple[list[NDArray[np.float64]], float, list[NDArray[np.float64]]]:
    """Roll ``policy`` ``horizon`` steps from the plant's current state on ``rollout_env``.

    Returns the *absolute* control trajectory applied, the horizon-summed economic reward
    (EPI), and the per-step *state* trajectory ``[x after step 0, ..., x after step H-1]``
    (its last element is the rollout's terminal state, the MPC's terminal target). The RL
    policy emits incremental actions, which the gym ``step`` maps to absolute controls --
    so ``rollout_env.u`` after each step is the absolute control. The state trajectory is a
    physically consistent forward guess used to warm-start the MPC's collocation states.
    """
    _copy_state_into(rollout_env, plant_env)
    obs = rollout_env._get_obs()
    controls: list[NDArray[np.float64]] = []
    states: list[NDArray[np.float64]] = []
    epi = 0.0
    for _ in range(horizon):
        action = np.asarray(policy(np.asarray(obs, dtype=float)), dtype=float)
        obs, _reward, _term, _trunc, info = rollout_env.step(action)
        controls.append(np.copy(rollout_env.u))
        states.append(np.copy(rollout_env.x))
        epi += float(info["EPI"])
    return controls, epi, states


def _score_controls(  # pragma: no cover - needs gl_gym
    rollout_env: Any, plant_env: Any, controls: list[NDArray[np.float64]]
) -> float:
    """Re-roll an absolute control sequence on the model and return its horizon EPI."""
    _copy_state_into(rollout_env, plant_env)
    epi = 0.0
    for u in controls:
        _obs, _reward, _term, _trunc, info = rollout_env.step_raw_control(
            np.asarray(u, dtype=float)
        )
        epi += float(info["EPI"])
    return epi


class PhysicsInformedMPC(PhysicsController):
    """Arm 4: RL-in-MPC with trajectory selection between the RL rollout and the MPC.

    Args:
        nominal_params: the controller's model parameters (never the sim's truth).
        policy: the guiding RL policy, ``obs -> incremental action`` (e.g. a PPO
            ``model.predict`` wrapper).
        rollout_env: a dedicated TomatoEnv with the SAME weather/config as the plant,
            used to roll and score trajectories without touching the plant.
        mpc: the physics economic MPC (arm 3); defaults to an economic ``PhysicsOnlyMPC``
            if ``economic``/``climate_bounds`` are supplied.
    """

    control_kind = "absolute"

    def __init__(
        self,
        nominal_params: NominalParameters,
        policy: Policy,
        rollout_env: Any,
        mpc: PhysicsOnlyMPC,
    ) -> None:
        super().__init__(nominal_params)
        self.policy = policy
        self.rollout_env = rollout_env
        self.mpc = mpc
        self.horizon = mpc.horizon
        self._n_selected_rl = 0  # diagnostics: how often the RL rollout wins
        self._n_steps = 0
        # Advisor reliability (reported like solve_stats): guided steps vs per-step
        # fallbacks to plain MPC when the advisory path fails (see act()).
        self.advisor_stats: dict[str, int] = {"guided": 0, "fallback": 0}

    def reset(self, seed: int = 0) -> None:  # pragma: no cover - rollout_env needs gl_gym
        self.mpc.reset(seed)
        # reset the rollout env once so its weather_data (loaded on reset) is available;
        # per step _copy_state_into overwrites the state but keeps the loaded weather.
        self.rollout_env.reset(seed=seed)
        self._n_selected_rl = 0
        self._n_steps = 0
        self.advisor_stats = {"guided": 0, "fallback": 0}

    def act(self, ctx: ControlContext) -> NDArray[np.float64]:  # pragma: no cover - integration
        """One control step: RL-advised MPC with a per-step fallback to plain MPC.

        The advisory path (rollout, guided solve, scoring, selection) steps a scratch
        copy of the plant, which the stiff integrator can occasionally fail to
        integrate under extreme states (observed twice in ~120 seasons, both at high
        mismatch). Such a failure is an ADVISOR failure, not a controller failure: the
        step falls back to the plain economic MPC (no terminal target, no warm start),
        the event is tallied in ``advisor_stats``, and the season continues. Decided
        2026-07-16 (design decision); previously the exception aborted the
        whole season and counted as a gate violation.
        """
        self._n_steps += 1
        try:
            # (1) RL rollout over the horizon: controls, planning score, state trajectory
            rl_controls, epi_rl, rl_states = rollout_policy(
                self.rollout_env, ctx.env, self.policy, self.horizon
            )
            x_terminal = rl_states[-1]
            # (2) physics economic MPC. If it carries a terminal cost (arm 4 proper), the
            # RL rollout's terminal state injects the RL's longer-horizon economics into
            # the short-horizon MPC (van Laatum Eqs 17-18); the rollout's state trajectory
            # warm-starts the collocation states (improves reliability on stiff corners).
            terminal = x_terminal if self.mpc.terminal_weight > 0.0 else None
            u_mpc0 = self.mpc.solve(
                ctx.state, ctx.forecast, terminal_target=terminal, state_guess=rl_states
            )
            pred = self.mpc._last_pred
            mpc_controls = pred["u"] if pred is not None else [u_mpc0] * self.horizon
            epi_mpc = _score_controls(self.rollout_env, ctx.env, mpc_controls)
            # (3) trajectory selection: apply the first control of the higher-scoring plan
            self.advisor_stats["guided"] += 1
            if epi_rl > epi_mpc:
                self._n_selected_rl += 1
                chosen = rl_controls[0]
            else:
                chosen = u_mpc0
        except Exception:  # deliberately broad: any advisory-path failure degrades, never dies
            self.advisor_stats["fallback"] += 1
            # The MPC was built WITH a terminal cost (weight > 0), so solve() requires a
            # target even without the advisor. Use the current state: a benign
            # stay-near-here pull at the small terminal weight, zero advisory content.
            # (First production fallback, 2026-07-17, failed on exactly this contract.)
            chosen = self.mpc.solve(
                ctx.state, ctx.forecast, terminal_target=np.asarray(ctx.state, dtype=float)
            )
        return np.clip(np.asarray(chosen, dtype=float), 0.0, 1.0)  # guard solver rounding

    @property
    def rl_selection_rate(self) -> float:
        """Fraction of steps where the RL rollout out-scored the MPC (diagnostic)."""
        return self._n_selected_rl / self._n_steps if self._n_steps else 0.0
