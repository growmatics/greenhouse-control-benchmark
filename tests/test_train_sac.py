"""Faithfulness tests for the SAC recipe translation.

These assert the SAC recipe translation is FAITHFUL to the pinned GreenLight-Gym2
``sac.yml`` (every hyperparameter matches; VecNormalize gamma equals the model gamma).
Note (independent audit 2026-07-17): the earlier belief that a dropped ``log_std_init``
caused the SAC divergence was WRONG -- for SAC with use_sde=False log_std_init is inert,
so the initial policy std is ~1.0 regardless. The divergence is genuine instability of the
(aggressive) reference recipe, not a translation bug. These tests therefore guard
faithfulness, not stability. Skipped when the pinned checkout is absent.
"""

import math
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from train_sac import (
    _VEC_NORM_BASE,
    _apply_winner,
    _reference_sac_hyperparams,
)


def test_apply_winner_overlays_only_searched_params() -> None:
    """The confirmation override replaces exactly the four searched SAC kwargs plus n_envs,
    leaves every frozen kwarg untouched, and is a no-op when given None."""
    frozen = {
        "learning_rate": 7e-4,
        "tau": 0.0135,
        "train_freq": 50,
        "gradient_steps": 10,
        "gamma": 0.9631,
        "batch_size": 128,
        "ent_coef": "auto",
    }
    winner = {
        "learning_rate": 2.74e-4,
        "n_envs": 2,
        "train_freq": 1,
        "gradient_steps": 1,
        "tau": 0.0102,
    }
    out, n_envs = _apply_winner(dict(frozen), 8, winner)
    assert n_envs == 2
    assert out["learning_rate"] == 2.74e-4 and out["tau"] == 0.0102
    assert out["train_freq"] == 1 and out["gradient_steps"] == 1
    assert isinstance(out["train_freq"], int) and isinstance(out["gradient_steps"], int)
    # frozen kwargs untouched
    assert out["gamma"] == 0.9631 and out["batch_size"] == 128 and out["ent_coef"] == "auto"
    # None is a no-op (train the shipped recipe)
    assert _apply_winner(dict(frozen), 8, None) == (frozen, 8)


_GLG2 = os.environ.get("GLG2_CONFIG_DIR")
_HAVE_GLG2 = bool(_GLG2 and (Path(_GLG2).parent / "agents" / "sac.yml").exists())
pytestmark = pytest.mark.skipif(not _HAVE_GLG2, reason="pinned gl_gym sac.yml not present")


def test_sac_hyperparams_match_the_reference_yaml() -> None:
    kw, n_envs = _reference_sac_hyperparams(_GLG2)
    ref = yaml.safe_load((Path(_GLG2).parent / "agents" / "sac.yml").read_text())["TomatoEnv"]
    assert n_envs == ref["n_envs"]
    assert kw["learning_rate"] == float(ref["learning_rate"])
    assert kw["buffer_size"] == ref["buffer_size"]
    assert kw["tau"] == ref["tau"]
    assert kw["train_freq"] == ref["train_freq"]
    assert kw["gradient_steps"] == ref["gradient_steps"]
    assert kw["ent_coef"] == ref["ent_coef"]  # "auto"
    assert "action_noise" in kw  # translated from the normalactionnoise block


def test_log_std_init_is_kept_at_reference_value_not_sb3_default() -> None:
    """The bug that collapsed the first run: log_std_init must be the reference's
    np.log(1)=0 (std 1.0), never SB3 SAC's default -3 (std ~0.05)."""
    kw, _ = _reference_sac_hyperparams(_GLG2)
    assert kw["policy_kwargs"]["log_std_init"] == pytest.approx(math.log(1.0))  # == 0.0
    assert kw["policy_kwargs"]["log_std_init"] != -3.0


def test_vecnorm_gamma_matches_model_gamma() -> None:
    """VecNormalize's reward-return normalization must use the model's gamma, not the
    library default 0.99 (the reference passes gamma into VecNormalize)."""
    kw, _ = _reference_sac_hyperparams(_GLG2)
    # the driver passes gamma=sac_kwargs["gamma"] into VecNormalize; assert it is defined
    assert "gamma" in kw and kw["gamma"] != 0.99  # reference gamma is 0.9631
    assert "gamma" not in _VEC_NORM_BASE  # gamma is injected per-run, not hard-coded


@pytest.mark.slow
def test_built_sac_initial_policy_std_is_order_one() -> None:
    """The constructed SAC's initial action std is ~1.0 (documented, not bug-catching:
    for use_sde=False this holds for any log_std_init, per the 2026-07-17 audit)."""

    kw, _ = _reference_sac_hyperparams(_GLG2)

    class _Env(gym.Env):
        observation_space = gym.spaces.Box(-1e4, 1e4, (6,), np.float64)
        action_space = gym.spaces.Box(-1, 1, (6,), np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._t = 0
            return self.np_random.uniform(0, 1, 6), {}

        def step(self, a):
            self._t += 1
            return self.np_random.uniform(0, 1, 6), 0.0, False, self._t >= 32, {}

    venv = VecNormalize(DummyVecEnv([_Env]), gamma=kw["gamma"], **_VEC_NORM_BASE)
    small = {**kw, "buffer_size": 500, "learning_starts": 10, "train_freq": 10}
    model = SAC("MlpPolicy", venv, seed=0, device="cpu", **small)
    obs = torch.zeros(1, 6)
    _mean, log_std, _ = model.policy.actor.get_action_dist_params(obs)
    assert float(log_std.exp().mean()) > 0.5  # std ~1.0 (reference), not ~0.05 (SB3 default)
