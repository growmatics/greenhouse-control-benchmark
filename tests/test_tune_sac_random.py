"""Tests for the SAC random-search sampler and winner selection (pure logic; no GLG2).

These guard that the sealed round-2 protocol is implemented faithfully: the search
space is respected, sampling is reproducible from the fixed seed, only the searched
hyperparameters are overridden (the frozen reference recipe is untouched), and the
winner rule matches the pre-registration (highest validation net, tie-break lower
n_envs then lower learning rate).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from tune_sac_random import (
    _EXCLUDED_TRIALS,
    _N_TRIALS,
    _REFINE_SPACE,
    _REFINE_UPDATE_BAND,
    _SEARCH_SPACE,
    _apply_overrides,
    gradient_updates,
    sample_configs,
    sample_refinement_configs,
    select_ranked,
    select_winner,
)


def test_sampling_is_reproducible_from_the_fixed_seed() -> None:
    assert sample_configs(seed=0) == sample_configs(seed=0)


def test_sampling_is_seed_sensitive() -> None:
    assert sample_configs(seed=0) != sample_configs(seed=1)


def test_default_trial_count_matches_the_sealed_protocol() -> None:
    assert len(sample_configs()) == _N_TRIALS == 40


def test_every_sample_lies_inside_the_declared_space() -> None:
    for cfg in sample_configs():
        for name, (kind, spec) in _SEARCH_SPACE.items():
            if kind == "loguniform":
                lo, hi = spec
                assert lo <= cfg[name] <= hi, (name, cfg[name])
            else:  # choice
                assert cfg[name] in spec, (name, cfg[name])


def test_choice_samples_are_plain_python_ints() -> None:
    # JSON-serialisable: guards against numpy scalars leaking into the record.
    cfg = sample_configs()[0]
    for name in ("n_envs", "train_freq", "gradient_steps"):
        assert type(cfg[name]) is int


def test_overrides_touch_only_the_searched_keys() -> None:
    ref = {
        "learning_rate": 7e-4,
        "tau": 0.0135,
        "train_freq": 50,
        "gradient_steps": 10,
        "gamma": 0.9631,  # frozen
        "buffer_size": 576_100,  # frozen
        "batch_size": 128,  # frozen
        "policy_kwargs": {"net_arch": {"pi": [256, 256, 256]}},  # frozen
    }
    cfg = {"learning_rate": 3e-4, "tau": 0.006, "train_freq": 1, "gradient_steps": 1, "n_envs": 1}
    out = _apply_overrides(ref, cfg)
    assert out["learning_rate"] == 3e-4
    assert out["tau"] == 0.006
    assert out["train_freq"] == 1
    assert out["gradient_steps"] == 1
    # frozen keys are unchanged, and n_envs is NOT a SAC kwarg (it sizes the vec env)
    assert out["gamma"] == 0.9631
    assert out["buffer_size"] == 576_100
    assert out["batch_size"] == 128
    assert out["policy_kwargs"] == {"net_arch": {"pi": [256, 256, 256]}}
    assert "n_envs" not in out
    assert ref["learning_rate"] == 7e-4  # input dict not mutated


def test_gradient_updates_matches_the_sb3_sac_schedule() -> None:
    # updates = budget / (train_freq * n_envs) * gradient_steps
    assert gradient_updates({"train_freq": 8, "n_envs": 4, "gradient_steps": 4}) == 125_000
    assert gradient_updates({"train_freq": 1, "n_envs": 2, "gradient_steps": 10}) == 5_000_000


def test_amendment_excludes_only_multiday_trials() -> None:
    sealed = sample_configs()
    assert _EXCLUDED_TRIALS == {21, 25, 29}
    # every excluded trial is in the multi-day class (>= 2M gradient updates),
    # and the single heaviest draw (t29, 5M) is among them
    assert all(gradient_updates(sealed[i]) >= 2_000_000 for i in _EXCLUDED_TRIALS)
    assert max(range(_N_TRIALS), key=lambda i: gradient_updates(sealed[i])) in _EXCLUDED_TRIALS


def test_refinement_sampling_is_reproducible_and_in_space() -> None:
    configs = sample_refinement_configs()
    assert configs == sample_refinement_configs()
    assert len(configs) == 2
    lo_u, hi_u = _REFINE_UPDATE_BAND
    for cfg in configs:
        for name, (kind, spec) in _REFINE_SPACE.items():
            if kind == "loguniform":
                lo, hi = spec
                assert lo <= cfg[name] <= hi, (name, cfg[name])
            else:
                assert cfg[name] in spec, (name, cfg[name])
        assert lo_u <= gradient_updates(cfg) <= hi_u


def test_refinement_cannot_recreate_a_multiday_trial() -> None:
    # the update-band cap bounds wall clock: every replacement sits in the
    # winners' 20k-160k range, far below the excluded 2M/5M-update draws
    for cfg in sample_refinement_configs(n_trials=20):
        assert gradient_updates(cfg) <= 160_000


def test_winner_is_highest_validation_net() -> None:
    trials = [
        {"trial": 0, "val_net": 1.0, "config": {"n_envs": 1, "learning_rate": 3e-4}},
        {"trial": 1, "val_net": 2.5, "config": {"n_envs": 8, "learning_rate": 7e-4}},
        {"trial": 2, "val_net": 2.0, "config": {"n_envs": 4, "learning_rate": 5e-4}},
    ]
    assert select_winner(trials)["trial"] == 1


def test_winner_tie_breaks_toward_lower_n_envs_then_lower_lr() -> None:
    trials = [
        {"trial": 0, "val_net": 2.0, "config": {"n_envs": 8, "learning_rate": 3e-4}},
        {
            "trial": 1,
            "val_net": 2.0,
            "config": {"n_envs": 1, "learning_rate": 7e-4},
        },  # lower n_envs
        {
            "trial": 2,
            "val_net": 2.0,
            "config": {"n_envs": 1, "learning_rate": 3e-4},
        },  # then lower lr
    ]
    assert select_winner(trials)["trial"] == 2


def test_select_ranked_returns_winner_then_runnerup_by_the_same_rule() -> None:
    """The confirmation run's winner (x5 seeds) and runner-up (x1 seed) come from the same
    sealed selection rule; top-2, best first, and consistent with select_winner."""
    trials = [
        {"trial": 0, "val_net": 1.0, "config": {"n_envs": 1, "learning_rate": 3e-4}},
        {"trial": 1, "val_net": 2.5, "config": {"n_envs": 8, "learning_rate": 7e-4}},
        {"trial": 2, "val_net": 2.0, "config": {"n_envs": 4, "learning_rate": 5e-4}},
    ]
    top2 = select_ranked(trials, n=2)
    assert [t["trial"] for t in top2] == [1, 2]  # winner then runner-up
    assert top2[0] == select_winner(trials)  # winner agrees with select_winner
    # tie at the top resolves by the same tie-break (lower n_envs, then lower lr)
    tied = [
        {"trial": 0, "val_net": 2.0, "config": {"n_envs": 8, "learning_rate": 3e-4}},
        {"trial": 1, "val_net": 2.0, "config": {"n_envs": 1, "learning_rate": 3e-4}},
    ]
    assert [t["trial"] for t in select_ranked(tied, n=2)] == [1, 0]
