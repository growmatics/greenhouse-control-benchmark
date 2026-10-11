"""Arm 1: run the rule-based incumbent for one season and report its benchmark result.

Drives GLG2's shipped rule-based controller (the competence-standard incumbent) through
the uniform ``run_season_controller`` harness on the default benchmark env (Amsterdam,
60-day tomato, dt=900). Its net economic value and per-class constraint exceedances are
the deployment baseline every learned arm must beat.

    python examples/run_arm1.py
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from greenlight_bench.controllers.rule_based import RuleBasedIncumbent
from greenlight_bench.envs.glg2 import default_glg2_paths, make_tomato_env, run_season_controller


def main() -> None:
    config_dir = os.environ.get("GLG2_CONFIG_DIR")
    weather_dir = os.environ.get("GLG2_WEATHER_DIR")
    config_path = Path(config_dir) if config_dir else default_glg2_paths()[0]
    env = make_tomato_env(
        config_dir,
        weather_dir,
        overrides={"start_train_year": 2010, "end_train_year": 2010},
    )
    # the incumbent's competence-standard parameters, from GLG2's shipped rule_based.yml
    agents_yml = config_path.parent / "agents" / "rule_based.yml"
    rb_params = yaml.safe_load(agents_yml.read_text())["TomatoEnv"]
    controller = RuleBasedIncumbent(rb_params)

    result = run_season_controller(env, controller, seed=0)
    print("Arm 1 (rule-based incumbent) - one season, default benchmark env:")
    print(f"  net economic value = {result.net_economic_value:+.4f} EUR/m2")
    for cls, ex in result.exceedances.items():
        steps = int((ex > 0).sum())
        print(f"  {cls:12s}: steps>0 {steps:5d}  cumulative {ex.sum():9.1f}  worst {ex.max():7.2f}")


if __name__ == "__main__":
    main()
