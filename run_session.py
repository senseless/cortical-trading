"""Run a trading-game session.

Usage:
    python run_session.py                                   # config/default.toml
    python run_session.py --config config/my_experiment.toml
    python run_session.py --set market.synthetic.regime=trend --set session.episodes=3
"""

from __future__ import annotations

import argparse
import json

from src.config import Config, load_config
from src.session import SessionRunner


def apply_override(cfg: Config, dotted: str) -> None:
    if "=" not in dotted:
        raise SystemExit(f"--set expects key.path=value, got: {dotted}")
    path, raw_value = dotted.split("=", 1)
    parts = path.strip().split(".")
    target = cfg
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise SystemExit(f"unknown config section: {path}")
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise SystemExit(f"unknown config key: {path}")
    current = getattr(target, leaf)
    value: object = raw_value
    if isinstance(current, bool):
        value = raw_value.strip().lower() in ("1", "true", "yes", "on")
    elif isinstance(current, int):
        value = int(raw_value)
    elif isinstance(current, float):
        value = float(raw_value)
    setattr(target, leaf, value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a cortical trading-game session")
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set market.source=live")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for override in args.set:
        apply_override(cfg, override)
    cfg.validate()

    runner = SessionRunner(cfg)
    metrics = runner.run()

    print(json.dumps({k: v for k, v in metrics.items() if k != "episodes"}, indent=2))
    print(f"\nsession dir: {runner.out_dir}")
    print(f"score ({metrics['score_metric']}): {metrics['score']}")


if __name__ == "__main__":
    main()
