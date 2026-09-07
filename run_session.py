"""Run a trading-game session.

Usage:
    python run_session.py                                   # config/default.toml
    python run_session.py --config config/my_experiment.toml
    python run_session.py --set market.synthetic.regime=trend --set session.episodes=3
"""

from __future__ import annotations

import argparse
import json

from src.config import apply_override, load_config
from src.session import SessionRunner


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
