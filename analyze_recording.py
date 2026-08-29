"""Analyze finished sessions: reports, control comparisons, readout training.

Usage:
    python analyze_recording.py data/sessions/<dir>
    python analyze_recording.py data/sessions/<a> --compare data/sessions/<b>
    python analyze_recording.py data/sessions/<dir> [more dirs] --train-readout --out data/readouts/r1.joblib
"""

from __future__ import annotations

import argparse
import json

from src.analysis import analyze_session, compare_sessions, train_readout


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze trading-game sessions")
    parser.add_argument("sessions", nargs="+", help="session directory (or several for --train-readout)")
    parser.add_argument("--compare", metavar="DIR", help="second session to compare against (Mann-Whitney U)")
    parser.add_argument("--metric", default="pnl_dollars",
                        help="episode metric for comparison (pnl_dollars, directional_accuracy, ...)")
    parser.add_argument("--train-readout", action="store_true", help="train a reservoir readout from the sessions")
    parser.add_argument("--lags", type=int, default=2, help="spike-count lag windows for readout features")
    parser.add_argument("--out", default="data/readouts/readout.joblib", help="readout output path")
    args = parser.parse_args()

    if args.train_readout:
        result = train_readout(args.sessions, args.out, lags=args.lags)
        print(json.dumps(result, indent=2))
        if result["cv_accuracy_mean"] is not None:
            edge = result["cv_accuracy_mean"] - result["majority_baseline"]
            print(f"\nreadout edge over majority baseline: {edge:+.4f}")
        print(f"use it: run_session.py --set session.mode=reservoir --set neural.decoding.readout_path={result['out']}")
        return

    for session in args.sessions:
        summary = analyze_session(session)
        print(json.dumps(summary, indent=2))

    if args.compare:
        result = compare_sessions(args.sessions[0], args.compare, metric=args.metric)
        print("\n=== comparison ===")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
