"""Silicon baseline gate: can a conventional learner extract edge from what
the neurons see, at the game's cadence, after real costs?

Usage:
    python train_baseline.py                                  # synthetic sine (should PASS: sanity check)
    python train_baseline.py --data synthetic:random_walk     # unlearnable control (should FAIL: no leakage)
    python train_baseline.py --data data/market/MBTU26_XCME_20260828_203316.jsonl.gz
    python train_baseline.py --data <file1> --data <file2>    # multiple recordings
    python train_baseline.py --horizons 5,15,30,60 --band 0.1

Interpretation: if no model beats the majority-class share (outside the
binomial CI) on real data at any horizon, the current encoding carries no
signal and the game design needs work before renting wetware. The comparison
is against the majority class, not 0.5: on a drifting test slice, always-up
already scores the up-share. The 'extended' feature set shows whether
candidate Phase-2 channels would help.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from src.baseline.backtest import backtest_proba
from src.baseline.dataset import build_dataset, load_recording_series, synthetic_series
from src.baseline.models import binomial_margin, fit_logistic, fit_mlp
from src.config import load_config
from src.market.synthetic import REGIMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Train silicon baseline probes and backtest them")
    parser.add_argument("--data", action="append", default=None,
                        help="'synthetic:<regime>' or path to a recording; repeatable (default synthetic:sine)")
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--steps", type=int, default=20000, help="samples for synthetic data")
    parser.add_argument("--snr", type=float, default=None, help="synthetic snr override")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizons", default="5,15,30,60", help="label horizons in seconds")
    parser.add_argument("--step-interval", type=float, default=1.0, help="grid/game cadence seconds")
    parser.add_argument("--min-move", type=float, default=0.0, help="drop labels with |move| <= this (points)")
    parser.add_argument("--band", type=float, default=0.1, help="hold band: act when p outside 0.5 +/- band")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--label", default="", help="tag for the output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    horizons = [float(h) for h in args.horizons.split(",")]
    data_items = args.data or ["synthetic:sine"]

    segments = []
    descriptions = []
    rec_ranges: list[tuple[float, float, str]] = []
    for item in data_items:
        if item.startswith("synthetic:"):
            regime = item.split(":", 1)[1]
            if regime not in REGIMES:
                raise SystemExit(f"unknown regime '{regime}', expected one of {REGIMES}")
            scfg = dataclasses.replace(cfg.market.synthetic, regime=regime, seed=args.seed)
            if args.snr is not None:
                scfg.snr = args.snr
            segments += synthetic_series(scfg, args.steps, args.step_interval)
            descriptions.append(f"{item} steps={args.steps} snr={scfg.snr} seed={args.seed}")
        else:
            segs = load_recording_series(item, args.step_interval)
            segments += segs
            rec_ranges += [(float(s.t[0]), float(s.t[-1]), str(item)) for s in segs]
            descriptions.append(str(item))

    # Recordings must not overlap in time: two files covering the same market
    # window (e.g. a rotated file plus its overlap) would put identical prices
    # on both sides of the train/test boundary -- leakage that reads as SIGNAL.
    rec_ranges.sort()
    for (a0, a1, pa), (b0, b1, pb) in zip(rec_ranges, rec_ranges[1:]):
        if b0 <= a1:
            raise SystemExit(f"recordings overlap in time:\n  {pa}\n  {pb}\n"
                             "the same prices would land in both train and test")
    if rec_ranges and not any(item.startswith("synthetic:") for item in data_items):
        # Chronological splits regardless of the order files were passed in.
        segments.sort(key=lambda s: float(s.t[0]))

    ds = build_dataset(segments, cfg, horizons, args.step_interval, args.min_move)
    n_tr, n_va, n_te = (int(ds.splits[k].sum()) for k in ("train", "val", "test"))
    print(f"data: {'; '.join(descriptions)}")
    print(f"samples: {ds.n} (train {n_tr} / val {n_va} / test {n_te}), "
          f"segments: {len(segments)}, cadence {args.step_interval}s")

    # CUDA is required unless CPU is requested explicitly: a silent fallback
    # would run the gate on a slower/different backend than the one the
    # results are calibrated on, without anyone noticing.
    try:
        import torch
    except ImportError:
        raise SystemExit("torch is not installed; the baseline gate needs it: "
                         "pip install torch --index-url https://download.pytorch.org/whl/cu126")
    if args.device == "cpu":
        dev = "cpu"
    elif torch.cuda.is_available():
        dev = "cuda"
    else:
        raise SystemExit(
            "CUDA is not available but the baseline gate requires it "
            "(install the CUDA build: pip install torch --index-url "
            "https://download.pytorch.org/whl/cu126 -- or pass --device cpu "
            "to knowingly run on CPU)")
    dev_name = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    print(f"torch device: {dev} ({dev_name})")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data/baseline") / (f"{stamp}_{args.label}" if args.label else stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr, va, te = ds.splits["train"], ds.splits["val"], ds.splits["test"]
    report: dict = {
        "data": descriptions, "n_samples": ds.n, "horizons_s": horizons,
        "band": args.band, "step_interval_s": args.step_interval,
        "instrument": dataclasses.asdict(cfg.instrument), "broker": dataclasses.asdict(cfg.broker),
        "results": {},
    }
    curves: dict[str, dict] = {}
    equities: dict[str, np.ndarray] = {}

    model_kinds = ["logistic", "mlp"]
    for set_name, X in ds.X.items():
        for kind in model_kinds:
            key = f"{set_name}/{kind}"
            per_h = {}
            best = None  # (val_pnl_dollars, horizon, proba_test_full)
            for h in horizons:
                y = ds.y[h]
                m_tr = tr & (y >= 0)
                m_va = va & (y >= 0)
                m_te = te & (y >= 0)
                # A min-move filter (or a short recording) can empty a split or
                # leave train single-class; fitting anyway would crash or emit garbage.
                if not (m_tr.any() and m_va.any() and m_te.any()) or len(np.unique(y[m_tr])) < 2:
                    print(f"{key:22s} @{h:.0f}s: skipped (not enough labeled rows in every split)")
                    continue
                if kind == "logistic":
                    predict = fit_logistic(X[m_tr], y[m_tr])
                else:
                    predict = fit_mlp(X[m_tr], y[m_tr], X[m_va], y[m_va], device=dev, seed=args.seed)
                val_acc = float(np.mean((predict(X[m_va]) > 0.5) == (y[m_va] == 1)))
                p_te = predict(X[m_te])
                test_acc = float(np.mean((p_te > 0.5) == (y[m_te] == 1)))
                # The no-skill benchmark is the majority class, not 0.5: on a
                # drifting slice, predicting "always up" already scores the
                # up-share without any signal.
                up_share = float(np.mean(y[m_te] == 1)) if m_te.any() else 0.5
                majority_acc = max(up_share, 1.0 - up_share)
                # Adjacent labels overlap by (h - step) seconds, so the
                # effective sample count is far below the row count.
                n_te_h = int(m_te.sum() / max(1.0, h / args.step_interval))
                # Horizon selection is by validation PnL, not accuracy: a
                # phase-shifted signal can be highly accurate yet untradeable.
                val_bt, _ = backtest_proba(predict(X[va]), args.band, ds.t[va], ds.bid[va],
                                           ds.ask[va], ds.segment_id[va], cfg.instrument,
                                           cfg.broker, n_random=0, seed=args.seed)
                per_h[h] = {"val_acc": round(val_acc, 4), "val_pnl": val_bt.dollars,
                            "test_acc": round(test_acc, 4), "majority_acc": round(majority_acc, 4),
                            "n_test": n_te_h, "ci_half_width": round(binomial_margin(n_te_h), 4)}
                if best is None or val_bt.dollars > best[0]:
                    best = (val_bt.dollars, h, predict(X[te]))
            if best is None:
                report["results"][key] = {"per_horizon": per_h,
                                          "skipped": "no horizon had enough labeled data"}
                print(f"{key:22s} skipped entirely (no horizon had enough labeled data)")
                continue
            _, best_h, proba_full = best
            bt, rnd = backtest_proba(proba_full, args.band, ds.t[te], ds.bid[te], ds.ask[te],
                                     ds.segment_id[te], cfg.instrument, cfg.broker, seed=args.seed)
            report["results"][key] = {
                "per_horizon": per_h, "best_horizon_s": best_h,
                "backtest": {"dollars": bt.dollars, "points": bt.points, "n_trades": bt.n_trades,
                             "win_rate": bt.win_rate, "n_signals": bt.n_signals},
                "random_policy": rnd,
            }
            curves[key] = per_h
            equities[key] = bt.equity
            edge = per_h[best_h]["test_acc"] - per_h[best_h]["majority_acc"]
            sig = "SIGNAL" if edge > per_h[best_h]["ci_half_width"] else "no signal"
            print(f"{key:22s} best@{best_h:.0f}s: acc {per_h[best_h]['test_acc']:.3f} "
                  f"vs maj {per_h[best_h]['majority_acc']:.3f} "
                  f"(CI +/-{per_h[best_h]['ci_half_width']:.3f}, {sig}) | "
                  f"PnL ${bt.dollars:+.2f} in {bt.n_trades} trades "
                  f"(random {rnd['mean_dollars']:+.2f} +/- {rnd['std_dollars']:.2f})")

    # Buy-and-hold context on the test slice.
    te_idx = np.flatnonzero(te)
    if len(te_idx) >= 2:
        bh_points = float(ds.mid[te_idx[-1]] - ds.mid[te_idx[0]])
        report["buy_hold_test"] = {"points": round(bh_points, 2),
                                   "dollars": round(bh_points * cfg.instrument.point_value, 2)}
        print(f"buy-and-hold over test: {bh_points:+.1f} pts "
              f"(${bh_points * cfg.instrument.point_value:+.2f})")

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plot(curves, equities, horizons, out_dir)
    print(f"report -> {out_dir}")


def _plot(curves: dict, equities: dict, horizons: list[float], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for key, per_h in curves.items():
        hs = [h for h in horizons if h in per_h and "test_acc" in per_h[h]]
        if not hs:
            continue
        accs = [per_h[h]["test_acc"] for h in hs]
        errs = [per_h[h]["ci_half_width"] for h in hs]
        ax1.errorbar(hs, accs, yerr=errs, marker="o", capsize=3, label=key)
    ax1.axhline(0.5, color="gray", ls="--", lw=1)
    ax1.set_xlabel("horizon (s)")
    ax1.set_ylabel("test directional accuracy")
    ax1.set_title("Predictability of forward move")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    for key, eq in equities.items():
        ax2.plot(eq, label=key, lw=1)
    ax2.axhline(0.0, color="gray", ls="--", lw=1)
    ax2.set_xlabel("test step")
    ax2.set_ylabel("equity after costs ($)")
    ax2.set_title("Policy backtest (test split)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "baseline_report.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
