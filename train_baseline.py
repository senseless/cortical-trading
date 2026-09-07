"""Silicon baseline gate: can a conventional learner extract edge from what
the neurons see, at the game's cadence, after real costs?

Usage:
    python train_baseline.py                                  # synthetic sine (should PASS: sanity check)
    python train_baseline.py --data synthetic:random_walk     # unlearnable control (should FAIL: no leakage)
    python train_baseline.py --data data/market/<recording>.jsonl.gz
    python train_baseline.py --data <file1> --data <file2>    # multiple recordings: read as one
                                                              # stream (rotated files are stitched)
    python train_baseline.py --horizons 5,15,30,60 ...        # seconds-scale horizons (diagnostics only:
                                                              # /MBT moves at these horizons cannot clear costs)

Label horizons default to holding-period scale (1m-1h): an /MBT round trip
costs ~90 points (spread + commissions) and the typical move only reaches
that from ~5-30 minutes out, so predictability at shorter horizons is not
tradeable even when real. Long horizons need correspondingly long recordings
(the effective test count shrinks by horizon/step due to label overlap).

Interpretation: if no model beats the majority-class share (outside the
binomial CI) on real data at any horizon, the current encoding carries no
signal and the game design needs work before renting wetware. The comparison
is against the majority class, not 0.5: on a drifting test slice, always-up
already scores the up-share. The default probes the 'encoded' set -- exactly
the channels the layout allocates (the momentum strip and the book-imbalance
strip), computed by the live FeatureTracker; --sets strip removes the
imbalance strip as a control, and extended / imbalance are diagnostics.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from src.baseline.backtest import backtest_proba
from src.baseline.dataset import (STALE_S, build_dataset, calibrate_imbalance_scales, calibrate_momentum_scale,
                                  load_recording_series, longest_feature_window_s, synthetic_series)
from src.baseline.models import binomial_margin, fit_logistic, fit_mlp
from src.config import apply_override, load_config
from src.market.synthetic import REGIMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Train silicon baseline probes and backtest them")
    parser.add_argument("--data", action="append", default=None,
                        help="'synthetic:<regime>' or path to a recording; repeatable (default synthetic:sine)")
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set instrument.tick_size=0.25")
    parser.add_argument("--sets", default="encoded",
                        help="comma list of feature sets to probe (default: encoded, what the neurons "
                             "see). Others: strip (momentum ladder alone, the control for the "
                             "imbalance channel), extended, imbalance (candidate-column diagnostics)")
    parser.add_argument("--calibrate-scale", action="store_true",
                        help="set neural.encoding.momentum_scale and imbalance_scales from the training "
                             "rows (median |input| -> |norm| 0.48, the rule the /MBT defaults follow) "
                             "so every channel is read in this product's units")
    parser.add_argument("--steps", type=int, default=50000,
                        help="usable samples for synthetic data (~14 h at 1 s; hour-scale label "
                             "horizons need room for the split-boundary purge). A pre-roll "
                             "covering the longest feature window is generated on top, so "
                             "every column is live from the first sample, as in a live session")
    parser.add_argument("--snr", type=float, default=None, help="synthetic snr override")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizons", default="60,300,900,3600",
                        help="label horizons in seconds (default: holding-period scale, 1m-1h)")
    parser.add_argument("--step-interval", type=float, default=1.0, help="grid/game cadence seconds")
    parser.add_argument("--min-move", type=float, default=0.0, help="drop labels with |move| <= this (points)")
    parser.add_argument("--band", type=float, default=0.1, help="hold band: act when p outside 0.5 +/- band")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--label", default="", help="tag for the output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for override in args.set:
        apply_override(cfg, override)
    cfg.validate()
    horizons = [float(h) for h in args.horizons.split(",")]
    data_items = args.data or ["synthetic:sine"]

    segments = []
    descriptions = []
    recordings = [item for item in data_items if not item.startswith("synthetic:")]
    for item in data_items:
        if item.startswith("synthetic:"):
            regime = item.split(":", 1)[1]
            if regime not in REGIMES:
                raise SystemExit(f"unknown regime '{regime}', expected one of {REGIMES}")
            scfg = dataclasses.replace(cfg.market.synthetic, regime=regime, seed=args.seed)
            if args.snr is not None:
                scfg.snr = args.snr
            segments += synthetic_series(scfg, args.steps, args.step_interval,
                                         warm_s=longest_feature_window_s(cfg))
            descriptions.append(f"{item} steps={args.steps} snr={scfg.snr} seed={args.seed}")
    if recordings:
        # One quote stream across all files (rotated files are stitched; only
        # real gaps split it) in chronological order regardless of argument
        # order, so the train/val/test split is temporal. The loader rejects
        # files that overlap in time (leakage across the split).
        # Gap handling follows the configured live session: idle past
        # stale_quote_s (a live game idles through a maintenance break with
        # its tracker intact), and a new segment only once the gap outlasts
        # the longest feature window, where no state would survive anyway.
        live = cfg.market.live
        try:
            segments += load_recording_series(
                recordings, args.step_interval,
                max_gap_s=longest_feature_window_s(cfg),
                stale_s=live.stale_quote_s if live.stale_quote_s > 0 else STALE_S)
        except RuntimeError as exc:
            raise SystemExit(str(exc))
        descriptions += recordings

    enc = cfg.neural.encoding
    imbalance_scale_calibrated = False
    if args.calibrate_scale:
        enc.momentum_scale = calibrate_momentum_scale(segments, enc.momentum_scale_window_s,
                                                      args.step_interval)
        print(f"momentum_scale calibrated on training rows: {enc.momentum_scale:.4g} points/s "
              f"at {enc.momentum_scale_window_s:g}s")
        imb_scales = calibrate_imbalance_scales(segments, enc.imbalance_windows_s, args.step_interval)
        if imb_scales is not None:
            enc.imbalance_scales = imb_scales
            imbalance_scale_calibrated = True
            print("imbalance_scales calibrated on training rows: " + ", ".join(
                f"{w:g}s {s:.3f}" for w, s in zip(enc.imbalance_windows_s, imb_scales)))
        else:
            print("imbalance_scales left at defaults: the book carries no sizes in this data")

    ds = build_dataset(segments, cfg, horizons, args.step_interval, args.min_move)
    n_tr, n_va, n_te = (int(ds.splits[k].sum()) for k in ("train", "val", "test"))
    idle_rows = sum(int((~s.fresh_mask()).sum()) for s in segments)
    print(f"data: {'; '.join(descriptions)}")
    print(f"samples: {ds.n} (train {n_tr} / val {n_va} / test {n_te}), "
          f"segments: {len(segments)}, idle rows skipped: {idle_rows}, cadence {args.step_interval}s")
    enc_names = ds.feature_names["encoded"]
    dark = {name: float(np.mean(ds.X["encoded"][:, k] == 0.0)) for k, name in enumerate(enc_names)}
    print(f"encoded (what the neurons see, {len(enc_names)} channels): " + ", ".join(enc_names))
    print("  share of rows silent per channel: " + ", ".join(f"{n} {v:.0%}" for n, v in dark.items()))

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
        "momentum_scale": cfg.neural.encoding.momentum_scale,
        "momentum_scale_calibrated": bool(args.calibrate_scale),
        "imbalance_windows_s": list(cfg.neural.encoding.imbalance_windows_s),
        "imbalance_scales": [round(s, 5) for s in cfg.neural.encoding.imbalance_scales],
        "imbalance_scales_calibrated": imbalance_scale_calibrated,
        "encoded_channels": enc_names, "encoded_silent_share": {k: round(v, 4) for k, v in dark.items()},
        "segments": len(segments), "idle_rows_skipped": idle_rows,
        "test_range_utc": [
            time.strftime("%Y-%m-%d %H:%M", time.gmtime(float(ds.t[te][0]))),
            time.strftime("%Y-%m-%d %H:%M", time.gmtime(float(ds.t[te][-1])))],
        "test_hours": round(float(te.sum()) * args.step_interval / 3600.0, 1),
        "results": {},
    }
    curves: dict[str, dict] = {}
    equities: dict[str, np.ndarray] = {}

    model_kinds = ["logistic", "mlp"]
    wanted = [s.strip() for s in args.sets.split(",") if s.strip()] or list(ds.X)
    unknown = [s for s in wanted if s not in ds.X]
    if unknown:
        raise SystemExit(f"unknown feature set(s) {unknown}; available: {list(ds.X)}")
    for set_name in wanted:
        X = ds.X[set_name]
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
                             "win_rate": bt.win_rate, "n_signals": bt.n_signals,
                             "n_long": sum(1 for x in bt.trades if x["direction"] == "long"),
                             "n_short": sum(1 for x in bt.trades if x["direction"] == "short"),
                             "trades": bt.trades},
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
    majority: dict[float, float] = {}
    for key, per_h in curves.items():
        hs = [h for h in horizons if h in per_h and "test_acc" in per_h[h]]
        if not hs:
            continue
        accs = [per_h[h]["test_acc"] for h in hs]
        errs = [per_h[h]["ci_half_width"] for h in hs]
        ax1.errorbar(hs, accs, yerr=errs, marker="o", capsize=3, label=key)
        for h in hs:
            majority[h] = per_h[h]["majority_acc"]
    # The no-skill reference is the majority class at each horizon (the same
    # test labels for every probe), not 0.5: a drifting slice makes "always
    # up" score the up-share for free.
    if majority:
        hs = sorted(majority)
        ax1.plot(hs, [majority[h] for h in hs], color="gray", ls="--", lw=1, marker="_",
                 label="majority class (no skill)")
    ax1.set_xlabel("label horizon (s)")
    ax1.set_ylabel("test directional accuracy")
    ax1.set_title("Predictability of forward move (all inputs, per target horizon)")
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
