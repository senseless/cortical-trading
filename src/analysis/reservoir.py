"""Reservoir-mode readout training.

Treats the culture as a fixed nonlinear reservoir (Brainoware-style): a linear
readout is trained on per-channel spike counts (with optional lags) to predict
the direction of the next price move. Trained offline from session logs --
every agent-mode session doubles as reservoir training data at no extra
neuron cost. Readouts drift with the culture; retrain periodically.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .session_io import load_session


def build_dataset(session_dirs: list[str | Path], lags: int = 2) -> tuple[np.ndarray, np.ndarray, int]:
    xs, ys = [], []
    n_channels = None
    for d in session_dirs:
        steps = load_session(d)["steps"]
        # group by episode; never build features across episode/session boundaries
        episodes: dict[int, list[dict]] = {}
        for row in steps:
            if row.get("counts"):
                episodes.setdefault(row["episode"], []).append(row)
        for rows in episodes.values():
            counts = [np.asarray(r["counts"], dtype=float) for r in rows]
            mids = [r["mid"] for r in rows]
            if n_channels is None and counts:
                n_channels = len(counts[0])
            for i in range(lags, len(rows) - 1):
                delta = mids[i + 1] - mids[i]
                if delta == 0:
                    continue
                feat = np.concatenate(counts[i - lags: i + 1])
                xs.append(feat)
                ys.append(1 if delta > 0 else -1)
    if not xs:
        raise ValueError("no usable steps found (need counts + next-step price moves)")
    return np.vstack(xs), np.asarray(ys), int(n_channels)


def train_readout(session_dirs: list[str | Path], out_path: str | Path,
                  lags: int = 2) -> dict:
    X, y, n_channels = build_dataset(session_dirs, lags=lags)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))

    n_splits = max(2, min(5, len(y) // 40))
    accs = []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        if len(set(y[train_idx])) < 2:
            continue
        model.fit(X[train_idx], y[train_idx])
        accs.append(float(model.score(X[test_idx], y[test_idx])))
    model.fit(X, y)

    majority = float(max(np.mean(y == 1), np.mean(y == -1)))
    bundle = {
        "model": model,
        "scaler": None,           # scaling lives inside the pipeline
        "lags": lags,
        "n_channels": n_channels,
        "cv_accuracy": accs,
        "majority_baseline": majority,
        "n_samples": int(len(y)),
        "trained_on": [str(d) for d in session_dirs],
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    return {
        "out": str(out),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "cv_accuracy_mean": round(float(np.mean(accs)), 4) if accs else None,
        "cv_accuracy_folds": [round(a, 4) for a in accs],
        "majority_baseline": round(majority, 4),
        "lags": lags,
    }
