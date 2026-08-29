"""Decoders: motor-region spike counts -> trading actions.

Agent mode follows the Embodied Neurocomputation paper: spike counts per motor
region over the step window, normalized against baseline spontaneous activity
collected from the interactions preceding each episode, argmax with a hold
threshold.

Reservoir mode treats the culture as a fixed nonlinear reservoir: a linear
readout trained offline (see src/analysis/reservoir.py) maps per-channel spike
features to a next-move prediction.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..config import DecodingCfg
from ..game.state import Action
from .layout import ElectrodeLayout


class BaselineTracker:
    """Rolling per-region spontaneous activity statistics (per step window)."""

    def __init__(self, regions: list[str], maxlen: int):
        self.regions = regions
        self._windows: dict[str, deque[float]] = {r: deque(maxlen=maxlen) for r in regions}

    def add_window(self, region_counts: dict[str, float]) -> None:
        for region, count in region_counts.items():
            if region in self._windows:
                self._windows[region].append(count)

    @property
    def n_windows(self) -> int:
        return min((len(d) for d in self._windows.values()), default=0)

    def stats(self) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for region, values in self._windows.items():
            if values:
                arr = np.asarray(values, dtype=float)
                out[region] = (float(arr.mean()), float(arr.std()))
            else:
                out[region] = (0.0, 0.0)
        return out


class AgentDecoder:
    def __init__(self, cfg: DecodingCfg, layout: ElectrodeLayout):
        self.cfg = cfg
        self.layout = layout
        if not {"buy", "sell"} <= set(layout.motor):
            raise ValueError("agent mode requires motor regions named 'buy' and 'sell'")

    def region_counts(self, channel_counts: np.ndarray) -> dict[str, float]:
        return {region: float(channel_counts[chans].sum()) for region, chans in self.layout.motor.items()}

    def normalize(self, counts: dict[str, float], baseline: BaselineTracker) -> dict[str, float]:
        stats = baseline.stats()
        out: dict[str, float] = {}
        for region, count in counts.items():
            mean, std = stats.get(region, (0.0, 0.0))
            if self.cfg.normalization == "zscore":
                out[region] = (count - mean) / max(std, 1e-6)
            else:  # ratio
                out[region] = count / max(mean, self.cfg.min_baseline_count)
        return out

    def decide(self, channel_counts: np.ndarray, baseline: BaselineTracker) -> tuple[Action, dict]:
        raw = self.region_counts(channel_counts)
        norm = self.normalize(raw, baseline)
        diff = norm["buy"] - norm["sell"]
        if diff > self.cfg.threshold:
            action = Action.BUY
        elif diff < -self.cfg.threshold:
            action = Action.SELL
        else:
            action = Action.HOLD
        debug = {"raw": raw, "norm": norm, "diff": diff}
        return action, debug


class ReservoirDecoder:
    """Linear readout on per-channel spike features (trained offline)."""

    def __init__(self, cfg: DecodingCfg, n_channels: int):
        import joblib

        bundle = joblib.load(cfg.readout_path)
        self.model = bundle["model"]
        self.scaler = bundle.get("scaler")
        self.lags = int(bundle.get("lags", 0))
        expected = int(bundle.get("n_channels", n_channels))
        if expected != n_channels:
            raise ValueError(f"readout was trained on {expected} channels, session has {n_channels}")
        self.prob_threshold = cfg.readout_prob_threshold
        self._buffer: deque[np.ndarray] = deque(maxlen=self.lags + 1)

    def decide(self, channel_counts: np.ndarray, baseline: BaselineTracker | None = None) -> tuple[Action, dict]:
        self._buffer.append(channel_counts.astype(float))
        if len(self._buffer) < self.lags + 1:
            return Action.HOLD, {"warmup": True}
        x = np.concatenate(list(self._buffer)).reshape(1, -1)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        proba = self.model.predict_proba(x)[0]
        classes = list(self.model.classes_)
        p_up = float(proba[classes.index(1)]) if 1 in classes else 0.0
        p_down = float(proba[classes.index(-1)]) if -1 in classes else 0.0
        if p_up >= self.prob_threshold and p_up > p_down:
            action = Action.BUY
        elif p_down >= self.prob_threshold and p_down > p_up:
            action = Action.SELL
        else:
            action = Action.HOLD
        return action, {"p_up": p_up, "p_down": p_down}
