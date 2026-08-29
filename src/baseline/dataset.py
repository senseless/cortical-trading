"""Dataset construction for the silicon baseline.

Sources (recordings or synthetic regimes) are resampled onto a fixed step grid
(the game cadence). Two feature sets are built:

- "encoded": exactly the market signal the neurons receive today -- the
  tanh-normalized momentum from FeatureTracker, computed with the same code
  and config. One column.
- "extended": candidate Phase-2 sensory channels -- returns over several
  windows (fixed-scale and vol-normalized), short/long volatility ratio,
  spread in ticks, book imbalance, level proximity (position within the
  rolling 5-min/30-min range and range width in vol units), time of day.

Labels are the sign of the forward mid move over each horizon. All features
use only past data; standardization happens later on train statistics only.
Gaps in recordings split the series into segments so no feature or label ever
spans a discontinuity.
"""

from __future__ import annotations

import gzip
import json
import math
import zlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Config, SyntheticCfg
from ..game.engine import FeatureTracker
from ..market.source import MarketSnapshot
from ..market.synthetic import SyntheticSource


@dataclass
class Segment:
    """One contiguous stretch of market data on the step grid."""

    t: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    mid: np.ndarray
    bid_size: np.ndarray
    ask_size: np.ndarray

    def __len__(self) -> int:
        return len(self.t)


def load_recording_series(path: str | Path, step_s: float = 1.0, max_gap_s: float = 120.0) -> list[Segment]:
    """Resample a JSONL(.gz) recording onto the step grid, splitting at gaps."""
    quotes: list[tuple[float, float, float, float, float]] = []
    opener = gzip.open if str(path).endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn tail line in a live file
                if ev.get("type") == "quote":
                    quotes.append((ev["t"], ev["bid"], ev["ask"], ev.get("bs", 0.0), ev.get("as", 0.0)))
    except (EOFError, zlib.error):
        pass  # truncated/corrupted tail (crashed or live recorder); use what decompressed cleanly
    if len(quotes) < 2:
        raise RuntimeError(f"{path}: not enough quotes to build a series")
    quotes.sort(key=lambda q: q[0])

    segments: list[Segment] = []
    rows: list[tuple[float, float, float, float, float]] = []
    idx = 0
    grid_t = quotes[0][0]
    last = quotes[0]
    while idx < len(quotes):
        nxt = quotes[idx]
        if nxt[0] - last[0] > max_gap_s and rows:
            segments.append(_segment_from_rows(rows))
            rows = []
            grid_t = nxt[0]
            # The new segment must start from the first post-gap quote; carrying
            # the pre-gap quote forward would plant a phantom price jump at the
            # segment start (and inflate rolling-vol features for minutes).
            last = nxt
        while grid_t <= nxt[0]:
            rows.append((grid_t, last[1], last[2], last[3], last[4]))
            grid_t += step_s
        last = nxt
        idx += 1
    if rows:
        segments.append(_segment_from_rows(rows))
    return [s for s in segments if len(s) >= 10]


def _segment_from_rows(rows: list[tuple[float, float, float, float, float]]) -> Segment:
    arr = np.asarray(rows, dtype=np.float64)
    return Segment(
        t=arr[:, 0], bid=arr[:, 1], ask=arr[:, 2],
        mid=(arr[:, 1] + arr[:, 2]) / 2.0,
        bid_size=arr[:, 3], ask_size=arr[:, 4],
    )


def synthetic_series(cfg: SyntheticCfg, steps: int, step_s: float = 1.0) -> list[Segment]:
    source = SyntheticSource(cfg)
    t = np.arange(steps, dtype=np.float64) * step_s
    bid = np.empty(steps)
    ask = np.empty(steps)
    bsz = np.empty(steps)
    asz = np.empty(steps)
    for i, ti in enumerate(t):
        snap = source.snapshot(float(ti))
        bid[i], ask[i] = snap.bid, snap.ask
        bsz[i], asz[i] = snap.bid_size, snap.ask_size
    return [Segment(t=t, bid=bid, ask=ask, mid=(bid + ask) / 2.0, bid_size=bsz, ask_size=asz)]


# ---------------------------------------------------------------------------

RETURN_WINDOWS_S = (1.0, 5.0, 15.0, 30.0, 60.0, 90.0, 120.0)
ZRET_WINDOWS_S = (5.0, 30.0, 120.0)     # vol-normalized momentum (sigma units)
LEVEL_WINDOWS_S = (300.0, 1800.0)       # near-term level structure windows
VOL_SHORT_S = 30.0
VOL_LONG_S = 300.0

EXTENDED_NAMES = (
    [f"ret_{int(w)}s" for w in RETURN_WINDOWS_S]
    + [f"zret_{int(w)}s" for w in ZRET_WINDOWS_S]
    + ["vol_ratio", "spread_ticks", "book_imbalance"]
    + [f"range_pos_{int(w)}s" for w in LEVEL_WINDOWS_S]
    + ["range_width", "tod_sin", "tod_cos"]
)


@dataclass
class BaselineDataset:
    """Aligned arrays across all segments, plus per-horizon labels and splits."""

    step_s: float
    horizons_s: list[float]
    feature_names: dict[str, list[str]]           # set name -> column names
    X: dict[str, np.ndarray]                      # set name -> (n, k)
    t: np.ndarray                                 # (n,)
    bid: np.ndarray
    ask: np.ndarray
    mid: np.ndarray
    segment_id: np.ndarray                        # (n,) int
    y: dict[float, np.ndarray]                    # horizon -> (n,) in {1, 0, -1=unlabeled}
    splits: dict[str, np.ndarray] = field(default_factory=dict)  # train/val/test -> bool mask

    @property
    def n(self) -> int:
        return len(self.t)


def build_dataset(
    segments: list[Segment],
    cfg: Config,
    horizons_s: list[float],
    step_s: float = 1.0,
    min_move_points: float = 0.0,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> BaselineDataset:
    enc_cfg = cfg.neural.encoding
    tick = cfg.instrument.tick_size
    warmup = int(max(max(RETURN_WINDOWS_S), enc_cfg.momentum_window_s) / step_s) + 1
    max_h = int(max(horizons_s) / step_s)

    xs_enc, xs_ext, meta_rows, seg_ids = [], [], [], []
    labels: dict[float, list[np.ndarray]] = {h: [] for h in horizons_s}

    for seg_id, seg in enumerate(segments):
        n = len(seg)
        if n <= warmup + max_h:
            continue

        # Encoded feature: identical computation to the live game.
        tracker = FeatureTracker(enc_cfg)
        momentum = np.empty(n)
        for i in range(n):
            snap = MarketSnapshot(t=float(seg.t[i]), bid=float(seg.bid[i]), ask=float(seg.ask[i]),
                                  last=float(seg.mid[i]))
            momentum[i] = tracker.update(snap)["momentum_norm"]

        # Extended features, built by name so columns stay aligned with EXTENDED_NAMES.
        eps = 1e-9
        diffs = np.diff(seg.mid, prepend=seg.mid[0])
        vol_s = _rolling_std(diffs, int(VOL_SHORT_S / step_s))
        vol_l = _rolling_std(diffs, int(VOL_LONG_S / step_s))
        cols: dict[str, np.ndarray] = {}

        # Multi-timescale momentum, fixed scale (matches the live encoder's units).
        for w_s in RETURN_WINDOWS_S:
            w = int(w_s / step_s)
            ret = np.zeros(n)
            ret[w:] = (seg.mid[w:] - seg.mid[:-w]) / (w * step_s)
            cols[f"ret_{int(w_s)}s"] = np.tanh(ret / enc_cfg.momentum_scale)

        # Vol-normalized momentum: the same move in sigma units, so the signal
        # is comparable across volatility regimes (and across products).
        for w_s in ZRET_WINDOWS_S:
            w = int(w_s / step_s)
            move = np.zeros(n)
            move[w:] = seg.mid[w:] - seg.mid[:-w]
            cols[f"zret_{int(w_s)}s"] = np.tanh(move / (vol_l * math.sqrt(w) + eps) / 3.0)

        cols["vol_ratio"] = np.clip(np.where(vol_l > 1e-12, vol_s / (vol_l + eps), 1.0), 0.0, 3.0)
        cols["spread_ticks"] = (seg.ask - seg.bid) / tick
        size_sum = seg.bid_size + seg.ask_size
        with np.errstate(invalid="ignore", divide="ignore"):
            cols["book_imbalance"] = np.where(size_sum > 0, (seg.bid_size - seg.ask_size) / size_sum, 0.0)

        # Near-term level structure: where price sits within the rolling range
        # (0 = at the low, 1 = at the high; the stochastic-%K idea) and how
        # wide that range is in vol units (tight = squeeze, wide = trending).
        rng_longest, w_longest = None, 1
        for w_s in LEVEL_WINDOWS_S:
            w = int(w_s / step_s)
            hi = _rolling_extreme(seg.mid, w, "max")
            lo = _rolling_extreme(seg.mid, w, "min")
            rng = hi - lo
            cols[f"range_pos_{int(w_s)}s"] = np.where(rng > eps, (seg.mid - lo) / (rng + eps), 0.5)
            rng_longest, w_longest = rng, w
        cols["range_width"] = np.tanh(rng_longest / (vol_l * math.sqrt(w_longest) + eps) / 3.0)

        # Time of day (UTC), cyclically encoded; even 24/7 BTC has session structure.
        tod = np.mod(seg.t, 86400.0) / 86400.0
        cols["tod_sin"] = np.sin(2.0 * np.pi * tod)
        cols["tod_cos"] = np.cos(2.0 * np.pi * tod)

        ext = np.column_stack([cols[name] for name in EXTENDED_NAMES])

        lo, hi = warmup, n - max_h
        xs_enc.append(momentum[lo:hi, None])
        xs_ext.append(ext[lo:hi])
        meta_rows.append(np.column_stack([seg.t[lo:hi], seg.bid[lo:hi], seg.ask[lo:hi], seg.mid[lo:hi]]))
        seg_ids.append(np.full(hi - lo, seg_id, dtype=np.int64))

        for h_s in horizons_s:
            h = int(h_s / step_s)
            move = seg.mid[lo + h:hi + h] - seg.mid[lo:hi]
            y = np.where(move > min_move_points, 1, np.where(move < -min_move_points, 0, -1))
            labels[h_s].append(y.astype(np.int64))

    if not xs_enc:
        raise RuntimeError("no segment is long enough for the requested warmup + horizon")

    meta = np.vstack(meta_rows)
    ds = BaselineDataset(
        step_s=step_s,
        horizons_s=list(horizons_s),
        feature_names={"encoded": ["momentum_norm"], "extended": list(EXTENDED_NAMES)},
        X={"encoded": np.vstack(xs_enc), "extended": np.vstack(xs_ext)},
        t=meta[:, 0], bid=meta[:, 1], ask=meta[:, 2], mid=meta[:, 3],
        segment_id=np.concatenate(seg_ids),
        y={h: np.concatenate(v) for h, v in labels.items()},
    )

    n = ds.n
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    # Purge max_h rows before each boundary: labels look up to max_h steps
    # ahead, so without the gap the tail of one split is labeled with prices
    # from inside the next (leakage across the temporal split).
    for name, sl in (("train", slice(0, max(0, i_train - max_h))),
                     ("val", slice(i_train, max(i_train, i_val - max_h))),
                     ("test", slice(i_val, n))):
        mask = np.zeros(n, dtype=bool)
        mask[sl] = True
        if not mask.any():
            raise ValueError(
                f"'{name}' split is empty after the {max_h}-row purge ({n} usable rows "
                "total); use a longer recording or shorter/fewer horizons")
        ds.splits[name] = mask
    return ds


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    """Rolling population std, partial windows from the start (vectorized)."""
    n = len(x)
    i = np.arange(n)
    j = np.maximum(0, i - w + 1)
    cnt = (i - j + 1).astype(np.float64)
    cs = np.cumsum(x * x)
    cm = np.cumsum(x)
    s = cs - np.where(j > 0, cs[j - 1], 0.0)
    m = (cm - np.where(j > 0, cm[j - 1], 0.0)) / cnt
    out = np.sqrt(np.maximum(s / cnt - m * m, 0.0))
    out[cnt < 2] = 0.0
    return out


def _rolling_extreme(x: np.ndarray, w: int, mode: str) -> np.ndarray:
    """Rolling max/min via monotonic deque, partial windows from the start."""
    keep = (lambda a, b: a >= b) if mode == "max" else (lambda a, b: a <= b)
    out = np.empty(len(x))
    dq: deque[int] = deque()
    for k in range(len(x)):
        while dq and dq[0] <= k - w:
            dq.popleft()
        while dq and not keep(x[dq[-1]], x[k]):
            dq.pop()
        dq.append(k)
        out[k] = x[dq[0]]
    return out
