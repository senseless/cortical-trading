"""Dataset construction for the silicon baseline.

Sources (recordings or synthetic regimes) are resampled onto a fixed step grid
(the game cadence). Feature sets:

- "encoded": exactly the market signal the neurons receive -- the
  tanh-normalized momentum strip from FeatureTracker (one column per
  chronotopic window) plus the book-imbalance channel (time-averaged
  top-of-book imbalance, tanh-scaled) when the layout allocates its
  electrodes, computed with the same code and config as the live encoder.
  Windows longer than the history available so far read zero, exactly as the
  live encoder renders them (silence), so short recordings still build a
  dataset with a partly dark strip rather than being rejected outright.
- "strip": the momentum ladder alone -- the control for what the imbalance
  channel adds. Note the asymmetry with a live
  session, which pre-rolls the strip from history before the first step: in a
  recording-built dataset the long windows are dark for the first hours of
  every segment (a segment is a gap-free stretch of the whole library, not a
  file -- rotated files are stitched). Synthetic series are generated with a
  pre-roll (Segment.warm_rows) covering the longest window, so there every
  column is live from the first usable row and the gate measures the full
  strip.
- "extended" and "imbalance": candidate-column diagnostics, not what the
  neurons see. Extended is returns over windows from
  1 s to 8 h (fixed-scale and vol-normalized), short/long volatility ratio,
  spread in ticks, book imbalance (instantaneous and smoothed over several
  windows -- top-of-book size on micros is a few contracts, so the moment's
  ratio is mostly noise; the documented signal is the time average), level
  proximity (position within the rolling 5-min/30-min range and range width
  in vol units), time of day.
  Windows longer than WARMUP_S report 0 until enough history accumulates
  within the segment -- the live encoder's "zero = no information" semantics
  -- so short recordings still build a dataset; the long-window columns just
  carry nothing there. Evaluating the hour-scale windows requires multi-day
  contiguous recordings.

Labels are the sign of the forward mid move over each horizon. All features
use only past data; standardization happens later on train statistics only.

Gaps follow the live session's rules rather than splitting at the first quiet
minute. A live game idles once no quote has arrived for stale_quote_s (60 s):
the tracker is not updated and no decision is taken, but nothing is reset, and
play resumes with the next quote -- through a CME maintenance break, an
overnight lull or a reconnect alike (outage_timeout_s only ends a session
whose stream is *disconnected*). The loader mirrors that: rows more than
`stale_s` past the last real quote are carried on the grid (so label horizons
stay in true time) but excluded from X and labels, the strip tracker skips
them just as the runner does, and a new segment starts only after a gap longer
than the longest feature window (`max_gap_s`, 8 h), because past that no
tracker state survives and a cold start is what live would see as well. The
distinction matters: splitting at each daily break left the 8 h channel dark
for the first 8 h of every 23 h day -- 37% of /MES rows -- and splitting at
two minutes turned a week of /MBT into 46 segments over weekend quote
silences.
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
    # Leading rows that exist only to warm the feature windows; they feed the
    # rolling computations but are excluded from X and labels.
    warm_rows: int = 0
    # Per-row: a real quote arrived within stale_s (the live game would be
    # acting on this row). None means every row is fresh (synthetic data).
    fresh: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.t)

    def fresh_mask(self) -> np.ndarray:
        return np.ones(len(self.t), dtype=bool) if self.fresh is None else self.fresh


Quote = tuple[float, float, float, float, float]  # t, bid, ask, bid_size, ask_size


def _read_quotes(path: str | Path) -> list[Quote]:
    quotes: list[Quote] = []
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
    quotes.sort(key=lambda q: q[0])
    return quotes


MAX_GAP_S = 28800.0  # the longest feature window: past this no tracker state survives a gap anyway
STALE_S = 60.0       # live stale_quote_s: the game idles (no decision, no tracker update) past this


def load_recording_series(paths: str | Path | list[str | Path], step_s: float = 1.0,
                          max_gap_s: float = MAX_GAP_S, stale_s: float = STALE_S) -> list[Segment]:
    """Resample JSONL(.gz) recordings onto the step grid, splitting at real gaps.

    Several files are read as ONE quote stream: the recorder rotates files
    every few hours with only a seconds-long seam between them, and a segment
    boundary there would restart every feature window (the strip's 8 h window
    never becomes live inside a 6 h file) and discard a label horizon's worth
    of rows per file. Only a gap longer than max_gap_s (the longest feature
    window: the daily maintenance break and a weekend are on opposite sides
    of it) splits the series; shorter silences stay inside the segment with
    their rows marked not-fresh past stale_s, which is exactly the stretch a
    live game would idle through. Files must not overlap in time: the same
    prices in two files would straddle the train/test boundary as leakage.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    per_file = []
    for path in paths:
        quotes = _read_quotes(path)
        if len(quotes) < 2:
            # A rotation slot the market slept through (weekend, daily break)
            # holds no quotes; it is not an error, there is just nothing there.
            continue
        per_file.append((path, quotes))
    if not per_file:
        raise RuntimeError("no quotes in any recording: " + ", ".join(str(p) for p in paths))
    per_file.sort(key=lambda item: item[1][0][0])
    for (pa, qa), (pb, qb) in zip(per_file, per_file[1:]):
        if qb[0][0] <= qa[-1][0]:
            raise RuntimeError(f"recordings overlap in time:\n  {pa}\n  {pb}\n"
                               "the same prices would land in both train and test")
    quotes = [q for _, qs in per_file for q in qs]

    segments: list[Segment] = []
    rows: list[tuple[float, float, float, float, float]] = []
    fresh: list[bool] = []
    idx = 0
    grid_t = quotes[0][0]
    last = quotes[0]
    while idx < len(quotes):
        nxt = quotes[idx]
        if nxt[0] - last[0] > max_gap_s and rows:
            segments.append(_segment_from_rows(rows, fresh))
            rows, fresh = [], []
            grid_t = nxt[0]
            # The new segment must start from the first post-gap quote; carrying
            # the pre-gap quote forward would plant a phantom price jump at the
            # segment start (and inflate rolling-vol features for minutes).
            last = nxt
        while grid_t <= nxt[0]:
            rows.append((grid_t, last[1], last[2], last[3], last[4]))
            fresh.append(grid_t - last[0] <= stale_s)
            grid_t += step_s
        last = nxt
        idx += 1
    if rows:
        segments.append(_segment_from_rows(rows, fresh))
    return [s for s in segments if len(s) >= 10]


def calibrate_momentum_scale(segments: list[Segment], window_s: float, step_s: float = 1.0,
                             frac: float = 0.6, target_norm: float = 0.48) -> float:
    """Momentum scale (points/s) that puts the median |norm| at target_norm.

    This is the rule the /MBT default (1.2 pts/s at 30 s) was calibrated by:
    median |velocity| over the calibration window maps to tanh^-1(0.48), so
    the strip spends its dynamic range on typical moves instead of pinning
    near silence (a slow product) or saturating (a fast one). Uses only the
    first `frac` of rows -- the training portion of a temporal split -- so
    the scale carries no information from the evaluation slice.
    """
    w = max(1, int(round(window_s / step_s)))
    total = sum(len(s) for s in segments)
    budget = int(total * frac)
    speeds = []
    for seg in segments:
        take = min(len(seg), budget)
        if take > w:
            mid = seg.mid[:take]
            v = np.abs(mid[w:] - mid[:-w]) / (w * step_s)
            speeds.append(v[seg.fresh_mask()[w:take]])  # idle rows are not moves the game saw
        budget -= take
        if budget <= 0:
            break
    if not speeds:
        raise RuntimeError("not enough rows to calibrate the momentum scale")
    v = np.concatenate(speeds)
    median_v = float(np.median(v))
    if median_v <= 0:
        # Tick-quantized slow products (e.g. /M6E at 0.0001) do not move at
        # all in most 30 s windows, so the median is zero. Use the mean
        # absolute velocity and the Gaussian ratio median/mean = 0.845, which
        # is what the median would be if the moves were not quantized.
        median_v = 0.845 * float(np.mean(v))
    if median_v <= 0:
        raise RuntimeError("the series does not move on this grid; cannot calibrate the momentum scale")
    return median_v / math.atanh(target_norm)


def calibrate_imbalance_scale(segments: list[Segment], window_s: float, step_s: float = 1.0,
                              frac: float = 0.6, target_norm: float = 0.48) -> float | None:
    """Imbalance scale that puts the median |smoothed imbalance| at target_norm.

    The same rule as the momentum scale, applied to the imbalance channel's
    input (the window mean of the top-of-book ratio) on the training rows.
    Returns None when the book carries no sizes (synthetic data, candle
    history), in which case the configured default stands and the channel is
    silent anyway.
    """
    w = max(1, int(round(window_s / step_s)))
    total = sum(len(s) for s in segments)
    budget = int(total * frac)
    vals = []
    for seg in segments:
        take = min(len(seg), budget)
        if take > w:
            size_sum = seg.bid_size[:take] + seg.ask_size[:take]
            with np.errstate(invalid="ignore", divide="ignore"):
                imb = np.where(size_sum > 0, (seg.bid_size[:take] - seg.ask_size[:take]) / size_sum, 0.0)
            smooth = _rolling_mean(imb, w)[w:]
            vals.append(np.abs(smooth[seg.fresh_mask()[w:take]]))
        budget -= take
        if budget <= 0:
            break
    if not vals:
        return None
    median = float(np.median(np.concatenate(vals)))
    if median <= 0:
        return None
    return median / math.atanh(target_norm)


def _segment_from_rows(rows: list[tuple[float, float, float, float, float]],
                       fresh: list[bool] | None = None) -> Segment:
    arr = np.asarray(rows, dtype=np.float64)
    return Segment(
        t=arr[:, 0], bid=arr[:, 1], ask=arr[:, 2],
        mid=(arr[:, 1] + arr[:, 2]) / 2.0,
        bid_size=arr[:, 3], ask_size=arr[:, 4],
        fresh=None if fresh is None else np.asarray(fresh, dtype=bool),
    )


def synthetic_series(cfg: SyntheticCfg, steps: int, step_s: float = 1.0,
                     warm_s: float = 0.0) -> list[Segment]:
    """Generate `steps` usable samples, preceded by `warm_s` of pre-roll.

    The pre-roll plays the same role as the live session's history seed: the
    generated path is long enough that every feature window is defined at the
    first usable row, so the gate evaluates the strip the neurons actually get
    rather than a version whose long end is dark for hours.
    """
    warm_rows = int(math.ceil(warm_s / step_s)) if warm_s > 0 else 0
    total = warm_rows + steps
    source = SyntheticSource(cfg)
    t = np.arange(total, dtype=np.float64) * step_s
    bid = np.empty(total)
    ask = np.empty(total)
    bsz = np.empty(total)
    asz = np.empty(total)
    for i, ti in enumerate(t):
        snap = source.snapshot(float(ti))
        bid[i], ask[i] = snap.bid, snap.ask
        bsz[i], asz[i] = snap.bid_size, snap.ask_size
    return [Segment(t=t, bid=bid, ask=ask, mid=(bid + ask) / 2.0, bid_size=bsz, ask_size=asz,
                    warm_rows=warm_rows)]


# ---------------------------------------------------------------------------

# Candidate momentum ladder: seconds for the game's cadence, then 5m through
# 8h where moves grow large enough to clear round-trip costs.
RETURN_WINDOWS_S = (1.0, 5.0, 15.0, 30.0, 60.0, 90.0, 120.0,
                    300.0, 900.0, 1800.0, 3600.0, 14400.0, 28800.0)
ZRET_WINDOWS_S = (5.0, 30.0, 120.0, 900.0, 3600.0, 14400.0, 28800.0)  # vol-normalized momentum (sigma units)
LEVEL_WINDOWS_S = (300.0, 1800.0)       # near-term level structure windows
IMB_WINDOWS_S = (5.0, 30.0, 120.0)      # book-imbalance smoothing windows
VOL_SHORT_S = 30.0
VOL_LONG_S = 300.0
# Rows discarded at each segment start so every window at or below this is
# fully defined. Longer windows zero-fill until their history exists instead
# of pushing the warmup to 4 h (which would discard short recordings wholesale).
WARMUP_S = 120.0


def longest_feature_window_s(cfg: Config) -> float:
    """History needed before the first row for every column, in both sets, to be live."""
    return max(cfg.neural.encoding.momentum_windows_s[-1], RETURN_WINDOWS_S[-1],
               ZRET_WINDOWS_S[-1], LEVEL_WINDOWS_S[-1], VOL_LONG_S)

EXTENDED_NAMES = (
    [f"ret_{int(w)}s" for w in RETURN_WINDOWS_S]
    + [f"zret_{int(w)}s" for w in ZRET_WINDOWS_S]
    + ["vol_ratio", "spread_ticks", "book_imbalance"]
    + [f"imb_{int(w)}s" for w in IMB_WINDOWS_S]
    + [f"range_pos_{int(w)}s" for w in LEVEL_WINDOWS_S]
    + ["range_width", "tod_sin", "tod_cos"]
)
# The raw book on its own: top-of-book imbalance, instantaneous and smoothed
# over several windows -- a diagnostic for whether the book carries anything
# at these horizons beyond the single smoothed channel the encoder delivers.
IMBALANCE_NAMES = ["book_imbalance"] + [f"imb_{int(w)}s" for w in IMB_WINDOWS_S]


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
    # Warm-up covers the short end of the ladder; longer strip windows (like the
    # long extended columns) zero-fill until their history exists rather than
    # pushing the warmup out to the longest window and discarding whole
    # recordings. Matches the live encoder's "zero = no information".
    warmup = int(max(WARMUP_S, enc_cfg.momentum_windows_s[0]) / step_s) + 1
    max_h = int(max(horizons_s) / step_s)
    # The encoded set is exactly what the encoder sends: the strip, plus the
    # book-imbalance channel when the layout allocates its electrodes (the
    # encoder skips groups absent from the layout, and so does this).
    strip_names = [f"mom_{w:g}s" for w in enc_cfg.momentum_windows_s]
    n_strip = len(strip_names)
    imbalance_channel = "imbalance_bid" in cfg.neural.sensory and "imbalance_ask" in cfg.neural.sensory
    encoded_names = strip_names + ([f"imb_norm_{enc_cfg.imbalance_window_s:g}s"] if imbalance_channel else [])

    xs_enc, xs_ext, xs_imb, meta_rows, seg_ids = [], [], [], [], []
    labels: dict[float, list[np.ndarray]] = {h: [] for h in horizons_s}
    imb_idx = [EXTENDED_NAMES.index(name) for name in IMBALANCE_NAMES]

    for seg_id, seg in enumerate(segments):
        n = len(seg)
        # A segment with pre-roll rows (synthetic) discards those instead of
        # the short warm-up; the windows are then live from the first row kept.
        first = max(warmup, seg.warm_rows)
        if n <= first + max_h:
            continue

        # Encoded features: identical computation to the live game, one column
        # per chronotopic strip window. Idle rows (no quote for stale_s) are
        # skipped exactly as the runner skips them -- snapshot() returns None
        # there and the tracker is not updated -- so the strip meets the same
        # time jump on the first fresh quote that it would live.
        fresh = seg.fresh_mask()
        tracker = FeatureTracker(enc_cfg)
        momentum = np.zeros((n, len(encoded_names)))
        for i in range(n):
            if not fresh[i]:
                momentum[i] = momentum[i - 1] if i else 0.0
                continue
            snap = MarketSnapshot(t=float(seg.t[i]), bid=float(seg.bid[i]), ask=float(seg.ask[i]),
                                  last=float(seg.mid[i]), bid_size=float(seg.bid_size[i]),
                                  ask_size=float(seg.ask_size[i]))
            feats = tracker.update(snap)
            momentum[i, :n_strip] = feats["momentum_norms"]
            if imbalance_channel:
                momentum[i, n_strip] = feats["imbalance_norm"]

        # Extended features, built by name so columns stay aligned with EXTENDED_NAMES.
        eps = 1e-9
        diffs = np.diff(seg.mid, prepend=seg.mid[0])
        vol_s = _rolling_std(diffs, int(VOL_SHORT_S / step_s))
        vol_l = _rolling_std(diffs, int(VOL_LONG_S / step_s))
        cols: dict[str, np.ndarray] = {}

        # Multi-timescale momentum in the live encoder's units, each window on
        # its own 1/sqrt(w) scale -- the same law the chronotopic strip uses,
        # so a candidate column that shows signal here can be added to the
        # strip without recalibration.
        for w_s in RETURN_WINDOWS_S:
            w = int(w_s / step_s)
            ret = np.zeros(n)
            ret[w:] = (seg.mid[w:] - seg.mid[:-w]) / (w * step_s)
            cols[f"ret_{int(w_s)}s"] = np.tanh(ret / enc_cfg.scale_for_window(w_s))

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
        # Smoothed imbalance: the per-moment ratio flips with every one-lot
        # order on a thin micro book; persistent one-sided pressure over
        # seconds-to-minutes is the version with predictive evidence behind
        # it -- and the version a sensory channel would encode.
        for w_s in IMB_WINDOWS_S:
            cols[f"imb_{int(w_s)}s"] = _rolling_mean(cols["book_imbalance"], int(w_s / step_s))

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

        # Labels are computed on the full grid so a horizon is always h seconds
        # of real time; idle rows are then dropped from X and labels, since the
        # live game takes no decision there.
        lo, hi = first, n - max_h
        keep = fresh[lo:hi]
        if not keep.any():
            continue
        xs_enc.append(momentum[lo:hi][keep])
        xs_ext.append(ext[lo:hi][keep])
        xs_imb.append(ext[lo:hi][keep][:, imb_idx])
        meta_rows.append(np.column_stack([seg.t[lo:hi], seg.bid[lo:hi], seg.ask[lo:hi],
                                          seg.mid[lo:hi]])[keep])
        seg_ids.append(np.full(int(keep.sum()), seg_id, dtype=np.int64))

        for h_s in horizons_s:
            h = int(h_s / step_s)
            move = seg.mid[lo + h:hi + h] - seg.mid[lo:hi]
            y = np.where(move > min_move_points, 1, np.where(move < -min_move_points, 0, -1))
            labels[h_s].append(y.astype(np.int64)[keep])

    if not xs_enc:
        raise RuntimeError("no segment is long enough for the requested warmup + horizon")

    meta = np.vstack(meta_rows)
    X_enc, X_ext, X_imb = np.vstack(xs_enc), np.vstack(xs_ext), np.vstack(xs_imb)
    ds = BaselineDataset(
        step_s=step_s,
        horizons_s=list(horizons_s),
        # encoded: what the neurons see. strip: the momentum ladder alone
        # (control for what the imbalance channel adds). extended / imbalance:
        # candidate-column diagnostics.
        feature_names={"encoded": encoded_names, "strip": strip_names,
                       "extended": list(EXTENDED_NAMES), "imbalance": list(IMBALANCE_NAMES)},
        X={"encoded": X_enc, "strip": X_enc[:, :n_strip], "extended": X_ext, "imbalance": X_imb},
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


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Rolling mean, partial windows from the start (vectorized)."""
    n = len(x)
    i = np.arange(n)
    j = np.maximum(0, i - w + 1)
    cnt = (i - j + 1).astype(np.float64)
    cm = np.cumsum(x)
    return (cm - np.where(j > 0, cm[j - 1], 0.0)) / cnt


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
