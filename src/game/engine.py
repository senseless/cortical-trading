"""Game engine: the market world the neurons play in.

Owns the position state machine (short -1 / flat 0 / long +1, one contract),
mark-to-market PnL, and the trade log. Also computes the market features the
encoder turns into stimulation.
"""

from __future__ import annotations

import math
import statistics
from collections import deque

from ..broker.base import Broker
from ..config import EncodingCfg, InstrumentCfg
from ..market.source import MarketSnapshot
from .state import Action, StepResult, Trade


class FeatureTracker:
    """Rolling market features derived from snapshots at each game step.

    Momentum is computed over the configured ladder of windows -- the
    chronotopic strip -- one velocity per window, each normalized by its own
    scale (see EncodingCfg.scale_for_window). A window reports exactly zero
    until that much market history exists: right after a reset the anchor is
    seconds old, and (mid - anchor)/elapsed over ~1 s is noise several times
    the calibrated scale, so the tanh would saturate in a random direction.
    Zero means "no information" and the encoder renders it as silence, so the
    strip's long end simply stays dark until it has the history to speak.

    The same saturation would follow any *gap* in the samples (a stale-quote
    idle through CME maintenance, a stream reconnect): total history is still
    long, but the oldest sample inside a short window is seconds old, so the
    velocity is again noise over a tiny span. Hence a window also stays silent
    until its anchor spans at least MIN_SPAN_FRAC of the window -- silence,
    not a random-direction burst, is what the culture gets after a gap.

    With hour-scale windows on the ladder, waiting for in-session history is
    not an option (nobody rents 8 hours of wetware warm-up), so seed() accepts
    pre-roll history from the market source -- recorded data before the replay
    offset, backfilled candles on live, generated path on synthetic -- and the
    whole strip is live from the first step.

    Window anchors advance monotonically (one cursor per window) instead of
    rescanning history, because the ladder retains hours of quotes and the
    baseline dataset builder calls update() once per row of a recording.

    The book-imbalance strip is the same construction on a different
    quantity: the time average of the top-of-book imbalance over each window
    of its own (seconds-scale) ladder, with the same silence rule.
    """

    VOL_WINDOW_S = 60.0     # volatility is a step-scale statistic, not part of the ladder
    # A window speaks only when its anchor is at least this fraction of the
    # window old. Normal 1 Hz sampling gives ~1.0; a feedback pause dents the
    # 30 s window to ~0.87; 1-minute pre-roll candles anchor the 5-minute
    # window at >= 0.8. Anything shorter is a post-gap span whose noise would
    # saturate the tanh (scale mismatch at the threshold is only 1/sqrt(0.75)).
    MIN_SPAN_FRAC = 0.75
    _COMPACT_AT = 4096      # drop consumed history once this many samples are dead

    def __init__(self, cfg: EncodingCfg):
        self.cfg = cfg
        self.windows: list[float] = list(cfg.momentum_windows_s)
        self.scales: list[float] = cfg.momentum_scales
        self.imb_windows: list[float] = list(cfg.imbalance_windows_s)
        self.imb_scales: list[float] = list(cfg.imbalance_scales)
        self._t: list[float] = []
        self._mid: list[float] = []
        self._imb: list[float] = []                 # top-of-book imbalance per sample
        self._head = 0                              # oldest live sample
        self._cursor = [0] * len(self.windows)      # per-window anchor
        # Imbalance strip: each window keeps the index of its oldest sample
        # and a running sum of the samples from there to the present, so the
        # window mean is O(1) per step (the dataset builder calls update()
        # once per second of a week-long recording).
        self._imb_cursor = [0] * len(self.imb_windows)
        self._imb_sum = [0.0] * len(self.imb_windows)
        self._vol: deque[tuple[float, float]] = deque()

    def reset(self) -> None:
        self._t.clear()
        self._mid.clear()
        self._imb.clear()
        self._head = 0
        self._cursor = [0] * len(self.windows)
        self._imb_cursor = [0] * len(self.imb_windows)
        self._imb_sum = [0.0] * len(self.imb_windows)
        self._vol.clear()

    def _compact(self) -> None:
        """Drop consumed history; every index into the sample lists shifts."""
        del self._t[:self._head]
        del self._mid[:self._head]
        del self._imb[:self._head]
        self._cursor = [max(0, c - self._head) for c in self._cursor]
        self._imb_cursor = [max(0, c - self._head) for c in self._imb_cursor]
        self._head = 0
        # Re-sum the imbalance windows exactly; the running sums drift by a
        # rounding error per step and this is the natural place to reset it.
        self._imb_sum = [sum(self._imb[c:]) for c in self._imb_cursor]

    def seed(self, snaps: list[MarketSnapshot]) -> None:
        """Prepend pre-roll history so the long windows speak immediately.

        Only snapshots older than the oldest live sample are used: a live
        source's backfill arrives seconds after real quotes started flowing
        (async candle fetch after a roll), and appending those older times
        would break the monotonic-time assumption the cursors rely on.
        Volatility is left untouched -- it is a step-scale statistic and
        pre-roll history is far coarser than the step grid.
        """
        if self._head:
            # Cursors index the lists; keep them valid even if nothing is
            # prepended below (an early return with stale indices would make
            # the next update() read past the end).
            self._compact()
        cut = self._t[0] if self._t else math.inf
        hist = sorted((s.t, s.mid) for s in snaps if s.t < cut)
        if not hist:
            return
        self._t[:0] = [t for t, _ in hist]
        self._mid[:0] = [m for _, m in hist]
        # Candle history carries no book sizes: the prepended samples read a
        # balanced book (0), and since they are older than every imbalance
        # window they fall out of the sums on the next update anyway.
        self._imb[:0] = [0.0] * len(hist)
        self._cursor = [0] * len(self.windows)
        self._imb_cursor = [0] * len(self.imb_windows)
        self._imb_sum = [sum(self._imb)] * len(self.imb_windows)

    def update(self, snap: MarketSnapshot) -> dict[str, float | list[float]]:
        # Sources without sizes (candle pre-roll, synthetic) read a balanced
        # book: exactly zero, which the encoder renders as silence.
        size_sum = snap.bid_size + snap.ask_size
        imb_now = (snap.bid_size - snap.ask_size) / size_sum if size_sum > 0 else 0.0
        self._t.append(snap.t)
        self._mid.append(snap.mid)
        self._imb.append(imb_now)
        for k in range(len(self._imb_sum)):
            self._imb_sum[k] += imb_now     # before any compaction re-sums the lists
        now, mid = snap.t, snap.mid

        # Retain a little more than the longest window so its anchor exists.
        retain = max(self.windows[-1], self.imb_windows[-1]) + 60.0
        n = len(self._t)
        while self._head < n - 1 and now - self._t[self._head] > retain:
            self._head += 1
        if self._head >= self._COMPACT_AT:
            self._compact()
            n = len(self._t)

        history_s = now - self._t[self._head]
        pps: list[float] = []
        norms: list[float] = []
        for k, window in enumerate(self.windows):
            c = max(self._cursor[k], self._head)
            while c + 1 < n and now - self._t[c] > window:
                c += 1
            self._cursor[k] = c
            v = 0.0
            span = now - self._t[c]
            if history_s >= window and span > 0 and span >= self.MIN_SPAN_FRAC * window:
                v = (mid - self._mid[c]) / span
            pps.append(v)
            scale = self.scales[k]
            norms.append(math.tanh(v / scale) if scale else 0.0)

        # Volatility of ~step-scale moves; skip pairs spanning a rest/roll gap
        # (steps are ~1 s apart -- a > 5 s jump is not a market move).
        self._vol.append((now, mid))
        while self._vol and now - self._vol[0][0] > self.VOL_WINDOW_S:
            self._vol.popleft()
        pts = list(self._vol)
        diffs = [b[1] - a[1] for a, b in zip(pts[:-1], pts[1:]) if b[0] - a[0] <= 5.0]
        vol = statistics.pstdev(diffs) if len(diffs) >= 2 else 0.0

        # Book imbalance strip: the time average of the top-of-book ratio over
        # each window of its ladder. Same silence rule as the momentum strip --
        # a window speaks once it holds three quarters of its span, so a
        # post-gap or fresh-start value (one or two one-lot quotes) is not
        # rendered as a saturated burst in a random direction. A window holds
        # the samples younger than its length (w samples at 1 Hz, the same
        # definition the gate's calibration uses), so a window no longer than
        # the sampling interval (the 1 s window at 1 Hz) holds exactly the
        # current sample and is live at once: its mean *is* the present book,
        # which is the right reading, not a post-gap artifact.
        dt_last = now - self._t[n - 2] if n >= 2 else 0.0  # unknown cadence on the very first sample: silence
        imb_means: list[float] = []
        imb_norms: list[float] = []
        for k, window in enumerate(self.imb_windows):
            c = self._imb_cursor[k]
            while c < n - 1 and now - self._t[c] >= window:
                self._imb_sum[k] -= self._imb[c]
                c += 1
            self._imb_cursor[k] = c
            count = n - c
            span = now - self._t[c]
            mean = 0.0
            if span >= self.MIN_SPAN_FRAC * window or (count == 1 and window <= dt_last):
                mean = self._imb_sum[k] / count
            imb_means.append(mean)
            scale = self.imb_scales[k]
            imb_norms.append(math.tanh(mean / scale) if scale else 0.0)

        return {
            "mid": mid,
            "spread": snap.spread,
            "momentum_pps": pps,
            "momentum_norms": norms,
            "vol_points": vol,
            "imbalance": imb_now,
            "imbalance_means": imb_means,
            "imbalance_norms": imb_norms,
        }


class GameEngine:
    def __init__(self, instrument: InstrumentCfg, broker: Broker):
        self.instrument = instrument
        self.broker = broker
        self.position = 0
        self.open_trade: Trade | None = None
        self.trades: list[Trade] = []
        self.realized_points = 0.0
        self.realized_dollars = 0.0
        self.total_costs = 0.0
        self._last_mid: float | None = None

    # -- queries ---------------------------------------------------------------

    @property
    def last_mid(self) -> float | None:
        """Mid of the most recent mark-to-market, if any."""
        return self._last_mid

    def unrealized_points(self, mid: float) -> float:
        if self.position == 0 or self.open_trade is None:
            return 0.0
        return self.open_trade.direction * (mid - self.open_trade.entry_price)

    # -- transitions -------------------------------------------------------------

    def step(self, action: Action, snap: MarketSnapshot) -> StepResult:
        """Mark to market on the held position, then apply the action."""
        prev_position = self.position
        mtm = 0.0
        if self._last_mid is not None:
            mtm = prev_position * (snap.mid - self._last_mid)
        self._last_mid = snap.mid

        result = StepResult(
            action=action, executed=False, position=self.position,
            prev_position=prev_position, mtm_points=mtm, mid=snap.mid,
        )

        side = action.side
        if side != 0 and side != self.position:
            fill = self.broker.execute(side, snap)
            result.fills.append(fill)
            result.executed = True
            self.total_costs += fill.commission
            if self.position == 0:
                self.open_trade = Trade(
                    direction=side, entry_t=fill.t, entry_price=fill.price, costs=fill.commission,
                )
                self.position = side
            else:
                trade = self.open_trade
                trade.close(fill, self.instrument.point_value)
                self.trades.append(trade)
                self.realized_points += trade.points
                self.realized_dollars += trade.dollars
                result.closed_trade = trade
                self.open_trade = None
                self.position = 0

        result.position = self.position
        result.unrealized_points = self.unrealized_points(snap.mid)
        result.realized_points = self.realized_points
        result.realized_dollars = self.realized_dollars
        return result

    def flatten(self, snap: MarketSnapshot) -> StepResult | None:
        """Force-close any open position (episode end or contract roll)."""
        if self.position == 0:
            return None
        action = Action.SELL if self.position > 0 else Action.BUY
        return self.step(action, snap)

    def on_roll(self) -> None:
        """Re-anchor after a contract roll (position must already be flat).

        Forgetting the last mid prevents the basis jump between contracts from
        appearing as a mark-to-market move on the first post-roll step.
        """
        self._last_mid = None
