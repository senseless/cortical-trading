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
    """Rolling market features derived from snapshots at each game step."""

    def __init__(self, cfg: EncodingCfg):
        self.cfg = cfg
        self._hist: deque[tuple[float, float]] = deque()

    def reset(self) -> None:
        self._hist.clear()

    def update(self, snap: MarketSnapshot) -> dict[str, float]:
        self._hist.append((snap.t, snap.mid))
        horizon = max(self.cfg.momentum_window_s * 2, 60.0)
        while self._hist and snap.t - self._hist[0][0] > horizon:
            self._hist.popleft()

        momentum_pps = 0.0
        window = self.cfg.momentum_window_s
        # Momentum is defined only once a full window of history exists. Right
        # after a reset the anchor is seconds old, and (mid - anchor)/elapsed
        # over ~1 s is dominated by noise ~6x the calibrated momentum_scale:
        # the tanh would saturate in a random direction for the first ~window
        # seconds of every episode. Until then, report zero (no information).
        if self._hist and snap.t - self._hist[0][0] >= window:
            anchor = None
            for t, mid in self._hist:
                if snap.t - t <= window:
                    anchor = (t, mid)
                    break
            if anchor and snap.t > anchor[0]:
                momentum_pps = (snap.mid - anchor[1]) / (snap.t - anchor[0])

        # Volatility of ~step-scale moves; skip pairs spanning a rest/roll gap
        # (steps are ~1 s apart -- a > 5 s jump is not a market move).
        pts = list(self._hist)
        diffs = [b[1] - a[1] for a, b in zip(pts[:-1], pts[1:]) if b[0] - a[0] <= 5.0]
        vol = statistics.pstdev(diffs) if len(diffs) >= 2 else 0.0

        return {
            "mid": snap.mid,
            "spread": snap.spread,
            "momentum_pps": momentum_pps,
            "momentum_norm": math.tanh(momentum_pps / self.cfg.momentum_scale) if self.cfg.momentum_scale else 0.0,
            "vol_points": vol,
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
