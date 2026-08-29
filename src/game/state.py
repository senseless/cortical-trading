"""Game state types: actions, trades, and per-step results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..broker.base import Fill


class Action(Enum):
    BUY = "buy"    # close short if short, open long if flat
    SELL = "sell"  # close long if long, open short if flat
    HOLD = "hold"

    @property
    def side(self) -> int:
        return {"buy": 1, "sell": -1, "hold": 0}[self.value]


@dataclass
class Trade:
    direction: int          # +1 long, -1 short
    entry_t: float
    entry_price: float
    exit_t: float = 0.0
    exit_price: float = 0.0
    costs: float = 0.0      # $ commissions+fees, both sides
    points: float = 0.0     # signed points captured (gross)
    net_points: float = 0.0  # points after costs (costs converted at point_value)
    dollars: float = 0.0    # $ PnL after costs

    def close(self, fill: Fill, point_value: float) -> None:
        self.exit_t = fill.t
        self.exit_price = fill.price
        self.costs += fill.commission
        self.points = self.direction * (self.exit_price - self.entry_price)
        self.dollars = self.points * point_value - self.costs
        # point_value > 0 is enforced at config load; a silent gross fallback
        # here would quietly disable cost-awareness in the reward signal.
        self.net_points = self.dollars / point_value


@dataclass
class StepResult:
    action: Action
    executed: bool                    # False if the action was a no-op (e.g. BUY while long)
    position: int                     # position after the action
    prev_position: int
    fills: list[Fill] = field(default_factory=list)
    closed_trade: Trade | None = None
    mtm_points: float = 0.0           # mark-to-market change on prev_position this step
    unrealized_points: float = 0.0
    realized_points: float = 0.0      # session cumulative
    realized_dollars: float = 0.0     # session cumulative, after costs
    mid: float = 0.0
