"""Broker interface: the seam where paper trading becomes live trading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..market.source import MarketSnapshot


@dataclass
class Fill:
    t: float
    side: int          # +1 bought, -1 sold
    price: float
    commission: float  # $ for this side


class Broker(ABC):
    """Executes single-contract market orders. The game engine owns position state."""

    @abstractmethod
    def execute(self, side: int, snap: MarketSnapshot) -> Fill:
        """Execute a market order for one contract. side: +1 buy, -1 sell."""
