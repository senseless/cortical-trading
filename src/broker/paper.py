"""Paper broker: instant deterministic fills at the live bid/ask plus slippage.

Deliberately used instead of the tastytrade sandbox: fills are immediate (the
reward stimulus must follow the action promptly for the neurons to associate
them) and fully under our control.
"""

from __future__ import annotations

from ..config import BrokerCfg, InstrumentCfg
from ..market.source import MarketSnapshot
from .base import Broker, Fill


class PaperBroker(Broker):
    def __init__(self, cfg: BrokerCfg, instrument: InstrumentCfg):
        self.cfg = cfg
        self.instrument = instrument

    def execute(self, side: int, snap: MarketSnapshot) -> Fill:
        slip = self.cfg.slippage_ticks * self.instrument.tick_size
        price = (snap.ask + slip) if side > 0 else (snap.bid - slip)
        return Fill(t=snap.t, side=side, price=price, commission=self.cfg.commission_per_side)
