"""Live tastytrade futures broker (Phase 4).

Deliberately not implemented yet: live order routing is gated on Phase 3 --
real neurons must beat random and shuffled-feedback controls across repeated
sessions first. The interface is final; only this class changes at cutover.
"""

from __future__ import annotations

from ..config import BrokerCfg, InstrumentCfg
from ..market.source import MarketSnapshot
from .base import Broker, Fill


class TastytradeBroker(Broker):
    def __init__(self, cfg: BrokerCfg, instrument: InstrumentCfg):
        raise NotImplementedError(
            "Live order routing is Phase 4 and gated on the Phase 3 evaluation "
            "(beat controls across repeated real-neuron sessions). Use broker.kind='paper'."
        )

    def execute(self, side: int, snap: MarketSnapshot) -> Fill:  # pragma: no cover
        raise NotImplementedError
