from .base import Broker, Fill
from .paper import PaperBroker

__all__ = ["Broker", "Fill", "PaperBroker", "build_broker"]


def build_broker(cfg):
    if cfg.broker.kind == "paper":
        return PaperBroker(cfg.broker, cfg.instrument)
    if cfg.broker.kind == "tastytrade":
        from .tastytrade_broker import TastytradeBroker

        return TastytradeBroker(cfg.broker, cfg.instrument)
    raise ValueError(f"unknown broker kind: {cfg.broker.kind}")
