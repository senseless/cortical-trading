from .source import MarketSnapshot, MarketSource
from .synthetic import SyntheticSource
from .replay import ReplaySource

__all__ = ["MarketSnapshot", "MarketSource", "SyntheticSource", "ReplaySource", "build_source"]


def build_source(cfg):
    """Construct the configured MarketSource from a Config."""
    kind = cfg.market.source
    if kind == "synthetic":
        return SyntheticSource(cfg.market.synthetic)
    if kind == "replay":
        return ReplaySource(cfg.market.replay)
    if kind == "live":
        from .live import LiveSource  # deferred: pulls in tastytrade SDK

        return LiveSource(cfg.market.live)
    raise ValueError(f"unknown market source: {kind}")
