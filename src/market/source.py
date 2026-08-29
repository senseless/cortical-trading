"""Market source interface and the snapshot type shared by all sources."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PendingRoll:
    """A contract roll detected by a live source, awaiting completion.

    The caller must flatten any open position and reset price-derived state
    (feature windows, mark-to-market anchor) before calling complete_roll().
    """

    old_streamer_symbol: str
    new_streamer_symbol: str
    new_trading_symbol: str
    note: str = ""


@dataclass
class MarketSnapshot:
    t: float  # source-relative time (seconds) for synthetic/replay, epoch seconds for live
    bid: float
    ask: float
    last: float
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


class MarketSource:
    """Pull-based market interface driven by the game clock.

    The game engine owns time and calls snapshot(t) once per game tick with the
    session-relative time in seconds. Synthetic and replay sources are
    deterministic functions of t; the live source ignores t and returns the
    latest streamed state.
    """

    def start(self) -> None:  # noqa: B027 - optional hook
        pass

    def stop(self) -> None:  # noqa: B027 - optional hook
        pass

    def snapshot(self, t: float) -> MarketSnapshot | None:
        raise NotImplementedError

    def pending_roll(self) -> PendingRoll | None:
        """Non-live sources never roll."""
        return None

    def complete_roll(self) -> None:  # noqa: B027 - optional hook
        pass

    def describe(self) -> str:
        return type(self).__name__

    def __enter__(self) -> "MarketSource":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
