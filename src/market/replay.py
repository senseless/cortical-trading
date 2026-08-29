"""Replay a recorded market session as a deterministic MarketSource.

Streams the JSONL file forward as the game clock advances, so arbitrarily long
recordings replay in constant memory. Supports playback speed multipliers and
a start offset so the same segment can be presented to different cultures or
configurations for clean comparisons.
"""

from __future__ import annotations

import gzip
import json
import math
import zlib
from pathlib import Path

from ..config import ReplayCfg
from .source import MarketSnapshot, MarketSource


class ReplaySource(MarketSource):
    def __init__(self, cfg: ReplayCfg):
        if not cfg.path:
            raise ValueError("market.replay.path is not set")
        self.cfg = cfg
        self.path = Path(cfg.path)
        if not self.path.exists():
            raise FileNotFoundError(f"replay file not found: {self.path}")
        self._fh = None
        self._t0: float | None = None  # recording timeline origin
        self._pending: dict | None = None
        self._bid = self._ask = self._last = float("nan")
        self._bid_size = self._ask_size = 0.0
        self.symbol = "?"
        self.events_consumed = 0

    def describe(self) -> str:
        return f"replay:{self.path.name} speed={self.cfg.speed} offset={self.cfg.start_offset_s}s"

    def start(self) -> None:
        self._fh = gzip.open(self.path, "rt", encoding="utf-8") if self.path.suffix == ".gz" \
            else open(self.path, "r", encoding="utf-8")
        first = self._next_event()
        if first and first.get("type") == "meta":
            self.symbol = first.get("symbol", "?")
            first = self._next_event()
        if first is None:
            raise ValueError(f"replay file {self.path} contains no events")
        self._t0 = float(first["t"])
        # Warm up through any leading trade events until the first quote, so
        # snapshot() never emits NaN bid/ask.
        ev = first
        while True:
            self._apply(ev)
            if not (math.isnan(self._bid) or math.isnan(self._ask)):
                break
            ev = self._next_event()
            if ev is None:
                raise ValueError(f"replay file {self.path} contains no quote events")
        self._pending = None
        if self.cfg.start_offset_s > 0:
            # Seek to the offset now: done lazily, the first in-loop snapshot()
            # would decompress and parse potentially hours of events inside a
            # 50 ms tick budget. This also makes an offset beyond the end of
            # the recording fail here, before any wetware time is spent.
            self.snapshot(0.0)

    def stop(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def _next_event(self) -> dict | None:
        while True:
            try:
                line = self._fh.readline()
            except (EOFError, zlib.error):
                # Truncated or corrupted gzip tail: recorder was killed
                # mid-write; everything decompressed so far is good data.
                return None
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                # Torn line from a crashed recorder -- everything before it is
                # good data; treat it as the end of the recording.
                return None

    def _apply(self, ev: dict) -> None:
        if ev["type"] == "quote":
            self._bid = float(ev["bid"])
            self._ask = float(ev["ask"])
            self._bid_size = float(ev.get("bs", 0.0))
            self._ask_size = float(ev.get("as", 0.0))
        elif ev["type"] == "trade":
            self._last = float(ev["price"])
        self.events_consumed += 1

    def snapshot(self, t: float) -> MarketSnapshot | None:
        if self._t0 is None:
            raise RuntimeError("ReplaySource.start() not called")
        target = self._t0 + self.cfg.start_offset_s + t * self.cfg.speed
        ev = self._pending
        self._pending = None
        while True:
            if ev is None:
                ev = self._next_event()
                if ev is None:
                    # The session has outlived the recording. Holding the last
                    # quote would silently simulate a frozen market; fail loudly
                    # so the session finalizes instead.
                    raise RuntimeError(
                        f"replay exhausted: {self.path.name} has no data past "
                        f"t={target - self._t0:.1f}s of the recording; shorten the "
                        "session or use a longer recording")
            if "t" not in ev:
                ev = None  # mid-file meta (recording path was reused/appended)
                continue
            if float(ev["t"]) > target:
                self._pending = ev
                break
            self._apply(ev)
            ev = None
        if self._last != self._last:  # NaN: no trade seen yet
            self._last = (self._bid + self._ask) / 2.0
        return MarketSnapshot(t=t, bid=self._bid, ask=self._ask, last=self._last,
                              bid_size=self._bid_size, ask_size=self._ask_size)
