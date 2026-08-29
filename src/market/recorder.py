"""Append-only JSONL(.gz) recorder building the replay library.

File creation is lazy: the file (with its meta header) is only created when
the first event is written, so a connected-but-idle market (weekend close,
CME maintenance) never litters the library with empty files -- reconnect
cycles cost no disk writes at all.

One file never spans two contracts: on a roll the live source calls rotate(),
which closes the current file and points at a new path for the new symbol.
All methods are thread-safe (writes come from the stream thread, flushes may
come from the caller's thread).
"""

from __future__ import annotations

import gzip
import json
import threading
import time
from pathlib import Path


def _open(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".gz"):
        return gzip.open(path, "at", encoding="utf-8")
    return open(path, "a", encoding="utf-8")


class MarketRecorder:
    def __init__(self, path: str | Path, symbol: str, meta: dict | None = None):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._fh = None            # created on first event write
        self._symbol = symbol
        self._meta = dict(meta) if meta else {}
        self._closed = False
        self.quotes = 0  # cumulative across rotations (written events only)
        self.trades = 0

    @property
    def created(self) -> bool:
        """True once the current file exists (at least one event was written)."""
        with self._lock:
            return self._fh is not None

    def _write(self, obj: dict) -> None:
        with self._lock:
            if self._closed:
                return  # straggler write after close(); do not resurrect the file
            if self._fh is None:
                self._fh = _open(self.path)
                header = {"type": "meta", "symbol": self._symbol,
                          "recorded_at": time.time(), **self._meta}
                self._fh.write(json.dumps(header, separators=(",", ":")) + "\n")
            self._fh.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def quote(self, t: float, bid: float, ask: float, bid_size: float = 0.0, ask_size: float = 0.0) -> None:
        self._write({"type": "quote", "t": t, "bid": bid, "ask": ask, "bs": bid_size, "as": ask_size})
        self.quotes += 1

    def trade(self, t: float, price: float, size: float = 0.0) -> None:
        self._write({"type": "trade", "t": t, "price": price, "size": size})
        self.trades += 1

    def rotate(self, symbol: str, path: str | Path, meta: dict | None = None) -> None:
        """Close the current file (if any) and continue into a new one (contract roll)."""
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            self.path = Path(path)
            self._symbol = symbol
            self._meta = dict(meta) if meta else {}

    def flush(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def default_recording_path(symbol: str, base_dir: str | Path = "data/market") -> Path:
    safe = symbol.replace("/", "").replace(":", "_")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path(base_dir) / f"{safe}_{stamp}.jsonl.gz"
