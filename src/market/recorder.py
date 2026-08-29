"""Append-only JSONL(.gz) recorder building the replay library.

One file never spans two contracts: on a roll the live source calls rotate(),
which closes the current file and starts a new one for the new symbol. All
methods are thread-safe (writes come from the stream thread, flushes may come
from the caller's thread).
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
        self._fh = _open(self.path)
        self.quotes = 0  # cumulative across rotations
        self.trades = 0
        self._write_meta(symbol, meta)

    def _write_meta(self, symbol: str, meta: dict | None) -> None:
        header = {"type": "meta", "symbol": symbol, "recorded_at": time.time()}
        if meta:
            header.update(meta)
        self._write(header)

    def _write(self, obj: dict) -> None:
        with self._lock:
            self._fh.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def quote(self, t: float, bid: float, ask: float, bid_size: float = 0.0, ask_size: float = 0.0) -> None:
        self._write({"type": "quote", "t": t, "bid": bid, "ask": ask, "bs": bid_size, "as": ask_size})
        self.quotes += 1

    def trade(self, t: float, price: float, size: float = 0.0) -> None:
        self._write({"type": "trade", "t": t, "price": price, "size": size})
        self.trades += 1

    def rotate(self, symbol: str, path: str | Path, meta: dict | None = None) -> None:
        """Close the current file and continue into a new one (contract roll)."""
        with self._lock:
            self._fh.close()
            self.path = Path(path)
            self._fh = _open(self.path)
        self._write_meta(symbol, meta)

    def flush(self) -> None:
        with self._lock:
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def default_recording_path(symbol: str, base_dir: str | Path = "data/market") -> Path:
    safe = symbol.replace("/", "").replace(":", "_")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path(base_dir) / f"{safe}_{stamp}.jsonl.gz"
