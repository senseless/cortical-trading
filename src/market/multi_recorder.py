"""Multi-product recording: one session, one DXLink websocket, N products.

Library-builder counterpart to LiveSource. Each product resolves to its
tradeable front month (same cutoff policy as trading), records into its own
per-contract file, and rolls independently: on a front-month change the feed
unsubscribes the old contract, subscribes the new one, and rotates its
recorder. Since nothing is traded here, rolls apply immediately without a
pending/complete handshake.

dxFeed pushes the last known quote/trade on every (re)subscribe; that
snapshot is history, not live data, and is never recorded. Combined with the
recorder's lazy file creation, reconnect cycles against a closed market
(weekends, CME maintenance) write nothing to disk.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
import traceback

from .recorder import MarketRecorder, default_recording_path

log = logging.getLogger(__name__)


def _f(value) -> float:
    if value is None:
        return math.nan
    return float(value)


class ProductFeed:
    def __init__(self, product: str):
        self.product = product
        self.symbol = ""            # streamer symbol, e.g. /MBTU26:XCME
        self.trading_symbol = ""    # e.g. /MBTU6
        self.desc = ""
        self.recorder: MarketRecorder | None = None
        self.quotes = 0             # events received (incl. subscribe snapshots)
        self.trades = 0
        self.bid = math.nan
        self.ask = math.nan
        self.last = math.nan
        # Per-subscription snapshot skips (touched only on the stream thread).
        self.skip_quote = True
        self.skip_trade = True


class MultiRecorder:
    def __init__(
        self,
        products: list[str],
        record_dir: str = "data/market",
        roll_cutoff_days: float = 7.0,
        roll_check_interval_s: float = 3600.0,
    ):
        self.products = [p.strip().upper() for p in products if p.strip()]
        self.record_dir = record_dir
        self.roll_cutoff_days = roll_cutoff_days
        self.roll_check_interval_s = roll_check_interval_s
        self.feeds: dict[str, ProductFeed] = {}   # keyed by streamer symbol
        self.skipped: list[tuple[str, str]] = []  # (product, reason)
        self.notices: list[str] = []              # roll messages for the driver to print
        self._lock = threading.Lock()
        self._event_count = 0
        self._connected = threading.Event()
        self._resolved = threading.Event()
        self._error: BaseException | None = None
        self._error_tb = ""
        self._stop: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._streamer = None
        self._Quote = self._Trade = None

    # -- lifecycle -----------------------------------------------------------

    def start(self, timeout_s: float = 90.0) -> None:
        from .tasty import build_session

        session = build_session()
        self._thread = threading.Thread(target=self._thread_main, args=(session,), daemon=True,
                                        name="dxlink-multi")
        self._thread.start()
        if not self._connected.wait(timeout=timeout_s):
            self.stop()
            if self._error:
                detail = f"\n{self._error_tb}" if self._error_tb else ""
                raise RuntimeError(f"multi recorder failed: {self._error}{detail}") from self._error
            raise TimeoutError(f"no market data received for any of {self.products} within {timeout_s}s")

    def stop(self) -> None:
        if self._loop and self._stop and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._stop.set)
            except RuntimeError:
                pass
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        with self._lock:
            for feed in self.feeds.values():
                if feed.recorder:
                    feed.recorder.close()

    def _thread_main(self, session) -> None:
        try:
            asyncio.run(self._stream(session))
        except BaseException as exc:
            self._error = exc
            self._error_tb = traceback.format_exc()

    # -- streaming -----------------------------------------------------------

    async def _stream(self, session) -> None:
        from tastytrade import DXLinkStreamer
        from tastytrade.dxfeed import Quote, Trade

        from .tasty import days_until_stop, resolve_trading_contract

        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        self._Quote, self._Trade = Quote, Trade

        for code in self.products:
            try:
                contract, note = await resolve_trading_contract(session, code, self.roll_cutoff_days)
            except Exception as exc:
                self.skipped.append((code, str(exc)))
                continue
            feed = ProductFeed(code)
            feed.symbol = contract.streamer_symbol
            feed.trading_symbol = contract.symbol
            feed.desc = (f"{contract.symbol} ({contract.streamer_symbol}) "
                         f"stops trading {contract.expiration_date} ({days_until_stop(contract):.1f}d)"
                         + (f" | {note}" if note else ""))
            feed.recorder = MarketRecorder(
                default_recording_path(feed.symbol, self.record_dir), symbol=feed.symbol,
                meta={"trading_symbol": feed.trading_symbol, "product": code},
            )
            with self._lock:
                self.feeds[feed.symbol] = feed
        self._resolved.set()
        if not self.feeds:
            raise RuntimeError(f"no products resolved; skipped: {self.skipped}")

        async with DXLinkStreamer(session) as streamer:
            self._streamer = streamer
            symbols = list(self.feeds)
            await streamer.subscribe(Quote, symbols)
            await streamer.subscribe(Trade, symbols)
            tasks = {
                asyncio.create_task(self._listen_quotes(streamer, Quote)),
                asyncio.create_task(self._listen_trades(streamer, Trade)),
                asyncio.create_task(self._watch_rolls(session)),
            }
            stop_task = asyncio.create_task(self._stop.wait())
            done, pending = await asyncio.wait(tasks | {stop_task}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                if task is not stop_task and task.exception():
                    raise task.exception()
            if stop_task not in done:
                raise RuntimeError("market data stream ended unexpectedly")

    async def _listen_quotes(self, streamer, Quote) -> None:
        async for q in streamer.listen(Quote):
            now = time.time()
            with self._lock:
                feed = self.feeds.get(q.event_symbol)
            if feed is None:
                continue
            bid, ask = _f(q.bid_price), _f(q.ask_price)
            if math.isnan(bid) or math.isnan(ask):
                continue
            with self._lock:
                feed.bid, feed.ask = bid, ask
                feed.quotes += 1
                self._event_count += 1
            self._connected.set()
            if feed.skip_quote:
                # dxFeed's subscribe snapshot: valid book, wrong time; the
                # in-memory state uses it but the recording must not.
                feed.skip_quote = False
            else:
                feed.recorder.quote(now, bid, ask, _f(q.bid_size), _f(q.ask_size))

    async def _listen_trades(self, streamer, Trade) -> None:
        async for tr in streamer.listen(Trade):
            now = time.time()
            with self._lock:
                feed = self.feeds.get(tr.event_symbol)
            if feed is None:
                continue
            price = _f(tr.price)
            if math.isnan(price):
                continue
            with self._lock:
                feed.last = price
                feed.trades += 1
                self._event_count += 1
            self._connected.set()
            if feed.skip_trade:
                feed.skip_trade = False  # subscribe snapshot, see quotes
            else:
                feed.recorder.trade(now, price, _f(tr.size))

    async def _watch_rolls(self, session) -> None:
        from .tasty import days_until_stop, resolve_trading_contract

        while True:
            await asyncio.sleep(self.roll_check_interval_s)
            # The session refreshes its own access token per request.
            with self._lock:
                feeds = list(self.feeds.values())
            for feed in feeds:
                try:
                    contract, note = await resolve_trading_contract(session, feed.product,
                                                                    self.roll_cutoff_days)
                except Exception as exc:
                    log.warning("roll check failed for %s: %s", feed.product, exc)
                    continue
                if contract.streamer_symbol == feed.symbol:
                    continue
                old = feed.symbol
                await self._streamer.unsubscribe(self._Quote, [old])
                await self._streamer.unsubscribe(self._Trade, [old])
                with self._lock:
                    del self.feeds[old]
                    feed.symbol = contract.streamer_symbol
                    feed.trading_symbol = contract.symbol
                    feed.desc = (f"{contract.symbol} ({contract.streamer_symbol}) "
                                 f"stops trading {contract.expiration_date} "
                                 f"({days_until_stop(contract):.1f}d)")
                    feed.bid = feed.ask = feed.last = math.nan
                    self.feeds[feed.symbol] = feed
                # Rotate before subscribing so the new contract's first fresh
                # events land in the new file (one file never spans two
                # contracts); the subscribe snapshot itself is skipped.
                feed.recorder.rotate(feed.symbol, default_recording_path(feed.symbol, self.record_dir),
                                     meta={"trading_symbol": feed.trading_symbol,
                                           "product": feed.product, "rolled_from": old})
                feed.skip_quote = feed.skip_trade = True
                await self._streamer.subscribe(self._Quote, [feed.symbol])
                await self._streamer.subscribe(self._Trade, [feed.symbol])
                with self._lock:
                    self.notices.append(f"contract roll [{feed.product}]: {old} -> {feed.symbol}"
                                        + (f" ({note})" if note else ""))

    # -- driver interface ------------------------------------------------------

    @property
    def total_events(self) -> int:
        with self._lock:
            return self._event_count

    def check_error(self) -> None:
        if self._error:
            detail = f"\n{self._error_tb}" if self._error_tb else ""
            raise RuntimeError(f"multi recorder stream failed: {self._error}{detail}") from self._error

    def drain_notices(self) -> list[str]:
        with self._lock:
            out, self.notices = self.notices, []
            return out

    def describe_lines(self) -> list[str]:
        with self._lock:
            return [f"{f.product:5s} {f.desc}" for f in self.feeds.values()]

    def status_line(self) -> str:
        with self._lock:
            parts = []
            for f in self.feeds.values():
                px = f"{f.bid:g}/{f.ask:g}" if not math.isnan(f.bid) else "-"
                parts.append(f"{f.product} q{f.quotes} t{f.trades} {px}")
            return " | ".join(parts)

    def flush_all(self) -> None:
        with self._lock:
            recorders = [f.recorder for f in self.feeds.values() if f.recorder]
        for rec in recorders:
            rec.flush()
