"""Live market source: tastytrade DXLink websocket in a background thread.

The asyncio streaming loop runs in a daemon thread and continuously updates a
lock-protected latest-state snapshot; the CL closed loop reads it without
blocking. An optional MarketRecorder archives every event into the replay
library while streaming.

Contract rolls: when the symbol is auto-resolved from a product code, a
watcher re-resolves it every roll_check_interval_s (honoring roll_cutoff_days,
see tasty.py). When the answer changes, pending_roll() becomes non-None; the
driver must flatten its position and reset price-derived state, then call
complete_roll(), which atomically resubscribes to the new contract, clears the
stale quote, and rotates the recorder into a fresh file.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
import traceback

from ..config import LiveCfg
from .recorder import MarketRecorder, default_recording_path
from .source import MarketSnapshot, MarketSource, PendingRoll


def _f(value) -> float:
    if value is None:
        return math.nan
    return float(value)


class LiveSource(MarketSource):
    def __init__(self, cfg: LiveCfg, recorder: MarketRecorder | None = None):
        self.cfg = cfg
        self.recorder = recorder
        self.record_dir = ""  # set before start() to auto-create a recorder
        self.record_path = ""  # explicit first-file path (overrides record_dir naming)
        self.symbol = cfg.streamer_symbol
        self.trading_symbol = ""
        self.contract_desc = cfg.streamer_symbol
        self._watch_rolls = not cfg.streamer_symbol  # pinned symbols never roll
        self._lock = threading.Lock()
        self._bid = self._ask = self._last = math.nan
        self._bid_size = self._ask_size = 0.0
        self._quote_count = 0
        self._trade_count = 0
        self._pending_contract = None  # (Future, note) under _lock
        self._connected = threading.Event()
        self._error: BaseException | None = None
        self._error_tb: str = ""
        self._stop: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._streamer = None
        self._Quote = self._Trade = None

    def describe(self) -> str:
        return f"live:{self.symbol or self.cfg.product_code}"

    # -- lifecycle -----------------------------------------------------------

    def start(self, timeout_s: float = 30.0) -> None:
        from .tasty import build_session

        session = build_session()
        self._thread = threading.Thread(target=self._thread_main, args=(session,), daemon=True, name="dxlink")
        self._thread.start()
        if not self._connected.wait(timeout=timeout_s):
            self.stop()
            if self._error:
                detail = f"\n{self._error_tb}" if self._error_tb else ""
                raise RuntimeError(f"DXLink connection failed: {self._error}{detail}") from self._error
            raise TimeoutError(
                f"no market data received for {self.symbol or self.cfg.product_code} within {timeout_s}s"
            )

    def stop(self) -> None:
        if self._loop and self._stop and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._stop.set)
            except RuntimeError:
                pass  # loop shut down between the check and the call
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        if self.recorder:
            self.recorder.close()

    def _thread_main(self, session) -> None:
        try:
            asyncio.run(self._stream(session))
        except BaseException as exc:  # surface to the main thread
            self._error = exc
            self._error_tb = traceback.format_exc()

    async def _stream(self, session) -> None:
        from tastytrade import DXLinkStreamer
        from tastytrade.dxfeed import Quote, Trade

        from .tasty import days_until_stop, resolve_trading_contract

        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        self._Quote, self._Trade = Quote, Trade
        if not self.symbol:
            # Must happen on this loop: the session's connection pool binds to
            # the loop of the first request (see tasty.py module docstring).
            contract, note = await resolve_trading_contract(
                session, self.cfg.product_code, self.cfg.roll_cutoff_days
            )
            self.symbol = contract.streamer_symbol
            self.trading_symbol = contract.symbol
            self.contract_desc = self._describe_contract(contract, days_until_stop(contract), note)
        if self.recorder is None and (self.record_dir or self.record_path):
            path = self.record_path or default_recording_path(self.symbol, self.record_dir or "data/market")
            self.recorder = MarketRecorder(path, symbol=self.symbol,
                                           meta={"trading_symbol": self.trading_symbol})
        async with DXLinkStreamer(session) as streamer:
            self._streamer = streamer
            await streamer.subscribe(Quote, [self.symbol])
            await streamer.subscribe(Trade, [self.symbol])
            tasks = {
                asyncio.create_task(self._listen_quotes(streamer, Quote)),
                asyncio.create_task(self._listen_trades(streamer, Trade)),
            }
            if self._watch_rolls:
                tasks.add(asyncio.create_task(self._watch_roll(session)))
            stop_task = asyncio.create_task(self._stop.wait())
            done, pending = await asyncio.wait(
                tasks | {stop_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done:
                if task is not stop_task and task.exception():
                    raise task.exception()

    @staticmethod
    def _describe_contract(contract, days_left: float, note: str) -> str:
        desc = (f"{contract.symbol} ({contract.streamer_symbol}) "
                f"stops trading {contract.expiration_date} ({days_left:.1f}d)")
        return f"{desc} | {note}" if note else desc

    # -- contract rolls --------------------------------------------------------

    async def _watch_roll(self, session) -> None:
        import inspect

        from .tasty import days_until_stop, resolve_trading_contract

        while True:
            await asyncio.sleep(self.cfg.roll_check_interval_s)
            with self._lock:
                if self._pending_contract is not None:
                    continue  # waiting for the driver to complete the previous roll
            try:
                refresh = session.refresh()
                if inspect.iscoroutine(refresh):
                    await refresh
                contract, note = await resolve_trading_contract(
                    session, self.cfg.product_code, self.cfg.roll_cutoff_days
                )
            except Exception:
                continue  # transient API failure; retry next interval
            if contract.streamer_symbol != self.symbol:
                with self._lock:
                    self._pending_contract = (contract, note)
            else:
                self.contract_desc = self._describe_contract(contract, days_until_stop(contract), note)

    def pending_roll(self) -> PendingRoll | None:
        with self._lock:
            pending = self._pending_contract
        if pending is None:
            return None
        contract, note = pending
        return PendingRoll(
            old_streamer_symbol=self.symbol,
            new_streamer_symbol=contract.streamer_symbol,
            new_trading_symbol=contract.symbol,
            note=note,
        )

    def complete_roll(self, timeout_s: float = 30.0) -> None:
        """Switch subscriptions to the pending contract. Flatten first."""
        with self._lock:
            if self._pending_contract is None:
                return
        if self._error or not self._loop or self._loop.is_closed():
            raise RuntimeError(f"cannot roll: stream is not running ({self._error})")
        future = asyncio.run_coroutine_threadsafe(self._apply_roll(), self._loop)
        future.result(timeout=timeout_s)

    async def _apply_roll(self) -> None:
        from .tasty import days_until_stop

        with self._lock:
            pending = self._pending_contract
        if pending is None:
            return
        contract, note = pending
        old = self.symbol
        await self._streamer.unsubscribe(self._Quote, [old])
        await self._streamer.unsubscribe(self._Trade, [old])
        with self._lock:
            self._bid = self._ask = self._last = math.nan
            self._bid_size = self._ask_size = 0.0
        self.symbol = contract.streamer_symbol
        self.trading_symbol = contract.symbol
        self.contract_desc = self._describe_contract(contract, days_until_stop(contract), note)
        await self._streamer.subscribe(self._Quote, [self.symbol])
        await self._streamer.subscribe(self._Trade, [self.symbol])
        if self.recorder:
            new_path = default_recording_path(self.symbol, self.recorder.path.parent)
            self.recorder.rotate(self.symbol, new_path, meta={"trading_symbol": self.trading_symbol,
                                                              "rolled_from": old})
        with self._lock:
            self._pending_contract = None

    # -- event listeners -------------------------------------------------------

    async def _listen_quotes(self, streamer, Quote) -> None:
        async for q in streamer.listen(Quote):
            now = time.time()
            if q.event_symbol != self.symbol:
                continue  # stale event from a contract we just rolled away from
            bid, ask = _f(q.bid_price), _f(q.ask_price)
            if math.isnan(bid) or math.isnan(ask):
                continue
            with self._lock:
                self._bid, self._ask = bid, ask
                self._bid_size, self._ask_size = _f(q.bid_size), _f(q.ask_size)
                self._quote_count += 1
            self._connected.set()
            if self.recorder:
                self.recorder.quote(now, bid, ask, _f(q.bid_size), _f(q.ask_size))

    async def _listen_trades(self, streamer, Trade) -> None:
        async for tr in streamer.listen(Trade):
            now = time.time()
            if tr.event_symbol != self.symbol:
                continue  # stale event from a contract we just rolled away from
            price = _f(tr.price)
            if math.isnan(price):
                continue
            with self._lock:
                self._last = price
                self._trade_count += 1
            self._connected.set()
            if self.recorder:
                self.recorder.trade(now, price, _f(tr.size))

    # -- reads ---------------------------------------------------------------

    @property
    def counts(self) -> tuple[int, int]:
        with self._lock:
            return self._quote_count, self._trade_count

    def snapshot(self, t: float) -> MarketSnapshot | None:
        if self._error:
            raise RuntimeError(f"DXLink stream failed: {self._error}") from self._error
        with self._lock:
            bid, ask, last = self._bid, self._ask, self._last
            bs, as_ = self._bid_size, self._ask_size
        if math.isnan(bid) or math.isnan(ask):
            return None
        if math.isnan(last):
            last = (bid + ask) / 2.0
        return MarketSnapshot(t=time.time(), bid=bid, ask=ask, last=last, bid_size=bs, ask_size=as_)
