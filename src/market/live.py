"""Live market source: tastytrade DXLink websocket in a background thread.

The asyncio streaming loop runs in a daemon thread and continuously updates a
lock-protected latest-state snapshot; the CL closed loop reads it without
blocking. An optional MarketRecorder archives every event into the replay
library while streaming.

Contract rolls: when the symbol is auto-resolved from a product code, a
watcher re-resolves it every roll_check_interval_s (honoring roll_cutoff_days,
see tasty.py). When the answer changes, pending_roll() becomes non-None; the
driver must flatten its position and reset price-derived state, then call
complete_roll(), which schedules the switch on the stream thread and returns
immediately (the CL closed loop is hard real-time and must never block on
network I/O). While the switch is in flight, quotes are cleared, so snapshot()
returns None and the driver skips steps until the new contract streams.

Two staleness rules:
- snapshot() returns None once no event has arrived for stale_quote_s (CME
  maintenance windows, dead feeds): the game idles instead of trading -- and
  the paper broker filling at -- a frozen book.
- dxFeed pushes the last known quote/trade immediately on (re)subscribe.
  That snapshot is history, not live data: it updates the in-memory state
  (the standing book is real) but is NOT recorded, so a connection cycle
  against a closed market writes nothing to the replay library.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
import traceback
from datetime import datetime, timezone

from ..config import LiveCfg
from .recorder import MarketRecorder, default_recording_path
from .source import MarketSnapshot, MarketSource, PendingRoll

log = logging.getLogger(__name__)


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
        self._last_event_wall = 0.0    # wall time of the newest quote/trade event
        # Per-subscription snapshot skip flags (touched only on the stream
        # thread's event loop, no lock needed).
        self._skip_snapshot_quote = True
        self._skip_snapshot_trade = True
        self._pending_contract = None  # (Future, note) under _lock
        self._rolling = False          # a roll switch is in flight on the stream thread
        self._roll_started = 0.0       # wall time complete_roll() was called
        self._connected = threading.Event()
        self._error: BaseException | None = None
        self._error_tb: str = ""
        self._stop: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._streamer = None
        self._Quote = self._Trade = None
        self._preroll_duration_s = 0.0         # remembered for post-roll backfill
        self._pending_preroll: list[MarketSnapshot] = []  # under _lock

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
            if stop_task not in done:
                # A listener ended without an exception: the server closed the
                # stream. Without this, snapshot() would serve the last quote
                # forever and the game would keep trading a frozen market.
                raise RuntimeError("market data stream ended unexpectedly")

    @staticmethod
    def _describe_contract(contract, days_left: float, note: str) -> str:
        desc = (f"{contract.symbol} ({contract.streamer_symbol}) "
                f"stops trading {contract.expiration_date} ({days_left:.1f}d)")
        return f"{desc} | {note}" if note else desc

    # -- historical backfill ---------------------------------------------------

    CANDLE_INTERVAL = "1m"          # preroll resolution; the strip's shortest window is 30 s,
                                    # but preroll only needs to anchor the minutes-to-hours end
    CANDLE_SILENCE_S = 3.0          # no candle for this long = history dump complete
    CANDLE_FETCH_TIMEOUT_S = 45.0

    def preroll(self, duration_s: float, step_s: float = 1.0) -> list[MarketSnapshot]:
        """Backfill history via DXLink candles so the strip starts warm.

        Blocks the caller (runner startup, before the closed loop) while the
        stream thread fetches. Failure is soft: the session starts with a
        cold long end rather than not at all.
        """
        self._preroll_duration_s = duration_s
        if duration_s <= 0 or not self._loop or self._loop.is_closed():
            return []
        fut = asyncio.run_coroutine_threadsafe(
            self._fetch_history(self.symbol, duration_s), self._loop)
        try:
            return fut.result(timeout=self.CANDLE_FETCH_TIMEOUT_S)
        except Exception as exc:
            log.warning("candle backfill failed (%s); the strip's long windows "
                        "stay dark until live history accumulates", exc)
            return []

    def take_preroll(self) -> list[MarketSnapshot]:
        with self._lock:
            snaps, self._pending_preroll = self._pending_preroll, []
        return snaps

    async def _fetch_history(self, symbol: str, duration_s: float) -> list[MarketSnapshot]:
        """Subscribe to 1m candles with a from-time, drain the history dump.

        dxFeed replays the requested range as a burst of Candle events, then
        keeps streaming the live candle; a few seconds of silence marks the
        end of the burst. Candle closes become bid=ask=mid snapshots stamped
        with the candle's epoch time -- the momentum strip only needs mids.
        """
        from tastytrade.dxfeed import Candle

        cutoff = time.time() - duration_s
        start = datetime.fromtimestamp(cutoff - 120.0, tz=timezone.utc)
        await self._streamer.subscribe_candle([symbol], self.CANDLE_INTERVAL, start_time=start)
        closes: dict[float, float] = {}
        gen = self._streamer.listen(Candle)
        deadline = self._loop.time() + self.CANDLE_FETCH_TIMEOUT_S - 5.0
        try:
            while True:
                remaining = deadline - self._loop.time()
                if remaining <= 0:
                    break
                try:
                    c = await asyncio.wait_for(
                        gen.__anext__(), timeout=min(self.CANDLE_SILENCE_S, remaining))
                except asyncio.TimeoutError:
                    break
                if c.event_symbol.split("{")[0] != symbol:
                    continue  # candle from a contract we rolled away from
                close = _f(c.close)
                if not math.isnan(close):
                    closes[c.time / 1000.0] = close  # dxFeed times are epoch ms
        finally:
            try:
                await gen.aclose()
            except Exception:
                pass
            try:
                await self._streamer.unsubscribe_candle(symbol, self.CANDLE_INTERVAL)
            except Exception:
                pass  # backfill is best-effort; never poison the stream for it
        return [MarketSnapshot(t=t, bid=px, ask=px, last=px)
                for t, px in sorted(closes.items()) if t >= cutoff]

    async def _backfill_after_roll(self) -> None:
        """Refill the strip's history with the new contract's candles.

        Runs as its own task after the roll switch completes, so the roll
        itself stays fast; the runner drains take_preroll() at the next
        boundary and seeds the (already reset) feature tracker.
        """
        try:
            snaps = await self._fetch_history(self.symbol, self._preroll_duration_s)
        except Exception as exc:
            log.warning("post-roll candle backfill failed (%s); the strip "
                        "rebuilds from live data", exc)
            return
        with self._lock:
            self._pending_preroll = snaps

    # -- contract rolls --------------------------------------------------------

    async def _watch_roll(self, session) -> None:
        import inspect

        from .tasty import days_until_stop, resolve_trading_contract

        failures = 0
        while True:
            await asyncio.sleep(self.cfg.roll_check_interval_s)
            with self._lock:
                if self._pending_contract is not None or self._rolling:
                    continue  # waiting for the driver to complete the previous roll
            try:
                refresh = session.refresh()
                if inspect.iscoroutine(refresh):
                    await refresh
                contract, note = await resolve_trading_contract(
                    session, self.cfg.product_code, self.cfg.roll_cutoff_days
                )
            except Exception as exc:
                # Transient API failures are expected; persistent ones mean
                # rolls have silently stopped and the contract will age out.
                failures += 1
                log.warning("roll check failed (%d consecutive): %s", failures, exc)
                continue
            failures = 0
            if contract.streamer_symbol != self.symbol:
                with self._lock:
                    self._pending_contract = (contract, note)
            else:
                self.contract_desc = self._describe_contract(contract, days_until_stop(contract), note)

    def pending_roll(self) -> PendingRoll | None:
        with self._lock:
            if self._rolling:
                return None  # switch already in flight
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

    def complete_roll(self) -> None:
        """Schedule the subscription switch on the stream thread. Flatten first.

        Returns immediately: the caller may be inside the CL closed loop's
        50 ms tick budget, so the network round trips must not be awaited
        here. Failures surface through snapshot() via self._error.
        """
        with self._lock:
            if self._pending_contract is None or self._rolling:
                return
            self._rolling = True
            self._roll_started = time.time()
        if self._error or not self._loop or self._loop.is_closed():
            with self._lock:
                self._rolling = False
            raise RuntimeError(f"cannot roll: stream is not running ({self._error})")
        asyncio.run_coroutine_threadsafe(self._apply_roll(), self._loop)

    async def _apply_roll(self) -> None:
        from .tasty import days_until_stop

        try:
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
            # Rotate the recorder *before* subscribing so the new contract's
            # first fresh events land in the new file (one file never spans
            # two contracts). The subscribe snapshot itself is skipped (it is
            # history, not live data); stale old-contract events are dropped
            # by the symbol filter meanwhile.
            if self.recorder:
                new_path = default_recording_path(self.symbol, self.recorder.path.parent)
                self.recorder.rotate(self.symbol, new_path, meta={"trading_symbol": self.trading_symbol,
                                                                  "rolled_from": old})
            self._skip_snapshot_quote = self._skip_snapshot_trade = True
            await self._streamer.subscribe(self._Quote, [self.symbol])
            await self._streamer.subscribe(self._Trade, [self.symbol])
            if self._preroll_duration_s > 0:
                # Detached on purpose: the strip refill takes seconds of
                # network time and must not extend the roll's data outage.
                asyncio.get_running_loop().create_task(self._backfill_after_roll())
        except BaseException as exc:
            self._error = exc
            self._error_tb = traceback.format_exc()
        finally:
            with self._lock:
                self._pending_contract = None
                self._rolling = False

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
                self._last_event_wall = now
            self._connected.set()
            if self._skip_snapshot_quote:
                # First quote after (re)subscribe is dxFeed's snapshot of the
                # last known state -- valid book, wrong time; don't record it.
                self._skip_snapshot_quote = False
            elif self.recorder:
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
                self._last_event_wall = now
            self._connected.set()
            if self._skip_snapshot_trade:
                self._skip_snapshot_trade = False  # subscribe snapshot, see quotes
            elif self.recorder:
                self.recorder.trade(now, price, _f(tr.size))

    # -- reads ---------------------------------------------------------------

    @property
    def counts(self) -> tuple[int, int]:
        with self._lock:
            return self._quote_count, self._trade_count

    ROLL_TIMEOUT_S = 60.0  # watchdog for the async subscription switch

    def snapshot(self, t: float) -> MarketSnapshot | None:
        if self._error:
            raise RuntimeError(f"DXLink stream failed: {self._error}") from self._error
        with self._lock:
            rolling, roll_started = self._rolling, self._roll_started
            bid, ask, last = self._bid, self._ask, self._last
            bs, as_ = self._bid_size, self._ask_size
            last_event = self._last_event_wall
        stale_s = self.cfg.stale_quote_s
        if stale_s > 0 and last_event > 0 and time.time() - last_event > stale_s:
            # No event for stale_quote_s: maintenance window or dead feed.
            # Serving the cached book would let the game fill paper trades at
            # prices nobody can actually trade; idle instead until data flows.
            return None
        if rolling:
            # The switch is in flight on the stream thread; until it clears the
            # cache, bid/ask still belong to the OLD contract the caller just
            # flattened. Serve nothing rather than let the game trade it. If
            # the switch hangs, fail loudly instead of starving forever.
            if time.time() - roll_started > self.ROLL_TIMEOUT_S:
                raise RuntimeError(
                    f"contract roll did not complete within {self.ROLL_TIMEOUT_S:.0f}s "
                    "(subscription switch hung on the stream thread)")
            return None
        if math.isnan(bid) or math.isnan(ask):
            return None
        if math.isnan(last):
            last = (bid + ask) / 2.0
        return MarketSnapshot(t=time.time(), bid=bid, ask=ask, last=last, bid_size=bs, ask_size=as_)
