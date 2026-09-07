"""Live market source: tastytrade DXLink websocket in a background thread.

The asyncio streaming loop runs in a daemon thread and continuously updates a
lock-protected latest-state snapshot; the CL closed loop reads it without
blocking. An optional MarketRecorder archives every event into the replay
library while streaming.

Reconnects: the DXLink connection is not permanent. The quote token tastytrade
issues is valid for 24 h and is cached account-wide, so every stream on the
account dies at the same wall-clock minute each day (see token_expires_at);
websockets also drop for ordinary network reasons. When the stream fails, the
book is cleared (snapshot() returns None, the game idles), the source
reconnects with capped backoff and resubscribes the current symbol, and play
resumes with the next quote. An outage longer than cfg.outage_timeout_s makes
snapshot() raise instead, so a dead feed cannot silently consume a wetware
session.

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
  That snapshot is history, not live data: it updates the in-memory book (the
  standing book is real) but is NOT recorded, so a connection cycle against a
  closed market writes nothing to the replay library, and it is aged by the
  exchange's own timestamp rather than by arrival time, so reconnecting
  across a closed market cannot pass a pre-closure book off as live.
"""

from __future__ import annotations

import asyncio
import base64
import json
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


def _leaves(exc: BaseException):
    """Flatten ExceptionGroups (the SDK's TaskGroup wraps streamer failures)."""
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            yield from _leaves(sub)
    else:
        yield exc


def _leaf_error(exc: BaseException) -> str:
    """Human-readable cause: the first real error inside any ExceptionGroup."""
    leaf = next(_leaves(exc), exc)
    return f"{type(leaf).__name__}: {leaf}"


def _is_stream_failure(exc: BaseException) -> bool:
    """True if this is a dead stream (retryable), not cancellation or Ctrl-C.

    Matters because a group containing CancelledError/KeyboardInterrupt must
    propagate: retrying it would spin the reconnect loop during shutdown.
    """
    return all(isinstance(leaf, Exception) for leaf in _leaves(exc))


def _event_wall(event_ms: float, now: float) -> float:
    """Local-clock time of an event from its exchange timestamp (epoch ms).

    Falls back to `now` when the feed's stamp is missing or implausible, so a
    product that does not populate it behaves exactly as before.
    """
    if math.isnan(event_ms) or event_ms <= 0:
        return now
    t = event_ms / 1000.0
    if t > now + 5.0 or t < now - 30.0 * 86400.0:
        return now  # clock skew, or a stamp that is not epoch milliseconds
    return t


def jwt_expiry(token: str) -> float | None:
    """Epoch expiry of a JWT's `exp` claim (no signature check), or None."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        exp = claims.get("exp")
        return float(exp) if exp is not None else None
    except Exception:
        return None


class LiveSource(MarketSource):
    RECONNECT_BACKOFF_S = 5.0
    RECONNECT_BACKOFF_MAX_S = 60.0

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
        self._error: BaseException | None = None   # fatal: the stream thread is gone
        self._error_tb: str = ""
        self._stop: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._streamer = None
        self._Quote = self._Trade = None
        self._preroll_duration_s = 0.0         # remembered for post-roll backfill
        self._pending_preroll: list[MarketSnapshot] = []  # under _lock
        # Reconnect state (under _lock unless noted).
        self._down_since = 0.0                 # wall time the stream dropped; 0 = up
        self._last_stream_error = ""           # stream thread only; read for diagnostics
        self._notices: list[str] = []          # human-readable events for the driver to log
        self._stale_noticed = False            # game thread only
        self.disconnects = 0                   # stream thread only
        self._connects = 0                     # stream thread only
        self.token_expires_at: float | None = None  # epoch; DXLink quote token expiry, if decodable

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
            hint = f" (last stream error: {self._last_stream_error})" if self._last_stream_error else ""
            raise TimeoutError(
                f"no market data received for {self.symbol or self.cfg.product_code} within {timeout_s}s{hint}"
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

    def drain_notices(self) -> list[str]:
        """Connection events (drops, reconnects) since the last call, oldest first."""
        with self._lock:
            out, self._notices = self._notices, []
        return out

    def _notify(self, message: str) -> None:
        with self._lock:
            self._notices.append(message)

    def _thread_main(self, session) -> None:
        try:
            asyncio.run(self._stream(session))
        except BaseException as exc:  # surface to the main thread
            self._error = exc
            self._error_tb = traceback.format_exc()

    async def _stream(self, session) -> None:
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

        watcher = asyncio.create_task(self._watch_roll(session)) if self._watch_rolls else None
        stop_task = asyncio.create_task(self._stop.wait())
        backoff = self.RECONNECT_BACKOFF_S
        try:
            while True:
                connects_before = self._connects
                try:
                    await self._stream_once(session, stop_task)
                    return  # stop requested
                except (Exception, BaseExceptionGroup) as exc:
                    if self._stop.is_set():
                        return
                    if not _is_stream_failure(exc):
                        raise  # cancellation / interrupt: do not retry
                    if self._connects > connects_before:
                        # This attempt did connect and ran for a while, so the
                        # backoff earned by earlier failures is spent; a daily
                        # token expiry must not inherit a minute of waiting.
                        backoff = self.RECONNECT_BACKOFF_S
                    self._on_disconnect(exc, backoff)
                done, _ = await asyncio.wait({stop_task}, timeout=backoff)
                if done:
                    return
                backoff = min(backoff * 2.0, self.RECONNECT_BACKOFF_MAX_S)
        finally:
            for task in (watcher, stop_task):
                if task is not None:
                    task.cancel()

    async def _stream_once(self, session, stop_task: asyncio.Task) -> None:
        """One websocket lifetime: connect, subscribe, listen until stop or failure."""
        from tastytrade import DXLinkStreamer

        async with DXLinkStreamer(session) as streamer:
            self._streamer = streamer
            # Every connection is handed the account's cached token; note when
            # it runs out so the driver can see the drop coming.
            self.token_expires_at = await self._quote_token_expiry(session) or self.token_expires_at
            self._skip_snapshot_quote = self._skip_snapshot_trade = True
            await streamer.subscribe(self._Quote, [self.symbol])
            await streamer.subscribe(self._Trade, [self.symbol])
            self._on_connected()
            tasks = {
                asyncio.create_task(self._listen_quotes(streamer, self._Quote)),
                asyncio.create_task(self._listen_trades(streamer, self._Trade)),
            }
            try:
                done, _ = await asyncio.wait(tasks | {stop_task}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            if stop_task in done:
                return
            for task in done:
                if task.exception():
                    raise task.exception()
            # A listener ended without an exception: the server closed the
            # stream. Without this, snapshot() would serve the last quote
            # forever and the game would keep trading a frozen market.
            raise RuntimeError("market data stream ended unexpectedly")

    def _on_connected(self) -> None:
        self._connects += 1
        with self._lock:
            was_down = self._down_since
            self._down_since = 0.0
        if was_down:
            self._notify(f"market data stream reconnected after {time.time() - was_down:.0f}s "
                         f"(subscribed {self.symbol})")

    def _on_disconnect(self, exc: BaseException, backoff: float) -> None:
        self.disconnects += 1
        self._last_stream_error = _leaf_error(exc)
        with self._lock:
            first = not self._down_since
            if first:
                self._down_since = time.time()
            # Never serve a dead stream's book: snapshot() returns None until
            # the resubscribe snapshot repopulates it.
            self._bid = self._ask = self._last = math.nan
            self._bid_size = self._ask_size = 0.0
        if first:
            self._notify(f"market data stream dropped: {self._last_stream_error}; reconnecting")
        log.warning("market data stream failed (%s); reconnecting in %.0fs", self._last_stream_error, backoff)

    @staticmethod
    async def _quote_token_expiry(session) -> float | None:
        """Expiry of the account's cached DXLink quote token (a JWT), best effort.

        tastytrade hands every connection the same token until it expires, so
        the stream is guaranteed to drop at this moment; the driver can warn
        when a planned session spans it.
        """
        try:
            data = await session._get("/api-quote-tokens")
            return jwt_expiry(data["token"])
        except Exception as exc:
            log.debug("quote token expiry unavailable: %s", exc)
            return None

    @staticmethod
    def _describe_contract(contract, days_left: float, note: str) -> str:
        desc = (f"{contract.symbol} ({contract.streamer_symbol}) "
                f"stops trading {contract.expiration_date} ({days_left:.1f}d)")
        return f"{desc} | {note}" if note else desc

    # -- historical backfill ---------------------------------------------------

    CANDLE_INTERVAL = "1m"          # preroll resolution; the strip's shortest window is 30 s,
                                    # but preroll only needs to anchor the minutes-to-hours end
    CANDLE_SILENCE_S = 3.0          # fallback: no candle for this long = history dump complete
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

        dxFeed replays the requested range as a snapshot burst whose last
        event carries the SNAPSHOT_END (or SNAPSHOT_SNIP) flag, then keeps
        streaming the live candle. The flag is the primary stop; a few seconds
        of silence and a hard deadline remain as fallbacks. Candle closes
        become bid=ask=mid snapshots stamped with the candle's epoch time --
        the momentum strip only needs mids.
        """
        from tastytrade.dxfeed import Candle

        streamer = self._streamer
        if streamer is None:
            raise RuntimeError("stream not connected")
        cutoff = time.time() - duration_s
        start = datetime.fromtimestamp(cutoff - 120.0, tz=timezone.utc)
        await streamer.subscribe_candle([symbol], self.CANDLE_INTERVAL, start_time=start)
        closes: dict[float, float] = {}
        gen = streamer.listen(Candle)
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
                if c.snapshot_end or c.snapshot_snip:
                    break  # last event of the history dump
        finally:
            try:
                await gen.aclose()
            except Exception:
                pass
            try:
                await streamer.unsubscribe_candle(symbol, self.CANDLE_INTERVAL)
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
        from .tasty import days_until_stop, resolve_trading_contract

        failures = 0
        while True:
            await asyncio.sleep(self.cfg.roll_check_interval_s)
            with self._lock:
                if self._pending_contract is not None or self._rolling:
                    continue  # waiting for the driver to complete the previous roll
            try:
                # The session refreshes its own access token per request.
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
        """Switch symbol, recorder, and subscriptions to the pending contract.

        The network calls are best-effort because the stream may be
        mid-reconnect: stale events for the old symbol are dropped by the
        listeners' symbol filter, and a reconnect subscribes whatever
        self.symbol is at that moment. Anything else failing here (a recorder
        rotation, say) can leave the switch half-applied -- symbol changed but
        still writing the old contract's file -- so it is fatal: the error goes
        to snapshot() and the session finalizes rather than silently mixing two
        contracts into one recording.
        """
        from .tasty import days_until_stop

        try:
            with self._lock:
                pending = self._pending_contract
            if pending is None:
                return
            contract, note = pending
            old = self.symbol
            streamer = self._streamer
            try:
                await streamer.unsubscribe(self._Quote, [old])
                await streamer.unsubscribe(self._Trade, [old])
            except Exception as exc:
                log.warning("unsubscribe %s failed (%s); stale events are filtered by symbol", old, exc)
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
            try:
                await streamer.subscribe(self._Quote, [self.symbol])
                await streamer.subscribe(self._Trade, [self.symbol])
            except Exception as exc:
                log.warning("subscribe %s failed (%s); the reconnect will subscribe it", self.symbol, exc)
            if self._preroll_duration_s > 0:
                # Detached on purpose: the strip refill takes seconds of
                # network time and must not extend the roll's data outage.
                asyncio.get_running_loop().create_task(self._backfill_after_roll())
        except BaseException as exc:
            self._error = exc
            self._error_tb = traceback.format_exc()
            # The switch is half-applied and the recorder may still point at the
            # old contract's file. Stop recording now (MarketRecorder refuses
            # writes after close) so the seconds before the session notices the
            # error cannot append the new contract's quotes to the old file.
            if self.recorder:
                self.recorder.close()
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
            # First quote after (re)subscribe is dxFeed's snapshot of the last
            # known state: a valid book, but as old as the last change. Age it
            # by the exchange's own stamp instead of "now", or a reconnect
            # across a market closure would present a pre-closure book as live
            # and stale_quote_s would never fire.
            snapshot_event = self._skip_snapshot_quote
            event_wall = (_event_wall(max(_f(q.bid_time), _f(q.ask_time)), now)
                          if snapshot_event else now)
            with self._lock:
                self._bid, self._ask = bid, ask
                self._bid_size, self._ask_size = _f(q.bid_size), _f(q.ask_size)
                self._quote_count += 1
                self._last_event_wall = max(self._last_event_wall, event_wall)
            self._connected.set()
            if snapshot_event:
                self._skip_snapshot_quote = False  # history, not a live tick: don't record
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
            snapshot_event = self._skip_snapshot_trade
            event_wall = _event_wall(_f(tr.time), now) if snapshot_event else now
            with self._lock:
                self._last = price
                self._trade_count += 1
                self._last_event_wall = max(self._last_event_wall, event_wall)
            self._connected.set()
            if snapshot_event:
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
            down_since = self._down_since
            bid, ask, last = self._bid, self._ask, self._last
            bs, as_ = self._bid_size, self._ask_size
            last_event = self._last_event_wall
        if down_since:
            # Reconnecting. Idle, unless the outage has gone on long enough
            # that idling is just burning the session.
            outage = time.time() - down_since
            if self.cfg.outage_timeout_s > 0 and outage > self.cfg.outage_timeout_s:
                raise RuntimeError(
                    f"market data stream has been down for {outage:.0f}s "
                    f"(> outage_timeout_s={self.cfg.outage_timeout_s:g}); last error: "
                    f"{self._last_stream_error}")
            return None
        stale_s = self.cfg.stale_quote_s
        if stale_s > 0 and last_event > 0 and time.time() - last_event > stale_s:
            # No event for stale_quote_s: maintenance window or dead feed.
            # Serving the cached book would let the game fill paper trades at
            # prices nobody can actually trade; idle instead until data flows.
            if not self._stale_noticed:
                self._stale_noticed = True
                self._notify(f"market data stale (no events for {time.time() - last_event:.0f}s); "
                             "idling until quotes resume")
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
        if self._stale_noticed:
            self._stale_noticed = False
            self._notify("market data resumed")
        return MarketSnapshot(t=time.time(), bid=bid, ask=ask, last=last, bid_size=bs, ask_size=as_)
