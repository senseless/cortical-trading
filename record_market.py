"""Record live market data into the replay library.

Usage:
    python record_market.py                        # front-month /MBT (micro bitcoin, trades 24/7)
    python record_market.py --forever              # run indefinitely: auto-reconnect, 6h file rotation
    python record_market.py --product MNQ          # any other CME product
    python record_market.py --product MBT,MET,MNQ,MES,MCL   # multi-product: one connection, one file per product
    python record_market.py --symbol "BTC/USD:CXTALP"    # explicit streamer symbol (single mode only)
    python record_market.py --duration 3600        # stop after N seconds (default: until Ctrl+C)

Contract rolls are handled automatically: when tastytrade flips the active
month (or the current contract enters the roll cutoff window), recording
rotates into a new file on the new contract. One file never spans two
contracts. In multi-product mode each product rolls independently.

--forever mode is the library builder: it restarts on any error with capped
backoff (surviving disconnects, token issues, and CME maintenance windows),
detects stalled streams, and starts a fresh timestamped file every rotation
interval so no single file grows unbounded. Note: equity/commodity products
only tick Sun 6pm - Fri 5pm ET; over the weekend only crypto (MBT, MET)
produces events, which is what keeps the stall detector fed.
"""

from __future__ import annotations

import argparse
import time

from src.config import LiveCfg
from src.market.live import LiveSource
from src.market.multi_recorder import MultiRecorder

STALL_TIMEOUT_S = 600.0  # no events for this long = stream is dead; reconnect


def record_once(args, duration_s: float, stall_timeout_s: float = 0.0) -> None:
    """One single-product recording session. Raises on stream failure or stall."""
    source = LiveSource(LiveCfg(product_code=args.product, streamer_symbol=args.symbol))
    source.record_path = args.out
    source.record_dir = "data/market"

    print(f"[{time.strftime('%H:%M:%S')}] connecting ({args.symbol or 'front month ' + args.product})...",
          flush=True)
    source.start()
    print(f"contract: {source.contract_desc}", flush=True)
    print(f"recording {source.symbol} -> {source.recorder.path}", flush=True)

    started = time.time()
    last_events = 0
    last_progress = started
    try:
        while True:
            time.sleep(5)
            roll = source.pending_roll()
            if roll:
                source.complete_roll()  # nothing to flatten while recording
                note = f" ({roll.note})" if roll.note else ""
                print(f"contract roll: {roll.old_streamer_symbol} -> {roll.new_streamer_symbol}{note}",
                      flush=True)
                print(f"recording continues -> {source.recorder.path}", flush=True)
            q, tr = source.counts
            snap = source.snapshot(0)
            px = f"bid {snap.bid} / ask {snap.ask} last {snap.last}" if snap else "no quote yet"
            print(f"[{time.strftime('%H:%M:%S')}] quotes={q} trades={tr} | {px}", flush=True)
            source.recorder.flush()
            now = time.time()
            if q + tr > last_events:
                last_events = q + tr
                last_progress = now
            elif stall_timeout_s and now - last_progress > stall_timeout_s:
                raise RuntimeError(f"no events for {stall_timeout_s:.0f}s; stream presumed dead")
            if duration_s and now - started >= duration_s:
                break
    finally:
        source.stop()
        q, tr = source.counts
        print(f"saved {q} quotes, {tr} trades (last file: {source.recorder.path})", flush=True)


def record_multi_once(products: list[str], duration_s: float, stall_timeout_s: float = 0.0) -> None:
    """One multi-product recording session over a single DXLink connection."""
    rec = MultiRecorder(products)
    print(f"[{time.strftime('%H:%M:%S')}] connecting ({', '.join(products)})...", flush=True)
    rec.start(timeout_s=90)
    for line in rec.describe_lines():
        print(f"  {line}", flush=True)
    for code, reason in rec.skipped:
        print(f"  {code:5s} SKIPPED: {reason}", flush=True)

    started = time.time()
    last_events = 0
    last_progress = started
    try:
        while True:
            time.sleep(30)
            rec.check_error()
            for notice in rec.drain_notices():
                print(f"[{time.strftime('%H:%M:%S')}] {notice}", flush=True)
            print(f"[{time.strftime('%H:%M:%S')}] {rec.status_line()}", flush=True)
            rec.flush_all()
            now = time.time()
            total = rec.total_events
            if total > last_events:
                last_events = total
                last_progress = now
            elif stall_timeout_s and now - last_progress > stall_timeout_s:
                raise RuntimeError(f"no events for {stall_timeout_s:.0f}s; stream presumed dead")
            if duration_s and now - started >= duration_s:
                break
    finally:
        rec.stop()
        print(f"[{time.strftime('%H:%M:%S')}] session closed: {rec.status_line()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record live market data to the replay library")
    parser.add_argument("--product", default="MBT",
                        help="CME product code(s), comma-separated (front months auto-resolved)")
    parser.add_argument("--symbol", default="", help="explicit dxFeed streamer symbol override (single product only)")
    parser.add_argument("--out", default="", help="output path (single product only; default: data/market/<symbol>_<stamp>.jsonl.gz)")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds to record (0 = until Ctrl+C)")
    parser.add_argument("--forever", action="store_true",
                        help="run indefinitely: reconnect on errors, rotate files every --duration seconds")
    args = parser.parse_args()

    products = [p.strip().upper() for p in args.product.split(",") if p.strip()]
    multi = len(products) > 1
    if multi and (args.symbol or args.out):
        print("--symbol/--out are ignored in multi-product mode")
        args.symbol = args.out = ""

    def run_once(duration_s: float, stall_timeout_s: float = 0.0) -> None:
        if multi:
            record_multi_once(products, duration_s, stall_timeout_s)
        else:
            record_once(args, duration_s, stall_timeout_s)

    if not args.forever:
        try:
            run_once(args.duration)
        except KeyboardInterrupt:
            print("stopping...")
        return

    if args.out:
        print("--out is ignored in --forever mode (files are timestamped per rotation)")
        args.out = ""
    rotate_s = args.duration or 21600.0  # default 6h files
    backoff = 30.0
    print(f"library builder: rotating files every {rotate_s / 3600:.1f}h, auto-reconnect on failure")
    while True:
        try:
            run_once(rotate_s, stall_timeout_s=STALL_TIMEOUT_S)
            backoff = 30.0  # clean rotation, reset backoff
        except KeyboardInterrupt:
            print("stopping...")
            return
        except Exception as exc:
            print(f"[{time.strftime('%H:%M:%S')}] recorder error: {exc}; retrying in {backoff:.0f}s",
                  flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 300.0)


if __name__ == "__main__":
    main()
