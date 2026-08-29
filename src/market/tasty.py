"""tastytrade session construction and futures contract selection.

tastytrade v13+ is async-first and a Session's httpx connection pool binds to
the event loop that makes the first request. All async calls that share one
Session must therefore run on the same loop -- LiveSource resolves symbols
inside its streaming loop via resolve_trading_contract for exactly this
reason. The sync resolve_front_month wrapper is for standalone scripts that
resolve a contract and exit without reusing the session on another loop.

Roll policy: we trade the contract tastytrade flags as active month, but never
one within `cutoff_days` of its last trading day. Whichever comes first --
tastytrade flipping the active-month flag or our cutoff -- moves us to the
next contract.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, time as dtime, timezone

from tastytrade import Session
from tastytrade.instruments import Future

from ..config import tastytrade_credentials


def build_session() -> Session:
    creds = tastytrade_credentials()
    if not creds["client_secret"] or not creds["refresh_token"]:
        raise RuntimeError(
            "tastytrade credentials missing. Copy .env.example to .env and set "
            "TASTYTRADE_CLIENT_SECRET and TASTYTRADE_REFRESH_TOKEN (see .env.example for setup steps)."
        )
    return Session(creds["client_secret"], creds["refresh_token"])


def days_until_stop(contract: Future, now: datetime | None = None) -> float:
    """Days until the contract stops trading (falls back to expiration date)."""
    now = now or datetime.now(timezone.utc)
    stop = contract.stops_trading_at
    if stop is None:
        stop = datetime.combine(contract.expiration_date, dtime(13, 30), tzinfo=timezone.utc)
    if stop.tzinfo is None:
        stop = stop.replace(tzinfo=timezone.utc)
    return (stop - now).total_seconds() / 86400.0


def select_trading_contract(
    futures: list[Future] | Future,
    product_code: str,
    cutoff_days: float = 0.0,
    now: datetime | None = None,
) -> tuple[Future, str]:
    """Pick the contract to trade and return (contract, note).

    Starts from tastytrade's active-month contract; if that is within
    cutoff_days of its last trading day, advances to the next contract in the
    chain. The note is non-empty only when we deviated from the plain
    active-month answer (worth surfacing in logs).
    """
    if isinstance(futures, Future):
        futures = [futures]
    tradeable = sorted((f for f in futures if f.is_tradeable), key=lambda f: f.expiration_date)
    if not tradeable:
        raise RuntimeError(f"no tradeable {product_code} contracts returned by tastytrade")

    active = [f for f in tradeable if f.active_month]
    pick = active[0] if active else tradeable[0]
    note = "" if active else f"no active-month flag on any {product_code} contract; using earliest expiry"

    if cutoff_days > 0 and days_until_stop(pick, now) <= cutoff_days:
        later = [f for f in tradeable if f.expiration_date > pick.expiration_date]
        if later:
            flagged = [f for f in later if f.next_active_month]
            skipped = pick
            pick = flagged[0] if flagged else later[0]
            note = (f"{skipped.symbol} stops trading in {days_until_stop(skipped, now):.1f}d "
                    f"(<= cutoff {cutoff_days:g}d); rolled to {pick.symbol}")
        else:
            note = (f"{pick.symbol} stops trading in {days_until_stop(pick, now):.1f}d "
                    f"and no later contract is listed yet")
    return pick, note


async def resolve_trading_contract(
    session: Session, product_code: str, cutoff_days: float = 0.0
) -> tuple[Future, str]:
    """Async contract resolution honoring the roll cutoff. Returns (contract, note)."""
    futures = await Future.get(session, product_codes=[product_code])
    return select_trading_contract(futures, product_code, cutoff_days)


def resolve_front_month(session: Session, product_code: str, cutoff_days: float = 0.0) -> Future:
    """Sync wrapper; do not reuse the session on another event loop afterwards."""
    contract, _ = asyncio.run(resolve_trading_contract(session, product_code, cutoff_days))
    return contract
