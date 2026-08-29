"""Backtest a probability stream through the real game engine and paper broker.

Uses the same Action semantics the neurons face: BUY from short closes the
short (one step to flatten, another to reverse), positions are one contract,
fills cross the spread and pay commission. The luck baseline circularly
time-shifts the signal stream: same trade structure and frequency, but no
alignment with prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..broker.paper import PaperBroker
from ..config import BrokerCfg, InstrumentCfg
from ..game.engine import GameEngine
from ..game.state import Action
from ..market.source import MarketSnapshot


@dataclass
class BacktestResult:
    dollars: float
    points: float
    n_trades: int
    win_rate: float
    n_signals: int
    equity: np.ndarray = field(repr=False, default=None)


def _signals_from_proba(proba: np.ndarray, band: float) -> np.ndarray:
    """+1 above 0.5+band, -1 below 0.5-band, 0 inside the hold band."""
    sig = np.zeros(len(proba), dtype=np.int64)
    sig[proba > 0.5 + band] = 1
    sig[proba < 0.5 - band] = -1
    return sig


def run_policy(
    signals: np.ndarray,
    t: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    segment_id: np.ndarray,
    instrument: InstrumentCfg,
    broker_cfg: BrokerCfg,
) -> BacktestResult:
    engine = GameEngine(instrument, PaperBroker(broker_cfg, instrument))
    equity = np.zeros(len(signals))
    prev_seg = segment_id[0] if len(segment_id) else 0
    for i in range(len(signals)):
        snap = MarketSnapshot(t=float(t[i]), bid=float(bid[i]), ask=float(ask[i]),
                              last=(float(bid[i]) + float(ask[i])) / 2.0)
        if segment_id[i] != prev_seg:
            engine.flatten(snap)  # never hold across a data gap
            engine.on_roll()
            prev_seg = segment_id[i]
        target = signals[i]
        if target > 0 and engine.position <= 0:
            action = Action.BUY
        elif target < 0 and engine.position >= 0:
            action = Action.SELL
        else:
            action = Action.HOLD
        engine.step(action, snap)
        # realized_dollars is already net of commissions (Trade.dollars subtracts costs)
        equity[i] = engine.realized_dollars + engine.unrealized_points(snap.mid) * instrument.point_value
    if len(signals):
        engine.flatten(MarketSnapshot(t=float(t[-1]), bid=float(bid[-1]), ask=float(ask[-1]),
                                      last=(float(bid[-1]) + float(ask[-1])) / 2.0))
    trades = engine.trades
    wins = sum(1 for tr in trades if tr.dollars > 0)
    return BacktestResult(
        dollars=round(engine.realized_dollars, 2),
        points=round(engine.realized_points, 2),
        n_trades=len(trades),
        win_rate=round(wins / len(trades), 3) if trades else 0.0,
        n_signals=int(np.count_nonzero(signals)),
        equity=equity,
    )


def backtest_proba(
    proba: np.ndarray,
    band: float,
    t: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    segment_id: np.ndarray,
    instrument: InstrumentCfg,
    broker_cfg: BrokerCfg,
    n_random: int = 20,
    seed: int = 42,
) -> tuple[BacktestResult, dict]:
    signals = _signals_from_proba(proba, band)
    result = run_policy(signals, t, bid, ask, segment_id, instrument, broker_cfg)

    if n_random <= 0:
        return result, {}

    # Luck baseline: circular shifts keep the signal's autocorrelation (hence
    # trade count/costs) but break any alignment with future prices.
    rng = np.random.default_rng(seed)
    n = len(signals)
    min_shift = max(60, n // 20)
    random_dollars = []
    for _ in range(n_random):
        shift = int(rng.integers(min_shift, max(n - min_shift, min_shift + 1)))
        rnd = run_policy(np.roll(signals, shift), t, bid, ask, segment_id, instrument, broker_cfg)
        random_dollars.append(rnd.dollars)
    random_stats = {
        "mean_dollars": round(float(np.mean(random_dollars)), 2),
        "std_dollars": round(float(np.std(random_dollars)), 2),
        "n_runs": n_random,
    }
    return result, random_stats
