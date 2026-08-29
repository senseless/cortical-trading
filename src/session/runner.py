"""Session runner: composes market, game, neural, and broker layers into the
closed loop and drives it through an episodic session structure.

Timeline (per the Embodied Neurocomputation findings -- episodic beats
continuous):

    [rest/baseline] [episode 1] [rest] [episode 2] ... [episode N] [done]

During rests, spontaneous activity is collected into the baseline used to
normalize motor-region spike counts. During episodes, every step
(~step_interval_s): decode spikes -> action -> broker -> reward feedback ->
encode new market state. Punishing feedback can pause play (like DishBrain's
post-miss rest) via FeedbackEvent.pause_s.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from ..broker import build_broker
from ..config import Config
from ..game import Action, FeatureTracker, GameEngine
from ..market import build_source
from ..market.live import LiveSource
from ..neural import (AgentDecoder, BaselineTracker, ElectrodeLayout, Encoder,
                      ReservoirDecoder, RewardGenerator)
from ..neural.backend import backend_name, get_cl
from ..neural.stim import StimScheduler, issue
from .console import ConsoleView
from .metrics import session_metrics


class SessionRunner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.layout = ElectrodeLayout(cfg.neural)
        self.source = build_source(cfg)
        if isinstance(self.source, LiveSource):
            self.source.record_dir = "data/market"  # recorder created once the symbol resolves
        self.broker = build_broker(cfg)
        self.engine = GameEngine(cfg.instrument, self.broker)
        self.features = FeatureTracker(cfg.neural.encoding)
        self.encoder = Encoder(cfg.neural.encoding, self.layout, cfg.session.step_interval_s)
        if cfg.session.mode == "reservoir":
            self.decoder = ReservoirDecoder(cfg.neural.decoding, cfg.neural.channels)
        else:
            self.decoder = AgentDecoder(cfg.neural.decoding, self.layout)
        self.baseline = BaselineTracker(list(self.layout.motor), cfg.session.baseline_interactions)
        self.reward = RewardGenerator(cfg.neural.reward, self.layout,
                                      seed=cfg.neural.sdk_seed if cfg.neural.sdk_seed >= 0 else None)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.out_dir = Path(cfg.session.output_dir) / f"{stamp}_{cfg.session.name}"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.steps_log: list[dict] = []
        self.n_predictable = 0
        self.n_unpredictable = 0

    # -----------------------------------------------------------------------

    def run(self) -> dict:
        cfg = self.cfg
        loop_hz = cfg.session.loop_hz
        ticks_per_step = max(1, round(loop_hz * cfg.session.step_interval_s))
        rest_ticks = max(ticks_per_step, round(loop_hz * cfg.session.rest_s))

        # Phase plan: rest before every episode.
        phases: list[tuple[str, int, int]] = []  # (kind, episode_index, duration_ticks)
        for ep in range(cfg.session.episodes):
            phases.append(("rest", ep, rest_ticks))
            phases.append(("episode", ep, cfg.session.steps_per_episode * ticks_per_step))

        cl_mod = get_cl(cfg.neural, accelerated=cfg.session.accelerated)
        (self.out_dir / "config_used.json").write_text(
            json.dumps(dataclasses.asdict(cfg), indent=2, default=str), encoding="utf-8")

        log_fh = open(self.out_dir / "steps.jsonl", "w", encoding="utf-8")
        wall_start = time.time()

        with self.source, ConsoleView(cfg.session.console) as view:
            view.log(f"market: {self.source.describe()}")
            contract_desc = getattr(self.source, "contract_desc", "")
            if contract_desc:
                view.log(f"contract: {contract_desc}")
            with cl_mod.open() as neurons:
                view.log(f"neurons: {backend_name(cl_mod)} | session -> {self.out_dir}")
                recording = None
                if cfg.session.record_cl:
                    recording = neurons.record(
                        file_location=str(self.out_dir), file_suffix=cfg.session.name,
                        include_raw_samples=False,
                        attributes={"label": cfg.session.label, "mode": cfg.session.mode},
                    )
                stream = neurons.create_data_stream(
                    name="trading_game", attributes={"instrument": cfg.instrument.symbol})
                self._last_stream_ts = 0

                scheduler = StimScheduler(loop_hz)
                counts = np.zeros(cfg.neural.channels, dtype=np.int64)
                phase_idx = 0
                tick_in_phase = 0
                step_in_episode = 0
                pause_ticks = 0
                dropped_stims = 0

                try:
                    for tick in neurons.loop(ticks_per_second=loop_hz):
                        i = tick.iteration
                        for spike in tick.analysis.spikes:
                            if 0 <= spike.channel < cfg.neural.channels:
                                counts[spike.channel] += 1
                        for cmd in scheduler.due(i):
                            if not issue(neurons, cl_mod, cmd):
                                dropped_stims += 1

                        if phase_idx >= len(phases):
                            break
                        kind, episode, duration = phases[phase_idx]
                        tick_in_phase += 1
                        boundary = tick_in_phase % ticks_per_step == 0

                        if kind == "rest" and boundary:
                            if self.source.pending_roll():
                                self._handle_roll(neurons, stream, scheduler, i, episode,
                                                  i / loop_hz, view, log_fh)
                            region_counts = {r: float(counts[ch].sum()) for r, ch in self.layout.motor.items()}
                            self.baseline.add_window(region_counts)
                            counts[:] = 0
                            view.update(self._view_state(phase="rest", episode=episode, step=self.baseline.n_windows))

                        elif kind == "episode" and boundary:
                            if pause_ticks > 0:
                                pause_ticks = max(0, pause_ticks - ticks_per_step)
                                if pause_ticks == 0:
                                    counts[:] = 0  # discard noise-period spikes
                            elif self.source.pending_roll():
                                # Roll consumes this decision window: flatten, re-anchor, resubscribe.
                                self._handle_roll(neurons, stream, scheduler, i, episode,
                                                  i / loop_hz, view, log_fh)
                                counts[:] = 0
                            else:
                                t_game = i / loop_hz
                                pause_s = self._do_step(neurons, stream, scheduler, i, counts, episode,
                                                        step_in_episode, t_game, view, log_fh)
                                counts[:] = 0
                                step_in_episode += 1
                                if pause_s > 0:
                                    pause_ticks = round(pause_s * loop_hz)

                        if tick_in_phase >= duration or (kind == "episode" and step_in_episode >= cfg.session.steps_per_episode):
                            if kind == "episode":
                                self._end_episode(neurons, stream, scheduler, i, episode, i / loop_hz, view, log_fh)
                                step_in_episode = 0
                            phase_idx += 1
                            tick_in_phase = 0
                            pause_ticks = 0
                            if phase_idx < len(phases) and phases[phase_idx][0] == "episode":
                                # Prime the first decision window with the current market state.
                                self._prime_episode(scheduler, i, counts)
                except KeyboardInterrupt:
                    view.log("interrupted -- finalizing session")

                if recording is not None:
                    recording.stop()

        log_fh.close()
        metrics = session_metrics(self.steps_log, cfg.session.score_metric)
        metrics.update({
            "label": cfg.session.label,
            "mode": cfg.session.mode,
            "market": self.source.describe(),
            "wall_seconds": round(time.time() - wall_start, 1),
            "dropped_stims": dropped_stims,
            "n_predictable": self.n_predictable,
            "n_unpredictable": self.n_unpredictable,
        })
        (self.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    # -----------------------------------------------------------------------

    def _decide(self, window_counts: np.ndarray) -> tuple[Action, dict]:
        action, debug = self.decoder.decide(window_counts, self.baseline)
        return action, debug

    def _prime_episode(self, scheduler, now_tick: int, counts: np.ndarray) -> None:
        """Deliver the sensory stimulus for the current state at episode start."""
        snap = self.source.snapshot(now_tick / self.cfg.session.loop_hz)
        if snap is None:
            return
        feats = self.features.update(snap)
        sensory = self.encoder.encode_step(feats, self.engine.position,
                                           self.engine.unrealized_points(snap.mid))
        scheduler.schedule(now_tick, sensory)
        counts[:] = 0

    def _do_step(self, neurons, stream, scheduler, now_tick, counts, episode, step,
                 t_game, view, log_fh) -> float:
        """One agent-environment interaction. Returns pause seconds (0 if none)."""
        snap = self.source.snapshot(t_game)
        if snap is None:
            return 0.0  # live source warming up

        feats = self.features.update(snap)
        window_counts = counts.copy()
        action, debug = self._decide(window_counts)
        result = self.engine.step(action, snap)

        feedback = None
        if self.cfg.session.mode == "agent":
            feedback = self.reward.evaluate(result)
            if feedback:
                scheduler.schedule(now_tick, feedback.commands)
                if feedback.kind == "predictable":
                    self.n_predictable += 1
                else:
                    self.n_unpredictable += 1

        sensory = self.encoder.encode_step(feats, self.engine.position, result.unrealized_points)
        scheduler.schedule(now_tick, sensory)

        self._log_row(log_fh, stream, neurons, dict(
            t=round(t_game, 3), wall_t=time.time(), episode=episode, step=step,
            mid=snap.mid, bid=snap.bid, ask=snap.ask,
            momentum_norm=round(feats["momentum_norm"], 4), vol_points=round(feats["vol_points"], 4),
            action=action.value, executed=result.executed, position=result.position,
            mtm_points=round(result.mtm_points, 4),
            unrealized_points=round(result.unrealized_points, 4),
            realized_points=round(result.realized_points, 4),
            realized_dollars=round(result.realized_dollars, 2),
            closed_trade=dataclasses.asdict(result.closed_trade) if result.closed_trade else None,
            reward=feedback.reward if feedback else 0.0,
            feedback_kind=feedback.kind if feedback else None,
            feedback_full=feedback.full if feedback else None,
            decoder=_jsonable(debug),
            counts=window_counts.tolist(),
        ))

        view.update(self._view_state(
            phase="episode", episode=episode, step=step, mid=snap.mid,
            momentum_norm=feats["momentum_norm"], action=action.value,
            diff=debug.get("diff", 0.0) if isinstance(debug, dict) else 0.0,
            buy_count=debug.get("raw", {}).get("buy", 0) if isinstance(debug, dict) else 0,
            sell_count=debug.get("raw", {}).get("sell", 0) if isinstance(debug, dict) else 0,
            position=result.position, unrealized_points=result.unrealized_points,
            realized_dollars=result.realized_dollars, n_trades=len(self.engine.trades),
        ))
        return feedback.pause_s if feedback else 0.0

    def _flatten_with_feedback(self, neurons, stream, scheduler, now_tick, episode,
                               t_game, log_fh, label: str) -> None:
        """Force-close any open position; the outcome is real, so feedback still flows."""
        snap = self.source.snapshot(t_game)
        if snap is None:
            return
        result = self.engine.flatten(snap)
        if result is None:
            return
        feedback = None
        if self.cfg.session.mode == "agent":
            feedback = self.reward.evaluate(result)
            if feedback:
                scheduler.schedule(now_tick, feedback.commands)
                if feedback.kind == "predictable":
                    self.n_predictable += 1
                else:
                    self.n_unpredictable += 1
        self._log_row(log_fh, stream, neurons, dict(
            t=round(t_game, 3), wall_t=time.time(), episode=episode, step=-1,
            mid=snap.mid, bid=snap.bid, ask=snap.ask,
            momentum_norm=0.0, vol_points=0.0,
            action=label, executed=result.executed, position=result.position,
            mtm_points=round(result.mtm_points, 4),
            unrealized_points=0.0,
            realized_points=round(result.realized_points, 4),
            realized_dollars=round(result.realized_dollars, 2),
            closed_trade=dataclasses.asdict(result.closed_trade) if result.closed_trade else None,
            reward=feedback.reward if feedback else 0.0,
            feedback_kind=feedback.kind if feedback else None,
            feedback_full=feedback.full if feedback else None,
            decoder={}, counts=[],
        ))

    def _end_episode(self, neurons, stream, scheduler, now_tick, episode, t_game, view, log_fh) -> None:
        self._flatten_with_feedback(neurons, stream, scheduler, now_tick, episode, t_game, log_fh, "flatten")
        view.log(f"episode {episode} done: realized ${self.engine.realized_dollars:+.2f} "
                 f"({len(self.engine.trades)} trades)")

    def _handle_roll(self, neurons, stream, scheduler, now_tick, episode, t_game, view, log_fh) -> None:
        """Contract roll: flatten on the old contract, re-anchor, then resubscribe.

        Order matters -- the flatten must execute against the old contract's
        quotes, and the feature window and mark-to-market anchor must be reset
        before the first snapshot of the new contract arrives, so the basis
        jump between contracts never reaches the network or the PnL.
        """
        info = self.source.pending_roll()
        if info is None:
            return
        self._flatten_with_feedback(neurons, stream, scheduler, now_tick, episode,
                                    t_game, log_fh, "roll_flatten")
        self.features.reset()
        self.engine.on_roll()
        self.source.complete_roll()
        note = f" ({info.note})" if info.note else ""
        view.log(f"contract roll: {info.old_streamer_symbol} -> {info.new_streamer_symbol}{note}")

    # -----------------------------------------------------------------------

    def _log_row(self, log_fh, stream, neurons, row: dict) -> None:
        self.steps_log.append(row)
        log_fh.write(json.dumps(row) + "\n")
        log_fh.flush()
        try:
            ts = neurons.timestamp()
            if ts <= self._last_stream_ts:
                ts = self._last_stream_ts + 1
            self._last_stream_ts = ts
            stream.append(ts, {k: row[k] for k in ("mid", "action", "position", "realized_dollars", "reward")})
        except Exception:
            pass  # data stream is best-effort; the JSONL log is authoritative

    def _view_state(self, **kw) -> dict:
        state = dict(
            title=f"{self.cfg.session.name} [{self.cfg.session.label}] {self.cfg.instrument.symbol}",
            n_predictable=self.n_predictable, n_unpredictable=self.n_unpredictable,
        )
        state.update(kw)
        return state


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj
