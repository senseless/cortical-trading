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
import math
import sys
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
            self.decoder = ReservoirDecoder(
                cfg.neural.decoding, cfg.neural.channels,
                layout={"sensory": cfg.neural.sensory, "motor": cfg.neural.motor})
        else:
            self.decoder = AgentDecoder(cfg.neural.decoding, self.layout)
        self.baseline = BaselineTracker(list(self.layout.motor), cfg.session.baseline_interactions)
        self.reward = RewardGenerator(cfg.neural.reward, self.layout,
                                      seed=cfg.neural.sdk_seed if cfg.neural.sdk_seed >= 0 else None,
                                      step_interval_s=cfg.session.step_interval_s)

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

        # Phase plan: rest before every episode. The first rest is extended to
        # warmup_s because the momentum strip's long windows are silent until
        # that much market history exists -- rests feed the feature tracker
        # (they just don't stimulate), so the pre-session rest warms the strip
        # while it collects the spontaneous-activity baseline.
        warmup_ticks = max(rest_ticks, round(loop_hz * cfg.session.warmup_s))
        phases: list[tuple[str, int, int]] = []  # (kind, episode_index, duration_ticks)
        for ep in range(cfg.session.episodes):
            phases.append(("rest", ep, warmup_ticks if ep == 0 else rest_ticks))
            phases.append(("episode", ep, cfg.session.steps_per_episode * ticks_per_step))

        cl_mod = get_cl(cfg.neural, accelerated=cfg.session.accelerated)
        (self.out_dir / "config_used.json").write_text(
            json.dumps(dataclasses.asdict(cfg), indent=2, default=str), encoding="utf-8")

        log_fh = open(self.out_dir / "steps.jsonl", "w", encoding="utf-8")
        wall_start = time.time()
        dropped_stims = 0

        try:
            with self.source, ConsoleView(cfg.session.console) as view:
                view.log(f"market: {self.source.describe()}")
                contract_desc = getattr(self.source, "contract_desc", "")
                if contract_desc:
                    view.log(f"contract: {contract_desc}")
                # Pre-roll: seed the feature tracker with history covering the
                # ladder's longest window, so the whole strip is live from the
                # first episode instead of taking hours to warm in-session.
                longest_window = cfg.neural.encoding.momentum_windows_s[-1]
                preroll_snaps = self.source.preroll(longest_window, cfg.session.step_interval_s)
                covered = 0.0
                if preroll_snaps:
                    self.features.seed(preroll_snaps)
                    covered = preroll_snaps[-1].t - preroll_snaps[0].t + cfg.session.step_interval_s
                    view.log(f"preroll: {covered / 3600.0:.2f} h of history seeded the momentum strip")
                if covered + warmup_ticks / loop_hz < longest_window:
                    view.log(f"warning: preroll ({covered:.0f}s) + warm-up ({warmup_ticks / loop_hz:.0f}s) "
                             f"cover less than the longest momentum window ({longest_window:g}s); the "
                             "strip's long end will read silence until enough history accumulates")
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
                    last_stim_end_tick = -(10 ** 9)  # long before the first window

                    try:
                        for tick in neurons.loop(ticks_per_second=loop_hz):
                            i = tick.iteration
                            for spike in tick.analysis.spikes:
                                if 0 <= spike.channel < cfg.neural.channels:
                                    counts[spike.channel] += 1
                            due_cmds = scheduler.due(i)
                            if due_cmds:
                                last_stim_end_tick = max(last_stim_end_tick, i)
                            for cmd in due_cmds:
                                if not issue(neurons, cl_mod, cmd):
                                    dropped_stims += 1
                                elif cmd.count > 1 and cmd.rate_hz > 0:
                                    # A burst keeps playing (count-1)/rate seconds
                                    # after issue; the baseline stim-free guard
                                    # must measure from the END of playback, or
                                    # a sensory burst's tail (up to ~1 s) leaks
                                    # into the first "spontaneous" window.
                                    end = i + math.ceil((cmd.count - 1) / cmd.rate_hz * loop_hz)
                                    last_stim_end_tick = max(last_stim_end_tick, end)

                            if phase_idx >= len(phases):
                                # Drain before ending: the final flatten's
                                # feedback is scheduled with delays up to a few
                                # seconds, and it is already counted and logged
                                # as delivered -- breaking immediately would
                                # silently drop it.
                                if scheduler.pending == 0:
                                    break
                                continue
                            kind, episode, duration = phases[phase_idx]
                            tick_in_phase += 1
                            boundary = tick_in_phase % ticks_per_step == 0

                            if kind == "rest" and boundary:
                                self._drain_backfill(view)
                                if self.source.pending_roll():
                                    self._handle_roll(neurons, stream, scheduler, i, episode,
                                                      i / loop_hz, view, log_fh)
                                # Keep feeding the feature tracker through the
                                # rest: the market moves while the culture
                                # rests, and the momentum strip's long windows
                                # need unbroken history to stay defined. No
                                # stimulation is issued, so the baseline stays
                                # spontaneous.
                                rest_snap = self.source.snapshot(i / loop_hz)
                                if rest_snap is not None:
                                    self.features.update(rest_snap)
                                # Baseline means *spontaneous* activity: skip any
                                # window that overlapped stim playback (the
                                # episode-end feedback tail plays into early rest
                                # by design, and roll flattens can stim mid-rest).
                                if i - last_stim_end_tick >= ticks_per_step:
                                    region_counts = {r: float(counts[ch].sum()) for r, ch in self.layout.motor.items()}
                                    self.baseline.add_window(region_counts)
                                counts[:] = 0
                                view.update(self._view_state(phase="rest", episode=episode, step=self.baseline.n_windows))

                            elif kind == "episode" and boundary:
                                self._drain_backfill(view)
                                if pause_ticks > 0:
                                    pause_ticks = max(0, pause_ticks - ticks_per_step)
                                    if pause_ticks == 0:
                                        counts[:] = 0  # discard feedback-period spikes
                                        # Re-prime: no sensory input was delivered
                                        # during the pause, so without a fresh
                                        # stimulus the next decision would decode
                                        # unstimulated spontaneous activity.
                                        self._prime_episode(scheduler, i, counts)
                                elif self.source.pending_roll():
                                    # Roll consumes this decision window: flatten, re-anchor, resubscribe.
                                    pause_s = self._handle_roll(neurons, stream, scheduler, i, episode,
                                                                i / loop_hz, view, log_fh)
                                    counts[:] = 0
                                    if pause_s > 0:
                                        # The roll flatten's full feedback must not
                                        # bleed into the next decision window.
                                        pause_ticks = round(pause_s * loop_hz)
                                else:
                                    t_game = i / loop_hz
                                    final_step = (tick_in_phase >= duration
                                                  or step_in_episode + 1 >= cfg.session.steps_per_episode)
                                    pause_s = self._do_step(neurons, stream, scheduler, i, counts, episode,
                                                            step_in_episode, t_game, view, log_fh,
                                                            final_step=final_step)
                                    counts[:] = 0
                                    step_in_episode += 1
                                    if pause_s > 0:
                                        pause_ticks = round(pause_s * loop_hz)

                            if tick_in_phase >= duration or (kind == "episode" and step_in_episode >= cfg.session.steps_per_episode):
                                if kind == "episode":
                                    self._end_episode(neurons, stream, scheduler, i, episode, i / loop_hz, view, log_fh)
                                    step_in_episode = 0
                                # Do NOT clear the scheduler here: the final
                                # step's and the flatten's feedback are scheduled
                                # on this very tick and must reach the culture
                                # (they are counted and logged as delivered).
                                # Baseline purity is preserved by the stim-free
                                # window guard in the rest branch above.
                                counts[:] = 0
                                phase_idx += 1
                                tick_in_phase = 0
                                pause_ticks = 0
                                if phase_idx < len(phases) and phases[phase_idx][0] == "episode":
                                    # Prime the first decision window with the current market state.
                                    # The decoder's lag state (reservoir mode) is cleared first:
                                    # its readout was trained on within-episode lag stacks only,
                                    # so windows from before the rest must not feed the first
                                    # decisions of the new episode.
                                    self.decoder.reset()
                                    self._prime_episode(scheduler, i, counts)
                    except KeyboardInterrupt:
                        view.log("interrupted -- finalizing session")
                    finally:
                        # Finalize the HDF5 recording even when the loop dies
                        # (tick-budget TimeoutError, stream failure): a rented
                        # wetware session's data must survive its crash.
                        if recording is not None:
                            try:
                                recording.stop()
                            except Exception:
                                view.log("warning: recording.stop() failed")
        finally:
            failure = sys.exc_info()[1]
            try:
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
                if failure is not None:
                    metrics["error"] = f"{type(failure).__name__}: {failure}"
                (self.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            except Exception:
                if failure is None:
                    raise  # the metrics write is the only failure; surface it
                # otherwise never mask the session error with a metrics error
        return metrics

    # -----------------------------------------------------------------------

    def _decide(self, window_counts: np.ndarray) -> tuple[Action, dict]:
        action, debug = self.decoder.decide(window_counts, self.baseline)
        return action, debug

    def _drain_backfill(self, view) -> None:
        """Seed history that arrived asynchronously (post-roll candle backfill)."""
        snaps = self.source.take_preroll()
        if snaps:
            self.features.seed(snaps)
            view.log(f"momentum strip re-seeded from {len(snaps)} backfilled candles")

    def _prime_episode(self, scheduler, now_tick: int, counts: np.ndarray) -> None:
        """Deliver the sensory stimulus for the current state (episode start / pause end)."""
        snap = self.source.snapshot(now_tick / self.cfg.session.loop_hz)
        if snap is None:
            return
        feats = self.features.update(snap)
        sensory = self.encoder.encode_step(feats, self.engine.position,
                                           self.engine.unrealized_points(snap.mid))
        scheduler.schedule(now_tick, sensory)
        counts[:] = 0

    def _do_step(self, neurons, stream, scheduler, now_tick, counts, episode, step,
                 t_game, view, log_fh, final_step: bool = False) -> float:
        """One agent-environment interaction. Returns pause seconds (0 if none)."""
        snap = self.source.snapshot(t_game)
        if snap is None:
            return 0.0  # live source warming up

        feats = self.features.update(snap)
        window_counts = counts.copy()
        action, debug = self._decide(window_counts)
        # An open on the episode's final step would be flattened at this same
        # snapshot moments later: entry at the ask, exit at the bid, a
        # guaranteed spread+commission loss and a punishment the culture cannot
        # learn to avoid (it has no time-remaining input). The backtester
        # already refuses to open into a boundary; mirror it here. The decoded
        # action is still logged as the culture's prediction.
        suppressed_open = final_step and self.engine.position == 0 and action != Action.HOLD
        result = self.engine.step(Action.HOLD if suppressed_open else action, snap)

        feedback = None
        if self.cfg.session.mode == "agent":
            # No holding mini on the final step: the episode-end flatten's full
            # feedback is scheduled on this same tick, and stacking both trains
            # (plus sensory) would breach the 200 Hz per-channel stim budget.
            feedback = self.reward.evaluate(result, allow_holding=not final_step)
            if feedback:
                scheduler.schedule(now_tick, feedback.commands)
                if feedback.kind == "predictable":
                    self.n_predictable += 1
                else:
                    self.n_unpredictable += 1

        if feedback is None or feedback.pause_s <= 0:
            sensory = self.encoder.encode_step(feats, self.engine.position, result.unrealized_points)
            scheduler.schedule(now_tick, sensory)
        # else: play is paused for the feedback window and the pause-expiry
        # re-prime delivers fresh sensory state; scheduling sensory here too
        # would only mix market-dependent stim into the feedback being
        # delivered (making the "predictable" reward less predictable) and
        # double the sensory dose around every pause.

        row = dict(
            t=round(t_game, 3), wall_t=time.time(), episode=episode, step=step,
            mid=snap.mid, bid=snap.bid, ask=snap.ask,
            momentum_norms=[round(v, 4) for v in feats["momentum_norms"]],
            vol_points=round(feats["vol_points"], 4),
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
        )
        if suppressed_open:
            row["suppressed_open"] = True  # decoded action logged above; not executed
        self._log_row(log_fh, stream, neurons, row)

        view.update(self._view_state(
            phase="episode", episode=episode, step=step, mid=snap.mid,
            momentum_norms=feats["momentum_norms"], action=action.value,
            diff=debug.get("diff", 0.0) if isinstance(debug, dict) else 0.0,
            buy_count=debug.get("raw", {}).get("buy", 0) if isinstance(debug, dict) else 0,
            sell_count=debug.get("raw", {}).get("sell", 0) if isinstance(debug, dict) else 0,
            position=result.position, unrealized_points=result.unrealized_points,
            realized_dollars=result.realized_dollars, n_trades=len(self.engine.trades),
        ))
        return feedback.pause_s if feedback else 0.0

    def _flatten_with_feedback(self, neurons, stream, scheduler, now_tick, episode,
                               t_game, log_fh, label: str) -> tuple[bool, float]:
        """Force-close any open position; the outcome is real, so feedback still flows.

        Returns (ok, pause_s). ok is False only when a position is open but no
        quote is available to close it against -- the caller must not proceed
        as if flat. pause_s is the feedback's play-suspension window: episode-end
        callers can ignore it (rest follows anyway), but a mid-episode roll must
        honor it or the full feedback's evoked spikes land in the next decision.
        """
        snap = self.source.snapshot(t_game)
        if snap is None:
            return self.engine.position == 0, 0.0
        result = self.engine.flatten(snap)
        if result is None:
            return True, 0.0  # nothing to flatten
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
            momentum_norms=[], vol_points=0.0,
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
        return True, (feedback.pause_s if feedback else 0.0)

    def _end_episode(self, neurons, stream, scheduler, now_tick, episode, t_game, view, log_fh) -> None:
        ok, _ = self._flatten_with_feedback(neurons, stream, scheduler, now_tick, episode,
                                            t_game, log_fh, "flatten")
        if not ok:
            view.log(f"warning: episode {episode} position NOT flattened (no market data); "
                     "it will carry into the next episode")
        view.log(f"episode {episode} done: realized ${self.engine.realized_dollars:+.2f} "
                 f"({len(self.engine.trades)} trades)")

    def _handle_roll(self, neurons, stream, scheduler, now_tick, episode, t_game, view, log_fh) -> float:
        """Contract roll: flatten on the old contract, re-anchor, then resubscribe.

        Order matters -- the flatten must execute against the old contract's
        quotes, and the feature window and mark-to-market anchor must be reset
        before the first snapshot of the new contract arrives, so the basis
        jump between contracts never reaches the network or the PnL.

        Returns the flatten feedback's pause window (seconds): a mid-episode
        caller must suspend play for it, exactly as after a per-step trade
        close, or the feedback's evoked spikes contaminate the next decision.
        """
        info = self.source.pending_roll()
        if info is None:
            return 0.0
        ok, pause_s = self._flatten_with_feedback(neurons, stream, scheduler, now_tick, episode,
                                                  t_game, log_fh, "roll_flatten")
        if not ok:
            # No quote to close against: do NOT roll with an open position (the
            # inter-contract basis jump would land in the trade's PnL). The
            # pending roll persists; retry at the next boundary.
            return 0.0
        self.features.reset()
        self.decoder.reset()  # lag windows from the old contract are stale context
        self.engine.on_roll()
        self.source.complete_roll()
        note = f" ({info.note})" if info.note else ""
        view.log(f"contract roll: {info.old_streamer_symbol} -> {info.new_streamer_symbol}{note}")
        return pause_s

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
