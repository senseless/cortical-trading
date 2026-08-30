"""Reward generator: the free-energy-principle learning signal.

Favorable outcomes earn predictable stimulation (structured 100 Hz bursts, per
DishBrain/gridworld); adverse outcomes earn unpredictable stimulation (random
sites, random timing). Full (per-trade) feedback spans encoding and decoding
regions and pauses play for its delivery window; mini feedback is delivered to
sensory channels only, because it has no pause and its evoked spikes land in
the next decision window.

Timing modes:
- hybrid (default): full feedback on trade close (net of costs) plus
  *holding* feedback -- every holding_interval_s while a position is open,
  its unrealized PnL is judged against holding_deadband_points and earns mini
  predictable (winner) or unpredictable (loser) stimulation. Culture credit
  assignment operates on seconds, so feedback delivered only at close cannot
  shape the entry decision across a 15-minute hold; holding feedback makes
  the contingency available continuously, and turns a losing position into an
  aversive stimulus the culture can terminate at any moment by closing --
  closed-loop stimulus-removal conditioning (Shahaf & Marom; le Feber). The
  deadband sits at ~the round-trip cost hurdle so a fresh entry (half a
  spread underwater by construction) is not punished for existing.
- per_tick (experimental arm): mini feedback on 1 s mark-to-market deltas.
- per_trade: close-time feedback only.

The shuffle flag decouples feedback from outcomes -- the control condition
that distinguishes real learning from stimulation artifacts. Valence is drawn
from a resampled history of *true* outcomes rather than a coin flip, so the
control arm receives the same marginal reward/punishment rates (and pause
budget) as the experimental arm; only the contingency is broken.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..config import RewardCfg
from ..game.state import StepResult
from .layout import ElectrodeLayout
from .stim import StimCommand


@dataclass
class FeedbackEvent:
    kind: str                                  # "predictable" | "unpredictable"
    full: bool                                 # full (per-trade) vs mini (holding/per-tick shaping)
    reward: float                              # signed reward value (points)
    commands: list[StimCommand] = field(default_factory=list)
    pause_s: float = 0.0                       # suspend env steps while noise plays out


class RewardGenerator:
    def __init__(self, cfg: RewardCfg, layout: ElectrodeLayout, seed: int | None = None,
                 step_interval_s: float = 1.0):
        self.cfg = cfg
        self.layout = layout
        self._rng = np.random.default_rng(seed)
        # Holding feedback cadence in env steps; the counter tracks how long
        # the current position has been held so the first evaluation happens a
        # full interval after entry, never on the entry step itself.
        self._holding_steps = max(1, round(cfg.holding_interval_s / step_interval_s))
        self._steps_in_position = 0
        # Full feedback spans encoding AND decoding regions (play is paused for
        # its window, so its evoked spikes never reach a decision). Mini
        # (holding / per-tick) feedback has no pause -- its evoked activity
        # lands in the very next decision window -- so it must not stimulate
        # motor channels directly, or the decoder reads the feedback as market
        # opinion. Sensory-only mini feedback also matches DishBrain, which
        # applies outcome stimulation to the sensory area.
        self._feedback_channels = sorted(set(layout.all_sensory) | set(layout.all_motor))
        self._mini_channels = list(layout.all_sensory)
        # True-outcome histories for the shuffle control, kept separately for
        # full and mini feedback (their favorable rates differ).
        self._outcomes: dict[bool, deque[bool]] = {True: deque(maxlen=50), False: deque(maxlen=50)}

    # -- outcome evaluation ----------------------------------------------------

    def evaluate(self, step: StepResult, allow_holding: bool = True) -> FeedbackEvent | None:
        """Map a game step outcome to a feedback event (or None).

        allow_holding=False suppresses holding minis while keeping the hold
        clock ticking: the runner passes it on an episode's final step, where
        the flatten's full feedback lands on the same tick -- a mini stacked
        on top would push the concurrent per-channel stim rate (sensory
        encoding + mini + full burst) past the CL1 200 Hz budget that config
        validation guarantees for any two trains.
        """
        timing = self.cfg.timing
        if step.position == 0:
            self._steps_in_position = 0
        # Full feedback on closed trades (per_trade and hybrid). Net of costs:
        # on /MBT the ~$4 round trip is worth 40 points, so rewarding gross
        # points would praise trades that lose money after commissions and
        # train the culture to churn.
        if timing in ("per_trade", "hybrid") and step.closed_trade is not None:
            reward = step.closed_trade.net_points
            # Deadband on the outcome too: a scratch trade within +/-deadband
            # of net breakeven is noise -- full 3 s punishment for -0.5 net
            # points would weigh it the same as a disaster.
            if abs(reward) <= self.cfg.deadband_points:
                return None
            return self._build(reward, full=True)
        # Mark-to-market delta shaping (per_tick experimental arm).
        if timing == "per_tick" and abs(step.mtm_points) > self.cfg.deadband_points:
            return self._build(step.mtm_points, full=False)
        # Holding-level shaping (hybrid): judge the open position's unrealized
        # PnL on a slow cadence. No pause is attached (mini feedback), which is
        # a requirement, not a convenience: the punishment must never lock the
        # culture out of the closing action that would end it.
        if timing == "hybrid" and step.position != 0:
            self._steps_in_position += 1
            if allow_holding and self._steps_in_position % self._holding_steps == 0 \
                    and abs(step.unrealized_points) > self.cfg.holding_deadband_points:
                return self._build(step.unrealized_points, full=False)
        return None

    def _build(self, reward: float, full: bool) -> FeedbackEvent:
        favorable = reward > 0
        if self.cfg.shuffle:
            # Resample a past true outcome instead of flipping a coin: a coin
            # flip would give the control arm 50/50 feedback while the real
            # arm skews toward punishment early on, confounding the comparison
            # with stim-dose and pause-budget differences.
            history = self._outcomes[full]
            history.append(favorable)
            favorable = bool(history[int(self._rng.integers(0, len(history)))])
        if favorable:
            # Full reward suspends play for the delivery window, mirroring the
            # punishment pause: the burst drives every motor channel, so the
            # next decision must not decode a window full of evoked spikes.
            pause = self.cfg.predictable_window_s if full else 0.0
            return FeedbackEvent(kind="predictable", full=full, reward=reward,
                                 commands=self._predictable(full), pause_s=pause)
        pause = (self.cfg.unpredictable_s + self.cfg.pause_after_loss_s) if full else 0.0
        return FeedbackEvent(kind="unpredictable", full=full, reward=reward,
                             commands=self._unpredictable(full), pause_s=pause)

    # -- stimulus construction ---------------------------------------------------

    def _predictable(self, full: bool) -> list[StimCommand]:
        """Structured bursts at predictable_hz (full: all channels; mini: sensory only)."""
        channels = self._feedback_channels if full else self._mini_channels
        n_bursts = self.cfg.predictable_bursts if full else 1
        burst_pulses = max(1, int(round(self.cfg.predictable_hz * self.cfg.predictable_burst_ms / 1000.0)))
        window = self.cfg.predictable_window_s if full else 0.0
        spacing = window / n_bursts if n_bursts > 1 else 0.0
        return [
            StimCommand(
                channels=channels, rate_hz=self.cfg.predictable_hz,
                count=burst_pulses, delay_s=i * spacing,
                amplitude_ua=self.cfg.feedback_amplitude_ua, tag="reward:predictable",
            )
            for i in range(n_bursts)
        ]

    def _unpredictable(self, full: bool) -> list[StimCommand]:
        """Random-site, random-time single stims -- maximally unpredictable input."""
        channels = self._feedback_channels if full else self._mini_channels
        if not channels:
            return []  # layout with no sensory groups: mini feedback has no sites
        duration = self.cfg.unpredictable_s if full else 0.3
        # Event budget scaled like DishBrain's 5 Hz over its 8-electrode sensory area.
        n_events = max(1, int(round(self.cfg.unpredictable_rate_hz * duration * len(channels) / 8.0)))
        commands = []
        for _ in range(n_events):
            channel = int(self._rng.choice(channels))
            delay = float(self._rng.uniform(0.0, duration))
            commands.append(StimCommand(
                channels=[channel], rate_hz=0.0, count=1, delay_s=delay,
                amplitude_ua=self.cfg.feedback_amplitude_ua, tag="reward:unpredictable",
            ))
        return commands
