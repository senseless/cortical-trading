"""Reward generator: the free-energy-principle learning signal.

Favorable outcomes earn predictable stimulation (structured 100 Hz bursts
across encoding and decoding regions, per DishBrain/gridworld); adverse
outcomes earn unpredictable stimulation (random sites, random timing). Reward
timing is the main experimental variable: per-tick mark-to-market shaping,
per-trade outcomes, or hybrid (small shaping + full feedback on trade close).

The shuffle flag decouples feedback from outcomes (coin flip) -- the control
condition that distinguishes real learning from stimulation artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import RewardCfg
from ..game.state import StepResult
from .layout import ElectrodeLayout
from .stim import StimCommand


@dataclass
class FeedbackEvent:
    kind: str                                  # "predictable" | "unpredictable"
    full: bool                                 # full (per-trade) vs mini (per-tick shaping)
    reward: float                              # signed reward value (points)
    commands: list[StimCommand] = field(default_factory=list)
    pause_s: float = 0.0                       # suspend env steps while noise plays out


class RewardGenerator:
    def __init__(self, cfg: RewardCfg, layout: ElectrodeLayout, seed: int | None = None):
        self.cfg = cfg
        self.layout = layout
        self._rng = np.random.default_rng(seed)
        self._feedback_channels = sorted(set(layout.all_sensory) | set(layout.all_motor))

    # -- outcome evaluation ----------------------------------------------------

    def evaluate(self, step: StepResult) -> FeedbackEvent | None:
        """Map a game step outcome to a feedback event (or None)."""
        timing = self.cfg.timing
        # Full feedback on closed trades (per_trade and hybrid).
        if timing in ("per_trade", "hybrid") and step.closed_trade is not None:
            reward = step.closed_trade.points
            return self._build(reward, full=True)
        # Mark-to-market shaping (per_tick and hybrid).
        if timing in ("per_tick", "hybrid"):
            if abs(step.mtm_points) > self.cfg.deadband_points:
                return self._build(step.mtm_points, full=False)
        return None

    def _build(self, reward: float, full: bool) -> FeedbackEvent:
        favorable = reward > 0
        if self.cfg.shuffle:
            favorable = bool(self._rng.integers(0, 2))
        if favorable:
            return FeedbackEvent(kind="predictable", full=full, reward=reward,
                                 commands=self._predictable(full), pause_s=0.0)
        pause = (self.cfg.unpredictable_s + self.cfg.pause_after_loss_s) if full else 0.0
        return FeedbackEvent(kind="unpredictable", full=full, reward=reward,
                             commands=self._unpredictable(full), pause_s=pause)

    # -- stimulus construction ---------------------------------------------------

    def _predictable(self, full: bool) -> list[StimCommand]:
        """Structured bursts at predictable_hz across all encoding+decoding channels."""
        n_bursts = self.cfg.predictable_bursts if full else 1
        burst_pulses = max(1, int(round(self.cfg.predictable_hz * self.cfg.predictable_burst_ms / 1000.0)))
        window = self.cfg.predictable_window_s if full else 0.0
        spacing = window / n_bursts if n_bursts > 1 else 0.0
        return [
            StimCommand(
                channels=self._feedback_channels, rate_hz=self.cfg.predictable_hz,
                count=burst_pulses, delay_s=i * spacing,
                amplitude_ua=self.cfg.feedback_amplitude_ua, tag="reward:predictable",
            )
            for i in range(n_bursts)
        ]

    def _unpredictable(self, full: bool) -> list[StimCommand]:
        """Random-site, random-time single stims -- maximally unpredictable input."""
        duration = self.cfg.unpredictable_s if full else 0.3
        # Event budget scaled like DishBrain's 5 Hz over its 8-electrode sensory area.
        n_events = max(1, int(round(self.cfg.unpredictable_rate_hz * duration * len(self._feedback_channels) / 8.0)))
        commands = []
        for _ in range(n_events):
            channel = int(self._rng.choice(self._feedback_channels))
            delay = float(self._rng.uniform(0.0, duration))
            commands.append(StimCommand(
                channels=[channel], rate_hz=0.0, count=1, delay_s=delay,
                amplitude_ua=self.cfg.feedback_amplitude_ua, tag="reward:unpredictable",
            ))
        return commands
