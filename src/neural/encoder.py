"""Encoder: market state -> place + rate coded stimulation.

Mirrors the DishBrain/gridworld encoding scheme: which electrode group fires
carries the signal's identity and sign (place coding); stimulation frequency
between f_min and f_max carries magnitude (rate coding). Sensory channels are
momentum, book imbalance, position state, and unrealized PnL, each claiming
its own electrode groups from the layout. Groups absent from the layout are
skipped, so channels can be enabled/disabled purely via config.

Momentum is delivered on two chronotopic strips (one per direction): the k-th
configured window is stimulated on the k-th channel of the strip, so timescale
is a spatial axis ordered short -> long. Different windows can disagree in
sign, and that is the point -- a short-term pullback inside a longer uptrend
lights the down strip's short end and the up strip's long end at the same
time, a pattern a single-window encoding cannot represent.
"""

from __future__ import annotations

import math

from ..config import EncodingCfg
from .layout import ElectrodeLayout
from .stim import StimCommand


class Encoder:
    def __init__(self, cfg: EncodingCfg, layout: ElectrodeLayout, step_interval_s: float):
        self.cfg = cfg
        self.layout = layout
        self.step_interval_s = step_interval_s

    def _rate(self, magnitude: float) -> float:
        """Map |magnitude| in [0,1] to a stim rate in [f_min, f_max]."""
        m = min(max(abs(magnitude), 0.0), 1.0)
        return self.cfg.f_min_hz + m * (self.cfg.f_max_hz - self.cfg.f_min_hz)

    def _burst(self, channels: list[int], rate_hz: float, tag: str,
               start_offset_s: float = 0.0) -> StimCommand | None:
        """A train at rate_hz that ends inside the step window.

        The CL SDK reserves a channel for count/rate seconds per burst (the
        trailing inter-pulse interval counts) and serializes commands per
        channel, so a train that runs even slightly past the step boundary
        delays the next step's train, and the offset compounds every step.
        Taking floor(rate * budget) pulses keeps count/rate <= budget: the
        inter-pulse interval -- the rate code itself -- is exact, and the
        train can never push the following one. start_offset_s carves out
        the head of the step for feedback that must play first.
        """
        budget = self.step_interval_s - start_offset_s
        if budget <= 0.0:
            return None
        count = max(1, int(math.floor(rate_hz * budget + 1e-9)))
        return StimCommand(
            channels=channels, rate_hz=rate_hz, count=count, delay_s=start_offset_s,
            amplitude_ua=self.cfg.amplitude_ua, pulse_width_us=self.cfg.pulse_width_us,
            tag=f"sensory:{tag}",
        )

    def encode_step(self, features: dict, position: int, unrealized_points: float,
                    start_offset_s: float = 0.0) -> list[StimCommand]:
        """Sensory stimulation for one step.

        start_offset_s delays every train by that much and shortens it to fit
        the remaining window: the runner passes the duration of a mini
        feedback issued on the same tick, so the feedback plays first at its
        exact time and the market state follows, instead of queueing behind
        it and dragging every later step along.
        """
        cmds: list[StimCommand] = []
        lay = self.layout

        def add(channels: list[int], rate_hz: float, tag: str) -> None:
            cmd = self._burst(channels, rate_hz, tag, start_offset_s)
            if cmd is not None:
                cmds.append(cmd)

        # 1. Momentum strips: sign -> which strip (place), window index -> which
        # electrode along it (place), magnitude -> rate. Exactly zero (warm-up,
        # or no move) is silence, not a weak "up" -- stimulating at f_min for
        # unknown momentum would bake in a directional bias, and it is silence
        # that lets the strip's long end stay dark until it has history.
        for k, v in enumerate(features.get("momentum_norms", ())):
            if v == 0.0:
                continue
            group = "momentum_up" if v > 0 else "momentum_down"
            if not lay.has(group):
                continue
            channels = lay.group(group)
            if k >= len(channels):
                continue  # strip shorter than the ladder (config validation rejects this)
            add([channels[k]], self._rate(v),
                f"momentum{'+' if v > 0 else '-'}[{self.cfg.momentum_windows_s[k]:g}s]")

        # 1b. Book imbalance: sign -> bid-heavy or ask-heavy group (place),
        # magnitude of the time-averaged imbalance -> rate. Zero (balanced
        # book, no sizes, or a window still filling after a gap) is silence.
        imb = features.get("imbalance_norm", 0.0)
        if imb != 0.0:
            imb_group = "imbalance_bid" if imb > 0 else "imbalance_ask"
            if lay.has(imb_group):
                add(lay.group(imb_group), self._rate(imb), f"imbalance{'+' if imb > 0 else '-'}")

        # 2. Position state: pure place coding at a fixed rate (the neurons feel their paddle).
        pos_group = {1: "position_long", 0: "position_flat", -1: "position_short"}[int(position)]
        if lay.has(pos_group):
            add(lay.group(pos_group), self.cfg.position_rate_hz, pos_group)

        # 3. Unrealized PnL: sign -> group, magnitude -> rate, only while holding
        # a position. This is the honest mark, not a valence cue: a fresh entry
        # is half a spread underwater (filled at the ask, marked at mid), so the
        # down group fires near f_min on the entry step; exactly zero is silence.
        if self.cfg.pnl_channel_enabled and position != 0:
            norm = math.tanh(unrealized_points / self.cfg.pnl_scale_points) if self.cfg.pnl_scale_points else 0.0
            if norm != 0.0:
                pnl_group = "pnl_up" if norm > 0 else "pnl_down"
                if lay.has(pnl_group):
                    add(lay.group(pnl_group), self._rate(norm), pnl_group)

        return cmds
