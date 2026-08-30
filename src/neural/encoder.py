"""Encoder: market state -> place + rate coded stimulation.

Mirrors the DishBrain/gridworld encoding scheme: which electrode group fires
carries the signal's identity and sign (place coding); stimulation frequency
between f_min and f_max carries magnitude (rate coding). Sensory channels are
momentum, position state, and unrealized PnL, each claiming its own electrode
groups from the layout. Groups absent from the layout are skipped, so channels
can be enabled/disabled purely via config.

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

    def _burst(self, channels: list[int], rate_hz: float, tag: str) -> StimCommand:
        # Fill the step window with pulses at rate_hz (rate coding for this interval).
        count = max(1, int(round(rate_hz * self.step_interval_s)))
        return StimCommand(
            channels=channels, rate_hz=rate_hz, count=count,
            amplitude_ua=self.cfg.amplitude_ua, pulse_width_us=self.cfg.pulse_width_us,
            tag=f"sensory:{tag}",
        )

    def encode_step(self, features: dict, position: int,
                    unrealized_points: float) -> list[StimCommand]:
        cmds: list[StimCommand] = []
        lay = self.layout

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
            cmds.append(self._burst([channels[k]], self._rate(v),
                                    f"momentum{'+' if v > 0 else '-'}[{self.cfg.momentum_windows_s[k]:g}s]"))

        # 2. Position state: pure place coding at a fixed rate (the neurons feel their paddle).
        pos_group = {1: "position_long", 0: "position_flat", -1: "position_short"}[int(position)]
        if lay.has(pos_group):
            cmds.append(self._burst(lay.group(pos_group), self.cfg.position_rate_hz, pos_group))

        # 3. Unrealized PnL: sign -> group, magnitude -> rate. Only while holding
        # a position; exactly zero (fresh entry) is silence, same as momentum.
        if self.cfg.pnl_channel_enabled and position != 0:
            norm = math.tanh(unrealized_points / self.cfg.pnl_scale_points) if self.cfg.pnl_scale_points else 0.0
            if norm != 0.0:
                pnl_group = "pnl_up" if norm > 0 else "pnl_down"
                if lay.has(pnl_group):
                    cmds.append(self._burst(lay.group(pnl_group), self._rate(norm), pnl_group))

        return cmds
