"""Synthetic market generator with a tunable signal-to-noise dial.

Provides the learnability curriculum: regimes range from trivially predictable
(sine, clean trend) to unlearnable (random walk). A culture that cannot learn
to trade a sine wave will not learn a real market; this is the control that
makes "did it learn?" answerable.

Deterministic: the same seed and parameters produce the same price path,
regardless of query timing, because the path is advanced in fixed dt steps.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import SyntheticCfg
from .source import MarketSnapshot, MarketSource

REGIMES = ("sine", "trend", "trend_noise", "mean_revert", "random_walk")


class SyntheticSource(MarketSource):
    def __init__(self, cfg: SyntheticCfg):
        if cfg.regime not in REGIMES:
            raise ValueError(f"unknown synthetic regime '{cfg.regime}', expected one of {REGIMES}")
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        self._t = 0.0
        self._noise = 0.0  # integrated brownian noise (points)
        self._ou = 0.0     # OU deviation for mean_revert (points)

    def describe(self) -> str:
        c = self.cfg
        return f"synthetic:{c.regime} seed={c.seed} snr={c.snr} noise_vol={c.noise_vol}"

    def _advance_to(self, t: float) -> None:
        dt = self.cfg.dt_s
        noise_mult = 3.0 if self.cfg.regime == "trend_noise" else 1.0
        sigma = self.cfg.noise_vol * noise_mult
        while self._t + dt <= t:
            z = self._rng.standard_normal()
            self._noise += sigma * math.sqrt(dt) * z
            # OU: reversion strength scaled by the snr dial (stronger pull = more
            # predictable). Exact exponential decay instead of explicit Euler:
            # Euler's (1 - k*dt) factor diverges when k*dt > 2, i.e. an extreme
            # --snr would silently turn "easiest regime" into numeric overflow.
            k = self.cfg.mean_revert_rate * self.cfg.snr
            self._ou = self._ou * math.exp(-k * dt) + sigma * math.sqrt(dt) * z
            self._t += dt

    def _signal(self, t: float) -> float:
        c = self.cfg
        if c.regime == "sine":
            return c.amplitude * math.sin(2.0 * math.pi * t / c.period_s)
        if c.regime in ("trend", "trend_noise"):
            return c.drift_per_s * t
        return 0.0

    def _price(self, t: float) -> float:
        c = self.cfg
        if c.regime == "mean_revert":
            return c.start_price + self._ou
        if c.regime == "random_walk":
            return c.start_price + self._noise
        # Signal regimes: the snr dial scales the signal against the noise path.
        # snr=4 (default) leaves the configured amplitude unchanged; snr=0 removes the signal.
        return c.start_price + self._signal(t) * (c.snr / 4.0) + self._noise

    def snapshot(self, t: float) -> MarketSnapshot:
        self._advance_to(t)
        mid = self._price(t)
        half = self.cfg.spread / 2.0
        return MarketSnapshot(t=t, bid=mid - half, ask=mid + half, last=mid, bid_size=10, ask_size=10)
