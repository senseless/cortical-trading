"""Configuration loading: TOML file -> typed dataclasses, plus .env credentials."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


@dataclass
class SessionCfg:
    name: str = "dev"
    label: str = "experiment"
    mode: str = "agent"
    episodes: int = 5
    steps_per_episode: int = 120
    step_interval_s: float = 1.0
    rest_s: float = 90.0
    baseline_interactions: int = 60
    loop_hz: int = 20
    accelerated: bool = False
    score_metric: str = "pnl_after_costs"
    output_dir: str = "data/sessions"
    record_cl: bool = True
    console: bool = True


@dataclass
class SyntheticCfg:
    regime: str = "sine"
    seed: int = 42
    start_price: float = 78000.0
    spread: float = 40.0
    dt_s: float = 0.1
    amplitude: float = 150.0
    period_s: float = 120.0
    drift_per_s: float = 1.0
    mean_revert_rate: float = 0.05
    noise_vol: float = 7.0
    snr: float = 4.0


@dataclass
class ReplayCfg:
    path: str = ""
    speed: float = 1.0
    start_offset_s: float = 0.0


@dataclass
class LiveCfg:
    product_code: str = "MBT"
    streamer_symbol: str = ""
    roll_cutoff_days: float = 7.0  # never trade a contract this close to its last trading day
    roll_check_interval_s: float = 3600.0  # how often the live source re-resolves the contract
    stale_quote_s: float = 60.0  # no events for this long -> snapshot() returns None (game idles); 0 disables


@dataclass
class MarketCfg:
    source: str = "synthetic"
    synthetic: SyntheticCfg = field(default_factory=SyntheticCfg)
    replay: ReplayCfg = field(default_factory=ReplayCfg)
    live: LiveCfg = field(default_factory=LiveCfg)


@dataclass
class InstrumentCfg:
    symbol: str = "/MBT"
    point_value: float = 0.1
    tick_size: float = 5.0


@dataclass
class BrokerCfg:
    kind: str = "paper"
    commission_per_side: float = 2.00
    slippage_ticks: int = 0


@dataclass
class EncodingCfg:
    f_min_hz: float = 4.0
    f_max_hz: float = 40.0
    amplitude_ua: float = 2.5
    pulse_width_us: float = 80.0
    position_rate_hz: float = 8.0
    pnl_channel_enabled: bool = True
    momentum_window_s: float = 30.0
    momentum_scale: float = 1.2
    pnl_scale_points: float = 200.0


@dataclass
class DecodingCfg:
    normalization: str = "ratio"
    threshold: float = 0.25
    min_baseline_count: float = 1.0
    readout_path: str = ""
    readout_prob_threshold: float = 0.6


@dataclass
class RewardCfg:
    timing: str = "hybrid"
    shuffle: bool = False
    deadband_points: float = 10.0
    predictable_hz: float = 100.0
    predictable_burst_ms: float = 80.0
    predictable_bursts: int = 5
    predictable_window_s: float = 1.0
    unpredictable_s: float = 2.0
    unpredictable_rate_hz: float = 5.0
    pause_after_loss_s: float = 1.0
    feedback_amplitude_ua: float = 2.5


@dataclass
class NeuralCfg:
    channels: int = 64
    sdk_seed: int = 42
    sensory: dict[str, list[int]] = field(default_factory=dict)
    motor: dict[str, list[int]] = field(default_factory=dict)
    encoding: EncodingCfg = field(default_factory=EncodingCfg)
    decoding: DecodingCfg = field(default_factory=DecodingCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)


@dataclass
class Config:
    session: SessionCfg = field(default_factory=SessionCfg)
    market: MarketCfg = field(default_factory=MarketCfg)
    instrument: InstrumentCfg = field(default_factory=InstrumentCfg)
    broker: BrokerCfg = field(default_factory=BrokerCfg)
    neural: NeuralCfg = field(default_factory=NeuralCfg)
    raw: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        # A group name in both sections would let one dict shadow the other in
        # merged views, silently hiding a channel-overlap error.
        collide = set(self.neural.sensory) & set(self.neural.motor)
        if collide:
            raise ValueError(f"group name(s) {sorted(collide)} appear in both "
                             "neural.layout.sensory and neural.layout.motor")
        used: dict[int, str] = {}
        for group, chans in {**self.neural.sensory, **self.neural.motor}.items():
            for ch in chans:
                if not (0 <= ch < self.neural.channels):
                    raise ValueError(f"layout group '{group}' channel {ch} outside 0..{self.neural.channels - 1}")
                if ch in used:
                    raise ValueError(f"channel {ch} assigned to both '{used[ch]}' and '{group}'")
                used[ch] = group
        # CL1 hardware: grid corners 0/7/56/63 are unused and channel 4 is the
        # measurement reference. The simulator accepts them, but on hardware
        # those stims are silently dead -- fail at config load instead.
        if self.neural.channels == 64:
            reserved = {0, 4, 7, 56, 63} & set(used)
            if reserved:
                raise ValueError(
                    f"layout assigns CL1 reserved channels {sorted(reserved)} "
                    "(0, 7, 56, 63 = unused corners; 4 = reference channel)")
        # CL1 hard limit: max 200 Hz stimulation per channel (cell protection).
        for name, hz in (("neural.encoding.f_max_hz", self.neural.encoding.f_max_hz),
                         ("neural.reward.predictable_hz", self.neural.reward.predictable_hz)):
            if hz > 200.0:
                raise ValueError(f"{name} = {hz} exceeds the CL1 200 Hz per-channel stim limit")
        # Reward bursts must not overlap each other: with spacing (window /
        # bursts) shorter than a burst's duration, two predictable_hz trains
        # play concurrently on the same channels -- instantaneously over the
        # 200 Hz limit even though each train alone passes the checks above.
        r = self.neural.reward
        if r.predictable_bursts < 1:
            raise ValueError("neural.reward.predictable_bursts must be >= 1")
        if r.predictable_bursts > 1:
            spacing_s = r.predictable_window_s / r.predictable_bursts
            if spacing_s < r.predictable_burst_ms / 1000.0:
                raise ValueError(
                    f"neural.reward: {r.predictable_bursts} bursts of {r.predictable_burst_ms:g} ms "
                    f"within predictable_window_s={r.predictable_window_s:g} overlap "
                    f"(spacing {spacing_s * 1000:.0f} ms < burst length); they would stack "
                    "past the CL1 200 Hz per-channel limit")
        # The same limit applies to *concurrent* trains: feedback bursts hit
        # every sensory+motor channel while sensory encoding is still playing
        # into the same step window, so the worst-case per-channel rate is the
        # sum, not the max.
        max_sensory_hz = max(self.neural.encoding.f_max_hz, self.neural.encoding.position_rate_hz)
        if max_sensory_hz + self.neural.reward.predictable_hz > 200.0:
            raise ValueError(
                f"sensory ({max_sensory_hz} Hz) + feedback ({self.neural.reward.predictable_hz} Hz) "
                "can stimulate the same channel concurrently at "
                f"{max_sensory_hz + self.neural.reward.predictable_hz} Hz, over the CL1 200 Hz limit")
        if self.instrument.point_value <= 0 or self.instrument.tick_size <= 0:
            raise ValueError("instrument.point_value and tick_size must be positive "
                             "(PnL and net-points feedback divide by them)")
        if self.session.score_metric not in ("pnl_after_costs", "directional_accuracy"):
            raise ValueError(f"unknown session.score_metric: {self.session.score_metric!r}")
        if self.neural.decoding.min_baseline_count <= 0:
            raise ValueError("neural.decoding.min_baseline_count must be > 0 "
                             "(it floors the normalization denominator)")
        if self.market.synthetic.dt_s <= 0 or self.market.synthetic.period_s <= 0:
            raise ValueError("market.synthetic.dt_s and period_s must be positive")
        if self.session.episodes <= 0 or self.session.steps_per_episode <= 0:
            raise ValueError("session.episodes and steps_per_episode must be positive")
        if self.session.loop_hz <= 0 or self.session.step_interval_s <= 0:
            raise ValueError("session.loop_hz and step_interval_s must be positive "
                             "(they define the closed-loop clock)")
        if self.session.rest_s < 0 or self.session.baseline_interactions <= 0:
            raise ValueError("session.rest_s must be >= 0 and baseline_interactions > 0")
        if self.session.mode not in ("agent", "reservoir"):
            raise ValueError(f"unknown session mode: {self.session.mode}")
        if self.market.source not in ("synthetic", "replay", "live"):
            raise ValueError(f"unknown market source: {self.market.source}")
        if self.session.mode == "reservoir" and not self.neural.decoding.readout_path:
            raise ValueError("reservoir mode requires neural.decoding.readout_path (train one via analyze_recording.py --train-readout)")
        if self.session.accelerated and self.market.source == "live":
            raise ValueError("accelerated time cannot be used with the live market source")


def _apply(dc: Any, data: dict[str, Any], subsections: tuple[str, ...] = ()) -> None:
    """Copy scalar keys onto a config dataclass; typos are hard errors.

    Configs *are* the experiments here -- a silently ignored misspelled key
    means running a different experiment than the one written down.
    """
    name = type(dc).__name__
    for key, value in data.items():
        if isinstance(value, dict):
            if key not in subsections:
                raise ValueError(f"unknown config subsection '{key}' under {name}")
            continue
        if not hasattr(dc, key):
            raise ValueError(f"unknown config key '{key}' in {name}")
        setattr(dc, key, value)


def load_config(path: str | Path) -> Config:
    """Load a TOML config file into a Config, layered over defaults."""
    load_dotenv()
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(raw) - {"session", "market", "instrument", "broker", "neural"}
    if unknown:
        raise ValueError(f"unknown top-level config section(s): {sorted(unknown)}")
    cfg = Config(raw=raw)

    _apply(cfg.session, raw.get("session", {}))
    market = raw.get("market", {})
    _apply(cfg.market, market, subsections=("synthetic", "replay", "live"))
    _apply(cfg.market.synthetic, market.get("synthetic", {}))
    _apply(cfg.market.replay, market.get("replay", {}))
    _apply(cfg.market.live, market.get("live", {}))
    _apply(cfg.instrument, raw.get("instrument", {}))
    _apply(cfg.broker, raw.get("broker", {}))

    neural = raw.get("neural", {})
    _apply(cfg.neural, neural, subsections=("layout", "encoding", "decoding", "reward"))
    layout = neural.get("layout", {})
    unknown = set(layout) - {"sensory", "motor"}
    if unknown:
        raise ValueError(f"unknown neural.layout subsection(s): {sorted(unknown)}")
    cfg.neural.sensory = {k: list(v) for k, v in layout.get("sensory", {}).items()}
    cfg.neural.motor = {k: list(v) for k, v in layout.get("motor", {}).items()}
    _apply(cfg.neural.encoding, neural.get("encoding", {}))
    _apply(cfg.neural.decoding, neural.get("decoding", {}))
    _apply(cfg.neural.reward, neural.get("reward", {}))

    cfg.validate()
    return cfg


def tastytrade_credentials() -> dict[str, str]:
    """Read tastytrade OAuth credentials from the environment (.env is loaded)."""
    load_dotenv()
    return {
        "client_secret": os.getenv("TASTYTRADE_CLIENT_SECRET", ""),
        "refresh_token": os.getenv("TASTYTRADE_REFRESH_TOKEN", ""),
    }
