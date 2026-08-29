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
class GameCfg:
    max_position: int = 1
    flatten_at_episode_end: bool = True


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
    backend: str = "auto"
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
    game: GameCfg = field(default_factory=GameCfg)
    broker: BrokerCfg = field(default_factory=BrokerCfg)
    neural: NeuralCfg = field(default_factory=NeuralCfg)
    raw: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        used: dict[int, str] = {}
        for group, chans in {**self.neural.sensory, **self.neural.motor}.items():
            for ch in chans:
                if not (0 <= ch < self.neural.channels):
                    raise ValueError(f"layout group '{group}' channel {ch} outside 0..{self.neural.channels - 1}")
                if ch in used:
                    raise ValueError(f"channel {ch} assigned to both '{used[ch]}' and '{group}'")
                used[ch] = group
        if self.session.mode not in ("agent", "reservoir"):
            raise ValueError(f"unknown session mode: {self.session.mode}")
        if self.market.source not in ("synthetic", "replay", "live"):
            raise ValueError(f"unknown market source: {self.market.source}")
        if self.session.mode == "reservoir" and not self.neural.decoding.readout_path:
            raise ValueError("reservoir mode requires neural.decoding.readout_path (train one via analyze_recording.py --train-readout)")
        if self.session.accelerated and self.market.source == "live":
            raise ValueError("accelerated time cannot be used with the live market source")


def _apply(dc: Any, data: dict[str, Any]) -> None:
    for key, value in data.items():
        if hasattr(dc, key) and not isinstance(value, dict):
            setattr(dc, key, value)


def load_config(path: str | Path) -> Config:
    """Load a TOML config file into a Config, layered over defaults."""
    load_dotenv()
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    cfg = Config(raw=raw)

    _apply(cfg.session, raw.get("session", {}))
    market = raw.get("market", {})
    _apply(cfg.market, market)
    _apply(cfg.market.synthetic, market.get("synthetic", {}))
    _apply(cfg.market.replay, market.get("replay", {}))
    _apply(cfg.market.live, market.get("live", {}))
    _apply(cfg.instrument, raw.get("instrument", {}))
    _apply(cfg.game, raw.get("game", {}))
    _apply(cfg.broker, raw.get("broker", {}))

    neural = raw.get("neural", {})
    _apply(cfg.neural, neural)
    layout = neural.get("layout", {})
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
