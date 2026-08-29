"""Hardware-agnostic stimulation commands and their delivery to the CL API."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class StimCommand:
    channels: list[int]
    rate_hz: float          # burst rate (ignored when count == 1)
    count: int = 1          # pulses in the burst
    delay_s: float = 0.0    # schedule offset relative to "now"
    amplitude_ua: float = 2.5
    pulse_width_us: float = 80.0
    tag: str = ""           # for logging: "sensory:momentum", "reward:predictable", ...


MAX_STIM_HZ = 200.0  # CL1 hard limit per channel (cell protection)


def issue(neurons, cl_mod, cmd: StimCommand) -> bool:
    """Deliver a StimCommand via the CL API. Returns False if the queue was full."""
    if not cmd.channels:
        return True
    # Pulse width must be an integer multiple of 20 us (50 kHz max stim rate).
    width = max(20, int(round(cmd.pulse_width_us / 20.0)) * 20)
    design = cl_mod.StimDesign(width, -abs(cmd.amplitude_ua),
                               width, abs(cmd.amplitude_ua))
    channel_set = cl_mod.ChannelSet(list(cmd.channels))
    rate_hz = cmd.rate_hz
    if rate_hz > MAX_STIM_HZ:
        log.warning("stim rate %.0f Hz for %s exceeds the CL1 %d Hz limit, clamping",
                    rate_hz, cmd.tag, int(MAX_STIM_HZ))
        rate_hz = MAX_STIM_HZ
    try:
        if cmd.count > 1:
            neurons.stim(channel_set, design, cl_mod.BurstDesign(cmd.count, rate_hz))
        else:
            neurons.stim(channel_set, design)
        return True
    except cl_mod.ChannelQueueFull:
        log.warning("stim queue full, dropped %s on channels %s", cmd.tag, cmd.channels)
        return False


@dataclass
class StimScheduler:
    """Executes StimCommands at their due loop tick (never sleeps in the CL loop)."""

    loop_hz: float
    _queue: list[tuple[int, StimCommand]] = field(default_factory=list)

    def schedule(self, now_tick: int, commands: list[StimCommand]) -> None:
        for cmd in commands:
            due = now_tick + int(round(cmd.delay_s * self.loop_hz))
            self._queue.append((due, cmd))
        self._queue.sort(key=lambda item: item[0])

    def due(self, now_tick: int) -> list[StimCommand]:
        ready: list[StimCommand] = []
        while self._queue and self._queue[0][0] <= now_tick:
            ready.append(self._queue.pop(0)[1])
        return ready

    @property
    def pending(self) -> int:
        return len(self._queue)

    def clear(self) -> None:
        self._queue.clear()
