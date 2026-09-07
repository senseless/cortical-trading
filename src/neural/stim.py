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
    # Any rejected plan (queue full, sync or deferred-interrupt limits, stale
    # run timestamp) means this command was not delivered; on hardware letting
    # it propagate would end the session, so report it and carry on.
    rejected = getattr(cl_mod, "TransactionRejected", cl_mod.ChannelQueueFull)
    try:
        if cmd.count > 1:
            neurons.stim(channel_set, design, cl_mod.BurstDesign(cmd.count, rate_hz))
        else:
            neurons.stim(channel_set, design)
        return True
    except rejected as exc:
        log.warning("stim rejected (%s), dropped %s on channels %s",
                    type(exc).__name__, cmd.tag, cmd.channels)
        return False


def occupancy_s(cmd: StimCommand) -> float:
    """Seconds a channel is reserved by this command, per the CL SDK.

    A burst reserves count/rate (the interval after the last pulse counts,
    not just the span up to it); a single pulse is effectively instantaneous
    at loop-tick resolution.
    """
    if cmd.count > 1 and cmd.rate_hz > 0:
        return cmd.count / min(cmd.rate_hz, MAX_STIM_HZ)
    return 0.0


@dataclass
class StimScheduler:
    """Executes StimCommands at their due loop tick (never sleeps in the CL loop).

    Also mirrors the SDK's per-channel availability so callers can tell when
    stimulation really ends: a command issued to a busy channel starts when
    the channel frees up (never stacks), and a multi-channel command starts
    on all its channels together, at the latest of their free times (the SDK
    inserts a sync). Units are loop ticks (float).
    """

    loop_hz: float
    _queue: list[tuple[int, StimCommand]] = field(default_factory=list)
    _free_at: dict[int, float] = field(default_factory=dict)

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

    def mark_issued(self, issue_tick: int, cmd: StimCommand) -> tuple[float, float]:
        """Record that cmd was handed to the SDK during loop tick issue_tick.

        Returns (delay_ticks, end_tick). delay_ticks > 0 means the command
        queued behind earlier stimulation instead of playing on time; end_tick
        is when its channels are free again, on the tick axis of the loop's
        frame windows (tick k covers [k, k+1)), so a spike-count window
        [k - n, k) is stim-free iff k - end_tick >= n.

        A command issued from the body of tick k is stamped by the SDK with
        the end of that tick's frames, so its nominal start is k + 1.
        """
        nominal = float(issue_tick + 1)
        start = nominal
        for ch in cmd.channels:
            start = max(start, self._free_at.get(ch, start))
        end = start + occupancy_s(cmd) * self.loop_hz
        for ch in cmd.channels:
            self._free_at[ch] = end
        return start - nominal, end

    @property
    def pending(self) -> int:
        return len(self._queue)

    def clear(self) -> None:
        self._queue.clear()
