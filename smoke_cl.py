"""Smoke test for the CL SDK simulator: loop, spikes, stim, data stream, recording.

Verifies the exact API surface the session runner depends on. Run with:
    .venv\\Scripts\\python smoke_cl.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("CL_SDK_RANDOM_SEED", "42")
os.environ.setdefault("CL_SDK_VISUALISATION", "0")

OUT_DIR = Path("data/tmp")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    import cl
    from cl import BurstDesign, ChannelSet, StimDesign

    print(f"cl module: {cl.__file__}")

    with cl.open() as neurons:
        stream = neurons.create_data_stream(name="smoke_stream", attributes={"score": 0})
        recording = neurons.record(file_location=str(OUT_DIR))

        spike_counts: dict[int, int] = {}
        stims_issued = 0
        stim_design = StimDesign(80, -2.5, 80, 2.5)
        burst = BurstDesign(4, 40)

        t_start = time.time()
        for tick in neurons.loop(ticks_per_second=20, stop_after_seconds=3):
            for spike in tick.analysis.spikes:
                spike_counts[spike.channel] = spike_counts.get(spike.channel, 0) + 1
            if tick.iteration == 10:
                neurons.stim(ChannelSet(0, 1, 2, 3), stim_design, burst)
                stims_issued += 1
            if tick.iteration == 20:
                ts = neurons.timestamp()
                stream.append(ts, {"price": 21000.0, "action": "hold"})
                stream.set_attribute("score", 1)
        elapsed = time.time() - t_start

        recording.stop()

    total_spikes = sum(spike_counts.values())
    channels_seen = sorted(spike_counts)
    print(f"loop ran 3s (wall {elapsed:.2f}s) at 20 Hz")
    print(f"spikes detected: {total_spikes} across {len(channels_seen)} channels")
    if channels_seen:
        print(f"channel range: {channels_seen[0]}..{channels_seen[-1]}")
    print(f"stims issued: {stims_issued}")

    recordings = sorted(OUT_DIR.glob("*.h5"), key=lambda p: p.stat().st_mtime)
    if recordings:
        newest = recordings[-1]
        print(f"recording file: {newest} ({newest.stat().st_size} bytes)")
        from cl import RecordingView

        with RecordingView(str(newest)) as view:
            print(f"recorded spikes: {len(view.spikes)}, stims: {len(view.stims)}")
            print(f"data streams: {list(view.data_streams.keys())}")
    else:
        print("WARNING: no .h5 recording found in", OUT_DIR)

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
