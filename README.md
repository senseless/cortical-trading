# Cortical Trading

A DishBrain-style closed-loop "trading game" that embodies Cortical Labs (CL1)
biological neurons in the /MBT micro Bitcoin futures market. Market state is
encoded as electrical stimulation (place + rate coding), spike activity in
motor regions is decoded into long/short/flat actions, and the learning signal
is structured feedback: predictable stimulation after favorable outcomes,
unpredictable noise after adverse ones (free energy principle).

/MBT was chosen because CME moved cryptocurrency futures to 24/7 trading in
May 2026 (only maintenance gaps: 2 min on weekdays, 2 h on Saturday mornings),
which increases training hours available per rented CL1 month compared
to equity index futures and removes weekend gap risk.

Phase 1 runs entirely against the free CL SDK simulator (`cl-sdk`) with an
internal paper broker on live, replayed, or synthetic market data. The same
code later connects to a rented CL1 on Cortical Cloud (the CL API is a 1:1
drop-in for the simulator). By design, this system runs as a proxy on the
local machine — market data in, neural I/O out to the remote CL1, orders out
to the broker — rather than deploying code into Cortical's environment.
Research background and sources: [NOTES.md](NOTES.md).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env   # then fill in tastytrade OAuth credentials (live data only)

# The silicon baseline requires the CUDA build of torch (NVIDIA):
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Requires Python 3.11+. Synthetic and replay sessions need no credentials.

## Usage

```powershell
# Smoke-test the CL simulator (loop, stim, recording round trip)
.venv\Scripts\python smoke_cl.py

# Run a session with the default config (synthetic sine market, agent mode)
.venv\Scripts\python run_session.py

# Run with a custom config / overrides
.venv\Scripts\python run_session.py --config config/default.toml --set market.source=live
.venv\Scripts\python run_session.py --set market.synthetic.regime=trend --set session.label=trend-test

# Record live market data into the replay library (Ctrl+C to stop)
.venv\Scripts\python record_market.py --product MBT

# Library builder: all liquid CME micros over one connection, one file per
# product, independent rolls, auto-reconnect, 6h file rotation. Crypto ticks
# 24/7; the rest tick Sun 6pm - Fri 5pm ET.
.venv\Scripts\python record_market.py --product "MBT,MET,MNQ,MES,M2K,MYM,MCL,MGC,SIL,MHG,M6E" --forever

# Replay a recorded session
.venv\Scripts\python run_session.py --set market.source=replay --set "market.replay.path=data/market/<recording>.jsonl.gz"

# Silicon baseline gate: can a conventional learner (GPU MLP + logistic) find
# edge in what the neurons see? Sanity-check on sine, control on random walk,
# then point it at real recordings.
.venv\Scripts\python train_baseline.py                                # synthetic sine (must pass)
.venv\Scripts\python train_baseline.py --data synthetic:random_walk   # control (must show no signal)
.venv\Scripts\python train_baseline.py --data data/market/<recording>.jsonl.gz
.venv\Scripts\python train_baseline.py --data <recording> --horizons 60,300,900,3600   # day-trader horizons (needs multi-day recordings)

# Analyze a finished session (learning curves, action-reward alignment)
.venv\Scripts\python analyze_recording.py data/sessions/<session_dir>

# Compare two sessions (e.g. experiment vs. random control), Mann-Whitney U
.venv\Scripts\python analyze_recording.py data/sessions/<a> --compare data/sessions/<b>

# Train a reservoir-mode readout from one or more session logs
.venv\Scripts\python analyze_recording.py data/sessions/<session_dir> --train-readout
```

## How a session works

1. The market source (synthetic / replay / live DXLink) provides quotes; the
   game engine tracks position, mark-to-market PnL, and the trade log.
2. Every step (~1 s), the encoder translates market features into stimulation:
   which electrode group fires = signal direction (place coding), stimulation
   frequency 4-40 Hz = magnitude (rate coding). The neurons' current position
   is also place-coded so they can "feel" it.
3. Spikes are counted per motor region over the step window, normalized
   against pre-episode baseline activity, and decoded into an action:
   buy / sell / hold (open long, open short, close long, close short).
4. The paper broker fills at bid/ask with configurable slippage/commission.
5. Feedback: net-profitable outcomes (after commissions) trigger predictable
   100 Hz bursts; net losses trigger unpredictable random-site stimulation
   (configurable timing: per-tick, per-trade, or hybrid).
6. Sessions are episodic (default 5 episodes with rest periods, per the
   Cortical Labs gridworld findings) and everything is logged: a JSONL step
   log per session plus the CL HDF5 recording with spikes, stims, and game
   state as a data stream.

## Electrode layout

The 64 electrodes form an 8x8 grid (channel = row * 8 + column). CL1 hardware
reserves five of them: 0, 7, 56, 63 (unused grid corners) and 4 (measurement
reference). The simulator accepts stims on those channels but hardware
silently drops them, so the config validator rejects any layout that assigns
them. The default layout (`[neural.layout]` in `config/default.toml`):

```
        c0    c1    c2    c3    c4    c5    c6    c7
  r0    xx    M+    M+    M+    xx    M+     .     xx
  r1    M-    M-    M-    M-     .     .     .     .
  r2    PL    PL     .     .    U+    U+     .     .
  r3    PF    PF     .     .    U-    U-     .     .
  r4    PS    PS     .     .     .     .     .     .
  r5   BUY   BUY   BUY   BUY   SEL   SEL   SEL   SEL
  r6   BUY   BUY   BUY   BUY   SEL   SEL   SEL   SEL
  r7    xx     .     .     .     .     .     .    xx
```

`xx` reserved &nbsp;|&nbsp; `.` unassigned (still recorded) &nbsp;|&nbsp;
sensory: `M+/M-` momentum up/down, `PL/PF/PS` position long/flat/short,
`U+/U-` unrealized PnL up/down &nbsp;|&nbsp; motor: `BUY`/`SEL` decode regions

Why this shape:

- **Place + rate coding.** Each signal gets a dedicated electrode group
  (*where* = meaning); stimulation frequency (4-40 Hz) carries magnitude.
  Exactly-zero signals are silence, not a weak positive.
- **Opposite valences on separate rows.** Stimulation activates tissue beyond
  the target electrode (~100 um), so groups whose meanings are opposites
  (`M+`/`M-`, `U+`/`U-`) get physical separation — blending them would
  destroy the signal they carry.
- **Sensory top, motor bottom.** Maximizing distance between stimulation
  sites and decode regions keeps directly evoked activity from dominating
  the spike counts the decoder reads; the decision should ride on network
  dynamics, not on stim artifacts.
- **Contiguous motor blocks.** Buy and sell are 2x4 regions side by side
  (mirroring DishBrain's two paddle regions); decoding compares region-level
  spike counts, which is robust to single-electrode noise.

### Planned: chronotopic input strips (Phase 2, gated on the silicon baseline)

If the baseline gate finds after-cost signal at several momentum horizons,
the two momentum groups become two topographic *strips*: timescale ordered
short -> long along each strip — "chronotopy", by analogy with the tonotopic
frequency axis of auditory cortex — one strip for up-momentum, one for down,
separated by a guard row.

The rationale: adjacent electrodes recruit overlapping neural populations,
and neurons in culture are more likely to be synaptically connected the
closer they are. Momentum at neighboring timescales is strongly correlated,
so placing those signals on neighboring sites turns electrode cross-talk
into generalization rather than interference. Blending is *desired* along
the timescale axis and *destructive* across valence — hence parallel strips
with a guard row between them. This also gives classic multi-timeframe
structure a spatial signature: a short-term pullback inside a long-term
uptrend lights the down strip at its short end and the up strip at its long
end simultaneously — a pattern a single-window encoding cannot represent.
The layout is pure configuration (`[neural.layout.sensory]`); the encoder
additionally needs a per-window rate scale, since raw points-per-second
momentum shrinks roughly with the square root of the window length.

## Contract rolls

Live sessions and recordings always start on the tradeable front month: the
tastytrade active-month contract, unless it is within `roll_cutoff_days`
(default 7) of its last trading day, in which case the next contract is used.
While running, the live source re-checks hourly; when the front month changes,
any open position is flattened on the old contract, the feature window and
mark-to-market anchor are reset, and the stream resubscribes to the new
contract. Recordings rotate into a new file at the roll, so one file never
spans two contracts — never stitch files across contracts, since each trades
at a different price level (basis). The network itself never sees absolute
price (only velocity, position, and PnL), so a roll feels like a fresh market
open, not a price shock.

## Curriculum

Synthetic regimes in increasing difficulty: `sine`, `trend`, `trend_noise`,
`mean_revert`, `random_walk` (unlearnable control), then `replay` of real
/MBT days, then `live`. The `market.synthetic.snr` dial scales signal vs.
noise. A culture must beat its own random baseline on predictable regimes
before graduating.

The silicon baseline (`train_baseline.py`) is the gate in front of all of it:
if no conventional learner can extract edge from the encoded features at game
cadence after real costs, the game design needs work before spending wetware
rental time. It reports directional accuracy per horizon against the
majority-class benchmark (with overlap-adjusted confidence intervals) for
both the current neural encoding and an extended candidate-channel set —
returns over a day-trader ladder of windows from 1 s to 4 h, vol-normalized
variants, volatility regime, spread, book imbalance, range position, time of
day — and backtests the resulting policies through the real game engine and
paper broker against time-shifted luck baselines. The hour-scale windows and
label horizons only mean anything on multi-day contiguous recordings, which
is what the `--forever` library builder accumulates.

## Project layout

- `src/market/` - market sources (synthetic, replay, live DXLink), recorder
- `src/game/` - game engine: position state machine, PnL, trade log
- `src/neural/` - CL backend wrapper, electrode layout, encoder, decoder, reward
- `src/broker/` - paper broker (live tastytrade broker arrives in the live phase)
- `src/session/` - session runner, metrics, console view
- `src/analysis/` - session analysis, control comparisons, reservoir readout training
- `src/baseline/` - silicon baseline: dataset builder, logistic/GPU-MLP probes, policy backtest
- `config/` - TOML session configs

## Important notes

- The CL SDK simulator generates random spikes and does not learn. Simulator
  sessions are pipeline validation and the random-control baseline, nothing more.
- The tastytrade sandbox is deliberately not used (15-minute delayed data,
  daily resets). Paper trading = live production data + internal fill engine.
- Live order routing (Phase 4) is gated on beating controls across repeated
  sessions on real neurons (Phase 3); the broker interface is already in place.
