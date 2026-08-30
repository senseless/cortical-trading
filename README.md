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
# label horizons default to holding-period scale (60,300,900,3600 s); the
# hour-scale ones only mean anything on multi-day recordings

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
   frequency 4-40 Hz = magnitude (rate coding). Momentum is delivered on two
   chronotopic strips (up and down), with timescale as a spatial axis: the
   k-th window of `momentum_windows_s` stimulates the k-th electrode along
   the strip. The neurons' current position is also place-coded so they can
   "feel" it.
3. Spikes are counted per motor region over the step window, normalized
   against pre-episode baseline activity, and decoded into an action:
   buy / sell / hold (open long, open short, close long, close short).
4. The paper broker fills at bid/ask with configurable slippage/commission.
5. Feedback: closing a trade with a net profit (after commissions) triggers
   full predictable 100 Hz bursts; a net loss triggers full unpredictable
   random-site stimulation. While a position is open, its unrealized PnL is
   also judged every `holding_interval_s` (default 20 s) against a deadband
   of roughly the round-trip cost (default 100 points): holding a winner
   earns recurring mini predictable feedback, holding a loser earns
   recurring unpredictable noise that stops the moment the culture closes
   the position -- closed-loop stimulus removal, the best-replicated
   conditioning protocol in the culture literature. Culture credit
   assignment operates on seconds, so close-time-only feedback could not
   shape entries across a 15-minute hold.
6. Sessions are episodic (rest periods between episodes, per the Cortical
   Labs gridworld findings). Episodes default to 15 minutes because /MBT's
   cost structure makes shorter trades unprofitable (see the ladder rationale
   below) — the culture must be able to *hold* a position at the horizon
   where being right pays. Everything is logged: a JSONL step log per session
   plus the CL HDF5 recording with spikes, stims, and game state as a data
   stream.
7. Before the first episode, the market source pre-rolls history covering the
   strip's longest window (recorded data before the replay offset, DXLink
   candle backfill on live, generated path on synthetic), so even the 8 h
   momentum channel is live from the first step.

## Electrode layout

The 64 electrodes form an 8x8 grid (channel = row * 8 + column). CL1 hardware
reserves five of them: 0, 7, 56, 63 (unused grid corners) and 4 (measurement
reference). The simulator accepts stims on those channels but hardware
silently drops them, so the config validator rejects any layout that assigns
them. The default layout (`[neural.layout]` in `config/default.toml`):

```
        c0    c1    c2    c3    c4    c5    c6    c7
  r0    xx    U+    U+     .    xx    U-    U-    xx
  r1   M+1   M+2   M+3   M+4   M+5   M+6   M+7   M+8    up strip:   30s -> 8h
  r2     .     .     .     .     .     .     .     .    guard row
  r3   M-1   M-2   M-3   M-4   M-5   M-6   M-7   M-8    down strip: 30s -> 8h
  r4    PL    PL     .    PF    PF     .    PS    PS
  r5    PL    PL     .    PF    PF     .    PS    PS
  r6     .   BUY   BUY   BUY   SEL   SEL   SEL     .
  r7    xx   BUY   BUY   BUY   SEL   SEL   SEL   xx
```

`xx` reserved &nbsp;|&nbsp; `.` unassigned (still recorded) &nbsp;|&nbsp;
sensory: `M+k`/`M-k` momentum up/down at the k-th timescale (`momentum_windows_s`
= 30 s, 1 m, 5 m, 15 m, 30 m, 1 h, 4 h, 8 h), `PL/PF/PS` position
long/flat/short, `U+/U-` unrealized PnL up/down &nbsp;|&nbsp; motor:
`BUY`/`SEL` decode regions

Why this shape:

- **Place + rate coding.** Each signal gets a dedicated electrode group
  (*where* = meaning); stimulation frequency (4-40 Hz) carries magnitude.
  Exactly-zero signals are silence, not a weak positive.
- **Chronotopic momentum strips.** Timescale runs along each strip, short to
  long — "chronotopy", by analogy with the tonotopic frequency axis of
  auditory cortex. Adjacent electrodes recruit overlapping populations, and
  neurons in culture are likelier to be synaptically connected the closer
  they are; since momentum at neighboring timescales is strongly correlated,
  putting those signals on neighboring sites turns electrode cross-talk into
  generalization rather than interference. Blending is *wanted* along the
  timescale axis and *destructive* across valence — hence two parallel strips
  with a guard row (r2) between them.
- **Multi-timeframe structure becomes spatial.** Windows can disagree in
  sign, and that is the point: a short-term pullback inside a long-term
  uptrend lights the down strip's short end and the up strip's long end at
  the same time — a pattern no single-window encoding can represent.
- **Opposite valences separated.** Stimulation activates tissue beyond the
  target electrode (~100 um), so groups whose meanings are opposites
  (`M+`/`M-` on separate rows, `U+`/`U-` split across reserved c4, `PL`/`PS`
  at opposite ends of their rows with guard columns c2/c5) get physical
  separation — blending them would destroy the signal they carry. The cost of
  a 64-electrode budget is that r0's PnL groups sit directly above the up
  strip; cross-*signal* bleed is tolerated where cross-*valence* bleed is not.
- **Sensory top, motor bottom.** Maximizing distance between stimulation
  sites and decode regions keeps directly evoked activity from dominating
  the spike counts the decoder reads; the decision should ride on network
  dynamics, not on stim artifacts. Motor lives on the bottom two rows, three
  rows below the nearest high-rate strip; the mildest sensory group
  (position, a fixed 8 Hz) is the one placed nearest.
- **Contiguous, symmetric motor blocks.** Buy and sell are mirror-image 2x3
  regions (echoing DishBrain's two paddle regions); decoding compares
  region-level spike counts, so equal size and geometry keep the buy/sell
  differential unbiased, and region sums are robust to single-electrode
  noise. Position states are 2x2 blocks for the same redundancy.

### Why the ladder runs 30 s to 8 h

The window ladder follows the instrument's cost structure, not taste. An
/MBT round trip costs about 90 points before the market has to move at all:
a ~50-point median spread (measured on live off-peak data; 35-45 at busier
times) plus 40 points of commissions ($2.00/side at $0.10/point). Against
measured seconds-scale volatility (~3 points/sqrt(s) on a quiet Saturday,
2-3x that in busy sessions), the *typical* absolute move first matches that
hurdle somewhere between 5 and 30 minutes, and runs 3-4x costs at 4-8 h:

| window | 30s | 5m | 15m | 30m | 1h | 4h | 8h |
|---|---|---|---|---|---|---|---|
| typical move / cost | 0.14x | 0.45x | 0.79x | 1.1x | 1.6x | 3.1x | 4.4x |

Sub-30 s momentum can never pay for its own trade on this instrument, so the
strip does not spend electrodes on it; the short end (30 s - 1 m) is entry
*timing* context for positions whose thesis lives at 15 min - 8 h. This is
also why episodes are 15 minutes: an episode must be able to hold a trade at
the horizon where being right clears costs.

### Per-window rate scales and pre-roll

Momentum is a velocity, and for a diffusive price the move over a window
grows like sqrt(window) — so velocity shrinks like 1/sqrt(window). A single
scale would saturate the short windows and pin the long ones near zero, so
each window carries its own scale under that law (`momentum_scale` calibrated
at `momentum_scale_window_s`), and every strip channel spans the same
4-40 Hz range. A sustained drift therefore reads *hardest* at the long end,
which is correct: holding 1 point/s for an hour is a far bigger anomaly than
holding it for 30 s.

A window cannot speak before it has that much history, and reports exactly
zero (silence) until then. Nobody rents 8 hours of wetware warm-up, so the
session runner asks the market source to *pre-roll* history covering the
longest window before the first episode: the replay source resamples the
recording before `start_offset_s` (set the offset at least 8 h into the file
for a fully warm strip), the live source backfills 1-minute DXLink candles,
and the synthetic source generates the path. `session.warmup_s` (default
120 s) remains as pre-session rest for baseline collection, and rests keep
feeding the feature tracker without stimulating, so history stays unbroken
across them. Note the externalization: the culture needs no 8-hour memory —
the tracker holds the history and the strip delivers its summary as a
present-tense spatial pattern. One honest caveat: the long-end channels are
quasi-static for tens of minutes at a time, and cultures habituate to
unvarying stimulation, so the 4-8 h channels act as slow context bias rather
than dynamic drive.

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

Resetting the feature window darkens the momentum strip — carrying momentum
across the inter-contract basis jump would inject a fictitious move — but
the live source immediately backfills the *new* contract's candle history in
the background and re-seeds the strip at the next step boundary, so the long
timescales come back within seconds rather than hours. Rolls are monthly and
checked hourly, so a given session rarely sees one.

## Curriculum

Synthetic regimes in increasing difficulty: `sine`, `trend`, `trend_noise`,
`mean_revert`, `random_walk` (unlearnable control), then `replay` of real
/MBT days, then `live`. The `market.synthetic.snr` dial scales signal vs.
noise. A culture must beat its own random baseline on predictable regimes
before graduating.

The silicon baseline (`train_baseline.py`) is the gate in front of all of it:
if no conventional learner can extract edge from the encoded features at game
cadence after real costs, the game design needs work before spending wetware
rental time. Label horizons default to holding-period scale (1 m - 1 h),
matching where /MBT moves clear costs. It reports directional accuracy per
horizon against the majority-class benchmark (with overlap-adjusted
confidence intervals) for both the current neural encoding (the chronotopic
strip, one column per window, computed by the same `FeatureTracker` the live
encoder uses) and an extended candidate-channel set — returns over windows
from 1 s to 8 h, vol-normalized variants, volatility regime, spread, book
imbalance, range position, time of day — and backtests the resulting policies
through the real game engine and paper broker against time-shifted luck
baselines. Extended return columns use the same per-window 1/sqrt(w) scale as
the strip, so a candidate column that shows signal can be promoted onto the
strip without recalibration. The hour-scale windows and label horizons only
mean anything on multi-day contiguous recordings, which is what the
`--forever` library builder accumulates — the 4 h and 8 h strip windows ship
*unvalidated* until those recordings exist, and the per-window gate results
are how the ladder gets pruned or extended on evidence.

On the synthetic sine sanity check (re-parameterized to a 30-minute period so
the signal lives at holding-period scale), the strip reads SIGNAL on every
probe at the 60 s horizon with positive after-cost PnL (+$121 to +$459)
against deeply negative luck baselines — while the old single-window encoding
is completely blind at these horizons: its calibrated probabilities never
leave the hold band, zero trades. The random-walk control still reads no
signal on every probe. A subtle earlier result is worth remembering: with the
sine's original 120 s period, probes hit 93% accuracy at the 60 s horizon yet
lost money on every accurate horizon (predicting direction at exactly half
the cycle period means entering at the zero-capture phase) — the
accurate-but-untradeable trap the backtest layer exists to catch.

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
