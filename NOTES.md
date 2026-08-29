# tradeai-cortical — Research Notes

Goal: train a living neural culture (Cortical Labs CL1) to trade a single CME futures
contract by embodying it in a market "game world", DishBrain/Pong style. Paper trade
first, go live only after demonstrated edge.

---

## 1. Cortical Labs platform (as of mid-2026)

### Access options
- **CL1 hardware**: US$35k per unit. Overkill for now.
- **Cortical Cloud** (wetware-as-a-service): ~US$300/week (~$1,700–2,200/month) per
  CL1 instance. Sign up at https://cloud.corticallabs.com. Public since ~March 2026.
  Melbourne biological data centre houses 120 units. Browser-based (Jupyter),
  Python SDK, no lab required.
- **CL SDK Simulator**: free, `pip install cl-sdk`. 1:1 API parity with the real
  device. Generates random (Poisson) or replayed spike data. **It does not learn or
  respond to stimulation** — useful for building/debugging the whole pipeline and as
  a random-baseline control, useless for validating that the neurons learn.

### CL API essentials (Python `cl` module)
- `with cl.open() as neurons:` — main entry point.
- `neurons.loop(ticks_per_second=N)` — closed-loop driver, up to 25 kHz. Each tick
  exposes `tick.analysis.spikes` (list of `Spike(timestamp, channel)`) and
  `tick.analysis.stims`. Hard real-time: exceeding tick budget raises `TimeoutError`
  (use `recover_from_jitter()` or lower tick rates).
- Stimulation: `neurons.stim(ChannelSet, StimDesign, BurstDesign)`.
  - `StimDesign(width_us, current_uA, ...)` mono/bi/tri-phasic pulses, widths in
    multiples of 20 µs, negative leading edge recommended.
  - `BurstDesign(count, hz)` for repeated stims.
  - **Hard limit: max 200 Hz stim per channel** (cell protection).
- `neurons.create_data_stream(name, attributes)` — log arbitrary game state
  (e.g. price, position, PnL) time-aligned with spikes into the HDF5 recording.
- `neurons.record()` → HDF5 with raw samples, spikes, stims, data streams.
  `RecordingView` to load/analyze offline. Recordings can be replayed in the sim.
- Docs: https://docs.corticallabs.com/ · CL API paper: arXiv:2602.11632

## 2. DishBrain / Pong training paradigm (the template)

From Kagan et al. 2022 (*Neuron*), "In vitro neurons learn and exhibit sentience
when embodied in a simulated game-world":

- ~800k–1M neurons on an HD-MEA (1,024 routed electrodes).
- **Sensory encoding (input)**: 8 stimulation electrodes in a defined sensory area.
  - *Place coding*: which electrode fires ↔ ball's y-position (topographic).
  - *Rate coding*: stim frequency 4→40 Hz ↔ ball distance from paddle.
  - 75 mV sensory stim level.
- **Motor decoding (output)**: spike counts binned every **10 ms** in two predefined
  motor regions. Region 1 activity → paddle up, region 2 → paddle down; the more
  active region wins. Game world also updates every 10 ms.
- **Feedback (the learning signal — Free Energy Principle)**:
  - *Hit (good)*: **predictable** stim — 75 mV, 100 Hz for 100 ms across all
    sensory electrodes simultaneously.
  - *Miss (bad)*: **unpredictable** stim — 150 mV, 5 Hz, random sites, random
    timing, for 4 s, then 4 s of silence, then game restarts on a random vector.
  - Theory: neurons self-organize to make their sensory world predictable, so they
    learn behaviors that avoid the noise. Learning was measurable within ~5 minutes;
    effect sizes real but modest (longer rallies vs. controls).
- Control conditions that mattered: silent feedback and no-feedback conditions did
  NOT learn — the structured predictable/unpredictable feedback is the active
  ingredient. In-silico random controls also didn't learn. We should replicate this
  structure: our cl-sdk simulator run *is* the random control baseline.
- Session length constraints existed (DishBrain-era cells disliked >1.5 h sessions).
  CL1 life support keeps cultures alive up to 6 months, but training will still be
  discrete sessions, not 24/7.

## 3. Tastytrade side

### Market data (production) — good
- DXLink websocket (dxFeed): get quote token from `/api-quote-tokens`, connect to
  `wss://tasty-openapi-ws.dxfeed.com/realtime`, open FEED channel, subscribe.
- Event types: `Quote` (bid/ask + sizes), `Trade` (prints), `Candle`
  (e.g. `/MNQZ26:XCME{=1s}` or `{=5m}`, historical backfill via `fromTime`).
- Futures streamer symbols: `/MNQZ26:XCME` style (product + month code + `:XCME`).
- Real-time CME data comes with a funded tastytrade account (needs futures approval,
  "The Works" trading level for futures).
- Auth: OAuth2 w/ refresh tokens recommended (session tokens expire in 15 min).
  Careful: repeated failed logins → ~8 h IP block.

### Sandbox (cert environment) — NOT usable for training
- `api.cert.tastyworks.com`: quotes are **always 15-min delayed**, environment
  **resets every 24 h**, limited symbol support, no control over fills.
- Verdict: fine for one-time order-payload validation, useless as the paper-trading
  environment for a system that learns from live feedback.

### Paper trading approach → build our own
- **Internal sim broker**: live production market data + our own fill engine
  (market orders fill at current bid/ask ± slippage model, track position & PnL).
  Same interface as the live broker adapter so cutover is a config switch.
- This also gives us instant, deterministic fills — important because the reward
  stim must follow the action promptly for the neurons to associate them.

## 4. Latency reality check ("HFT" framing)

True HFT (µs–ms) is not achievable here and shouldn't be the goal:
- Retail websocket data: ~50–300 ms behind the matching engine.
- Cortical Cloud instances live in Melbourne — market data must travel there.
- Tastytrade order round-trip: ~100+ ms, plus no colocation.

What IS fast: the neurons' internal decision loop (10 ms bins, like Pong). So the
realistic shape is **high-frequency decisions, human-scale execution**: game ticks
every 10–100 ms, actions/trades on the seconds-to-minutes scale. The "video game"
framing still holds completely.

Architecture implication: run the entire game loop (data client + game engine +
sim broker + encoder/decoder) ON the Cortical Cloud instance next to the neurons.
Cortical Cloud is advertised as network-enabled for real-time external data.
(Verify outbound websocket access once we have an account — fallback would be a
relay we host, or replay-based training.)

## 5. Draft game design (to discuss)

### The game: "hold the right position"
Instead of Pong's ball-and-paddle, the neurons manage one position in one
instrument: **short (-1), flat (0), long (+1)**, one contract.

- **Sensory input** (place + rate coding, mirroring Pong):
  - Price momentum/direction over a short window → which electrode group
    (place) + magnitude → stim rate (4–40 Hz analog).
  - Current position state (long/flat/short) on a dedicated electrode group —
    the neurons should "feel" their own position, like feeling the paddle.
  - Optionally: unrealized PnL direction as a third channel group.
- **Motor output**: two motor regions, spike counts per bin:
  - Region 1 wins → move position toward +1 (buy: open long / close short).
  - Region 2 wins → move position toward -1 (sell: open short / close long).
  - Below activity threshold / tie → hold. (This covers the user's 4 actions:
    open long, open short, close long, close short.)
- **Feedback**:
  - Favorable mark-to-market move / profitable close → predictable burst
    (100 Hz, 100 ms, all sensory electrodes).
  - Adverse move / losing close → unpredictable noise (5 Hz random-site, ~4 s).
  - Open question: reward on every MTM tick vs. on trade close vs. hybrid
    (small tick rewards + big close rewards). Pong rewarded discrete events
    (hit/miss); a per-trade reward is the closest analog.

### Honest expectations
Pong is a much easier task: the ball's physics are deterministic and the neurons'
actions directly cause the feedback. Markets are near-random and the neurons don't
influence the price. The experiment tests whether structured feedback can shape
the culture toward *any* persistent, better-than-random policy (e.g. momentum
following). Sharpe expectations should be near zero going in; treat it as research
with a rigorous baseline comparison (real neurons vs. cl-sdk random control vs.
shuffled-feedback control).

## 6. Phase sketch (to refine into the plan)

1. **Phase 0 — access**: Cortical Cloud signup (waitlist?), tastytrade OAuth app +
   futures data confirmation.
2. **Phase 1 — pipeline on simulator (free, local)**: DXLink client, game engine,
   encoder/decoder, sim broker, session recorder/metrics — all against `cl-sdk`.
   This is 90% of the code and costs nothing.
3. **Phase 2 — real neurons, paper trading**: deploy to Cortical Cloud, run
   training sessions, compare vs. random/control baselines. Iterate on encoding
   and reward design (this is where the real experimentation happens).
4. **Phase 3 — evaluation gate**: predefined metrics (win rate, PnL after costs,
   consistency across sessions/cultures) that must beat controls before any live
   trading.
5. **Phase 4 — live**: micro contract (/MES or /MNQ), max position 1, daily loss
   limit, kill switch, same code path with live broker adapter.

## 7. Literature review — what actually works (and what to steal)

### A. Embodied Neurocomputation (Cortical Labs, arXiv:2605.13315, 2026) — THE blueprint
First large-scale optimization of encoding parameters on CL1 hardware: ~1,300
configurations, 26 cultures, ~4,000 hours of closed-loop interaction. Task was a
gridworld agent navigating to "food" via a scalar odor gradient with 3 actions
(forward/left/right) — structurally almost identical to our game (scalar market
signals → 3 actions: buy/sell/hold).

**Optimal encoding parameters found (adopt as our defaults):**
- Rate encoding: min frequency 4 Hz, **max frequency 40–60 Hz** (strongest single
  performance driver; 80–100 Hz was worse)
- Amplitude: **2.5 µA** (higher was better within 1.0–2.5 range)
- Pulse width: **40–80 µs** (shorter was better; 160 µs worse)
- Pulse shape: symmetric biphasic, negative-polarity leading phase
- Tick rate 1–2 Hz, i.e. the agent acted roughly **once per second** — the proven
  interaction speed is ~1 Hz, not DishBrain's 10 ms. Our game tick should target
  ~1 action opportunity per second.
- Decoding: spike counts in one region per action, **normalized against baseline
  spontaneous activity from the 60 interactions preceding each episode** (adopt
  this baseline normalization — raw counts are biased by uneven culture activity).
- Feedback: reward > 0 → five structured 100 Hz bursts (80 ms) across BOTH
  encoding and decoding regions; reward ≤ 0 → random stimulation. Reward was
  graded (+2 goal, −0.2 collision, continuous shaping on odor change) →
  supports a hybrid reward: small continuous MTM shaping + big per-trade events.
- **Episodic structure beats continuous**: 5 × 30-step episodes with 2-minute
  rests significantly outperformed one 150-step run. Design trading sessions as
  short episodes with rest breaks, not marathon sessions.
- Optimized BNN agents beat sample-matched DQN by 1.18–1.25x and culture
  baselines by 3.6–5.6x. But **only a small subset of encoding regimes learned
  at all** — per-culture tuning is expected, and they used Optuna to search.

### B. Classic culture-learning literature — expectation setting
- Shahaf & Marom 2001 (J. Neurosci): first demonstration that cultures can be
  trained via closed-loop stimulus removal (learning = stimulus avoidance).
- Replications (le Feber et al. 2010, PLoS ONE; others): the effect is real *on
  average* but only ~50% of cultures learn, learning curves are erratic, and
  connectivity changes are partly uncontrolled. → Never conclude from one
  culture/session; build the metrics pipeline for repeated sessions and
  statistical comparison from day one.

### C. Biological reservoir computing — a second, lower-risk architecture
Instead of asking the neurons to *learn* the task (FEP feedback), treat them as a
fixed nonlinear dynamical reservoir and train a simple silicon readout (ridge/
logistic regression) on their spike responses:
- **Brainoware** (Cai et al., Nature Electronics 2023): brain organoid + trained
  readout predicted the Hénon chaotic map; regression score 0.36 → 0.81 over 4
  training epochs. Plasticity-blocked controls didn't improve.
- Sumi et al. (PNAS 2023): cultured BNNs act as "generalization filters" —
  reservoir + linear decoder classified spoken digits and generalized across
  datasets where a direct linear decoder failed. BNN short-term memory is
  several hundred ms.
- IJCNN 2024 (Lindell et al.): in vitro reservoirs + ridge regression predicted
  logistic-map chaotic series 4–6 steps ahead, ~30% MAE improvement over
  control — but only 2 of 6 networks worked (variability again).
- Chaos-controlled RC paper (arXiv 2604.02552): readout accuracy drifts over
  ~12 h as the culture's representation drifts → readouts need periodic
  retraining.

**Implication for us:** build the decoder with two modes sharing the same
encoder/game/broker:
1. **Agent mode** (primary experiment): DishBrain/gridworld-style motor-region
   counts + predictable/unpredictable feedback. Tests whether neurons *learn to
   trade*.
2. **Reservoir mode** (fallback with stronger literature support for time-series
   prediction): log full per-channel spike features every tick, train a ridge
   readout offline on recordings (predict next-interval direction), then run it
   online with periodic retraining. Even if FEP learning fails, this can still
   produce a functioning trading signal — and it reuses the same recordings.

## 8. Open questions

- Cortical Cloud: waitlist status, session scheduling, outbound network access,
  whether the same culture persists across our sessions (continuity of learning
  matters a lot for this project).
- Tastytrade account: futures approval level, CME data entitlement on API.
- Reward design: per-tick vs per-trade vs hybrid.
- Which instrument: /MNQ vs /MES (micro contracts; NQ moves more per unit time —
  richer signal, harsher noise).
- Session protocol: how long, how often, same culture vs fresh cultures.
