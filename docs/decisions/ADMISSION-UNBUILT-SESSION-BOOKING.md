# An admitted session that owns no route state yet keeps its charge booked (2026-11-12)

Date: 2026-11-12. Issue memra#1255, branch `fix/1255-same-pass-admission`, receipts
`research/same-pass-admission-20261112/`. Cards: one RTX 5090 Laptop GPU (24 GB; the enforced
power limit read 150 W idle and 160 W for most of each timed run, recorded per run and not
changed) and one DGX Spark (GB10). Qwen3.8-27B NVFP4 + Q5_K with the DFlash2 drafter.

## Problem

The VRAM gate admits a request when the device's effective free memory (driver free plus the
pool's cached bytes) covers its charge plus the transient reserve. The reading is live: it sees
what admitted sessions have allocated, and nothing else.

`admit` builds the plain cache, the MTP spec session and every restored or resumed session, so
their charge shows in the next reading. A DFlash, gemma-spec or GLM cold prime builds its target
caches and drafter plane in the tick that follows (`dspark_prime_start`,
`dspark_spec_session_new`, `gemma_spec_session_new`, `glm5_prime_start`). Until then it has
allocated nothing. A second arrival read in the same admission pass, or in the pass that replays
a step-OOM park, was priced against a reading that still held the first session's whole charge,
and both were admitted on the same bytes.

On the agentic replay at c = 2 (memra#1255, `research/drafter-admission-20261112/agentic-c2/`),
two of four boots admitted two cold DFlash requests in one pass. The two admission trims read
the same driver free and pool use (2,118,254,592 and 21,479,402,552 bytes in `main-133251`), so the
second request was charged against 3,020 MB of effective free while the first one's 770 MB was
still owed. From the trim to the failure, effective free fell 1,856 MB, about the two charges
(1,899 MB). The transient reserve was spent and the step ran out of memory with 36 MB of driver
free. A diagnostic build (logging only) showed it directly: every cold DFlash admission left the
session with no cache and no drafter session, and its prime start then took 518 to 1,017 MiB.

## Chosen

- **The booking.** Each admitted session records the charge the gate admitted it on
  (`Session::admitted_charge_bytes`, and its prefill-workspace part). While the session owns no
  route state (`session_owns_route_state`: no plain cache, MTP spec session or prime, gemma-spec
  session, DFlash session or prime, GLM session or prime), that charge is added to the bookings
  every reading subtracts (`unbuilt_session_booking`, through `pending_booked` and
  `AdmissionHeadroom::less_pending`). The tick builds the state, and the next reading shows what
  it took. On the armed memory door (the RTX PRO 6000 Workstation Edition) the workspace part is
  left out, since the door's pending-prime term books it already.
- **The settle wait.** An arrival that does not fit the booked reading while some charge is
  still owed is requeued for the next pass, before the reclaim, the eager arm, the drafter
  refusal and the capacity reject (`settle_wait`). Each of those would decide on a reading the
  tick is about to change: a reclaim would evict prefix entries and parked sessions the real
  reading may not need, and a drafter refusal holds for the request's life. A settle wait lasts
  until the owing session's prime starts, normally one tick. `[admission] settle wait:` prints
  once per pass; `[admission] settled:` names the passes and milliseconds a request waited when
  it is admitted; `/metrics` gains `admission_settle_waits` and `admission_settle_wait_ms`.
- **Routes that build their state in `admit` are unchanged.** Their sessions own their state when
  the next arrival is read, so nothing is booked for them.

## Rejected

- **Turning on the memory door (memra#680) on this card.** Its pending-prime term books the
  prime workspace, checkpoint snapshots and pending seeds of sessions that are still priming
  (`pending_prime_bytes_v2`), not the KV and drafter plane of a session that has none yet, so it
  does not close this gap. The door also changes the open-output charge, the host-tier demotion
  and the bounded defer that ends in a 429; the RTX 5090 class stays out of it until its own cell
  reads (`ADMIT-BY-MEMORY-DEFAULT.md`).
- **One cold DFlash admission per pass.** It would serialize admissions that fit together, and
  it misses the step-OOM replay, which re-admits beside a session admitted in an earlier pass.
- **Deciding on the booked reading at once** (reclaim, eager arm, drafter refusal). The booked
  reading is a bound, not a measurement; the real one arrives a tick later. Refusing a request
  its drafter for life, or evicting the prefix cache, on that bound trades a one-tick wait for a
  lasting cost.
- **Booking the charge less what `admit` was seen to allocate.** A session that owns no state
  allocated nothing in `admit`; the predicate is exact where the subtraction would be noise.

## Not covered

- **PP and TP deployments.** Their charge is planned per stage; the scalar booking is applied on
  single-device deployments only (`is_multi_device_deployment`). A multi-device route that builds
  its state in the tick (GLM PP) keeps the old reading until a per-stage booking exists.
- **A lone cold prime that overruns the reserve** (memra#1258): one cold DFlash admission, with
  its charge and the reserve met and no peer admitted in its pass (`active=0`, earlier sessions
  parked), ran out of memory in its first prime chunk. It was the first request of the pair in
  the first cut of this change's gate cell, on a diagnostic build of main
  (`research/same-pass-admission-20261112/diag/gate-g1/`). That is a different gap (the chunk's
  driver draw against pool-cached bytes and the calibrated floor), measured and fixed in its own
  change.

## Measured

Receipts: `research/same-pass-admission-20261112/` (its README names every binary). "Base" is
`445f964ea7`, memra#1248's head, whose landing `9d7c086c67` changes no code after it; "the
change" is this PR's code on it. Every gate run is a single run.

### Same-pass gate (RTX 5090)

`tools/same-pass-admission-gate.py`: two cold prompts (5,772 and 5,741 prompt tokens, charges
816 and 814 MB, transient floor 1,576 MB) fired together beside a decoding keeper, a ballast
placing effective free at 2,799 MB, where one fits (2,392 MB) and two do not (3,206 MB).

| run | exit | failed verdicts | settle waits | out-of-memory lines | the keeper |
|---|---|---|---|---|---|
| base | 1 | V2 (both admitted in one pass), V3 (step OOM) | 0 | 2 | not parked after 3,872 streamed tokens: 503 |
| the change | 0 | none | 1 | 0 | 8,000 tokens, 200 |

- **Base.** The two `[admit-trim]` lines of the pass read the same `driver_free_after`
  (2,137,063,424 bytes) and `still_owned` (20,410,368,036): the second request was charged on
  the reading the first was admitted on. The first prime chunk ran out of memory with 22 MB of
  driver free. One pair session parked and replayed, and both pair requests answered 200 with
  the reference text; the keeper, mid-stream, could not be parked.
- **The change.** The second request read `effective free 1985MB after 814MB booked ... < cost
  815MB + reserve 1653MB` and waited one pass. After the first session's prime started the real
  reading (1,981 to 2,112 MB) still did not hold it, so it took the ordinary VRAM defer until
  the first finished, and was admitted 7,637.6 ms after its settle wait. Both answered 200 with
  the reference greedy text.

### Agentic replay at c = 2

The 29-turn agentic workload (8 sessions, tool calls, thinking on, T = 1.0, seed 1234),
two closed-loop users, a fresh boot per row.

| card | boot | turns ok | tokens | agg gen tok/s | TTFT p50 / p95 s | settle waits | settled wait ms | step OOM |
|---|---|---|---|---|---|---|---|---|
| RTX 5090 | base | 29/29 | 10,818 | 35.51 | 3.91 / 16.48 | 0 | 0 | 0 |
| RTX 5090 | change | 29/29 | 10,818 | 36.11 | 4.00 / 15.52 | 1 | 3,264 | 0 |
| RTX 5090 | change | 29/29 | 10,818 | 37.02 | 3.61 / 15.56 | 0 | 0 | 0 |
| RTX 5090 | base | 29/29 | 9,765 | 35.86 | 3.29 / 15.05 | 0 | 0 | 1 |
| GB10 | base | 29/29 | 11,125 | 28.03 | 3.58 / 11.71 | 0 | 0 | 0 |
| GB10 | change | 29/29 | 11,125 | 28.01 | 3.57 / 11.73 | 0 | 0 | 0 |
| GB10 | change | 29/29 | 11,125 | 28.03 | 3.57 / 11.72 | 0 | 0 | 0 |
| GB10 | base | 29/29 | 11,125 | 27.95 | 3.58 / 11.73 | 0 | 0 | 0 |

- **The out-of-memory on base** is the same shape as the gate's: in the second base boot two
  `[admit-trim]` lines of one pass read the same `driver_free_after` (2,136,342,528) and
  `still_owned` (21,736,474,480), and the first prime chunk ran out of memory. The parked
  session replayed and the boot answered every turn, but wrote 9,765 tokens and 2 of 29 turns
  with another text; every other boot on the card wrote the same 29 texts.
- **The settle wait's cost** in the replay: one wait in four change boots, 3,264 ms from the
  wait to the admission (that span includes the first session's prime chunk the arrival waited
  behind). Throughput and TTFT on the change are within the spread of base.
- **GB10** has the memory to admit both, so neither binary waited or deferred there; the four
  boots wrote the same 29 texts and the same throughput within 0.3%.

### Normal-concurrency cells

DFlash2 drafter, closed-loop clients at c = 1, 2, 4, 8, 6 requests per client, greedy chat,
160 new tokens, streamed. On GB10 (one boot each, base then the change) every c wrote the same
text per prompt in both boots, with no settle wait, defer or out-of-memory line; tok/s base |
change: c1 27.16 | 27.03, c2 32.76 | 33.20, c4 46.67 | 46.59, c8 79.13 | 81.00.

### CPU

Reverting either half of the change by one line (the booking books nothing; the settle wait
never fires) fails `an_admitted_session_with_no_route_state_books_its_charge_until_the_tick_builds_it`.
The change's tree passes both worker tests, the full `memra-server` suite, clippy with
`-D warnings`, fmt, the flags census and the numeric-scope check.
