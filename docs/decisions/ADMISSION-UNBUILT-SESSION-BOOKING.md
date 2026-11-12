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
  left out and the door's pending-prime term books it: one prime call's slab and returned rows,
  and the whole-prompt stack of every session whose walker has not started. A session that owns
  no route state counts as such a session (`prime_walker_unstarted`); before this change that
  term counted only an MTP session before its walker starts, so a cold DFlash or GLM session's
  stack was booked by neither term.
- **The settle wait.** An arrival the booked reading does not hold, while some charge is still
  owed and the reading without that booking would hold it, is requeued for the next pass, before
  the reclaim, the eager arm, the drafter refusal and the capacity reject (`settle_wait`). Each
  of those would decide on a reading the tick is about to change: a reclaim would evict prefix
  entries and parked sessions the real reading may not need, and a drafter refusal holds for the
  request's life. A settle wait lasts until the owing sessions are built, one tick: every spec
  session steps each tick and its first step builds the state. An arrival the reading does not
  hold even without the booking gains nothing from a wait (the tick can only lower the reading,
  a retire aside), so it goes on to those verdicts at once, decided on the booked reading so
  none of them spends the bytes an unbuilt session will take.
- **The settle wait holds the line.** Like a VRAM defer, an interactive settle wait arms the
  head-of-line memory hold for its devices (`C8-SERVING-24GB.md`): later requests on those
  devices wait behind it, so a smaller arrival neither spends the memory it waits for nor, as
  one more cold prime, becomes another unbuilt booking ahead of it. The hold set is made anew at
  the top of every admission pass, so it lasts the pass the request waited in. Behind it a
  dark-lane request is shed with a 429 for that pass, exactly as behind a VRAM defer. An arrival
  that would not fit even without the booking takes no settle wait and so arms no settle hold;
  its VRAM defer arms the hold only when it can fit once its device's sessions retire, as before.
  The hold's cost: a later arrival on those devices waits at most as long as the settle wait,
  until the unbuilt sessions are built, even one that would fit. No cell measured here reaches it
  (in the gate and at agentic c = 2 nothing queues behind the waiting request), so its cost is
  unmeasured.
- **What it prints and counts.** `[admission] settle wait:` once per pass;
  `[admission] settled:` when `admit` has built the session, with the passes the request waited
  and the milliseconds those waits took. The settle time ends at the pass that lets the request
  past the settle check, before any VRAM defer that follows. `/metrics`:
  `admission_settle_waits` (once per pass a request waits) and `admission_settle_wait_ms`.
- **Routes that build their state in `admit` are unchanged.** Their sessions own their state when
  the next arrival is read, so nothing is booked for them.
- **A partial reclaim's booking counts it twice, toward safety.** Door off, after a partial
  reclaim, the reclaim booking books what the priming sessions owe through the same pending-prime
  term, which now includes an unbuilt session's stack and call, while the unbuilt booking books
  that session's whole charge, its workspace included. Until the tick builds the session the
  workspace is counted twice; the error can only add a settle pass, and it ends when the session
  is built.

## Rejected

- **Turning on the memory door (memra#680) on this card.** Its pending-prime term books the
  prime workspace, checkpoint snapshots and pending seeds of sessions that are still priming
  (`pending_prime_bytes_v2`), not the KV and drafter plane of a session that has none yet, so it
  does not close this gap. The door also changes the open-output charge, the host-tier demotion
  and the bounded defer that ends in a 429; the RTX 5090 class stays out of it until its own cell
  reads (`ADMIT-BY-MEMORY-DEFAULT.md`).
- **One cold DFlash admission per pass.** It would serialize admissions that fit together, and
  it misses the step-OOM replay, which re-admits beside a session admitted in an earlier pass.
- **Deciding at once on the booked reading for an arrival the booking alone keeps out**
  (reclaim, eager arm, drafter refusal). The booked reading is a bound, not a measurement; the
  real one arrives a tick later. Refusing a request its drafter for life, or evicting the prefix
  cache, on that bound trades a one-tick wait for a lasting cost.
- **Booking the charge less what `admit` was seen to allocate.** A session that owns no state
  allocated nothing in `admit`; the predicate is exact where the subtraction would be noise.
- **Building the route state in `admit`, in this change** (memra#1263 proposes it). Calling
  `dspark_prime_start` in the admission pass would put the target caches and drafter plane in
  the live reading at once, and most settle waits would not happen. The wait's cost is real: the
  one settle wait in the agentic replay lasted 3,263.7 ms, one tick, because the tick that built
  the owing session also ran its first 4,096-row prime chunk (`[prime-yield] ... wall_ms=3258.586`,
  the next line). It is not a small change: the admission pass would need the drafter that the
  tick borrows mutably, the capture decision and the TTFT prime mark; the yield-off rollback
  (`dspark_spec_session_new`), gemma-spec and GLM build and prime in one call and would need a
  split; and an allocation failure in the pass would need the park and requeue a failure in the
  tick takes today. It would not remove the booking either: the walker's whole-prompt stack and
  call workspace are still allocated chunk by chunk in the tick, so that part of the charge stays
  owed until the walker starts.

## Not covered

- **Multi-device deployments.** The booking is a scalar on the primary device, applied only
  when `is_multi_device_deployment` is false. That excludes PP and TP, whose charge is planned
  per stage, and also any deployment whose loaded models span more than one device, such as two
  single-card models on a two-card box. A multi-device route that builds its state in the tick
  (GLM PP) keeps the old reading until a per-device booking exists.
- **A lone cold prime that overruns the reserve** (memra#1258): one cold DFlash admission, with
  its charge and the reserve met and no peer admitted in its pass (`active=0`, earlier sessions
  parked), ran out of memory in its first prime chunk. It was the first request of the pair in
  the first cut of this change's gate cell, on a diagnostic build of main
  (`research/same-pass-admission-20261112/diag/gate-g1/`). That is a different gap (the chunk's
  driver draw against pool-cached bytes and the calibrated floor), measured and fixed in its own
  change.

## Measured

Receipts: `research/same-pass-admission-20261112/` (its README names every binary). "Base" is
`445f964ea7`, memra#1248's head; "the head" is this PR's final code on it (`fix3`), and "the
first cut" is its code before the review (`fix`: no head-of-line hold on a settle wait, a settle
wait also for an arrival that did not fit even without the booking, the settle time measured to
admission, no unbuilt stack in the pending-prime term). On this card only the last is reached in
these runs, through the door-off reclaim booking after a partial reclaim in the agentic replay. The PR head is the same code rebased onto `11b368f7d7`; the landings between change
`crates/memra-engine/src/spec.rs` and `unified_memory.rs` only (the MTP pre-capture guard, its
narrow trim on unified memory only), which none of these runs reaches, so the device runs were
not repeated; the CPU checks were. Every gate run is a single run.

### Same-pass gate (RTX 5090)

`tools/same-pass-admission-gate.py`: two cold prompts (5,772 and 5,741 prompt tokens, charges
816 and 814 MB) fired together beside a decoding keeper. The gate applied a reserve of 1,653 MB
(the boot floor of 1,576 MB plus the verify-graph pool's debt, read from a probe that defers),
so one request fits at 2,469 MB of effective free and two at 3,283 MB; a ballast placed the
reading at 2,874 MB.

| run | exit | failed verdicts | settle waits | out-of-memory lines | the keeper |
|---|---|---|---|---|---|
| base | 1 | V2 (both admitted in one pass), V3 (step OOM) | 0 | 2 | not parked after 4,896 streamed tokens: 503 |
| the head | 0 | none | 1 | 0 | 12,000 tokens, 200 |

- **Base.** The two `[admit-trim]` lines of the pass read the same `driver_free_after`
  (2,137,063,424 bytes) and `still_owned` (20,536,576,036): the second request was charged on
  the reading the first was admitted on. The first prime chunk ran out of memory with 22 MB of
  driver free. One pair session parked and replayed, and both pair requests answered 200 with
  the reference text; the keeper, mid-stream, could not be parked.
- **The head.** The second request read `effective free 2059MB after 815MB booked ... < cost
  814MB + reserve 1653MB` and waited one pass, 3,540.3 ms: the tick that built the first session
  and ran its first prime chunk. The real reading then still did not hold it, so it took the
  ordinary VRAM defer until the first finished. Both answered 200 with the reference greedy text.
- **The first cut**, against base, before the gate probed the reserve (band from the floor,
  2,392 to 3,206 MB, placement 2,799 MB, inside the applied band too): the same verdicts, base
  exit 1 with 2 out-of-memory lines and the keeper's 503, the first cut exit 0 with 1 settle wait.

### Agentic replay at c = 2

The 29-turn agentic workload (8 sessions, tool calls, thinking on, T = 1.0, seed 1234),
two closed-loop users, a fresh boot per row, two boots per lease in the order B F, F B, B F, F B
on the RTX 5090 (the first two pairs on the first cut, the last two on the head) and one boot
per lease B F F B on GB10.

| card | boot | turns ok | tokens | agg gen tok/s | TTFT p50 / p95 s | settle waits, ms | step OOM | texts as boot 1 |
|---|---|---|---|---|---|---|---|---|
| RTX 5090 | base | 29/29 | 10,818 | 35.51 | 3.91 / 16.48 | 0 | 0 | 29/29 |
| RTX 5090 | first cut | 29/29 | 10,818 | 36.11 | 4.00 / 15.52 | 1, 3,264 | 0 | 29/29 |
| RTX 5090 | first cut | 29/29 | 10,818 | 37.02 | 3.61 / 15.56 | 0 | 0 | 29/29 |
| RTX 5090 | base | 29/29 | 9,765 | 35.86 | 3.29 / 15.05 | 0 | 1 | 27/29 |
| RTX 5090 | base | 29/29 | 10,262 | 37.08 | 3.13 / 14.77 | 0 | 2 | 27/29 |
| RTX 5090 | head | 29/29 | 10,818 | 37.94 | 3.84 / 15.10 | 1, 3,190 | 0 | 29/29 |
| RTX 5090 | head | 29/29 | 10,818 | 35.37 | 3.25 / 15.53 | 1, 2,182 | 0 | 29/29 |
| RTX 5090 | base | 29/29 | 9,765 | 36.45 | 3.23 / 15.48 | 0 | 1 | 27/29 |
| GB10 | base | 29/29 | 11,125 | 28.03 | 3.58 / 11.71 | 0 | 0 | 29/29 |
| GB10 | the first cut | 29/29 | 11,125 | 28.01 | 3.57 / 11.73 | 0 | 0 | 29/29 |
| GB10 | the first cut | 29/29 | 11,125 | 28.03 | 3.57 / 11.72 | 0 | 0 | 29/29 |
| GB10 | base | 29/29 | 11,125 | 27.95 | 3.58 / 11.73 | 0 | 0 | 29/29 |

- **Base ran out of memory in three of four boots on the RTX 5090**, four step OOMs in all, each
  in the gate's shape: two `[admit-trim]` lines of one pass read the same `driver_free_after`
  (2,136,342,528) and `still_owned` (21,736,474,480), and the first prime chunk after them ran out
  of memory. The parked sessions replayed and every turn was answered, but each of those boots
  wrote 2 of 29 turns with another text (9,765 or 10,262 tokens against 10,818). The change ran
  out of memory in none of its four boots and wrote the same 29 texts in each.
- **The settle wait's cost.** Three of the change's four boots took one settle wait each:
  3,264, 3,190 and 2,182 ms, each the tick that built the owing session and ran its first prime
  chunk. Throughput is level (base 35.51, 35.86, 37.08, 36.45; the change 36.11, 37.02, 37.94,
  35.37 tok/s), and so is TTFT p95. TTFT p50 reads 3.13 to 3.91 s on base and 3.25 to 4.00 s on
  the change; the base boots that ran out of memory wrote other, shorter texts, so the two are
  not the same work.
- **GB10** has the memory to admit both, so neither binary waited or deferred there; the four
  boots wrote the same 29 texts and the same throughput within 0.3%.

### Normal-concurrency cells

DFlash2 drafter, closed-loop clients at c = 1, 2, 4, 8, 6 requests per client, greedy chat,
160 new tokens, streamed. No request in any boot took a settle wait or ran out of memory, and
every boot wrote the same text per prompt at every c.

| card | c | base tok/s | change tok/s |
|---|---|---|---|
| RTX 5090 (B F F B, the head) | 1 | 73.35, 72.11 | 72.19, 71.93 |
| RTX 5090 | 2 | 81.46, 78.32 | 78.24, 77.05 |
| RTX 5090 | 4 | 121.35, 120.94 | 121.09, 121.61 |
| RTX 5090 | 8 | 146.50, 146.27 | 146.70, 146.81 |
| GB10 (B F, the first cut) | 1 | 27.16 | 27.03 |
| GB10 | 2 | 32.76 | 33.20 |
| GB10 | 4 | 46.67 | 46.59 |
| GB10 | 8 | 79.13 | 81.00 |

On the RTX 5090 the first base boot is the high one at c1 and c2 and the second sits with the
head's two, the shape of drift across the lease; the change took no settle wait in these cells,
so its only work there is the booking's sum over the active sessions each pass.

### CPU

Each half of the head reverted by one line fails a test: the booking that books nothing or the
settle wait that never fires fails
`an_admitted_session_with_no_route_state_books_its_charge_until_the_tick_builds_it` (first cut,
`cpu/`); the settle wait without the unbooked fit fails it too, the pending-prime term without
the unbuilt sessions' stacks fails `the_armed_door_books_the_whole_prompt_stack_of_an_unbuilt_session`,
and the settle wait without its hold fails
`every_route_state_field_counts_as_built_and_the_settle_wait_precedes_every_verdict`
(`cpu-head/`). The head's tree passes all three, the full `memra-server` suite (1,218 passed),
clippy with `-D warnings`, fmt, the flags census and the numeric-scope check, on its base and
again rebased onto `11b368f7d7`.
