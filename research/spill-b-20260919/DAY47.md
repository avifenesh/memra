# WP-B day 47: O7, DAY24's step-OOM and client-disconnect fault arms in `tools/health-fault-gate.sh`

OWED.md O7. DAY24 section 5 pre-registered two fault arms for the health gate and DAY25 left them unbuilt: (c) an OOM
at a session step parks and requeues, no 5xx to peers, peers' streams complete; (d) a client disconnect mid-stream
retires the session within one tick, peers unaffected. Both doors exist (`MEMRA_STEP_OOM_FAULT`, and a client
closing its socket). This day adds them to the gate as arms `g` and `h`, each with a red twin. No server code changes.

## 1. Pre-registration

Committed and pushed before any day-47 code and before any day-47 run. Nothing in section 1 changes after a number is
seen; a failed clause is recorded as it reads and a revision is a new, dated addendum pushed before its code.

### 1.1 Arm g: a step OOM parks, requeues and completes; peers complete

- Boot with `MEMRA_STEP_OOM_FAULT=1` (the next session step reports the door's synthetic CUDA OOM; the park branch,
  the teardown fence, `park_requeue` and the retry budget are production logic). Three concurrent streamed chat
  requests (`max_tokens=48`, greedy, distinct prompts).
- Green: exactly one `[admit-oom] step OOM parked session back to queue` line; all three requests end `200` with a
  `finish_reason` and `[DONE]`; no 5xx; no `panicked` or `[worker] PANIC` line; `/health` 200 after.
- **Red twin g-red:** `MEMRA_STEP_OOM_FAULT=4` (one more than the default `MEMRA_STEP_OOM_RETRIES` of 3), so the same
  session walks into the bounded-retry honest error. The green assertion ("all three complete") must read FAIL there,
  with the faulted request's error quoted and the two peers still `200`; the line reads `RED as expected` when it
  does, and `-> FAIL` (the twin has no teeth) when it does not.

### 1.2 Arm h: a client disconnect retires the session; the peer completes

- One boot, two concurrent streamed chat requests (`max_tokens=256`); the gate closes the first client's connection
  after its 8th frame and keeps the second.
- Green: an `[abort] client disconnected:` line for the closed request; the time from the client's close to that line
  at most 1,000 ms (the worker checks the channel at every tick; a tick here is well under that; the bound is stated,
  not fitted); the peer ends `200` with `finish_reason` and `[DONE]`; after both, `/metrics` reads no active session
  and `/health` 200.
- **Red twin h-red:** the same boot shape with the first client NOT closed. The green assertion (an abort line for
  that request) must read FAIL there; the line reads `RED as expected` when it does.
- **Out of the engine:** DAY24's `client_disconnected` ledger row belongs to the metering implementation, which the
  stock binary does not carry (`MeteringFactory` returns `Ok(None)`); the row's assertion is darklanes' and is listed
  for the lead to route.

### 1.3 Runs

Each arm and twin boots its own server, like the gate's other arms (`HFG_ARMS=g,h` runs them; the default arm list
gains `g,h`). The 5090 (the 9B) and the target card (the 27B, `HFG_READY_WAIT_S` raised for its load), under the
card's lock, twice each (two runs of the same verdicts, a stability reading).

### 1.4 CPU before the cards

`bash -n` and shellcheck on the gate; a dry run of the new verdict parsing against fixture logs.

### 1.5 Price

About 0.3 agent-day of gate code; about 15 min per run on the 5090, about 30 min on the target card.

## 2. Results

Written after the runs. Section 1 is unchanged.
