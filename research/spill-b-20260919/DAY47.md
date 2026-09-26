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

### 2.1 The 5090, the registered arms (`rtx5090-day47/r1`, `r2`; the 9B; 2026-09-26 14:53 and 15:03Z)

Run 1, verbatim (run 2 reads the same four verdicts; `h` there: `frames_at_close=11 ... close_to_abort_line_ms=96`):

```
HFG (g) step-oom-parks-and-completes: fault=1 parked_lines=0 completed=0/3 codes={r0:200,r1:200,r2:200} http_5xx=0 panic_lines=0 health_after=200 -> FAIL
HFG (g-red) step-oom-past-the-retry-budget: fault=4 parked_lines=0 completed=0/3 codes={r0:200,r1:200,r2:200} faulted_error={r0:"message":"the model is temporarily at capacity; retry after the Retry-After delay";r1:"message":"the model is temporarily at capacity; retry after the Retry-After delay";r2:"message":"the model is temporarily at capacity; retry after the Retry-After delay";} http_5xx=0 panic_lines=0 green_assertion_fired=true -> FAIL
HFG (h) client-disconnect-retires-within-1000ms: frames_at_close=15 abort_lines=1 close_to_abort_line_ms=97 peer_complete=true peer_finish=length active_sessions_after=0 health_after=200 -> PASS
HFG (h-red) no-disconnect: frames_at_close=15 abort_lines=0 closed_request_complete=true peer_complete=true green_assertion_fired=true -> PASS
health-fault-gate: arms=g,h pass=2 documented=0 fail=2 receipts=/home/avifenesh/projects/wt-spill-b/research/spill-b-20260919/rtx5090-day47/r1
```

- **h PASS and h-red PASS on both runs**: the closed client's session retired 97 and 96 ms after the close, the peer
  completed, the box idled with no active session; without the close there is no abort line and the green assertion
  fires.
- **g FAIL and g-red FAIL on both runs, as registered.** Placed from the logs: with three concurrent streamed requests
  the fault fired on the BATCHED decode chunk (`MEMRA_STEP_OOM_FAULT fired: this batched decode chunk reports a
  synthetic CUDA OOM (3 session(s))`), and the chunk's error arm (`worker.rs`, the `decode_step_batch_sampled_lean`
  error branch) ends every session of the chunk with the typed overloaded error (`the model is temporarily at
  capacity; retry after the Retry-After delay`, an SSE error event on a 200 stream): no park, no requeue, peers
  included. 1.1 assumed the fault would land on one session's own step; in the served batched regime every
  concurrent session shares the chunk. No 5xx and no panic on either arm. The registered clause reads FAIL; the
  batched chunk's missing OOM recovery is a server gap against DAY24 (c)'s acceptance ("peers' streams complete") and
  is opened as O14.

### 1.6 Addendum A (2026-09-26, after 2.1, before any code of the revision)

- **Arm g, reshaped to the door's documented branch:** one non-streamed request (nothing reaches the client before it
  completes), so the fault fires on its own non-batching step (`MEMRA_STEP_OOM_FAULT fired: this non-batching step`):
  green is one `step OOM parked session back to queue` line and the request ends `200` with a `finish_reason`, no
  5xx, no panic, `/health` 200 after. **g-red:** `MEMRA_STEP_OOM_FAULT=4` walks it into the bounded-retry honest
  error, and the green assertion must fire.
- **Arm g-batch, a DOCUMENTED reading (not a pass):** the three-stream shape of 1.1, reporting how many sessions of
  the chunk ended with the overloaded error, the park count and the 5xx count, so the O14 revision has its before
  receipt in the gate.
- The runs repeat as 1.3 (twice on each card). Arm h and its twin are unchanged.

### 2.2 Addendum A's first run on the 5090 (`rtx5090-day47/a1`, 2026-09-26 15:18Z)

```
HFG (g) step-oom-parks-and-completes: fault=1 fired_lines=0 parked_lines=1 http=200 finish_reason=length completion_tokens=48 panic_lines=0 health_after=200 -> FAIL
HFG (g-red) step-oom-past-the-retry-budget: fault=4 fired_lines=0 parked_lines=3 http=503 error={the model is temporarily at capacity; retry after the Retry-After delay} panic_lines=0 green_assertion_fired=true -> PASS
HFG (g-batch) step-oom-on-a-batched-chunk: batched_fired_lines=1 parked_lines=0 completed=1/3 ended_with_error_event=2/3 http_5xx=0 (the chunk's error arm ends every session of the chunk; owed O14) -> DOCUMENTED
HFG (h) client-disconnect-retires-within-1000ms: frames_at_close=15 abort_lines=1 close_to_abort_line_ms=97 peer_complete=true peer_finish=length active_sessions_after=0 health_after=200 -> PASS
HFG (h-red) no-disconnect: frames_at_close=15 abort_lines=0 closed_request_complete=true peer_complete=true green_assertion_fired=true -> PASS
health-fault-gate: arms=g,h pass=3 documented=1 fail=1 receipts=/home/avifenesh/projects/wt-spill-b/research/spill-b-20260919/rtx5090-day47/a1
```

- **g reads FAIL on a condition the gate added beyond addendum A.** Addendum A's green clause is one park line, `200`
  with a `finish_reason`, no 5xx, no panic, `/health` 200 after, and the run meets every term of it (`parked_lines=1
  http=200 finish_reason=length completion_tokens=48 panic_lines=0 health_after=200`). The gate also required one
  `fired` line matching `this non-batching step`, and the plain route's solo step prints the other injection point's
  text (`MEMRA_STEP_OOM_FAULT fired: this step reports a synthetic CUDA OOM (model hfg, generated 0, oom_retries
  0/3)`), so `fired_lines=0`. The line reads FAIL as printed.
- **g-red PASS** (three parks, then `step OOM NOT parked (... retries 3/3 ...): reporting honestly`, `503` overloaded);
  **g-batch DOCUMENTED** (the batched chunk's fault ends 2 of 3 streams with the overloaded event, no park: O14);
  **h and h-red PASS**.

### 1.7 Addendum B (2026-09-26, after 2.2, before the next run)

The gate's `fired` count matches either solo-step injection point (`this step reports` or `this non-batching step
reports`). Two more runs (`a2`, `a3`) on the 5090 with the revised gate, then the target card. No clause changes.

### 2.3 The target card (the eleventh sitting, one RTX PRO 6000 Blackwell Workstation Edition, 2026-09-26 17:45 to 17:47Z)

Tree `8d870f04e` (the gate as addendum B revised it); the gate's release binary built on the box from that tree,
sha256 `5c62447d...8c88f112`; the 27B. Receipts at `pro-single-day47/box/` (the box mirror manifest checked, the binary
by hash only). The card's `card.csv` from the dry run reads `power.limit 505.00 W`; each run's `source.txt` reads
600.00 W. Nothing in these arms is timed. Verbatim, `b1` then `b2`:

```
HFG (g) step-oom-parks-and-completes: fault=1 fired_lines=1 parked_lines=1 http=200 finish_reason=length completion_tokens=48 panic_lines=0 health_after=200 -> PASS
HFG (g-red) step-oom-past-the-retry-budget: fault=4 fired_lines=4 parked_lines=3 http=503 error={the model is temporarily at capacity; retry after the Retry-After delay} panic_lines=0 green_assertion_fired=true -> PASS
HFG (g-batch) step-oom-on-a-batched-chunk: batched_fired_lines=0 parked_lines=1 completed=3/3 ended_with_error_event=0/3 http_5xx=0 (the chunk's error arm ends every session of the chunk; owed O14) -> DOCUMENTED
HFG (h) client-disconnect-retires-within-1000ms: frames_at_close=10 abort_lines=1 close_to_abort_line_ms=188 peer_complete=true peer_finish=length active_sessions_after=0 health_after=200 -> PASS
HFG (h-red) no-disconnect: frames_at_close=10 abort_lines=0 closed_request_complete=true peer_complete=true green_assertion_fired=true -> PASS
health-fault-gate: arms=g,h pass=4 documented=1 fail=0 receipts=/root/spill-receipts/b-day47/b1
HFG (g) step-oom-parks-and-completes: fault=1 fired_lines=1 parked_lines=1 http=200 finish_reason=length completion_tokens=48 panic_lines=0 health_after=200 -> PASS
HFG (g-red) step-oom-past-the-retry-budget: fault=4 fired_lines=4 parked_lines=3 http=503 error={the model is temporarily at capacity; retry after the Retry-After delay} panic_lines=0 green_assertion_fired=true -> PASS
HFG (g-batch) step-oom-on-a-batched-chunk: batched_fired_lines=1 parked_lines=0 completed=2/3 ended_with_error_event=1/3 http_5xx=0 (the chunk's error arm ends every session of the chunk; owed O14) -> DOCUMENTED
HFG (h) client-disconnect-retires-within-1000ms: frames_at_close=10 abort_lines=1 close_to_abort_line_ms=187 peer_complete=true peer_finish=length active_sessions_after=0 health_after=200 -> PASS
HFG (h-red) no-disconnect: frames_at_close=10 abort_lines=0 closed_request_complete=true peer_complete=true green_assertion_fired=true -> PASS
health-fault-gate: arms=g,h pass=4 documented=1 fail=0 receipts=/root/spill-receipts/b-day47/b2
```

- **g, g-red, h and h-red PASS on both runs.** The two runs read the same verdicts; the close-to-abort times are 188 and
  187 ms against the 1,000 ms bound. g-batch is DOCUMENTED on both runs, as registered (O14's subject). In `b1` the
  fault landed on a solo step (`batched_fired_lines=0 parked_lines=1`), so the probe did not exercise the batched
  chunk. In `b2` it landed on the batched chunk and ended one of the three streams with the error event.
- The target card ran before the 5090's `a2` and `a3` (addendum B named the 5090 first): the local card was held by
  another lane's cells. The 5090 runs follow from queue-l. No clause changes.
