# WP-B day 42: O12, the admission reclaim flush off the tick (a deferring flush inside `MEMRA_ADMIT_BY_MEMORY`)

OWED.md O12 (the lead's ruling at integ62): lane A's design V (`research/spill-a-20260919/DAY47.md`, receipts under
its day-47 `pro-single-*` dirs) moved the agent-pause sweep's demotes off the tick and left the admission reclaim flush
on it, because an off-tick flush frees nothing until its D2H lands, so moving it means deferring the arrival until the
landings, which is this lane's door. This day builds that deferring flush behind the door, measures it against today's
on-tick flush, and folds in O10 (the door's part (b) with the host tier armed, which days 31 to 36 never ran). Every cell
is `executed-not-qualified`; no default moves.

## 1. Pre-registration

Committed and pushed before any day-42 code and before any day-42 cell. Nothing in section 1 changes after a number is
seen; a failed clause is recorded as it reads and a revision is a new, dated addendum pushed before its code.

### 1.1 Today's flush (the code on the lane tip)

Armed (`MEMRA_ADMIT_BY_MEMORY=1`), an arrival the headroom cannot fit reaches the reclaim ladder. The door's `decide`
over the pre-flush readings gives the demote budget B (the arrival's shortfall when the host tier can hold it, else 0).
`evict_all_demoting` then walks the evictable prefix entries oldest first: while B is unspent and the tier accepts, it
demotes the entry by reference on the tick (`ContractD2h::OnTick`: the D2H, its receipt, the owner's wait and the bind
inside the tick, every tenant waiting on them); then it removes every evictable entry (demoted or not). The arrival is
re-read and admits in the same tick. The host tier holds one demote in flight per worker; a second demote of any route
settles the first blocking.

### 1.2 The arm (`MEMRA_ADMIT_RECLAIM_OFFTICK`, default unset; decide-by 2026-10-09)

Read only when `MEMRA_ADMIT_BY_MEMORY=1` and the host tier's contracts door (`MEMRA_KV_HOST_CONTRACTS=1`, the only
route with an off-tick D2H) are both armed; otherwise today's flush runs byte for byte. Armed:

- **The plan, once per arrival.** At the arrival's first reclaim pass, with B from `decide` as today: D = the oldest
  evictable entries whose cumulative bytes stay within B and that the tier would accept, R = every other evictable
  entry. R is removed at once (today's drop half, the same entries today's flush drops). D is the arrival's demote plan,
  kept on the request.
- **One landing at a time.** The plan's first entry leaves the device index and is demoted on the sink's off-tick route
  (`host_demote_prefix_entry`'s shape: the shell rides the pending demote, its planes return to the pool at the settle).
  No second submission while the tier's demote slot is held (a second would settle the first blocking, which is the
  stall this removes). The arrival is re-read: if it fits (R's drops sufficed), it admits now and the plan's remaining
  entries stay resident on the device; otherwise it defers with reason `reclaim-landing`.
- **What an arrival waits on.** The tick-top settle that publishes the in-flight demote (the existing
  `host_demote_settle_pending(.., Poll, "the tick top")`). At each later admission pass the arrival is re-read; while
  it is still short and the slot is free, the plan's next entry is submitted; when it fits it admits.
- **How long it may wait.** The door's own bound, `MEMRA_ADMIT_DEFER_BUDGET_MS` (default 8,000 ms) from its first memory
  defer. Past it and still short, the door's typed refusal (429 with `Retry-After`), `reason=reclaim-landing-timeout`
  on its `[admit-mem]` line. The in-flight demote completes on its own and publishes; the plan's unsubmitted entries
  stay resident.
- **What it books while it waits.** Nothing: it is not admitted. The in-flight demote's planes stay allocated, so the
  device reading does not count them free until the settle returns them; the plan's unsubmitted entries stay resident
  and evictable. The admission queue's FIFO order is unchanged.
- **Receipts.** `[admit-mem] reclaim off-tick: plan <D> demote + <R> drop (<bytes>MB demote budget) for <id>`,
  `[admit-mem] reclaim off-tick: submitted <n> of <D> (<MB>) for <id>`, the settle's existing publication line, and the
  defer and refuse lines' `reason=`.
- **The one-numeric-program argument.** The flush moves prefix-cache entries between tiers and drops some; it changes
  no running session's numbers. Every later request (the arrival itself included) meets one of three states for any
  entry: resident on the device (a device restore), published on the host (the promote path, whose bytes pass the
  tier's verify digest, so the restored cache is byte-identical to the device entry), or gone (a cold prime). A request
  that meets an entry while it is `Demoting` parks and promotes after the publication (lane A's design P). Each is a
  program that request already has under today's flush; the arm changes which state an entry is in at a given moment,
  never the program a state runs.
- **Fault arms (`MEMRA_KV_HOST_FAULT`, existing values).** `d2h-delay`: the landing is late, the arrival waits and then
  admits, or refuses typed at its budget. `flip-demote`: the landing fails its verify, the entry drops, its bytes free,
  the arrival admits, no host entry publishes. `sources-helper-gone` (or `hash-never-lands`): the tier latches off, the
  plan's unsubmitted entries drop as today's past-budget entries do, the arrival admits. A client that disconnects while
  it waits is dropped from the queue; its in-flight demote completes and publishes.

### 1.3 CPU before the cards

- A pure plan function (D and R from the evictable entries, B and the tier's acceptance) and a pure per-pass step (wait,
  submit, admit, refuse from the readings, the slot and the wait) with unit tests over every arm, including the
  budget's edge, a slot held by a demote of another route, and the timeout.
- Census tests: the arm is read only at the reclaim site and only with both doors; one submission per pass; R's drop is
  the same set today's flush drops; the tick-top settle is unchanged; unarmed, the site's text is today's.

### 1.4 The cell (`day42-client.py`, one boot = one arm)

Door ON (`MEMRA_ADMIT_BY_MEMORY=1`, open output 8192), the host tier armed (`MEMRA_KV_HOST_MB`: 8,192 on the 5090,
32,768 on the target card), the contracts door on unless named. Phases in one boot:

1. **Warm.** W distinct long prompts (8 prompts of 8,192 tokens from `docs/SERVING.md`, `max_tokens=1`) seed the prefix
   cache.
2. **Tenants.** T = 4 streamed chat generations (`max_tokens=2048`) start and run through the next phase; each records
   its inter-token gaps.
3. **Burst.** B = 32 (5090) or 64 (target) open-output arrivals released together, sized to reach the reclaim flush.
4. **Warmth.** The W prompts again, each followed by its cold twin in a fresh namespace, `max_tokens=32`.

Arms: `ontick` (the new flag unset) and `offtick` (`=1`), both orders (O1 `ontick` then `offtick`, O2 the reverse), one
binary; `ontick-nocontracts` (the host tier armed, the contracts door off: O10's other arm); fault boots on `offtick`:
`d2h-delay`, `flip-demote`, `sources-helper-gone`. 8 boots per card.

### 1.5 Clauses

- **F1 identity.** Every tenant row and every warmth row has the same digest on `ontick` and `offtick` in each order;
  every warmth row equals its cold twin (a device restore, a host promote or a cold prime are the same program by the
  tier's verify and the grid law).
- **F2 the flush ran.** On `ontick`, at least one `[admit-mem] reclaim demoted` line; on `offtick`, at least one plan line,
  at least one submitted line and at least one publication, and no `[admit-mem] reclaim demoted` line (no on-tick demote).
- **F3 health.** No `CUDA_ERROR_OUT_OF_MEMORY` line, no 503, no crash line; every 429 carries `Retry-After` in 1..60.
- **F4 faults.** `d2h-delay`: F3, and every waiting arrival either admits or gets the typed refusal with
  `reclaim-landing-timeout`. `flip-demote`: F3, no host publication for the flipped entry, its arrival admits.
  `sources-helper-gone`: F3, the tier's latch line, the arrival admits.

Readings, no bound: the tenants' inter-token gap p50, p99 and max inside the burst window per arm (the stall today's
flush imposes); the arrivals' TTFT per arm (the wait the arm imposes); demoted entries and bytes per arm; host entries
resident after the burst; warmth hits by tier (device, host promote, cold) per arm; `ontick-nocontracts`' demoted count,
bytes and flush tick cost (O10). Every median states N and the 250 ms regime.

### 1.6 What each card decides

Nothing is promoted by this day: the door's verdict and this flag's are the owner's. The 5090 reading and the target
card's are each their own; no timing crosses cards.

### 1.7 Price

Code: about 0.5 agent-day. Cells: about 2 h on the 5090 (after its reset) and about 3 h on the target card, in the next
sitting.

### 1.8 Addendum A (2026-09-25, while writing the cells, before any cell)

Writing the fault boots against the code found that 1.2's `flip-demote` arm is the wrong fault for "the landing fails
its verify": `flip-demote` corrupts the host copy after its D2H receipt, the demote publishes, and the verify arm
refuses the entry at its PROMOTE. The fault that makes the landing itself fail is `d2h-source-flip` (one byte of the
first KV item's device source flips after its receipt digest and before its copy; the bind refuses the image). So:

- **F4's flip arm is `fault-d2h-source-flip`:** F3, the fault's armed line, no `[prefix-host] demote:` publication for
  the flipped demote (the boot's publication count is below its submitted count), and its arrival admits or gets the
  typed refusal.
- `fault-flip-demote` leaves the list; the promote-side verify is the host tier's own gate (lane A's), not this flush's.
- The other clauses, arms and readings are unchanged.
- The tenants are streamed `/v1/completions` requests with `prompt_ids` (1,024 ids, `max_tokens=2048`), like every
  other phase of the client, not chat requests as 1.4 wrote.

### 1.9 Addendum B (2026-09-25, from the spill review patterns, before any cell)

Reading the arm's state machine against the review patterns found one defect of the "a wait that never ends" shape:
1.2 makes the plan once per arrival. Entries that become evictable after the plan (other sessions' seeds) are then
never dropped for that arrival, and `decide` keeps answering `DemoteThenAdmit` while any evictable bytes exist, so an
arrival whose plan ran out could defer past its budget without ever reaching the refusal. Today's flush has no such
case because every pass removes every evictable entry.

- **The fix.** When the plan is empty and the tier's demote slot is free, the next reclaim pass makes a new plan from
  the current evictable set (today's demote rule over the pass's own budget; everything else drops), with its own plan
  line. Each plan drops or demotes every entry evictable at its making, so the evictable set shrinks to what arrives
  later; with none left, `decide`'s own rule defers and then refuses as today.
- The other patterns: moves happen after the shape is decided (`px_find_evictable`, then `remove_at`); the waiting
  arrival joins the parked-only bounded wait; the reader's counts come from the run's own lines; a refused submission
  drops its entry and a refused or timed-out arrival drops its plan, leaving the unsubmitted entries resident; the
  demoted entry's shell rides the pending demote, whose settle returns its planes; the demote budget is `decide`'s
  shortfall, capped by the host tier's room; the arm hands no memory back on a reply.
- No clause, bound or reading of 1.4 and 1.5 changes.

### 1.10 Addendum C (2026-09-25, after the target-card sitting read the flush as never reached; before any cell of this shape)

Section 2.1 places why the registered cell never reached the reclaim flush on the target card: the warm phase seeded
nothing (a `max_tokens=1` request on the spec route parks its session in the spec pool and publishes no prefix entry;
the 8,192-token entries appear only in the warmth phase, whose 32-token requests run a burst), and the burst never
deferred (64 open-output arrivals at 8,192 charged about 2.6 GB each and all fit beside the prefix cache on a 96 GB
card). The arm, its clauses and its readings are unchanged; the shape that exercises them:

- **Warm.** W prompts, `max_tokens=16` (a spec burst runs, so the boundary capture publishes each entry), W and the
  prompt length per card: 24 of 8,192 tokens on the target card (about 10 GB of evictable device entries at 415 MB
  each), 16 of 4,096 tokens on the 5090.
- **Pressure.** `MEMRA_ADMIT_OPEN_OUTPUT_TOKENS` large enough that the burst cannot all fit: 131,072 on the target card
  (about 4.2 GB of context per arrival at 31,552 B/token), 32,768 on the 5090 (about 580 MB at 16,704 B/token), with
  the burst of 1.4 (64 and 32). The first arrival that does not fit then has a shortfall under the evictable bytes,
  which is `decide`'s demote arm.
- **P0 precondition (a new line, not a clause).** Per boot, the reader prints the warm entries published before the
  burst window and the reclaim passes; a boot with no warm publication or no reclaim pass reads `NOT-EXERCISED`, and
  its F2 and F4 lines are not a verdict on the arm.
- Everything else of 1.2 to 1.9 stands: arms, orders, faults (`d2h-source-flip` per addendum A), clauses F1 to F4, the
  readings.

## 2. Results

Written after the runs. Section 1 is unchanged except by its dated addenda.

### 2.1 The first target-card sitting (one RTX PRO 6000 Blackwell Workstation Edition, 2026-09-25 11:21 to 14:18Z)

`1d11d5426` (server `65473af7...`), eight boots, receipts `pro-single-day42/box/` (208 files, checked against the box
manifest). The reader's lines, verbatim:

```
DAY42 F3 card=pro6000 boot=ontick-O1 oom_lines=0 crash_lines=0 r503=0 bad_429=[] burst_status={200: 64} counts={'ontick_demoted': 0, 'plans': 0, 'submitted': 0, 'published': 25, 'landing_defers': 0, 'landing_refusals': 0} -> PASS
DAY42 F3 card=pro6000 boot=offtick-O1 oom_lines=0 crash_lines=0 r503=0 bad_429=[] burst_status={200: 64} counts={'ontick_demoted': 0, 'plans': 0, 'submitted': 0, 'published': 25, 'landing_defers': 0, 'landing_refusals': 0} -> PASS
DAY42 F3 card=pro6000 boot=offtick-O2 oom_lines=0 crash_lines=0 r503=0 bad_429=[] burst_status={200: 64} counts={'ontick_demoted': 0, 'plans': 0, 'submitted': 0, 'published': 25, 'landing_defers': 0, 'landing_refusals': 0} -> PASS
DAY42 F3 card=pro6000 boot=ontick-O2 oom_lines=0 crash_lines=0 r503=0 bad_429=[] burst_status={200: 64} counts={'ontick_demoted': 0, 'plans': 0, 'submitted': 0, 'published': 25, 'landing_defers': 0, 'landing_refusals': 0} -> PASS
DAY42 F3 card=pro6000 boot=ontick-nocontracts oom_lines=0 crash_lines=0 r503=0 bad_429=[] burst_status={200: 64} counts={'ontick_demoted': 0, 'plans': 0, 'submitted': 0, 'published': 26, 'landing_defers': 0, 'landing_refusals': 0} -> PASS
DAY42 F3 card=pro6000 boot=fault-d2h-delay oom_lines=0 crash_lines=0 r503=0 bad_429=[] burst_status={200: 64} counts={'ontick_demoted': 0, 'plans': 0, 'submitted': 0, 'published': 25, 'landing_defers': 0, 'landing_refusals': 0} -> PASS
DAY42 F3 card=pro6000 boot=fault-d2h-source-flip oom_lines=0 crash_lines=0 r503=0 bad_429=[] burst_status={200: 64} counts={'ontick_demoted': 0, 'plans': 0, 'submitted': 0, 'published': 24, 'landing_defers': 0, 'landing_refusals': 0} -> PASS
DAY42 F3 card=pro6000 boot=fault-sources-helper-gone oom_lines=0 crash_lines=0 r503=0 bad_429=[] burst_status={200: 64} counts={'ontick_demoted': 0, 'plans': 0, 'submitted': 0, 'published': 25, 'landing_defers': 0, 'landing_refusals': 0} -> PASS
DAY42 F1 card=pro6000 order=O1 rows=20 arm_differ=[] cold_differ=[] -> PASS
DAY42 F2 card=pro6000 order=O1 ontick={'ontick_demoted': 0, 'plans': 0, 'submitted': 0, 'published': 25, 'landing_defers': 0, 'landing_refusals': 0} offtick={'ontick_demoted': 0, 'plans': 0, 'submitted': 0, 'published': 25, 'landing_defers': 0, 'landing_refusals': 0} -> FAIL
DAY42 F1 card=pro6000 order=O2 rows=20 arm_differ=[] cold_differ=[] -> PASS
DAY42 F2 card=pro6000 order=O2 ontick={'ontick_demoted': 0, 'plans': 0, 'submitted': 0, 'published': 25, 'landing_defers': 0, 'landing_refusals': 0} offtick={'ontick_demoted': 0, 'plans': 0, 'submitted': 0, 'published': 25, 'landing_defers': 0, 'landing_refusals': 0} -> FAIL
DAY42 F4 card=pro6000 boot=fault-d2h-delay burst=64 all_200_or_429=True landing_refusals=0 armed=True -> PASS
DAY42 F4 card=pro6000 boot=fault-d2h-source-flip burst=64 all_200_or_429=True armed=True submitted=0 published=24 -> FAIL
DAY42 F4 card=pro6000 boot=fault-sources-helper-gone burst=64 all_200_or_429=True latch_lines=12 -> PASS
DAY42 READING card=pro6000 boot=ontick-O1 tenant_gap_ms_in_burst N=7576 p50=139.1 p99=142.1 max=13292.5 burst_ttft_ms N=64 p50=27608.5 p95=62996.2 demoted_MB=5348 warmth_hits=0/8
DAY42 READING card=pro6000 boot=offtick-O1 tenant_gap_ms_in_burst N=7596 p50=139.4 p99=142.1 max=14033.6 burst_ttft_ms N=64 p50=29140.1 p95=64829.9 demoted_MB=5348 warmth_hits=0/8
DAY42 READING card=pro6000 boot=offtick-O2 tenant_gap_ms_in_burst N=7592 p50=139.3 p99=141.9 max=14021.0 burst_ttft_ms N=64 p50=29120.1 p95=205777.7 demoted_MB=5348 warmth_hits=0/8
DAY42 READING card=pro6000 boot=ontick-O2 tenant_gap_ms_in_burst N=7596 p50=139.5 p99=141.9 max=14030.3 burst_ttft_ms N=64 p50=29129.3 p95=64807.6 demoted_MB=5348 warmth_hits=0/8
DAY42 READING card=pro6000 boot=ontick-nocontracts tenant_gap_ms_in_burst N=7596 p50=139.4 p99=142.1 max=15119.8 burst_ttft_ms N=64 p50=29232.4 p95=64912.2 demoted_MB=5566 warmth_hits=0/8
DAY42 READING card=pro6000 boot=fault-d2h-delay tenant_gap_ms_in_burst N=7596 p50=139.5 p99=142.0 max=14049.9 burst_ttft_ms N=64 p50=29179.9 p95=64860.0 demoted_MB=5348 warmth_hits=0/8
DAY42 READING card=pro6000 boot=fault-d2h-source-flip tenant_gap_ms_in_burst N=7572 p50=139.4 p99=142.2 max=14055.9 burst_ttft_ms N=64 p50=29186.0 p95=64900.6 demoted_MB=5158 warmth_hits=0/8
DAY42 READING card=pro6000 boot=fault-sources-helper-gone tenant_gap_ms_in_burst N=7572 p50=139.4 p99=142.1 max=14103.1 burst_ttft_ms N=64 p50=29141.7 p95=64823.0 demoted_MB=5348 warmth_hits=0/8
```

**The registered cell did not exercise the arm.** Placed from the server logs:

- **No reclaim pass ran on any boot.** Not one `[admit-mem]` line has `verdict=defer`, `refuse` or `demote-then-admit`;
  every burst arrival admitted (`burst_status={200: 64}`). The 64 open-output arrivals charged about 2.6 GB each at
  the open-output value 8,192 and all fit on the 96 GB card beside the prefix cache. `ontick_demoted=0` and `plans=0`
  on every boot: neither flush ran, so F2 reads FAIL in both orders and F4's `d2h-source-flip` line reads FAIL on
  `submitted=0`; those lines are not a reading of the arm.
- **The warm phase published nothing.** The eight 8,192-token warm requests (`max_tokens=1`) ran on the spec route
  (`[spec-k] ... tenant="warm" K=3 source=cold-long`) and parked their sessions in the spec pool
  (`spec-affinity: declined (history diverged at 0 of checkpoint 8192; 2 parked ...)`); no 8,192-token prefix entry
  exists until the warmth phase, whose 32-token requests run a burst and publish (`insert (spec-boundary): 8192 tokens,
  415.4MB`). So `warmth_hits=0/8` and every warmth row equals its cold twin trivially.
- **What the 25 publications are.** `published=25` counts the host tier's capacity-eviction demotes (the prefix cache's
  own sink, on inserts past its 15,883 MB budget), not the admission flush.
- What stands as read: F3 PASS on all eight boots (no OOM, no 503, no crash line); F1 PASS in both orders (tenant and
  warmth digests equal across the arms; not a test of the flush). The tenants' gap p99 is 142 ms on every boot and the
  max 13 to 15 s (a reading; without a flush it is the burst's own admission and prime work).

Addendum C (1.10) pre-registers the shape that reaches the flush.

### 1.11 Addendum D (2026-09-25, while writing addendum C's chains, before any cell of that shape)

Addendum C's pressure (a large open-output value) makes every admitted burst arrival generate up to that value
(131,072 tokens of raw document continuation on the target card: hours per boot). The pressure is the booked and
allocated context, not the output, so:

- **The burst arrivals send `max_tokens=64` and `max_ctx = 2,048 + M`** (M = 131,072 on the target card, 32,768 on the
  5090). A request-supplied `max_ctx` is the charged and allocated context (`request_ctx_cap`'s authoritative arm), so
  each arrival books and allocates M + 2,048 rows and generates 64 tokens. `MEMRA_ADMIT_OPEN_OUTPUT_TOKENS` stays at its
  8,192 default (no burst request is open-output).
- Everything else of addendum C stands (the warm set at `max_tokens=16`, the P0 line).
