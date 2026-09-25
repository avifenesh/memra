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
