# WP-B owed ledger (written 2026-09-24, day 37 start, on main `17dceb981`)

Every open item owed to or by this lane, read from the lane's records (STATE.md through day 36, DAY2 to DAY36,
KV-RESIDENCY-DESIGN.md, the park and admission rows of `docs/FLAGS.md`, `docs/decisions/KV-PHYSICAL-RECLAIM.md`) and
from the lead's record (`research/spill-lead-20260919/INTEGRATION-DAY12.md`, rulings 44 to 53). Owner order,
2026-09-24: "every improvment and tunning should be done, no shortcut or fast path". So every item below is executed
in full: a real design, the real mechanism, measurement on the local RTX 5090 Laptop GPU and on the target card (one
RTX PRO 6000 Blackwell Workstation Edition), and the gates. Defaults and door verdicts stay the owner's; the lane
states which value each pre-registered rule selects and picks none.

Order of work: by decide-by date (O1, O2, O3), then O4 to O10. One item per DAYnn.md from day 37 on. Status words:
`open` (not started), `pre-registered`, `running`, `receipts banked` (the lane's work is done and the item waits only
for the owner or the lead), `closed` (with its closing pointer), `owner-only` (not worked by this lane).

## Door deciding cells

### O1. `--kv-allocator vmm`, decide-by 2026-10-04

- Source: `docs/decisions/KV-PHYSICAL-RECLAIM.md` ("Decide-by 2026-10-04: Promote to the naked default for the tiered
  materializer only with the 32k residual classified on the target card and the serving-shape gates run; otherwise
  delete the door"); `KV-RESIDENCY-DESIGN.md` option (b) (grow-on-demand planes on a reserved VA range) and its
  recommended order ("(b) second, on the VMM door's own decide-by"); lead rulings 6 and 7 (DAY11, DAY12).
- Acceptance on record: the 32k residual classification is met on both card classes
  (`ACTIVE-32K G1 PASS (classified one-time-driver-mapping-metadata, 5 cycles)`, DAY12). Owed: the serving-shape
  gates. Option (b)'s gate list: a G1 grow series (N>=5 granule-boundary crossings in one process, driver free falling
  by exactly the mapped granules, drift 0); the byte cells on both allocators with equal lines (`kernel-check`,
  `run-gen`, `run-spec` K=1..8, `prefix-newest-turn-fits`, `qwen-hit-gate`, `serve-smoke`, `cache-meter`); a serving
  stall cell at the granule boundary (tick latency, N>=5, both orders); and the admission accounting cell (booked
  against mapped against used). Ruling 7's 8k cycles control ("revisited only if the door is promoted at decide-by")
  rides the same item.
- The tuning the allocator needs to be measured at its best, from the design note: grow-on-demand (map granules as
  `pos` crosses a boundary, never move a byte), pre-mapping the next granule off the boundary tick, the parked-session
  tail release (a parked cache keeps its VA and captured graphs; only unused granules go back), and admission and
  metrics that count mapped bytes (`effective_free_bytes`, `cuda_pool_cached_bytes` do not see VMM planes).
- Status: `running`. Target class (RTX PRO 6000 Blackwell Workstation): the rule of DAY37 1.7 reads
  PROMOTE-ELIGIBLE from the r4 source across the two sittings (DAY37 2.5, 2.6); the two boxes' stage-0 probes disagree
  (55.3 against 271.2 us busy p95; inline serving grows p99 139 and 141 us), which the owner weighs with the door. The
  5090 class has no reading: its r4 half stopped at the card's reset (queue-e runs the rest when the card is healthy).
- Price: 3 to 4 agent-days (design note), plus a target-card sitting of about 8 h (the byte cells on both allocators,
  the stall cell both orders, the grow series, the accounting cell) and the matching 5090 holds.
- 5090 class (DAY37 2.7): FAIL (no reading) on A1, the pooled control arm's admit-mem burst gate RED. Placed (DAY39
  2.3): the r4 tree lacks DAY39's addendum B, and the registered DAY39 5090 half on the same binary reads the same red
  (8 parked prefill OOMs, `pending_prime=0`). DAY37 addendum G's repro (r4 twice, v3 once) runs from queue-k; a rerun of
  O1's 5090 cell on a tree with addendum B is owed once it reads.

### O2. `MEMRA_KV_PARK_COMPACT`, decide-by 2026-10-06

- Source: its `docs/FLAGS.md` row; `DAY28.md` section 2; `KV-RESIDENCY-DESIGN.md` day-28 addendum ("The cell that
  decides the door").
- Acceptance on record: plain path (`MEMRA_SERVE_SPEC=0`), both cards, default against `=1`, both orders, N>=5 per arm
  per length, one binary: (i) the compacted-park resume byte-identity gate on both resume shapes (an exact-extension
  continuation resuming a compacted entry, against the same continuation resuming a plain-parked entry, against a
  cold prime), digests equal on every request, `plain-affinity` hit lines present on the resumed arms; (ii) the
  step-OOM adjacency replay (`retire_may_park(_, true)` refuses the park, no `park-compact` line, no entry left);
  (iii) the park-time copy cost per park at the served context on both cards (the local 9B pair is owed).
- Status: `running`. Target class: the rule of DAY38 1.6 reads PROMOTE-ELIGIBLE under addendum D (DAY38 2.3: P1'
  PASS in both orders with 30 of 30 resumes on both arms, P2' PASS with the red arm red). 3 of 30 verbatim-extension
  resumes flip against cold on both arms (the pool's near-tie residual, O11). The 5090 half waits for the card
  (queue-e). Day 27 has the target-card plain-path receipt for (iii) only.
- Price: 1 agent-day (design note), plus about 3 h on each card.

### O3. `MEMRA_ADMIT_BY_MEMORY` ON rows on the capped seed booking, decide-by 2026-10-07

- Source: `DAY36.md` 2.2 and 2.6; integ56 packet option (4); ruling 51 ("runs only if the owner asks"); the owner's
  2026-09-24 order makes it owed.
- Why: day 36 ran the uncapped seed booking (`809c16444`). The cap `seed_booking_cap` (`34a4b7f23`, merged in #705)
  limits the booked seed to the prefix cache's remaining budget. Warm-cache ON rows may be over-booked.
- Acceptance: DAY34 1.6's terms unchanged (V-DOOR and its members) on the capped tree, both cards, both orders, plus
  a term that says whether the cap bound on each boot (from the `pending_seed=` field against the uncapped sum), so
  the rerun says what the cap moved rather than only repeating day 36.
- Status: target card `read` (DAY40 2.1): every DAY34 1.6 term PASS on 8 boots, V-DOOR PASS; SELECT R1 and R2 select
  none, R3's registry value 32,768; `on32768` admits 46 of 64 against day 36's 44; the seed cap bound on every ON boot.
  The 5090 half runs from queue-e. Pre-registered (DAY40.md), on the final booking (the capped seed and day 39's revised prime term, GREEN on
  the target card). Was `open`, DAY40. It follows O5 (DAY39): O5 corrects the door's own prime booking (a burst books the
  shared slab once per session today), and the ON rows are only the owner's input on the booking that would ship.
  Running them first would bank a second known over-booking beside day 36's.
- Price: 0.1 agent-day plus about 4.5 h local and 6 h on a target card (STATE, day 36).

## Admission and memory improvements

### O4. The outstanding-only `W` release at prime completion

- Source: `DAY24.md` sections 1 and 3 ("release the workspace charge when the prime completes, keep the persistent
  terms ... needs a prime-completion seam in both books and a retire-side accounting change; it is named, not done").
- What: the predictive book (`AdmissionBook`, `shadow_kv_hat`) and the real book (`booked_kv_bytes`) carry `W` for a
  session's whole life; the workspace is live only during the prime. Under `MEMRA_ADMIT_BY_MEMORY` the booked
  reading already counts `W` only for still-priming sessions (`pending_prime`, day 33), so this item is the two books'
  side.
- Status: `running`. Pre-registered (DAY45.md, `3182da256`), addendum A (the release site after the command drain);
  `MEMRA_ADMIT_W_RELEASE` coded (`21b081ee1`, decide-by 2026-10-10; memra-server 957 passed, clippy clean). The tenth
  sitting (`pro-single-b-sitting10.sh`, 4 boots, about 1.5 h) and queue-k (the 5090) run the cells. Price: about 0.3
  agent-day plus about 40 min on the 5090 and 1.5 h on the target card.

### O5. The shared prime slab charged per request

- Source: `DAY24.md` section 2 (ii) ("the physical gate charges `W`'s `call_row_bytes` term per request while the
  prime slab pool is one retained per-device allocation shared by every concurrent prime ... OVER-books concurrency by
  about 5.6 GB here ... named for the lead, not fixed by this lane").
- Why here: day 33's `pending_prime` sums every still-priming session's full `W`, so the door's booked reading carries
  the same over-count on a burst. The fix is the door's booking measured at its best, and it feeds O3.
- Status: `receipts banked` on the target card: DAY39 2.2 GREEN (v3: no OOM, 46 x 200 against v1's 44; v2 reproduces
  the 10 OOMs). The 5090 half and the admission gate on v3 wait for the card (queue-e). Feeds O3.
- Price: about 0.5 agent-day plus a cell on each card.

### O6. The enforcing predictive door on the fuller charge

- Source: `DAY24.md` section 4 ("the enforcing door was not exercised (it would now refuse on the fuller charge, which
  is the intended change and needs its own cell against a budget arm)"); STATE through day 32 ("the enforcing door on
  the fuller charge"). Dropped from STATE at day 33 with no closing record; restored here.
- Acceptance to pre-register: `MEMRA_ADMIT_PREDICT_ENFORCE=1` against a budget arm on both cards, the day-24 sequence
  and a burst, before and after the day-24 charge, every refusal a typed 429 with its `Retry-After`, no OOM.
- Status: `pre-registered` (DAY46.md, text only until DAY37 addendum G's repro and the ninth and tenth sittings read):
  arms `shadow`, `enforce`, `enforce-wrel` (with DAY45's W release) at the boot-derived budget, both orders, both cards,
  DAY24's sequence then a burst; P1 typed refusals, P2 no OOM, P3 within the budget, P4 identity, P5 the release reaches
  the door. No new engine or server code. Price: about 0.2 agent-day plus about 1 h on the 5090 and 2 h on the target
  card.

### O7. DAY24's step-OOM and client-disconnect fault arms

- Source: `DAY24.md` section 5 arms (c) and (d); `DAY25.md` sections 1 and 4 ("remain pre-registered only ... both have
  their doors: `MEMRA_STEP_OOM_FAULT`, and a client close").
- Acceptance on record: (c) an OOM at prime or step under a tiny headroom parks or requeues, no 5xx to peers, peers'
  streams complete; (d) a client disconnect mid-stream retires the session within one tick, ledger row
  `client_disconnected`, peers unaffected. Each with a red twin, added to `tools/health-fault-gate.sh`.
- Status: `running`. DAY47 2.1 (the 5090, twice): h and h-red PASS (retired 96 to 97 ms after the close); g and g-red
  FAIL as registered (the fault landed on the batched chunk, whose error arm ends every session: O14). Addendum A
  reshapes g to one non-streamed request (the door's documented park branch) plus a DOCUMENTED g-batch reading; the
  reruns and the target card follow.

### O8. The `[spec-vg]` predictive gap on MoE and linear families

- Source: `DAY28.md` 1.1 and 3.3 (the verify-graph pool grows per new key and is charged on the physical side only,
  as `vg_debt`; the predictive book does not carry it; not measurable on the dense 9B and 27B).
- Needs a MoE plus linear-attention model: a 35B-A3B NVFP4 artifact of that family is on the local disk (20 GB, a
  tight fit on the 24 GB card) and must be staged on the target card.
- Status: `pre-registered` (DAY48.md, text only until DAY46 reads): `MEMRA_ADMIT_PREDICT_VG_DEBT` makes the
  predictive verdict subtract the same `vg_debt` the physical side reserves; the cell runs the Ornith-1.5-35B-A3B NVFP4
  MTP artifact (20 GB, on the local disk, to be staged on the target card) with `enforce` against `enforce-vg`, V1 the
  pool engages, V2 no OOM, V3 typed refusals, V4 identity. Price: about 0.3 agent-day plus about 1 h on the 5090 and
  1.5 h on the target card.

### O14. A batched decode chunk's OOM ends every session of the chunk (DAY47 2.1)

- Source: DAY47 2.1: the step-OOM door on three concurrent streamed requests fired on the batched decode chunk, and
  the chunk's error arm ended all three with the typed overloaded error (no park, no requeue, no retry), where DAY24
  (c)'s acceptance says peers' streams complete. The non-batching step has the park branch; the batched chunk has none.
- What: a pre-registered recovery for a quoted CUDA OOM on a batched decode chunk: reclaim (the prefix cache and parked
  sessions, the step-OOM ladder), then either retry the chunk when no session's cache was touched (the fault before
  device work, or a torn-state check), or split it, keeping one numeric program per request (a retried chunk is the
  same batched step); sessions that never emitted park as today. Design, the torn-state argument and the cells are
  pre-registered before code (DAY49).
- Status: `open`. Price: about 1 agent-day plus a cell on each card.

### O9. memra#464's guard seed

- Source: `DAY30.md` section 6; the lane's comment on #464 (2026-09-22): two guard-only seeds, "the format is your
  call"; the code is darklanes' `recover_request_ledger` (the money path).
- Status: waiting on the owner's format choice (sidecar `requests.jsonl.accounted`, which the issue itself names, or
  a zero-amount `debit` row per carried id). The implementation is 0.5 agent-day once chosen. The lane does not
  choose a money-path format.

### O10. The part (b) arm of `MEMRA_ADMIT_BY_MEMORY` with the host tier armed

- Source: lane C's packet (`ADMIT-BY-MEMORY-DECISION-PACKET.md`) section 5 items 3 and 4 and appendix D items 2
  ("Part (b)'s `reclaim demoted` count, bytes and tick cost with the host tier armed, on either card") and 4 (the two
  doors ON together). Days 31 to 36 ran with the host tier unarmed, so part (b) is the OFF program in every receipt.
- Acceptance to pre-register: an armed host tier (`MEMRA_KV_HOST_MB > 0`), door ON, a warm prefix cache and a burst
  that reaches the reclaim flush; the `reclaim demoted` lines, bytes and the flush's tick cost; identity against
  door OFF; the same with `MEMRA_KV_HOST_CONTRACTS=1`. Both cards.
- Status: folded into O12 (DAY42's `ontick` and `ontick-nocontracts` arms are this item's cell). Feeds O3.

### O11. The grid-checkpoint rewind arm for the verbatim-extension resume (lead's order, 2026-09-25)

- Source: the lead's message of 2026-09-25 ("pre-register and price the grid-checkpoint rewind arm ... so the owner
  decides the near-tie residual question on receipts. Do not change the serving default"); DAY38 2.1 to 2.3.
- What: `MEMRA_RESUME_GRID_REWIND` (default unset) makes a plain or spec pool exact-extension resume rewind to the
  entry's grid checkpoint and re-prime from there, so the resumed turn is cold-identical by the grid law; the cells
  price it against keeping the decoded rows (re-primed rows, TTFT, throughput, memory, fanout reach), both cards.
- Status: `running`, revised on the owner's direction ("resume vs rewind - i think its not or or question, but more
  of we didnt make it right yet"): the exact AND fast resume, DAY44 (`MEMRA_RESUME_EXACT`, decide-by 2026-10-10). Code:
  the in-call grid capture (`35ece04e7`), the door, exact hits and settle queue (`9fa281cff`, `138790651`), the
  prime-only spec settle (`7a4abb4c9`). GPU on the 5090: the capture equals a split prime's state bitwise; a resume
  from it and a settle then resume are cold-exact. Local smoke (not registered): 0 flips against cold on both routes
  (keep 12 of 20); gapped turns resume from settled points faster than keep (plain 42 to 43 against 45 ms, spec 52 to
  54 against 99 to 107 ms); a next turn that arrives before or during a G=256 settle pays the re-prime (plain x2.3,
  spec x1.29). The ninth sitting (`pro-single-b-sitting9.sh`, about 12 h) and queue-j (the 5090) run the registered
  cells. The measurement arms' readings stay banked (DAY41 2.1 and 2.2); `MEMRA_RESUME_GRID_REWIND` stays a
  measurement arm until DAY44 reads.

### O12. The admission reclaim flush off the tick (lead's ruling at integ62)

- Source: the lead's ruling at integ62; lane A's design V (`research/spill-a-20260919/DAY47.md` section 1: "Site 1
  stays on the tick in this design, and goes to the lead as a question ... an off-tick flush means deferring the
  arrival until the landings, which is the memory-admission door's own decision"), receipts under lane A's day-47
  `pro-single-*` dirs.
- What: `MEMRA_ADMIT_RECLAIM_OFFTICK` (default unset, only with `MEMRA_ADMIT_BY_MEMORY=1` and
  `MEMRA_KV_HOST_CONTRACTS=1`): the reclaim flush drops today's drop set at once and demotes its demote set off the
  tick, one landing at a time, and the arrival defers on the landings within the door's defer budget. Folds in O10.
- Status: `running`. BOX26 (addenda C and D, DAY42 2.2): exercised; F1 to F3 PASS; F4 d2h-delay and helper-gone PASS,
  source-flip FAIL as registered (the reader's count took the capacity sink's publications; the flipped flush demote
  published nothing). The price: tenants' largest gap 3.9 s against 8.6 s, burst TTFT p50 7.2 to 8.0 s against 11.8 to
  12.2 s, warmth kept 1 of 24 against 10 of 24 (a per-arrival plan). Addenda E and F: one worker-level demote queue
  (`0cf59870a`). The sixth sitting read it (DAY42 2.3): P0 EXERCISED, F1 to F4 PASS on all boots, warmth 10 of 24 on
  both arms, the tenants' largest gap 4.0 and 3.96 s against 8.9 and 9.0 s, burst TTFT p50 7.9 and 6.9 s against 11.8
  and 12.0 s; `offtick` admits 15 and 16 of 64 against 17 (arrivals deferred on the landing, refused at the defer
  budget). Target card done; the 5090 half runs from queue-f (`rtx5090-day42e`).

### O13. The spec pool's exact-extension miss after an overshooting final burst (DAY41 2.1)

- Source: DAY41 2.1 (R2 on the spec route, both arms): the default spec route resumed 34 of 60 exact-extension turns on
  the 27B, because `SpecSession::committed` keeps the final burst's accepted drafts past `max_tokens` and the exact
  probe needs the whole `committed` in the next prompt. The DSpark engine already clamps its commits at the budget.
- What: a default-OFF door that clamps the MTP session's final round at the request budget (accepted drafts past the
  budget are treated as rejected at that column, the same rollback a rejection takes), so the parked `committed`
  equals the public stream. The design, the one-numeric-program argument (the emitted tokens are the same accepted
  drafts; K=1..8 self-consistency), the census and the cells are pre-registered before code (DAY43).
- Status: target card `read` (DAY43 2.1): C2 to C5 PASS; C1 FAIL as registered (turn-3 prompts follow turn 2's
  completion; every same-prompt row equal on both arms). The clamp resumes 60 of 60 spec turns (today 34), later-turn
  TTFT p50 211 against 1,622 ms, p95 417 against 9,149 ms, throughput 9.91 against 9.34 tokens/s; the resumed turns
  carry the keep residual (24 of 60 flip), which O11's exact resume removes. The 5090 half runs from queue-i. Code
  `87d9e00d1`, `MEMRA_SPEC_BUDGET_CLAMP` default-OFF, decide-by 2026-10-10.

## Owner-only (listed, not worked)

- memra#476's boot pre-grow booking point (`DAY28.md` 3.3: `fa_dcw_pool_ensure` in `run_boot_calibration`; it moves the
  boot footprint, so it is the owner's).
- darklanes `lane/budget-journal-order-20260922` at `20f8140e8` (the money path), and `DAY30.md` section 5's two format
  forks (source-file top-ups have no row; the journal's stamp meaning).
- The park policy on the spec pool (TTL, per-pool cap, scope; memra#539, recorded on the issue as the owner's product
  switch).
- `phase=warming` on a first boot (`DAY25.md` 4: binding before the load is a serving-order change).
- Every door verdict and default: `MEMRA_ADMIT_BY_MEMORY` and its open-output value, `--kv-allocator vmm`,
  `MEMRA_KV_PARK_COMPACT`.
- The verbatim-extension continuation resume (plain and spec pools) keeps decode-computed rows, so a resumed turn can
  differ from the same prompt's cold prime at near-ties: `docs/SERVING.md` documents it as a near-tie residual,
  `CLAUDE.md`'s one-numeric-program rule reads such a crossing as a bug unless forbidden or proven bit-identical. The
  two conflict on this path. Measured: one flip in 19 exact-extension resumes on the 27B (DAY38 2.1, 2.2). A rewind
  to the entry's grid checkpoint would make the resume cold-identical by the grid law, at the cost of re-priming the
  previous turn's generated tokens; which contract holds is the owner's.

## Open in the records, outside this lane's clauses (for the lead to route)

- memra#445 sections 2 and 3 (`DAY22.md` section 3): gemma prefix capture on the spec route (needs memra#151's
  ring-aware snapshot first) and gemma spec under concurrency (a c=1/4/8 ladder). Mapped by this lane, never owed by
  it.
- The step35 continuation split, 16 against 17 rows (`DAY23.md` section 5): a confirmation under #614, owed to a rig
  that holds a Step-3.7-Flash GGUF.
- `KV-RESIDENCY-DESIGN.md` options (c) paged KV and (d) copy-on-write prefix sharing: the note makes (c) conditional on
  the residual measured after (a) and (b), and the order is the owner's. O1's accounting cell produces that residual.

## Closed since the records named them

- DAY31 2.11's stale engine text (`admit_memory.rs:48-51`, `lib.rs:3428-3429`): the lead's `584cbe2cf` (integ50).
- DAY22's target-card `kernel-check` line: `DAY23.md` (`ALL GREEN (109 cells, ...)`).
- DAY29's clean-card twin run: `DAY29.md` table (`twin27-off ... -> PASS` on a clean card).
- memra#680: closed with integ55 (ruling 50).
