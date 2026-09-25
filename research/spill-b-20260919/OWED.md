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
- Status: `running` (DAY37.md). The serving arm `MEMRA_KV_ALLOCATOR=vmm` is built (addenda A to E; r4 =
  `c6f9282c2`). Target card, first sitting (2.5): A2, A1-MIX, A1-STREAM, A3 (ii), A4, A5-BUILD and A7 PASS; stage 0
  selects inline grows (now the class default in code, `0b75283fe`); the gate set refused on a box without `ss`/`lsof`,
  A6's main binary refused a lane-only env name (harness), A5's two lines were reader defects. Addendum F (1.15)
  reruns those on the r4 source in the second sitting (`pro-single-b-sitting2.sh`). 5090: the r4 chain is at its
  stream pairs (a foreign process on the card stretched the waits); the seven boots its first batch missed run next
  (`rtx5090-queue-c.sh`).
- Price: 3 to 4 agent-days (design note), plus a target-card sitting of about 8 h (the byte cells on both allocators,
  the stall cell both orders, the grow series, the accounting cell) and the matching 5090 holds.

### O2. `MEMRA_KV_PARK_COMPACT`, decide-by 2026-10-06

- Source: its `docs/FLAGS.md` row; `DAY28.md` section 2; `KV-RESIDENCY-DESIGN.md` day-28 addendum ("The cell that
  decides the door").
- Acceptance on record: plain path (`MEMRA_SERVE_SPEC=0`), both cards, default against `=1`, both orders, N>=5 per arm
  per length, one binary: (i) the compacted-park resume byte-identity gate on both resume shapes (an exact-extension
  continuation resuming a compacted entry, against the same continuation resuming a plain-parked entry, against a
  cold prime), digests equal on every request, `plain-affinity` hit lines present on the resumed arms; (ii) the
  step-OOM adjacency replay (`retire_may_park(_, true)` refuses the park, no `park-compact` line, no entry left);
  (iii) the park-time copy cost per park at the served context on both cards (the local 9B pair is owed).
- Status: `running`. Target card (DAY38 2.1): FAIL (no reading) as registered; P1's cold-twin clause asserted, for
  shape X, what `docs/SERVING.md` documents as the verbatim-extension near-tie residual (2.2), and the workload never
  let `off` resume. Addendum D (1.11) re-registers P1 and P2 (shape Xp under `max_ctx`, door identity resume against
  resume, cold identity on the affinity rewind; the non-batching fault point after the prime, `45f1b948d`). The
  registered 5090 half runs as registered (queue-b), addendum D's on both cards (queue-c, the second sitting).
  Day 27 has the target-card plain-path receipt for (iii) only.
- Price: 1 agent-day (design note), plus about 3 h on each card.

### O3. `MEMRA_ADMIT_BY_MEMORY` ON rows on the capped seed booking, decide-by 2026-10-07

- Source: `DAY36.md` 2.2 and 2.6; integ56 packet option (4); ruling 51 ("runs only if the owner asks"); the owner's
  2026-09-24 order makes it owed.
- Why: day 36 ran the uncapped seed booking (`809c16444`). The cap `seed_booking_cap` (`34a4b7f23`, merged in #705)
  limits the booked seed to the prefix cache's remaining budget. Warm-cache ON rows may be over-booked.
- Acceptance: DAY34 1.6's terms unchanged (V-DOOR and its members) on the capped tree, both cards, both orders, plus
  a term that says whether the cap bound on each boot (from the `pending_seed=` field against the uncapped sum), so
  the rerun says what the cap moved rather than only repeating day 36.
- Status: `pre-registered` (DAY40.md), on the final booking (the capped seed and day 39's revised prime term, GREEN on
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
- Status: `open`. Price: about 0.5 agent-day plus a cell on each card.

### O5. The shared prime slab charged per request

- Source: `DAY24.md` section 2 (ii) ("the physical gate charges `W`'s `call_row_bytes` term per request while the
  prime slab pool is one retained per-device allocation shared by every concurrent prime ... OVER-books concurrency by
  about 5.6 GB here ... named for the lead, not fixed by this lane").
- Why here: day 33's `pending_prime` sums every still-priming session's full `W`, so the door's booked reading carries
  the same over-count on a burst. The fix is the door's booking measured at its best, and it feeds O3.
- Status: `running`. Target card (DAY39 2.1): NOT-GREEN, G-NOOM fails on both green runs (10 and 12 parked prefill
  OOMs): the section-1 term missed the plain checkpoint snapshot (156.9 MB per session on the 27B) and subtracted the
  slab from the call's returned rows. Addendum B (1.8) revises the term (`be2177ead`); `v1`/`v2`/`v3` cells on both
  cards (queue-c, the second sitting); the registered 5090 half still runs as registered (queue-b). Before O3.
- Price: about 0.5 agent-day plus a cell on each card.

### O6. The enforcing predictive door on the fuller charge

- Source: `DAY24.md` section 4 ("the enforcing door was not exercised (it would now refuse on the fuller charge, which
  is the intended change and needs its own cell against a budget arm)"); STATE through day 32 ("the enforcing door on
  the fuller charge"). Dropped from STATE at day 33 with no closing record; restored here.
- Acceptance to pre-register: `MEMRA_ADMIT_PREDICT_ENFORCE=1` against a budget arm on both cards, the day-24 sequence
  and a burst, before and after the day-24 charge, every refusal a typed 429 with its `Retry-After`, no OOM.
- Status: `open`. Price: about 0.3 agent-day plus a cell on each card.

### O7. DAY24's step-OOM and client-disconnect fault arms

- Source: `DAY24.md` section 5 arms (c) and (d); `DAY25.md` sections 1 and 4 ("remain pre-registered only ... both have
  their doors: `MEMRA_STEP_OOM_FAULT`, and a client close").
- Acceptance on record: (c) an OOM at prime or step under a tiny headroom parks or requeues, no 5xx to peers, peers'
  streams complete; (d) a client disconnect mid-stream retires the session within one tick, ledger row
  `client_disconnected`, peers unaffected. Each with a red twin, added to `tools/health-fault-gate.sh`.
- Status: `open`. Price: about 0.5 agent-day plus the gate on each card.

### O8. The `[spec-vg]` predictive gap on MoE and linear families

- Source: `DAY28.md` 1.1 and 3.3 (the verify-graph pool grows per new key and is charged on the physical side only,
  as `vg_debt`; the predictive book does not carry it; not measurable on the dense 9B and 27B).
- Needs a MoE plus linear-attention model: a 35B-A3B NVFP4 artifact of that family is on the local disk (20 GB, a
  tight fit on the 24 GB card) and must be staged on the target card.
- Status: `open`. Price: about 0.5 agent-day plus a cell on each card.

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
- Status: `open`. Price: about 0.3 agent-day plus a cell on each card. Feeds O3.

### O11. The grid-checkpoint rewind arm for the verbatim-extension resume (lead's order, 2026-09-25)

- Source: the lead's message of 2026-09-25 ("pre-register and price the grid-checkpoint rewind arm ... so the owner
  decides the near-tie residual question on receipts. Do not change the serving default"); DAY38 2.1 to 2.3.
- What: `MEMRA_RESUME_GRID_REWIND` (default unset) makes a plain or spec pool exact-extension resume rewind to the
  entry's grid checkpoint and re-prime from there, so the resumed turn is cold-identical by the grid law; the cells
  price it against keeping the decoded rows (re-primed rows, TTFT, throughput, memory, fanout reach), both cards.
- Status: `pre-registered` (DAY41.md). Price: about 0.5 agent-day plus about 3 h on the 5090 and 4.5 h on the target
  card.

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
