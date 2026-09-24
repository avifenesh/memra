# WP-A owed ledger (written 2026-09-24, day 37, before any day-37 code)

Every open item owed to or by this lane, with its source, its acceptance clause where one was registered, and its
status. Order of work is top to bottom (section 1). Section 2 holds the items that are closed, delivered, or held by
another owner, each with the receipt or ruling that closes it, so nothing is dropped silently. Owner order,
2026-09-24: "every improvment and tunning should be done, no shortcut or fast path". Door decisions
(`MEMRA_KV_HOST_CONTRACTS`, decide-by 2026-10-05) are the owner's; the receipts for them are this lane's.

Resync for this ledger: `origin/main` `17dceb981` (integ58 merged A day 36 as `aadfea44d`) fast-forwarded into the
lane; STATE.md, DAY34 to DAY36, OWNER-THREAD-OFFLOAD.md, the lead's integ45 to integ58 (rulings 41 to 53), and lane C's
STATE.md and DAY39 section 7 re-read.

Status words: `open` (no pre-registration yet), `pre-registered` (its DAYnn section pushed), `built` (code in the lane,
cells pending), `5090 done` (the 5090 half read), `target owed` (its target-card half is the next sitting), `closed`
(receipts in place).

## 1. Open, in order of work

### 1. A's finding 5: the two native span cells fail once in parallel in one process

- Source: `DAY32.md` section 6 finding 5 (`rtx5090-day32/unit/engine-span-cells-parallel.log`); integ52 lead review
  ("A's finding 5 is owed, not waived"); ruling 47. Quoted failure: `a batch with a running span has not landed`.
- Acceptance: DAY37 section 1 (the all and pair arms 100 of 100 in parallel, serial green, the red arm, the target
  card's all arm 20 of 20).
- Status: 5090 done, target owed (DAY37 sections 2 to 9: the cause placed at same-context `cuMemFreeHost`,
  synchronous `cuMemFree` and module load by a two-thread probe; the fix `ba5da705d`, one pool context per native
  cell; pair 100/100, all 100/100, serial 3/3, the red arm failing as required).

### 2. Hash 1 (the demote's D2H receipt digest) off the owner thread

- Source: `DAY35.md` sections 2, 6 to 8 (design M refuted: M1's reply added a tick-top poll to the copy phase and a
  copy-phase hit does not park; `DAY35 M D order=o2 .. m-minus-base=+26.50 rule <=+25.0 .. -> FAIL`, identity
  default-on `4 FAILURE(S)`, failure-on `1 FAILURE(S)`, fault-default `13 FAILURE(S)`); `DAY36.md` section 5; rulings
  52 and 53 ("no off-thread form known"). Candidates named in the record: a copy-phase hit that parks (`DAY29.md`, the
  `Hashing` park with "one predicate change"; `DAY35.md` section 6); the device-side form of `DAY27.md` option (c)
  (a GPU digest behind Move 2 owed item 1, which is now closed).
- Price today: `copy settle` 8.33 ms per demote on the 5090 (write-combined leases, `DAY35.md` section 8), 0.70 ms on
  BOX5 (`DAY36.md` section 4, cached leases).
- Acceptance: none registered. DAY35 section 2's (a) to (d) are the template; the new design's own clauses are
  pre-registered before its code.
- Status: open (survey first, then pick on evidence).

### 3. The same-tick fill (design F) on slower CPUs

- Source: `DAY34.md` section 7 and finding 2 (BOX4: `156.9MB filled by the hash helper in 11.4ms`, the fill plus the
  96 spans outlast the 13.1 ms to the next tick top, `polls [2] (counts [90])`), section 9 ("a multi-threaded or
  chunked fill on the copy stream, not pre-registered"); `DAY35.md` section 4 (F kept on the 5090, flat on BOX4);
  rulings 49, 52 and 53.
- Acceptance: none registered. It must decide on a slower-CPU host as well as a 9950X-class host; the 5090 host decides
  neither.
- Status: open.

### 4. The strong-form receipt of the recurrent spans (both directions)

- Source: `DAY30.md` sections 3 and 9 (D2H: "the device four-lane digest of each span source plus the CPU oracle over
  the landed bytes; about +71 ms on the target card's helper"); `DAY31.md` section 2 and `DAY32.md` section 7 (H2D:
  "the helper's SHA-256 of the staged bytes required equal to the plane's share recorded at the demote, about +73 ms
  on the helper per promote on the target card"); rulings 42, 44, 47, 52, 53.
- Acceptance: none registered.
- Status: open.

### 5. The helper's promote-side fail-closed arms have no serving-shape fault cell (found in this ledger's read)

- Source: `DAY34.md` section 2 item 3 (design K's `Sources` job: "a helper gone, a reply that does not describe the
  job, or digests past the 10 s deadline latch the tier"); the fault gate's `hash-helper-gone` exits the helper on its
  FIRST job of any kind and `hash-never-lands` discards a `Hash` reply (`worker.rs` `HostHashWorker::spawn`), so in the
  gate's cells the first job is a demote's `Hash` and the promote's `Sources` arms are reached by no cell. The
  precedent: DAY30 section 9 owed a serving-shape span-refusal cell because only a GPU unit cell covered it (closed
  on day 31, ruling 44); DAY35 section 5 named the same gap for M1's receipt step. Every new helper job this ledger
  adds (items 2 and 4) carries the same obligation.
- Acceptance: none registered. A red arm per arm (helper gone, foreign reply, deadline) in the fault gate, each with
  its typed line, the tier latched, nothing published, the parked request served cold, byte-compared with a door-OFF
  boot as the other promote cells are.
- Status: open.

### 6. Move 1 item 3: the by-reference demote routes keep the blocking program

- Source: `OWNER-THREAD-OFFLOAD.md` Move 1 lists (day 17 item 3, day 18 item 3, days 27 to 29 item 3): "the admission
  reclaim flush (`evict_all_demoting`), the pause sweep and the handoff keep the blocking program; ... The pause sweep's
  park half holds live device state and needs the `Demoting` state to protect the park until publication"; ruling 47.
  Code today: `ContractD2h::OnTick` at `worker.rs` `evict_all_demoting`, the pause sweep's shape 1 (plain park) and
  shape 2 (deepest resident entry), and the handoff export's drain-demote.
- Acceptance: none registered.
- Status: open.

### 7. Lane C: why b1 shows no first-touch pre-submit step

- Source: `research/spill-c-20260919/DAY39.md` section 7 and STATE.md; ruling 43 ("a per-demote allocation line, lane
  A's engine code"). b1 `pre_submit` 43.71 (first three) and 42.44 (4th on) in C's cell against day 28's double-park
  step (39 to 45 ms on the first two demotes, about 6 ms steady).
- Acceptance: none registered (an attribution: a log-only per-demote allocation or first-touch line, read on the cell
  shape that shows it).
- Status: open.

### 8. Lane C: the b2 helper's `hashed_in` rise, 73.2 to 104.8 / 107.3 ms, unattributed

- Source: C DAY39 section 7; ruling 43. `DAY30.md` section 6 readings record the day-30 helper paying the heap `Vec`
  first touch (about 106 ms on demotes 1 to 3, 79.8 steady) in the double-park cell.
- Acceptance: none registered (attribution first; if a first touch recurs on every demote, the improvement that
  removes it is pre-registered as its own design).
- Status: open. Items 7 and 8 share a hypothesis (fresh heap pages on every demote where no host entry frees) that is
  tested, not assumed.

### 9. Lane C: promote tick 2 on b1 and b2 on the target card

- Source: C DAY39 section 3 (`40 of 40 runs lack tick 2` for promote ON on b1 and b2) and section 7; ruling 43.
- Acceptance: none registered. C's tick reader defines tick 2 as the second stretched tick after the fire; a reading
  of the promote's tick placement on the current tree, pre-registered, with the day-33 timeline fields that b1 and b2
  lack.
- Status: open.

### 10. Move 2 item 2: the publishes still on the tick

- Source: `OWNER-THREAD-OFFLOAD.md` days 24 to 26 lists item 2 ("the fanout leader's snapshot and the pause sweep's
  boundary snapshot (`prefix_snapshot` direct); the `dspark-boundary` publish; the `glm5-boundary` publish; every
  `OnTick` refusal"), carried as "2 to 4 unchanged" through day 36.
- Acceptance: none registered.
- Status: open.

### 11. Move 1 item 4: the decision cell (i), both classes, same window

- Source: `OWNER-THREAD-OFFLOAD.md` Move 1 cells (i) (day 16's clause: `stall_median(second stream) <= idle p99` of
  the same sitting for both classes, N=5 per arm per order, both orders, door ON in both); C day 29 ran it on tree
  `653c997f4` against the day-16 tree `1646d421b` (integ40): `DAY29 CELL(i) CLAUSE: NOT MET (demote=False
  promote=False admissible=True); executed-not-qualified`; ruling 47 carries item 4 as owed.
- Acceptance: day 16's clause, verbatim, unchanged.
- Status: open (the NOT MET reading is on a tree before options (a) and 2a, the D2H and H2D spans, K, F and M'; the
  cell is owed on the current tree; the clause is read as written).

## 2. Closed, delivered, or held by another owner

- **Move 1 item 1, the settle-time owner wait for an H2D**: closed on day 19 by `c96d51862` (`DAY19.md`; the target
  card's gates green in both arms; `OWNER-THREAD-OFFLOAD.md` day-19 section "It did"). In code today:
  `CudaTransfers::install_consumer_wait` (`tier_transfer.rs`) and `submit_batch`'s rule that an off-owner H2D is
  fenced at the settle, never at submit, pinned by the census `copy_stream_issue_and_owner_wait_rules_are_as_stated`.
  The Move 1 lists of days 27 to 29 and DAY31 / DAY32 section 7 restated it as "unchanged from day 18"; those lines
  were stale, and ruling 47 inherited them. Correction recorded here; the census is re-run in the day-37 checks.
- **Move 1 item 2 (the receipt hashes)**: the bundle hash closed on ruling 41 (days 28 and 29); hash 2 closed on
  rulings 52 and 53 (M'); hash 1 is item 2 above.
- **Move 2 item 1, the recurrent f32 state off the tick**: the D2H half closed on ruling 42 (day 30), the H2D half on
  ruling 47 (day 32), the D2D half on ruling 53 (day 36, `DAY36 PRICE VERDICT (target card) .. -> CLOSES`). Its
  strong-form receipt is item 4 above.
- **DAY30 section 9's small items** (the governor charge of the staging, the span-refusal fault cell, the staging
  return on a post-take abort): closed on ruling 44 (day 31).
- **The 9B conv, ssm and hidden split** (C's STATE.md "engine line"): closed by day 31's 1d (the copy-complete line's
  `by slot class` tally; ruling 44).
- **The H2D completion checksum on the owner thread** (DAY33 section 5a): closed by design K (ruling 49).
- **Move 2 item 3, the receipt's price read by the door review** (cell (v) of day 22 and its 5090 twin, reported
  verbatim, nothing relaxed): delivered as the owner's input to the 2026-10-05 door decision; the reading is the
  owner's.
- **Move 2 item 4, the lead's ruling on day 26's clause 2**: closed on ruling 37 ("the code stays"; clause 2
  re-derived on the token-emission reading, no gate moved).
- **DAY25 proposal 2** (a source-session identity on `PendingCapture`): closed on ruling 36 (the seam stays; 0.4 ms is
  the recorded price).
- **The darklanes VERDICT line** (`DAY36.md` section 4): drafted and handed; ruling 53 lands it through the lead's own
  PR in the darklanes repository.
- **The M1 receipt step's missing fault cells** (DAY35 section 5): moot with M1 reverted (`6ce8b1aea`); the obligation
  carries to any new helper job (item 5).
- **Stated limits, not owed** (integ49 lead review, "no finding"): the staging charges release at the latch while a
  quarantined ticket or a detached helper may still hold pinned buffers (a latched tier only); a multi-model context's
  staging set grows per distinct plane length; a fill on a detached helper holds staging (DAY32 section 7).
- **Recorded as outside the door's scope, no clause owed**: the one-token request's token emitted a tick after its
  prime (`advance_sample_emit`, DAY26 clause 2's reading, ruling 37: "a scheduler shape outside Move 2"); the verify
  arm's owner-stream digest (`MEMRA_KV_HOST_VERIFY`, a diagnostic arm, DAY30 finding 5).
- **Owner decisions** (not this lane's to take): the contracts door `MEMRA_KV_HOST_CONTRACTS`, decide-by 2026-10-05.
