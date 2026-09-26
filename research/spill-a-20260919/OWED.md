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
- Status: **closed** (DAY37 sections 2 to 9: the cause placed at same-context `cuMemFreeHost`, synchronous `cuMemFree`
  and module load by a two-thread probe; the fix `ba5da705d`, one pool context per native cell; pair 100/100, all
  100/100, serial 3/3, the red arm failing as required; the target card's all arm on BOX7 `DAY37 FINDING5 TARGET
  all-arm green=20 of 20 rule 20 of 20 -> PASS`, DAY38 section 11a).

### 2. Hash 1 (the demote's D2H receipt digest) off the owner thread

- Source: `DAY35.md` sections 2, 6 to 8 (design M refuted: M1's reply added a tick-top poll to the copy phase and a
  copy-phase hit does not park; `DAY35 M D order=o2 .. m-minus-base=+26.50 rule <=+25.0 .. -> FAIL`, identity
  default-on `4 FAILURE(S)`, failure-on `1 FAILURE(S)`, fault-default `13 FAILURE(S)`); `DAY36.md` section 5; rulings
  52 and 53 ("no off-thread form known"). Candidates named in the record: a copy-phase hit that parks (`DAY29.md`, the
  `Hashing` park with "one predicate change"; `DAY35.md` section 6); the device-side form of `DAY27.md` option (c)
  (a GPU digest behind Move 2 owed item 1, which is now closed).
- Price today: `copy settle` 8.33 ms per demote on the 5090 (write-combined leases, `DAY35.md` section 8), 0.70 ms on
  BOX5 (`DAY36.md` section 4, cached leases).
- Acceptance: DAY38 section 3, (a) to (e).
- Status: **closed as G4** (ruling 54, integ59, `research/spill-lead-20260919/INTEGRATION-DAY12.md`: G4 is the door's
  form of hash 1 off the owner). G'' failed (d) on BOX7 (the tenant's per-demote decode hump); the bisection (DAY38 13a
  to 13k) placed the hump on two non-owner streams running kernels. G4 (`26676c037`: one side stream; the D2H receipt
  ahead of the copies on the copy stream, the D2D classes there too, `d2h-delay` a host-side hold) passes (a) to (f) on
  BOX7 (DAY38 section 19b: copy settle 0.53 ms, e2e -1.17 / -1.10 ms, hump +0.035, every gate green) and (a) to (e) on
  the 5090 (sections 19, 19a). Its 5090 (f) FAIL (section 19, +0.299 ms in a hot hold, 87 to 88 C, the clock falling)
  **stands as registered** (ruling 54: the base-controlled cell of section 20a read base and G4 flat in a cooler hold, 60
  to 78 C, and its rule presumed G4 would rise there; the cause is unplaced between the card's thermal regime and the
  design). Owed, **item 16** below: section 20's cell replicated in the G4 hold's thermal regime. G''' (`9ab5c1265`) is
  the measured alternative; G''' against G4 at long entries is item 15's cell (both arms, the evidence decides).

### 3. The same-tick fill (design F) on slower CPUs

- Source: `DAY34.md` section 7 and finding 2 (BOX4: `156.9MB filled by the hash helper in 11.4ms`, the fill plus the
  96 spans outlast the 13.1 ms to the next tick top, `polls [2] (counts [90])`), section 9 ("a multi-threaded or
  chunked fill on the copy stream, not pre-registered"); `DAY35.md` section 4 (F kept on the 5090, flat on BOX4);
  rulings 49, 52 and 53.
- Acceptance: DAY39 section 1's rule picks the design per host class; the design's acceptance is pre-registered after
  the host's reading.
- Status: **closed for the slower-CPU host class** (DAY39 section 7: BOX7, `polls==1 90` of 90, HK - FT +12.64 /
  +12.55 ms e2e and +12.70 ms PIN against pair noise 0.10 to 0.23, every gate green; the 5090's (e) PASS, section 6).
  Design T is `0153316d4` (the fill split across `min(12, cpus / 2)` threads), the door's fill program (ruling 54).
  The 9950X-class reading (DAY44 section 2, an `AMD Ryzen 9 9950X3D2` host): `DAY39 T TARGET (a) and (b) -> PASS`, the
  gates and the unit cells green (attempt 2 on the arms' tree); every T fits there, T=1 included. **Item 3 closed for
  both host classes.**

### 4. The strong-form receipt of the recurrent spans (both directions)

- Source: `DAY30.md` sections 3 and 9 (D2H: "the device four-lane digest of each span source plus the CPU oracle over
  the landed bytes; about +71 ms on the target card's helper"); `DAY31.md` section 2 and `DAY32.md` section 7 (H2D:
  "the helper's SHA-256 of the staged bytes required equal to the plane's share recorded at the demote, about +73 ms
  on the helper per promote on the target card"); rulings 42, 44, 47, 52, 53.
- Acceptance: DAY40 section 3 (amended 3a), (a) to (e).
- Status: **open, design S reverted** (`8a044a1c0`). S (`a40e5b334`, DAY40 sections 3 to 4) passed its correctness
  cells on the 5090 (every span digest bitwise, both red arms witnessed, identity, failure and hit gates green) and FAILED
  its price clauses there (section 5: (c) wall +40.25 / +39.30 ms, e2e +1.47 / +1.16 ms; (d) e2e +1.53 ms on o1): the
  trace (section 7) prices the span receipt at about 3.1 ms of copy-stream time (source digests 0.61, landed 2.49 ms),
  enough to push the landing past the first tick-top poll. The revision owed (section 7): the span receipt off the
  landing path, required at publication with the sources and staging held until its event, the digests in one launch
  each; pre-registered with its acceptance before its code; its 5090 and target sittings then. Ruling 54: S refuted and
  reverted. The revision, design S2 (`DAY42.md` sections 1 and 1a, built as `7ce3f3243`), **failed (c) on the target
  card and is reverted** (DAY42 section 3: e2e +3.16 / +3.06 ms against +1.0; (d), (e), every gate and the unit cells
  green): its span digests ran grids of 12288 and 6144 blocks that filled the card for about 3 ms per demote, and the
  owner stream ran no kernel meanwhile. Design S3 (the grid bounded, `DAY46.md`) **failed (c) too and is reverted**
  (DAY46 section 3: e2e +3.34 / +3.27 ms): the price is the release paths' whole-copy-stream drain, which the capture
  settle at the same tick top pays for the landed digests (3.23 ms held against 0.13 to 0.21 on G4). The revision,
  design S4 (the drain made precise, `DAY48.md`), **passes (a) to (e) on the target card** (DAY48 section 3: e2e +0.40 /
  +0.37 ms, PIN +0.10, hump +0.012, every gate green). Owed: the 5090 half, after the card's reset.

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
- Status: **closed.** Pre-registered (DAY41 section 1) and built (`4ca4bb36e`: `sources-helper-gone`,
  `sources-never-land`, `sources-foreign-reply` on the `MEMRA_KV_HOST_FAULT` row, keyed on the first `Sources` job; three
  fault gate cells). On the 5090 (DAY41 section 2) the fault gate default and plain `ALL GREEN` (229 ok each), each arm
  latching the tier in its own words with no promote published. The target half: BOX7's fault gate on the G''' tree,
  default and plain `ALL GREEN` (229 ok each, the three cells green; DAY38 section 16d).

### 6. Move 1 item 3: the by-reference demote routes keep the blocking program

- Source: `OWNER-THREAD-OFFLOAD.md` Move 1 lists (day 17 item 3, day 18 item 3, days 27 to 29 item 3): "the admission
  reclaim flush (`evict_all_demoting`), the pause sweep and the handoff keep the blocking program; ... The pause sweep's
  park half holds live device state and needs the `Demoting` state to protect the park until publication"; ruling 47.
  Code today: `ContractD2h::OnTick` at `worker.rs` `evict_all_demoting`, the pause sweep's shape 1 (plain park) and
  shape 2 (deepest resident entry), and the handoff export's drain-demote.
- Acceptance: DAY47 section 1 (design V: the pause sweep's two shapes off the tick; the export stays by its contract;
  the admission flush stays, a lead question: a deferring flush is the memory-admission door's decision).
- Status: **built (V, on S4) and passes (a) to (d) on the target card** (DAY47 sections 3 and 3b: the pause's tenant
  stall 202.9 ms on the tick, 3.2 ms off it; every gate green, the new pause gate after its section 3a revision). Owed:
  the 5090 half, after the card's reset. **Closed as V** (the lead, integ62): the admission reclaim flush stays on the
  tick; its deferring form belongs inside `MEMRA_ADMIT_BY_MEMORY` and is lane B's owed item (V's receipts its source).

### 7. Lane C: why b1 shows no first-touch pre-submit step

- Source: `research/spill-c-20260919/DAY39.md` section 7 and STATE.md; ruling 43 ("a per-demote allocation line, lane
  A's engine code"). b1 `pre_submit` 43.71 (first three) and 42.44 (4th on) in C's cell against day 28's double-park
  step (39 to 45 ms on the first two demotes, about 6 ms steady).
- Acceptance: none registered (an attribution: a log-only per-demote allocation or first-touch line, read on the cell
  shape that shows it).
- Status: **closed** (DAY49 section 3, the target card, run by the lead: `DAY49 ITEM7 -> b1 step attributed to H (the
  heap first touch)`; the same mechanism as item 8, taken on b1's owner thread and now on the helper).

### 8. Lane C: the b2 helper's `hashed_in` rise, 73.2 to 104.8 / 107.3 ms, unattributed

- Source: C DAY39 section 7; ruling 43. `DAY30.md` section 6 readings record the day-30 helper paying the heap `Vec`
  first touch (about 106 ms on demotes 1 to 3, 79.8 steady) in the double-park cell.
- Acceptance: none registered (attribution first; if a first touch recurs on every demote, the improvement that
  removes it is pre-registered as its own design).
- Status: **closed as attributed** (DAY49 section 3: `DAY49 ITEM8 pages=38306 nofree_minflt=38306 (rule >= 19153)
  free_minflt=0 (rule <= 9576) copy_ms nofree=23.73 free=7.64 diff=+16.09 (rule >= +5.0) -> H attributed`). The first
  touch recurs on every demote while the host tier fills and costs the publication one tick-top poll (wall 101.3 against
  88.6 ms), so its remedy is owed as item 17.

### 9. Lane C: promote tick 2 on b1 and b2 on the target card

- Source: C DAY39 section 3 (`40 of 40 runs lack tick 2` for promote ON on b1 and b2) and section 7; ruling 43.
- Acceptance: none registered. C's tick reader defines tick 2 as the second stretched tick after the fire; a reading
  of the promote's tick placement on the current tree, pre-registered, with the day-33 timeline fields that b1 and b2
  lack.
- Status: **closed** (DAY50 sections 2 and 3: one stretched tick, tick 2 absent in 100 of 100 runs on G4 and S4 and
  in 30 of 30 on the tip, `DAY50 POOLED runs=30 tick1=30 tick2=0 .. submit_to_publish_ticks={1: 25, 2: 2}
  owner_segment_med=0.52`). The one-tick-late publications recur on the tip (2 of 27), owed as item 18.

### 10. Move 2 item 2: the publishes still on the tick

- Source: `OWNER-THREAD-OFFLOAD.md` days 24 to 26 lists item 2 ("the fanout leader's snapshot and the pause sweep's
  boundary snapshot (`prefix_snapshot` direct); the `dspark-boundary` publish; the `glm5-boundary` publish; every
  `OnTick` refusal"), carried as "2 to 4 unchanged" through day 36.
- Acceptance: none registered.
- Status: **priced** (DAY54 section 3, the lead's run on a 5900XT host): the fanout publisher **design next** (`DAY54
  PRICE fanout (short) .. o1=+6.50 o2=+6.50 ms .. -> DESIGN NEXT`; its own parts the snapshot 0.72 ms and three
  restores 1.19 ms, its insert's 2.26 ms is the evicted entry's demote pre-submit, item 19; the long cell's +896 ms is
  the four members' own suffix primes past the 1024-token cap, recorded as read); the pause park snapshot **closed as
  priced** (0.73 ms); no route refusal on the 27B; the DFlash, GLM-5 and latent publishers not measured here, owed to
  their artifacts and rigs. The fanout design is **pre-registered** (DAY59 section 1: an attribution of the snapshot's
  and restores' owner time on the 5090, then design B1 (batched copies) or B2 (one pool reservation) by a stated rule).

### 11. Move 1 item 4: the decision cell (i), both classes, same window

- Source: `OWNER-THREAD-OFFLOAD.md` Move 1 cells (i) (day 16's clause: `stall_median(second stream) <= idle p99` of
  the same sitting for both classes, N=5 per arm per order, both orders, door ON in both); C day 29 ran it on tree
  `653c997f4` against the day-16 tree `1646d421b` (integ40): `DAY29 CELL(i) CLAUSE: NOT MET (demote=False
  promote=False admissible=True); executed-not-qualified`; ruling 47 carries item 4 as owed.
- Acceptance: day 16's clause, verbatim, unchanged.
- Status: **pre-registered** (DAY60 section 1: C's day-29 cell verbatim with arm X the current tip; the clause read
  as written; a target card).
  cell is owed on the current tree; the clause is read as written).

### 12. Every CPU hash over the 5090's write-combined leases reads at the direct rate (found by DAY38's survey)

- Source: `DAY38.md` section 2 (`SURVEY WC kind=write-combined bytes=1153434 N=5 direct_ms median=9.841 ..
  streamed_ms median=0.341`); the reads it covers: K's promote-side checksums on the helper (8.8 ms per promote on the
  5090, DAY34; 88 of 90 promotes land one tick later for it), M''s hash 2 on the helper (about 8 ms inside the
  `Hashing` job, DAY35 section 8), the verify arm's host digest, and hash 1 until item 2 lands.
- Acceptance: none registered (a streaming read for write-combined pinned sources in the hash path, the program
  unchanged, bitwise, priced on the 5090 and a no-regression reading on the target card's cached leases).
- Status: **pre-registered** (DAY61 section 1: design W, a streamed read through a cached bounce buffer into the
  same checksum program).

### 13. The capture retire seam's `Block` settle holds the owner thread when the capture's copy is queued behind other copy-stream work (found by DAY38; see DAY38 section 7: part of the observed hold is the receipt twin's free, item 2's G'')

- Source: `DAY38.md` section 4 (`capture published off the tick (seed): .. settled synchronously by a session retire; the
  settle held the owner thread 2607.12ms`, under the `d2h-delay` red arm); day 25 priced the seam at 0.4 ms on a copy
  that had landed (`DAY25.md` Task 2), and ruling 36 kept the seam at that price. Any long copy-stream work ahead of a
  seed capture (a demote's copies, a promote's fill and spans, a large receipt kernel) makes the source session's
  retire wait for it on the owner thread.
- Acceptance: none registered (the seam's owner hold priced with a capture queued behind a known amount of copy-stream
  work, then a design that does not block the owner there, for example the retiring session's source planes held by
  the pending capture until it lands).
- Status: **pre-registered** (DAY62 section 1: the lines, a price cell with the source and no-source shapes, designs
  R1 and R2 selected by it).

### 14. The host tier's pinned lease frees run `cuMemFreeHost` on the owner thread (found by DAY37 and DAY38)

- Source: `DAY37.md` sections 4 and 8 (`cuMemFreeHost` waits for every stream's queued work in the context and holds every
  other thread's calls meanwhile); `DAY38.md` section 7 (a per-batch pinned twin's free held the owner thread 2944.94
  ms behind a 3 s receipt-stream spin). The contract lease backings (`PinnedBacking::drop`: `event.synchronize()` then
  `free_host`) are freed when a host entry leaves the tier (host LRU eviction at insert, a VERIFY FAILED drop, a tenant
  purge, the latch): 32 leases per 27B entry, each a context-wide wait on the owner thread for whatever the copy and
  receipt streams hold at that moment.
- Acceptance: none registered (the owner's hold priced at a host eviction with known copy-stream and receipt-stream work
  queued, then a design that frees no pinned memory on the serving path, for example the leases returned to a pool the
  governor still charges).
- Also read (DAY51 section 3 reading 5, BOX22, a 9950X host, the base arm): each publication that replaces a host
  entry of the same prompt holds the owner 9.36 to 9.39 ms in `take-back bind and publish` (the replaced entry's heap
  payloads and 32 pinned leases freed on the owner thread) in the chain cell's shape. DAY52 section 3 (the publication
  split, log only, its base arm): 8.6 ms of that is the 32 pinned lease frees (about 270 us per
  `cuMemFreeHost`), the heap payloads 0.02 ms; and P's reserve made those frees about 1 ms slower (item 17).
- Status: **pre-registered** (DAY63 section 1: design L, a pinned backing pool (L1) and the staging set at boot
  (L2), items 14 and 19 together; item 17 re-read on top).

### 15. The D2H receipt kernel's price at long entries (found by DAY38 section 17)

- Source: `DAY38.md` sections 2 and 17 (the survey: one thread per item, about 40 MB/s per thread, `SURVEY G items=32
  bytes_each=4194304 .. median=107.120` ms on the 5090); under G4 every later copy-stream consumer (a capture, a
  restore, a promote's fill and copies) queues behind it, because no side placement off the copy stream stayed flat on
  both cards (sections 13 to 16).
- Acceptance: none registered. A kernel bitwise equal to the program (`memra_tier::contracts::checksum`) on every size
  and offset, priced on both cards at 32 x 60 KiB and 32 x 4 MiB, pre-registered with a bound before its code. Ruling
  54: G''' against G4 at long entries is not an owner choice; the 4096-token cell is pre-registered with both arms and
  runs on the next target card, and the registered rule decides.
- Status: **closed** (DAY43 section 2: `ITEM15 -> G4 STAYS the single placement`; term (1), the chained request that
  waits on a demote, read -0.11 / -0.12 ms against +2.0, while G''' shortened the long copy phase 5.4 ms; the kernel's
  price 104.9 ms per 32 x 4 MiB on the RTX PRO 6000, 107.1 on the 5090).

### 16. The 5090 hump replicate in G4's hot regime (ruling 54)

- Source: `DAY38.md` sections 19 (G4's 5090 (f) FAIL, +0.299 ms, 87 to 88 C, the clock falling) and 20a (the
  base-controlled cell, base +0.072, G4 +0.032, 60 to 78 C); ruling 54.
- Acceptance: none registered. Section 20's cell replicated in the G4 hold's thermal regime (the card driven to that
  regime before the boots, the clock and the temperature recorded per boot), pre-registered before it runs, with a rule
  that places the cause (the regime or the design) either way.
- Status: **pre-registered** (DAY45 section 1: the G4 hold's own warm-up, section 20's eight boots, the regime check and
  the placing rule; `rtx5090-day45/`); waits for the 5090's reset.

### Order of work from day 51 (the lead's order after integ62)

Item 17 first (item 8's remedy; P refuted on day 51, P2 on day 52; blocked on item 14, the lead's ruling), then item 21
(the lead: the server test failure is a defect to place; closed on day 53), then item 10's pricing (day 54), then items
22 and 23 (the lead, after DAY54: a flaky or slow suite hurts every lane's CI; 22 closed on day 55; 24 and 25 found by
it, placed after 23 as the same class), then item 10's fanout design, then items 11 to 14 (item 10 per publisher; item 19 designed with item 14, one lease
design for both directions), then items 18 and 20; the three 5090 cells (item 4's and item 6's halves, item 16) when
the card is reset.

### 17. Remove the helper's heap first touch from the demote's path (item 8's remedy)

- Source: DAY49 section 3 (`-> H attributed`): 38306 minor faults per 156.9 MB copy, 16.09 ms of the helper's 83.50 ms,
  the steady publication one tick-top poll later (wall 101.0 to 101.5 ms against 88.4 to 88.8 where entries free);
  DAY49 section 1 ("the improvement that removes the first touch ... is pre-registered as its own design before its
  code, with its own price clauses").
- Acceptance: DAY51 section 1, (a) to (g).
- Status: **open, design P refuted and reverted** (`a089a5c25`). P (DAY51, `d82738c14`) passed (a) to (f) on the
  target card (BOX22, a 9950X host: the copy -15.2 ms with no faults, the publication -12.4 ms, one poll earlier) and
  FAILED (g) in both orders (DAY51 section 3: `DAY51 P (g) cell=chain order=o1 chain p-minus-base=+1.40 rule <=+1.00
  .. -> FAIL`, o2 +1.54); the chain's extra millisecond sits in the publication's `take-back bind and publish` segment
  (+0.89 / +1.07 ms), unplaced within it. The revision, P2 (DAY52: P with an arming rule, `f9c849389`, over the
  publication split `5990945cd`), passed (a) to (f) on BOX25 (DAY51's machine) and **FAILED (g) in o2** (`DAY52 P2 (g)
  cell=chain order=o2 chain p2-minus-base=+1.15 rule <=+1.00 | first p2-minus-base=+1.11 .. -> FAIL`); **reverted**
  (the P2 commit alone; the split lines stay). The split placed P's millisecond: `DAY52 PLACING -> placed in insert,
  kv`, the replaced twin's 32 pinned lease frees (+0.91 / +0.99 ms; the heap frees flat), and in two of five o2 boots
  P2 kept that cost for the whole boot although it disarmed at the fourth demote. Status: **open, blocked on item 14**
  (proposed): both designs pass their mechanism and every gate and fail only through the pinned lease frees they slow;
  the next revision is re-read on top of item 14's lease design (with 19), where no `cuMemFreeHost` reaches the tick.

### 18. S4's H2D destination digests ride the promote's landing (found by DAY50)

- Source: DAY50 sections 2 and 3: on S4's tree 9 of 90 steady promotes publish one tick after the next tick top (0 of
  90 on G4), and 2 of 27 on the tip; the destination digests of S4's span receipt are queued on the promote's landing
  path (DAY42 section 1 step 5). The delay did not reach S4's (d) (PIN +0.10 ms per order).
- Acceptance: none registered (the destination digests off the landing path, still required before the publication,
  as S2 did for the demote; pre-registered with its own clauses before its code).
- Status: **pre-registered** (DAY64 section 1: per-poll lines, the promote cell, a placing rule, a design per place).

### 19. The host tier's pinned allocations run on the owner thread (found by DAY49)

- Source: DAY49 section 3 readings 2 and 3: a long (5122-token) demote's pre-submit holds the owner 19.66 ms, 19.26 of
  it allocating its 32 pinned KV destinations (151.1 MB, 36896 minor faults) fresh on every demote while no entry frees;
  the first demote of every context holds it about 20 ms allocating the staging set (`spans 19.74` and `20.43 ms`).
  Both are the tenant's tick. On BOX22 (a 9950X host, DAY51 section 3 reading 5) the chain's long pre-submit reads 25.8
  to 26.0 ms, 25.3 to 25.5 of it in the leases. DAY54 section 3 (a 5900XT host): the fanout's insert holds the owner
  2.26 ms (short entries) and 18.25 ms (long) for the evicted entry's demote pre-submit, and each boot's first demote 78
  ms in `spans`.
- Acceptance: none registered (the owner's hold priced at a long demote and at the first demote, then a design that
  allocates no pinned memory on the owner thread's serving path, pre-registered with item 14's: one lease design for
  both directions).
- Status: **pre-registered** (DAY63 section 1: design L, a pinned backing pool (L1) and the staging set at boot
  (L2), items 14 and 19 together; item 17 re-read on top).

### 20. The hash helper's per-payload work runs on one thread (found by DAY49)

- Source: DAY49 section 3: the helper's 83.50 ms per 64-token 27B demote is copy 23.73 plus hash 59.05 over 97
  independent payloads, and at long entries the bind's KV re-hash adds about 56 ms over 32 independent lease views
  (helper 139.80 ms); the publication waits for the whole job (wall 101 ms, 362 ms at long entries). Design T split
  the promote's fill across `min(12, cpus / 2)` threads (item 3); the helper's copies and digests are the same shape of
  work.
- Acceptance: none registered (the same program per payload and per view, bitwise, the digests in the job's order;
  priced on the target card against the tip, the tenant's hump and the promote's PIN inside S's bounds; pre-registered
  before its code).
- Status: **pre-registered** (DAY65 section 1: design T-H, the helper's payloads and views across scoped threads,
  the digest program unchanged).

### 21. `tests::responses_carry_rate_limit_headers_and_slot_frees` failed once under the full server suite (found by DAY52)

- Source: DAY52 section 2: one full `cargo test -p memra-server --lib` run on the P2 tree (under the rig's CPU quota,
  default test threads) read `test tests::responses_carry_rate_limit_headers_and_slot_frees ... FAILED`, panicked at
  `crates/memra-server/src/lib.rs:22740:9`. That line is `assert_eq!(resp.status(), StatusCode::OK);` of the test's
  second request (the streaming `/v1/completions`): the stream was answered with a status other than 200. (Corrected on
  day 53: DAY52 section 2 and this entry first named the next assertion, `stream in flight holds the slot`, from the
  source rather than the panic line.) The rerun of the whole suite passed, and the test alone passed 6 of 6. The panic's
  left and right values were not kept (that run's output was filtered to its summary lines), so the status it read is
  unknown. The test holds `drain_lock()` against its shared-state peers.
- The lead (2026-09-25): a defect to place, not a flake to leave; worked after item 17's reading.
- Acceptance: none registered (a reproduction under the suite's concurrency first, pre-registered, then the placing
  and the fix with their own clauses).
- Status: **closed** (DAY53 section 7: F1 `22f1872d6`, `admission_counters_guard()` takes `drain_lock()` first; 400 of
  400 full suites with the target green and no handler 429, against A''s 7 of 200; H1 placed by intervention). The
  cost, a reading: the suite's median `finished in` 6.47 s to 7.94 s; item 23.

### 22. Three server timing tests fail under CPU starvation (found by DAY53's arm B)

- Source: DAY53 section 2, arm B (40 full `memra-server` suites, `--test-threads 48`, `CPUQuota=400%`):
  `tests::an_extended_stream_commits_prefill_then_injects_the_original_deadline` failed 4 of 40 (`the bridge waited for
  the first-token deadline instead of committing`), `worker::tests::slow_constraint_compile_times_out_while_normal_decode_and_heartbeat_progress`
  4 of 40 (`heartbeat declared stalled: .. no forward progress for 80 ms (.. threshold 50 ms)`; `normal decode stopped
  at 9 steps`), `dsv4_serve::c4_host_budget_tests::coalesced_rows_each_get_their_own_token_once_per_step` 1 of 40. None
  failed in arm A's 40 runs (default threads, `CPUQuota=1200%`). DAY53 section 5 (A', 200 runs of arm A's shape):
  `tests::deep_schema_fails_while_normal_decode_keeps_stepping` 1 of 200 (`bad schema stalled or replaced the normal
  decode`). DAY53 section 7 (F1, 400 runs of arm A's shape): `dsv4_serve::c4_host_budget_tests::coalesced_rows_each_get_their_own_token_once_per_step`
  1 of 400 and `health::tests::no_progress_source_is_the_pre_fix_beat_age_verdict` 1 of 400 (`left: 41 right: 40`).
- Acceptance: none registered (each placed: a real defect, or a wall-clock threshold that a starved runner cannot
  meet; pre-registered after item 21).
- Status: **closed** (DAY55 section 7): T-a (a snapshot sampled the clock twice; fixed in `health.rs`), T-b (tokio's
  paused clock), T-c (the step clock, a test-only health clock, a non-blocking guard) and T-e (the coalescer's window a
  field, two mechanism cells) fixed and accepted: R1 and R2 0 of 100 each, red arms 10 of 10, none red in 200 full
  suites (arm A 100 of 100, arm B 98 of 100 with two other tests, items 24 and 25). T-d not reproduced since its one A'
  failure (R1, R2, 600 later full suites), left unchanged.

### 23. The admission-counter test isolation costs 1.5 s of every full server suite (found by DAY53)

- Source: DAY53 section 7: F1 orders the counter writers behind `drain_lock()`, and the suite's median `finished in`
  went from 6.47 s (A', N=200) to 7.94 s (F1, N=400).
- Acceptance: none registered (the writers isolated without serializing them, for example `reserve_pending_admit`'s
  test entry taking its lane counters as a parameter so a writer never touches the process-global ones; the same 400-run
  shape green for the target and its siblings, the suite's time back to A''s).
- Status: **closed** (DAY56 section 2: F2 `10b9329cc`, the median 6.62 s against 6.70; section 3, from integ65's review:
  the pending-admits gauge the path still wrote, now one `AdmitCounters` pair, F2b `b4d6f95c2`; its deterministic red
  arm `(1, [0, 0, 0])` against `(0, [0, 0, 0])`, 200 of 200 green in R3's shape on the fix).

### 24. `darklane::tests::stop_mode_full_cycle_launch_yield_resume_shutdown` times out under starvation (found by DAY55)

- Source: DAY55 section 7, arm B's shape (100 full suites, `--test-threads 48`, `CPUQuota=400%`): 1 of 100, `timed out
  (3000ms) waiting for: yield to T` (darklane.rs:602).
- Acceptance: none registered (reproduce and place it as DAY55 did: a defect, or a wall-clock bound).
  DAY56 section 2: 1 of 400 in arm A's shape too (run 179).
- Status: **closed** (DAY57 section 2: reproduced 1 of 100 beside sixteen burners; the waits made acknowledgement
  waits under a 30 s hang guard, `a327f486c`; 0 of 100 in R2 and R3 after, the red arm 10 of 10).

### 25. `tests::a_fake_route_memory_door_refuses_defers_and_recovers_through_the_handler` reads a running row after the cancel (found by DAY55)

- Source: DAY55 section 7, arm B's shape: 1 of 100, lib.rs:19036: after the loop saw `cancelled == 1`, `(waiting,
  running, inflight)` read `(0, 1, 0)` against `(0, 0, 0)`. Either the test reads a route book mid-update or the book
  publishes `cancelled` before it takes the row out of `running` (a snapshot a reader could see in production).
- Acceptance: none registered (reproduce, then place: the book's order of updates or the test's read).
  DAY56 section 2: 1 of 400 in arm A's shape too (run 156). Cause read from the code (DAY58 registers it):
  `RouteRun::cancel` counts `cancelled` before its `Drop` takes the row out of `running`, and the snapshot loads
  `running` before `cancelled`, so a reader can see both.
- Status: **closed** (DAY58 section 2: a real ordering defect, reproduced by a stress cell, 34011 and 27753 of
  100,000 snapshots; fixed in `route_telemetry.rs`, `62a29cfe0`: 0 of 100,000 twice, both red arms read it, the test 100
  of 100 beside sixteen burners).

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
