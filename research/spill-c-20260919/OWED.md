# WP-C owed ledger (written 2026-09-24, day 40, before any code)

Every open item owed to or by this lane, with its source, its acceptance clause if one is registered, and
its status. Owner order, 2026-09-24, verbatim: "every improvment and tunning should be done, no shortcut or
fast path". Nothing here is parked, waived or optional: an item is open, closed (with the receipt that closed
it), another lane's (listed so the ledger is complete, not worked here), or an owner decision. The ledger is
worked in the order below, nearest decide-by first; each item gets its own `DAYnn.md` from day 40 on.

Resync for this ledger: lane tip `a7608c268` (day 39) fast-forwarded to `origin/main` `d61012658` (#716); main
carries integ48 (C day 39) and every lead integ through integ58 (A day 36, ruling 53). Read: repo and user
`CLAUDE.md`, `STATE.md`, `DAY18.md`, `DAY39.md`, `MOE-SLOT-CACHE-DOOR.md`, `DOOR-DECISION-PACKET.md` item 3
and item 7, `HOSTPREFIX-DOOR.md` section D, the lead record `research/spill-lead-20260919/INTEGRATION-DAY12.md`
(integ25, integ27, integ48 ruling 43, integ49 ruling 44, integ58 ruling 53) and `HANDOVER-20260920.md` (ruling 8).

## C1. The MoE slot cache door: tune its program, then the deciding cell (decide-by 2026-10-04)

- **Source.** `MOE-SLOT-CACHE-DOOR.md` item 4; `DAY18.md` cell `overlap`, verbatim `OVERLAP-PAIR rule
  decode_off_s=0.408 decode_on_s=2.343 decode_ratio=5.743 ratio_o1=5.694 ratio_o2=5.833 ...
  door_cost_ms_per_decode_token=60.47 ... -> sync_miss_path_slower` (integ25); lead ruling 8 (a CLI door carries
  its decide-by in its design doc); the owner order above; repo `CLAUDE.md` "Hy3 spilling" (one
  storage-to-compute pipeline: positioned reads, bounded pinned host buffers, residency caching, asynchronous
  prefetch and overlap, PCIe transfer, kernels; H2D and cache publication on the CUDA owner thread; the stages
  measured together).
- **Work, in order.** (a) Attribute the 60 ms per decode token stage by stage: the per-lease compute-stream
  drains in `admit_banked` and `admit_native`, the host demand (host-tier hit or physical read, the per-record
  SHA-256 verify, the host-slot allocation), the per-demand trace print, the GPU slot reserve and eviction, the
  H2D, `finish`. (b) Design, pre-register and land the improvements one at a time, each with its own receipt,
  never trading correctness: the door's identity (one token tape across arms) and integrity checks (per-record
  checksum, lease identity, publication after a proven copy) stay in every cell. (c) The pre-registered deciding
  cell on the RTX 5090 and on the target card (one RTX PRO 6000 Blackwell Workstation Edition): balanced
  interleaved A/B against the legacy SLRU slot cache, both orders, N>=5 per arm per order, raw hashes, 250 ms
  telemetry. (d) If the tuned door loses: the deletion in this lane ("Decision at decide-by" list of
  `MOE-SLOT-CACHE-DOOR.md`), receipts banked, the verdict in the `docs/FLAGS.md` "Removed doors" ledger. If it
  wins: the owner's promotion call, and C2 becomes the promotion work.
- **Acceptance.** Registered per step in its `DAYnn.md` before code or boots; the deciding cell's rule is
  registered before either card runs it.
- **Status.** Step (a) done day 40 on both cards (`DAY40 ATTRIB ... top=demand`, `rig=rtx5090` and `rig=pro-single`).
  Step (b) done: ten rungs and two fixes, every RTX 5090 verdict in its DAY file (days 43 to 50, 57, 58; I6, I8 and I5
  fixed rather than reverted), the target card's ladder read (`DAY52.md` section 11). Step (c): the target card's
  deciding verdict `DAY51 VERDICT rig=pro-single integrity=ok -> door_wins` (`decide-b`; the first `decide` void on a
  trace term I10 contradicts), with REF (the legacy's own prefetch) faster than the door, reported. The RTX 5090's G1,
  G2 and `decide-b` wait on the card's reset (`rtx5090-fault-20260925/`: `GPU requires reset` at 01:25:23Z), queued
  (`rtx5090-queue-v5-20260925.sh`). Open until that cell reads; then the owner's promotion call (2026-10-04).

## C2. The slot cache door's promotion prerequisites (the door doc's pending items 1, 2, 3, 5, 6)

- **Source.** `MOE-SLOT-CACHE-DOOR.md` "What is pending": item 1 (the bank owner under PP: registry is
  thread-local, a stage thread refuses `WrongOwner`), item 2 (mixed-layout budgets need per-class minima; the
  GPU budget reads free VRAM at install time), item 3 (the hash lock to one artifact; scale admission, whose native
  refusal cell needs a scale-bearing artifact), item 5 (LFU, size-aware classes, pread backends and frozen
  residency refuse with the bank installed), item 6 (no `memra-server` installer, so no serving-shape bit-identity
  gate, banked against native, solo against batched). "Decision at decide-by": either items 1 to 3 land with their
  gates and the door is promoted, or the door is deleted.
- **Dependency, stated.** These are the door's promotion work. They are executed after C1's deciding cell if the
  tuned door wins it; if it loses, door hygiene deletes the door and these items with it in the same lane. This is
  the door doc's own decision rule, not a deferral.
- **Status.** Open, sequenced after C1(c).

## C3. The contracts door decision packet kept current (owner decision 2026-10-05)

- **Source.** `DOOR-DECISION-PACKET.md` (status line: "after days 31, 33, A day 27, A days 28 and 29, and C days
  38 and 39"); the lead's integ49 to integ58 records (A day 31 items 1a to 1d, ruling 44; A day 32 the H2D half,
  design H; A days 33 and 34 designs F and K; A day 35 F settled KEEP, M refuted and reverted, M' PASS on the 5090;
  A day 36 the D2D half closes, M' PASS on the target card, ruling 53).
- **Work.** Read A days 31 to 36 into the packet's sections 2, 4, 5 item 7 and 6 and the appendix, verbatim lines
  only, no recommendation; `HOSTPREFIX-DOOR.md` section E the same.
- **Status.** Current through lane A day 36 and ruling 53 (day 42, `DAY42.md`: `DAY42 PACKET LINES checked=68
  missing=0 -> PASS`). Stays open until the review: every later receipt bearing on the door is read in before
  2026-10-05. At 2026-09-24 21:10Z lane A's days 37 to 41 are in flight on its branch (DAY38's G'' and G''' sittings,
  DAY39's design T, DAY40's span-receipt survey, DAY41's design K red arms); they are read in when they land. Item 3's
  open question (which slice moved the demote's landing) answered from this lane's day 54 and read in verbatim.

## C4. The double-park slice (the contracts door, 2026-10-05)

- **Source.** `DOOR-DECISION-PACKET.md` section 5 item 3 and section 6: "Which slice between `0713c1a79` and the
  integ38 tip moved the copy's landing is still not determined" (the inline demote published `after 1 poll(s)` in
  a median 22.4 ms on the integ38 tree against 97.0 on the day-23 tree); `DAY29.md`.
- **Work.** A pre-registered bisect over the slices between `0713c1a79` and the integ38 tip (`643ecbb28`), the
  day-29 cell shape on each slice's binary, the reading the demote's landing time.
- **Acceptance.** Registered in `DAY54.md` (the eight slice binaries, day 29's promote boot, ten boots per binary
  interleaved both orders, the landing and park readings and the rule).
- **Status.** CLOSED day 54 on the target card: `DAY54 VERDICT -> moved_at s5=58b814abe` (#638, integ37's
  parked-only wait moves the demote's landing 89.1 to 26.2 ms and brings the second park), read into
  `DOOR-DECISION-PACKET.md` item 3.

## C5. The DFlash tail slice of the contracts door (`HOSTPREFIX-DOOR.md` section D item 2)

- **Source.** `DAY19.md` Task 3 (pre-registration: a third entry class with the drafter's export-directory byte
  manifest as its artifact identity, `DflashCfg` as its plan, the f32 tail through the contract D2H route; the
  identity gate OFF against ON with `MEMRA_DSPARK_SPEC=1` and the drafter, `ALL GREEN` both arms, equal demote
  bytes, one receipt per tail plane per draft layer, a promote that re-arms the drafter).
- **Work.** Engine code first (the manifest digest, the class, the tail through the route), then the identity gate
  grows a drafter arm, then the cells on the 5090 (drafter export present locally) and the target card.
- **Acceptance.** `DAY19.md` Task 3 "Rule", as registered; the design and cells in `DAY56.md`.
- **Status.** Target card PASS: `DAY56 DFLASH TAIL rig=pro-single -> PASS` (attempt 4, `DAY56.md` section 3, after
  three attempts that found a cell shape error and two defects of `MEMRA_DSPARK_PARTIAL_RESTORE`, both fixed:
  `c4e18a4e3`, `62e848b1f`). The RTX 5090's attempt 4 waits on the card's reset (queue v5). Follow-ups C5b and C5c
  (the tail's hash on the helper, the tail through the contract route as spans) edit lane A's in-flight helper and
  span code (A's DAY38 design G''' and its receipt streams); sequenced after that lands, through the lead. Open, not
  waived.

## C6. Verify digest v3 (the draft plane inside `MEMRA_KV_HOST_VERIFY`)

- **Source.** `HOSTPREFIX-DOOR.md` section D item 3; the day-14 finding "The verify arm is blind to the draft
  plane" (a `memra-prefix-split-state-v3` digest covering the draft plane changes the digest strings the OFF arm
  prints in `VERIFY FAILED` lines, so it is its own slice and gate line).
- **Acceptance.** Registered in `DAY53.md` (sections 1 and 1a).
- **Status.** CLOSED: ALL GREEN on both cards (RTX 5090 `DAY53.md` section 3; target card section 4, after a first
  attempt that failed only on a wrong literal in the gate, section 2).

## C7. The arena lease handoff under the contracts door (`HOSTPREFIX-DOOR.md` section D item 1)

- **Source.** `DAY19.md` (scoped: an arena-backed lease type in `tier_transfer.rs`, worker charge-once accounting,
  a budget ruling); lead ruling 28 (integ27): "the arena handoff stays scoped until the HOSTPREFIX decide-by
  review; the budget question (one pinned budget or two) is decided there with the door."
- **Status.** Owner decision first (the budget question at the 2026-10-05 review); the build follows the ruling.

## C8. An always-admitted prime arm on the RTX 5090 class

- **Source.** `DAY35.md` (pass 1's prime arm inadmissible: the memory admission refused 7 of 10 intruders beside a
  co-tenant, `[admit-oom] capacity reject ... HTTP 400 context_length_exceeded`), `DAY37.md` section 7 (the prime
  control's low first pass, cause not separated), `DOOR-DECISION-PACKET.md` item 7.
- **Work.** A new pre-registration (a prime the memory admission admits in every run: a shorter prime or another
  `MEMRA_CTX`), then the 5090 cell.
- **Status.** CLOSED day 55: `DAY55 VERDICT -> prime_always_admitted` (every boot guard=clean, no co-tenant, no
  admission defer or reject in either prime boot).

## C9. The 9B entry's conv, ssm and hidden split (closes on a reading)

- **Source.** `DAY38.md` section 4 (bounds from the banked logs: KV planes 950,272 B exactly, 256 B of token ids,
  logits in (950,272, 1,099,744) B, conv + ssm + hidden in [52,599,728, 52,699,728) B; the per-class line is engine
  code); `DAY39.md` section 7.
- **What changed.** A day 31 item 1d landed that line (`by slot class: conv C (B B), ssm S (B B), hidden H (B B),
  logits L (B B)` on `demote copy complete off the tick`, ruling 44), and A's DAY31 reads the 9B plain lines as conv
  24 (2,359,296 B), ssm 24 (50,331,648 B), hidden 1 (0 B), logits 1 (993,280 B), the draft-bearing lines hidden
  16,384 B (`research/spill-a-20260919/DAY31.md` 1d table).
- **Work.** The reconciliation against day 38's registered bounds, arithmetic on the banked lines; no new cell.
- **Status.** CLOSED day 41 (`DAY41.md`): `DAY41 9B SPLIT plain_tuples=1 -> PASS` over 1,120 banked lines; conv
  2,359,296 B, ssm 50,331,648 B, hidden 0 B (plain) or 16,384 B (draft-bearing), logits 993,280 B.

## Closed items this ledger records

- **The server accepting the door's flag silently** (`DAY18.md` item 6 hygiene finding): closed by memra#617 (C day
  19, integ27): `memra-server` refuses an unknown argument at boot, `SERVERDOOR19 rule flag_exit=2 flag_refused=True
  ...` on the local 5090; merged by the lead on the binary-boot tests.
- **The 5090 demote-class tenant-stall cells and the target card's** (days 35, 37, 39; rulings 43 and 41): closed
  as registered.

## Lane A's items (listed for completeness; A is working them, this lane does not)

- Why b1 shows no first-touch pre-submit step in the day-39 cell (a per-demote allocation line), ruling 43.
- The b2 helper's `hashed_in` rise of 31.6 to 34.0 ms (off the tick, unattributed), ruling 43.
- The promote class's tick 2 on b1 and b2 on the target card (`not_defined`), ruling 43.
- Move 2 owed item 1's remaining terms after ruling 53: hash 1 on the owner thread, the fill on slower CPUs, the
  strong-form receipt; the split of the door's second stretched tick into the owner thread's segments on the 5090
  class (a timestamped segment line, the door's owner-thread code).

## Owner decisions (not this lane's to make; receipts in place per the items above)

- 2026-10-04: the MoE slot cache door (C1's deciding cell is its receipt) and the VMM door (lane B's).
- 2026-10-05: the contracts door (C3 to C7 feed it).
- 2026-10-06: the park door (lane B's).
- 2026-10-07: `MEMRA_ADMIT_BY_MEMORY` (lane B's packet, integ56).
