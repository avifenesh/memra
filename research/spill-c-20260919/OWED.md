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
  G2 and `decide-b` ran after the reset and the reboot (queue v9, `DAY51.md` section 5): G1 and G2 PASS, `DAY51
  VERDICT rig=rtx5090 integrity=ok -> door_flat`, a per-card input; earlier: waited on the card's reset, queued
  (`rtx5090-queue-v5-20260925.sh`). Both cards have read; what remains is the owner's promotion call (2026-10-04). The
  lead accepted `door_wins` on the corrected reading (integ60 ruling). The gap to REF and its tuning are C11.

## C10. `MEMRA_MOE_PREFETCH=1`'s deciding cell (lead integ60 owed item 1, decide-by 2026-10-04)

- **Source.** The lead's integ60 resume: REF beat both the door and the legacy on the target card (`DAY51.md`
  section 4); its `docs/FLAGS.md` row lacked a decide-by. "Pre-register its deciding cell (correctness gates plus the
  timing A/B, both orders, N>=5, both cards) and add the decide-by date to its row. The winner becomes the per-card
  default under the flags doctrine; the owner decides promotions."
- **Acceptance.** `DAY59.md` section 1 (G1 tapes at two shapes, G2 run-spec K=1..8, G3 serving byte-equality; `pftime`
  and `pfnaked`, 20 runs each, both orders; `pf_wins`, `pf_flat`, `pf_loses`); the row's decide-by 2026-10-04
  (`97bd6d889`, `55fbad45a`).
- **Status.** Target card done (BOX12, `DAY59.md` section 2): G1, G2, G3 PASS, `DAY59 VERDICT rig=pro-single
  shape=pftime integrity=ok -> pf_wins`, `DAY59 VERDICT rig=pro-single shape=pfnaked integrity=ok -> pf_flat`: it
  qualifies as the target card's naked default; the promotion is the owner's call. The RTX 5090 (queue v9,
  `DAY59.md` section 3): G1 to G3 PASS, `pf_wins` under pressure, `pf_flat` naked: it qualifies there too.

## C11. The door's gap to REF, attributed and tuned (lead integ60 owed item 2)

- **Source.** The lead's integ60 resume: "Attribute the door's gap to REF stage by stage and tune the door to match
  or beat it ..., each improvement with its own pre-registration and cell. If the tuned door still loses to REF,
  record it plainly; the owner reads both against the 2026-10-04 date."
- **Acceptance.** `DAY60.md` (the log-only `--moe-dispatch-clock`, cell `gap`, readings R1 to R4); `DAY61.md` (the
  CPU profile, I11 and I12 with per-change CPU gates and ladder, cell `i11` with its rules).
- **Status.** The instrument landed (`fec3c582f`). The CPU profile put about 5.4 us on each host-hit prefetch
  cycle; I11 changes 1 to 5 took it to 3.6 us on the local CPU (a sixth change read flat and was reverted,
  `DAY61.md` section 2a); I12 landed (`117302725`). Target card (BOX12): `DAY60 GAP rig=pro-single integrity=ok
  window: wall_gap=+0.437 cpu_gap=+0.699 top=prefetch_ns cpu_side; gen: wall_gap=+0.688 cpu_gap=+1.478 top=prefetch_ns
  cpu_side` (R2 `over_bound` by 0.003 ms per token at the printed resolution) and `DAY61 VERDICT rig=pro-single
  integrity=ok i11=improves i12=flat door=i12 vs_ref=loses (window: i11=improves i12=flat vs_ref=loses)`: the tuned
  door still loses to REF (+0.34 ms per generated token, +0.25 per window token, from +0.69 and +0.44). The RTX
  5090's cells wait on its reset (queue v6). Day 63 (`DAY63.md`): the owner demand split (log only, `ad73d242c`); I13
  (the governor without temporaries, one body per lease, the retire side in one lookup, the SLRU on an Fx hasher,
  `9bbab60ed` to `c9379c051`) takes the host-hit prefetch cycle from 3569 to 2940 ns on the local CPU in one window;
  on the target card (BOX13, `DAY63.md` section 4) `DAY63 VERDICT rig=pro-single integrity=ok i13=improves door=i13
  vs_ref=loses (window: i13=improves vs_ref=loses)`: the door at 0.264 s gen-only and 0.232 s window against REF's
  0.255 and 0.226 (+0.28 and +0.19 ms per token); the RTX 5090's cell is queued (queue v7). Day 64 (`DAY64.md`): I14
  (the catalog and the host cache hashed, `8e7faf4ec`, `83f03d9b7`) and I15 (one ticket per prefetched expert,
  `2243b1fe2`), CPU gates green; the host-hit prefetch per block 2995 ns at I13 to 1997 grouped in one window; the card
  cell `i15` ran on BOX15 (a Ryzen 9 9950X host; `DAY64.md` section 4): `DAY64 VERDICT rig=pro-single integrity=ok
  i14=flat i15=flat door=i15 vs_ref=matches (...)`, as the rule reads it, with noise terms (0.053 to 0.074 s) set by
  the door's per-boot host-CPU bimodality on that host (fast boots near REF, slow ones 62 ms behind); the rerun on the
  285K class with a pre-registered admissibility clause (IQR at most 0.005 s per arm) is prepared (`day64b-box.sh`);
  its first attempt (BOX16) was void, nvcc segfaulted in a build and no cell ran (`DAY64.md` section 5a). The rerun
  ran on BOX14 (`DAY64.md` section 5b): `DAY64 ADMISSIBILITY ... -> admissible`, `DAY64 VERDICT rig=pro-single
  integrity=ok i14=flat i15=flat door=i15 vs_ref=loses (window: i14=flat i15=flat vs_ref=loses)`: both steps stay;
  the door 0.28 and 0.19 ms per token behind REF, as at I13, while its CPU-side work halved, so the gap is off the
  CPU-side door work. Day 72 (`DAY72.md`): the gap re-attributed at I15 before any improvement, cell `gap15` (DAY60's
  clocked arms plus REF and the door under Nsight Systems), sitting ready (`day72-box.sh`, the 285K class); the RTX
  5090's half ran (queue v10, `DAY72.md` section 2): `admissible=no`, recorded, deciding nothing (beside it the
  door's GPU idles 0.16 ms per window token more than REF's with the same kernels and copies); the 5090's DAY60 gap
  (`cpu_side`), i11 (`flat`, `flat`), i13 (`flat`) read, i15 `void (inadmissible)`; queue v11 reruns i15 and gap15 on
  the 5090 as new holds (i15 void with a foreign compute app on the card; gap15 inadmissible again). The target card
  (BOX29, `DAY72.md` section 3): `DAY72 GAP15 VERDICT rig=pro-single integrity=ok admissible=yes partA=cpu_side
  partB=gpu_stall`: the door's GPU work equals REF's and its GPU waits on the door's prefetch path. Day 75
  (`DAY75.md`): I16 (`eeacfaf50`, the door's next-expert prefetch issued after the current expert's launch) read on
  BOX29 `DAY75 VERDICT rig=pro-single integrity=ok i16=regresses door=i15 vs_ref=loses` (admissible; the next
  expert's copy exposed) and reverted (`26aa54c12`); the 5090's `i16` inadmissible. Day 77 (`DAY77.md`): I17
  (`d4ab19f1d`, the group's residency and staging each in one owner-registry entry, the same program), CPU gates
  green, the profile 70 to 150 ns per block below I15; on BOX32 `DAY77 VERDICT rig=pro-single integrity=ok i17=flat
  door=i17 vs_ref=loses` (stays; the door 10 ms over 32 tokens behind REF); the 5090's `i17` inadmissible. Day 79
  (`DAY79.md`): I18 (`c7294b912`, a bank ticket's records by position instead of a set and a map of cloned ids, the
  host-hit demand's largest part), CPU gates green, the profile about 150 ns per block below I17; on BOX34 `DAY79
  VERDICT rig=pro-single integrity=ok i18=flat door=i18 vs_ref=loses (window: i18=flat vs_ref=matches)` (stays); the
  5090's `i18` inadmissible. Day 81 (`DAY81.md`): I19 (the next-but-one expert prefetched after the current
  expert's accumulate) withdrawn before any card, its order is I18's (the local check's host demand sequence
  byte-for-byte I18's), reverted (`5f8824d6b`). Day 82 (`DAY82.md`): an allocation census of the grouped host-hit
  cycle chose I20 (`8efea3a54`, the bank's stage without its two `BudgetRequest` clones: 22 to 16 allocations per
  cycle), CPU gates green, P9 unresolved on the local CPU; the card cell `i20` adds I15 as a fifth arm (I20 against
  I15 printed beside the registered step, deciding nothing); sitting ready (`day82-box.sh`, the 285K class, then a
  9950X).

## C12. The door's sensitivity to its owner thread's host placement (the 9950X class)

- **Source.** `DAY64.md` section 4: on a Ryzen 9 9950X host every door arm is bimodal by boot (the CPU-side stage
  parts about 1.85 times larger in the slow boots, the GPU copies unchanged), REF is not; the runner's 12 pinned CPUs
  span both core complexes.
- **Work.** Place the cause with the placement sampler's receipts (`ev/placement.tsv`, from the rerun onward) and a
  pre-registered cell on a 9950X-class host that pins the run to one core complex against the two-complex pin; read
  beside lane A's owed 9950X-class fill reading.
- **Status.** `DAY65.md`'s cell `pin` ran on BOX17 (BOX15's machine): `DAY65 PIN VERDICT rig=pro-single integrity=ok ->
  pin_does_not placement_does_not_track; ...`: the L3-domain hypothesis refuted (slow in 8 of 10 boots on one domain
  as on two, the owner thread home in every sample). The next reading is registered (`DAY66.md`, cell `freq`: the
  owner core's clock and the process's huge-page backing per boot) ran on BOX18 (BOX15's machine): `DAY66 FREQ VERDICT
  rig=pro-single integrity=ok -> clock_does_not_track thp_does_not_track` (about 5720 MHz and no huge pages in every
  boot). `DAY67.md`'s cell `probe` ran on BOX19: `DAY67 PROBE VERDICT rig=pro-single integrity=ok -> none_tracks` by
  the strict rule, while 19 of 20 door boots split by mode on the compute, L1 and L2 chases (about 1.4x, DRAM
  unchanged: a lower effective core clock than the requested one DAY66 read), the slow state ending within the probe
  in 9 of 10 slow boots. `DAY68.md`'s cell `eclock` ran on BOX21: `DAY68 ECLOCK VERDICT rig=pro-single integrity=ok ->
  gate_tracks`: the slow state is in place before the decode and absent at the start, ends by itself about a second
  after the window, with the reported clock and the temperature unchanged. `DAY70.md`'s cell `sched` ran on BOX23:
  `DAY70 SCHED VERDICT rig=pro-single integrity=ok -> none_tracks` (no run-queue wait, nothing else the scheduler
  accounts on the owner's CPU or its sibling, the owner on its CPU for the whole span; this kernel accounts no
  hard-interrupt time; DAY67's DRAM reading corrected: 1.65x slower in the two boots whose slow state lasted through
  it). Next: `DAY71.md`, one cell `core` for the remaining candidates at once (the core's TSC, APERF and MPERF around
  the gate probe through `RDPRU`, every CPU's interrupt and softirq counts, the sibling's idle states, the package's
  energy and sensors, the card's PCIe traffic), on BOX15's machine and on a second 9950X machine for the class
  question. The probe change landed (`6bad38150`, binary `p71`). Machine `b` (BOX24) ran: `DAY71 CORE VERDICT
  rig=pro-single-b integrity=ok -> not_reproduced` (19 slow, 1 fast); beside it the door's gate probe at 12.4 to 14.2
  cycles per step at a full clock with MPERF over TSC 1.000, and compaction failing to migrate 70,000 to 78,000 pages
  a second through every slow door span, none in REF's (`DAY71.md` section 2). A reader defect fixed and `intr_rate`
  and `migrate_fail` registered before BOX15's half (section 3). Day 73 (`DAY73.md`): the compaction question as its own
  cell `compact` on any 9950X machine (the state per run from the gate probe, compaction per span, the process's
  pinned and huge-page memory, buddyinfo, system calls under strace) ran on BOX31 (a 9950X, as registered): `DAY73
  COMPACT VERDICT rig=box31 integrity=ok -> not_reproduced` (no slow run, no compaction in any span); a diagnostic on
  BOX30 (a 9950X3D2, outside the class) read its one slow door run as the sitting's only span with compaction. Day 74
  (`DAY74.md`): compaction induced on purpose in half the runs of REF and the door, cell `induce`, sitting ready
  (`day74-box.sh`): BOX31 `not_run` (page cache), then `not_induced` (every huge-page burst came back whole); BOX29
  `void` (no `RDPRU` on Intel). `DAY74.md` section 4 registers `induce-b` (all but 2 GiB of free memory fragmented,
  the artifact reread before every run, the wall-time state on Intel), ready (`day74b-box.sh`); on BOX29 (285K):
  `not_induced` by rule (REF+I 0 of 6), and beside it the 285K's first slow state: the 3 door+I runs with compaction
  in their span, 1.48x slower per chain step from the gate on, REF never compacting (`DAY74.md` section 5). Day 76
  (`DAY76.md`): does the door's one large pinned allocation draw the compaction: the diagnostic flag
  `--expert-bank-pool-chunk-bytes` (`7a162e6b7`, decide-by 2026-10-10) and cell `chunk` (REF+I, D+I, DC+I under the
  fragmentation) on BOX32: `DAY76 CHUNK VERDICT rig=box32-285k integrity=ok -> chunk_does_not`; beside it the slow
  runs are those whose compaction fails to migrate nearly every page it isolates (`DAY76.md` section 2). Day 78
  (`DAY78.md`): which of the door's pages: the diagnostic flag `--expert-bank-pool-pageable` (`a1786bc32`, decide-by
  2026-10-10; a local GPU check reads MATCH and the door's tape), cell `pages` (REF+I, D+I, DP+I under the
  fragmentation, `fail_heavy` per span, a per-mapping page census that uses frame numbers and `kpageflags` where the
  container allows) on BOX34: `DAY78 PAGES VERDICT rig=box34-285k integrity=ok -> pool_draws` (fail_heavy di 8 of 8,
  dpi 0 of 8); the census shows the pinned pool is a shared `/dev/zero` (shmem) mapping, whose pinned pages compaction
  isolates and cannot move. Day 80 (`DAY80.md`): the fix, a pool of private anonymous memory pinned with
  `cuMemHostRegister` (`57086efc8`, `--expert-bank-pool-registered`, decide-by 2026-10-10), cells `regtime` and
  `regpool` ready (`day80-box.sh`, the 285K and a 9950X); on BOX37 (285K) `registered_clears` and `dr=flat`
  (admissible); the local 5090 check MATCH with the same tape; the 5090's `regtime` inadmissible; on BOX38 (9950X)
  `regtime` admissible `dr=flat`, `no_natural_slow`, `regpool` `not_run` (a valid half by section 3). Open: the
  owner's question (`DAY80.md` section 4a), then the flip and its qualification sitting; `induce-b` on a 9950X with at least 98 GiB
  `MemFree`; DAY71's default half on BOX15's machine, then the class line.

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
  2026-10-05. Day 62 (`DAY62.md`): current through lane A day 41 and ruling 54 (`DAY62 PACKET LINES checked=63
  missing=0 -> PASS`). Day 69 (`DAY69.md`): current through lane A day 48 and ruling 57 (`DAY69 PACKET LINES checked=32
  missing=0 -> PASS`). At 2026-09-24 21:10Z lane A's days 37 to 41 were in flight on its branch (DAY38's G'' and G''' sittings,
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
  `c4e18a4e3`, `62e848b1f`). The RTX 5090 (queue v9, `DAY56.md` section 4):
  `DAY56 DFLASH TAIL rig=rtx5090 -> PASS`. Follow-ups C5b and C5c
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
