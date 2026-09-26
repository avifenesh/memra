# WP-A day 59: OWED item 10, the fanout publisher (an attribution first, then the design it selects)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, on `bff73418f` (origin/main `4b24740f4`, integ64, merged). The
lead's order: the fanout publisher's design from DAY54's price, pre-registered.

## 1. Pre-registration (committed before any code)

**The price, and what it is made of** (DAY54 section 3, a 5900XT host, the 27B, four identical 97-token prompts). The
fanout adds 6.50 ms to the tenant's stall over one prime of the same prompt, in both orders. Its own on-tick parts, from
the lines: the leader's snapshot 0.72 ms and three sibling restores 1.19 to 1.24 ms. The insert's 2.24 ms is the evicted
entry's demote pre-submit (item 19, not the fanout's). The rest, about 2.3 ms, is not the publisher: the dedup line
covers the prefix exactly (`prefix=97` of 97 prompt tokens), so no member primes a suffix, and the next decode step runs
five sessions (the tenant and the four) where the control runs two. So the publisher's share is the snapshot and the
restores, about 2.0 ms of owner time, above DAY54's 1.0 ms rule.

**What those 2.0 ms are is not known**, and the two candidate designs differ by it:

- the device-call enqueue cost: a snapshot allocates 64 fresh KV planes and 96 recurrent planes and issues a copy or a
  clone for each (about 160 calls), and each restore issues about 160 copies into a sibling's cache; or
- the allocations: the fresh planes come from the device pool one call at a time; or
- the copies' own GPU time on the owner stream (160 MB per snapshot and per restore), which the owner thread's wall
  clock sees only where it synchronizes.

**Step 1, the attribution (log only; its own commit).** `prefix_snapshot` and `prefix_restore` time their parts on the
owner thread: allocation (count, ms), copies and clones issued (count, ms), and the rest; the fanout's on-tick line
gains `snapshot (alloc A ms over N, copies C ms over M)` and the same per restore. A test-only census pins that the
parts only time and print (the same calls in the same order).

**The cell.** The local RTX 5090 (the development rig, under `/tmp/memra-5090.lock`, a bounded wait; lanes B and C
queue on it): DAY54's `fanout` against `prime-short` at 72 words, five boots per mode per order, the 27B NVFP4 MTP
artifact the rig carries, door ON. A reading that selects the design; the design's price is decided on the target card.

**The selection rule, stated before the cell.** On the fanout mode's steady ticks (the second and later of each boot):

- the calls (copies and clones) at least 60% of the snapshot-plus-restores owner time: **design B1**, the snapshot's and
  each restore's copies as one batched device copy (one launch over the `(source, destination, bytes)` items, the span
  kernels' shape), the same bytes in the same destinations;
- the allocations at least 60%: **design B2**, the snapshot's fresh planes taken as one pool reservation instead of one
  allocation per plane;
- neither: both parts are priced and the larger is designed first, stated with its numbers.

**The design's acceptance** (registered now, for whichever the rule selects; the target card decides):

- (a) the entry's planes and every sibling's restored cache bitwise equal to the old program's (a GPU cell comparing
  both programs' bytes on the same inputs), the identity gate door ON default and plain, the hit gate door ON;
- (b) the fanout's own owner time (snapshot plus restores) at most half of the old program's, per order;
- (c) the tenant's stall, fanout minus prime, at least 1.0 ms smaller than the old program's in both orders (DAY54 read
  +6.50 on the old program);
- (d) the fanout members' e2e medians no worse than the old program's plus 1.0 ms.

**What each card decides.** The 5090 selects the design (a reading); the target card decides its adoption (b to d).

**Budget.** 0.5 agent-day: the lines 0.1, the 5090 reading 0.1, the design and its GPU cell 0.2, the target sitting
0.1.

## 2. Step 1 as built (`8b5e5e213`), and the 5090 sitting prepared

- The split: `prefix_copy_timed(kind, f)` wraps each device call of `prefix_snapshot` (the KV planes' `alloc_u8` and
  `copy_u8_into`, the recurrent planes' `clone_dtod`) and of `prefix_restore_at` (the KV `copy_u8_into`, the length
  `set_i32_one`, the recurrent `copy_into`), accumulating owner time and counts on the calling thread; the fanout takes
  the split after the snapshot and after the restores and prints `[prefix-dedup] on-tick split: ..` under the door.
  Census `day59_the_fanout_copy_split_is_log_only` (the calls and their order unchanged; two takes; no decision).
  Server lib `933 passed; 0 failed; 25 ignored`; clippy `-D warnings`; fmt.
- Read from the code while building: each restore also writes each KV layer's length to the device with
  `set_i32_one`, a 4-byte host-to-device copy from pageable memory. **Before the cell runs**, the rule's "calls"
  includes these length sets (a restore's device calls), and the reader prints them apart.
- The sitting `rtx5090-day59/card-run.sh <out> <model> <memra-server>` (one bounded hold of `/tmp/memra-5090.lock`, 60 x
  120 s; the card idle with 20000 MiB free; DAY54's short cell shape, 20 boots) and the reader `day59-reading.py`, both
  written before the cell runs. The model is the local copy of the target's artifact
  (`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, sha256 `1facf36c2db359dc..`).

## 3. The first 5090 run, cancelled; the re-run

- The first run (on `8b5e5e213`'s binary) waited 23 of its 60 lock attempts behind lanes B and C and never held the
  card; it was cancelled when the lead interrupted with item 23's addendum, whose builds would have perturbed an
  owner-time cell. Banked as `rtx5090-day59/cell-cancelled/`.
- The re-run is the same cell on the tip's binary (`6f844ccf0c00d9a0`, tree `071e1126a`, whose fanout path equals
  `8b5e5e213`'s), with one addition recorded before it runs: each boot's `BOOT.txt` also records the host's load
  average at the boot's start (the rig is shared with lanes B and C, and the cell reads owner-thread time), a reading.

## 4. A target-card twin of the cell, prepared (`pro-single-day59/`)

- The 5090 cell measures owner-thread time on a rig whose lock and CPUs lanes B and C share (the re-run waits behind
  their queues; their CPU work runs beside any cell that gets the card). So the same cell is prepared for a quiet target
  card: `build.sh <tip>`, then `driver.sh` (DAY54's short paired cell, 20 boots, one collector hold, then
  `day59-reading.py` over `/root/spill-receipts/a-d59`). The rule is section 1's and is read on whichever card runs
  first; if both run, the target card's selection is the one that decides and the 5090's is a reading beside it
  (stated now, before either result).

## 5. The 5090 re-run: NOT RUN

- The re-run (`rtx5090-day59/cell-not-run/`) waited through the lock queue behind lanes B and C, took the hold, and
  then read a compute app on the card for all 15 idle checks (`card not idle under the hold (apps=[279749] free=22581
  MiB)`, 03:09Z to 03:23Z) and stopped as registered: `NOT RUN: the card never went idle under the hold`. The process is
  not this lane's and was not touched. No boot ran; nothing is read.
- The attribution waits for the target-card twin (section 4), whose selection decides.

## 6. The target twin: read as registered, DESIGN B1

- Run by the lead on one RTX PRO 6000 Blackwell Workstation card (a 16-core, 123 GB host), binary built at `d4117fb2d`
  (whose fanout path is `8b5e5e213`'s), 03:20Z, one collector hold, `ab-short-cell rc=0`. Mirror
  `pro-single-day59/box/`: 103 receipts, sha256-checked against the box manifest. 20 boots, none failed; the boots'
  start temperatures 30 C to 65 C (the 250 ms telemetry is in `ab-short-cell/command.gpu.csv`).
- Verbatim (`box/reading-day59.log`), N=45 steady fanout ticks per order:

      DAY59 READING order=o1 N=45 snapshot=0.50 restores=0.84 | alloc=0.06 copies=0.71 clones=0.34 sets=0.06 ms | stall fanout=68.51 prime-short=63.78 fanout-minus-prime=+4.73 ms
      DAY59 READING order=o2 N=45 snapshot=0.50 restores=0.85 | alloc=0.06 copies=0.71 clones=0.34 sets=0.06 ms | stall fanout=68.74 prime-short=63.81 fanout-minus-prime=+4.93 ms
      DAY59 SHARES order=o1 calls=0.83 allocs=0.04 of 1.34 ms
      DAY59 SHARES order=o2 calls=0.82 allocs=0.04 of 1.35 ms
      DAY59 SELECT -> DESIGN B1 (batched copies)

- Read: the device calls are 82% to 83% of the publisher's 1.34 ms of owner time and the allocations 4%. B1 is
  selected. On this card the fanout costs the tenant +4.73 / +4.93 ms over one prime (DAY54 read +6.50 on a 5900XT
  host with the 5090 cell's shape).

## 7. Design B1, pre-registered (committed before any code)

**The program.** The same bytes into the same destinations, in one device launch per snapshot and one per restore.

- (B1.1) A kernel `copy_batch_items_u8` (`cu/kernels.cu`, beside `copy_batch_uniform_f32`): a u64 table
  `[src x n, dst x n, bytes x n, set_dst x m, set_val x m]`, grid `(chunks, n + 1)`. Block row `r < n` copies item
  `r` (16-byte vectors when both pointers are 16-byte aligned, then the tail byte by byte; bytes only otherwise). Row
  `n` writes the `m` i32 sets. `Engine::copy_batch_items_u8(items, sets)` uploads the table (one H2D), launches
  once on the owner stream and frees the table in stream order.
- (B1.2) `prefix_snapshot`: each KV plane with bytes and each recurrent plane is allocated without a memset, because
  the batch writes every byte of it. A zero-byte KV plane keeps its zeroed 1-byte allocation. The layer loop collects
  the items, and one batch call follows the loop, before the TP shards. The latent planes keep their own calls in the
  loop.
- (B1.3) `prefix_restore_at`: the KV copies, the length sets and the recurrent copies go into one batch after the
  loop. The latent restores stay in the loop, and the TP shard restore follows the batch. A destination or source
  too short for its range fails the request with a named error, as `copy_u8_into` does now; it never panics the
  worker.
- (B1.4) Cross-stream order: every source keeps its read event and every destination its write event. The cudarc
  guards are held across the launch, so a consumer on another stream (the host tier's copy stream) waits for the
  batch exactly as it waited for the copies.
- (B1.5) The split keeps its line and wording. The allocations stay kind 0, and the batch is timed as kind 1 (one
  "copy" per snapshot and one per restore); clones and length sets read `0 over 0`. `day59-reading.py` parses the
  lines unchanged. The census `day59_the_fanout_copy_split_is_log_only` is rewritten to pin the new program: one batch
  call in the snapshot and one in the restore, and no per-plane copy, clone or length set left in either.
- (B1.6) No door. The rollback is the previous binary. The bytes are the same for every request, and the program
  switches for all requests at once with the binary, never mid-request (one numeric program per request). The A/B
  compares two binaries built from one clone.

**The cells.**

- (a1) The engine GPU cell `copy_batch_items_u8_is_the_memcpy_program`: random items with 16-byte-aligned and
  unaligned pointers, sizes 0, 1, 15, 17, 4096 + 3 and 4 MiB + 5, plus sets. Each destination range equals its
  source, every byte outside the ranges is unchanged, and each set reads its value. The red arm is a scratch patch in
  which the kernel skips each item's last byte, grep-checked in its test binary; it must fail the cell.
- (a2) The server GPU cell `b1_snapshot_and_restore_are_the_copy_program`: `Cache::new` on the 27B's own config
  (`MEMRA_B1_MODEL`, read from its GGUF metadata only) and on a tiny synthetic hybrid config. The planes are filled
  with random bytes at `pos`. The snapshot's every plane equals the source's `[0, len * tok_bytes)`, and a zero-byte
  plane reads zero. A restore into a fresh cache pre-filled with a pattern: the KV `[0, kb)` equals the entry, the
  bytes past `kb` keep the pattern, `len_d` reads `restore_len`, and the recurrent planes equal the entry. The same
  red arm must fail it.
- (a3) On the B1 binary, door OFF and ON: the identity gate default and plain, and the hit gate. The registered
  clause is door ON; the OFF arms are added because B1's path does not depend on the door.
- (b) to (d) The paired cell: two binaries (base: B1's parent; b1: the B1 tip) by two modes (`fanout`,
  `prime-short`), 72 words, the prefix cache at 256 MB, door ON, the S sittings' environment. o1 runs base-fanout,
  b1-fanout, base-prime, b1-prime five times; o2 runs the reverse sequence five times. That is 40 boots in one
  collector hold, with 250 ms telemetry and each boot's start temperature and SM clock.
  - (b) b1's snapshot-plus-restores median at most half of base's, per order.
  - (c) base's fanout-minus-prime stall minus b1's at least 1.0 ms, in both orders.
  - (d) b1's fanout members' `wall_ms` median at most base's plus 1.0 ms, per order.
  Complete: 40 boots with receipts, `errors` empty, 40 `STALL REPLAY: PASS`, and a split line for every fanout
  publish line. An incomplete cell reads nothing and repeats whole once.

**The decision.**

- (a1) to (d) all pass: B1 is adopted as the naked program.
- (a) fails: B1 is refuted and reverted in one commit, with its red receipts banked.
- (a) passes and (b), (c) or (d) fails: the failed clause is recorded as read, and B1 is reverted in one commit. Any
  revised design goes under a new pre-registration; no bound moves.

**What each card decides.** The target card (one RTX PRO 6000 Blackwell, any CPU class) runs (a1) to (d) and decides
adoption. The CPU suites (server lib, clippy, fmt) run here. The 5090 runs (a1) and (a2) when the card is free: a
compatibility reading, not a veto (5090 rows follow per-hardware rules). The 5090's price for this change is owed
with the three 5090 cells already queued.

**Budget.** 0.3 agent-day: the kernel and seams 0.1, the cells 0.1, the sitting 0.1.

## 8. Revuto's finding on the split (#731), and whether it moved the selection

- **The defect** (revuto on #731, review at `3214d1e13`, inline on `worker.rs:34774`). `PREFIX_COPY_SPLIT` is a
  thread-local that every `prefix_snapshot` and `prefix_restore_at` call adds to, but only the fanout takes it, and
  only after its snapshot returns. So whatever another owner-thread caller left since the previous fanout lands in the
  next fanout's snapshot line. Those callers are the retire capture's on-tick snapshot (`worker.rs:22104`), the park
  snapshot (`:29550`), the hit restores (`:32048`, `:32203`) and the `prefix_restore` wrapper. Restore-kind calls
  (kinds 1 and 3) then inflate `a.copy_ms` and `a.copies`, and the gap between two fanouts is unbounded. The step 1
  comment ("the take also drops whatever an earlier caller left") is wrong: the take adds the leftover to the line
  rather than dropping it. The fix is DAY66, its own registered step.
- **Could it have moved `DAY59 SELECT -> DESIGN B1`?** Every timed call adds to its kind's count, so a leftover shows
  in the line's counts. The clean counts follow from the 27B's layout:
  - the snapshot: 2 allocations and 2 copies per attention layer (16 layers: 32 and 32) and 2 clones per recurrent
    layer (48 layers: 96);
  - three sibling restores: 3 x (32 + 96) = 384 copies and 3 x 16 = 48 length sets.

  All 100 fanout split lines in the twin's receipts, including each boot's first tick, carry exactly those counts:

      100 on-tick split: snapshot alloc X ms over 32, copies X ms over 32, clones X ms over 96; restores copies X ms over 384, len sets X ms over 48

  (`grep -ho 'on-tick split: .*' box/short/ab/*/b*-fanout/server.log`, the times masked, `sort | uniq -c`). No line
  took a leftover call, so no line took a leftover's time: in this cell's shape no other snapshot or restore ran on
  the owner thread between fanouts. The door-ON captures went off the tick, and the fanout's fresh prompts never hit.
  **The selection stands as read**, and no rerun is needed for it. Under the fix the same cell would print the same
  lines.

## 9. Design B1 as built (`e522a9417`), and its target sitting prepared

- (B1.1) The kernel `copy_batch_items_u8` (`cu/kernels.cu`) and `Engine::copy_batch_items_u8` (`lib.rs`), with a
  KERNELS.md row. Zero-byte items are dropped host-side, and the grid is `(min(64, max_bytes / 4096), n + 1)`.
- (B1.2) `prefix_snapshot`: `prefix_plane_alloc` gives a KV plane with bytes an uninitialized buffer and a zero-byte
  plane a zeroed 1-byte one; the recurrent planes use `alloc_f32_uninit`. Then `prefix_snapshot_batch` runs, with one
  launch, before the TP shards.
- (B1.3) `prefix_restore_at` calls `prefix_restore_batch` right after the validation: the KV `[0, kb)` copies, the
  recurrent copies and the length sets in one launch, every range checked before any device write. The host lengths
  and the latent restores follow in the loop, and the TP shards after that.
- (B1.4) Both batch fns hold the source (read) and destination (write) guards across the launch and drop them after
  it.
- (B1.5) The census `day59_the_fanout_copy_split_is_log_only` is rewritten as registered: allocate, then batch, then
  TP; validate, then batch, then the host lengths; one launch per batch fn, with the guards dropped after it; no
  per-plane `copy_u8_into`, `clone_dtod`, `copy_into` or `set_i32_one` in the four fns. The split lines keep their
  wording (clones and length sets now read `0 over 0`).
- The cells (a1) `fused_gate_bounds_tests::copy_batch_items_u8_is_the_memcpy_program` (engine) and (a2)
  `worker::tests::b1_snapshot_and_restore_are_the_copy_program` (server) are `#[ignore]` GPU cells. They were not run
  here: the 5090's lock was free, but another project's compute app held 1.4 GiB of the card, so by the lane's rule
  I waited rather than share it. The target card runs them.
- CPU: server lib `935 passed; 0 failed; 26 ignored` (`b1/server-lib.log`); clippy `-D warnings` on the server and
  the engine, all targets (`b1/clippy.log`); fmt clean.
- **The sitting** `pro-single-b1/`, receipts root `/root/spill-receipts/a-b1`:
  - `build.sh <tip> 9ab479d9c` builds b1 (the tip's server plus its engine and server lib test executables), red
    (`red-arm.patch`: the kernel copies each item's bytes minus one, and the wrapper prints `[b1 red arm] ..`; test
    executables only) and base (the tip's crates taken to B1's parent). One clone, the tree checked back after each
    arm, and the markers in `markers.txt`.
  - `driver.sh`, each step under one collector hold:
    - `unit-cells.sh`: (a1) and (a2) green on b1 with `MEMRA_B1_MODEL` set to the 27B; both must fail on red with
      the marker printed; the day54, day59 and day66 censuses.
    - `gates.sh`: the identity gate default and plain, door OFF and ON.
    - `hitgate.sh`: the hit gate, OFF then ON.
    - `ab.sh short fanout prime-short 256`: 40 boots.
  - Then `b1-reading.py`, whose last line is `B1 VERDICT -> ..`. The reader was dry-run on DAY59's receipts mapped
    as identical arms. It read (b) and (c) FAIL and (d) PASS, which is right for two identical arms.
  - About 75 minutes of card time plus the three builds.

## 10. B1's sitting, read as registered: ADOPT

- Run by the lead on one RTX PRO 6000 Blackwell Workstation card (a 16-core host), `build.sh 7edc329d9 9ab479d9c`
  then `driver.sh`, 05:0xZ to 05:50Z. Mirror `pro-single-b1/box/`: 346 receipts, sha256-checked against the box
  manifest. The six executables are recorded by hash only (`binaries.sha256`): b1 server `806579de14f7317f..`, base
  `e8f8de9794c0990d..`, b1 tests `ee1f8d97..` (engine) and `bf6a2356..` (server), red tests `3bbae521..` and
  `7d24dfa7..`. `markers.txt`: the batch wording is in b1 (3) and absent from base (0); the red marker is in both red
  test binaries and absent from b1's. 40 boots with start temperatures of 44 C to 65 C, and 250 ms telemetry in
  `ab-short-cell/`.
- Verbatim (`box/reading-b1.log`):

      B1 (a) UNIT a1-green=0 a2-green=0 a1-red=101 (marker 1) a2-red=101 (marker 1) censuses=0
      B1 (a) gates {'identity-default-off': '0', 'identity-default-on': '0', 'identity-plain-off': '0', 'identity-plain-on': '0', 'hitgate-off': '0', 'hitgate-on': '0'}
      B1 READING order=o1 N_own=45 own base=1.36 b1=0.25 ms | fanout-minus-prime base=+4.77 b1=+3.74 (gain +1.03) ms | members wall base=95.9 b1=94.9 ms
      B1 READING order=o2 N_own=45 own base=1.34 b1=0.25 ms | fanout-minus-prime base=+4.90 b1=+3.75 (gain +1.15) ms | members wall base=96.0 b1=94.8 ms
      B1 (b) PASS per order [True, True]
      B1 (c) PASS per order [True, True]
      B1 (d) PASS per order [True, True]
      B1 VERDICT -> ADOPT (B1 is the naked program)

- (a2) ran on the tiny config and on the 27B's own geometry at pos 1 and 97. The 27B has 17 attention layers, the
  last of them the MTP layer, which the cell holds absent at capture, and 48 recurrent layers
  (`[b1 cell] /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf pos=97: 17 attention and 48 recurrent layers equal`).
  The red arm failed each cell on its first shortened item: `tiny pos 1 K 1` in (a2), and (a1)'s byte check.
- **Read.**
  - The fanout's own owner time falls from 1.36 / 1.34 ms to 0.25 ms per publish, 18% of base (the bound was 50%).
  - The tenant's stall over one prime falls by 1.03 / 1.15 ms (the bound was 1.0 ms). Both orders clear it, o1 by
    0.03 ms.
  - The members' walls are 1.0 / 1.2 ms faster.
  - Every clause of (a) holds: the bytes are the per-plane program's on both configs, and the six gates are green.
- **Adopted** as registered: B1 is the naked program, with no door; the rollback is the previous binary. It goes to
  integ67 after integ66 merges.
  - `MEMRA_B1_MODEL` stays as the adopted cell's test input. Its FLAGS row drops the decide-by, because a flag a gate
    sets stays under the door-hygiene rule.
  - The 5090's (a1) and (a2) stay owed as a compatibility reading, not a gate.
- Item 10's remaining publishers (DFlash, GLM-5, latent) stay owed to their artifacts and rigs (DAY54). The fanout's
  insert (2.2 ms, the evicted entry's demote pre-submit) is item 19's.

## 11. integ67's two asks: the grid limit (fixed) and B1's 5090 half (running)

- **The grid limit** (review of `e522a9417`). The batched launch uses `gridDim.y = n + 1`, and CUDA caps it at 65535.
  A batch of more than 65534 items would have failed at launch, after the table upload.
  - `Engine::copy_batch_items_rows(n)` now refuses it before any device work, naming the count (`batched copy of N
    items refused: N+1 grid rows exceed CUDA's gridDim.y limit of 65535`).
  - The CPU test `copy_batch_items_refuses_more_rows_than_the_grid_holds` covers 0, 128 (the 27B's snapshot), 65534
    (the largest legal batch) and the refusal at 65535. Engine lib `576 passed`; clippy; fmt. Commit `231fba087`.
  - Red arm (`b1/grid-red-arm.patch`, the check skipped, with a marker): the test fails on `called
    Result::unwrap_err() on an Ok value: 65536` (`b1/grid-red-arm.log`).
  - Today's shapes are far below the limit (the 27B's restore is 129 items), so the program and its bytes are
    unchanged.
- **B1's 5090 half.** B1 changes the naked snapshot and restore program on every card, so by the per-hardware rule it
  needs a 5090 reading before main. `rtx5090-b1/` runs it on the target sitting's own trees:
  - b1 at `7edc329d9`, base at `9ab479d9c`, red at b1 plus `pro-single-b1/red-arm.patch`, built in one scratch
    worktree with its own target dir (both removed after), the executables outside `/tmp`.
  - One bounded hold of `/tmp/memra-5090.lock` with the idle rule: (a1) and (a2) green and red with
    `MEMRA_B1_MODEL` set to the 27B, the censuses, the identity gate default and plain door OFF and ON, and the hit
    gate OFF and ON, all through `--external-lock 9`.
  - Then the 40-boot paired cell in the target sitting's environment, and `b1-reading.py`.
  - A 5090 regression makes B1 a per-card default question, not a revert on the target.
  - It runs ahead of W's 5090 half. W's run had taken the hold at 07:33Z and passed (a) and four identity gates when
    I stopped it at 07:36Z, before its timed cell, so B1 could go first. It is banked as
    `rtx5090-w/cell/stopped-0736Z/` and repeats whole after B1.
