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
