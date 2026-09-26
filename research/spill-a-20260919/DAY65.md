# WP-A day 65: OWED item 20, the hash helper's per-payload work across threads (design T-H)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Behind `MEMRA_KV_HOST_CONTRACTS` (default OFF). Every cell
`executed-not-qualified`. Pre-registered while DAY59's cells wait for a card; its code follows in order (after items
11 to 14, 17 and 18). DAY64 is reserved for item 18.

## 1. Pre-registration (committed before any code)

**The price on record** (DAY49 section 3, DAY51 section 3, DAY52 section 3): the helper's job is one thread: at
64-token 27B entries copy 23.7 ms plus hash 59.1 ms over 96 independent staged payloads (83.5 ms), and at 5122-token
entries the bind's KV re-hash adds about 56 ms over 32 independent lease views (helper 122 to 140 ms). The demote
publishes only after the job (wall 101 ms at 64 tokens, 362 to 444 ms at long entries), and a request that meets the
entry `Demoting` waits for it (DAY43: at long entries the chained hit waits on the helper, not the copy).

**Design T-H.** The helper's `Hash` job splits its work across `min(8, available_parallelism / 2)` scoped threads (at
least one): the staged payloads (copy and digest, each whole on one thread) and the lease views (digest each), in
contiguous shares of the job's order; the reply keeps the job's order. The digest program per payload and per view is
unchanged (the same `checksum` over the same bytes). The `Sources` job (the promote's checksums) splits its views the
same way. Design T's thread rule for the fill (item 3) is the precedent.

**Acceptance.**

- (a) Bitwise: every digest equal to the one-thread helper's on the same job (a CPU cell over synthetic payloads of the
  27B's shapes, 1 to 8 threads); the identity, fault and hit gates door ON; the pause gate.
- (b) The demote cell (the 27B, 64 tokens): the helper time median at most half of the tip's, and the steady wall t0
  to publication at most the tip's minus 20 ms, per order.
- (c) The chain cell (DAY52's): the chained request's e2e at most the tip's minus 30 ms per order.
- (d) The tenant's stall and the hump at most the tip's plus 1.0 ms and 0.15 ms; the promote cell's PIN and e2e at most
  the tip's plus 1.0 ms.
- Readings: the helper's CPU time and the threads it used; the 5090's half on its own host class.

**What each card decides.** The target card (b) to (d); the 5090 its own (b) and (d) after.

**Budget.** 0.5 agent-day.

## 2. Design T-H as built (`a839d3494`), and its target sitting prepared

- `host_hash_threads(parallelism) = clamp(parallelism / 2, 1, 8)`, read once when the helper spawns (8 on a 16-core
  host and on this rig).
- `host_scoped_map(items, threads, f)` maps `f` over contiguous shares on scoped threads, with the results in the
  items' order; one thread or one item runs inline.
- The helper's three maps use it:
  - the `Hash` job's payloads (each payload whole on one thread: the staged copy, then `host_hash_payload_digest`);
  - its lease views (`(slot, v.len(), v.digest())`);
  - the `Sources` job's views (`v.digest()`).
  - The digest program per payload and per view is unchanged.
- The helper split line's copy and hash terms are now thread time summed over the shares, `helper` stays the job's
  wall, and the line ends `; N threads`. Stated for the readers: DAY49's and DAY52's split regexes read the same
  fields.
- Cells:
  - `day65_the_shared_helper_digests_equal_the_one_thread_helper_bitwise`: 96 payloads of mixed lengths, 1 to 8
    threads, every digest equal and in order; the thread rule's table.
  - The census `day65_the_helper_splits_its_three_maps_and_keeps_the_program`.
  - Two older censuses follow the new shape. Day 35's lease map is now found inside its share call, and day 49's
    split starts with the thread count; both keep the copy-then-hash order.
  - The existing `hash_helper_digests_equal_the_owner_thread_digests_bitwise` runs the real helper at 8 threads on
    this rig.
- The red arm (`day65/red-arm.patch`: the shares come back reversed, with a marker) fails the bitwise cell at 2
  threads (`day65/red-arm.log`).
- Server lib `943 passed; 0 failed; 26 ignored`; clippy `-D warnings`; fmt (`day65/`).
- **The sitting** `pro-single-th/`, receipts `/root/spill-receipts/a-th`: `build.sh <tip> <T-H's parent>`, then
  `driver.sh`, each step under one collector hold:
  - the 11 gates on th;
  - base against th, 20 boots each, in the demote, chain and promote cells (P2's cell environment);
  - the hump cell (4 boots, base th th base);
  - then `th-reading.py`, whose last line is `TH VERDICT -> ..`.
  - The reader was dry-run on P2's receipts mapped as the two arms. It parsed base's helper 82.8 / 83.2 ms, wall
    100.8 / 101.0 ms, PIN 25.9 / 26.1 ms and hump +0.51 ms (`INCOMPLETE` only from P2's three-arm chain cell).
  - About 2 hours of card time.

## 3. T-H's A/B base, re-derived after L's revert and re-application and W's revert

- The sitting's first base, `1cba80185` (T-H's parent), no longer differs from the tip by T-H alone. Since then the
  tip has reverted L, re-applied it as L', reverted W, and added DAY64 step 1's timing lines.
- The base is now the tip with T-H taken back out: branch `lane/spill-a-th-base-20260926` at `c6369b507`. It is the
  revert of `a839d3494` on tip `771fc2a8f`, with the one test-block conflict resolved by removing only T-H's cells.
  Server lib `942 passed`. It is never merged.
- `pro-single-th/build.sh` fetches that branch too. The sitting's commands are `build.sh <tip> c6369b507`, then
  `driver.sh`, still only after L' adopts: the tip carries L'.
