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

## 4. The void first start of T-H's sitting, banked

- The lead's chain started T-H at 14:27Z with the old pair (`build.sh 21984b527 1cba80185`), before section 3's base
  reached it. The lead stopped it in its gates cell (the lead's processes only; the card emptied) and banked it as
  `/root/spill-receipts/a-th-void-stale-base-1cba80185/` with a `VOID.txt`.
- Mirrored as `pro-single-th/box-void-stale-base/` (200 receipts, sha256-checked against the box manifest; the
  executables by hash). Nothing in it is read: its base differs from the tip by more than T-H.
- The sitting restarted at 14:34Z on section 3's pair (`build.sh 50fdbcfaf c6369b507`). Its gates cell finished rc 0
  at 14:54Z.

## 5. T-H's sitting, read as registered: REVERT (b)

- Run by the lead on one RTX PRO 6000 Blackwell Workstation card, `build.sh 50fdbcfaf c6369b507` then `driver.sh`.
  Mirror `pro-single-th/box/`, sha256-checked against the box manifest (0 mismatches); the executables are recorded by
  hash.
- Verbatim (`box/reading-th.log`):

      TH READING cell=demote order=o1 stall base=64.02 th=64.15 | helper base=82.5 th=24.9 ms (th threads [8], thread time 95.1 ms) | wall base=101.0 th=88.8 ms
      TH READING cell=chain order=o1 stall base=68.10 th=68.26 | chain base=371.1 th=245.7 ms
      TH READING cell=promote order=o1 stall base=62.49 th=62.53 | pin base=25.80 th=25.90 | e2e base=113.9 th=114.1 ms
      TH READING cell=demote order=o2 stall base=64.21 th=64.18 | helper base=82.8 th=25.1 ms (th threads [8], thread time 95.5 ms) | wall base=101.2 th=88.9 ms
      TH READING cell=chain order=o2 stall base=68.11 th=68.26 | chain base=371.1 th=245.6 ms
      TH READING cell=promote order=o2 stall base=62.48 th=62.55 | pin base=25.90 th=25.80 | e2e base=113.9 th=114.0 ms
      TH READING hump base=+0.022 th=+0.029 ms
      TH (b) FAIL [False, False]
      TH (c) PASS [True, True]
      TH (d) PASS [True, True, True, True, True, True, True, True, True]
      TH VERDICT -> REVERT ((a) passed; failed b): recorded as read, reverted in one commit

- Read:
  - (b)'s helper half passes: 82.5 / 82.8 against 24.9 / 25.1 ms, a ratio of 0.30. Its wall half fails: 101.0 / 101.2
    against 88.8 / 88.9 ms, -12.2 / -12.3 against the -20 bound.
  - (c) passes by far more than its bound: the chained request falls from 371.1 to 245.7 ms (-125 ms against -30).
  - (d) passes: the stall, the PIN, the e2e and the hump are unchanged.
- **Reverted** as registered, in one commit (`06b2d31db`), with the receipts kept. P2's reserve, which sat on T-H's
  shares (DAY67 section 2), returns to its own sequential payload map, DAY52's code verbatim.
- Whether the 64-token wall was the right measure for T-H is a new registration's question, argued from this
  section's text before any rerun (section 6), not a moved bound.
