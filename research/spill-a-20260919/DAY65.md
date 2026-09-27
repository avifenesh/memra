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

## 6. T-H', pre-registered (committed before any rerun): the clause that measures what DAY65 named

- **The argument, from section 1's own text.** Section 1 priced the helper in two regimes. At 64 tokens: copy 23.7 ms
  plus hash 59.1 ms, the wall 101 ms. At 5122-token entries: helper 122 to 140 ms, the wall 362 to 444 ms.
  - It named the waiting case: "a request that meets the entry `Demoting` waits for it (DAY43: at long entries the
    chained hit waits on the helper, not the copy)".
  - Clause (b)'s wall bound was taken at 64 tokens on the assumption that the 64-token publication waits on the
    helper. Section 5 shows it does not: the helper fell 58 ms there and the wall 12 ms. The 64-token publication is
    bounded by its copy phase and its tick-top polls.
  - The regime section 1 named, long entries, is the chain cell's. There (c) was registered and passed.
- **T-H'** is T-H's code unchanged, re-applied on the tree at its sitting (merged with P2's reserve if P2 adopts,
  under the same assignment rule DAY67 section 2 named). Its (b) is restated from section 1's long-entry numbers:
  - **(b')** In the chain cell (long entries), th's helper median at most half of base's, and th's steady wall t0 to
    publication at most base's minus 50 ms, per order.
    - The bound is set from section 1's prices. Half of the 122 to 140 ms helper is 61 to 70 ms; 50 ms leaves room
      for the publication's other terms.
    - The refuted sitting's chain numbers are not used to set it, and that sitting is not reused.
  - The 64-token demote wall becomes a reading, stated with section 5's result and its reason.
  - (a), (c) and (d) are section 1's, verbatim: the chain e2e at most base's minus 30 ms; the stall, the hump, and the
    promote PIN and e2e bounds as registered.
- **The rule.** T-H' adopts if (a), (b'), (c) and (d) hold in both orders; otherwise it is reverted in one commit. It
  is a new sitting on a fresh pair, after P2's verdict fixes the tree it is merged onto.
- This is the lead's and the owner's to accept or refuse before any code. The argument is that (b) measured a regime
  section 1's own text did not name as the waiting one. Nothing here moves a bound on the regime that was named.

## 7. The owner accepts T-H' as a new experiment (2026-09-27)

- Owner ruling, relayed by the lead on 2026-09-27: T-H' is accepted as a new experiment, and T-H's REVERT stands.
- T-H' runs as section 6 registered it, on a fresh sitting:
  - T-H's code re-applied on the current tree, which is main with L', design Q, the purge scrub and P2;
  - (b') set from section 1's pre-result prices;
  - adopting only if (a), (b'), (c) and (d) hold in both orders.
- Nothing in section 6 changes.

## 8. T-H' as built (`67957b4f8`), and its sitting prepared

- **The tree.** Branch `lane/spill-a-th2-20260927`, cut from main `80f734c77`, which carries L', R1, Q, P and P2. T-H'
  is `67957b4f8`, the revert of T-H's revert `06b2d31db`.
  - It applied without a conflict. So the helper's three maps run on T-H's scoped threads again, and P2's reserve is
    handed out on the helper thread in the job's order before the shares fill and hash: DAY67 section 2's assignment,
    restored verbatim.
  - The helper split line reads `(helper Y ms); reserve H of S staged; T threads`.
- **(a)'s CPU half:**
  - server lib `989 passed; 0 failed; 27 ignored` (`day65b/server-lib.log`), with T-H's
    `day65_the_shared_helper_digests_equal_the_one_thread_helper_bitwise`, its census, the real-helper
    `hash_helper_digests_equal_the_owner_thread_digests_bitwise`, and P2's `day51_`/`day52_` cells green;
  - clippy `-p memra-server --all-targets -D warnings` clean; fmt; `tools/check-flags.sh`.
  - Built under `nice -n 19`, a 600% CPU quota and `MemoryMax=12G`, the lead's limits while four lanes work.
  - **Red arm:** T-H's own `day65/red-arm.patch` (the shares come back reversed, with a marker) applies to this tree
    unchanged. It fails the bitwise cell and the real-helper cell, verbatim `[day65 red arm] the shares come back in
    reverse order` (`day65b/red-arm.log`).
- **The sitting** `pro-single-th2/` is T-H's `pro-single-th/` scripts with the receipt root `a-th2`. The build fetches
  the branch and main and records the markers `staged; {} threads` and `; reserve {} of {} staged` per binary. The
  reader is `th2-reading.py`. The cells, environments, boot counts and orders are T-H's:
  - the 11 gates on th;
  - base against th, 20 boots each, in the demote, chain and promote cells;
  - the hump (4 boots);
  - then `th2-reading.py`, whose last line is `TH2 VERDICT -> ..`.
- **The reader** `th2-reading.py` is T-H's reader with three changes:
  - (b') is on the chain cell: th's helper median at most 0.5 x base's, and th's wall t0 to publication at most base's
    minus 50 ms, per order, over the steady lines (each boot's second and later);
  - the demote cell's helper and wall are a reading;
  - the split regex admits P2's reserve term.
  - Dry-run for parsing only on P2L2's receipts, mapping its p2 arm as th (`day65b/reader-dry-run-p2l2.log`). It read
    the chain cell's helper at 122.4 ms against 122.4 and its wall at 410.0 against 409.9 (N=100 per arm per order),
    so it parses every term. It reads `INCOMPLETE` and `(a)` failed only because that sitting's hump arms and gate set
    differ.
  - The refuted T-H sitting's chain numbers were not read.
- **Commands** on the rented box (receipts `/root/spill-receipts/a-th2`):
  - `bash research/spill-a-20260919/pro-single-th2/build.sh <branch tip> 80f734c77`;
  - then `bash research/spill-a-20260919/pro-single-th2/driver.sh`.
  - About 2 hours of card time.
- **The box:** one RTX PRO 6000 Blackwell Workstation card, the only tenant for the sitting.
  - At least 16 logical CPUs, so the helper's thread rule gives 8 threads as it did at T-H's sitting.
  - At least 64 GB of RAM.
  - The 27B at `/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` (sha256 `1facf36c..`).
  - The CUDA 13 toolchain and Rust to build; the lock `/tmp/memra-gpu.lock` taken by the collector per cell.

## 9. T-H''s sitting, started

- Run by the lead on BOX44, a fresh box: a Core Ultra 9 285K class host (24 threads), one RTX PRO 6000 Blackwell
  Workstation card at 600 W, 249 GB of RAM.
  - Acceptance passed: no power brake, zero PCIe replays, and the FMA spin at 2797 MHz.
  - It is the host class that carried L''s, R1's and P2L2's sittings. It is not the machine with the hourly host-wide
    stall that lane C found.
- The helper's thread rule gives `clamp(24 / 2, 1, 8) = 8` threads there, T-H's count at its sitting.
- The lead's chain:
  - stages the 27B, sha-checked;
  - clones `lane/spill-a-th2-20260927` and checks the tip is `67af1b71e`;
  - runs `build.sh 67af1b71e 80f734c77`, then `driver.sh`, into `/root/spill-receipts/a-th2`.
  - It ends with `TH2-CHAIN-DONE` and the reader's `TH2 VERDICT -> ..` line.
- The lead mirrors the receipts to a staging directory outside this worktree when the chain finishes. They are copied
  in and read here as registered in sections 6 and 8.

## 10. T-H''s sitting, read as registered: ADOPT

- Run by the lead on BOX44 (section 9), 17:15Z to 18:44Z on 2026-09-27, tree `67af1b71e` (`build.sh 67af1b71e
  80f734c77`), sole tenant.
  - Mirror `pro-single-th2/box/`, checked against the lead's box manifest (`MIRROR-CHECK.txt`, and here `sha256sum -c
    LEAD-MANIFEST.sha256` rc 0). The executables are recorded by hash; th and base differ (`binaries.sha256`).
- Verbatim (`box/reading-th2.log`; re-read here by the same reader on the mirror, identical):

      TH2 READING cell=demote order=o1 stall base=68.04 th=68.11 | helper base=48.3 th=22.5 ms (N=40, th threads [8], thread time 66.0 ms) | wall base=94.7 th=94.9 ms (N=40)
      TH2 READING cell=chain order=o1 stall base=71.76 th=71.79 | helper base=82.8 th=29.8 ms (N=100, th threads [8], thread time 84.9 ms) | wall base=370.8 th=313.8 ms (N=100) | chain base=315.2 th=249.4 ms
      TH2 READING cell=promote order=o1 stall base=66.33 th=66.62 | pin base=27.20 th=15.50 | e2e base=120.7 th=117.4 ms
      TH2 READING cell=demote order=o2 stall base=68.03 th=68.17 | helper base=48.3 th=21.4 ms (N=40, th threads [8], thread time 60.6 ms) | wall base=94.8 th=94.8 ms (N=40)
      TH2 READING cell=chain order=o2 stall base=71.67 th=71.80 | helper base=82.7 th=30.9 ms (N=100, th threads [8], thread time 85.4 ms) | wall base=370.9 th=313.4 ms (N=100) | chain base=315.2 th=249.6 ms
      TH2 READING cell=promote order=o2 stall base=66.35 th=66.60 | pin base=16.70 th=15.40 | e2e base=117.4 th=110.0 ms
      TH2 READING hump base=+0.049 th=+0.036 ms
      TH2 (b') PASS [True, True]
      TH2 (c) PASS [True, True]
      TH2 (d) PASS [True, True, True, True, True, True, True, True, True]
      TH2 VERDICT -> ADOPT (T-H prime is the naked program)

  (a): all 11 gates `.exit` 0, and the CPU cells and red arm of section 8.
- **Read:**
  - **(b') holds.** In the chain cell's long entries, the helper falls from 82.8 / 82.7 to 29.8 / 30.9 ms (0.36 and
    0.37 of base) on 8 threads. The publication wall falls from 370.8 / 370.9 to 313.8 / 313.4 ms (-57.0 / -57.5
    against -50).
  - **(c) holds:** the chained request falls from 315.2 to 249.4 / 249.6 ms (-66 against -30).
  - **(d) holds:** the stall, the hump, the PIN and the promote e2e are all within their bounds.
  - **The 64-token demote cell, a reading as registered:** the helper halves (48.3 to 22.5 / 21.4 ms) and the wall is
    flat (94.7 against 94.9, 94.8 against 94.8). As section 5 read it, the 64-token publication is bounded by its copy
    phase and its tick-top polls, not by the helper.
- **Adopted as registered:** T-H' is the naked program.
  - Its 5090 half is owed, section 1's "the 5090 its own (b) and (d) after", now (b') and (d). It is registered after
    the running DAY68 chain, as P2's is.
  - For the integ, T-H''s commit is rebased onto main `21ce97836`.
- **A harness defect, placed.** `markers.txt` reads `threads wording: 0` for both binaries. T-H's own sitting
  (`pro-single-th/box/markers.txt`) read the same.
  - The marker greps the binary for a whole format string with its placeholder (`staged; {} threads`, and `helper
    {:.1} ms); {} threads` in T-H's). A compiled binary holds the pieces between placeholders, not the string, so the
    marker can match nothing in either arm.
  - The arms are told apart by their own logs instead. In the chain cell, all 210 of th's helper split lines end
    `staged; 8 threads`, and none of base's 210 carry a threads term. The binary hashes differ.
  - A later marker greps a literal piece (` staged; `, which only th's format holds).
