# WP-A resumable state (2026-09-27: F2 built and its sitting frozen, NEED TARGET CARD (lane/spill-a-f2-20260927 502e780bd, base d6132710e); the DAY68 5090 chain rebuilding and running)

- Lane `lane/spill-a-20260919`; worktree `wt-spill-a`. integ67 takes B1 (`e522a9417`, adopted on both cards) and the
  grid refusal (`231fba087`, cherry-picked as 95f275859).
- Closed: items 10 (B1), 11 (reading), 12 (W reverted, DAY61 section 5), 13 (R1 ADOPTED, `04554e99f`; R2 reverted),
  21 to 25; DAY66 (the fanout split's scope).
- **Items 14 and 19 (DAY63) closed: L' ADOPTED** (section 6, `348d2e8f3`, with the gate change `217ace3fd`); L' and R1
  go to integ69.
- **Item 18 (DAY64).** The split selected the fill (8.63 ms of 12.9), not D1 (section 5). Design F (the fill on its
  own stream in chunks) is built (`568f33c7b`), with its sitting `pro-single-f/` (`build.sh <tip> 0a835a75b`, then
  `driver.sh`). **F read REVERT ((a) passed with the worker step; (b), (c) failed) and is reverted (`ef8cd47d8`,
  DAY64 section 8).** Item 18 stays open; F2 (pinned resident payloads) is due next, registration first.
- **Item 20 (DAY65).** T-H read REVERT (b) and was reverted (`06b2d31db`). T-H' is registered (section 6: (b')
  measured at long entries, from DAY65's own text), awaiting the lead's and the owner's acceptance before any code.
- **Item 17 (DAY67).** P2 on L': the first sitting was stopped as a diagnostic (T-H was in both arms). The corrected
  pair P2L2 read (b) to (g) PASS and (a)'s repeated unit step read all green on `a2419d3e1`: **P2 ADOPTED** (DAY67
  section 5). For the next integ: `lane/spill-a-p2-20260926` (`409be61f8`, P2 on integ69's fix tip `a57f85897`). P2's 5090 half
  is owed after the DAY68 chain.
- **integ69** took `6c60d798f` (L' + R1 + the gate change + the two test-only cell fixes); revuto found the purge
  retention defect (DAY69): **design P, `4f297e7bd` on `lane/spill-a-integ69-20260926`**, is the fix (the pool
  drained with an epoch, the staging set zeroed, at every purge); CPU battery green, both red arms caught; its two GPU
  cells run in the lead's battery. On this lane it is `59376ebeb`. Worktree `wt-spill-a-i69` holds that branch until
  integ69 merges; remove it then. Revuto round 2: **design Q, `a57f85897`** (DAY70: a lease charged its length,
  the pool its idle backings and the leases' tails within its cap), CPU battery green, red arm caught; GPU cells
  named for BOX43 in DAY70 section 4. Owed after the fixes: the GLM-5 arena's purge scrub and the pooled-lease read guard
  (DAY69 sections 1 and 2).
- **The owed 5090 cells (DAY68), registered and built:** R1's half and L''s half (section 1, the target sittings' own
  pairs and scripts, derived by `rtx5090-derive.py`), item 16 (section 3, DAY45's cell from `7b849a817`), then S4's and
  V's halves (section 4). All run from frozen copies under `/home/avifenesh/spill-a-cells/`, chained by
  `rtx5090-chain-day68.sh` (run as the copy `spill-a-cells/chain-day68.sh`), each in its own bounded hold after every
  build has finished. Receipts go to `rtx5090-{r1,l2,s4,v}/cell/` and `rtx5090-day45/cell/`; then `rtx5090-half*.sh
  clean` and the i16 worktree removal.
- Local cells run their scripts from a frozen copy of the tree, never from this worktree (DAY61 section 5's
  lesson). No build of this lane runs while one of its own timed cells holds the 5090.
- The local 5090 is shared: lanes B, C and F queue on it, and another project's process sometimes lands on it. The
  idle rule stays.
- Scratch to remove when the lane closes: the four lines added to the shared
  `/home/avifenesh/projects/memra/.git/info/exclude`, and `/home/avifenesh/spill-a-cells/` (the DAY68 trees, target
  dirs and binaries while the halves run).

- **integ70:** `lane/spill-a-integ70-20260927` at `d5156f468` on main `b1fcf40aa`: `6ae34c277` (P2, the p2 arm's
  program; one textual conflict with main's new test at the top of `worker.rs`'s test module, both kept) and
  `d5156f468` (this directory at `aa1616662`, records only). Not built here (the DAY68 chain's timed cells); the
  lead's CPU battery is its first build. P2 adds no GPU cell (its `day51_`/`day52_` cells are CPU); its 5090 half is
  owed after the DAY68 chain, registered then. `lane/spill-a-p2-20260926` is superseded by it (delete once integ70
  merges).

- **DAY68 first chain read (section 8):** L' ADOPT on the 5090; R1 closed by the lead as "unexercised on the 5090"
  (not a PASS; section 10), its timed cell repeating once without builds to place o2 `retire-seam`'s +18.2 ms; item 16
  regime not reproduced; S4 and V not run. **Second chain** (section 9) started 17:10Z from
  `spill-a-cells/chain-day68b.sh`: R1's cell, item 16 with the warm-up doubled, S4, V. P2's 5090 half after it.
- **T-H' (DAY65 sections 7 and 8):** sitting ready on `lane/spill-a-th2-20260927` (`67af1b71e`, base `80f734c77`):
  `build.sh 67af1b71e 80f734c77` then `driver.sh`, last line `TH2 VERDICT -> ..`; the lead rents a clean PRO 6000 WS
  (not BOX43's machine) and sends the box on acceptance.

- **T-H' ADOPTED** on BOX44 (DAY65 section 10). For the integ: `lane/spill-a-integ73-20260927` at `20bf78b42` on main
  `21ce97836`: `43a956340` (T-H', the sitting's diff line for line) and `20bf78b42` (this directory at `3a2e2886f`).
  Not built here (R1's repeat holds the 5090 in a timed cell). T-H' adds no GPU cell (its cells are CPU); its 5090 half
  ((b') and (d)) is owed after the DAY68 chain, beside P2's. Delete `lane/spill-a-th2-20260927` once it merges.

- **F2 (DAY71):** built on `lane/spill-a-f2-20260927` (`7b38cc013` code, `502e780bd` records and sitting) over
  main's T-H' tree `d6132710e`. CPU green (server 988, engine 667, clippy), red arm caught. The sitting:
  `build.sh 502e780bd d6132710e` then `driver.sh`, last line `F2 VERDICT -> ..`, about 2.5 h on one PRO 6000 WS.
  Its two GPU cells (`option_b_published_spans_stay_resident_in_their_staging`,
  `option_c_resident_spans_read_the_entry_in_place`) go to the target battery.
- **DAY68 chain, third start** (section 12, sccache bypassed): cell root `~/.local/share/memra-lane-a-cells/`,
  `IN-USE.txt` kept.

- stopped 2026-09-28 by the owner's order; queue chain-day68c killed mid-run at R1's card (its first hold, waiting on the 5090 lock at attempt 36 of 180, after every build); F2's BOX46 sitting stopped by the lead after gates, demote and chain (promote killed mid-run), partial receipts in pro-single-f2/box-partial/.
