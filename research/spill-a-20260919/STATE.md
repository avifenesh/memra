# WP-A resumable state (2026-09-26, NEED TARGET CARD: F's sitting (running), then /root/units-chain.sh on BOX31; T-H' awaits the owner)

- Lane `lane/spill-a-20260919`; worktree `wt-spill-a`. integ67 takes B1 (`e522a9417`, adopted on both cards) and the
  grid refusal (`231fba087`, cherry-picked as 95f275859).
- Closed: items 10 (B1), 11 (reading), 12 (W reverted, DAY61 section 5), 13 (R1 ADOPTED, `04554e99f`; R2 reverted),
  21 to 25; DAY66 (the fanout split's scope).
- **Items 14 and 19 (DAY63) closed: L' ADOPTED** (section 6, `348d2e8f3`, with the gate change `217ace3fd`); L' and R1
  go to integ69.
- **Item 18 (DAY64).** The split selected the fill (8.63 ms of 12.9), not D1 (section 5). Design F (the fill on its
  own stream in chunks) is built (`568f33c7b`), with its sitting `pro-single-f/` (`build.sh <tip> 0a835a75b`, then
  `driver.sh`, last line `F VERDICT -> ..`), running on BOX31. F adopts only on `F VERDICT -> ADOPT` AND the added
  worker span step `pro-single-f/unit-worker.sh` at `01ce7ed10` reading `door-rc=0` (section 7, accepted). F2 (pinned
  resident payloads) is recorded for after P2's verdict.
- **Item 20 (DAY65).** T-H read REVERT (b) and was reverted (`06b2d31db`). T-H' is registered (section 6: (b')
  measured at long entries, from DAY65's own text), awaiting the lead's and the owner's acceptance before any code.
- **Item 17 (DAY67).** P2 on L': the first sitting was stopped as a diagnostic (T-H was in both arms). The corrected
  pair P2L2 read (b) to (g) PASS; (a)'s unit step is void (the stale cell arithmetic) and repeats whole on
  `lane/spill-a-p2l2-unit-20260926` (`a2419d3e1`) via `pro-single-p2l2/unit-rerun.sh`. ADOPT if all green, else
  REVERT (section 4, accepted). Queued on BOX31 after F's sitting, in `/root/units-chain.sh`.
- **integ69** takes `6c60d798f` (L' + R1 + the gate change + the two test-only cell fixes `411177fea`, `28c7aa6c1`); its
  18 worker span cells read 18 of 18 on the local 5090 (DAY63 section 7). Its BOX39 rerun follows lane C's DAY82.
- The owed 5090 cells: S4's half, V's half, item 16, R1's half, and L''s and T-H's halves once they adopt.
- Local cells run their scripts from a frozen copy of the tree, never from this worktree (DAY61 section 5's
  lesson). No build of this lane runs while one of its own timed cells holds the 5090.
- The local 5090 is shared: lanes B, C and F queue on it, and another project's process sometimes lands on it. The
  idle rule stays.
- Scratch to remove when the lane closes: the four lines added to the shared
  `/home/avifenesh/projects/memra/.git/info/exclude`, and `/home/avifenesh/spill-a-cells/` (empty).
