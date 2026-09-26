# WP-A resumable state (2026-09-26, NEED TARGET CARD: item 17's re-read on L'; T-H and DAY64's split running on BOX31)

- Lane `lane/spill-a-20260919`; worktree `wt-spill-a`. integ67 takes B1 (`e522a9417`, adopted on both cards) and the
  grid refusal (`231fba087`, cherry-picked as 95f275859).
- Closed: items 10 (B1), 11 (reading), 12 (W reverted, DAY61 section 5), 13 (R1 ADOPTED, `04554e99f`; R2 reverted),
  21 to 25; DAY66 (the fanout split's scope).
- **Items 14 and 19 (DAY63) closed: L' ADOPTED** (section 6, `348d2e8f3`, with the gate change `217ace3fd`); L' and R1
  go to integ69.
- **Item 17 (DAY67).** P2 re-applied on L' (`c58f32f7d`, merged with T-H's shares, section 2). Base branch
  `lane/spill-a-p2l-base-20260926` at `f4e84f11e`. Sitting `pro-single-p2l/` (`build.sh <tip> f4e84f11e`, then
  `driver.sh`, last line `DAY52 P2 -> ..`).
- **Item 18 (DAY64).** Placed at `receipt` (176 of 180 late, section 3). The design was revised before code (section 4).
  Step 1's timing lines are built (`08cdd9736`); their sitting is `pro-single-day64b/` (`build.sh <tip>`, then
  `driver.sh`, last line `DAY64B SELECT -> ..`). D1 (overlapped span digests) follows if the digests dominate.
- **Item 20 (DAY65, T-H).** Built (`a839d3494`). Its sitting `pro-single-th/` (`build.sh <tip> c6369b507`, the base branch
  `lane/spill-a-th-base-20260926` = the tip with T-H taken out, DAY65 section 3) runs only if L' adopts.
- The owed 5090 cells: S4's half, V's half, item 16, R1's half, and L''s and T-H's halves once they adopt.
- Local cells run their scripts from a frozen copy of the tree, never from this worktree (DAY61 section 5's
  lesson). No build of this lane runs while one of its own timed cells holds the 5090.
- The local 5090 is shared: lanes B, C and F queue on it, and another project's process sometimes lands on it. The
  idle rule stays.
- Scratch to remove when the lane closes: the four lines added to the shared
  `/home/avifenesh/projects/memra/.git/info/exclude`, and `/home/avifenesh/spill-a-cells/` (empty).
