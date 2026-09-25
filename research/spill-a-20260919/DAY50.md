# WP-A day 50: OWED item 9, the promote's tick placement on the current tree (a reading)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. OWED item 9 (C DAY39 sections 3 and 7, ruling 43): `40 of 40 runs
lack tick 2` for promote ON on b1 and b2 on the target card (C's tick reader: tick 2 is the second stretched tick after
the fire). Acceptance registered by the ledger: "a reading of the promote's tick placement on the current tree,
pre-registered, with the day-33 timeline fields that b1 and b2 lack". Every cell `executed-not-qualified`.

## 1. Pre-registration (committed before the reader runs on any receipt)

**The reader** (`day50-reading.py`, written now): per promote-arm run, C's tick definition unchanged (a gap stretched
when above 3 x the run's median gap; tick 1 the first stretched gap at an index at or after the fire, tick 2 the next);
per boot, the day-33 timeline of each `promote published off the tick` line (the submission's tick and the
publication's tick) and the submission's owner segment. Printed per boot and pooled.

**The receipts it reads.**

1. Existing, not yet read for ticks: design S4's promote A/B on the target card (DAY48 section 3,
   `pro-single-s2/box-design-s4/promote/ab/*/b*`), both arms (g4 and S4's tree, whose promote program is the current
   tree's: V and the split lines do not touch the promote path). Their PIN and e2e were read by DAY48; their ticks were not.
2. New: a promote block in DAY49's sitting (`pro-single-day49/cell.sh`, three boots of `stall_cell.py --mode promote
   --n 5` on the tip binary after the demote arms), so the reading also runs on the exact current binary.

**What it can say, stated now.** Whether the current tree's promote stretches one tenant tick or two, and where the
timeline puts the submission and the publication. **Predictions:** tick 2 absent in every run (as C read on b1 and b2);
the timeline's submission in tick N and its publication at the tick top of N + 1 (`submit_to_publish_ticks` 1 on every
steady promote); the submission's owner segment about 0.5 ms (below any stretch), so the one stretched tick is N + 1
(the publication, the restore and the parked request's prime). If a tick 2 appears, its placement is described against
the timeline, and the promote's second stretched tick becomes its own owed item.

**What each card decides.** The target card only.

**Budget.** 0.1 agent-day.
