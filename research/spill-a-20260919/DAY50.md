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

## 2. The reading on S4's promote receipts (`day50/reading-s4-sitting-{g4,s2}.log`)

- Pooled, verbatim: g4 `DAY50 POOLED runs=100 tick1=100 tick2=0 tick1_med=75.01 tick2_med=nan stretched_med=1
  submit_to_publish_ticks={1: 90} owner_segment_med=0.50`; S4 (`s2` arm) `DAY50 POOLED runs=100 tick1=100 tick2=0
  tick1_med=75.04 tick2_med=nan stretched_med=1 submit_to_publish_ticks={1: 81, 2: 9} owner_segment_med=0.53`.
- **What it says.** On the current promote program the promote stretches exactly one tenant tick (tick 1 about 75 ms,
  tick 2 absent in 100 of 100 runs on both trees, as C read on b1 and b2): the submission's own tick costs the owner
  about 0.5 ms, below any stretch, and the one stretched tick is the publication's, where the restore and the parked
  request's prime run. C's `tick 2 not_defined` is the promote's shape, not a missing measurement. The timeline places
  the publication at the next tick top in 90 of 90 steady promotes on G4 and 81 of 90 on S4; on S4, 9 of 90 publish one
  tick later: S4's H2D destination digests ride the promote's landing (DAY42 section 1 step 5), and about one promote in
  ten misses its first poll. That delay did not reach (d) (PIN +0.10 ms per order) and is recorded here as a reading.
- The tip's own promote block runs in DAY49's sitting; item 9 closes on that reading if it agrees, and the S4 late
  landings become an owed improvement (the destination digests off the landing path, required before the publication)
  if they recur there.
