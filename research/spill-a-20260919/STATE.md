# WP-A resumable state (2026-09-26, days 56 to 58 closed; stopped at an integrable milestone)

- Lane `lane/spill-a-20260919`; worktree `wt-spill-a`. origin/main `968c0fa68` (integ63) merged into the lane
  (`621502678`); integ64 takes `90fe89abf`; days 56 to 58 are new.
- Item 23 closed (DAY56, F2 `10b9329cc`): the reservation path takes its lane counters; the shed tests isolated; the
  suite's median 6.62 s against 6.70 (N=400).
- Item 24 closed (DAY57, `a327f486c`): the darklane stop-mode cycle waits for acknowledgements under a 30 s guard
  (reproduced 1 of 100 beside sixteen burners; 0 of 100 after).
- Item 25 closed (DAY58, `62a29cfe0`): a real route-book ordering defect (an ended run counted as running in 27 to 34%
  of boundary snapshots); fixed in `route_telemetry.rs`.
- Next: the fanout publisher's design (DAY54's price), items 11 to 14 (19 with 14, 17 re-read on top), 18 and 20, and
  the owed 5090 cells (S4's half, V's half, item 16) under `/tmp/memra-5090.lock`.
- Local scratch: none.
