# WP-A resumable state (2026-09-25, day 51 read and closed)

- Lane `lane/spill-a-20260919`; worktree `wt-spill-a`. integ62 (#726, main `5228ff0cd`) merged A days 42 to 48; main
  is merged into the lane (`ff41669d1`).
- Items 7 and 8 closed (DAY49 section 3: `-> H attributed`, `-> b1 step attributed to H`). Item 9 closed (DAY50
  section 3).
- Item 17: design P ran on BOX22 (a 9950X host) and FAILED (g) in both orders (the chained request +1.40 / +1.54 ms
  against +1.0; (a) to (f) pass: the copy -15.2 ms, the publication -12.4 ms where the tier fills). Reverted in one
  commit (`a089a5c25`); the crates equal `e4de9c804`'s. Receipts `pro-single-p/box/` (1040 of 1040). BOX22 released.
- Next: item 17's revision, pre-registered anew (a split of the publication segment to place the chain's millisecond,
  and a reserve that refills only while the copy would fault); then items 10 to 14 (19 with 14), then 18 and 20; the
  three 5090 cells after the card's reset.
- Local scratch: none.
