# WP-A resumable state (2026-09-25, day 52, stopped at NEED TARGET CARD for item 17's revision)

- Lane `lane/spill-a-20260919`; worktree `wt-spill-a`. integ62 (#726, main `5228ff0cd`) merged A days 42 to 48; the lead
  carries days 49 to 51 into integ63.
- Item 17: P refuted on DAY51 (g) and reverted (`a089a5c25`). Its revision (DAY52): step 1, the demote publication
  split, log only (`5990945cd`, the base arm); step 2, design P2, P with an arming rule (`f9c849389`); CPU cells green
  (server lib 928 passed). The sitting `pro-single-p2/`: `build.sh <tip> 5990945cd`, then `driver.sh`; about 2.6 hours
  on one RTX PRO 6000 Blackwell with the 27B artifact, a 9950X-class host (DAY51's). The reading:
  `reading-day52.log`, last line `DAY52 P2 -> ..`, and `DAY52 PLACING -> ..` (where P's millisecond sits).
- Next after P2's reading: items 10 to 14 (19 with 14), then 18 and 20; the three 5090 cells (S4's half, V's half,
  item 16) after the card's reset.
- Local scratch: none.
