# WP-A resumable state (2026-09-25, day 51, stopped at NEED TARGET CARD for item 17)

- Lane `lane/spill-a-20260919`; worktree `wt-spill-a`. integ62 (#726, main `5228ff0cd`) merged A days 42 to 48; main
  is merged into the lane (`ff41669d1`).
- Items 7 and 8 closed (DAY49 section 3, the lead's run on BOX10: `-> H attributed`, `-> b1 step attributed to H`).
  Item 9 closed (DAY50 section 3: one stretched tick, tick 2 absent 30 of 30 on the tip).
- Item 17 (H's remedy): design P pre-registered (DAY51 section 1, `e4de9c804`, the base arm) and built (`d82738c14`,
  CPU cells green: server lib 925 passed, clippy, fmt, diff check, flags census). The sitting `pro-single-p/` is
  prepared: `build.sh <tip> e4de9c804`, then `driver.sh`; about 2.2 hours on one RTX PRO 6000 Blackwell, the 27B
  artifact, a 9950X-class host (DAY49's class). The reading: `reading-day51.log`, last line `DAY51 P -> ..`.
- New owed items from the readings: 18 (S4's H2D destination digests off the promote's landing), 19 (pinned
  allocations on the owner thread: 19.3 ms per long demote, about 20 ms at a context's first demote), 20 (the hash
  helper's per-payload work on one thread).
- Order of work after P's sitting: items 10 to 14 (19 with 14), then 18 and 20; the three 5090 cells (S4's half, V's
  half, item 16) after the card's reset.
- Local scratch: none.
