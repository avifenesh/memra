# WP-A resumable state (2026-09-25, day 52 read and closed)

- Lane `lane/spill-a-20260919`; worktree `wt-spill-a`. integ62 (#726, main `5228ff0cd`) merged A days 42 to 48; the lead
  carries days 49 to 52 into integ63.
- Item 17: P (DAY51) and P2 (DAY52) each passed (a) to (f) and failed (g); both reverted. The publication split
  (`5990945cd`, log only) stays and placed P's millisecond in the replaced twin's pinned lease frees. Item 17 is
  proposed blocked on item 14's lease design (with 19). Receipts `pro-single-p2/box/` (1091 of 1091). BOX25 released.
- Next: item 21 (the server test failure under the full suite: a reproduction under the suite's concurrency,
  pre-registered), then items 10 to 14 (19 with 14; item 17 re-read on top of 14), then 18 and 20; the three 5090
  cells after the card's reset.
- Local scratch: none.
