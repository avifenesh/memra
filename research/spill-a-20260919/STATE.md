# WP-A resumable state (2026-09-25, day 53 closed)

- Lane `lane/spill-a-20260919`; worktree `wt-spill-a`. The lead carries days 49 to 53 into integ63.
- Item 17: P and P2 refuted and reverted; blocked on item 14's lease design (the lead's ruling), re-read on top of it.
- Item 21 closed (DAY53): the admission-counter writer tests and the handler tests held different locks; F1
  (`22f1872d6`) orders them (400 of 400 full suites green for the target, no handler 429). New: item 22 (timing tests
  red under load: five named), item 23 (F1's +1.47 s of suite time).
- Next by the lead's order: items 10 to 14 (item 10 per publisher; 19 with 14; then 17 re-read on top), then 18 and
  20; items 22 and 23 placed after 21's class in the ledger, their order the lead's call; the three 5090 cells after
  the card's reset.
- Local scratch: none.
