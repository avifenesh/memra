# WP-A resumable state (2026-09-25, stopped at an integrable milestone)

- Lane `lane/spill-a-20260919`; worktree `wt-spill-a`; tip merged with `origin/main` `9eb1326e9` (as `cc136e129`), CPU
  checks green (engine tier_transfer 15, server lib 911, tier crate, clippy, fmt, check-flags, diff --check).
- Integrable: item 2 as G4 (`26676c037`), item 3 as T (`0153316d4`), item 5 (`4ca4bb36e`), item 1 (`ba5da705d`, earlier);
  S (item 4) reverted (`8a044a1c0`) and owed. Records: DAY37 to DAY41, OWED.md; BOX7's receipts mirrored under
  `pro-single-day38/box`, `pro-single-g3/box`, `pro-single-g4/box`, `pro-single-s/box` (2547 files checked against the box
  manifests; binaries and nsys reports by hash, `MIRROR-CHECK.txt`). BOX7 released: its receipt roots and clone removed.
- Owner items: the 5090's (f) for G4 read from the base-controlled hold (DAY38 section 20a); G''' against G4 for long
  entries (item 15's cell, needs a target card).
- Next in the ledger: item 4's revision (DAY40 section 7: the span receipt off the landing path, required at
  publication, batched launches; pre-register, 5090, then a target card); then items 6 to 15.
- Local scratch: none (every `/tmp/wt-a*` removed).
