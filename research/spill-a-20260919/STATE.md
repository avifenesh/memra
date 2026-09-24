# WP-A resumable state (2026-09-25, during the BOX7 G4 and S sittings)

- Lane `lane/spill-a-20260919`; worktree `wt-spill-a`. Code tips: G4 `26676c037` (item 2's delivered placement: one side
  stream), S `a40e5b334` (item 4: device-side span receipts on the copy stream), T `0153316d4` (item 3), day 41's K arms
  `4ca4bb36e` (item 5). Records: DAY38 sections 11 to 20a (the hump, its bisection, G''' and G4), DAY39 sections 4 to 6
  (T), DAY40 sections 3 to 4 (S), DAY41 (item 5).
- BOX7 (the lead's box for this lane): receipt roots `/root/spill-receipts/a-day38` (the day-38 sitting and diag to
  diag8), `a-g3` (G''' sitting, done), `a-g4` (G4 sitting with item 3's cell, running), `a-s` (S sitting: its build queued
  after the G4 driver; its driver to start after the 5090 S half reads). Clone `/root/wt-a`. Before release: mirror all
  four roots file for file against a box sha256 manifest (binaries and nsys reports by hash only), remove them and the
  clone, confirm no process of this lane remains, report `BOX7 RELEASED`.
- 5090: the S card run (`rtx5090-day40/s/`) in its hold. Local scratch: `/tmp/wt-a-d38g4` (G4, gpp, base binaries),
  `/tmp/wt-a-d38g4-base-target`, `/tmp/wt-a-d38g34` (the G''' rebuild), `/tmp/wt-a-d40s` (S binaries); remove when banked.
- OWED: 1 closed; 2 G4 5090 (a)-(e) PASS, (f) PASS in the base-controlled hold, BOX7 (c), (d), (f) PASS so far; 3 T 5090
  PASS, target cell in the G4 sitting; 4 S built, sittings running; 5 closed; 6 to 15 open (15 new: the receipt kernel's
  price at long entries). Owner items: the 5090's (f) reading (base-controlled hold), G''' against G4 for long entries.
