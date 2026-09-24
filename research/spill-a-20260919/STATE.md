# WP-A resumable state (2026-09-25, during the BOX7 sittings)

- Lane `lane/spill-a-20260919`; Linux worktree `wt-spill-a`; tip `3c57fdfca` (G''' `9ab5c1265`, T `0153316d4`, day 41's
  K arms `4ca4bb36e`, their records). BOX7 (one RTX PRO 6000 WS, EPYC 9B14 class, 92 CPUs) is the lead's box for this
  lane: receipts under `/root/spill-receipts/a-day38` (the day-38 sitting and the diagnosis diag to diag8) and
  `/root/spill-receipts/a-g3` (the G''' sitting); the box's clone `/root/wt-a`. Mirror file for file against a box
  sha256 manifest, then remove both receipt roots and `/root/wt-a`, and report `BOX7 RELEASED`.
- OWED (`OWED.md`): item 1 closed (BOX7 all arm 20 of 20). Item 2: G'' failed (d) on BOX7 (the tenant's per-demote
  decode hump); the bisection (DAY38 13a to 13k) placed it on two non-owner streams running kernels; G''' (every engine
  kernel on the receipt stream, the D2D classes with it; `d2h-delay` a host-side hold) pre-registered (section 14) and
  built; its 5090 sitting (`rtx5090-day38/g3-card-run.sh`) then BOX7's (`pro-single-g3/`) owed. Item 3: design T built,
  5090 PASS (DAY39 section 6), target cell rides the G''' sitting (`item3.sh`). Item 5: 5090 PASS (DAY41 section 2),
  target rides the G''' sitting's fault gate. Item 4: device-side span receipt picked (DAY40), design waits for G'''.
  Items 6 to 14 open.
- Local scratch: `/tmp/wt-a-d38g3` (the G''' 5090 binaries), `/tmp/wt-a-d38g3-base-target` (the base build's target
  dir), `/tmp/wt-a-d38x` (a scratch worktree at `b214bd2cf` for the diag patches). Remove when their cells are banked.
