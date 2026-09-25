# WP-A resumable state (2026-09-25, stopped at an integrable milestone on the target card; the 5090 down)

- Lane `lane/spill-a-20260919`; worktree `wt-spill-a`; merged with `origin/main` `d6515742f` (integ61) as `35978fc42`
  (one FLAGS.md conflict, both rows merged by content); CPU suites green on the merge (server lib 920, engine lib 570,
  the tier crate, clippy, fmt, check-flags).
- Item 4: S2 and S3 refuted on the target card's (c) and reverted (`16904e97d`, `7197c1a9d`); design S4 (`ef4b097ad`:
  S3 plus the release paths draining the owner stream only) passes (a) to (e) on the target card (DAY48 section 3).
- Item 6: design V (`a324503df`, on S4) passes (a) to (d) on the target card (DAY47 sections 3, 3a, 3b).
- Item 15 closed (DAY43 section 2: G4 stays the single placement). Item 3 closed for the 9950X class (DAY44 section 2).
- The 5090 reads `GPU requires reset` since 01:25Z (Xid 119, then 154); owed there: S4's half (`rtx5090-day42/` with
  S4's tip), V's half, item 16 (`rtx5090-day45/`). The owner's reset.
- Receipts mirrored (all against the box's own manifests): `pro-single-i15/box`, `pro-single-t9950/box`,
  `pro-single-s2/box-design-s2`, `box-design-s3`, `box-design-s4`, `pro-single-v/box`.
- Next in the ledger: items 7 to 14 (item 13's capture-settle hold behind other copy-stream work is S4's precise drain
  in part: the capture settle reads 0.16 ms on S4's demote boots).
- BOX10 released: /root/wt-a, /root/spill-receipts and the /tmp scratch removed, no lane process, no compute app.
- Local scratch: none.
