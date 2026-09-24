# WP-A day 35 resumable state

- Lane `lane/spill-a-20260919`; Linux worktree `wt-spill-a`; base `05fbec3b2` (integ54). DAY35 sections 1 and 2 pre-registered `a0f915d8a`; section 7 (M') `e748231e1`.
- F settled on the 5090: `DAY35 F DECISION -> KEEP` (receipts `cbf52cc16`). integ54 records it.
- Design M refuted on the 5090 (DAY35 section 6, red receipts `e727b0072`): (c) PASS, (d) o2 FAIL (+26.50 over +25.0), gates red (identity default-on, failure-on, fault-default); cause placed (M1's reply adds a poll to the copy phase; a copy-phase hit does not park). Reverted whole `6ce8b1aea` (the code byte-identical to `a0f915d8a`).
- Design M' (hash 2 alone in the Hashing job) coded `55ae87616`, binary `68c7dd8fa746fef0..`; its 5090 cell (`rtx5090-day35/m2/`, `m-card-run.sh`, reader `day35m-reading.py --m2`): attempt 1 NOT RUN (a foreign compute app held the card through the idle check, 04:33:43Z to 04:48:44Z); the relaunch waits outside the lock for an idle card.
- If M' fails: revert it; the tip returns to the F-settled code. Owed: hash 1 stays on the owner; the fill's speed; the D2D half; the strong-form receipt.
