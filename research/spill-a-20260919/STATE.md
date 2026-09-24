# WP-A day 35 resumable state

- Lane `lane/spill-a-20260919`; Linux worktree `wt-spill-a`. Merged `origin/main` `fa73d0e6c` (integ55, integ56, #706, #708) as `21e082cf3`, no conflict. DAY35 sections 1 and 2 pre-registered `a0f915d8a`; section 7 (M') `e748231e1`.
- F settled on the 5090: `DAY35 F DECISION -> KEEP` (receipts `cbf52cc16`); integ54 records it.
- Design M refuted (DAY35 section 6, red receipts `e727b0072`), reverted `6ce8b1aea`.
- Design M' (hash 2 alone on the helper) `55ae87616`: every clause PASS on the 5090 (DAY35 section 8, receipts `9794558fc`): take-back 8.25 to 0.06 ms, wall -8.15 / -8.40 ms, e2e -7.68 / -8.04 ms, gates ALL GREEN. Integrable.
- Owed: hash 1 on the owner (no off-thread form that keeps the copy phase at one poll); the fill's speed on slower CPUs; the D2D half; the strong-form receipt. No BOX4 sitting was pre-registered for M'.
