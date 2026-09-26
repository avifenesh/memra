# WP-A day 60: OWED item 11, Move 1's decision cell (i) on the current tree, both classes, same window

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Every cell `executed-not-qualified`. Prepared while DAY59's 5090
cell waits for the card (text and scripts only; nothing here runs before DAY59's reading).

## 1. Pre-registration

**The item** (`OWNER-THREAD-OFFLOAD.md` Move 1 cells, day 16's clause, verbatim): "(i) The census stall cell repeated
with the second stream: the tenant's ITL max during the demote and promote arms against the day-16 baseline, N=5 per
arm per order, both orders, door ON in both ... Decision clause: `stall_median(second stream) <= idle p99` of the same
sitting for both classes". Lane C ran it on day 29 (`research/spill-c-20260919/DAY29.md`) on tree `653c997f4` against
the day-16 tree `1646d421b`: `DAY29 CELL(i) CLAUSE class=demote stall_median(second stream)=150.0 idle_p99_sitting=14.9
-> clause_not_met` (and the promote class the same way); ruling 47 carries the cell as owed on the current tree, the
clause read as written.

**The cell, unchanged but for arm X's tree.** C's day-29 scripts, verbatim (`research/spill-c-20260919/`:
`day29-stall-cell.sh`, `day29-box-run.sh`, `day29-stall-reading.py`): arm X the lane's current tip (the copy-stream
program with everything since, S4's receipts, V, the split lines), arm Y the day-16 tree `1646d421b` in its own
worktree (the owner-stream program); every boot the day-16 `on` boot; `stall_cell.py --mode demote` then `--mode
promote`, N=5 per arm per order inside each boot; order 1 X Y .. x5, order 2 Y X .. x5; one dry boot of Y first; one
RTX PRO 6000 Blackwell, one collector hold; the 27B NVFP4 MTP artifact.

**The rule, as written (no bound of mine).** Per class: `stall_median(arm X) <= idle p99 of the same sitting` is the
clause; the reader prints it met or not. The same-window pair `y_minus_x` per class and order is a reading.

**What it can say, stated now.** The clause compares the whole intruder's stall with the idle tenant: the demote and
promote intruders prime a fresh prompt on the tick (the day-16 cell's shape), so the clause is expected **not met** on
any tree while the intruder primes on the tick (day 29: 150.0 against 14.9); the pair then reads how much of the
class's own on-tick share the copy stream removed on the current tree. If the clause is not met, item 11 closes with
the reading, and whether a cell that isolates a class from its intruder's prime (Move 2's owed item 3, "a stall cell
that isolates each class") replaces it is the lead's call.

**What each card decides.** The target card (the day-16 clause's own rig class); the 5090 has no role.

**Budget.** 0.15 agent-day to prepare; about 40 minutes of a target card.
