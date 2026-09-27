# WP-C day 87 (2026-09-27): OWED C3, the contracts-door decision packet read up to lane A day 70 and ruling 65

Records only: no card, no engine code. Written while BOX43 carries DAY86's load before `slow86`. The packet
(`DOOR-DECISION-PACKET.md`) is the owner's input for `MEMRA_KV_HOST_CONTRACTS` at its decide-by, 2026-10-05; its last
read-in was day 69 (lane A days 42 to 48, ruling 57). Since then lane A landed days 49 to 70 and the lead ruled 58 to 65
(`lead/INTEGRATION-DAY12.md`, integ63 to integ70). Both are on `main` at `359e850d0`, the tree this read-in uses. The
packet recommends nothing, before or after this update.

## 1. Pre-registration (committed before the packet is edited)

**What is read in.** Lane A days 49 to 70 (`A/DAY49.md` to `A/DAY70.md`) and rulings 58 to 65, item by item:
- items 7 and 8, the demote's first touch (A day 49), and item 9, the promote's tick placement (A day 50);
- item 17, the payload reserve: design P (A day 51) and P2 (A day 52), both refuted and reverted, then P2 re-applied on
  top of L' (A day 67) and made the naked program on the target card (ruling 65);
- item 10, the publishes on the tick: priced (A day 54), then the fanout publisher B1 (A day 59, adopted, ruling 62)
  and its split's scope (A day 66);
- item 11, Move 1's decision cell (A day 60), item 12, design W (A day 61), item 13, R2 refuted and R1 adopted (A day
  62), items 14 and 19, L refuted and L' adopted (A day 63), item 18, F reverted (A day 64), item 20, T-H reverted (A day
  65);
- items 21 to 25, the server tests and the route book (A days 53, 55 to 58), each closed;
- the owed RTX 5090 halves (A day 68);
- the two correctness defects the integ69 review found under L' and their designs (A day 69, the purge scrub P; A day
  70, one unit for the LRU and the requests, Q);
- rulings 58 to 65 where they bear on the door, verbatim.

**What changes in the packet.** The status paragraph (a day-87 line); section 2 (a bullet "Since A days 49 to 70",
the naked program on the target card as the rulings state it); section 3 (one row: the door's gate lines in the latest
integration GPU battery the lead recorded); section 4's target rows (the adopted forms' registered clauses as read, the
refuted forms with the clause each failed); section 5 (the two review-found defects, placed and fixed, as findings the
review weighs); item 7 (what the rulings close leaves the list; what they leave owed is listed verbatim); section 6
(the naked-default bullet's programs as ruled); appendix A (the day-87 command). No verdict is restated in other words;
every number and verdict is a copy of a line in a named file.

**How every added line is checked.** `day87-packet-lines.py`, day 69's three forms (a whole line of a `.log` or `.txt`
receipt, after any timestamp tab; a substring of an `.md` record with its line breaks read as spaces; a leading part of
one receipt line), reading every file through `git show 359e850d0:research/<path>` so the check does not depend on this
lane's checkout; it prints `DAY87 PACKET LINES checked=N missing=0 -> PASS` only if every line is present. The packet
edit is committed only after that PASS, and the command goes into appendix A.

**Cross-box rule, restated.** Each box's rows stay its own; no row of one machine is subtracted from another's.

## 2. The read-in, as done

`python3 research/spill-c-20260919/day87-packet-lines.py 359e850d0` into `day87-cpu/packet-lines.log`: `DAY87 PACKET
LINES tree=359e850d0 checked=60 missing=0 -> PASS` (35 whole receipt lines, 2 leading parts of a receipt line, 23
substrings of lane A's and the lead's records). A second pass read every backticked or quoted span the day-87 additions
carry against the checker's list: every verdict, clause, reading and price line among them is one of the 60 (the
section 6 figures are parts of those lines). The packet changed where section 1 said: the status paragraph's day-87
line; section 2's bullet "Since A days 49 to 70" (rulings 58, 59, 60, 62, 64 and 65 quoted, W's 5090 half and its
revert, the four adopted forms named); section 3's day-87 row (integ70's GPU battery on BOX43 and integ69 run 4's
purge cells); section 4's nine target rows (items 7 and 8, P and P2's refuted forms, P2 on L' with its repeated unit
step, item 10's price, B1, W, R2 and R1, L and L', F and T-H); section 5 item 2's day-87 lines (cell (i) on the current
tree, not met again, and A's candidate for the owner) and a new item 8 (the purge and the ledger defects under L',
placed, fixed as designs P and Q, and the review's two latent hazards); item 7's day-87 paragraph; section 6's two
day-87 lines; appendix A's day-87 command. It recommends nothing.

Beside it, stated plainly: item 7's owed list is what the rulings and A's records name; it is not this lane's
reading of what the door needs. `OWED.md` C3 is current through lane A day 70 and ruling 65; it stays open for any
later receipt bearing on the door before 2026-10-05 (the RTX 5090 halves A day 68 registered are the next).
