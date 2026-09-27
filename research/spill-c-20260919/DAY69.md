# WP-C day 69 (2026-09-25): OWED C3, the contracts-door decision packet read up to lane A day 48

Records only: no card, no engine code. Written while the lead runs DAY68's cell `eclock`. The packet
(`DOOR-DECISION-PACKET.md`) is the owner's input for `MEMRA_KV_HOST_CONTRACTS` at its decide-by, 2026-10-05; its last
read-in was day 62 (lane A days 37 to 41, ruling 54). Since then lane A landed days 42 to 48 and the lead ruled 57
(`lead/INTEGRATION-DAY12.md` integ62, on `main` since #726). The packet recommends nothing, before or after this update.

## 1. Pre-registration (committed before the packet is edited)

**What is read in.** Lane A days 42 to 48 (`A/DAY42.md` to `A/DAY48.md`) and ruling 57:

- Item 4, the strong-form span receipts: S2 (A day 42) and S3 (A day 46) each failed only their price clause and were
  reverted; S4 (A day 48: the source digests batched ahead of the copies, the landed digests at the seal, required
  before the demote publishes, and `take_plane` and `release_device` draining the owner stream only) passes its demote
  and promote clauses, its hump clause and the gate set on the target card (BOX10, a Ryzen 9 9950X3D2 host).
- Item 6, design V (A day 47): the pause sweep's two demote shapes off the tick, the park released when the demote
  publishes; its stall clause and the new pause gate.
- Item 15 closed: G4 stays the single placement at long entries.
- Item 3 closed for the 9950X class (T's target clauses on the 9950X3D2 host).
- Ruling 57: the deferred admission reclaim flush is lane B's owed item inside `MEMRA_ADMIT_BY_MEMORY`; owed by A: the
  RTX 5090 halves of S4 and V and item 16, then items 7 to 14.

**What changes in the packet.** The status paragraph (a day-69 update line); section 2 (a bullet "Since A days 42 to
48"); section 3 (one row: the gate set on A days 42 to 48 on BOX10, the fault gate at 255 `ok:` per arm, the new pause
gate); section 4's target table (S2 and S3 as the failed forms, S4, V, item 15's clauses, T on the 9950X3D2 host);
item 7 (what ruling 57 closes leaves the list, what it leaves owed is listed); section 6 (the naked-default bullet's
owner-thread costs as the day-69 rows read them, the longer-door bullet's candidates); appendix A (the day-69
command). No verdict is restated in other words; every number is a copy of a line in a named file.

**How every added number is checked.** `day69-packet-lines.py`, day 62's form: each quoted line beside its file, a whole
line of a `.log` or `.txt` receipt, a substring of lane A's or the lead's own `.md` record with line breaks read as
spaces, and line counts; it prints `DAY69 PACKET LINES checked=N missing=0 -> PASS` only if every one is present. The
packet edit is committed only after that PASS, and the command goes into appendix A.

**Cross-box rule, restated.** BOX10 (a Ryzen 9 9950X3D2 host) is another machine of the target card class; its rows are
never subtracted from BOX3's, BOX4's, BOX5's, BOX7's or any other box's.

## 2. The read-in, as done

`python3 research/spill-c-20260919/day69-packet-lines.py research` into `day69-cpu/packet-lines.log`: `DAY69 PACKET
LINES checked=32 missing=0 -> PASS`. One check form was added to section 1's three, stated here: the two item-15
clause lines are long, so the packet quotes their leading part and the check requires that part inside one line of
the receipt (a stricter test than the `.md` form, which reads line breaks as spaces). The packet changed where
section 1 said: the status paragraph's day-69 line;
section 2's bullet "Since A days 42 to 48" (ruling 57 quoted, S4's form and why S2 and S3 failed, V, items 15 and 3);
section 3's day-69 gate row (BOX10; the fault gate's 255 `ok:` per arm, the pause gate's 40); section 4's target rows
(S2 and S3 as the failed forms, S4, V, item 15 on A day 43, T on the 9950X3D2 host on A day 44); item 7's day-69
paragraph (closed: items 4, 6, 15, and 3 for the 9950X class; owed verbatim from ruling 57; the reclaim flush moved to
lane B); section 6's two day-69 lines; appendix A's day-69 command. It recommends nothing. `OWED.md` C3 is current
through lane A day 48 and ruling 57; it stays open for any later receipt bearing on the door before 2026-10-05.
