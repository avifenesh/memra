# WP-C day 62 (2026-09-25): OWED C3, the contracts-door decision packet read up to lane A day 41

Records only: no card, no engine code. Written while the days 59 to 61 target sitting runs on BOX12. The packet
(`DOOR-DECISION-PACKET.md`) is the owner's input for `MEMRA_KV_HOST_CONTRACTS` at its decide-by, 2026-10-05; its last
read-in was day 42 (lane A days 31 to 36, rulings 44, 47, 49, 53) plus day 54's answer to item 3. Since then lane A
landed days 37 to 41 and the lead ruled 54 (`lead/INTEGRATION-DAY12.md` integ59, on `main` since #723). The packet
recommends nothing, before or after this update.

## 1. Pre-registration (committed before the packet is edited)

**What is read in.** Lane A days 37 to 41 (`A/DAY37.md` to `A/DAY41.md`) and ruling 54:

- A day 37 (item 1, finding 5): the native span cells' parallel failure placed (a pinned or synchronous free, or a
  module load, on one owner thread of a context holds every other owner thread of that context) and fixed in test
  code (one pool context per native cell); the 5090's acceptance and BOX7's all-arm 20 of 20.
- A day 38 (item 2, hash 1 off the owner thread): designs G, G', G'', G''' and G4, each pre-registered; G4 the
  delivered form (every side kernel on the copy stream, the D2H device receipt a framed SHA-256 kernel ahead of the
  copies); its clauses (c), (d), (f) and the gate set on BOX7 and on the 5090; the 5090's (f) FAIL and section 20's
  base-controlled cell.
- A day 39 (item 3, the fill on slower CPUs): design T, the threaded staging fill; its clauses on BOX7 and the 5090.
- A day 40 (item 4, the strong-form receipt): design S refuted on the 5090's price clauses and reverted; its target
  sitting cancelled.
- A day 41 (item 5): the three Sources fault cells, the contract fault gate 229 `ok:` per arm on the 5090 and BOX7.
- Ruling 54: items 1 and 5 close; G4 is the door's form of hash 1 off the owner thread; T is the door's fill
  program; S refuted, item 4 open under `A/DAY40.md` section 7's revision; owed: item 4, items 6 to 15, the
  9950X-class fill reading, the 5090 hump replicate, G''' against G4 at long entries (item 15's cell).

**What changes in the packet.** The status paragraph (a day-62 update line); section 2 (a bullet "Since A days 37 to
41", and ruling 53's "no off-thread form known" for hash 1 marked as superseded by ruling 54, its words kept);
section 3 (one row: the gate set on A days 38 to 41, BOX7 and the RTX 5090, and finding 5's cells); section 4 (target
card: G4 against its base on BOX7, G'' on BOX7 as the failed form, T on BOX7; the RTX 5090: G4 against its base, the
hump cell and the base-controlled hump cell, T, S); item 7 (what ruling 54 closes leaves the list; what it leaves
owed is listed); section 6 (the naked-default bullet's owner-thread costs as the day-38 and day-39 rows read them,
and the longer-door bullet's candidates); appendix A (the day-62 command). No verdict of any day is restated in other
words; every number is a copy of a line in a named receipt file.

**How every added number is checked.** `day62-packet-lines.py` holds each line the update quotes, beside the file it
comes from (paths under `research/`), and prints `DAY62 PACKET LINES checked=N missing=0 -> PASS` only if every line is
present in its file: verbatim as a whole line of a `.log` or `.txt` receipt; for a verdict that exists only in lane
A's own record (the 5090 half of finding 5 is written in `A/DAY37.md` section 9, from `pool/run.log`'s raw runs), as a
substring of that `.md` file with its line breaks read as spaces. The packet edit is committed only after that PASS,
and the command goes into appendix A. A number that cannot be matched to a line of a named file is not added.

**Cross-box rule, restated.** BOX7 is the BOX4-class slow host of the target card class; its rows are never
subtracted from BOX3's, BOX4's, BOX5's or BOX8's, and the 5090's rows never from any target-card row.
