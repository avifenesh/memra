# WP-C day 42 (2026-09-24): OWED C3, the contracts-door decision packet read up to lane A day 36

Records only: no card, no engine code. Written while the day-40 `attrib` cell waits for the RTX 5090. The packet
(`DOOR-DECISION-PACKET.md`) is the owner's input for `MEMRA_KV_HOST_CONTRACTS` at its decide-by, 2026-10-05; its last
update was day 39 ("after days 31, 33, A day 27, A days 28 and 29, and C days 38 and 39"). Since then lane A landed days
31 to 36 and the lead ruled 44, 47, 49, 53. It recommends nothing, before or after this update.

## 1. Pre-registration (committed before the packet is edited)

**What is read in.** Lane A days 31 to 36 (`A/DAY31.md` to `A/DAY36.md`) and rulings 44, 47, 49 and 53
(`lead/INTEGRATION-DAY12.md` integ49, integ52, integ54, integ57, integ58):

- A day 31 (ruling 44): the staging back on every post-take refusal, the staging charged to the governor, the
  span-refusal fault cell, the per-slot-class byte tally; the double-park cell on the day-31 tree (target card, BOX3).
- A day 32 (ruling 47): design H, the promote's recurrent planes as H2D spans; B1 to B5 on the target card; the
  pre-H2D and H2D binaries in one hold; DAY28 clause 1b read over its bound on the H2D binary.
- A day 33: design F (the fill as one copy-stream host function) on the 5090; the owner-hold cell; the cause placed
  (the completing poll's SHA-256 over write-combined leases); the D2D half pre-registered, the capture half refuted by
  construction.
- A day 34 (ruling 49): design K (the H2D completion checksum on the hash helper); BOX4 with the d32, d33 and d34
  binaries in one hold; the 5090 readings.
- A day 35 (ruling 49's F decision, integ57): `DAY35 F DECISION -> KEEP` on the 5090; design M refuted and reverted;
  design M' (the bind re-hash on the helper) PASS on the 5090.
- A day 36 (ruling 53): the D2D restore price cell `-> CLOSES` on the target card (BOX5); M' PASS on the target card.

**What changes in the packet.** The status paragraph (a day-42 update line); section 2's bullet "What still runs on the
tick under ON" (the promote's recurrent planes, the fills, the two helper hashes, the D2D restore price); section 4's
target-card and RTX 5090 tables (one row per cost reading of those days, OFF and ON with the receipt file); item 7 (the
items rulings 44, 47 and 53 closed leave the list, what they leave owed stays: the strong-form receipt, hash 1 on the
owner thread, the fill on slower CPUs); section 6 (the naked-default bullet's owner-thread costs on the target card as
the receipts now read them, and the longer-door bullet's candidates); appendix A (the command below). No verdict of
any day is restated in other words; every number is a copy of a line in a named receipt file.

**How every added number is checked.** `day42-packet-lines.py` holds each line the update quotes, beside the receipt
file it comes from (paths under `research/`), and prints `DAY42 PACKET LINES checked=N missing=0 -> PASS` only if every
line is present verbatim in its file. The packet edit is committed only after that PASS, and the command goes into
appendix A. A number that cannot be matched to a receipt line is not added.

**Cross-box rule, restated.** BOX3, BOX4 and BOX5 are different machines of the target card class; their rows are
never subtracted from one another (BOX4's OFF stall reads 93.1 where BOX3's read 85.4, the same cell).

## 2. What was done

- **The check first.** `python3 research/spill-c-20260919/day42-packet-lines.py research` into
  `day42-cpu/packet-lines.log`: `DAY42 PACKET LINES checked=68 missing=0 -> PASS` (52 lines or line fragments present
  verbatim in their named receipt files, and 16 gate-count checks: per target-card sitting of A days 31, 32, 34 and 36,
  4 identity, 2 failure and 2 hit `ALL GREEN` lines, and 1 contract-fault line, 2 on day 36).
- **The packet** (`DOOR-DECISION-PACKET.md`): the status paragraph gains the day-42 update line; section 2 gains the
  bullet "Since A days 31 to 36" after the "What still runs on the tick under ON" bullet; section 4's target-card table
  gains four rows (A day 31 on BOX3, A day 32's same-hold pair on BOX3, A day 34 on BOX4, A day 36 on BOX5) and the RTX
  5090 table six (the day-41 split, A days 33, 34, 35 twice, 36); item 7 gains its day-42 paragraph (what closed, what
  stays); section 6's naked-default and longer-door bullets gain a day-42 line each; appendix A names the check.
- **One addition beyond the registered list, stated.** Section 3 (correctness evidence) was not in section 1's list of
  changes; the owner needs the gate set's verdicts on the trees the new cost rows come from, so one row was added there
  (the gate set on A days 31, 32, 34 and 36), its counts checked by the same script before the commit.
- **`HOSTPREFIX-DOOR.md`** section E gains a DAY 42 paragraph pointing here; the question is unchanged.
- No recommendation, before or after. No number is compared across BOX3, BOX4 and BOX5.

**Status.** C3 is current through lane A day 36 and ruling 53. It stays open in `OWED.md` until the review: any later
lane A or lane C receipt that bears on the contracts door is read in the same way before 2026-10-05.
