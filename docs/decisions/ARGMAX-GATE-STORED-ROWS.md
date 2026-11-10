# The argmax margin gate compares the window with main's stored rows (2026-11-10)

**Status:** adopted (memra#1204), after [ARGMAX-GATE-GAP-SHIFT-FLOOR.md](ARGMAX-GATE-GAP-SHIFT-FLOOR.md)
(memra#1199). Receipts:
[`research/argmax-stored-rows-20261110/`](../../research/argmax-stored-rows-20261110/).

## Question

The floor rule judges a candidate's prefill against its own decode, inside one build, against
the largest gap shift main's two programs show where they agree. On the roster that floor is 3.1
to 4.5 logits, set by one row of the prompt each time. So a candidate that moves a token by 1 to
3 logits inside the window passes, and so does a token change at a near tie. Main's rows at the
same positions are known: the calibration measured them. Can the gate compare with them
directly, without building or running main at gate time, and with what tolerance?

## Decision

**A stored table per cell.** `tools/argmax-margin-rows/` holds one `.rows` file per cell, keyed
by the model's name and the sha256 of its bytes, the prompt's name and sha256, the decode
program, the exact device name, the window and where the MoE experts ran (`resident`, `slru` or
`none`, read from the log's `[moe] resident-experts decision` lines). The body is the probe's table lines of the
window as printed. The gate looks a table up by the whole key, never by file name. A changed
prompt or artifact reads as "no stored table", not as a difference. Two tables for one key, or a
file that does not parse, are errors.

**Byte comparison, no tolerance.** The candidate's printed table lines must equal the stored
ones. A differing row prints both sides (top-1 ids, margins, spread and the probe's `detail`
line) and fails the cell. There is no tolerance knob: every table measured twice on one device
so far is byte-identical, across processes, builds and other cells on the card in between.

**Both checks.** A cell passes only if the floor rule and the stored rows both pass. The floor
rule still judges prefill against decode within the candidate; the stored rows judge the
candidate against main.

**The expert placement is part of the key.** An MoE model whose experts do not fit the free
memory at load (a shared card) runs them through the SLRU cache, another dispatch program. On
the RTX 5090 Laptop GPU, Ornith-1.5-35B-A3B with `MEMRA_MOE_RESIDENT=0` passes the floor rule
and differs from main's resident rows at all 12 positions, eager and batch. So a run whose
placement differs from main's reads as no stored table ("not comparable", named in the
refusal), never as a difference, and the release battery waits for the model's size plus
2,048 MiB free, above the resident planner's 2.0e9-byte reserve. The independent review found
this case; it was not covered by any record (every RTX 5090 Ornith record ran resident).

**No stored table.** Outside the release battery the check is not applied and the gate says so.
With `--require-floor` (the battery) the cell is `NO-STORED-ROWS`, refused, on a device of a
maintained class or an unknown name, and `NOT-JUDGED` on a device of a class nobody maintains:
the same reading as a missing floor row. Only a cell the floor rule passed is refused or not
judged this way: a floor-rule failure is a FAIL whether or not main's rows are stored, and a
differing row fails whatever the judge's exit code. The release-qualification and cell-evidence
checkers require one `STORED-ROWS: identical` line in each passing argmax cell.

**The model digest.** The key needs the artifact's sha256, and hashing 15 to 20 GB on every gate
run is slow. The gate reads it from a cache under `${XDG_CACHE_HOME:-~/.cache}/memra/` keyed by the
resolved path, the size, the modification and change times and the inode; `--rehash-model` hashes
the bytes again.

**Where the rows come from.** The floor row's calibration log already holds main's rows. The
stored table of a battery cell is the last 12 rows of that receipt, and the census
(`tools/check-argmax-floors.py`) checks it: its receipt field must be the floor row's, and its
rows the receipt's last 12, byte for byte. `--calibrate --rows-out FILE --commit C` writes a
stored table with the floor row.

**Changing the rows.** A bit-identical change shows no difference. A change of numeric class
that a decision record accepts recalibrates in the same PR: new calibration log, floor row and
stored table, with the old and the new rows of the window shown (docs/TESTING.md).

**Teeth.** The canary writes a stored table from the measured window, moves one margin of its
last row by 1 logit in a copy, and requires the comparison to call the unmoved copy identical
and to name exactly the moved row.

## Rejected

- **A tolerance.** No repeat on one device has moved a byte, so any bound would be a guess. If a
  device turns out not to be byte-stable, the measured run-to-run maximum for that cell is the
  only acceptable bound, and only after the source of the movement is known.
- **Building and running main at gate time.** It costs a build and a GPU run per cell and gives
  the same answer while main is byte-stable on the device.
- **Leaving the expert placement out of the key.** Measured above: the SLRU program's window
  differs at every row, so a shared card would read as a change of numbers.
- **One table per device class.** The tables differ across devices: decode is byte-identical
  between the RTX 5090 Laptop GPU and GB10 but differs on the RTX PRO 6000 Max-Q after row 94,
  and prefill is byte-identical between the two sm_120a cards but differs on GB10 after row 14
  (memra#1203). The key is the exact device name.
- **The model's file name as its identity.** A re-quantized or replaced file under the same name
  would read as a difference of numbers; the digest makes it "no stored table".
- **Hashing the model on every run.** The cache covers the same ground: a rewrite changes the
  file's change time even when its size and modification time are kept, and `--rehash-model`
  hashes regardless. A key of size and modification time alone was rejected in review: a file
  rewritten with both set back kept its stale digest.
- **Storing the whole prompt.** The gate measures its window only; the whole table stays in the
  calibration receipt.

## Evidence

CPU, red and green arms (`cargo test -p memra-engine --lib argmax_margin`, `cargo test -p
memra-engine --bin argmax-margin-probe`, `python3 tools/test_argmax_margin_gate.py`):

| table against main's stored rows | floor rule | stored rows |
|---|---|---|
| identical | pass | pass, `STORED-ROWS: identical` |
| one margin moved at one row (by 1 logit in the probe's tests, 0.2 in the wrapper's), gap shift inside the floor | pass | fail, `ROW-DIFF` with both sides |
| a token change at a near tie | pass (a near tie within the budget) | fail, `ROW-DIFF` |
| other prompt bytes, another artifact, device, decode program or window | (its own floor row) | `STORED-ROWS: none` |

<!-- SEEDS-EVIDENCE:START -->
The seeds are the last 12 rows of the ten battery floor rows' receipts. Each equals the 12-row
gate table of every cell-evidence record of its exact device: 197 tables from 85 records, each
record at its own commit (39 on the RTX 5090 Laptop GPU, 40 on GB10, 6 on the RTX PRO 6000
Max-Q). The GB10 records come from both units and the RTX PRO 6000 records from three cards, so
units of one device name give the same table on these cells, and the key does not need the unit.

On a real table from another build (v0.141.0's 12-row Ornith eager window on the RTX 5090 Laptop
GPU): as measured, `identical`, exit 0; one decode margin moved by 1 logit, floor rule pass and
`ROW-DIFF`, exit 1; one row flipped at a near tie, floor rule pass (a near tie within the
budget) and `ROW-DIFF`, exit 1 (`research/argmax-stored-rows-20261110/red-green/`).

Repeats with this branch's gate on the RTX 5090 Laptop GPU (`--require-floor`, the PR's probe
build): each of the four roster cells twice in two processes, then once more after the other
three cells used the card. All 12 runs read `STORED-ROWS: identical` and pass; their tables
hash to the same values as the cell-evidence records'.
<!-- SEEDS-EVIDENCE:END -->

## Not measured yet

The premise is byte stability per device and placement. The records show it across builds,
processes and days on every seeded cell, and across units on GB10 (both units for Ornith and
for Qwen3.8-27B, eager 26 and 14 records per unit; each batch arm rests on 20 records from one
unit and 1 from the other) and on three RTX PRO 6000 Max-Q cards; the repeats show it within one build on the RTX 5090 Laptop GPU. The GB10
repeats (two processes, then after other cells) are in `research/argmax-stored-rows-20261110/`
when run; the RTX PRO 6000 has no repeat with this branch's gate. A stored table for another
placement than resident, and for a gemma model, is not seeded: a cell that would need one reads
as no stored table.
