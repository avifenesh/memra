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
program, the exact device name and the window. The body is the probe's table lines of the
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

**No stored table.** Outside the release battery the check is not applied and the gate says so.
With `--require-floor` (the battery) the cell is `NO-STORED-ROWS`, refused, on a device of a
maintained class or an unknown name, and `NOT-JUDGED` on a device of a class nobody maintains:
the same reading as a missing floor row.

**The model digest.** The key needs the artifact's sha256, and hashing 15 to 20 GB on every gate
run is slow. The gate reads it from a cache under `${XDG_CACHE_HOME:-~/.cache}/memra/` keyed by the
resolved path, the size and the modification time; `--rehash-model` hashes the bytes again.

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
- **One table per device class.** The tables differ across devices: decode is byte-identical
  between the RTX 5090 Laptop GPU and GB10 but differs on the RTX PRO 6000 Max-Q after row 94,
  and prefill is byte-identical between the two sm_120a cards but differs on GB10 after row 14
  (memra#1203). The key is the exact device name.
- **The model's file name as its identity.** A re-quantized or replaced file under the same name
  would read as a difference of numbers; the digest makes it "no stored table".
- **Hashing the model on every run.** The cache with `--rehash-model` covers the same ground; a
  file rewritten with the same size and modification time is the case the flag exists for.
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
<!-- SEEDS-EVIDENCE:END -->

## Not measured yet

The premise is byte stability per device. The records above show it across builds, processes
and days on every seeded cell, and across units for Ornith on GB10 and all four cells on the RTX
PRO 6000 Max-Q. Not yet run with this branch's gate: the gate window twice from one build in two
processes and once more after another cell used the card, per device class, model and arm
(12-row runs of 40 to 42 s on the RTX 5090 Laptop GPU), and Qwen3.8-27B on the second GB10 unit.
If one moves, the key needs the unit or the source needs a fix, and that is reported first.
