# Main's stored rows for the argmax margin gate (memra#1204)

The argmax margin gate after memra#1199 judges each row of a cell's window against a floor that
main's own two programs set on the whole prompt. On the roster that floor is 3.1 to 4.5 logits,
so a candidate that moves a token by 1 to 3 logits inside the window passes it. This lane adds
the second check: the candidate's window against main's own table for the same cell, line by
line. Decision record: [`docs/decisions/ARGMAX-GATE-STORED-ROWS.md`](../../docs/decisions/ARGMAX-GATE-STORED-ROWS.md).

## What is stored

`tools/argmax-margin-rows/` holds one `.rows` file per cell: a header (model name and the sha256
of its bytes, prompt name and sha256, decode program, exact device name, window, the commit of
the binary that measured the rows, the receipt) and then the probe's table lines of the window
as printed. `memra_engine::argmax_margin::StoredTable` reads and writes the form; the gate looks
a table up by the whole key, never by file name.

The seeds are the battery cells that have a floor row: the last 12 rows of each floor row's
calibration receipt (main 031f146690's probe over the whole of `board-2048`,
`research/argmax-gate-rule-20261110/calibration/`), written by the probe's own
`--judge <receipt> --calibrate --rows-out` (`tools/seed_rows.sh`). The model digest of each
device's artifact is in [`model-sha256.tsv`](model-sha256.tsv).

<!-- SEEDS:START -->
Ten seeds, every battery cell that has a floor row at this commit (Ornith-1.5-35B-A3B on the
RTX 5090 Laptop GPU, the RTX PRO 6000 Max-Q and GB10; Qwen3.8-27B on the first two; eager and
batch each). The two GB10 Qwen3.8-27B cells get theirs with their floor rows. The model files on
all three devices have the same sha256 (`model-sha256.tsv`), so the key differs only by device.
`seeds.txt` is the run's output.

Each seed against the 12-row gate tables of the cell-evidence records of its exact device
(`tools/against_records.py`, `against-records.md`):

| stored table | records with this cell | identical | devices (distinct uuid) |
|---|---|---|---|
| Ornith eager, RTX 5090 Laptop GPU | 39 | 39 | 1 |
| Ornith batch, RTX 5090 Laptop GPU | 18 | 18 | 1 |
| Qwen3.8-27B eager, RTX 5090 Laptop GPU | 39 | 39 | 1 |
| Qwen3.8-27B batch, RTX 5090 Laptop GPU | 18 | 18 | 1 |
| Ornith eager, RTX PRO 6000 Max-Q | 6 | 6 | 3 |
| Ornith batch, RTX PRO 6000 Max-Q | 5 | 5 | 3 |
| Qwen3.8-27B eager, RTX PRO 6000 Max-Q | 6 | 6 | 3 |
| Qwen3.8-27B batch, RTX PRO 6000 Max-Q | 5 | 5 | 3 |
| Ornith eager, GB10 | 40 | 40 | 2 |
| Ornith batch, GB10 | 21 | 21 | 2 |

Every record equals its seed: 197 tables from 85 records (39 on the RTX 5090 Laptop GPU, 40 on
GB10, 6 on the RTX PRO 6000 Max-Q), each record at its own commit, on both GB10 units and on
three RTX PRO 6000 Max-Q cards. Across the builds these records cover, no roster cell on these
devices has moved a byte of its window.

Red and green on a real table (`tools/red_green.sh`, `red-green/`): v0.141.0's 12-row Ornith
eager table on the RTX 5090 Laptop GPU (another build than the seed's), judged with the floors
table and the stored rows:

| table | floor rule | stored rows | exit |
|---|---|---|---|
| as measured | 0 flips, pass | identical | 0 |
| decode margin of row 2047 moved by 1 logit (1.157 to 2.157) | 0 flips, pass | `ROW-DIFF pos=2047`, DIFFERENT (1 of 12) | 1 |
| row 2037 flipped to another token at decode margin 0.05 | 1 near tie within the budget, pass | `ROW-DIFF pos=2037`, DIFFERENT (1 of 12) | 1 |
<!-- SEEDS:END -->

## Why a byte comparison, with no tolerance

Every table measured twice on one device so far is byte-identical
(`research/argmax-gate-rule-20261110/README.md`, "Reproduction"):

- RTX 5090 Laptop GPU, Ornith-1.5-35B-A3B eager, all 2,045 rows of `board-2048`: main's probe
  and #1202's probe, 80 minutes apart with other cells on the card between.
- The last 12 rows of every whole-prompt calibration table equal the 12-row table of a
  cell-evidence record of its device from another build (RTX 5090 Laptop GPU, RTX PRO 6000
  Max-Q and GB10; the table below checks every seed against every record of its device).

Across devices the tables differ, so a stored table is valid for its exact device only (whole
prompt, the two sides of the table compared as printed; memra#1203):

| model, arm | side | RTX 5090 Laptop GPU against RTX PRO 6000 Max-Q | RTX 5090 Laptop GPU against GB10 |
|---|---|---|---|
| Ornith eager | prefill | 2,045 of 2,045 identical | 12 identical (rows 3 to 14), top-1 differs at 177 |
| Ornith eager | decode | 92 identical (rows 3 to 94), top-1 differs at 151 | 2,045 of 2,045 identical |
| Ornith batch | decode | 92 identical, top-1 differs at 146 | 2,045 of 2,045 identical |
| Qwen3.8-27B eager | decode | 92 identical, top-1 differs at 28 | not measured |

A tolerance would be a measured run-to-run maximum for the cell. None has been needed: no
repeat has moved a byte. If a device turns out not to be byte-stable, the rows that move and by
how much come first, then the source, then the choice between fixing it and a measured bound.

## Repeats on GPU and the expert placement

**RTX 5090 Laptop GPU, this branch's gate** (`repeats/rtx5090-laptop/`, the PR head's probe,
`--require-floor`, 12-row window): each roster cell twice in two processes in one lease, then
once more in a later lease after the other three cells had used the card.

| cell | runs | table sha256 (first 16) | stored rows | gate |
|---|---|---|---|---|
| Ornith eager | 3 | `706af873d259891a` all three | identical | PASS |
| Ornith batch | 3 | `e356081d72028d74` all three | identical | PASS |
| Qwen3.8-27B eager | 3 | `b25c41e5e8e34315` all three | identical | PASS |
| Qwen3.8-27B batch | 3 | `4083a49bdc9adb40` all three | identical | PASS |

The four hashes are the cell-evidence records' (research/argmax-gate-rule-20261110/README.md).
The first run of a model hashed its file and later runs read the cache (`model sha256: ...
(hashed)`, then `(cached)`).

**The expert placement (the independent review's question).** An MoE model loads its experts
on the card only when the free memory covers them, the trunk and a 2.0e9-byte reserve; otherwise
the SLRU cache serves them, another dispatch program. With `MEMRA_MOE_RESIDENT=0`, Ornith on the
RTX 5090 Laptop GPU passes the floor rule and differs from main's resident rows at all 12
positions, eager and batch (`repeats/rtx5090-laptop/rep-or-*-slru*/`). A second SLRU eager run gives
the same table (`76dce3c92cf0ef8f`), so the SLRU path is a second stable program, not noise. No record had measured that
path: every RTX 5090 Ornith record ran resident. So the placement is part of the key, read from
the log's `[moe] resident-experts decision` lines, and a run placed otherwise reads as "not
comparable", not as a difference. The release battery waits for the model's size plus 2,048 MiB
free, above the reserve.

**GB10** (`repeats/gb10/`, main's sm_121a probe, the build the stored rows come from, judged on
CPU with this branch's probe): two processes and one later run per cell on the unit that
calibrated it, and Qwen3.8-27B on the other unit. They run after the GB10 Qwen3.8-27B floor rows,
under the units' queue rules. Before them, the records already cover both units: Ornith and
Qwen3.8-27B eager 26 and 14 records per unit, each batch arm 20 and 1.

**RTX PRO 6000 Max-Q**: no repeat with this branch's gate. The records cover three cards for all
four cells.

## Files

- `tools/seed_rows.sh <probe>`: writes the seeds from the floor rows' receipts.
- `tools/against_records.py`: each seed against the 12-row tables of the cell-evidence records of its device.
- `model-sha256.tsv`: the model digests per device.
- `seeds.txt`: the seed run's output; `against-records.md`: the comparison.
- `repeats/`: the GPU repeat cells, one directory per cell and run (`runs.txt` per cell).
