# A per-flip rule for the argmax margin gate that can fail (memra#1199)

2026-11-10. CPU analysis of the receipts the tree already held, then one whole-prompt
calibration cell per gate cell. Decision record:
[`docs/decisions/ARGMAX-GATE-GAP-SHIFT-FLOOR.md`](../../docs/decisions/ARGMAX-GATE-GAP-SHIFT-FLOOR.md).

## Outcome

1. The gate's old per-flip rule ("margin-explained": the spread at the contending ids exceeds
   the smaller top-2 margin) is true of every flip. All 1,309 distinct flips in the tree's
   receipts pass it, at margins up to 15.8 logits; the smallest spread over smaller margin is
   1.036 (`ANALYSIS.md` section 1). The gate's verdicts rested on the flip count alone.
2. The rule that replaces it judges each row by its **gap shift**: the sum of the two programs'
   top-2 margins at a flip, their difference at an agreeing row. A cell's **floor** is the
   largest gap shift at an agreeing row over the whole prompt. A flip inside the floor is a near
   tie; a flip or an agreeing row above it fails; a cell with no floor is UNCALIBRATED.
3. On the kv-quant lane's rows (Qwen3.8-27B, GB10, 2,048 rows per document) that rule fails 0
   of 468 flips on real prose and a CSV table, and 25 of the 47 flips on a JSONL log: exactly
   the 25 whose margin sum is 1.97 or more. The 22 it passes have a margin sum of 0.41 or
   less. The old rule fails none of the 47. A floor built on the spread instead misses 4 of
   the 25 (`ANALYSIS.md` section 3).
4. A floor cannot come from the gate's 12 rows: a floor taken from 12 rows is exceeded by an
   agreeing row in 54% of other 12-row windows of the same text (`ANALYSIS.md` section 5). Each
   cell is calibrated over every row of its prompt.

## The rule

`memra_engine::argmax_margin` (CPU only). Per row: prefill's top-1 `i` with margin `m_p`,
decode's top-1 `j` with margin `m_d`.

- A flip needs `p_i - p_j >= m_p` and `d_j - d_i >= m_d`, so the programs disagree about `i`
  against `j` by at least `m_p + m_d`. Gap shift of a flip: `m_p + m_d`.
- Gap shift of an agreeing row: `|m_p - m_d|`.
- Floor: the largest gap shift at an agreeing row of the calibration table.
- `judge(rows, floor, budget)`: near tie (flip, gap shift <= floor, counted), `WIDE-FLIP`,
  `WIDE-ROW`, `TOO-MANY-NEAR-TIES`, `UNCALIBRATED-FLIP`.

The same inequality is why the old rule could not fail: the spread over the contending ids is
at least `(m_p + m_d) / 2`, which is at least the smaller margin.

## Receipts read (CPU, no new measurement)

`tools/collect.py` gathers every probe row in the tree: the probe tables in research logs and
cell-evidence records, and the kv-quant lane's per-row dumps (top-5 logits per program per row,
`research/kv-quant-repetitive-20261110/`). `tools/analyze.py` writes `ANALYSIS.md` from them:

| section | what it shows |
|---|---|
| 1 | the old predicate on every flip: 1,309 of 1,309 "explained" |
| 2 | per (model, prompt, pair, device): the largest agreeing-row gap shift and spread, the flips |
| 3 | candidate rules and floor statistics on the kv-quant gate pairs: false fails on the controls, wide flips caught on the JSONL log |
| 4 | the gate's 12-row window cut from those rows: the old gate fails 3 of 680 JSONL windows (by count) while 24 hold a wide flip; near ties per window on prose at 32,768 tokens (0 / 1 / 2 / 3 in 83 / 59 / 23 / 5 of 170 windows) |
| 5 | calibration size: share of 12-row windows with a row above a floor taken from N rows |
| 6 | every distinct 12-row table with the floor it would need |

Sources at main 2628ece940. The kv-quant rows are analysis only: their programs are not the
gate's (prefill rows from one forward, decode starting from a prime), so none of them is a
floor in the gate's table.

## Red arm

`red-arm.txt` (`tools/red_arm.sh`): three synthetic tables through the gate script at the base
commit and through this one, the measurement stubbed, each script's own verdict.

| table | base gate | this gate |
|---|---|---|
| wide flip: prefill holds its token by 14.4, decode picks another by 2.5 | exit 0, "explained" | exit 1, `WIDE-FLIP` |
| near tie: margins 0.02 and 0.03 | exit 0 | exit 0, `near-tie` |
| wide row: both pick one token, at margins 14.4 and 7.2 | exit 0 | exit 1, `WIDE-ROW` |

CPU tests (dev desk, sm_120a build, no GPU): `cargo test -p memra-engine --lib argmax_margin`
(13: the verdict on synthetic logit rows, the table and floors parsers, the old predicate true
for every flip of 20,000 random row pairs), `cargo test -p memra-engine --bin
argmax-margin-probe` (11: `--judge` on a wide flip, a near tie, a budget overrun, a wide row,
an uncalibrated cell, `--calibrate`), `python3 tools/test_argmax_margin_gate.py` (29: the
wrapper with the built probe as its judge).

## Calibration

One cell per (model, decode program, device): main's unmodified `argmax-margin-probe` over
every row of `research/e2e/prompts/board-2048.txt` the probe can give (a window of 2,045 on
T = 2,048; 2,100 on T = 2,103 for the gemma models), run by `tools/cell.sh`. The cell checks the
probe's sha256 before the run, records it again after, and writes the model's byte size, the
prompt's sha256 and the device name `nvidia-smi` reports. `tools/floor_rows.sh` turns each
banked log into its floors-table row through the gate's own `--judge --calibrate`.

- **Probe builds.** Main 031f146690, unmodified. sm_120a: built on a CPU box with the record
  recipe (`--locked`, fat LTO, one codegen unit), sha256 `723a3f7ff8dd...6035`. sm_121a: the
  cell-evidence build of the same commit, sha256 `6e6650071f22...2b5f`.
- **The build reproduces the records.** The last 12 rows of the 27B eager timing tables on the
  RTX 5090 Laptop GPU hash to `b25c41e5e8e34315`, the hash of the 12-row table in that card's
  cell-evidence records.
- **Main moved while the cells ran** (to 2628ece940, the GB10 deep split rung). The floors are
  measured on 031f146690's engine. `board-2048` holds 2,048 keys and the rung starts past
  16,384. The four roster 12-row tables in the cell-evidence records are byte-identical between
  c7708aec49 and 12c3f554c4, on GB10 and on the RTX 5090 Laptop GPU (eight hashes equal).
- **Cost.** RTX 5090 Laptop GPU at a 95 W limit, 27B eager: 128 rows in 342 s, 512 rows in
  1,051 s (the second from file times; `timing/`). The window's rows are the
  longest prefixes, so the whole prompt costs between 2,313 s (cost follows tokens) and 3,883 s
  (cost follows forwards) on that card.

**Cells and their state at this commit.** No whole-prompt cell has finished yet, so
`tools/argmax-margin-floors.tsv` holds no row and every gate cell is UNCALIBRATED: the release
battery would refuse each one. Rows are added, with their banked logs, as cells finish.

| device | cells | state |
|---|---|---|
| RTX 5090 Laptop GPU | Ornith-1.5-35B-A3B and Qwen3.8-27B, eager and batch | queued, one lease per cell |
| RTX 5090 Laptop GPU | Qwen3.5-9B NVFP4 (the fast-gate default that card resolves to), eager and batch | queued |
| RTX 5090 Laptop GPU | gemma-4-31B q4_0 and gemma-4-26B q4_0, eager | queued |
| GB10 | Ornith-1.5-35B-A3B and Qwen3.8-27B, eager and batch | queued |
| RTX PRO 6000 Blackwell | Ornith-1.5-35B-A3B and Qwen3.8-27B, eager and batch | running |

Not calibrated here: `ornith-1.0-9b` Q8_0 and `Qwen3.6-27B-Q8_0`, the first two models the gate
script tries when it is given none (neither file is on the calibration card; the script skips a
candidate whose file is absent and takes the next), and Qwen3.8-27B on `p4-16k` on GB10 (its
window sits past the GB10 split rung, so its floor has to come from a probe built at
2628ece940 or later).

## History under the floors

`REJUDGE.md` is written once floor rows exist. With no row, the rule reads every table that
holds a flip as unclassified (`UNCALIBRATED-FLIP`) and every table without one as
`PASS: UNCALIBRATED`. Floor-independent facts are in `ANALYSIS.md` section 6, which gives for
each distinct 12-row table the smallest floor it would pass under:

- the split-rung lane's synthetic 16,390-token cell (Qwen3.8-27B, GB10, eager) has a flip at
  row 16,385 with margins 14.386 and 2.481: it needs a floor of 16.87. Main on the same prompt
  has no flip and an agreeing row with margins 14.386 and 7.156: it needs 7.23. Both fail under
  any floor measured on text where the two programs agree. What passed that cell is the hand
  examination in `docs/decisions/FA-SPLIT-DEEP-RUNG.md`, candidate against main row by row;
- the roster's 12-row tables on GB10 and the RTX 5090 Laptop GPU hold no flip.

## Files

- `ANALYSIS.md`: `tools/analyze.py` over `tools/collect.py`'s rows.
- `REJUDGE.md`: `tools/rejudge.py`, every 12-row table in the tree under the floors table.
- `red-arm.txt`: `tools/red_arm.sh`.
- `calibration/<device>/<cell>/`: `probe.log` (the probe's raw output over the whole prompt),
  `result.txt` (model, decode program, prompt and probe digests, device name, exit code, wall
  time). `calibration/SUMMARY.md`: `tools/floor_rows.sh`.
- `tools/cell.sh`: the cell runner.
