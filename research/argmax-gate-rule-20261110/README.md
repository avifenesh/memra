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
- `judge(rows, floor, known, budget)`: near tie (flip, gap shift <= floor, counted), `WIDE-FLIP`,
  `WIDE-ROW`, `TOO-MANY-NEAR-TIES`, `UNCALIBRATED-FLIP`, and for a known row `KNOWN-ROW` (named,
  not counted) or `KNOWN-ROW-WIDER`.

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

`red-arm.txt` (`tools/red_arm.sh`): three synthetic tables through the gate script at
031f146690 and through this one, the measurement stubbed, each script's own verdict.

| table | base gate | this gate |
|---|---|---|
| wide flip: prefill holds its token by 14.4, decode picks another by 2.5 | exit 0, "explained" | exit 1, `WIDE-FLIP` |
| near tie: margins 0.02 and 0.03 | exit 0 | exit 0, `near-tie` |
| wide row: both pick one token, at margins 14.4 and 7.2 | exit 0 | exit 1, `WIDE-ROW` |

CPU tests (a CPU box, sm_120a build, no GPU): `cargo test -p memra-engine --lib argmax_margin`
(16: the verdict on synthetic logit rows, known rows in both directions, the table and floors
parsers, the old predicate true for every flip of 20,000 random row pairs), `cargo test -p
memra-engine --bin argmax-margin-probe` (13: `--judge` on a wide flip, a near tie, a budget
overrun, a wide row, a known row and a widened one, an uncalibrated cell, `--calibrate`),
`python3 tools/test_argmax_margin_gate.py` (30: the wrapper with the built probe as its judge).

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
  c7708aec49 and 12c3f554c4, on GB10 and on the RTX 5090 Laptop GPU (eight hashes equal). Main
  then moved to 3de59bbf80 (v0.141.0: a DSv4 server fix and a speculative-burst fix, nothing on
  the prefill or decode programs). Its probe is another binary (sm_120a sha256 `a98dfa943a6e...`
  against `723a3f7ff8dd...`, the version bump among the reasons) and its 12-row Ornith eager table
  on the RTX 5090 Laptop GPU equals the calibration's last 12 rows byte for byte
  (`main-v0.141.0/ornith-eager-12/`).
- **Cost.** RTX 5090 Laptop GPU at a 95 W limit, 27B eager: 128 rows in 342 s, 512 rows in
  1,051 s (the second from file times; `timing/`). The window's rows are the
  longest prefixes, so the whole prompt costs between 2,313 s (cost follows tokens) and 3,883 s
  (cost follows forwards) on that card.

**What the cells show** (`calibration/SUMMARY.md` has every column).

<!-- FLOORS-TABLE:START -->
| device | model | decode | flips / rows | most flips in 12 rows | p50 | p99 | p99.9 | floor (row) | known rows |
|---|---|---|---|---|---|---|---|---|---|
| GB10 | Ornith-1.5-35B-A3B | eager | 191 / 2045 | 6 | 0.219 | 1.616 | 3.013 | 3.581 (727) | none |
| GB10 | Ornith-1.5-35B-A3B | batch | 189 / 2045 | 6 | 0.236 | 1.707 | 4.013 | 4.450 (306) | none |
| RTX 5090 Laptop GPU | Ornith-1.5-35B-A3B | eager | 199 / 2045 | 5 | 0.197 | 1.522 | 2.265 | 3.558 (306) | 41, 1809 |
| RTX 5090 Laptop GPU | Ornith-1.5-35B-A3B | batch | 171 / 2045 | 5 | 0.199 | 1.582 | 3.339 | 3.956 (740) | none |
| RTX PRO 6000 Max-Q | Ornith-1.5-35B-A3B | eager | 173 / 2045 | 7 | 0.199 | 1.370 | 2.848 | 3.918 (306) | 41 |
| RTX PRO 6000 Max-Q | Ornith-1.5-35B-A3B | batch | 181 / 2045 | 5 | 0.195 | 1.504 | 2.763 | 3.695 (1169) | none |
| RTX 5090 Laptop GPU | Qwen3.8-27B | eager | 42 / 2045 | 4 | 0.047 | 0.340 | 0.960 | 3.730 (1198) | none |
| RTX 5090 Laptop GPU | Qwen3.8-27B | batch | 40 / 2045 | 3 | 0.046 | 0.364 | 0.663 | 3.088 (1199) | none |
| RTX PRO 6000 Max-Q | Qwen3.8-27B | eager | 44 / 2045 | 3 | 0.048 | 0.394 | 1.139 | 3.960 (1198) | none |
| RTX PRO 6000 Max-Q | Qwen3.8-27B | batch | 41 / 2045 | 4 | 0.049 | 0.371 | 1.222 | 4.242 (1198) | none |
| RTX 5090 Laptop GPU | Qwen3.5-9B NVFP4 | eager | 152 / 2045 | 6 | 0.147 | 1.425 | 2.810 | 4.010 (909) | none |
| RTX 5090 Laptop GPU | Qwen3.5-9B NVFP4 | batch | 156 / 2045 | 6 | 0.151 | 1.413 | 2.969 | 3.871 (909) | none |
| RTX 5090 Laptop GPU | gemma-4-31B q4_0 | eager | 114 / 2100 | 4 | 0.139 | 1.192 | 3.353 | 6.399 (594) | none |
| RTX 5090 Laptop GPU | gemma-4-26B q4_0 | eager | 250 / 2100 | 6 | 0.278 | 1.918 | 3.421 | 3.556 (1458) | 150, 943, 981, 1013, 1117, 1540 |
<!-- FLOORS-TABLE:END -->

- **Every floor is set by one row and is loose.** The roster floors sit at 3.1 to 4.5 logits
  where the 99.9th percentile is 0.7 to 4.0. On the dense 27B one row sets each floor, with the
  next agreeing row at 0.84 to 1.69: row 1,198 (prefill margin 4.92, decode 8.65 to 9.16) in
  three cells (3.73 eager on the RTX 5090 Laptop GPU, 3.96 and 4.24 on the RTX PRO 6000), and
  row 1,199 (prefill 9.49, decode 12.58) at 3.09 in the RTX 5090 Laptop GPU's batch cell, whose
  decode holds row 1,198 at 4.95, next to prefill. A flip whose two margins add up to 3 passes
  as a near tie on these cells. The rule catches the
  flip at margins 14.4 and 2.5 the old rule passed; it does not catch one at 2.0 and 1.5. The
  coordinator kept the maximum with these numbers on the table; the guard for that range is
  memra#1204 (the candidate's rows against main's stored rows at the same positions).
- **Main's own programs flip above their own floor.** Ornith eager at row 41 (margins 1.472
  and 2.899) on both discrete cards and at row 1,809 on the RTX 5090 Laptop GPU (prefill holds
  its token by 4.079, decode picks another by 0.070; on the PRO 6000 the same row agrees with a
  gap shift of 3.48). GB10 has none on Ornith. gemma-4-26B has six on the RTX 5090 Laptop GPU
  (rows 150, 943, 981, 1013, 1117, 1540; margin sums 3.7 to 5.3). gemma-4-31B has none: one
  agreeing row (594: prefill margin 0.43, decode 6.82) sets its floor at 6.40, the highest in
  the table, where the next agreeing row is at 3.37. These are the **known rows**
  of their floor rows: the gate names one when a judged window holds it, and a candidate may
  not widen it. Which program is right there is memra#1203. The gate's window, the last 12
  rows, holds none of them.
- **Flip rates differ by model, not by device.** Over the whole prompt: Ornith 8.4 to 9.7% of
  rows (up to 7 inside one 12-row window), Qwen3.5-9B 7.4 to 7.6% (up to 6), gemma-4-26B 11.9%
  (up to 6), gemma-4-31B 5.4% (up to 4), the 27B 2.0 to 2.2% (up to 4). The flip budgets hold at
  the gate's window (0 or 1 flip there on the roster models, 2 on gemma-4-26B and 1 on
  gemma-4-31B against their budget of 3), not across the prompt. No budget is changed.

**Reproduction.** Tables that were measured twice:

| pair | same? |
|---|---|
| RTX 5090, 27B eager: the 128-row timing table against the last 128 rows of the 512-row one (one binary) | byte-identical (`0dcbb5bbd14536ba`) |
| RTX 5090, Ornith eager and batch: the last 12 rows of the whole-prompt table against the 12-row table of that card's cell-evidence record at 12c3f554c4 (another build) | byte-identical (`706af873d259891a`, `e356081d72028d74`) |
| RTX PRO 6000 Max-Q, all four cells: the last 12 rows against the card's cell-evidence record at e6613b4eee22 (another build) | byte-identical (four hashes) |
| GB10, 27B eager: the last 12 rows of the 128-row timing table against the unit's cell-evidence record | byte-identical (`f5ad5475e41d3eec`) |
| RTX 5090, 27B eager and batch: the last 12 rows of the whole-prompt table against the card's cell-evidence record at 12c3f554c4 (another build) | byte-identical (`b25c41e5e8e34315`, `4083a49bdc9adb40`) |
| GB10, Ornith eager and batch: the last 12 rows of the whole-prompt table against the unit's cell-evidence record at 12c3f554c4 | byte-identical (`0d9b64dd0373157a`, `3484571a10e0190c`) |
| RTX 5090, gemma-4-26B eager: the last 12 rows against the 12-row table banked in `research/gemma-one-program-20261104/` (another build, six days earlier) | byte-identical (`3768249ed8dbaaf2`) |
| RTX 5090, Ornith eager: v0.141.0's probe (3de59bbf80), 12 rows, against the last 12 rows of the whole-prompt calibration table | byte-identical (`706af873d259891a`); PRIME-GATE MATCH |
| RTX 5090, gemma-4-31B eager: the last 12 rows against both 12-row tables banked in `research/gemma-one-program-20261104/` (`argmax-g31-off`, `argmax-g31-on`; another build, six days earlier) | byte-identical (`1ad3ce860bb2bcdb`) |
| RTX 5090, Ornith eager, the whole prompt twice: main's probe (031f146690) and this branch's probe (`--calibrate`, on 2628ece940's engine), 80 minutes apart with other cells on the card between | all 2,045 rows byte-identical; the two floor rows equal in every data field |

Two RTX 5090 cells were stopped mid-run for a release ticket and rerun from scratch; a stopped
cell wrote no table (the probe prints at the end), so there is no partial table to compare.

**Cells still to run at this commit:** Qwen3.8-27B eager and batch on GB10 (the two roster
rows the census still names); Qwen3.5-9B NVFP4 eager and batch on GB10 (the model the gate
script resolves to there when given none). Until a cell's row exists it is UNCALIBRATED and the
release battery refuses it.

Not calibrated here: `ornith-1.0-9b` Q8_0 and `Qwen3.6-27B-Q8_0`, the first two models the gate
script tries when it is given none (neither file is on a box the gate runs on; the script skips
a candidate whose file is absent and takes the next); Qwen3.5-9B on the RTX PRO 6000 box
(nothing is built there, so only the battery with the roster's two models calls the gate
there); and Qwen3.8-27B on `p4-16k` on GB10 (its window sits past the GB10 split rung, so its
floor has to come from a probe built at 2628ece940 or later).

## Green arm on a GPU

This branch's gate script with this branch's probe (built at 83052e765d on a CPU box, sha256
`7b26703b2ca2...`), RTX 5090 Laptop GPU, Ornith-1.5-35B-A3B, one lease per cell (`green-arm/`).

| cell | result |
|---|---|
| the gate, eager, 12 rows | PASS, 0 flips, floor 3.5582 printed with its percentiles and known rows, prime MATCH; 42 s |
| the gate, `--decode batch`, 12 rows | PASS, 0 flips, served MATCH; 40 s |
| `--canary`, eager | CANARY PASS: `WIDE-FLIP pos=9999` at the injected row, the injected prime line read as MISMATCH-STRUCTURED |
| `--canary`, batch | CANARY PASS: `WIDE-FLIP pos=9999`, the injected served line read as MISMATCH |
| `--calibrate`, eager (window `all`, 2,045 rows, 731 s) | the `FLOOR-ROW` it prints equals the table's row from main's probe in all 12 data fields (floor 3.558178902e0, 1,846 rows, row 306, the percentiles, known rows 41 and 1809 with the same ids and margins) |

The 12-row tables of the two gate cells are byte-identical to the card's cell-evidence record.
`green-arm/ornith-eager-calibrate/known-rows-judged.txt` is the whole-prompt table judged
against the floors table: the two known rows named with their `detail` lines (each program's
top-2 ids, text and logits), 197 near ties, nothing failing.

## History under the floors

`REJUDGE.md` (`tools/rejudge.py`): every distinct 12-row table in the tree under the floors
table as it stands at this commit. A table whose cell has no floor row yet reads
`UNCALIBRATED`; a flip there fails closed.

| what | old verdict | under this rule |
|---|---|---|
| roster cells in the cell-evidence records, RTX PRO 6000 Max-Q (4 cells) | PASS | PASS. Ornith's flip at row 2,041 (0.240 + 0.038 eager, 0.240 + 0.048 batch) is a near tie under floors of 3.9 and 3.7 |
| roster cells, RTX 5090 Laptop GPU, Ornith eager and batch | PASS | PASS, no flip; largest agreeing gap shift 0.98 and 0.59 |
| roster cells, RTX 5090 Laptop GPU 27B eager and batch; GB10 Ornith eager and batch | PASS | PASS, no flip |
| roster cells still without a row (GB10 27B eager and batch) | PASS | no floor row yet: PASS, UNCALIBRATED, no flip |
| the split-rung lane's synthetic 16,390-token cell, 27B eager, GB10: candidate | PASS | FAIL under any floor measured where the programs agree: a flip at row 16,385 with margins 14.386 and 2.481 (gap shift 16.87) |
| the same prompt on main | PASS | FAIL the same way: an agreeing row at 16,385 with margins 14.386 and 7.156 (gap shift 7.23), memra#1175 |
| the split-rung landing's real-text served-batch cells (`landing-cells/`), main and the landed head | PASS | no flip in either arm; no floor for that prompt, so UNCALIBRATED. Largest agreeing gap shift 1.655 at row 16,378 in both arms (below the rung, identical rows) |
| GB10-F8F4-GEMM-V3's `p4-16k` cells (3 flips, budget 3) | PASS | no floor for that prompt: its three flips (margin sums 0.044, 0.098, 0.233) are unclassified. Its evidence is the byte-identical tables |
| gemma-4-26B, RTX 5090 Laptop GPU (budget 3) | PASS | PASS: its two flips are near ties under the floor of 3.556 (row 2,091: 0.097 + 0.048; row 2,096: 0.014 + 1.300, a flip where decode holds its token by 1.3, which this floor does not separate from a tie) |
| the other gemma tables (31B on every card, 26B and 12B on the 170-SM card, the Q6K ship artifact) | PASS | no floor on those cells: flips unclassified |

The rule is not tuned around the split-rung cell. It fails, main fails on the same prompt, and
what passed it is the hand examination in `docs/decisions/FA-SPLIT-DEEP-RUNG.md`, candidate
against main row by row. The gate now prints what that examination reads: for every flip and
wide row, each program's token and margin, the spread, the gap shift and the floor.

## Files

- `ANALYSIS.md`: `tools/analyze.py` over `tools/collect.py`'s rows.
- `REJUDGE.md`: `tools/rejudge.py`, every 12-row table in the tree under the floors table.
- `red-arm.txt`: `tools/red_arm.sh`.
- `calibration/<device>/<cell>/`: `probe.log` (the probe's raw output over the whole prompt:
  the cell's full per-row table, kept for memra#1204), `result.txt` (model, decode program,
  prompt and probe digests, device name, exit code, wall time). `calibration/SUMMARY.md`:
  `tools/floor_rows.sh`. The RTX PRO 6000 cells ran on another box through
  `calibration/rtxpro6000-maxq/pro6000_cal.sh`; their `result.txt` restates `meta.txt` and
  `runs.log`, and box paths, card ids and the user name are replaced in the logs.
- `green-arm/`: this branch's gate, canary and `--calibrate` on a GPU.
- `landing-cells/`: the split-rung landing's real-text served-batch cells, both arms.
- `timing/`: the cells that sized the calibration.
- `tools/cell.sh`: the cell runner.
