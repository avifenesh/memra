# The argmax margin gate judges a flip by its gap shift against a calibrated floor (2026-11-10)

**Status:** adopted (memra#1199). The gate's per-flip rule could not fail; it is replaced by
one that can, with a floor measured per cell. Receipts:
[`research/argmax-gate-rule-20261110/`](../../research/argmax-gate-rule-20261110/).

## Question

The calibrated argmax margin gate (`tools/argmax-margin-gate.sh`, `argmax-margin-probe`)
compares the prefill reference's logits with a decode program's at the last 12 positions of a
prompt. Its verdict had two parts: every flip had to be "margin-explained" (the spread at the
contending ids above the smaller of the two top-2 margins), and the flips had to fit a
per-model budget.

The first part holds for every flip by arithmetic. If prefill's top-1 is `i` at margin `m_p`
and decode's is `j` at margin `m_d`, then `p_i - p_j >= m_p` and `d_j - d_i >= m_d`, so
`(p_i - d_i) + (d_j - p_j) >= m_p + m_d` and the spread over the contending ids is at least
the mean of the two margins, which is at least the smaller one. All 1,309 distinct flips in
the tree's receipts are "explained" (smallest spread over smaller margin: 1.036), up to
margins of 15.8 logits. The only working check was the count. A defect that flips one row in
twelve at a wide margin passed.

What is a benign flip, measured, and what does the gate do with a row it cannot classify?

## Decision

**Gap shift.** The inequality above is the rule's base: a flip needs the two programs to
disagree about `i` against `j` by at least `m_p + m_d`. At a row where they agree, they
disagree about the top-1 against the runner-up by `|m_p - m_d|` (exact when both programs have
the same runner-up). That number is the row's gap shift. It is one statistic for every row, and
the probe table already holds it.

**Floor.** For one cell, a (model, prompt, decode program, device), the floor is the largest gap
shift at an agreeing row over every row of the prompt the probe can measure (T - 3 rows),
measured with the gate's own probe (`--calibrate`). Floors live in
`tools/argmax-margin-floors.tsv`, each row with its count of agreeing rows, the row that set it
and its receipt. The key includes the prompt's sha256 and the device name `nvidia-smi` reports.

**Verdict** (`memra_engine::argmax_margin::judge`, the one function the probe's `--judge` and
through it the gate script call):

| clause | row | result |
|---|---|---|
| near tie | a flip whose gap shift is inside the floor | counted against the flip budget |
| `WIDE-FLIP` | a flip whose gap shift is above the floor | fails at any count |
| `WIDE-ROW` | an agreeing row whose gap shift is above the floor | fails |
| `TOO-MANY-NEAR-TIES` | more near ties than the budget | fails |
| `UNCALIBRATED-FLIP` | a flip in a cell with no floor row | fails |

- **No floor row** means UNCALIBRATED: a flip fails closed, a table with no flip passes and
  says so, and its agreeing rows are not judged. The release battery runs the gate with
  `--require-floor` and reports such a cell as `REFUSED`.
- **Per device, no fallback.** A floor is used only on the device it was measured on. There is no
  default floor and no floor borrowed from another prompt, model or device.
- **Budgets keep their numbers** (1 per 12 rows by default; 3 for gemma-4-31B, gemma-4-26B and
  Qwen3.8-27B NVFP4 on `p4-16k`). What they count changed: near ties, flips inside the floor.
  The floor says what a near tie is, not how many a window holds, so a budget is still a
  measured count per (model, prompt).
- **Output.** For every flip and every wide row the gate prints the clause, both top-1 ids and
  margins, the spread, the gap shift and the floor, and the probe's `detail` line for the row:
  both programs' top-2 ids with their text and logits and each program's logit at the other's
  top-1.

**Procedures.**

- *Calibrating a cell:* `tools/argmax-margin-gate.sh <model> [--decode batch] --calibrate`,
  the log banked under `research/<lane>/`, the printed `FLOOR-ROW` added to the table.
- *A lane gating on its own prompt* calibrates the base binary (main's) on that prompt and
  passes that table with `--floors`; the gate's table holds no row measured on another prompt.
- *Recalibration:* a floor row changes only in a PR whose decision record accepts a change of
  numeric class for that cell. The PR reruns `--calibrate` on the changed binary and shows the
  old and the new row, here and in its body.

## What was rejected

Measured on the kv-quant lane's rows (`research/kv-quant-repetitive-20261110/`: Qwen3.8-27B on
GB10, 2,048 rows per document, prefill against the served batch decode and against the eager
decode). The controls are real prose and a CSV table at 32,768 and 65,535 tokens (468 flips;
none has a margin above 1.6 on either side). The JSONL log at the same depths has 47 flips: 22
with a margin sum of 0.41 or less and 25 with 1.97 or more (up to 16.6), which is the
population a rule has to split. Floors are the maximum over the agreeing rows of the control of
the same depth.

| rule | control flips failed | JSONL flips failed |
|---|---|---|
| the old rule | 0 / 468 | 0 / 47 |
| A: a margin above the largest agreeing-row **spread** | 0 / 468 | 21 / 47 |
| B: A, or the flip row's spread above that | 1 / 468 | 24 / 47 |
| **C: the margin sum above the largest agreeing-row gap shift (adopted)** | **0 / 468** | **25 / 47** |
| D: a margin above the largest agreeing-row gap shift | 0 / 468 | 23 / 47 |

- **The spread as the floor (A, B).** The spread at the contending ids includes moves that do
  not touch the decision. Its maximum is set by rare rows: one prose row in 1,966 has a spread
  of 14.68 where the 99.9th percentile is 4.82, and that floor lets a JSONL flip with a 5.8
  margin pass. On gemma-4-12B the agreeing rows of one table reach a spread of 3.8 and a gap
  shift of 0.77.
- **Each margin under the floor (D)** instead of their sum. A flip needs the sum; D passes two
  flips whose margins are each under the floor and add up to more.
- **A quantile floor.** The 99th percentile fails 20 of the 468 control flips, the 99.9th one of
  them and needs a thousand rows to exist. The maximum fails none. Its weakness, a floor set by
  one outlying row, is why each row of the table names the row that set it.
- **The exact shift between the two contenders** (from both programs' logits at `i` and `j`)
  in place of the margin sum. It is the truer number (at 27% of the kv-quant flips the
  contenders are not each other's runner-up and it is larger than the sum), it fails 2 of the
  468 control flips, and no receipt before this change holds the logits to re-judge. The probe
  now prints those logits per row, so the question can be reopened with data.
- **A floor from the gate's own 12 rows.** A floor taken from 12 rows is exceeded by an
  agreeing row in 54% of other 12-row windows of the same text, from 256 rows in 5.6%, from
  1,024 rows in 0.8% (order statistics give 12 / (N + 12): 50%, 4.5%, 1.2%). The calibration
  window is the whole prompt.
- **A fallback floor, or a proxy** (the largest device's floor on a device with none; a deep
  prose floor for a deep synthetic prompt). A cell without its own measurement is
  UNCALIBRATED.
- **Passing a clean uncalibrated cell silently, or refusing every uncalibrated cell.** A gate
  that passes by not having a row is not a gate, so the pass says UNCALIBRATED and the release
  battery refuses it; a lane probing a new model still gets the flip count.
- **A spread ceiling beside the gap shift.** memra#1175's second row (row 16,386: spread 4.79,
  margins 11.54 and 12.21) moves both programs' logits together and its gap shift is 0.67. The
  gap clause does not flag it. A second calibrated number per cell, with the spread's outliers,
  was not built.
- **Dropping the count.** A change that perturbs logits by less than the floor is, row by row,
  inside the calibrated difference; only the count of near ties in the fixed window shows it.

## The floors

`tools/argmax-margin-floors.tsv` is the table; the calibration receipts and their summary are in
`research/argmax-gate-rule-20261110/calibration/`. Each row is main 031f146690's unmodified
probe over every row of `board-2048` on one device. Main moved to 2628ece940 (the GB10 deep
split rung) while the cells ran: `board-2048` holds 2,048 keys and the rung starts past 16,384,
and the four roster 12-row tables in the cell-evidence records are byte-identical between
c7708aec49 and 12c3f554c4 on GB10 and on the RTX 5090 Laptop GPU.

No row is in the table at this commit: the cells are running. The change is not ready to land
until every roster cell (two models, two decode programs) has its row on GB10, the RTX 5090
Laptop GPU and the RTX PRO 6000 Blackwell, because the release battery refuses a cell without
one.

## History under the rule

`research/argmax-gate-rule-20261110/REJUDGE.md` re-judges every 12-row table in the tree once
the floor rows exist. Independent of any floor (`ANALYSIS.md` section 6):

- **The split-rung lane's synthetic cell fails, and so does main on the same prompt.** On the
  16,390-token synthetic record document (Qwen3.8-27B, GB10, eager) the candidate flips row
  16,385 with margins 14.386 and 2.481, a gap shift of 16.87. Main does not flip it and holds
  margins 14.386 and 7.156 there, a gap shift of 7.23 (memra#1175). No floor measured on text
  where the two programs agree covers either. The rule is not tuned around that cell: what
  passed it is the hand examination recorded in
  [FA-SPLIT-DEEP-RUNG.md](FA-SPLIT-DEEP-RUNG.md), candidate against main row by row, and the
  gate now prints what that examination needs.
- **Records that say "every flip margin-explained"** (GB10-F8F4-GEMM-V3 and the research
  records it cites, the gemma calibration records, the Ornith fork triage) carry a dated note:
  the phrase is true of any flip, and their evidence is the flip counts and, where they have
  them, byte-identical probe tables.
- **The gemma-4-31B budget of 3** was derived from "5 of 12 rows flippable", the same test.
  The banked gemma tables hold at most 2 flips. The number stays until a whole-prompt cell
  says otherwise.

## Limits

- A floor is one maximum over about two thousand rows of one prompt on one device. A legitimate
  change of numeric class can exceed it; the answer is the recalibration procedure, not a
  margin on the floor.
- An agreeing row's gap shift is exact only when both programs have the same runner-up;
  otherwise `|m_p - m_d|` is a lower bound, so a floor measured this way is, if anything, low.
- The gate still measures one prompt per cell. The kv-quant lane's rows show that on long
  machine-generated records main's own programs disagree by 5 to 20 logits at a few rows per
  thousand (memra#1175, memra#1177). A lane that gates there calibrates there, and a floor of
  several logits is a statement about those programs, not a pass.

## Receipts

`research/argmax-gate-rule-20261110/`: `ANALYSIS.md` (the candidate rules on the tree's
receipts, the 12-row window simulation, the calibration-size table), `calibration/` (the
whole-prompt probe logs behind every floor row), `REJUDGE.md` (every 12-row table in the tree
under the floors), `red-arm.txt` (the base gate and this gate on the same synthetic tables),
`tools/` (the scripts). CPU red arm: `cargo test -p memra-engine --lib argmax_margin`.
