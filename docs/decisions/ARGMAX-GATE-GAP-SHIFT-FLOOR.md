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
| `KNOWN-ROW` | a row the floor row lists as known, no wider than recorded | printed and named |
| `KNOWN-ROW-WIDER` | a known row whose gap shift exceeds the recorded one | fails |

- **Known rows.** The whole-prompt calibration of main holds flips above its own floor: rows
  where main's two programs already disagree by more than at any agreeing row of the prompt.
  The floor row lists each with its two top-1 ids and margins. A known row inside a judged
  window is printed and named, and does not fail the cell by itself. A candidate whose gap
  shift at a known row exceeds main's recorded shift at that row fails. A known row covers its
  own position only and is not counted against the budget. Which program is right at these
  rows is memra#1203.
- **The floor row states how loose it is.** Besides the count of agreeing rows and the row
  that set the maximum, it carries the p50, p99 and p99.9 of the agreeing rows' gap shift.
- **No floor row** means UNCALIBRATED: a flip fails closed, a table with no flip passes and
  says so, and its agreeing rows are not judged. The release battery runs the gate with
  `--require-floor` and reports such a cell as `REFUSED`.
- **Per device, no fallback.** A floor is used only on the device it was measured on. There is no
  default floor and no floor borrowed from another prompt, model or device. The device is its
  exact `nvidia-smi` name: a laptop and a desktop RTX 5090, or a Max-Q and a non-Max-Q RTX PRO
  6000, are different devices until measured equal.
- **Maintained classes.** The classes of push-gated build targets in
  `tools/change-impact.toml` (`rtx5090`, `rtxpro6000`, `gb10`) are maintained. With
  `--require-floor` and no floor row, a device of a maintained class is refused and so is a
  device name no class matches. A device of a class that is not maintained (H100, B200) prints
  `not judged: device class not maintained, no floor row` and neither passes nor refuses the
  battery. The census (`tools/check-argmax-floors.py`) demands, for every maintained class, a
  floor row for every argmax cell the battery lists.
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
  one outlying row, showed on the first whole-prompt cells: every roster floor on `board-2048`
  is set by one row and sits at several logits (the table below), where the p99.9 is under 3.
  The question went back to the coordinator with those numbers. The floor stays the maximum;
  each row states its percentiles and its known rows, the wide rows on main are memra#1203, and
  the tight guard that does not depend on the floor's looseness, the candidate's rows against
  main's stored rows at the same positions, is memra#1204.
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

`tools/argmax-margin-floors.tsv` is the table; the receipts are the whole-prompt probe logs in
`research/argmax-gate-rule-20261110/calibration/`. Each row is main 031f146690's unmodified
probe over every row of `board-2048` (2,045 of T = 2,048) on one device. Main moved to
2628ece940 (the GB10 deep split rung) while the cells ran: `board-2048` holds 2,048 keys and
the rung starts past 16,384, and the roster's 12-row tables in the cell-evidence records are
byte-identical between c7708aec49 and 12c3f554c4 on GB10 and on the RTX 5090 Laptop GPU. On
3de59bbf80 (v0.141.0) the probe's Ornith eager window on the RTX 5090 Laptop GPU is unchanged,
byte for byte. The last 12 rows of each whole-prompt table equal the 12-row table of its card's
record.

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

Rows still to come at this commit: Qwen3.8-27B eager and batch on GB10. The change does not
land before them: the release battery refuses a roster cell without its row, and the census
test is red until they exist. Outside the battery: Qwen3.5-9B on GB10.

What the cells showed, and what was decided with it:

- Every floor is set by one row. On the dense 27B the next agreeing row is at 0.84 to 1.69 and
  the p99.9 at 0.66 to 1.22: row 1,198 (prefill margin 4.92, decode 8.65 to 9.16) sets the
  floor at 3.7 to 4.2 in three cells, and row 1,199 (prefill 9.49, decode 12.58) sets it at
  3.09 in the RTX 5090 Laptop GPU's batch cell, whose decode holds row 1,198 at 4.95, next to
  prefill. On the MoE the top rows are closer together (3.6, 2.9,
  2.3, 2.2 on one cell).
- Main's own programs flip above their own floor: Ornith eager at row 41 (margins 1.472 and
  2.899) on both discrete cards and at row 1,809 on the RTX 5090 Laptop GPU (4.079 and 0.070),
  none on GB10; gemma-4-26B at six rows on the RTX 5090 Laptop GPU. The gate's window, the
  last 12 rows, holds none of them.
- The coordinator's ruling on those numbers: the floor stays the maximum; each floor row states
  its percentiles; a calibration flip above its own floor is a known row, printed and named
  when a window holds it, failing only when a candidate widens it. Issues: memra#1203 (the
  wide rows on main: which program is right, the router's top-k per layer in both programs)
  and memra#1204 (the tight guard: the candidate's rows against main's stored rows at the same
  positions, for which every calibration cell's full per-row table is kept).

## History under the rule

`research/argmax-gate-rule-20261110/REJUDGE.md` re-judges every distinct 12-row table in the
tree under the floors table as it stands.

- **The roster's cells pass.** On the RTX PRO 6000 Max-Q, Ornith's one flip (row 2,041, margins
  0.240 and 0.038 eager, 0.240 and 0.048 batch) is a near tie. On the RTX 5090 Laptop GPU and
  on GB10 the calibrated roster cells hold no flip. The cells whose rows are still to come
  read UNCALIBRATED and hold no flip.
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
  them, byte-identical probe tables. Their cells have no floor, so their flips read as
  unclassified.
- **The gemma budgets of 3** were derived from "5 of 12 rows flippable", the same test. The
  banked gemma tables hold at most 2 flips in the gate's window. The whole-prompt gemma-4-26B
  cell on the RTX 5090 Laptop GPU has 250 flips in 2,100 rows and up to 6 in one 12-row
  window; the gate's window holds 2, both near ties under its floor of 3.556 (one of them a
  flip where decode holds its token by 1.3). The budget is not contradicted at the gate's
  window and is not changed. The whole-prompt gemma-4-31B cell (the model `tools/local-ci.sh`
  gates on the RTX 5090 Laptop GPU) has 114 flips in 2,100 rows and up to 4 in one 12-row
  window; the gate's window holds 1, a near tie (margins 0.223 and 0.377) under its floor of
  6.399, which one agreeing row sets (row 594: prefill 0.426, decode 6.825; the next is at
  3.37). Its last 12 rows equal both 12-row tables banked in `research/gemma-one-program-20261104/`.

## Limits

- A floor is one maximum over about two thousand rows of one prompt on one device. A legitimate
  change of numeric class can exceed it; the answer is the recalibration procedure, not a
  margin on the floor.
- The maximum is loose where one row sets it. A flip whose two margins add up to less than a
  floor of 3.1 to 4.5 logits passes as a near tie on the roster cells. The gate catches the
  flip at margins 14.4 and 2.5 that the old rule passed; it does not catch one at 2.0 and 1.5.
  memra#1204 is the guard for that range.
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
