# The split-rung landing's real-text served-batch cells (2026-11-10)

The cell the split-rung lane (PR #1188) had not run: the calibrated argmax margin gate on
Qwen3.8-27B NVFP4, real text (the first 16,390 tokens of `research/e2e/prompts/p4-16k.txt`,
window rows 16,378 to 16,389, straddling GB10's 16,384-key split floor), with the served batch
decode program. GB10, one unit per arm, the gate script of 1a4f769861 (the old rule).

| arm | probe build | flips | result under the old rule |
|---|---|---|---|
| main 031f146690 | recorded build, sha256 `6e6650071f22...` | 0 | PASS, served MATCH |
| the landed head 12c3f554c4 | recorded build, sha256 `22293a1dac13...` | 0 | PASS, served MATCH |

Rows 16,378 to 16,383 (below the floor) are identical in both arms. `rt_cell.sh` ran on the
unit, `rt_driver.sh` on the driving machine. The home directory in the logs is written `~`.
`../REJUDGE.md` re-judges both tables under the rule that replaced the old one.
