# Card conditions around the RTX 5090 Laptop perf cell of 12c3f554c4fd

Receipts for the record next to this directory, `12c3f554c4fd-rtx5090-perf` (the landing of memra#1180, #1194, #1187, #1190 and #1189, PR memra#1188). They are not part of the record's payload manifest; the gate does not read them.

The 5090's battery and serving cells are in `12c3f554c4fd-rtx5090-battery-serving`. The perf cell ran as its own record, after a cool-down, on a quiet host, with the host build lock held so that no build could start during it.

## The card ran under a 95 W limit

The laptop was not in its performance power mode for this cell. `nvidia-smi` reported `power.default_limit` 95 W (maximum 175 W), the platform profile read `balanced`, and the software power cap flag was active in every one of the 327 samples. The card drew at most 93.4 W. In the earlier cells of this card (for example `../e6613b4eee22-rtx5090-perf-conditions/`) it drew 150 to 160 W.

So the absolute rates here are rates at 95 W, not this card's rates: Ornith-1.5-35B-A3B about 121 tok/s against about 215 before, Qwen3.8-27B about 31 against about 45. The cell is an interleaved A B B A A B comparison of two binaries under the same limit, and that comparison is what the record carries.

## Summary

| | |
|---|---|
| window (2026-11-10, local) | 15:22:14 to 15:27:41 |
| start | 59 C, 1-minute load 1.8 |
| host during the cell | mean run queue 3.0, 7% user CPU |
| power | limit 95 W, peak draw 93.4 W, software power cap active throughout |
| Ornith-1.5-35B-A3B | +0.50% (A 121.04, 121.28, 120.30; B 120.97, 121.95, 121.65 tok/s), spread 1.4% |
| Qwen3.8-27B | -0.03% (A 31.07, 31.19, 31.19; B 31.10, 31.18, 31.18 tok/s), spread 0.4% |
| verdict | pass, no arm separation on either model |

No thermal slowdown: the software and hardware thermal slowdown flags were not active in any sample, in any run. The card peaked at 79 C. At this limit it does not reach the 87 C at which the flag tripped in the earlier cells, and the run-to-run spread is far smaller than there (5 to 8% on Qwen3.8-27B at 150 to 160 W).

## Files

- `throttle-before.csv`, `throttle-after.csv`: the card's throttle flags, temperature, power, utilization and clock right before and right after the cell.
- `throttle-1hz.csv`: the same once a second during the cell.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run (end time, samples, samples with a thermal or hardware slowdown flag active, temperature range, peak power).
- `arm_conditions.py`: the script that joins the samples with the perf logs' end times to make `arms.tsv`.
