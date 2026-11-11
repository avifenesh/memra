# Card conditions around the RTX 5090 Laptop perf cell of 83d37f5179ad

Receipts for the record next to this directory, `83d37f5179ad-rtx5090-perf` (the landing of memra#1216, #1202, #1215 and #1219, PR memra#1220). They are not part of the record's payload manifest; the gate does not read them.

The 5090's battery and serving cells are in `83d37f5179ad-rtx5090-battery-serving`. The perf cell ran as its own record, after a cool-down, on a quiet host, with the host build lock held so that no build could start during it.

## The card ran under a 95 W limit

As for the landings before it, the laptop was not in its performance power mode for this cell. `nvidia-smi` reported `power.default_limit` 95 W (maximum 175 W), the platform profile read `balanced`, and the software power cap flag was active in every one of the 330 samples. The card drew at most 91.8 W.

So the absolute rates here are rates at 95 W, not this card's rates. The cell is an interleaved A B B A A B comparison of two binaries under the same limit, and that comparison is what the record carries.

## Summary

| | |
|---|---|
| window (2026-11-11, local) | 02:35:49 to 02:41:19 |
| start | 59 C, 1-minute load 2.9 |
| host during the cell | mean run queue 2.3, 6% user CPU |
| power | limit 95 W, peak draw 91.8 W, software power cap active throughout |
| Ornith-1.5-35B-A3B | +0.56% (A 121.47, 121.95, 120.77; B 120.91, 122.15, 122.27 tok/s) |
| Qwen3.8-27B | +0.23% (A 31.07, 31.09, 31.22; B 31.06, 31.19, 31.16 tok/s) |
| verdict | pass, no arm separation on either model |

The percentages are the record's median deltas, candidate (B, `83d37f5179`) against base (A, `11f3c0920b`). No thermal slowdown: the software and hardware thermal slowdown flags and the hardware slowdown flag were not active in any sample, in any run. The card peaked at 76 C.

## Files

- `throttle-before.csv`, `throttle-after.csv`: the card's throttle flags, temperature, power, utilization and clock right before and right after the cell.
- `throttle-1hz.csv`: the same once a second during the cell.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run (end time, samples, samples with a thermal or hardware slowdown flag active, temperature range, peak power).
- `arm_conditions.py`: the script that joins the samples with the perf logs' end times to make `arms.tsv`.
