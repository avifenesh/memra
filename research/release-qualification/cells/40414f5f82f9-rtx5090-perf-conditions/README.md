# Card conditions around the RTX 5090 Laptop perf cell of 40414f5f82f9

Receipts for the record next to this directory, `40414f5f82f9-rtx5090-perf` (the `release: v0.141.1` commit, PR memra#1218). They are not part of the record's payload manifest; the gate does not read them.

The 5090's battery and serving cells are in `40414f5f82f9-rtx5090-battery-serving`. The perf cell ran as its own record, after a cool-down, on a quiet host, with the host build lock held so that no build could start during it.

## The card ran under a 95 W limit

As for the landings before it, the laptop was not in its performance power mode for this cell. `nvidia-smi` reported `power.default_limit` 95 W (maximum 175 W), the platform profile read `balanced`, and the software power cap flag was active in every one of the 334 samples. The card drew at most 94.1 W.

So the absolute rates here are rates at 95 W, not this card's rates. The cell is an interleaved A B B A A B comparison of two binaries under the same limit, and that comparison is what the record carries.

## Summary

| | |
|---|---|
| window (2026-11-11, local) | 00:41:30 to 00:47:03 |
| start | 59 C, 1-minute load 2.5 |
| host during the cell | mean run queue 3.5, 7% user CPU |
| power | limit 95 W, peak draw 94.1 W, software power cap active throughout |
| Ornith-1.5-35B-A3B | +0.27% (A 121.07, 121.89, 121.43; B 120.78, 121.76, 122.69 tok/s) |
| Qwen3.8-27B | -0.26% (A 31.10, 31.17, 31.16; B 31.08, 31.07, 31.19 tok/s) |
| verdict | pass, no arm separation on either model |

The percentages are the record's median deltas, candidate (B, `40414f5f82`) against base (A, `3de59bbf80`). No thermal slowdown: the software and hardware thermal slowdown flags and the hardware slowdown flag were not active in any sample, in any run. The card peaked at 81 C.

## Files

- `throttle-before.csv`, `throttle-after.csv`: the card's throttle flags, temperature, power, utilization and clock right before and right after the cell.
- `throttle-1hz.csv`: the same once a second during the cell.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run (end time, samples, samples with a thermal or hardware slowdown flag active, temperature range, peak power).
- `arm_conditions.py`: the script that joins the samples with the perf logs' end times to make `arms.tsv`.
