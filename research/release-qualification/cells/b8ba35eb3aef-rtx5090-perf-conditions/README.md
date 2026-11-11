# Card conditions around the RTX 5090 Laptop perf cell of b8ba35eb3aef

Receipts for the record next to this directory, `b8ba35eb3aef-rtx5090-perf` (the landing of memra#1229, memra#1228, memra#1230 and memra#1224, PR memra#1231, after the SLRU trace fixture re-pin). They are not part of the record's payload manifest; the gate does not read them.

The 5090's battery and serving cells are in `b8ba35eb3aef-rtx5090-battery-serving`. The perf cell ran as its own record, after a cool-down, on a quiet host, with the host build lock held so that no build could start during it.

## The card ran under its 95 W default limit

As for the landings before it, the laptop was not in its performance power mode for this cell. `nvidia-smi` reported `power.default_limit` 95 W (maximum 175 W) and the platform profile read `balanced`. The software power cap flag was active in every one of the 378 samples, and the card drew at most 93.9 W. (Read after the cell, `enforced.power.limit` showed 150 W; the samples carry no limit column, and the draw stayed under 95 W throughout.)

So the absolute rates here are rates at that limit, not this card's best rates. The cell is an interleaved A B B A A B comparison of two binaries under the same limit, and that comparison is what the record carries.

## Summary

| | |
|---|---|
| window (2026-11-11, local) | 18:26:37 to 18:32:55 |
| start | 59 C, 1-minute load 3.94 |
| host during the cell | mean run queue 5.9, 14% user CPU (other sessions' CPU work; the build lock was held, so no build ran) |
| power | default limit 95 W, peak draw 93.9 W, software power cap active throughout |
| Ornith-1.5-35B-A3B | -0.36% (A 121.70, 120.62, 121.30; B 121.01, 120.86, 119.51 tok/s) |
| Qwen3.8-27B | -0.22% (A 31.01, 31.17, 31.15; B 31.05, 31.08, 31.15 tok/s) |
| verdict | pass, no arm separation on either model |

The percentages are the record's median deltas, candidate (B, `b8ba35eb3a`) against base (A, `ede1b8119f`). The arms overlap on both models. No thermal slowdown: the software and hardware thermal slowdown flags and the hardware slowdown flag were not active in any sample, in any run. The card peaked at 81 C. The same code (without the fixture re-pin, which changes no crate source) measured Ornith -1.34% and Qwen3.8-27B +0.16% at 8c4d70b927 earlier the same day, also with overlapping arms (`8c4d70b92732-rtx5090-perf-conditions/`).

## Files

- `throttle-before.csv`, `throttle-after.csv`: the card's throttle flags, temperature, power, utilization and clock right before and right after the cell.
- `throttle-1hz.csv`: the same once a second during the cell.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run (end time, samples, samples with a thermal or hardware slowdown flag active, temperature range, peak power).
- `arm_conditions.py`: the script that joins the samples with the perf logs' end times to make `arms.tsv`.
