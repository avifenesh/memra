# Card conditions around the RTX 5090 Laptop perf cell of 29e3632d87d8

Receipts for the record next to this directory, `29e3632d87d8-rtx5090-perf` (the landing of memra#1208, PR memra#1222). They are not part of the record's payload manifest; the gate does not read them.

The 5090's battery and serving cells are in `29e3632d87d8-rtx5090-battery-serving`. The perf cell ran as its own record, after a cool-down, on a quiet host, with the host build lock held so that no build could start during it.

## The card ran under its 95 W default limit

As for the landings before it, the laptop was not in its performance power mode for this cell. `nvidia-smi` reported `power.default_limit` 95 W (maximum 175 W) and the platform profile read `balanced`. The software power cap flag was active in every one of the 326 samples, and the card drew at most 92.2 W. (Read after the cell, `enforced.power.limit` showed 150 W; the samples carry no limit column, and the draw stayed under 95 W throughout.)

So the absolute rates here are rates at that limit, not this card's best rates. The cell is an interleaved A B B A A B comparison of two binaries under the same limit, and that comparison is what the record carries.

## Summary

| | |
|---|---|
| window (2026-11-11, local) | 05:13:31 to 05:18:57 |
| start | 59 C, 1-minute load 1.85 |
| host during the cell | mean run queue 2.9, 7% user CPU |
| power | default limit 95 W, peak draw 92.2 W, software power cap active throughout |
| Ornith-1.5-35B-A3B | -0.09% (A 122.18, 121.36, 120.26; B 121.07, 122.65, 121.25 tok/s) |
| Qwen3.8-27B | +0.22% (A 31.10, 31.13, 31.22; B 31.08, 31.20, 31.21 tok/s) |
| verdict | pass, no arm separation on either model |

The percentages are the record's median deltas, candidate (B, `29e3632d87`) against base (A, `66ee31fcb6`). No thermal slowdown: the software and hardware thermal slowdown flags and the hardware slowdown flag were not active in any sample, in any run. The card peaked at 77 C.

## Files

- `throttle-before.csv`, `throttle-after.csv`: the card's throttle flags, temperature, power, utilization and clock right before and right after the cell.
- `throttle-1hz.csv`: the same once a second during the cell.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run (end time, samples, samples with a thermal or hardware slowdown flag active, temperature range, peak power).
- `arm_conditions.py`: the script that joins the samples with the perf logs' end times to make `arms.tsv`.
