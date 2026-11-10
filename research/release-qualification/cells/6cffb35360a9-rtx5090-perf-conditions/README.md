# Card conditions around the RTX 5090 Laptop perf cell of 6cffb35360a9

Receipts for the record next to this directory, `6cffb35360a9-rtx5090-perf` (the landing of memra#1130, #1163, #1162, #1155 and #1156, PR memra#1164). They are not part of the record's payload manifest; the gate does not read them.

The 5090's battery and serving cells are in `6cffb35360a9-rtx5090-battery-serving`. The perf cell ran as its own record, after a cool-down, on a quiet host, with the host build lock held so that no build could start during it.

## Summary

| | |
|---|---|
| window (2026-11-10, local) | 04:12:32 to 04:17:30 |
| start | 59 C, 1-minute load 3.9 |
| host during the cell | mean run queue 3.7, 13% user CPU |
| Ornith-1.5-35B-A3B | +1.82% (A 217.28, 209.45, 207.96; B 213.26, 219.82, 211.64 tok/s) |
| Qwen3.8-27B | +2.29% (A 44.54, 43.69, 46.04; B 45.94, 45.01, 45.56 tok/s), spread 5.2% |
| verdict | pass, no arm separation on either model |

The software thermal slowdown flag was active in 9 of 297 one-second samples inside perf runs, all in Qwen3.8-27B runs: 08-B 1, 09-B 1, 10-A 2, 11-A 2, 12-B 3 (A 4, B 5). The hardware slowdown flags were never active. The Ornith runs had no slowdown sample. The card peaked at 87 C.

## Reading

This is the laptop's steady state for this load at the `balanced` platform profile, as measured before (`../e6613b4eee22-rtx5090-perf-conditions/README.md`): about 18 s of model load at about 30 W, then about 10 s of decode at 150 to 160 W, about 14 C of climb per burst, and the flag trips near 87 C at the tail of a burst once the floor between bursts has crept up. The A B B A A B order spreads the throttled samples over both arms.

Follow-up: memra#1160 asks the perf cell to record the throttle flags itself and to handle a throttled run.

## Files

- `throttle-before.csv`, `throttle-after.csv`: the card's throttle flags, temperature, power, utilization and clock right before and right after the cell.
- `throttle-1hz.csv`: the same once a second during the cell.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run (end time, samples, samples with a slowdown flag active, temperature range, peak power).
- `arm_conditions.py`: the script that joins the samples with the perf logs' end times to make `arms.tsv`.
