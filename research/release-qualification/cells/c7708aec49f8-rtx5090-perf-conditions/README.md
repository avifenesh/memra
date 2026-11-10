# Card conditions around the RTX 5090 Laptop perf cell of c7708aec49f8

Receipts for the record next to this directory, `c7708aec49f8-rtx5090-perf` (the landing of memra#1170, #1179 and #1183, PR memra#1182). They are not part of the record's payload manifest; the gate does not read them.

The 5090's battery and serving cells are in `c7708aec49f8-rtx5090-battery-serving`. The perf cell ran as its own record, after a cool-down, with the host build lock held so that no build could start during it.

## Summary

| | |
|---|---|
| window (2026-11-10, local) | 10:40:32 to 10:45:29 |
| start | 58 C, 1-minute load 3.9 |
| host during the cell | mean run queue 6.3, 18% user CPU (background indexing jobs at low priority; no build) |
| Ornith-1.5-35B-A3B | -1.20% (A 217.71, 216.57, 217.70; B 215.09, 216.46, 208.01 tok/s) |
| Qwen3.8-27B | -0.95% (A 42.70, 45.74, 43.36; B 42.27, 45.28, 42.95 tok/s), spread 7.9% |
| verdict | pass, no arm separation on either model |

The software thermal slowdown flag was active in 10 of 296 one-second samples inside perf runs, all in Qwen3.8-27B runs: 08-B 1, 09-B 1, 10-A 2, 11-A 2, 12-B 4 (A 4, B 6). The hardware slowdown flags were never active. The Ornith runs had no slowdown sample. The card peaked at 87 C.

The Qwen spread is wider than in the two cells before it (5.2% and 5.6%), and the host was busier. The rates move together in time across both arms: neighbouring runs of the two arms are within 1% of each other (07-A 42.70 and 08-B 42.27; 09-B 45.28 and 10-A 45.74; 11-A 43.36 and 12-B 42.95). The interleaved order carries the comparison.

## Reading

This is the laptop's steady state for this load at the `balanced` platform profile, as measured before (`../e6613b4eee22-rtx5090-perf-conditions/README.md`): about 18 s of model load at about 30 W, then about 10 s of decode at 150 to 160 W, about 14 C of climb per burst, and the flag trips near 87 C at the tail of a burst once the floor between bursts has crept up.

Follow-up: memra#1160 asks the perf cell to record the throttle flags itself and to handle a throttled run.

## Files

- `throttle-before.csv`, `throttle-after.csv`: the card's throttle flags, temperature, power, utilization and clock right before and right after the cell.
- `throttle-1hz.csv`: the same once a second during the cell.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run (end time, samples, samples with a slowdown flag active, temperature range, peak power).
- `arm_conditions.py`: the script that joins the samples with the perf logs' end times to make `arms.tsv`.
