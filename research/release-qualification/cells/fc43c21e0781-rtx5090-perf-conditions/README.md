# Card conditions around the RTX 5090 Laptop perf cell of fc43c21e0781

Receipts for the record next to this directory, `fc43c21e0781-rtx5090-perf` (the v0.140.0 release commit, memra#1161). They are not part of the record's payload manifest; the gate does not read them.

The 5090's battery and serving cells are in `fc43c21e0781-rtx5090-battery-serving`. The perf cell ran as its own record, after a cool-down, on a quiet host, with the host build lock held so that no build could start during it.

## Summary

| | |
|---|---|
| window (2026-11-10, local) | 02:20:58 to 02:25:50 |
| start | 59 C, 1-minute load 3.8 |
| host during the cell | mean run queue 4.2, 14% user CPU |
| Ornith-1.5-35B-A3B | +0.67% (A 215.39, 209.47, 209.40; B 210.87, 210.79, 213.62 tok/s) |
| Qwen3.8-27B | -2.31% (A 43.11, 45.80, 45.44; B 44.39, 43.95, 45.04 tok/s), spread 6.1% |
| verdict | pass, no arm separation on either model |

The software thermal slowdown flag was active in 14 of 291 one-second samples inside perf runs, all in Qwen3.8-27B runs: 08-B 1, 09-B 2, 10-A 2, 11-A 3, 12-B 6 (A 5, B 9). The hardware slowdown flags were never active. The Ornith runs had no slowdown sample. The card peaked at 90 C in the last run.

## Reading

This is the laptop's steady state for this load at the `balanced` platform profile, as measured on the commit before (`../e6613b4eee22-rtx5090-perf-conditions/README.md`): about 18 s of model load at about 30 W, then about 10 s of decode at 150 to 160 W, about 14 C of climb per burst, and the flag trips near 87 C at the tail of a burst once the floor between bursts has crept up. A cool start delays it by one or two Qwen runs. The A B B A A B order spreads the throttled samples over both arms.

Follow-up: memra#1160 asks the perf cell to record the throttle flags itself and to handle a throttled run.

## Files

- `throttle-before.csv`, `throttle-after.csv`: the card's throttle flags, temperature, power, utilization and clock right before and right after the cell.
- `throttle-1hz.csv`: the same once a second during the cell.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run (end time, samples, samples with a slowdown flag active, temperature range, peak power).
- `arm_conditions.py`: the script that joins the samples with the perf logs' end times to make `arms.tsv`.
