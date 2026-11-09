# Card conditions around the RTX 5090 Laptop perf cells of e6613b4eee22

Receipts for two records next to this directory (memra#1153):

- `e6613b4eee22-rtx5090-battery-perf-serving`: the full record. Its battery and serving cells are the 5090's evidence. Its perf cell is the "first cell" below.
- `e6613b4eee22-rtx5090-perf`: a perf-only record taken on a quiet host from a cool card, the "quiet cell" below. It is the 5090's perf evidence.

These files are not part of either record's payload manifest; the gate does not read them.

## Summary

The perf verdict is a clean pass in both cells: no arm separation, medians within 2%.

| cell | window (2026-11-10, local) | start | Ornith-1.5-35B-A3B | Qwen3.8-27B | Qwen spread |
|---|---|---|---|---|---|
| first | 00:38:13 to 00:42:47 | 76 C, straight after the battery and serving cells; host run queue 5.8, 15% user CPU | -0.16% | -0.94% | 9.6% (41.03 to 45.18 tok/s) |
| quiet | 00:49:30 to 00:54:32 | 59 C after a 4 minute cool-down; 1-minute load 3.0; no build running (the host build lock was held) | -1.27% | -1.86% | 5.3% (43.89 to 46.29 tok/s) |

The software thermal slowdown flag was active in some samples of the Qwen3.8-27B runs in both cells. The hardware slowdown flags were never active. The Ornith runs had no slowdown sample in either cell.

| cell | samples with `sw_thermal_slowdown` Active inside perf runs | by run | by arm |
|---|---|---|---|
| first | 22 of 274 | 08-B 1, 09-B 2, 10-A 4, 11-A 6, 12-B 9 | A 10, B 12 |
| quiet | 9 of 301 | 09-B 1, 10-A 2, 11-A 2, 12-B 4 | A 4, B 5 |

## Reading

The slowdown is this laptop's steady state for this load at the `balanced` platform profile. It is not an effect of the host load or of the start temperature.

- One Qwen3.8-27B run is about 18 s of model load at about 30 W, then about 10 s of decode at 150 to 160 W.
- The card climbs about 14 C inside one decode burst (70 to 84 C on the first run of the quiet cell).
- The flag trips at 87 C, at the tail of a burst, once the card's floor between bursts has crept up. By run 09 the card sits at 74 to 76 C between bursts.
- A cool start only delays it. The 59 C start bought two clean Qwen runs.

The A B B A A B order spreads the throttled samples over both arms (A 4, B 5 in the quiet cell), so the comparison stays balanced. The first cell ran hotter and under more host load, and its Qwen rates fall with time in both arms (A 45.18, 44.48, 41.94; B 44.18, 44.06, 41.03).

The battery and serving cells of the full record also ran partly throttled (208 of 483 samples before its perf cell). They are correctness cells; a slowdown cannot change their verdicts.

## Files

- `quiet-cell/`: the perf-only record's cell.
  - `throttle-before.csv`, `throttle-after.csv`: the card's throttle flags, temperature, power, utilization and clock right before and right after.
  - `throttle-1hz.csv`: the same once a second during the cell.
  - `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
  - `arms.tsv`: one row per run (end time, samples, samples with a slowdown flag active, temperature range, peak power).
- `first-cell/`: the same for the full record. Its sampler started at 00:30:10, during the battery, and runs to the end of the perf cell.
- `arms.tsv` comes from `../e6613b4eee22-rtxpro6000-perf-conditions/arm_conditions.py`.
