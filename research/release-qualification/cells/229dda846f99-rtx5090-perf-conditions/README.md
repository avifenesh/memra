# Card conditions around the RTX 5090 Laptop perf cell of 229dda846f99

Receipts for the record next to this directory, `229dda846f99-rtx5090-perf` (the landing of memra#1251, which closes issue memra#1227). They are not part of the record's payload manifest; the gate does not read them.

The 5090's battery and serving cells are in `229dda846f99-rtx5090-battery-serving`. The perf cell ran as its own record, after a rest, on a quiet host, with the host build lock held so that no build could start during it.

## One take, from a rested start

The take rested 10 minutes under its own lease after the battery and serving record and started at 54 C. No thermal slowdown sample in it, so it was not retaken.

## The supply did not change

`ac-profile-1hz.tsv` logs the AC adapter's `online` value and the platform profile once a second from 12:24:41, when the take's lease was granted (the rest included). All 277 samples inside the timed window, and all 878 of the log, read AC online and `balanced`.

## The card ran above its 95 W default limit

As in the landings before this one, the card did not sit at its 95 W default limit. `nvidia-smi` reported `power.default_limit` 95 W and `power.max_limit` 175 W; `enforced.power.limit` read 160 W before the landing's battery record and 150 W right after the perf cell, and the card drew up to 160.26 W (107 to 120 W peaks in the Ornith runs, 157 to 160 W in the Qwen3.8-27B runs). Under load the clock read 1,582 to 2,055 MHz (median 1,830). The platform profile read `balanced` throughout. The software power cap flag was active in every one of the 277 card samples; the 1 Hz samples carry no limit column.

So the absolute rates here match the previous landings' records (Ornith about 182 tok/s) and must not be compared with the older ones taken at 95 W (about 126). The cell is an interleaved A B B A A B comparison of two binaries under the same conditions, and that comparison is what the record carries.

## Summary

| | |
|---|---|
| window (2026-11-12, local) | 12:34:42 to 12:39:19 |
| start | 54 C, 1-minute load 0.70, after a 10 minute rest |
| host during the cell | mean run queue 2.1, 5% user CPU (other sessions' CPU work; the build lock was held, so no build ran) |
| supply | AC online and platform profile `balanced` in every 1 Hz sample |
| power | default limit 95 W, enforced limit 150 to 160 W, peak draw 160.26 W, software power cap flag active throughout |
| Ornith-1.5-35B-A3B | +0.44% (A 180.65, 180.53, 186.13; B 179.73, 181.44, 185.10 tok/s) |
| Qwen3.8-27B | +0.02% (A 40.40, 40.71, 40.69; B 40.47, 40.70, 40.70 tok/s) |
| verdict | pass, no arm separation on either model |

The percentages are the record's median deltas, candidate (B, `229dda846f`) against base (A, `46920874c0`). The arms overlap on both models. No thermal slowdown: the software and hardware thermal slowdown flags and the hardware slowdown flag were not active in any of the 277 samples, in any run. The card peaked at 87 C.

## Files

- `throttle-before.csv`, `throttle-after.csv`: the card's throttle flags, temperature, power, utilization and clock right before and right after the cell.
- `throttle-1hz.csv`: the same once a second during the cell.
- `ac-profile-1hz.tsv`: time, AC adapter online (1 or 0) and the platform profile, once a second.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run (end time, samples, samples with a thermal or hardware slowdown flag active, temperature range, peak power).
- `arm_conditions.py`: the script that joins the samples with the perf logs' end times to make `arms.tsv`.
