# Card conditions around the RTX 5090 Laptop perf cell of c41f81b5b7dd

Receipts for the record next to this directory, `c41f81b5b7dd-rtx5090-perf` (the landing of memra#1241, part of issue memra#1211). They are not part of the record's payload manifest; the gate does not read them.

The 5090's battery and serving cells are in `c41f81b5b7dd-rtx5090-battery-serving`. The perf cell ran as its own record, after a rest, on a quiet host, with the host build lock held so that no build could start during it.

## This record is the second take

The first take started one minute after the battery and serving record ended, at 59 C on a heat-soaked chassis (window 03:20:05 to 03:24:56). It passed the gate (Ornith +1.11%, Qwen3.8-27B -0.71%, arms overlapping on both models; A 181.79, 179.61, 179.30 and B 182.22, 181.35, 181.61 tok/s on Ornith; A 40.41, 40.68, 40.68 and B 40.33, 40.39, 40.68 tok/s on Qwen3.8-27B). But the software thermal slowdown flag was active in 4 of its 291 samples, single samples at 86 to 88 C at the end of the last three Qwen3.8-27B runs (one in each A run, two in the B run). No earlier landing's perf cell had one. The rule was set before the second take ran: retake once from a rested start, and a passing retake is the banked record whatever its numbers. The first take's per-run table and samples are in `first-take/`.

The second take rested 10 minutes under its lease and started at 53 C. No thermal slowdown sample in it.

## The supply did not change

`ac-profile-1hz.tsv` logs the AC adapter's `online` value and the platform profile once a second from 03:29:14, when the take obtained the build lock (the rest included). All 285 samples inside the timed window, and all 885 of the log, read AC online and `balanced`. The first take's log (`first-take/ac-profile-1hz.tsv`, 362 samples) reads the same.

## The card ran above its 95 W default limit

As in the landing before this one, the card did not sit at its 95 W default limit. `nvidia-smi` reported `power.default_limit` 95 W and `power.max_limit` 175 W; `enforced.power.limit` read 150 W before the landing's battery record and 160 W after each perf take, and the card drew up to 160.1 W (109 to 129 W peaks in the Ornith runs, 156 to 160 W in the Qwen3.8-27B runs). Under load the clock read 1,747 to 2,055 MHz (median 2,040). The platform profile read `balanced` throughout. The software power cap flag was active in every one of the 284 card samples; the samples carry no limit column.

So the absolute rates here match the previous landing's record (Ornith about 182 tok/s) and must not be compared with the older ones taken at 95 W (about 126). The cell is an interleaved A B B A A B comparison of two binaries under the same conditions, and that comparison is what the record carries.

## Summary

| | |
|---|---|
| window (2026-11-12, local) | 03:39:14 to 03:43:59 |
| start | 53 C, 1-minute load 1.89, after a 10 minute rest |
| host during the cell | mean run queue 2.8, 7% user CPU (other sessions' CPU work; the build lock was held, so no build ran) |
| supply | AC online and platform profile `balanced` in every 1 Hz sample |
| power | default limit 95 W, enforced limit 150 to 160 W, peak draw 160.1 W, software power cap flag active throughout |
| Ornith-1.5-35B-A3B | +0.98% (A 180.64, 185.20, 181.25; B 183.03, 183.01, 184.29 tok/s) |
| Qwen3.8-27B | +0.12% (A 40.45, 40.42, 40.70; B 40.45, 40.50, 40.71 tok/s) |
| verdict | pass, no arm separation on either model |

The percentages are the record's median deltas, candidate (B, `c41f81b5b7`) against base (A, `1d3be5b0df`). The arms overlap on both models. No thermal slowdown: the software and hardware thermal slowdown flags and the hardware slowdown flag were not active in any of the 284 samples, in any run. The card peaked at 85 C.

## Files

- `throttle-before.csv`, `throttle-after.csv`: the card's throttle flags, temperature, power, utilization and clock right before and right after the cell.
- `throttle-1hz.csv`: the same once a second during the cell.
- `ac-profile-1hz.tsv`: time, AC adapter online (1 or 0) and the platform profile, once a second.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run (end time, samples, samples with a thermal or hardware slowdown flag active, temperature range, peak power).
- `arm_conditions.py`: the script that joins the samples with the perf logs' end times to make `arms.tsv`.
- `first-take/`: the first take's `arms.tsv`, `throttle-1hz.csv` and `ac-profile-1hz.tsv`.
