# Card conditions around the RTX 5090 Laptop perf cell of f53bfef3cdeb

Receipts for the record next to this directory, `f53bfef3cdeb-rtx5090-perf` (the landing of memra#1235 and memra#1232, PR memra#1236). They are not part of the record's payload manifest; the gate does not read them.

The 5090's battery and serving cells are in `f53bfef3cdeb-rtx5090-battery-serving`. The perf cell ran as its own record, after a cool-down, on a quiet host, with the host build lock held so that no build could start during it.

## This is the second take; the first was discarded

A first perf-only record at this commit (window 22:09:40 to 22:17:08) is not banked. The laptop left AC power at 22:11:01, inside the window. Its battery hook set the platform profile to `quiet`, the CPU governor to powersave and GPU persistence mode off; AC came back at 22:13:35. The card's idle clock fell to 180 MHz and two base runs of Ornith measured 19.2 and 19.3 tok/s against 121.6 in the run before them, so that record's verdict (+536%) measured the power supply, not the binaries. No thermal or hardware slowdown flag was active in it either; only the 1 Hz clock and power columns and the system journal show the cause.

For this take `ac-profile-1hz.tsv` logs the AC adapter's `online` value and the platform profile once a second. All 402 samples inside the window read AC online and `balanced`.

## The card ran under its 95 W default limit

As for the landings before it, the laptop was not in its performance power mode for this cell. `nvidia-smi` reported `power.default_limit` 95 W (maximum 175 W) and the platform profile read `balanced`. The software power cap flag was active in every one of the 409 samples, and the card drew at most 94.1 W. (Read after the cell, `enforced.power.limit` showed 150 W; the samples carry no limit column, and the draw stayed under 95 W throughout.)

So the absolute rates here are rates at that limit, not this card's best rates. The cell is an interleaved A B B A A B comparison of two binaries under the same limit, and that comparison is what the record carries.

## Summary

| | |
|---|---|
| window (2026-11-11, local) | 22:21:04 to 22:27:54 |
| start | 56 C, 1-minute load 3.87 |
| host during the cell | mean run queue 4.6, 9% user CPU (other sessions' CPU work; the build lock was held, so no build ran) |
| supply | AC online and platform profile `balanced` in every 1 Hz sample |
| power | default limit 95 W, peak draw 94.1 W, software power cap active throughout |
| Ornith-1.5-35B-A3B | +0.57% (A 125.52, 128.11, 125.66; B 126.37, 126.06, 127.44 tok/s) |
| Qwen3.8-27B | -0.27% (A 32.86, 33.02, 32.99; B 32.87, 32.90, 32.98 tok/s) |
| verdict | pass, no arm separation on either model |

The percentages are the record's median deltas, candidate (B, `f53bfef3cd`) against base (A, `98337756cf`). The arms overlap on both models. No thermal slowdown: the software and hardware thermal slowdown flags and the hardware slowdown flag were not active in any sample, in any run. The card peaked at 76 C.

## Files

- `throttle-before.csv`, `throttle-after.csv`: the card's throttle flags, temperature, power, utilization and clock right before and right after the cell.
- `throttle-1hz.csv`: the same once a second during the cell.
- `ac-profile-1hz.tsv`: time, AC adapter online (1 or 0) and the platform profile, once a second from the lease's start.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run (end time, samples, samples with a thermal or hardware slowdown flag active, temperature range, peak power).
- `arm_conditions.py`: the script that joins the samples with the perf logs' end times to make `arms.tsv`.
