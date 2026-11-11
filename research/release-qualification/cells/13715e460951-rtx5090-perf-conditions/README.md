# Card conditions around the RTX 5090 Laptop perf cell of 13715e460951

Receipts for the record next to this directory, `13715e460951-rtx5090-perf` (the landing of memra#1237, issue memra#1234). They are not part of the record's payload manifest; the gate does not read them.

The 5090's battery cells are in `13715e460951-rtx5090-battery`. The perf cell ran as its own record, after a cool-down, on a quiet host, with the host build lock held so that no build could start during it.

## The supply did not change

`ac-profile-1hz.tsv` logs the AC adapter's `online` value and the platform profile once a second from the start of the lease (00:39:03, the cool-down included). All 351 samples inside the timed window, and all 1,315 of the lease, read AC online and `balanced`.

## The card ran above its 95 W default limit this time

The earlier landings' perf cells on this laptop sat at the card's 95 W default limit. This one did not. `nvidia-smi` reported `power.default_limit` 95 W and `power.max_limit` 175 W as before, but `enforced.power.limit` read 150 W before the landing's battery record and 160 W right after this cell, and the card drew up to 173.8 W (107 to 141 W peaks in the Ornith runs, 152 to 174 W in the Qwen3.8-27B runs). The platform profile read `balanced` throughout. The software power cap flag was active in every one of the 351 samples; the samples carry no limit column.

So the absolute rates here are higher than the earlier landings' records (Ornith about 182 tok/s against about 126) and must not be compared with them. The cell is an interleaved A B B A A B comparison of two binaries under the same conditions, and that comparison is what the record carries.

## Summary

| | |
|---|---|
| window (2026-11-12, local) | 00:55:08 to 01:00:59 |
| start | 57 C, 1-minute load 3.95 (after a 16 minute wait for other sessions' CPU checks to end) |
| host during the cell | mean run queue 4.2, 11% user CPU (other sessions' CPU work; the build lock was held, so no build ran) |
| supply | AC online and platform profile `balanced` in every 1 Hz sample |
| power | default limit 95 W, enforced limit 150 to 160 W, peak draw 173.8 W, software power cap flag active throughout |
| Ornith-1.5-35B-A3B | -0.09% (A 181.91, 180.35, 185.63; B 181.75, 182.33, 181.06 tok/s) |
| Qwen3.8-27B | -0.07% (A 40.45, 40.68, 40.68; B 40.65, 40.55, 40.68 tok/s) |
| verdict | pass, no arm separation on either model |

The percentages are the record's median deltas, candidate (B, `13715e4609`) against base (A, `d1ed663256`). The arms overlap on both models. No thermal slowdown: the software and hardware thermal slowdown flags and the hardware slowdown flag were not active in any sample, in any run. The card peaked at 85 C.

## Files

- `throttle-before.csv`, `throttle-after.csv`: the card's throttle flags, temperature, power, utilization and clock right before and right after the cell.
- `throttle-1hz.csv`: the same once a second during the cell.
- `ac-profile-1hz.tsv`: time, AC adapter online (1 or 0) and the platform profile, once a second from the lease's start.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run (end time, samples, samples with a thermal or hardware slowdown flag active, temperature range, peak power).
- `arm_conditions.py`: the script that joins the samples with the perf logs' end times to make `arms.tsv`.
