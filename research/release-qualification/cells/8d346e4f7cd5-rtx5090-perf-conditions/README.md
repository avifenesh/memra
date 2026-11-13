# Card conditions around the RTX 5090 Laptop perf cell of 8d346e4f7cd5

Receipts for the record next to this directory, `8d346e4f7cd5-rtx5090-perf` (the landing of memra#1261, issue memra#1255). They are not part of the record's payload manifest; the gate does not read them.

The 5090's battery and serving cells are in `8d346e4f7cd5-rtx5090-battery-serving`. The perf cell ran as its own record, after a rest, on a quiet host, with the host build lock held so that no build could start during it.

## One take, banked

All times are local, 2026-11-13 (+02:00). The take held its own GPU lease from 02:10:11 and the host build lock from 02:10:45 (a build of another lane held it until then). It rested 10 minutes from the lock and started the timed cell at 02:20:45, with the card at 59 C (under 60 C) and a 1-minute load of 2.48 (under 4). The cell ended at 02:25:46.

| | |
|---|---|
| window | 02:20:45 to 02:25:46 |
| host during the cell | mean run queue 4.5, 9.4% user CPU (other sessions' CPU work; the build lock was held, so no build ran) |
| supply | AC online and platform profile `balanced` in all 935 samples of the log (from 02:10:11, the rest included), 301 of them in the window |
| power | default limit 95 W, enforced limit 155 W at P0, peak draw 157.32 W, software power cap flag active in 92 of 300 samples |
| thermal | software thermal slowdown flag active in 7 of 300 samples (2.33%), all in the Qwen3.8-27B runs; no hardware thermal or hardware slowdown flag; 59 to 88 C |
| clock at 90% or more utilization | 1,515 to 2,467 MHz, median 1,740 (96 samples) |
| Ornith-1.5-35B-A3B | -0.09% (A 217.32, 216.20, 215.17; B 215.61, 216.01, 216.48 tok/s) |
| Qwen3.8-27B | +0.35% (A 45.55, 45.49, 45.27; B 45.57, 45.78, 45.65 tok/s) |
| verdict | pass, no regression on either model |

The percentages are the record's median deltas, candidate (B, `8d346e4f7c`) against base (A, `11b368f7d7`). The arms overlap on Ornith-1.5-35B-A3B. On Qwen3.8-27B the lowest candidate run (45.57) is above the highest base run (45.55), a separation in the candidate's favour; the gate reads a separation only as a regression when the candidate is below.

## The thermal flags

The software thermal slowdown flag came on in 7 one-second samples, near the end of the cell, as the card reached its 87 C target:

| run | arm | samples | flagged | median clock at 90% or more utilization |
|---|---|---|---|---|
| 09-B | candidate | 24 | 1 (02:24:38) | 1,732 MHz |
| 10-A | base | 22 | 2 (02:24:59, 02:25:00) | 1,725 MHz |
| 11-A | base | 21 | 3 (02:25:20 to 02:25:22) | 1,732 MHz |
| 12-B | candidate | 21 | 1 (02:25:42) | 1,725 MHz |

Both arms carry flags: base 5 of 157 samples (3.2%), candidate 2 of 142 (1.4%); in the Qwen3.8-27B runs alone, base 5 of 79 (6.3%), candidate 2 of 74 (2.7%). The flagged runs' median clocks under load match the unflagged Qwen3.8-27B runs (07-A 1,736 MHz, 08-B 1,762 MHz). More of the flags fell on the base arm, which can only lower the base: with the flagged base runs left out, the Qwen3.8-27B base is 45.55 (07-A alone) against the candidate's median 45.65, still +0.22%. So the flags do not hide a candidate regression.

## Power state

Nothing on the machine was changed by the landing. `platform-before.txt` (02:10:11, at the lease) and `platform-after.txt` (02:25:46) hold the same values: `ppt_pl1_spl` / `ppt_pl2_sppt` 140 / 175, `nv_tgp` 150, `nv_dynamic_boost` 25, `nv_temp_target` 87, profile `balanced`, AC online, CPU governor `powersave`; card default limit 95 W, enforced limit 155 W at P0, max limit 175 W.

The previous landing's perf record (`6b6734f786eb-rtx5090-perf`) ran with the enforced limit at 160 W at P0, drew up to 170.35 W, peaked at 86 C and had no thermal flag. Here the enforced limit read 155 W at P0 at the lease and after the cell, the card drew up to 157.32 W and peaked at 88 C. Absolute rates are not comparable across records; the cell is an interleaved A B B A A B comparison of two binaries under the same conditions, and that comparison is what the record carries.

## Files

- `throttle-before.csv`, `throttle-after.csv`: the card's throttle flags, temperature, power, utilization and clock right before and right after the cell.
- `throttle-1hz.csv`: the same once a second during the cell.
- `ac-profile-1hz.tsv`: time, AC adapter online (1 or 0) and the platform profile, once a second from the lease start.
- `platform-before.txt`, `platform-after.txt`: the asus-armoury power knobs, platform profile, AC, CPU governor and the card's power limits at the lease start and right after the cell.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run (end time, samples, samples with a thermal or hardware slowdown flag active, temperature range, peak power).
- `arm_conditions.py`: the script that joins the samples with the perf logs' end times to make `arms.tsv`.
