# Card conditions around the RTX 5090 Laptop perf cell of 7a32d9b10be6

Receipts for the record next to this directory, `7a32d9b10be6-rtx5090-perf`. It covers the landing of memra#1256 (issue memra#1244), carried on top of memra#1272. These receipts are not part of the record's payload manifest, and the gate does not read them.

The 5090's battery and serving cells are in `7a32d9b10be6-rtx5090-battery-serving`. The perf cell ran as its own record, with the host build lock held so that no build could start during it.

## One take, banked

All times are local, 2026-11-13 (+02:00).

- 05:47: the battery and serving records started under a GPU lease.
- 06:04:10: those records ended. The card did no GPU work from then until the perf cell started.
- 06:04:10: the first perf take began, in the same lease. Its start condition was a card under 60 C and a 1-minute load under 4, and it was never met: at about 06:13 the idle card read 61 to 63 C and the load was 6.4 to 6.6.
  - The heat comes from the laptop's shared heatpipe, which carries the CPU's heat while other sessions load the CPU.
  - The load came from other sessions' CPU work, not from a build.
- The start condition was relaxed to a card under 65 C and a 1-minute load under 8. The rig has 24 threads, and the decode cells are GPU-bound.
- The temperature and load only stand in for a quiet machine. The cell interleaves the two binaries in one lease (A B B A A B), so a steady background load reaches both arms alike. The guard that decides is the bank rule: slowdown flags at 5% or less of samples in both arms.
- 06:16:48: the first take was stopped before its timed cell; nothing ran.
- 06:16:52: this take's lease was granted, 4 s later, with no GPU work in between. So the rest counts from 06:04:10, 12 minutes before the start.
- 06:16:56: the timed cell started with the card at 64 C and a 1-minute load of 7.78. It ended at 06:22:40.

| | |
|---|---|
| window | 06:16:56 to 06:22:40 |
| start | card 64 C, 1-minute load 7.78 (start condition: under 65 C and under 8) |
| host during the cell | mean run queue 8.9; 22.8% user, 10.0% system, 56.7% idle CPU (other sessions' work; the build lock was held, so no build ran) |
| supply | AC online and platform profile `balanced` in all 347 samples of the log (from 06:16:52), 344 of them in the window |
| power | default limit 95 W; enforced limit 150 W at P8 before the cell and 155 W at P0 after it; peak draw 156.88 W; software power cap flag active in 87 of 344 samples |
| thermal | software thermal slowdown flag active in 9 of 344 samples (2.62%), all in the Qwen3.8-27B runs; no hardware thermal or hardware slowdown flag; 64 to 87 C |
| clock at 90% or more utilization | 1,515 to 2,460 MHz, median 1,747 (96 samples) |
| Ornith-1.5-35B-A3B | -0.07% (A 216.43, 214.86, 213.98; B 214.72, 216.34, 209.40 tok/s) |
| Qwen3.8-27B | -0.15% (A 45.52, 45.54, 45.30; B 45.36, 45.57, 45.45 tok/s) |
| verdict | pass, no regression on either model |

The percentages are the record's median deltas, candidate (B, `7a32d9b10b`) against base (A, `7a43ac1e56`). The arms overlap on both models. #1256 and #1272 change no code on the decode path these cells time; #1272 is a harness change.

## The thermal flags

The software thermal slowdown flag came on as the card reached its 87 C target in the Qwen3.8-27B runs:

| run | arm | samples | flagged | median clock at 90% or more utilization |
|---|---|---|---|---|
| 07-A | base | 37 | 0 | 1,740 MHz |
| 08-B | candidate | 37 | 1 (06:20:55) | 1,732 MHz |
| 09-B | candidate | 33 | 2 (06:21:28, 06:21:29) | 1,755 MHz |
| 10-A | base | 25 | 2 (06:21:53, 06:21:54) | 1,740 MHz |
| 11-A | base | 21 | 2 (06:22:14, 06:22:15) | 1,740 MHz |
| 12-B | candidate | 21 | 2 (06:22:35, 06:22:36) | 1,725 MHz |

Flag counts by arm:

| | base | candidate |
|---|---|---|
| whole cell | 4 of 182 samples (2.2%) | 5 of 161 (3.1%) |
| Qwen3.8-27B runs alone | 4 of 83 (4.8%) | 5 of 91 (5.5%) |

The flags fall evenly: one or two per run from 08-B on, in both arms. The flagged runs' median clocks under load match the unflagged 07-A (1,740 MHz). No Ornith run has a flag. Both arms stay under 5% over the cell, so the take is banked under the bank rule with these counts disclosed. The previous landing's record (`8d346e4f7cd5-rtx5090-perf`) was banked the same way, with 6.3% of its Qwen-only base samples flagged.

## Power state

Nothing on the machine was changed by the landing. `platform-before.txt` (06:16:52, at the lease) and `platform-after.txt` (06:22:40) hold the same knobs:

- `ppt_pl1_spl` / `ppt_pl2_sppt` 140 / 175
- `nv_tgp` 150
- `nv_dynamic_boost` 25
- `nv_temp_target` 87
- profile `balanced`, AC online, CPU governor `powersave`

The card's default limit was 95 W and its max limit 175 W. The enforced limit read 150 W at P8 before the cell and 155 W at P0 after it. Absolute rates are not comparable across records. The cell is an interleaved A B B A A B comparison of two binaries under the same conditions, and that comparison is what the record carries.

## Files

- `throttle-before.csv`, `throttle-after.csv`: the card's throttle flags, temperature, power, utilization and clock right before and right after the cell.
- `throttle-1hz.csv`: the same once a second during the cell.
- `ac-profile-1hz.tsv`: time, AC adapter online (1 or 0) and the platform profile, once a second from the lease start.
- `platform-before.txt`, `platform-after.txt`: the asus-armoury power knobs, platform profile, AC, CPU governor and the card's power limits at the lease start and right after the cell.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run (end time, samples, samples with a thermal or hardware slowdown flag active, temperature range, peak power).
- `arm_conditions.py`: the script that joins the samples with the perf logs' end times to make `arms.tsv`.
