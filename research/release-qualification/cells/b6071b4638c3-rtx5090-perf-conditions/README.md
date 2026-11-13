# Card conditions around the RTX 5090 Laptop perf cell of b6071b4638c3

Receipts for the record next to this directory, `b6071b4638c3-rtx5090-perf`. It covers the landing of memra#1267 (memra#1221 part 2). These receipts are not part of the record's payload manifest, and the gate does not read them.

The 5090's battery and serving cells are in `b6071b4638c3-rtx5090-battery-serving`. Both records ran in one GPU lease taken with `gpulock run --perf`, so the profile was performance and the CPU cap 100/110 W for the whole lease. The perf cell held the host build lock, so no build could start during it.

## One take, banked

All times are local, 2026-11-13 (+02:00).

- 14:11:25: the lease started (`power-at-lease.txt`): AC online, profile `performance`, `ppt_pl1_spl` / `ppt_pl2_sppt` 100 / 110 W, `nv_tgp` 150, `nv_dynamic_boost` 25, `nv_temp_target` 87.
- 14:27:51: the battery and serving record ended.
- The perf take started after a 10 minute rest, with the card under 70 C.
- The heat rule applied to the take. It would have been void if either arm had a slowdown flag in more than 5% of its 1 Hz samples, or if the card started at 86 C or hotter.
- An AC guard logged AC, profile and the CPU caps once a second. Any second off AC would have stopped the cell and voided the take.
- The first take counted, so it was banked.

| | |
|---|---|
| window | 14:37:52 to 14:45:01 |
| start | card 58 C, 1-minute load 2.90 |
| profile | performance, CPU cap 100/110 W (`gpulock run --perf`) |
| supply | AC online, profile `performance` and CPU cap 100/110 W in all 429 one-second samples of the log (428 in the window) |
| host during the cell | mean run queue 8.7; 16.4% user, 11.7% system, 42.9% idle CPU (other sessions' work; the build lock was held, so no build ran) |
| power | default limit 95 W; enforced limit 150 W at P8 before the cell and 166.03 W at P0 after it; peak draw 178.75 W; software power cap flag in 92 of 428 samples |
| thermal | software thermal slowdown flag in 6 of 428 samples (1.40%), in the last four Qwen3.8-27B runs; no hardware thermal or hardware slowdown flag; 58 to 88 C |
| heat rule | arm A 2 of 221 samples (0.90%), arm B 4 of 205 (1.95%), start 58 C: the take counts (`heat-rule.txt`) |
| clock at 90% or more utilization | 1,470 to 2,497 MHz, median 1,822 (95 samples) |
| Ornith-1.5-35B-A3B | -1.00% (A 221.89, 219.99, 221.00; B 221.83, 218.78, 215.66 tok/s) |
| Qwen3.8-27B | -0.78% (A 48.23, 44.31, 47.74; B 47.37, 48.42, 46.96 tok/s) |
| verdict | pass, no regression on either model |

The percentages are the record's median deltas, candidate (B, `b6071b4638`) against base (A, `c2d2388bd0`), and the perf gate passes both models. The arms overlap on both:
- On Ornith, B's best run (221.83) sits inside A's range (219.99 to 221.89). The lowest Ornith run, 06-B (215.66), drew at most 155.1 W, against 168.91 to 178.63 W for the other Ornith runs.
- On Qwen3.8-27B, A's lowest run (10-A, 44.31) is below every B run.

This record ran on the performance profile. Records taken before about 06:42 on 2026-11-13 ran on balanced. Their absolute rates are not comparable with this one; each record compares its own two arms under one profile.

## Power state

Nothing was changed by hand. `gpulock run --perf` set the profile and the CPU cap for the lease and lifts them at release. `platform-before.txt` (at the cell's start) and `platform-after.txt` (right after it) hold the same knobs:

- profile `performance`, `ppt_pl1_spl` / `ppt_pl2_sppt` 100 / 110 W
- `nv_tgp` 150, `nv_dynamic_boost` 25, `nv_temp_target` 87
- AC online, CPU governor `powersave`

## Files

- `throttle-before.csv`, `throttle-after.csv`, `throttle-1hz.csv`: the card's throttle flags, temperature, power, utilization and clock before, after and once a second during the cell.
- `ac-profile-1hz.tsv`: time, AC online, platform profile, `ppt_pl1_spl` and `ppt_pl2_sppt`, once a second from the cell's start (`ac_guard17.py`, which would have stopped the cell on any second off AC).
- `platform-before.txt`, `platform-after.txt`: the asus-armoury knobs, profile, AC, CPU governor and the card's power limits.
- `power-at-lease.txt`: the same check at the lease start.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run (end time, samples, samples with a thermal or hardware slowdown flag, temperature range, peak power), from `arm_conditions.py`.
- `heat-rule.txt`, `arm_share17.py`: the heat-rule verdict and the script that computes it from `arms.tsv`.
