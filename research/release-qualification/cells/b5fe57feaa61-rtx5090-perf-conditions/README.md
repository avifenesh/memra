# Card conditions around the RTX 5090 Laptop perf cell of b5fe57feaa61

Receipts for the record next to this directory, `b5fe57feaa61-rtx5090-perf`. It covers the landing of memra#1288 (memra#1276). These receipts are not part of the record's payload manifest, and the gate does not read them.

The 5090's battery and serving cells are in `b5fe57feaa61-rtx5090-battery-serving`. Both records ran in one GPU lease taken with `gpulock run --perf`, so the profile was performance and the CPU cap 100/110 W for the whole lease. The perf cell held the host build lock, so no build could start during it.

## One take, banked

All times are local, 2026-11-13 (+02:00).

- 15:50:20: the lease started (`power-at-lease.txt`): AC online, profile `performance`, `ppt_pl1_spl` / `ppt_pl2_sppt` 100 / 110 W, `nv_tgp` 150, `nv_dynamic_boost` 25, `nv_temp_target` 87.
- The lease waited inside itself for the CPU inventory's verdict, which came back green at 15:53:20.
- 16:13: the battery and serving record ended.
- The perf take started after a 10 minute rest, with the card under 70 C.
- The heat rule applied to the take. It would have been void if either arm had a slowdown flag in more than 5% of its 1 Hz samples, or if the card started at 86 C or hotter.
- An AC guard logged AC, profile and the CPU caps once a second. Any second off AC would have stopped the cell and voided the take.
- The first take counted, so it was banked.

| | |
|---|---|
| window | 16:22:51 to 16:28:32 |
| start | card 59 C, 1-minute load 3.59 |
| profile | performance, CPU cap 100/110 W (`gpulock run --perf`) |
| supply | AC online, profile `performance` and CPU cap 100/110 W in all 341 one-second samples of the window |
| host during the cell | mean run queue 5.0; 11.2% user, 8.5% system, 66.7% idle CPU (other sessions' work; the build lock was held, so no build ran) |
| power | default limit 95 W; enforced limit 150 W at P8 before the cell and 167.36 W at P0 after it; peak draw 180.34 W; software power cap flag in 90 of 341 samples |
| thermal | software thermal slowdown flag in 5 of 341 samples (1.47%), in the last three Qwen3.8-27B runs; no hardware thermal or hardware slowdown flag; 58 to 87 C |
| heat rule | arm A 3 of 182 samples (1.65%), arm B 2 of 157 (1.27%), start 59 C: the take counts (`heat-rule.txt`) |
| clock at 90% or more utilization | 1,665 to 2,497 MHz, median 1,924 (94 samples) |
| Ornith-1.5-35B-A3B | -0.32% (A 221.63, 221.50, 221.58; B 220.86, 222.48, 218.21 tok/s) |
| Qwen3.8-27B | +0.15% (A 48.65, 48.19, 46.36; B 48.23, 48.69, 48.26 tok/s) |
| verdict | pass, no regression on either model |

The percentages are the record's median deltas, candidate (B, `b5fe57feaa`) against base (A, `c0a292dc64`). The arms overlap on both models. memra#1288 is admission arithmetic: on a single device its charge equals main's at every length, and it changes no prime schedule or numeric program.

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
