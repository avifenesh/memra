# Card conditions around the RTX 5090 Laptop perf cell of 25ccfe509cde

Receipts for the record next to this directory, `25ccfe509cde-rtx5090-perf`. It covers the landing of the DeepSeek-V4.1 route: memra#1281, memra#1282, memra#1283 and memra#1285. These receipts are not part of the record's payload manifest, and the gate does not read them.

The 5090's battery and serving cells are in `25ccfe509cde-rtx5090-battery-serving`. Both records ran in one GPU lease taken with `gpulock run --perf`, so the profile was performance and the CPU cap 100/110 W for the whole lease. The perf cell held the host build lock, so no build could start during it.

## One take, banked

All times are local, 2026-11-13 (+02:00).

- 20:02:02: the lease started.
- 20:03:19: the battery began inside the lease without waiting for the CPU inventory, so the card did not idle; the inventory came back green at 20:24. The state at that point (`power-at-lease.txt`): AC online, profile `performance`, `ppt_pl1_spl` / `ppt_pl2_sppt` 100 / 110 W, `nv_tgp` 150, `nv_dynamic_boost` 25, `nv_temp_target` 87.
- 20:20:37: the battery and serving record ended.
- The perf take started after a 10 minute rest, with the card under 70 C.
- The heat rule applied to the take. It would have been void if either arm had a slowdown flag in more than 5% of its 1 Hz samples, or if the card started at 86 C or hotter.
- An AC guard logged AC, profile and the CPU caps once a second. Any second off AC would have stopped the cell and voided the take.
- The first take counted, so it was banked.

| | |
|---|---|
| window | 20:30:37 to 20:37:02 |
| start | card 59 C, 1-minute load 2.74 |
| profile | performance, CPU cap 100/110 W (`gpulock run --perf`) |
| supply | AC online, profile `performance` and CPU cap 100/110 W in all 385 one-second samples of the window |
| host during the cell | mean run queue 4.2; 7.5% user, 9.9% system, 56.5% idle CPU (other sessions' work; the build lock was held, so no build ran) |
| power | default limit 95 W; enforced limit 150 W at P8 before the cell and 175 W at P0 after it; peak draw 173.81 W; software power cap flag in 94 of 384 samples |
| thermal | software thermal slowdown flag in 1 of 384 samples (0.26%), in the last Qwen3.8-27B run; no hardware thermal or hardware slowdown flag; 59 to 87 C |
| heat rule | arm A 0 of 201 samples (0.00%), arm B 1 of 182 (0.55%), start 59 C: the take counts (`heat-rule.txt`) |
| clock at 90% or more utilization | 1,755 to 2,505 MHz, median 1,950 (93 samples) |
| Ornith-1.5-35B-A3B | +0.56% (A 223.05, 217.89, 218.42; B 219.65, 218.84, 220.94 tok/s) |
| Qwen3.8-27B | +0.23% (A 48.73, 48.36, 48.60; B 48.68, 48.78, 48.71 tok/s) |
| verdict | pass, no regression on either model |

The percentages are the record's median deltas, candidate (B, `25ccfe509c`) against base (A, `d6ae4e525d`). The arms overlap on both models. The landing adds the DeepSeek-V4.1 route in its own modules; the timed roster models run no code it changes.

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
- `power-at-lease.txt`: the same check when the battery began.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run (end time, samples, samples with a thermal or hardware slowdown flag, temperature range, peak power), from `arm_conditions.py`.
- `heat-rule.txt`, `arm_share17.py`: the heat-rule verdict and the script that computes it from `arms.tsv`.
