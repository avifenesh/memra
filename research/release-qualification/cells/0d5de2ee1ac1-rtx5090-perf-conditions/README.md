# Card conditions around the RTX 5090 Laptop perf cell of 0d5de2ee1ac1

Receipts for the record next to this directory, `0d5de2ee1ac1-rtx5090-perf`. It covers the landing of memra#1270, memra#1274 (memra#1258 part 1) and memra#1280. These receipts are not part of the record's payload manifest, and the gate does not read them.

The 5090's battery cells are in `0d5de2ee1ac1-rtx5090-battery`. Both records ran in one GPU lease taken with `gpulock run --perf`, so the profile was performance and the CPU cap 100/110 W for the whole lease. The perf cell held the host build lock, so no build could start during it.

## One take, banked

All times are local, 2026-11-13 (+02:00).

- 10:16:31: the lease started (`power-at-lease.txt`): AC online, profile `performance`, `ppt_pl1_spl` / `ppt_pl2_sppt` 100 / 110 W, `nv_tgp` 150, `nv_dynamic_boost` 25, `nv_temp_target` 87.
- 10:39:31: the battery record ended.
- The perf take started after a 10 minute rest, with the card under 70 C.
- The heat rule applied to the take. It would have been void if either arm had a slowdown flag in more than 5% of its 1 Hz samples, or if the card started at 86 C or hotter.
- An AC guard logged AC, profile and the CPU caps once a second. Any second off AC would have stopped the cell and voided the take.
- The first take counted, so it was banked.

| | |
|---|---|
| window | 10:49:32 to 10:56:28 |
| start | card 62 C, 1-minute load 6.01 |
| profile | performance, CPU cap 100/110 W (`gpulock run --perf`) |
| supply | AC online, profile `performance` and CPU cap 100/110 W in all 415 one-second samples of the window (`ac-profile-1hz.tsv`: time, AC, profile, ppt_pl1, ppt_pl2) |
| host during the cell | mean run queue 6.3; 17.8% user, 9.1% system, 48.1% idle CPU (other sessions' work; the build lock was held, so no build ran) |
| power | default limit 95 W; enforced limit 150 W at P8 before the cell and 174.91 W at P0 after it; peak draw 174.78 W; software power cap flag in 93 of 415 samples |
| thermal | software thermal slowdown flag in 3 of 415 samples (0.72%), in the last two Qwen3.8-27B runs; no hardware thermal or hardware slowdown flag; 61 to 87 C |
| heat rule | arm A 1 of 219 samples (0.46%), arm B 2 of 195 (1.03%), start 62 C: the take counts (`heat-rule.txt`) |
| clock at 90% or more utilization | 1,365 to 2,482 MHz, median 1,860 (96 samples) |
| Ornith-1.5-35B-A3B | -0.42% (A 214.46, 212.41, 220.46; B 212.26, 213.55, 217.66 tok/s) |
| Qwen3.8-27B | +1.83% (A 48.45, 46.93, 46.98; B 48.04, 45.99, 47.84 tok/s) |
| verdict | pass, no regression on either model |

The percentages are the record's median deltas, candidate (B, `0d5de2ee1a`) against base (A, `a17add18f3`). The arms overlap on both models, and the runs spread wider on the performance profile than on balanced. The change on the timed path is memra#1274's prime schedule, which only differs for prompts of 4,097 to 4,111 tokens. The perf prompt is not in that range.

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
