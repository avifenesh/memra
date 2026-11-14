# Card conditions around the RTX 5090 Laptop perf cell of 1184226212a4

Receipts for the record next to this directory, `1184226212a4-rtx5090-perf`. It covers the landing of memra#1302 (memra#1257). These receipts are not part of the record's payload manifest, and the gate does not read them.

The 5090's battery and serving cells are in `1184226212a4-rtx5090-battery-serving`. Both records come from one `gpulock run --perf` lease (00:10:06 to 00:44:29): battery and serving first, a 10 minute rest, then this perf take. The profile was performance and the CPU cap 100/110 W. The perf cell held the host build lock, so no build could start during it.

## One take, banked

All times are local, 2026-11-14 (+02:00).

The take started after a 10 minute rest, with the card under 70 C. The heat rule applied: the take would have been void if either arm had a slowdown flag in more than 5% of its 1 Hz samples, or if the card started at 86 C or hotter. An AC guard logged AC, profile and the CPU caps once a second; any second off AC would have stopped the cell and voided the take.

| | take 1 |
|---|---|
| window | 00:37:52 to 00:44:29 |
| start | card 63 C, 1-minute load 7.77 |
| profile | performance, CPU cap 100/110 W (`gpulock run --perf`) |
| supply | AC online, profile `performance` and CPU cap 100/110 W in all 397 one-second samples |
| host during the cell | mean run queue 7.2; 16.4% user, 11.1% system, 54.5% idle CPU (the build lock was held, so no build ran) |
| power | default limit 95 W; enforced limit 150 W at P8 before the cell and 165.57 W at P0 after it; peak draw 180.31 W; software power cap flag in 96 of 397 samples |
| thermal | software thermal slowdown flag in 2 of 397 samples (0.50%), both in the last Qwen3.8-27B run; no hardware thermal or hardware slowdown flag; 63 to 87 C |
| heat rule | arm A 0 of 209 samples (0.00%), arm B 2 of 187 (1.07%), start 63 C: the take counts (`heat-rule.txt`) |
| clock at 90% or more utilization | 1,695 to 2,475 MHz, median 1,950 (97 samples) |
| Ornith-1.5-35B-A3B | +1.22% (A 209.34, 212.90, 208.27; B 213.11, 210.50, 211.89 tok/s) |
| Qwen3.8-27B | +0.95% (A 42.39, 45.45, 46.21; B 45.88, 45.31, 46.38 tok/s) |
| verdict | pass, no regression on either model |

The percentages are the record's median deltas, candidate (B, `1184226212`) against base (A, `df1d23a764`). The arms overlap on both models: A's 212.90 is above B's 210.50 and 211.89 on Ornith, and A's 45.45 and 46.21 are above B's 45.31 on Qwen3.8-27B.

The host carried more background load than in the previous landings' takes (1-minute load 7.77 at the start against 2.7 to 3.9), and the absolute rates of both arms are lower than theirs (Ornith about 210 against about 220 tok/s). The comparison is between this take's two interleaved arms, which ran under the same load.

memra#1302 changes only admission's driver-headroom rung in the server: on a discrete card its free reading is `cuMemGetInfo` as before, and the `[admit-trim]` line gains `reading=driver`. The perf cell runs `run-gen`'s plain decode, which does not go through server admission.

## Power state

Nothing was changed by hand. `gpulock run --perf` set the profile and the CPU cap for the lease and lifts them at release. `platform-before.txt` and `platform-after.txt` hold the same knobs:

- profile `performance`, `ppt_pl1_spl` / `ppt_pl2_sppt` 100 / 110 W
- `nv_tgp` 150, `nv_dynamic_boost` 25, `nv_temp_target` 87
- AC online, CPU governor `powersave`

## Files

- `throttle-before.csv`, `throttle-after.csv`, `throttle-1hz.csv`: the card's throttle flags, temperature, power, utilization and clock before, after and once a second during the cell.
- `ac-profile-1hz.tsv`: time, AC online, platform profile, `ppt_pl1_spl` and `ppt_pl2_sppt`, once a second from the cell's start (`ac_guard17.py`).
- `platform-before.txt`, `platform-after.txt`: the asus-armoury knobs, profile, AC, CPU governor and the card's power limits.
- `power-at-lease.txt`: the same check at the lease's start.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run, from `arm_conditions.py`.
- `heat-rule.txt`, `arm_share17.py`: the heat-rule verdict and the script that computes it.
