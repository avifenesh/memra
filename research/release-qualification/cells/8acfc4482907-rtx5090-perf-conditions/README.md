# Card conditions around the RTX 5090 Laptop perf cell of 8acfc4482907

Receipts for the record next to this directory, `8acfc4482907-rtx5090-perf`. It covers the landing of memra#1297 (memra#1275) and memra#1320. These receipts are not part of the record's payload manifest, and the gate does not read them.

**This is not a quiet-box take.** The host's 1-minute load was 17.42 at the start and between 16.06 and 29.75 through the take; the two landings before this one took theirs at 4 to 5. The load came from other projects' work on the same machine and did not drop below 21 in a 40 minute wait (`load-quiet-wait-30s.tsv`, 10:41 to 11:21: 80 samples, median 29.1, maximum 67.1). The take was run at the recorded load because this landing's code is off the measured models' path: memra#1297 changes the step35 batched prime's tail split, which neither roster model runs, and memra#1320 changes a test binary. The record shows that the shared build did not regress; it is not a measurement of the models' rates on a quiet machine, and its absolute rates should not be compared with other records'.

The 5090's battery cell is in `8acfc4482907-rtx5090-battery`, from an earlier `gpulock run --perf` lease the same morning (10:09:27 to 10:40:21; battery 10:09:32 to 10:21:08). That lease ended without a perf take: the take waited for the load to fall under 10, and when it did for a moment the take script failed to start (it had been edited while it ran). This record comes from a second `--perf` lease (11:25:35 to 11:48:04), perf only. The profile was performance and the CPU cap 100/110 W. The perf cell held the host build lock, so no build could start during it.

## Two takes, the second banked

All times are local, 2026-11-14 (+02:00).

The heat rule applied to both: a take is void if either arm has a slowdown flag in more than 5% of its 1 Hz samples, or if the card starts at 86 C or hotter. An AC guard logged AC, profile and the CPU caps once a second; any second off AC would have stopped the cell and voided the take. There was no load gate (see above); the load was logged every 30 s through each take (`load-30s.tsv` for the banked one).

| | take 2 (void) | take 3 (banked) |
|---|---|---|
| window | 11:25:35 to 11:32:21 | 11:42:13 to 11:48:04 |
| start | card 67 C, 1-minute load 40.06 | card 69 C, 1-minute load 17.42, after a 5 minute rest and a wait for the card to read under 70 C |
| 1-minute load through the take | 20.54 to 42.78 | 16.06 to 29.75 |
| heat rule | arm A 10 of 219 samples (4.57%), arm B 11 of 185 (5.95%): void | arm A 8 of 181 (4.42%), arm B 8 of 169 (4.73%): the take counts (`heat-rule.txt`) |
| Ornith-1.5-35B-A3B | +1.18% (A 208.11, 218.16, 207.20; B 205.06, 211.65, 210.56 tok/s) | -0.23% (A 200.22, 209.82, 213.18; B 207.95, 211.61, 209.34 tok/s) |
| Qwen3.8-27B | -0.66% (A 47.46, 45.46, 46.66; B 47.08, 46.35, 45.64 tok/s) | +0.40% (A 47.69, 47.25, 47.12; B 47.50, 47.44, 47.19 tok/s) |

Take 3 in full:

| | take 3 |
|---|---|
| profile | performance, CPU cap 100/110 W (`gpulock run --perf`) |
| supply | AC online, profile `performance` and CPU cap 100/110 W in all 351 one-second samples |
| host during the cell | mean run queue 22.8; 52.5% user, 17.7% system, 27.3% idle CPU (the build lock was held, so no build ran) |
| power | default limit 95 W; enforced limit 150 W at P8 before the cell and 174.48 W at P0 after it; peak draw 179.44 W; software power cap flag in 81 of 350 samples |
| thermal | software thermal slowdown flag in 16 of 350 samples (4.57%), all in the six Qwen3.8-27B runs (11:45:26 to 11:48:03); no hardware thermal or hardware slowdown flag; 69 to 89 C |
| clock at 90% or more utilization | 1,642 to 2,430 MHz, median 1,905 (95 samples) |
| verdict | pass, no regression on either model |

The percentages are the record's median deltas, candidate (B, `8acfc44829`) against base (A, `0db086f87a`). In the banked take the arms overlap on both models: on Ornith A's 209.82 and 213.18 are above B's 207.95 and 209.34, and on Qwen3.8-27B A's 47.69 is above all three B runs. Take 2's arms overlap too; it is void for its thermal share only, and its runs are listed here because they were measured (`take2-void-arms.tsv`, `take2-void-heat-rule.txt`). The card idles near 70 C while the host is loaded, which is why the slowdown share is higher in both takes than in earlier landings' (1 to 2%).

## Power state

Nothing was changed by hand. `gpulock run --perf` set the profile and the CPU cap for the lease and lifts them at release. `platform-before.txt` and `platform-after.txt` hold the same knobs:

- profile `performance`, `ppt_pl1_spl` / `ppt_pl2_sppt` 100 / 110 W
- `nv_tgp` 150, `nv_dynamic_boost` 25, `nv_temp_target` 87
- AC online, CPU governor `powersave`

## Files

- `throttle-before.csv`, `throttle-after.csv`, `throttle-1hz.csv`: the card's throttle flags, temperature, power, utilization and clock before, after and once a second during the cell.
- `ac-profile-1hz.tsv`: time, AC online, platform profile, `ppt_pl1_spl` and `ppt_pl2_sppt`, once a second from the cell's start (`ac_guard17.py`).
- `platform-before.txt`, `platform-after.txt`: the asus-armoury knobs, profile, AC, CPU governor and the card's power limits.
- `power-at-lease.txt`: the same check at the second lease's start.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `load-30s.tsv`: time, the 1, 5 and 15 minute load and the card's temperature every 30 s from the rest before take 3 to its end (`load_log.py`).
- `load-quiet-wait-30s.tsv`: the same columns through the 40 minute wait for a quiet host, 10:41 to 11:21.
- `arms.tsv`: one row per run, from `arm_conditions.py`.
- `heat-rule.txt`, `arm_share17.py`: the heat-rule verdict and the script that computes it.
- `take2-void-arms.tsv`, `take2-void-heat-rule.txt`: the void take's rows and verdict.
