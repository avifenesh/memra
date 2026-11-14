# Card conditions around the RTX 5090 Laptop perf cell of d463a4c66361

Receipts for the record next to this directory, `d463a4c66361-rtx5090-perf`. It covers the landing of memra#1268 (memra#1266). These receipts are not part of the record's payload manifest, and the gate does not read them.

The 5090's battery and serving cells are in `d463a4c66361-rtx5090-battery-serving`, from a first `gpulock run --perf` lease (21:03 to 21:46). This perf record is the rested retake, from a second `--perf` lease. The profile was performance and the CPU cap 100/110 W in both. The perf cell held the host build lock, so no build could start during it.

## Two takes; the second is banked

All times are local, 2026-11-13 (+02:00).

**Take 1** (first lease, 21:40:22 to 21:46:33): not banked.
- It started at 57 C after a 10 minute rest and counted under the heat rule (slowdown flags in 1 of 189 base samples and 1 of 182 candidate samples).
- Its perf gate passed. But Ornith-1.5-35B-A3B's arms separated, every candidate run below every base run: A 222.39, 220.07, 220.76; B 217.58, 219.41, 216.26 tok/s, -1.44%. That is more than the 5090's per-run spread of about 1%, so the cell was retaken.
- Qwen3.8-27B was -0.39%, with the arms overlapping.

**Take 2** (second lease, from 21:54:38): banked.
- It started after a 10 minute rest, with the card under 70 C.
- The heat rule applied. The take would have been void if either arm had a slowdown flag in more than 5% of its 1 Hz samples, or if the card started at 86 C or hotter.
- An AC guard logged AC, profile and the CPU caps once a second. Any second off AC would have stopped the cell and voided the take.

| | take 2 |
|---|---|
| window | 22:04:39 to 22:10:01 |
| start | card 57 C, 1-minute load 2.73 |
| profile | performance, CPU cap 100/110 W (`gpulock run --perf`) |
| supply | AC online, profile `performance` and CPU cap 100/110 W in all 322 one-second samples |
| host during the cell | mean run queue 4.8; 12.3% user, 8.4% system, 65.6% idle CPU (the build lock was held, so no build ran) |
| power | default limit 95 W; enforced limit 150 W at P8 before the cell and 172.45 W at P0 after it; peak draw 178.21 W; software power cap flag in 89 of 322 samples |
| thermal | software thermal slowdown flag in 5 of 322 samples (1.55%), in the last three Qwen3.8-27B runs; no hardware thermal or hardware slowdown flag; 57 to 87 C |
| heat rule | arm A 3 of 167 samples (1.80%), arm B 2 of 154 (1.30%), start 57 C: the take counts (`heat-rule.txt`) |
| clock at 90% or more utilization | 1,695 to 2,505 MHz, median 1,912 (90 samples) |
| Ornith-1.5-35B-A3B | -1.27% (A 218.65, 220.61, 220.88; B 217.81, 218.85, 217.63 tok/s) |
| Qwen3.8-27B | +0.04% (A 48.47, 48.55, 48.63; B 48.57, 48.83, 48.42 tok/s) |
| verdict | pass, no regression on either model |

The percentages are the record's median deltas, candidate (B, `d463a4c663`) against base (A, `8031c7ef90`). In take 2 the arms overlap on both models: B's 218.85 is above A's 218.65.

Both takes put the Ornith candidate about 1.3% low. The perf cell runs `run-gen`, which is plain decode, not speculative decode. memra#1268's new code (`fa_rows_growth`, `Engine::fa_rows_growth_bytes` and a `VerifyWs` method) is called only from the speculative scheduler's pricing in `spec.rs`, so it is not on this cell's timed path. The gate reads the takes as no regression.

This record ran on the performance profile. Records taken before about 06:42 on 2026-11-13 ran on balanced. Their absolute rates are not comparable with this one; each record compares its own two arms under one profile.

## Power state

Nothing was changed by hand. `gpulock run --perf` set the profile and the CPU cap for each lease and lifts them at release. `platform-before.txt` and `platform-after.txt` (take 2) hold the same knobs:

- profile `performance`, `ppt_pl1_spl` / `ppt_pl2_sppt` 100 / 110 W
- `nv_tgp` 150, `nv_dynamic_boost` 25, `nv_temp_target` 87
- AC online, CPU governor `powersave`

## Files (take 2)

- `throttle-before.csv`, `throttle-after.csv`, `throttle-1hz.csv`: the card's throttle flags, temperature, power, utilization and clock before, after and once a second during the cell.
- `ac-profile-1hz.tsv`: time, AC online, platform profile, `ppt_pl1_spl` and `ppt_pl2_sppt`, once a second from the cell's start (`ac_guard17.py`).
- `platform-before.txt`, `platform-after.txt`: the asus-armoury knobs, profile, AC, CPU governor and the card's power limits.
- `power-at-lease.txt`: the same check at the second lease's start.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run, from `arm_conditions.py`.
- `heat-rule.txt`, `arm_share17.py`: the heat-rule verdict and the script that computes it.
