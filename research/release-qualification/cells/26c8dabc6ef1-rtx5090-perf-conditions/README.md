# Card conditions around the RTX 5090 Laptop perf cell of 26c8dabc6ef1

Receipts for the record next to this directory, `26c8dabc6ef1-rtx5090-perf`. It covers the landing of the DeepSeek-V4.1 CUTLASS lane (memra#1286) with memra#1294 and memra#1312. These receipts are not part of the record's payload manifest, and the gate does not read them.

The 5090's battery and serving cells are in `26c8dabc6ef1-rtx5090-battery-serving`. Both records come from one `gpulock run --perf` lease (04:38:44 to 05:09:38): battery and serving first, a 10 minute rest, then this perf take. The profile was performance and the CPU cap 100/110 W. The perf cell held the host build lock, so no build could start during it.

## One take, banked

All times are local, 2026-11-14 (+02:00).

The take started after a 10 minute rest, with the card under 70 C. The heat rule applied: the take would have been void if either arm had a slowdown flag in more than 5% of its 1 Hz samples, or if the card started at 86 C or hotter. An AC guard logged AC, profile and the CPU caps once a second; any second off AC would have stopped the cell and voided the take.

| | take 1 |
|---|---|
| window | 05:05:04 to 05:09:38 |
| start | card 58 C, 1-minute load 5.30 |
| profile | performance, CPU cap 100/110 W (`gpulock run --perf`) |
| supply | AC online, profile `performance` and CPU cap 100/110 W in all 274 one-second samples |
| host during the cell | mean run queue 5.7; 13.3% user, 9.3% system, 73.1% idle CPU (the build lock was held, so no build ran) |
| power | default limit 95 W; enforced limit 150 W at P8 before the cell and 175 W at P0 after it; peak draw 177.35 W; software power cap flag in 90 of 274 samples |
| thermal | software thermal slowdown flag in 5 of 274 samples (1.82%), all in the last three Qwen3.8-27B runs (05:08:53 to 05:09:35); no hardware thermal or hardware slowdown flag; 58 to 89 C |
| heat rule | arm A 3 of 146 samples (2.05%), arm B 2 of 127 (1.57%), start 58 C: the take counts (`heat-rule.txt`) |
| clock at 90% or more utilization | 1,650 to 2,505 MHz, median 1,912 (91 samples) |
| Ornith-1.5-35B-A3B | +1.18% (A 221.95, 223.88, 220.36; B 224.57, 224.88, 222.08 tok/s) |
| Qwen3.8-27B | +2.69% (A 48.83, 46.25, 47.50; B 48.78, 48.80, 48.32 tok/s) |
| verdict | pass, no regression on either model |

The percentages are the record's median deltas, candidate (B, `26c8dabc6e`) against base (A, `bbc6f83aea`). The arms overlap on both models: A's 223.88 is above B's 222.08 on Ornith, and A's 48.83 is above all three B runs on Qwen3.8-27B. The two slowest Qwen3.8-27B runs (A's 46.25 and 47.50) are the fourth and fifth of that model's six, the first runs in which the slowdown flag came on (1 and 2 samples, at 86 and 87 C).

The diff against the base is the deepseek-v4.1 route's modules and their CUTLASS units (vendored CUTLASS 4.8.1), the engine's build script and manifest, the removal of the DSV4 dense split-K archive, the `dsv41-gate` test binary, docs and receipts. The perf cell runs `run-gen`'s plain decode on the two roster models.

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
