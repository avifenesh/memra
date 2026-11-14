# Card conditions around the RTX 5090 Laptop perf cell of 9c2a3ddf5ba0

Receipts for the record next to this directory, `9c2a3ddf5ba0-rtx5090-perf`. It covers the landing of memra#1310 (memra#1278) and memra#1317 (memra#1307). These receipts are not part of the record's payload manifest, and the gate does not read them.

The 5090's battery and serving cells are in `9c2a3ddf5ba0-rtx5090-battery-serving`. Both records come from one `gpulock run --perf` lease (07:54:26 to 08:33:18): battery and serving first (07:59:48 to 08:18:42, after the lease waited five minutes for the head's CPU build to finish under a loaded host), a 10 minute rest, then this perf take. The profile was performance and the CPU cap 100/110 W. The perf cell held the host build lock, so no build could start during it.

## One take, banked

All times are local, 2026-11-14 (+02:00).

The take started after a 10 minute rest, with the card under 70 C. The heat rule applied: the take would have been void if either arm had a slowdown flag in more than 5% of its 1 Hz samples, or if the card started at 86 C or hotter. An AC guard logged AC, profile and the CPU caps once a second; any second off AC would have stopped the cell and voided the take.

This take had one more gate. Other work on the host had pushed its 1-minute load to about 50 half an hour earlier, so the take was set to start only with the 1-minute load under 10, as in the two landings before it (5 to 8). `load-wait-30s.tsv` has the load and the card's temperature every 30 s through the rest: 18.35 at 08:18:43, 8.01 at 08:24:43, 4.43 at the start. The gate did not delay the take past the rest.

| | take 1 |
|---|---|
| window | 08:28:43 to 08:33:18 |
| start | card 59 C, 1-minute load 4.43 |
| profile | performance, CPU cap 100/110 W (`gpulock run --perf`) |
| supply | AC online, profile `performance` and CPU cap 100/110 W in all 275 one-second samples |
| host during the cell | mean run queue 5.3; 15.6% user, 8.1% system, 74.5% idle CPU (the build lock was held, so no build ran) |
| power | default limit 95 W; enforced limit 150 W at P8 before the cell and 175 W at P0 after it; peak draw 178.16 W; software power cap flag in 88 of 274 samples |
| thermal | software thermal slowdown flag in 4 of 274 samples (1.46%), all in the last three Qwen3.8-27B runs (08:32:26 to 08:33:16); no hardware thermal or hardware slowdown flag; 59 to 87 C |
| heat rule | arm A 2 of 149 samples (1.34%), arm B 2 of 125 (1.60%), start 59 C: the take counts (`heat-rule.txt`) |
| clock at 90% or more utilization | 1,507 to 2,505 MHz, median 1,927 (91 samples) |
| Ornith-1.5-35B-A3B | +2.58% (A 218.94, 215.66, 223.16; B 226.35, 224.59, 222.14 tok/s) |
| Qwen3.8-27B | +2.77% (A 48.59, 47.29, 47.20; B 48.61, 48.60, 48.27 tok/s) |
| verdict | pass, no regression on either model |

The percentages are the record's median deltas, candidate (B, `9c2a3ddf5b`) against base (A, `63c351a2f4`). The arms overlap on both models: A's 223.16 is above B's 222.14 on Ornith, and A's 48.59 is above B's 48.27 on Qwen3.8-27B. The two slowest Qwen3.8-27B runs (A's 47.29 and 47.20) are the fourth and fifth of that model's six, the first runs in which the slowdown flag came on (1 sample each, at 86 C).

memra#1310 changes server admission (`worker.rs`: an eviction is reaped before a reading or a retry counts on it), and memra#1317 changes which free-memory reading the speculative verify paths take on unified memory. The perf cell runs `run-gen`'s plain decode on a discrete card, which goes through neither.

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
- `load-wait-30s.tsv`: time, the 1, 5 and 15 minute load and the card's temperature every 30 s from the end of the battery to the take's start (`load_log.py`).
- `arms.tsv`: one row per run, from `arm_conditions.py`.
- `heat-rule.txt`, `arm_share17.py`: the heat-rule verdict and the script that computes it.
