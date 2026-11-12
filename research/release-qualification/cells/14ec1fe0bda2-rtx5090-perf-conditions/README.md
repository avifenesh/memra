# Card conditions around the RTX 5090 Laptop perf cell of 14ec1fe0bda2

Receipts for the record next to this directory, `14ec1fe0bda2-rtx5090-perf` (the landing of memra#1248, which fixes issue memra#1173). They are not part of the record's payload manifest; the gate does not read them.

The 5090's battery and serving cells are in `14ec1fe0bda2-rtx5090-battery-serving`. The perf cell ran as its own record, after a rest, on a quiet host, with the host build lock held so that no build could start during it.

## Four takes; the fourth is the record

All times are local, 2026-11-12 (+02:00). Each take held its own GPU lease and rested 10 minutes in it before the timed cell, which started only under 60 C.

| take | lease | timed window | outcome |
|---|---|---|---|
| 1 | 15:34:50 | 15:44:51 to 15:54:54, stopped | void: the AC adapter went offline at 15:45:26. Only run 01-A (220.55 tok/s) finished on AC. Runs 02 to 06 ran on battery (enforced limit 33 W, about 19 tok/s) and the cell was stopped in run 07. |
| 2 | 15:56:32 | none | the lease waited for AC to return. The laptop shut down on battery at 17:02 first, so no timed cell ran and no cell receipt exists. |
| 3 | 18:31:53 | 18:43:58 to 18:48:56 | perf gate pass (Ornith -0.47%, Qwen3.8-27B +0.06%), but the software thermal slowdown flag was active in 12 of 297 samples (4.04%), over the 2% bar. Retaken. |
| 4 | 18:49:43 | 19:01:44 to 19:06:33 | perf gate pass (Ornith -0.01%, Qwen3.8-27B -0.14%), software thermal slowdown in 5 of 288 samples (1.74%), in both arms. Banked. |

Take 1's receipts are in `take1-void/` and take 3's in `take3-throttled/` (its conditions, its 12 perf logs, and its record as `perf-record.json`). The files at the top of this directory belong to take 4.

## The power table changed with the restart

The laptop came back up at 18:10 on AC. Its platform power table then differed from take 1 and from the landings before this one. Nothing was changed by the landing; `platform-before.txt` and `platform-after.txt` hold the values for each take.

| | take 1 (15:35) | takes 3 and 4 |
|---|---|---|
| `ppt_pl1_spl` / `ppt_pl2_sppt` | 90 / 120 | 140 / 175 |
| `nv_tgp`, `nv_dynamic_boost`, `nv_temp_target` | 150, 25, 87 | 150, 25, 87 |
| `enforced.power.limit` | 150 W (P8) | 175 W at P0, 150 W at P8 |
| `power.default_limit`, `power.max_limit` | 95 W, 175 W | 95 W, 175 W |
| platform profile | `balanced` | `balanced` |

The previous landings' records ran with the enforced limit at 150 to 160 W and a peak draw of about 160 W. Here the card drew up to 173.52 W (take 3) and 176.18 W (take 4).

## The thermal slowdown flag

At 175 W the card reached its 87 C temperature target in the Qwen3.8-27B runs (peaks 87 C in take 3, 88 C in take 4), and the software thermal slowdown flag came on for one to four seconds at a time. The hardware thermal slowdown and hardware slowdown flags were not active in any sample of either take. No Ornith run had a flagged sample.

| run | take 3 flagged | take 4 flagged |
|---|---|---|
| 07-A qwen | 0 | 0 |
| 08-B qwen | 1 | 0 |
| 09-B qwen | 3 | 1 |
| 10-A qwen | 2 | 1 |
| 11-A qwen | 2 | 1 |
| 12-B qwen | 4 | 2 |

The landings before this one used a 2% bar on flagged samples, set when the card ran at 150 to 160 W. After take 3 the bar for this 175 W state was set to: perf gate pass, flagged samples in both arms, and no more than about 5% of the window. Take 4 meets both the old and the new bar.

## Clock and power under load

The software power cap flag was active in 87 of take 4's 288 samples and 82 of take 3's 297 (in every sample in the previous landing's record).

| | take 3 | take 4 |
|---|---|---|
| clock at 90% or more utilization | 1,545 to 2,475 MHz, median 1,908 (92 samples) | 1,725 to 2,497 MHz, median 1,920 (90 samples) |
| peak draw | 173.52 W | 176.18 W |
| temperature | 57 to 87 C | 58 to 88 C |

The absolute rates (Ornith about 221 tok/s, Qwen3.8-27B about 48.7) are higher than in the previous landing's record (about 182 and 40.7). Do not compare them across records. The cell is an interleaved A B B A A B comparison of two binaries under the same conditions, and that comparison is what the record carries.

## Summary of take 4

| | |
|---|---|
| window (2026-11-12, local) | 19:01:44 to 19:06:33 |
| start | 58 C, 1-minute load 1.67, after a 10 minute rest |
| host during the cell | mean run queue 2.3, 5.0% user CPU (other sessions' CPU work; the build lock was held, so no build ran) |
| supply | AC online and platform profile `balanced` in all 890 samples of the log (from 18:51:44, the rest included), 290 of them in the window |
| power | default limit 95 W, enforced limit 175 W at P0, peak draw 176.18 W, software power cap flag active in 87 of 288 samples |
| thermal | software thermal slowdown in 5 of 288 samples (1.74%): 2 in arm A, 3 in arm B, all in Qwen3.8-27B runs; peak 88 C |
| Ornith-1.5-35B-A3B | -0.01% (A 221.19, 219.37, 223.02; B 221.17, 224.17, 219.57 tok/s) |
| Qwen3.8-27B | -0.14% (A 48.74, 48.72, 48.51; B 48.56, 48.75, 48.65 tok/s) |
| verdict | pass, no arm separation on either model |

The percentages are the record's median deltas, candidate (B, `14ec1fe0bd`) against base (A, `cdfdfd6006`). The arms overlap on both models.

## Files

- `throttle-before.csv`, `throttle-after.csv`: the card's throttle flags, temperature, power, utilization and clock right before and right after the cell.
- `throttle-1hz.csv`: the same once a second during the cell.
- `ac-profile-1hz.tsv`: time, AC adapter online (1 or 0) and the platform profile, once a second from the end of the AC wait.
- `platform-before.txt`, `platform-after.txt`: the asus-armoury power knobs, platform profile, AC, CPU governor and the card's power limits right before the rest and right after the cell.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run (end time, samples, samples with a thermal or hardware slowdown flag active, temperature range, peak power).
- `arm_conditions.py`: the script that joins the samples with the perf logs' end times to make `arms.tsv`.
- `take1-void/`, `take3-throttled/`: the same receipts for takes 1 and 3, with their perf logs.
