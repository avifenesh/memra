# Card conditions around the RTX 5090 Laptop perf cell of 6b6734f786eb

Receipts for the record next to this directory, `6b6734f786eb-rtx5090-perf` (the landing of memra#1259, part of issue memra#1221). They are not part of the record's payload manifest; the gate does not read them.

The 5090's battery and serving cells are in `6b6734f786eb-rtx5090-battery-serving`. The perf cell ran as its own record, after a rest, on a quiet host, with the host build lock held so that no build could start during it.

## One take, banked

All times are local, 2026-11-12 (+02:00). The take held its own GPU lease from 20:31:06, rested 10 minutes in it, and started the timed cell at 20:41:07, with the card at 57 C (under 60 C) and a 1-minute load of 2.29 (under 4). The cell ended at 20:46:04.

| | |
|---|---|
| window | 20:41:07 to 20:46:04 |
| host during the cell | mean run queue 2.6, 6.3% user CPU (other sessions' CPU work; the build lock was held, so no build ran) |
| supply | AC online and platform profile `balanced` in all 898 samples of the log (from 20:31:06, the rest included), 297 of them in the window |
| power | default limit 95 W, enforced limit 160 W at P0, peak draw 170.35 W, software power cap flag active in 96 of 297 samples |
| thermal | no slowdown flag (software thermal, hardware thermal, hardware) in any of the 297 samples; 57 to 86 C, peak in run 12-B |
| clock at 90% or more utilization | 1,530 to 2,467 MHz, median 1,800 (96 samples) |
| Ornith-1.5-35B-A3B | +0.40% (A 216.83, 218.04, 221.54; B 214.99, 218.92, 219.27 tok/s) |
| Qwen3.8-27B | +2.33% (A 46.29, 47.18, 44.05; B 47.37, 47.42, 46.84 tok/s) |
| verdict | pass, no arm separation on either model |

The percentages are the record's median deltas, candidate (B, `6b6734f786`) against base (A, `9d7c086c67`). The arms overlap on both models. Run 11-A's 44.05 tok/s is the lowest Qwen3.8-27B run; it has no flagged sample and its median clock under load (1,770 MHz) matches its neighbours (1,777 to 1,807 MHz).

## Power state

Nothing on the machine was changed by the landing. `platform-before.txt` (20:31:06, at the lease) and `platform-after.txt` (20:46:04) hold the same values: `ppt_pl1_spl` / `ppt_pl2_sppt` 140 / 175, `nv_tgp` 150, `nv_dynamic_boost` 25, `nv_temp_target` 87, profile `balanced`, AC online, CPU governor `powersave`; card default limit 95 W, enforced limit 160 W at P0, max limit 175 W.

The previous landing's perf record (`14ec1fe0bda2-rtx5090-perf`) ran with the enforced limit at 175 W at P0, drew up to 176.18 W and reached its 87 C target in the Qwen3.8-27B runs. Here the enforced limit read 160 W at P0 at the lease and after the cell, the card drew up to 170.35 W, peaked at 86 C, and no thermal flag came on. Absolute rates are not comparable across records; the cell is an interleaved A B B A A B comparison of two binaries under the same conditions, and that comparison is what the record carries.

## Files

- `throttle-before.csv`, `throttle-after.csv`: the card's throttle flags, temperature, power, utilization and clock right before and right after the cell.
- `throttle-1hz.csv`: the same once a second during the cell.
- `ac-profile-1hz.tsv`: time, AC adapter online (1 or 0) and the platform profile, once a second from the lease start.
- `platform-before.txt`, `platform-after.txt`: the asus-armoury power knobs, platform profile, AC, CPU governor and the card's power limits at the lease start and right after the cell.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run (end time, samples, samples with a thermal or hardware slowdown flag active, temperature range, peak power).
- `arm_conditions.py`: the script that joins the samples with the perf logs' end times to make `arms.tsv`.
