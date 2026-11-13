# Card conditions around the RTX 5090 Laptop perf cell of 2342f0de3e37

Receipts for the record next to this directory, `2342f0de3e37-rtx5090-perf` (memra#1273, the
no-finite-logit fix, against base 202e1411fe). They are not part of the record's payload manifest;
the gate does not read them. The battery and serving cells are in
`2342f0de3e37-rtx5090-battery-serving`, on the same profile.

The perf cell ran as its own record under a `gpulock --perf` lease, with these conditions for the
whole lease:
- the performance platform profile;
- the CPU burst capped at 100/110 W (ppt_pl1/pl2);
- the host build lock held;
- a 10 minute rest, then a card under 70 C before the cell.

The owner's heat and power rule decided whether the take counts:
- the card under 86 C at the start;
- sw thermal slowdown on at most 5% of the 1 Hz samples of each arm;
- no second off AC or off the asked profile.

## Summary

| | |
|---|---|
| window (2026-11-13, local) | 12:45:18 to 12:52:54 |
| power | AC online and profile=performance in all 1058 one-second samples of the lease; CPU cap 100/110 W before and after (`platform-before.txt`, `platform-after.txt`) |
| start | 61 C, 1-minute load 7.4 |
| host during the cell | mean run queue 6.0, 12% user CPU |
| Ornith-1.5-35B-A3B | +0.01% (base median 213.46, head 213.49 tok/s) |
| Qwen3.8-27B | -0.04% (base median 47.17, head 47.15 tok/s) |
| heat | sw thermal slowdown in 4 of 456 samples (A 2 of 236 = 0.85%, B 2 of 220 = 0.91%); card 61 to 87 C, peak power 178 W |
| verdict | pass, no arm separation on either model |

Compare A against B inside this record only. Records before 06:45 ran on the balanced profile, and
those before 09:30 had no CPU cap.

## Takes that did not count (`void-takes/`)

These were taken on f386f0e8a5, the same fix before its rebase onto 202e1411fe and before the
skip-census test change.
- 08:23: void. The take was thermally clean, but the adapter read offline at 08:29:31 and the
  profile then fell to balanced. The cause was CPU burst and GPU at 175 W together, with no
  adapter headroom. Since then the `--perf` lease caps the CPU, and the AC-return hook re-applies
  the lease's profile.
- 08:50: void and stopped. The rig ran off AC from 08:53:35, a charger fault the owner fixed at
  09:25.
- 09:44 to 09:52: clean (AC, profile, cap and heat), but superseded. The skip-census change and the
  rebase changed the tested inputs, so it was retaken here.

## Files

- `throttle-before.csv`, `throttle-after.csv`, `throttle-1hz.csv`: the card's throttle flags, temperature, power, utilization and clock before, after and once a second during the cell.
- `ac-profile-1hz.tsv`: AC online and the platform profile once a second over the lease.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run, from `arm_conditions.py`.
- `gate.json`: the heat and power rule's verdict for the take, from `throttle_gate.py`.
- `platform-before.txt`, `platform-after.txt`: the platform power state around the lease.
- `rec5090.sh`: the script that ran the leases.
