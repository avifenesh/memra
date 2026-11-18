# graph-reserve-once-20261118: the draft graph pool's growth charged once per stream, on GB10 (memra#1360)

Verdict: the split reads as intended on GB10 (PR #1360's head e25411f8cd). This is memra#1307's MTP serving on GB10 (unit-a, sm_121a): Qwen3.8-27B NVFP4 + Q5_K head, masked regime MTP draft, 262,144 context, 32 sessions. It ran under systemd with `MEMRA_PANIC_AFTER=1`, one worker respawn.

| | main 96390ef0e4 (`research/respawn-drafthw-20261118/`, PR #1353's branch) | this change |
|---|---|---|
| first boot, first capture | high-water 34 MB (parked 34, pool peak 2) | `draft graph-pool reserve: 32MB (the graph pool 0MB -> 32MB ...)`, then high-water 2 MB (parked 2 net of the 32 MB growth, pool peak 2) |
| per-session draft-state charge | +36 MB | +2 MB |
| boot calibration | measured 1,214 MB, floor 1,536 (static) | measured 1,246 MB, floor 1,536 (static) |
| after the respawn | high-water 2 MB, charge +2 MB | high-water 2 MB (growth 0), charge +2 MB |
| NRestarts, serving again after the panic | 0, 125 s | 0, 125 s |

- At 32 speculative sessions the draft-state charge goes from about 1.15 GB to about 64 MB. The 32 MB reserve is already in the free reading admission starts from once the boot probe's capture holds it.
- The boot calibration subtracts the per-session charge from the probe's peak, so its measured floor rises by the 32 MB it no longer subtracts. The floor stays the 1,536 MiB static floor here.
- Without #1361, the respawned worker finds the graph pool already held: growth 0, reserve 0, charge +2 MB. With #1361's trim, the respawned worker's first capture grows the pool again and records the 32 MB reserve as held, so the charge is the same +2 MB.
- Not settled here: the captures ran one at a time. Many concurrent speculative sessions sharing the 32 MiB is the assumption this charge rests on (#1360).

Conditions: 2026-11-18, one GB10 unit, no page-cache fill (MemAvailable about 118 GB), one run; nothing else ran on the unit. Raw: `gb10/wd-f1360/`. Harness: `tools/gb10-build.sh`, `tools/gb10-wd.sh ARM`.
