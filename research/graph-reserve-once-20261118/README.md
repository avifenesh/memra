# graph-reserve-once-20261118: the draft graph pool's growth charged once per stream, on GB10 (memra#1360)

Verdict: the split reads as intended on GB10 (PR #1362's first form, e25411f8cd; review round 1 replaced its held flag with a live reading of the graph pool, which this run did not exercise). This is memra#1307's MTP serving on GB10 (unit-a, sm_121a): Qwen3.8-27B NVFP4 + Q5_K head, masked regime MTP draft, 262,144 context, 32 sessions. It ran under systemd with `MEMRA_PANIC_AFTER=1`, one worker respawn.

| | main 96390ef0e4 (`research/respawn-drafthw-20261118/`, PR #1353's branch) | this change |
|---|---|---|
| first boot, first capture (the spec-cost calibration's) | high-water 34 MB (parked 34, pool peak 2) | `draft graph-pool reserve: 32MB (the graph pool 0MB -> 32MB ...)`, then high-water 2 MB (parked 2 net of the 32 MB growth, pool peak 2) |
| per-session draft-state charge | +36 MB | +2 MB |
| boot calibration | measured 1,214 MB, floor 1,536 (static) | measured 1,246 MB, floor 1,536 (static) |
| after the respawn | high-water 2 MB, charge +2 MB | high-water 2 MB (growth 0), charge +2 MB |
| NRestarts, serving again after the panic | 0, 125 s | 0, 125 s |

- At 32 speculative sessions the draft-state charge goes from about 1.15 GB to about 64 MB. The 32 MB reserve is already in the free reading admission starts from once the boot probe's capture holds it.
- The boot calibration subtracts the per-session charge from the probe's peak, so its measured floor rises by the 32 MB it no longer subtracts. The floor stays the 1,536 MiB static floor here.
- Review round 2 records the bracket's own growth (`graph1 - cap_graph0`) as the reserve, not the device-wide reading after the capture, so another tenant's graphs in the pool at that moment (the Qwen prime graph on the 5090, DSpark verify graphs) are not counted. On this run the two are the same: the pool read 0 MiB at the capture's entry.
- Without #1361, the respawned worker finds the graph pool already holding the dead worker's 32 MiB: its first capture grows the pool by 0, so it records no reserve, and the charge is the per-session 2 MB while the pool holds the 32 MiB. With #1361's trim, its first capture grows the pool from 0 again and records the 32 MiB reserve: the same +2 MB while the pool holds it. With the live reading, a trim between captures charges the 32 MB back until the pool holds them again.
- The sharing holds for four live uploaded draft graphs: the spec-cost calibration's pooled four-session cohort reads the graph pool at 32 / 32 MiB at each of its four brackets (`research/respawn-drafthw-20261118/gb10/wd-dm96/server.log` on PR #1353's branch, the `[diag1351b]` lines before the `pooled_overhead=Some((4, ...))` line). Graphs on one stream share their physical memory (CUDA programming guide §4.2.5.3.2), which needs every draft graph on the one stream and balanced alloc and free nodes.
- Not settled here: more than four concurrent speculative sessions, which is still owed.

Conditions: 2026-11-18, one GB10 unit, no page-cache fill (MemAvailable about 118 GB), one run; nothing else ran on the unit. Raw: `gb10/wd-f1360/`. Harness: `tools/gb10-build.sh`, `tools/gb10-wd.sh ARM`.
