# respawn-drafthw-20261118: the respawned worker's draft-state measurement on GB10 (memra#1351, PR #1353)

Verdict: the PR's change does not fix it. On GB10 (unit-a, sm_121a), memra#1307's MTP serving (Qwen3.8-27B NVFP4 + Q5_K head, masked regime MTP draft, 262,144 context, 32 sessions) runs under systemd with `MEMRA_PANIC_AFTER=1`. One worker respawn per run. The respawned worker measures the draft state at 2 MB on the PR's head as on main, and a second run that reads the graph pool at every capture shows why, and that the first boot is the one that over-charges.

| arm | first boot: high-water (parked, pool peak) | after the respawn | admission charge after the respawn |
|---|---|---|---|
| main 96390ef0e4 (`m96`) | 34 MB (34, 2) | 2 MB (1, 2), twice | +2 MB |
| PR head e6485a9792 (`f1353`) | 34 MB (34, 2) | 2 MB (1, 2), twice | +2 MB |
| PR head with `[diag1351]` lines (`d1353`) | 34 MB (34, 2) | 2 MB (1, 2), twice | +2 MB |

## What the bracket reads (`d1353`, MiB)

`tools/diag1351.patch` prints the readings at the measurement bracket's entry and at its end (the end only when a capture landed): raw `cuMemGetInfo` free, the delta reading (`delta_free_bytes`), the default pool's reserved and used bytes, the graph pool's reserved bytes, and `device_graph_mem_cached` (graph reserved less graph used).

| bracket | raw | delta | pool reserved / used | graph reserved | graph cached |
|---|---|---|---|---|---|
| first boot, entry | 4,387 | 83,718 | 34,144 / 18,147 | 0 | 0 |
| first boot, end | 4,355 | 83,686 | 34,144 / 18,149 | 32 | 0 |
| after the respawn, entry | 4,088 | 83,420 | 34,144 / 18,147 | 32 | 0 |
| after the respawn, end | 4,088 | 83,420 | 34,144 / 18,149 | 32 | 0 |

- At the first boot the capture grows the graph pool from 0 to 32 MiB, and the free readings drop by the same 32 MiB. That is the 32 MB of the 34 MB parked delta that the default pool's 2 MB peak does not see.
- The respawned worker finds the graph pool at 32 MiB already, held from the dead worker in the same process and context. Its capture takes nothing new: the free readings and the graph pool stay put, and the parked delta reads 1 MB.
- The PR counts the graph pool's reserved-but-unused bytes (`RESERVED_MEM_CURRENT - USED_MEM_CURRENT`), and the driver reports all 32 MiB as used at every reading, before the respawn and after it. So the new term is 0 at both ends, and the measurement is main's.
- Within one worker, later captures (the further entry/end pairs in `server.log`) take nothing new either: the graph pool stays at 32 MiB. The 34 MB per-session charge rests on the first capture alone.

## The graph pool's reserved and used bytes at every capture (review round 1, `dm96`, `df1353`)

`tools/diag1351b.py` writes main's and the PR head's `spec.rs` with one `[diag1351b]` line at every bracket entry and at every landed capture's end: the graph pool's `RESERVED_MEM_CURRENT` and `USED_MEM_CURRENT` (MiB), beside the free readings and the default pool. Both arms read the same, capture by capture:

| capture | graph reserved / used, entry | graph reserved / used, end | high-water line |
|---|---|---|---|
| first boot, the first capture (the boot probe) | 0 / 0 | 32 / 32 | 34 MB (parked 34) |
| first boot, every later capture (six in each run) | 32 / 32 | 32 / 32 | none (below the high-water) |
| the respawned worker, its first capture | 32 / 32 | 32 / 32 | 2 MB (parked 1) |
| the respawned worker, every later capture | 32 / 32 | 32 / 32 | 2 MB once more |

- (a) The respawned worker's capture does not raise graph USED: the dead worker's 32 MiB are still counted as used when the new worker starts, and its capture takes nothing new from the graph pool or the driver.
- (b) A second capture on the first boot leaves graph USED flat, and so does every later one: the 32 MiB are taken once per process (per stream: draft captures upload on the engine's one stream, and graphs uploaded on the same stream share their physical memory), not once per session. The first boot's 34 MB per-session charge counts that one-time 32 MiB for every speculative session. The respawned worker's 2 MB is the per-session part.

## What this settles and what it does not

- Settled: the PR's term reads 0 (the reserve is counted as used), so the PR does not change the measurement. The graph-pool part of the draft state is one-time per stream, so the first boot over-charges every speculative session by about 32 MB. The respawn reads the per-session part, so it does not under-charge, against what #1351 first assumed.
- Follow-up: charge the graph pool's growth once (at the first capture of a stream), not per session.
- Not settled here: whether many concurrent speculative sessions keep sharing the 32 MiB. The captures above ran one at a time.

Conditions: 2026-11-18, one GB10 unit, no page-cache fill (MemAvailable about 118 GB), the five runs back to back (the review round's two at 15:43 to 16:01); nothing else ran on the unit. Raw: `gb10/wd-{m96,f1353,d1353,dm96,df1353}/` (`summary.txt`, the timestamped `server.log`, systemd's `journal.txt`). Harness: `tools/gb10-build.sh`, `tools/gb10-wd.sh ARM` (as in `research/respawn-watchdog-20261118/` on PR #1350's branch), `tools/diag1351.patch`, `tools/diag1351b.py`.
