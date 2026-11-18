# respawn-graph-trim-20261118: a worker respawn trims the device graph pools before its reload, on GB10 (memra#1361)

Verdict: the trim gives the dead worker's graph memory back, and the respawned worker measures its draft state as a first boot does (PR #1361's head ec60547b7a, run as 54e7a91894, which differs only in how the trim line prints its figures).

This is memra#1307's MTP serving on GB10 (unit-a, sm_121a): Qwen3.8-27B NVFP4 + Q5_K head, masked regime MTP draft, 262,144 context, 32 sessions. It ran under systemd (`Type=notify`, `WatchdogSec=180`, `Restart=on-failure`) with `MEMRA_PANIC_AFTER=1`, one worker respawn.

| | main 96390ef0e4 (`research/respawn-drafthw-20261118/`, PR #1353's branch) | this change |
|---|---|---|
| the respawn's trim line | none | `respawn graph-pool trim dev0: reserved 32 used 32 -> reserved 0 used 0 (CUDA_SUCCESS)` |
| the respawned worker's draft-state high-water | 2 MB (parked 1, pool peak 2) | 34 MB (parked 34, pool peak 2), as the first boot |
| the respawned worker's admission charge | +2 MB | +36 MB, as the first boot |
| NRestarts, serving again after the panic | 0, 125 s | 0, 125 s |

- The dead worker's 32 MiB were still counted as used before the trim; after the synchronize they were released, and the trim returned them. The respawned worker's first capture then parks 34 MB, as a first boot's does: it takes its graph memory from the driver again. This run did not print the graph pool per capture; the first boot's 0 -> 32 MiB is in PR #1353's record.
- The charge is the first boot's, 36 MB per speculative session. #1360 makes that 2 MB per session and the graph reserve once per stream; it is a separate change.

Conditions: 2026-11-18, one GB10 unit, no page-cache fill (MemAvailable about 118 GB), one run; nothing else ran on the unit. Raw: `gb10/wd-f1361/` (`summary.txt`, the timestamped `server.log`, systemd's `journal.txt`). Harness: `tools/gb10-build.sh`, `tools/gb10-wd.sh ARM`.
