# respawn-graph-trim-20261118: a worker respawn trims the device graph pools before its reload, on GB10 (memra#1361)

Verdict: the trim gives the dead worker's graph memory back, and the respawned worker measures its draft state as a first boot does (PR #1363; run as 54e7a91894, before the head printed the trim line's figures plainly and moved the loop behind injected device operations).

This is memra#1307's MTP serving on GB10 (unit-a, sm_121a): Qwen3.8-27B NVFP4 + Q5_K head, masked regime MTP draft, 262,144 context, 32 sessions. It ran under systemd (`Type=notify`, `WatchdogSec=180`, `Restart=on-failure`) with `MEMRA_PANIC_AFTER=1`, one worker respawn.

| | main 96390ef0e4 (`research/respawn-drafthw-20261118/`, PR #1353's branch) | this change |
|---|---|---|
| the respawn's trim line | none | `respawn graph-pool trim dev0: reserved Some(32)MiB used Some(32)MiB -> reserved Some(0)MiB used Some(0)MiB (CUDA_SUCCESS)`, as 54e7a91894 printed it (the head prints `reserved 32MiB used 32MiB -> reserved 0MiB used 0MiB`) |
| the respawned worker's draft-state high-water | 2 MB (parked 1, pool peak 2) | 34 MB (parked 34, pool peak 2), as the first boot |
| the respawned worker's admission charge | +2 MB | +36 MB, as the first boot |
| NRestarts, serving again after the panic | 0, 125 s | 0, 125 s |

- Both readings are taken after the synchronize. Before the trim the dead worker's 32 MiB read as reserved and used (32 / 32), and the trim itself took them to 0 / 0: it releases what no live allocation and no running or queued graph holds, and the after reading is what those still hold. The respawned worker's first capture then parks 34 MB, as a first boot's does: it takes its graph memory from the driver again. This run did not print the graph pool per capture; the first boot's 0 -> 32 MiB is in PR #1353's record.
- The charge is the first boot's, 36 MB per speculative session. #1360 makes that 2 MB per session and the graph reserve once per stream; it is a separate change.

Conditions: 2026-11-18, one GB10 unit, no page-cache fill (MemAvailable about 118 GB), one run; nothing else ran on the unit. Raw: `gb10/wd-f1361/` (`summary.txt`, the timestamped `server.log`, systemd's `journal.txt`). Harness: `tools/gb10-build.sh`, `tools/gb10-wd.sh ARM`.
