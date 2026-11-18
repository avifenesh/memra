# respawn-watchdog-20261118: the systemd watchdog through a worker respawn's reload on GB10 (memra#1331)

Verdict: fixed on the cell the issue names (PR #1350, fold ee374c925e). On GB10 (unit-a, sm_121a),
memra#1307's MTP serving (Qwen3.8-27B NVFP4 + Q5_K head, masked regime MTP draft, 262,144 context,
32 sessions) runs as a systemd user service with the shipped unit's supervision (`Type=notify`,
`WatchdogSec=180`, `Restart=on-failure`, `TimeoutStartSec=600`) and `MEMRA_PANIC_AFTER=1`: the first
completed request panics the worker once and the in-process respawn reloads, in about 125 s.

With the panic about 70 s after a ping the worker answered live (the issue's worst phase: the
reload spans the next two ping slots), main withholds both pings and systemd kills the unit at the
deadline, 110 s into the reload; the fold pings through the reload and the respawn serves.

| arm | panic timing | pings withheld in the reload | systemd | NRestarts | main PID | serving again |
|---|---|---|---|---|---|---|
| main 5a6ae2438e | live slot + 70 s | 2 (at +20 s, +110 s) | `Watchdog timeout (limit 3min)!`, SIGABRT at +110 s, restart | 1 | new | 254 s after the panic (a second full boot) |
| the fold ee374c925e | live slot + 69 s | 0 | nothing | 0 | unchanged | 125 s after the panic (the respawn) |
| main 5a6ae2438e | untimed | 1 | nothing | 0 | unchanged | 125 s |
| the fold ee374c925e | untimed | 0 | nothing | 0 | unchanged | 125 s |

- Untimed, main survives: its panic fell so that one slot came inside the reload and the next
  after readiness. Main fails when the panic lands more than about 55 s after the last live ping
  (a 125 s reload then covers two slots), the fold in neither case.
- In each run the one `withholding` line before the first `listening` line is the first boot's
  load, before `READY=1`; systemd arms the watchdog only after it.
- The restarted main process panicked again on the run's second request (`MEMRA_PANIC_AFTER` is
  once per process) and was in its own respawn when the run stopped the unit
  (`TimeoutStopSec=60` then killed it); nothing after the restart is read as a result.

The fold's reload steps (seconds after the panic, timed run; the untimed run within 1.5 s):

| step | from | to | longest stretch with no load stamp |
|---|---|---|---|
| respawn backoff | 0 | 2.0 | |
| engine and model load | 2.0 | 9.1 | |
| KV planes and the decode-wave probe | 9.1 | 54.4 | 45 s |
| vision tower | 54.4 | 54.7 | |
| prefix cache, admission sizing, then the cost calibration's primes (odometer) | 54.7 | 120.6 | about 50 s before the first prime |
| boot probe, ready | 120.6 | 125.8 | |

Every stretch is under the 120 s stall bound (`MEMRA_HEALTH_STALL_S`), so the pings never stop.
A single step past that bound would still withhold them, as on main.

Side observation, main and fold alike, not changed here: the respawned worker measures the
per-session draft state at 2 MB (parked delta 1 MB, pool peak 2 MB) where the first boot measured
34 MB, and admission charges +2 MB per speculative session after a respawn.

Conditions: 2026-11-18, one GB10 unit, no page-cache fill (MemAvailable about 118 GB), the four
runs back to back; nothing else ran on the unit. Raw: `gb10/wd-{main,f1350}{,-t}/` (`summary.txt`,
the timestamped `server.log`, systemd's `journal.txt`, `steps.txt`, the two answers). Harness:
`tools/gb10-build.sh`, `tools/gb10-wd.sh ARM` (`TIMED=1` sends the panicking request 68 s after a
live ping slot: the pinger's armed line plus multiples of 90 s).
