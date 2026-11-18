# capture-oom-observation-20261118: a failed capture's draft-state record, and the landed capture on GB10 (memra#1319)

Verdict: the landed capture records as on main (PR #1347, head f0b696f18a). On GB10 (unit-a,
sm_121a), two boots per arm of memra#1307's MTP serving (Qwen3.8-27B NVFP4 + Q5_K head, masked
regime MTP draft, 262,144 context, 32 sessions): every boot prints one
`draft-session state high-water: 34MB (max of parked delta 34MB and capture-time pool peak 2MB)`
line, main and the fold alike, no capture `WARN` line and no capture out-of-memory line; the three
greedy answers of each boot are the same text in all four boots.

The failed-capture arm, which the PR changes, has no natural cell: no boot here or in memra#1307's
57 had a capture out-of-memory error. Its evidence is the CPU test and its five mutations
(PR body).

| arm | boot | ready after | high-water line | capture WARN | answers (sha256 of the text, first 16) |
|---|---|---|---|---|---|
| main 5a6ae2438e | 1 | 131 s | 34 MB = parked 34, pool peak 2 | 0 | 1966b0c38ee03a59 x3 |
| the fold f0b696f18a | 1 | 130 s | 34 MB = parked 34, pool peak 2 | 0 | 1966b0c38ee03a59 x3 |
| main 5a6ae2438e | 2 | 125 s | 34 MB = parked 34, pool peak 2 | 0 | 1966b0c38ee03a59 x3 |
| the fold f0b696f18a | 2 | 130 s | 34 MB = parked 34, pool peak 2 | 0 | 1966b0c38ee03a59 x3 |

- The pool peak of 2 MB against a parked state of 34 MB is the default pool's under-read of a
  capture that the review round found (its allocations land in the graph's own pool); the landed
  arm takes the larger of the two, as on main.
- Conditions: 2026-11-18, one GB10 unit, no page-cache fill (MemAvailable about 118 GB at each
  boot), boots alternating main and fold; nothing else ran on the unit.

Raw: `gb10/cap-{main,f1347}-b{1,2}/` (`summary.txt`, the timestamped `server.log`, `guard.log`,
the three answers). Harness: `tools/gb10-build.sh` (one source tree switched between `git archive`
copies of the two commits), `tools/gb10-cap.sh ARM BOOT` (memra#1307's
`unified-readers-20261114/tools/serve1307.sh` in `MODE=mtp`, three requests, then a clean stop).
