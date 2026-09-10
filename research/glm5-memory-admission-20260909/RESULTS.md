# Lane results: glm5-memory-admission-20260909 (`MEMRA_ADMIT_BY_MEMORY`, PR #431)

Serving cell for the door, run 2026-09-10 on vast 50431646 (2x B200 SXM, CUDA 13.0),
a non-production dev pair. Fleet consumer side and the launcher gate live in darklanes
PR #565; the lane write-up with every row is
`research/glm5-dev-pair-20260910/LANE.md` in darklanes (private).

Arms differ in exactly one thing, the binary, and therefore in the launcher's
capability gate:

| arm | binary | launcher |
| --- | --- | --- |
| control | `darkserve-v0137` | `memory-shaped admission unset; binary lacks MEMRA_ADMIT_BY_MEMORY; max_sessions=4 (the worst-case slot proxy)` |
| door | `darkserve-pr431` (this branch) | `memory-shaped admission armed; ... max_sessions=32 open_output_tokens=32768 (derived from models.toml) defer_budget_ms=8000` |

Everything else is the production 1M posture: TP-2 plain (`MEMRA_SERVE_SPEC=0`),
`MEMRA_KV_HOST_MB=294912` at tenant pct 100, `MEMRA_PREFIX_CACHE_MB=49152`,
`MEMRA_REUSE_POOL_GLOBAL_CAP=1`, `MEMRA_HYPER_SUFFIX_PRIME=1`, `MEMRA_CTX=1048576`,
served prompt cap 917504. All requests were sent at vendor-default sampling: no
sampling parameters on the wire. The long prompt measures 899,275 prompt tokens by
the server's own usage row.

Each arm's binary identity was asserted from `readlink /proc/<pid>/exe` before any
row was taken, after a first attempt silently measured the door arm against the
control binary (a 200 on `/v1/models` proves some server answers, never which one).

## The shape the door claims: one 899k cold prompt, then 8x 4k while it decodes

| row | control (cap 4) | door (cap 32) |
| --- | --- | --- |
| 899k cold | 200, t_hdr 10.105 s, ttft 246.652 s, total 281.634 s | 200, t_hdr 10.104 s, ttft 236.334 s, total 239.559 s |
| 8x 4k wave | 3x 200 (3.414 / 3.417 / 3.495 s), 5x pre-header 408 | 8x 200 (7.811 - 8.789 s) |
| served | 4 / 9 | **9 / 9** |
| `verdict=reject-slot` | 5 | 0 |
| `admitted` | 4 | 9 |
| `admission_session_defers` | 595 | 0 |
| `admission_vram_defers` | 0 | 0 |
| `step_oom_parks` | 0 | 0 |
| host-tier demotions | 0 | 0 |

The door admits nine concurrent sessions holding 932,043 prompt tokens against live
free VRAM with no VRAM defer, no OOM park and no reject-slot. The proxy refuses five
of the eight shorts because the single 899k session consumes one of only four slots.

## Simultaneous arrival, 9-way and 21-way

| row | control 9-way | door 9-way | control 21-way | door 21-way |
| --- | --- | --- | --- | --- |
| long prompt | 408 `deadline_exceeded` | 408 `deadline_exceeded` | 429 `shed_deadline` | 408 `deadline_exceeded` |
| shorts served | 4 / 8 | 1 / 8 | 4 / 20 | 0 / 20 |
| `verdict=reject-slot` | 5 | 0 | 16 | 0 |
| `prompt_tokens_in` | 16,384 | 932,043 | 279,198 | 262,814 |
| `admission_session_defers` | 464 | 0 | 1,995 | 0 |
| `step_oom_parks` | 0 | 0 | 0 | 0 |

Under simultaneous arrival the door admits everything, prefill is time-sliced across
every admitted request, and nobody commits a header inside the 10 s pre-header budget.
The slot proxy's queue was incidentally protecting header latency. The engine's own
accounting is not what fails here: zero VRAM defers, zero OOM parks and zero host
demotions in every arm, including the 21-way burst.

## 8-turn cache-on twin

Unchanged by the door. Both arms 8/8 200; cold turn 1 t_hdr 8.63 s (control) and
8.67 s (door); turns 2-8 in 0.218-0.324 s on both; `cached_tokens` walks
32,736 -> 32,886 identically.

## Verdict

The door is a **winner at the shape it claims** (4/9 -> 9/9 served, +125%) and it
introduces no new failure of its own: every arm is clean of OOM parks, VRAM defers
and demotions. What the cell also shows is that `MEMRA_MAX_SESSIONS=32`, the launcher
number that pairs with the door in darklanes #565, has no bound on *concurrent cold
prefills*, so a burst degrades every request's time to header instead of queueing.
The engine-side recommendation is to keep the door and to land the missing prefill
concurrency bound before the fleet raises the slot cap; that bound belongs in the
engine, not in a launcher constant.

`decide-by 2026-09-23` still stands; this cell is the positive receipt for the door
itself, not for the cap raise.

## What this cell does not show

- The reclaim ladder and host-tier demotion path were never exercised: booked real
  bytes peaked near 86 GB against roughly 180 GB of pair VRAM. Zero demotions in
  every arm is an absence of pressure, not evidence the path works.
- No greedy rows and no perf claim: these are served-shape sampled receipts only.

## Gates

No gates were run on the rig (owner rule). `MEMRA_SKIP_PERF_CI=1` on push. Engine
tests, clippy, fmt and the card cell for this branch were run on the lane box and
are recorded in the commit message of `serve: admit by memory instead of a
worst-case slot proxy (door OFF)`.
