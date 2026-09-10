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

## Cell 1c: cap, budget and scheduling are three different causes

The first pass left one question open: the door removes slot refusal outright, yet the 408
rate does not collapse. Cell 1c separates the three candidate causes so the verdict cannot be
read ambiguously. Every arm runs the same nine requests arriving together, one 899,275-token
cold prompt and eight 4k prompts, and **every short carries a distinct prompt length**
(4096 + 8i). The server names an abandoned request only by prompt length, so distinct lengths
are what make the abort lines join back to exactly one client row.

The measurement arm does not hand-type its environment. `attrib-door` boots through the
production launcher; its exec environment is read verbatim from `/proc/<pid>/environ`
(126 vars) and replayed with one named override, printed into the boot log as
`MEMRA_SSE_PREFILL_COMMIT_MS: 10000 -> 60000`. The launcher is untouched.

### The three arms

| arm | binary | cap | pre-header commit | served | the long prompt's fate |
| --- | --- | --- | --- | --- | --- |
| control | `darkserve-v0137` | 4 | 10 s | 4 / 9 | refused a slot (queued, never admitted) |
| door | `darkserve-pr431` | 32 | 10 s | 1 / 9 | admitted, then starved (0.00 s of work) |
| door + budget | `darkserve-pr431` | 32 | **60 s** | **9 / 9** | served (t_hdr 60.303 s, ttft 273.861 s) |

Per-request attribution, door arm at the production 10 s commit, every failure classified and
no abort line unmatched:

| tag | prompt tok | outcome | t_hdr | server work at abort | class |
| --- | --- | --- | --- | --- | --- |
| 899k | 899,275 | 408 | 10.062 s | **0.00 s** | admitted, starved |
| uniq-1 | 4096 | 408 | 10.001 s | 9.62 s | admitted, computing |
| uniq-2 | 4104 | 408 | 10.003 s | 9.68 s | admitted, computing |
| uniq-3 | 4112 | 408 | 10.003 s | 9.63 s | admitted, computing |
| uniq-4 | 4120 | 408 | 10.001 s | 9.60 s | admitted, computing |
| uniq-5 | 4128 | 408 | 10.002 s | 9.69 s | admitted, computing |
| uniq-6 | 4136 | 200 | 7.949 s | - | served |
| uniq-7 | 4144 | 408 | 10.001 s | 9.66 s | admitted, computing |
| uniq-8 | 4152 | 408 | 10.003 s | 9.65 s | admitted, computing |

`verdict=reject-slot` 0, "client disconnected while queued" 0, `admission_session_defers` 0,
VRAM defers 0, OOM parks 0. Control arm at the same shape: `reject-slot` 5,
queued-never-admitted 5, `admission_session_defers` 528, served rows at 3.975 - 4.533 s.

8-turn cache-on twin, all three arms 8/8 200, cold turn 8.670 - 9.139 s, turns 2-8 in
0.214 - 0.339 s, `cached_tokens` 0 -> 32,886. `budget-door` is the only arm that demoted to
the host tier at all (1 demotion, 0 VRAM defers, 0 OOM parks), because it is the only arm in
which all nine sessions actually ran to completion.

### The three-way reading

1. **The door clears slot refusal.** `reject-slot` 5 -> 0 and queued-never-admitted 5 -> 0.
   That is what #431 claims and it is what it delivers.
2. **The budget is what actually serves the shape.** Same binary, same cap, only
   `MEMRA_SSE_PREFILL_COMMIT_MS` 10000 -> 60000, and the identical nine requests go from
   1/9 to 9/9. None of the door arm's 408s was unserved capacity: seven of the eight shorts
   had already burned 9.6 s of real prefill when the client hung up at 10 s.
3. **Neither fixes the starved long prime.** The 899k was admitted, held a session, and had
   `0 generated` with `billed to abort point, 0.00s` while eight 4k prefills ran to
   completion. A larger cap makes that row worse, not better, and the raised budget only
   hides it in this particular shape by giving the shorts time to drain first. Filed as
   **#442**, and it is the failing case `MEMRA_PRIME_YIELD` (#389) has to be measured
   against.

### The knee: there is no concurrency knee, there is a temporal one

Ladder `N = 1, 2, 4, 8, 12, 16` of 4k shorts fired against one 899k cold prime, N=1 first as
the floor row. Client pre-header budget 900 s on both the long request and the shorts, echoed
into the arm log as `CLIENT BUDGETS ... 900s`, so every 408 below is the **server's**, issued
at its own 60 s commit, and no row is a client-side truncation. Server pre-header commit
60 s, cap 32, `darkserve-pr431`.

| wave N | concurrency with long | long still priming | served | t_hdr median | t_hdr max | wave start (s into phase) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | yes | **0 / 1** | - | - (408 at 60.032 s) | 12.11 |
| 2 | 3 | yes | **0 / 2** | - | - (408 at 60.059 s) | 76.15 |
| 4 | 5 | yes | **0 / 4** | - | - (408 at 60.119 s) | 140.21 |
| 8 | 9 | ends mid-wave | 8 / 8 | 41.831 s | 41.883 s | 204.33 |
| 12 | 13 | no | 12 / 12 | 14.777 s | 15.046 s | 280.87 |
| 16 | 17 | no | 16 / 16 | 18.202 s | 18.303 s | 339.34 |

The long request itself: 200, t_hdr 60.128 s, **ttft 264.054 s**, total 277.740 s, 899,275
prompt tokens, 297 completion tokens, 149 token events.

The ladder does not cross at a concurrency. It crosses at a *time*, the long prime's ttft:

- While the 899k is priming, **zero** shorts are served at N=1, N=2 or N=4. One single 4k
  request beside one long prime is already a total failure. Server counters for the arm:
  `reject-slot` 0, admitted-then-abandoned 0, **queued-never-admitted 7** - the seven shorts
  of waves 1, 2 and 4. They never got a slot at all while the long prime held the machine.
- Wave 8 straddles the end of the prime (starts at 204 s, prime ends at 264 s) and is served
  at 41.8 s, which is the tail of the prime, not a property of N=8.
- Once the prime is done, the same pair serves 12 concurrent shorts at 14.8 s median and 16
  at 18.2 s median, every one a 200. Seventeen concurrent sessions are comfortable.

`prime-rendezvous` lines 0 and `prime-chunk` yields 0 in this arm, which is expected:
`MEMRA_PRIME_YIELD` is OFF here. That is cell 4's variable.

#### The live instrument, and one caveat about it

`long_token_events` stayed at **0** for the first 266 s, across three waves of 408s, and then
went 0 -> 24 -> 68 -> 149 within ten seconds. The stall was visible as it happened, which is
what the sampler is for.

The `/metrics` counters are a *lagging* instrument here and must not be quoted as live
evidence: `admitted`, `active_sessions` and `prompt_tokens_in` all read 0 for the entire
266 s and then published together (0 -> 9 admitted, 0 -> 9 active sessions, 0 -> 932,267
prompt tokens in) at the instant the prime completed. They publish at prefill completion, not
at admission. The unambiguous live signal is the token arrival log.

#### The numbers this supports

| knob | value the data supports | receipt |
| --- | --- | --- |
| `MEMRA_MAX_SESSIONS` | 32 is **not** refuted by concurrency: 17 concurrent sessions serve 16/16 at 18.303 s worst-case time-to-headers, zero 408, zero reject-slot | knee waves 12 and 16 |
| `MEMRA_SSE_PREFILL_COMMIT_MS` | **60000**. 10000 is refuted outright: worst-case time-to-headers is 15.046 s at 12 concurrent shorts and 18.303 s at 16, on an otherwise idle pair with no long prime at all | knee waves 12 and 16, and budget-door 9/9 |
| the pair | **must not ship cap 32 with a 10 s budget** | every arm above |

The cap is not the lever for the long-prompt case and raising it does not help there: at cap
32 with a 60 s budget, a single short arriving during a 899k prime still gets nothing. That
failure is #442, and #389 is the lever aimed at it.

## Verdict

The door is a **winner at the shape it claims** (4/9 -> 9/9 served, +125%) and it
introduces no new failure of its own: every arm is clean of OOM parks, VRAM defers
and demotions. Cell 1c then separates what the door does from what it does not do, and
the three causes have three separate receipts:

| cause | evidence | fixed by the door? |
| --- | --- | --- |
| slot refusal | `reject-slot` 5 -> 0, queued-never-admitted 5 -> 0 | **yes**, this is #431 |
| the pre-header budget | same binary, same cap, 10 s -> 60 s takes the identical nine requests from 1/9 to 9/9 | no, it is a launcher constant |
| a long prime monopolising prefill | one 899k prime in flight serves **0 of 1** short at N=1, and 0 at N=2 and N=4, until the prime's ttft at 264.054 s | **no**, filed as #442 |

The knee arm settles the cap question directly. `MEMRA_MAX_SESSIONS=32` is not refuted
by concurrency: with no long prime in flight this pair serves 16 concurrent shorts, 17
sessions in all, at 18.303 s worst-case time-to-headers with zero 408 and zero
reject-slot. What is refuted is the pairing of that cap with a 10 s pre-header commit,
because worst-case time-to-headers is already 15.046 s at 12 concurrent shorts on an
otherwise idle pair. **Cap 32 with a 10 s budget must not ship.** The budget the data
supports is 60000 ms.

The engine-side recommendation is unchanged in direction and sharper in detail: keep the
door, and land the missing bound on *concurrent cold prefills* in the engine rather than
in a launcher constant. No cap value makes a long prime share the machine; that is #442,
and `MEMRA_PRIME_YIELD` (#389) is the lever aimed at it.

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
