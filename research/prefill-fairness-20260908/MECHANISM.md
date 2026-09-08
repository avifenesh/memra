# Speculative prime fairness: mechanism and proposed design

2026-09-08. Code read at `72aa777c3763a21281fb9c1c55b2499f2c14d1ee`.
Step 2 checkpoint, before implementation. No build, test, or GPU experiment has run.
Source line references below refer to that commit.

## Prior result and measured trigger

Read `research/global-prefill-budget-20260825/RESULTS.md`: its plain Qwen
4,860-token mixed90 arm moved prefill after decode and imposed a global token
budget. Throughput at c4/8/12/16 fell 7.99/8.79/7.85/3.14%; hit ITL p95 rose
13.38/12.76/12.20/2.82%. Both runtime candidates were removed. Do not repeat
their ordering/budget or water-filled concat-prime arms. This proposal exposes
yield points inside speculative prime calls that currently monopolize the worker;
it does not claim mixed forward execution or a throughput gain.

The read-only `mixed-rps-20260908/REPORT.md` receipt, archived under
`/home/avifenesh/.local/share/relay/receipts/`, records one RTX 5090:
127,625-token Ornith cold prefill, 0.20 req/s, small p95 TTFT 16.2284 s,
long TTFT 19.4535 s, long total 23.3947 s, zero OOM retries. A separate
59,847-input long-decode job reached small p95 TTFT 70.2282 s and queue peak 10.
These are different interference mechanisms, not one measured prime duration.

## Answers

**(a) Speculative prime does not yield.** Worker phase (a) calls each speculative
session synchronously before interactive prefill (b) and plain decode (c):
`crates/memra-server/src/worker.rs:15302`, `:15375`, `:15542`, `:15926`.
For MTP, `step_session` drains the whole suffix at `:22658` and calls
`generate_spec_session_*_prime_split` at `:22789` / `:22801`. The engine primes
all segments before decoding (`crates/memra-engine/src/spec.rs:10434`, `:10526`).
The serial trunk's range loop calls `prime_chunk` until completion at
`crates/memra-engine/src/hybrid_forward.rs:5735`. `note_prime_rows` at `:5750`
updates health progress; it does not run another session. Thus a 128k prime at
chunk 1024 remains one worker hold. Plain serving is different: `prefill_tick`
consumes a bounded prefix and returns (`worker.rs:21577`, `:21732`); its
interactive budget can widen for a solo session (`:177`).

**(b) Both waits exist; arrival time matters.** Requests arriving during that
hold wait before worker admission, in the command channel. It is polled at
`worker.rs:13475`, and successful admission pushes a session at `:14867`.
Peers already in `active` wait for the long phase-(a) call to return. Session-cap
and VRAM refusals independently requeue at `:13770` and `:14627`.
The HTTP `[meter] admit` line is not proof of worker admission
(`worker.rs:476`; `crates/memra-server/src/lib.rs:8300`).

Archived `ornith-large-prefill/telemetry.jsonl` last successful metrics report
8 session-cap deferrals, 0 VRAM deferrals and 0 step-OOM parks; the corresponding
long-decode archive reports 4,042, 0 and 0. Deferrals are repeated checks, not
distinct requests. This rules out measured VRAM deferrals as the explanation for
these runs. It does not assign every small request's wait to one phase: that needs
arrival/admission/prime/first-token timestamps in the new cell. The one-session
Qwen candidate cannot admit a peer until its slot frees, regardless of prime yields.

**(c) Chunk wall times remain to measure.** No per-chunk 1024-versus-4096 wall
receipt was found for either target model in the requested corpus, the archived
mixed-load logs, or the capacity write-ups. Do not divide total TTFT by chunk
count and call it a bound. The Ornith 2026-09-04 repro records whole-request
128k wall 16.7 s at chunk 0 and 17.8 s at 4096 on another card, not individual
chunk costs (`darklanes/research/memra146-ornith-repro-20260904/RESULTS.md`).
Measure early/middle/late trunk chunks, draft fill/ingest and finalization separately
at 1024 and 4096 after GPU handoff. The mixed-load profile does not explicitly
set `MEMRA_PRIME_CHUNK`; do not silently describe its old binary as a 1024 arm.

**(d) DFlash has stateful boundaries, but no resumable public prime API.** After
tap streaming, `prime_dflash_taps` still loops every segment/range in one call
(`crates/memra-engine/src/dflash.rs:5004`, `:5031`). Each chunk runs target prime,
takes any boundary snapshot, then ingests taps through `TapBatchCarry`
(`:5052`, `:5070`). Carry capacity is min(prompt rows, 256); pending rows and
absolute positions cross chunk and capture boundaries (`:159` to `:203`). The
final partial batch is flushed only once (`:5084`). Cold construction calls the
whole helper at `:5171`; worker cold and resumed primes are synchronous at
`worker.rs:23435` and `:23405`. These locals can become owned resumable state;
flushing partial ingestion at each scheduler yield would change its GEMM shapes.

**(e) MTP needs two resumable phases.** Trunk prime retains all hidden rows,
logits and exact stable-boundary captures (`spec.rs:10431` to `:10544`). Afterwards
draft scratch fills from predecessor hidden rows in its own chunk loop
(`:11805` to `:11881`), after draft-context setup. Yielding only the trunk leaves
that second hold. Resume must preserve the previous turn's last hidden, absolute
base, draft-fill cursor, snapshots, pending token and Philox counters; calling
ordinary generation once per prompt chunk would emit/sample premature boundaries.

## Proposed smallest change

Add `MEMRA_PRIME_YIELD`, default OFF, decide-by 2026-09-22, with both arms,
rollback and this receipt namespace in `docs/FLAGS.md` in the implementation PR.
OFF drains the current numerical operations synchronously. ON advances an owned
prime state by one existing chunk, then returns to the worker. Use the same core
advance routine for both arms so the OFF oracle and ON walker execute the same
range list. Freeze segment boundaries, tail folding, GDN grid, request-absolute
`seq_end`, overlay offsets and kernel selection before the first chunk. Do not
replace the MTP inner loop with arbitrary repeated outer `prime_cache` calls.

Represent DFlash cold/resume preparation with cache, draft KV, range cursor,
tap carry plus its device buffer, final logits and capture state. Represent MTP
preparation with trunk cursor/hidden stack, captures and a later draft-fill cursor;
move setup/final boundary draws into explicit once-only phases. Preserve source
buffer lifetimes and complete stream dependencies before allowing shared scratch
to be reused. Cover the MTP restored-suffix prime currently inside admission
(`worker.rs:19397`) as well, so it cannot bypass the yield machinery.

At each completed chunk, drain commands and run existing admission, then service
each eligible peer at most once: one unchanged prime chunk, or one committed spec
round / plain decode step. A newly admitted one-chunk cold request needs its prime
AND first-token emission in that service turn. Rotate peer order to avoid a fixed
index bias. Resume the original long prime after this bounded round even if more
arrivals appeared. Keep the existing plain prefill batching policy; there is no
global token budget and no new cross-request concat-prime partition.

Let C be the largest long-prime chunk wall, Q the largest peer service quantum
(including its necessary once-only finalization), N the admitted-session limit,
and m the number of yielding trunk/draft chunks. Added long-prime TTFT is bounded
in work by `(m-1)*(N-1)*Q + scheduler overhead`, rather than an unlimited peer drain.
With one long prime and one admitted small request fitting one chunk, the target
is residual C plus the small request's idle prime/first-token work and scheduler
overhead. With additional peers add at most their service quanta. Setup, graph
capture and snapshot maxima must be measured too: the formula is not yet a wall
SLO. Neither session/memory admission waits nor arbitrary work in another process
on the shared GPU is bounded by this process-local scheduler. Co-location stays;
the implementation makes no dedicated-card assumption or capacity reduction.

## Approved exactness gate (owner clarification, 2026-09-08)

Concurrency-based route choice and plain batch composition are existing serving
policy, not a cross-timing sampled output contract. Keep the per-request numerical
program intact; do not require sampled bytes to match across different timing.
The owner approved these gates:

1. c=1 greedy four-turn chains, OFF versus ON, byte-identical for Qwen and Ornith.
2. c=2 greedy long-prime plus small-request pair, with an actual `prime-yield`
   count > 0. Raise `MEMRA_SPEC_GATE_LOW`/`HIGH` for this cell, record the values,
   prove both requests keep K>0, and compare each output with its own c=1 oracle.
3. Resumed versus unyielded prime boundary logits/captures: reuse the DFlash
   boundary oracle from `research/dflash-tap-storage-20260908` and the MTP run-spec
   cell. This tests state preservation independently of final greedy token matches.
4. Performance remains vendor-default sampled fixed arrivals, OFF/ON interleaved
   x3. Record per-chunk 1024/4096 wall, small p95 TTFT and long TTFT penalty.

CPU tests should exercise the production cursor/policy through a fake executor:
identical operation tapes OFF/ON; arrival immediately after chunk start; peer
prefill before first token; bounded round and long progress under continuous arrivals;
non-divisible tails; carry across capture boundaries; cancel/error ownership;
no early boundary sample; cold, restored and resumed turns. Compile, unit tests,
fmt and release clippy run only on the authorized remote CPU. GPU cells remain
blocked until the explicit handoff, then require an empty compute-app list and
the shared lock for every launch.
