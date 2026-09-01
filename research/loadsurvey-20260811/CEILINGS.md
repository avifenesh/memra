# Axis-B load ceilings: admission and backpressure survey

Date: 2026-08-11

Baseline: `35b285c9124f0899bcfeb4f6d010cb6ad75e3404`

Status: code-and-receipt survey; **no GPU work or new measurement was performed**

## Verdict

**No-go on spending a rig on a new load lane now.** The next limit is already identifiable, but it
is shape-specific rather than one missing `c=128` number:

- On the 24 GB q9 stress shape, the present c=64 gate is a **completion/liveness floor**, not an
  active-capacity result. The recorded run reached 59 active sessions and then VRAM-deferred a
  queue of four; all 64 eventually completed. The next offered clients deepen the wait before they
  discover a new failure mode. `research/ctxcharge-20260809/raw/20260809T161809Z-gates/serve-stress-server.log:149-169`
  `research/ctxcharge-20260809/raw/20260809T161809Z-gates/serve-stress-gate.log:1-3`
- Independently of memory, batched interactive serving defaults to 64 active sessions, and later
  requests wait rather than reject. Therefore a 65th simultaneously live request is the first
  **certain count-wait** if the VRAM gate has not already queued it.
  `crates/memra-server/src/worker.rs:33-35`
  `crates/memra-server/src/worker.rs:3173-3195`
- On the current 2x RTX PRO 6000 Workstation target, short decode is already near a useful-throughput
  knee by c=8: the N=5 aggregate rises only 9.6% from c=4 (158.45 tok/s) to c=8
  (173.62 tok/s), while arithmetic per-stream share falls from 39.6 to 21.7 tok/s. The likely next
  short-request ceiling is therefore compute/QoS before either 64 active sessions or VRAM; c>8 on
  this exact target remains **needs-measurement**. `research/newboxgates-20260811/RESULTS.md:76-94`
- On that target at 262,144 context, the ceiling is already measured: with the validated SWA ring,
  the 13th simultaneous session is the first to wait; all 24 offered requests drain with no failure
  or park. This is N=1 for that exact rig and shape, not a general scaling factor.
  `research/newboxgates-20260811/RESULTS.md:105-128`

The cheapest work is to use the already-validated ring and request-owned context sizing, then
collect real arrival/context distributions. A deeper rig lane becomes justified only when real
traffic—not a synthetic c=64 robustness canary—repeatedly reaches one of the waits above and violates
an explicit first-content or completion SLO. `research/ringval-20260810/RESULTS.md:1-11`
`crates/memra-server/src/worker.rs:4850-4877`

## Evidence vocabulary

- **Receipt** means a committed run result, with its stated N and thermal regime.
- **Deduction** means a consequence of current code plus a committed receipt; it is not promoted to
  a measured number.
- **Needs-measurement** marks the exact boundary that a future triggered lane would have to measure.

The stress gate itself defines wall time and its `ttfb` percentiles as informational, not asserted
performance gates. `tools/serve-stress-gate.sh:8-15`

## Admission state machine

### 1. Arrive and admit-wait

| State/transition | Current mechanism | What it actually bounds |
|---|---|---|
| **ARRIVE -> QUEUED** | A valid generate command is appended to one worker-owned `VecDeque`. `crates/memra-server/src/worker.rs:2963-2973` `crates/memra-server/src/worker.rs:4745-4754` | Pending work, not active GPU residency. |
| **QUEUED -> COUNT-WAIT** | Each scheduler tick scans queued requests. Batched interactive requests use `MEMRA_MAX_SESSIONS`, default 64; at the cap they are appended to the requeue and are not shed. `crates/memra-server/src/worker.rs:3117-3130` `crates/memra-server/src/worker.rs:3173-3195` | Active interactive session count. It is neither a queue-depth limit nor a memory guarantee. |
| **QUEUED -> SHED** | Judge/harvest work is capped and SLO-gated; overload returns an immediate retryable error rather than waiting in the engine. `crates/memra-server/src/worker.rs:3117-3125` `crates/memra-server/src/worker.rs:3182-3208` | Protects interactive service; this survey's wait analysis is the interactive branch. |
| **QUEUED -> SHAPED** | Explicit `max_ctx` is authoritative. Without it, an omitted `max_tokens` inherits `MEMRA_CTX`; a finite `max_tokens` instead sizes to `prompt + max_tokens + 8`. `crates/memra-server/src/worker.rs:4850-4877` | Context-linear cache allocation for this request. |
| **SHAPED -> COSTED** | The model charges exact flat-context bytes plus physically capped ring rows and a learned fixed high-water residual; admission uses `max(ctx_cap, prompt + budget + 64)`. `crates/memra-server/src/worker.rs:595-610` `crates/memra-server/src/worker.rs:622-698` | The next request's modeled resident cost, not total process capacity by itself. |
| **COSTED -> RECLAIM** | Except for the first active session, admission requires effective free memory (driver free plus reusable async-pool bytes) to cover request cost plus reserve. Before waiting, it evicts the global-oldest parked plain/spec session and re-reads effective free. `crates/memra-server/src/worker.rs:3293-3338` `crates/memra-server/src/worker.rs:3338-3373` | Reclaimable dormant state yields before live work waits. The first session instead reaches allocation and may return an honest allocation error. |
| **COSTED -> VRAM-WAIT** | If effective free remains below `cost + reserve`, the request is appended to the FIFO requeue; otherwise it is admitted. `crates/memra-server/src/worker.rs:3374-3440` | Next-session safety and transient headroom. It deliberately turns overload into queueing. |
| **ADMIT -> STEP** | The worker owns one CUDA thread; each loop runs speculative sessions, prefill, then batched decode chunks. `crates/memra-server/src/worker.rs:1-16` `crates/memra-server/src/worker.rs:3442-3454` | One execution owner and interleaved forward progress, not parallel CUDA owners. |

The reserve is asymmetric. A spec-capable process pays the fixed 1.5 GiB floor; a plain-only path
pays `min(request cost, 1.5 GiB)`. `MEMRA_ADMIT_RESERVE_MB` overrides that floor only for the
teeth/diagnostic path and is explicitly not a tuning knob. `crates/memra-server/src/worker.rs:848-862`
`crates/memra-server/src/worker.rs:1842-1876`

The interactive queue is FIFO in retry order, but heterogeneous VRAM service is not strict FIFO:
the loop scans the whole queue, requeues a request that does not fit, and can still admit a cheaper
request encountered later before replacing `queue` with `requeue`. This fit-skipping behavior avoids
hard head-of-line blocking, but a large request can be postponed by a continuing stream of smaller
requests. That starvation risk is a **deduction** from the loop, not a measured failure.
`crates/memra-server/src/worker.rs:3130-3138` `crates/memra-server/src/worker.rs:3374-3440`

The same loop also makes one pass over every waiting request each tick. Host admission work therefore
grows with pending queue depth; the magnitude at c>64 is **needs-measurement**.
`crates/memra-server/src/worker.rs:3127-3130` `crates/memra-server/src/worker.rs:3394-3440`

### 2. Step OOM -> park/requeue, or honest failure

| Branch | Current mechanism | Boundary |
|---|---|---|
| **Spec step OOM before emission -> PARK** | Only driver text containing `CUDA_ERROR_OUT_OF_MEMORY` or `out of memory` qualifies. The default retry budget is three. `crates/memra-server/src/worker.rs:1878-1899` | A bounded transient-collision backstop, not an inference from a dead run. |
| **PARK -> REQUEUE-FRONT** | The speculative session is dropped, its original request is rebuilt, and the retry counter/spec K are preserved. After retire frees memory, parked requests are front-inserted in original order. `crates/memra-server/src/worker.rs:3960-4006` `crates/memra-server/src/worker.rs:4985-5028` `crates/memra-server/src/worker.rs:4631-4635` | Preserves queue position but discards KV, so the retry re-primes and pays latency again. |
| **Spec OOM after emission or retries exhausted -> ERROR** | A streamed prefix cannot safely be replayed; those sessions report the error instead of parking. `crates/memra-server/src/worker.rs:3974-3985` `crates/memra-server/src/worker.rs:4007-4016` | Prevents duplicate output and infinite retries. |
| **Plain batched step error -> ERROR CHUNK** | A batched decode failure sends `batch step: {err}` to every affected row and retires them; this branch does not call park/requeue. `crates/memra-server/src/worker.rs:4282-4288` `crates/memra-server/src/worker.rs:4347-4365` | Park/requeue does **not** protect the ordinary batched path. |

This distinction is load-bearing. With the correct reserve, the current c=64 run completed 64/64
with no failure. Forcing the reserve to 16 MB completed only 46/64 and produced 16 batched-step plus
two speculative-step OOM receipts; the gate correctly inverted to `TEETH OK`.
`research/ctxcharge-20260809/RESULTS.md:90-100`
`research/ctxcharge-20260809/raw/20260809T161809Z-gates/serve-stress-teeth-gate.log:1-12`

Therefore park/requeue is not a lever for increasing admitted load. Lowering the reserve merely
moves overload from wait to post-admission errors, some of which are structurally unparkable.
`crates/memra-server/src/worker.rs:1842-1876` `crates/memra-server/src/worker.rs:3960-4016`
`crates/memra-server/src/worker.rs:4347-4365`

### 3. Spec allocation miss -> right-size ladder

| Transition | Current mechanism | Boundary |
|---|---|---|
| **SPEC POOL MISS -> EVICT-FIRST** | After observing that a parked spec session and a new full-size allocation do not fit, the model is remembered and later misses evict dead-weight parked spec state before allocating. `crates/memra-server/src/worker.rs:558-575` `crates/memra-server/src/worker.rs:5640-5685` | Avoids repeating a known failed allocation walk on a VRAM-tight spec rig. |
| **FULL ALLOC FAIL -> LADDER** | With the pool empty, the ladder starts at the learned landing or half the requested context, clamps at `need`, and halves toward it. A landing must make embeddings resident and clear a 1.5 GiB allocation probe; the landing is memoized. `crates/memra-server/src/worker.rs:5686-5738` | Finds a safe reduced speculative cache that still covers this request's output contract. |
| **LADDER FAIL -> TOKENWISE** | If even `need` cannot land, the speculative session is abandoned for the tokenwise path. `crates/memra-server/src/worker.rs:5735-5744` | Clean fallback for a spec-session allocation miss. |
| **PLAIN CACHE ALLOC FAIL -> ONE RECLAIM RETRY** | The plain path evicts prefix state and, for a quoted allocation OOM, one global-oldest parked session; it retries once and then returns an allocation error. `crates/memra-server/src/worker.rs:5770-5803` | Plain allocation has bounded reclaim, not the speculative right-size ladder. |

The right-size ladder does not participate in ordinary admission defers or batched step errors.
`crates/memra-server/src/worker.rs:3374-3397` `crates/memra-server/src/worker.rs:4347-4365`
`crates/memra-server/src/worker.rs:5640-5744` It also cannot repair the historical c=64 failure:
that run allocated all sessions successfully and then OOMed at step time.
`research/serving-density-20260806/VERDICT.md:90-98`

## What each control does—and does not—mean

| Control | Protects | Does not establish |
|---|---|---|
| `MEMRA_MAX_SESSIONS` | Active interactive count; requests over the cap wait FIFO. `crates/memra-server/src/worker.rs:3173-3195` | Safe VRAM use, useful throughput, a bounded queue, or tenant fairness. |
| Request-shaped admission | Next-session cache bytes plus learned residual and transient reserve. `crates/memra-server/src/worker.rs:3210-3235` `crates/memra-server/src/worker.rs:3293-3338` | That all admitted sessions can survive a deliberately undersized reserve. |
| Park/requeue | Up to three quoted speculative step OOMs before output. `crates/memra-server/src/worker.rs:1878-1899` `crates/memra-server/src/worker.rs:3960-4006` | Recovery for post-output or batched/plain OOMs. |
| F5 right-size | Spec cache allocation fit after a pool miss. `crates/memra-server/src/worker.rs:5640-5744` | More physical memory, admission fairness, or a higher plain PP-2 ceiling. |
| SWA ring | Context bytes for ring-eligible rows stop growing after their physical row cap. `crates/memra-server/src/worker.rs:595-605` | A universal model/rig scaling ratio. |

## Committed ceiling receipts

### 24 GB q9: c=64 is green because admission waits

The original default c=64 cell failed 0/64 three times with quoted step OOM while the server stayed
alive. Capping active spec sessions at 16, 32, or 48, or disabling spec at cap 64, made all 64
complete. `research/serving-density-20260806/VERDICT.md:63-98`

The repaired gate later sustained 59 active with queue depth four, completed 64/64, and recorded no
park or OOM in three effective-free runs. `research/admit-oom-20260806/RESULTS.jsonl:7-10`
The newest committed gate still completes 64/64; its explicit 8,192-token cap deliberately preserves
the old per-session pressure despite short finite generations. `research/ctxcharge-20260809/RESULTS.md:88-100`

The newest raw run shows why “c=64” is not “64 resident.” The first two arrivals took K=3 spec and
later arrivals took K=0 plain; the first defer occurred at 59 active with a 240 MB request cost and
1,611 MB reserve. `research/ctxcharge-20260809/raw/20260809T161809Z-gates/serve-stress-server.log:14-26`
`research/ctxcharge-20260809/raw/20260809T161809Z-gates/serve-stress-server.log:149-169`
The single-card spec policy admits at low load and stops new spec arrivals at high load, while
sampled sessions cannot be demoted; at most the low-water residual remains speculative.
`crates/memra-server/src/worker.rs:1532-1560` `crates/memra-server/src/worker.rs:3668-3707`

**Deduction:** for this exact 8k-pressure shape, the next offered load is a queue-depth/tail-latency
experiment, not a search for another active slot. The physically observed active boundary is about
59, and the count boundary is 64. Whether c=65, 96, or 128 remains acceptable is
**needs-measurement** because acceptance requires a real first-content SLO, not merely eventual
completion.

### The stress `ttfb` number is keepalive-censored

The client stamps `ttfb` on the first raw response line before checking whether it begins with
`data:`. `tools/serve-stress-gate.sh:107-134` The server emits SSE keepalive comments every five
seconds. `crates/memra-server/src/main.rs:3252-3256` Its own TTFT instrumentation deliberately ignores
keepalives and recognizes the first application data frame instead.
`crates/memra-server/src/main.rs:100-122` `crates/memra-server/src/main.rs:4091-4097`

Accordingly, the recorded c=64 `ttfb p95=5.00s` is not evidence of a 5-second first-token tail; it
can be the keepalive censor. The 24.9/28.0-second wall percentiles and 64/64 well-formed completion
remain valid for that N=1 gate. `research/ctxcharge-20260809/raw/20260809T161809Z-gates/serve-stress-gate.log:1-3`
Any future load lane must use first `data:`/content TTFT or the server's TTFT trace.

### 2x 96 GB target: two different next ceilings

The current target receipt is a PP-2 process on two 97,887 MiB RTX PRO 6000 Workstation cards, with
262,144 server context and speculative serving off. `research/newboxgates-20260811/RESULTS.md:25-40`
Placement policy also makes sharded cross-device PP-2 plain-only by default.
`crates/memra-server/src/worker.rs:1505-1529` `crates/memra-server/src/worker.rs:1547-1559`

For short decode, the current controlled ladder stops at c=8. Its N=5 totals are 98.30, 158.45,
and 173.62 tok/s at c=1/4/8, with 75/75 successful requests and no wrong-length responses.
`research/newboxgates-20260811/RESULTS.md:76-103` **Deduction:** aggregate compute is flattening before
the admission-count limit, so raising the cap cannot be assumed to buy useful capacity. The exact
c=16/32/64 first-content tails and throughput are **needs-measurement** only after demand triggers
the work.

For 262k sessions, ring-ON is already a measured capacity lever: first defer moves from two to 12,
modeled session cost falls from 21,894 to 6,123 MB, all 24 offered requests complete, and no failure
or park is captured. `research/ringval-20260810/RESULTS.md:113-146` The independent current-target
receipt reproduces 2 -> 12 and explicitly limits the result to N=1 on that box/configuration.
`research/newboxgates-20260811/RESULTS.md:105-128` Thus the 13th active 262k ring session, not c=64,
is the known next memory wait.

Actual prompt length can bind compute well before a nominal context cap binds VRAM. In the older
Box1 receipt, a sustained c=8 workload with actual 8,000-token prompts completed only 5.12 aggregate
output tok/s and had 178.661-second median wall time, despite zero admission defer or park. This is
N=1 historical shape evidence, not a current-target prediction. `research/capbase-20260809/RESULTS.md:41-58`

## Ranked cheapest levers

These are ranked by implementation/operational cost first, then by whether committed evidence says
they attack the actual boundary. Effort and priority are survey judgments, not measured results.

| Rank | Lever | Cost | Why now / why not | Required gate before promotion |
|---:|---|---|---|---|
| 1 | **Use `MEMRA_SWA_RING=1` for the validated Step-3.7 PP-2 / 262k serving profile.** | Config-only | It is already recommended as serving config while remaining default-off, and the exact receipt raises first defer 2 -> 12. `research/ringval-20260810/RESULTS.md:1-11` | Preserve the pinned model/PP-2 shape and existing correctness/golden gates; do not generalize its 6x integer result to other models. `research/ringval-20260810/RESULTS.md:113-146` |
| 2 | **Require honest request bounds:** finite `max_tokens`; use explicit `max_ctx` for intentionally unbounded/large requests; do not stamp every short request as 8k or 262k. | Caller/config | Finite output sets the allocation cap to `prompt + max_tokens + 8`, with admission retaining its `+64` speculative safety bound; omitted output inherits the server cap. `crates/memra-server/src/worker.rs:691-698` `crates/memra-server/src/worker.rs:4850-4877` Live charges span 152 MB at 8k, 2,536 MB at 128k, and 4,968 MB at 256k on the q9 spec path. `research/ctxcharge-20260809/RESULTS.md:47-65` | Validate request-schema compatibility, then inspect real prompt/output/cap distributions. No rig is needed to adopt bounded requests. |
| 3 | **Hold the 64-session cap; choose a lower QoS cap only from a real SLO. Do not raise it as a capacity fix.** | Config-only | The cap only changes active count and queueing. On the current target, c=4 already delivers 91.3% of c=8 aggregate decode, so more residency may mostly divide the same compute. `crates/memra-server/src/worker.rs:3173-3195` `research/newboxgates-20260811/RESULTS.md:84-94` | A triggered c=4/8/16 comparison using first-content p50/p95, completion tails, aggregate rate, and per-stream rate under the real request mix. |
| 4 | **Bound per-tick admission work and add tenant-aware fairness among interactive waiters.** | Small/medium code | One shared queue is fully scanned each tick; fit-skipping can postpone large requests and arrival FIFO alone lets one tenant occupy the waiting order. `crates/memra-server/src/worker.rs:2963-2966` `crates/memra-server/src/worker.rs:3127-3138` `crates/memra-server/src/worker.rs:3374-3440` | CPU scheduler tests for progress/order/cancellation first; only then a two-tenant rig burst measuring true first-content max wait and step p99. Keep the current c=64 and teeth gates. |
| 5 | **Make the transient reserve depend on actual/potential live spec work, not only process-wide spec capability.** | Medium, correctness-sensitive code | Admission currently passes process capability to `admission_reserve`, even when this request is predicted plain. `crates/memra-server/src/worker.rs:3217-3235` `crates/memra-server/src/worker.rs:3293-3315` This may recover headroom after residual sampled spec sessions retire on the single-card q9 path; it cannot help the target PP-2 path, where spec is already off. `crates/memra-server/src/worker.rs:1505-1529` | Mixed sampled-spec + many K=0 gate, zero post-admit OOM, no parks at normal reserve, teeth still red, and unchanged output. Treat 1.5 GiB as safety until that proof exists. |
| 6 | **Replace repeated F5 halving with a free-bytes/bytes-per-token landing estimate, retaining the allocation probe.** | Medium code | It can reduce failed allocation churn on long-context single-card spec pool misses; it does not move plain PP-2, count, admission-wait, or batched-step ceilings. `crates/memra-server/src/worker.rs:5640-5744` | Fault-injected ladder tests plus the existing spec-pool exactness and a triggered long-context allocation receipt. |

**Explicit no-go lever:** do not lower `MEMRA_ADMIT_RESERVE_MB`. It exists to make the teeth arm fail,
not to expose more capacity. `crates/memra-server/src/worker.rs:1842-1861` The current forced-16 MB
receipt admits too far and loses 18/64 requests. `research/ctxcharge-20260809/RESULTS.md:90-100`

## Is a dedicated load lane worth a rig?

### Decision: no, not yet

The committed serving track proves controlled concurrency through c=8 and a synthetic stability
soak at one request every 20 seconds; it does not establish organic c=64 demand.
`research/vast-trial-20260810/perf-receipt.md:1-14`
`research/vast-trial-20260810/perf-receipt.md:27-54`
The current target adds N=5 c=1/4/8 and a controlled c=24 262k capacity receipt, but those remain
constructed cells rather than an arrival-distribution receipt. `research/newboxgates-20260811/RESULTS.md:76-94`
`research/newboxgates-20260811/RESULTS.md:105-128`

Spending a rig now would answer an unrequested synthetic question while the known cheap levers and
known shape-specific waits remain sufficient. Keep passive production/trial receipts and open the
lane only when at least one of these demand triggers is present:

1. real traffic repeatedly produces session or VRAM defers and breaches an agreed first-content or
   completion SLO;
2. short-request active count approaches 64 on the target;
3. real 262k ring traffic repeatedly fills the measured 12-active envelope; or
4. a default-reserve run records a quoted step OOM/park, which would invalidate the current safety
   model.

When triggered, the minimum rig matrix is c=4/8/16/32/64 and only the larger rungs justified by
observed demand, split by real request shapes (finite short, actual long prompt, and the context caps
seen in traffic). Record first application-data/content TTFT, completion p50/p95/max, aggregate and
per-stream output rate, active count, queue depth, session/VRAM defers, parks, failures, N, and
thermal regime. Preserve the c=64 normal/teeth pair. This paragraph is a proposed future gate, not
a measurement performed by this survey.

## Bottom line

- **24 GB q9, explicit 8k:** active VRAM wait is already visible at 59; a 65th simultaneously live
  request certainly adds count wait at best. The unknown is acceptable backlog latency/fairness,
  not basic survival. `research/ctxcharge-20260809/raw/20260809T161809Z-gates/serve-stress-server.log:149-169`
  `crates/memra-server/src/worker.rs:3173-3195`
- **2x 96 GB Step, short:** useful compute/QoS appears to bend by c=8; exact deeper rungs need a
  demand-triggered measurement. `research/newboxgates-20260811/RESULTS.md:76-94`
- **2x 96 GB Step, 262k ring:** the 13th active request is the measured next VRAM wait; c=24 drains.
  `research/newboxgates-20260811/RESULTS.md:105-128`
- **Rig decision:** no-go until a real receipt crosses one of those envelopes. Use ring, honest
  request sizing, and the existing cap first. `research/vast-trial-20260810/perf-receipt.md:1-14`
