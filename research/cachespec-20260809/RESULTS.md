# Cache/spec accumulation receipt — Step-3.7-Flash PP-2

## Verdict

The owner-reported slowdown is reproduced on both the live RunPod and box1. The first
dominant term is **prefill/TTFT, not decode**: after request 2 the prefix cache keeps
crediting the same 6,148-token snapshot while the prompt grows, so every later turn
recomputes an ever-larger suffix. On box1, requests 2 through 11 grew from 1,223 to
7,770 uncached tokens and from 2.349 s to 13.745 s TTFT (5.85x), while decode moved
from 82.26 to 78.75 tok/s (-4.27%). TTFT versus uncached tokens is linear at
1.758 ms/token with R2=0.99971 (N=10 consecutive turns).

The concurrency-4 burst exposes a second, independent term: admission pacing. Two
full-context plain continuation entries remain parked, never hit this rewritten-history
workload, and leave only about 7.98 GB effective free while the admission gate requires
11.174 GB session cost plus 11.174 GB reserve. The four requests therefore start one at
a time at roughly 25 s intervals and accumulate 4,659 VRAM-defer decisions. The pool is
bounded, so this is not unbounded leakage; it is bounded but expensive dead residency.

`MEMRA_SERVE_SPEC=0` does not remove either slope. The default and spec-off arms have
identical aggregate cache/admission counters, and their sequential TTFT differs by a
median 9.7 ms (N=12, maximum 56.0 ms). Every default-arm request logged
`K=0 source=pp2-placement`; the deployed PP-2 policy is already choosing plain decode.
Therefore the proposed spec-on cache-accounting failure is **not confirmed** and the
control does not assert that an actually enabled speculative path is correct. It proves
that the observed deployed-policy slowdown is in the plain path.

An additional forced-spec diagnostic (`MEMRA_SPEC_GATE=0`) did exercise K=3. It found a
real but non-causal alignment defect: speculative sessions bypass the prefix cache as
designed, and all 16 later requests missed the spec pool because the exact prompt diff
diverged two tokens before the saved checkpoint (with 21-token divergences between burst
siblings). Explicit `session_id` nomination did not help because bytes correctly vetoed the
unsafe rewind. Forced-spec TTFT therefore still rose from 9.068 s at 6,150 prompt tokens to
19.860 s at 13,918, with zero cached tokens. This does not explain the deployed default
(which is K=0), but it shows that merely forcing spec or enabling a client header is not a fix.

All cells below are single runs, not medians. Box1 was the controlled paired A/B: both
GPUs stayed P0 and 30--52 C (650 one-second GPU rows per arm, two GPUs). The RunPod was
the live owner rig and ranged 38--87 C across 604 GPU rows; its result is treated as a
reproduction, not the paired comparison.

## Sequential request receipt

`dPCev`, `dReuseEv`, and `dDefer` are per-request metric deltas from the default arm.
The two cold learning requests insert the 6,150-token seed and 6,148-token LCP split.
Every request from index 2 onward reports a prefix hit, but the credited depth never
advances beyond 6,148.

| idx | prompt tok | cached tok | uncached tok | default TTFT s | default wall s | spec-off TTFT s | default decode tok/s | dPCev | dReuseEv | dDefer |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6,150 | 0 | 6,150 | 9.104 | 18.382 | 9.094 | 82.78 | 0 | 0 | 0 |
| 1 | 6,948 | 0 | 6,948 | 12.268 | 21.590 | 12.264 | 82.39 | 0 | 0 | 0 |
| 2 | 7,371 | 6,148 | 1,223 | 2.349 | 11.685 | 2.351 | 82.26 | 0 | 1 | 0 |
| 3 | 8,173 | 6,148 | 2,025 | 3.530 | 12.951 | 3.533 | 81.51 | 0 | 1 | 0 |
| 4 | 8,972 | 6,148 | 2,824 | 4.958 | 14.597 | 4.960 | 79.68 | 0 | 1 | 0 |
| 5 | 9,125 | 6,148 | 2,977 | 5.224 | 14.915 | 5.211 | 79.25 | 0 | 1 | 0 |
| 6 | 9,921 | 6,148 | 3,773 | 6.659 | 16.309 | 6.635 | 79.58 | 0 | 1 | 0 |
| 7 | 10,725 | 6,148 | 4,577 | 8.101 | 17.739 | 8.092 | 79.69 | 0 | 1 | 0 |
| 8 | 11,524 | 6,148 | 5,376 | 9.520 | 19.203 | 9.510 | 79.31 | 0 | 1 | 0 |
| 9 | 12,324 | 6,148 | 6,176 | 10.919 | 20.612 | 10.913 | 79.23 | 0 | 1 | 0 |
| 10 | 13,121 | 6,148 | 6,973 | 12.285 | 21.938 | 12.341 | 79.56 | 0 | 1 | 0 |
| 11 | 13,918 | 6,148 | 7,770 | 13.745 | 23.496 | 13.764 | 78.75 | 0 | 1 | 0 |

The live RunPod independently showed the same shape: request 2 was 2.178 s TTFT at
1,223 uncached tokens and request 11 was 12.650 s at 7,770 uncached tokens (5.81x),
while decode changed 86.85 to 82.40 tok/s (-5.13%). Its TTFT/uncached-token fit is
1.619 ms/token with R2=0.99958 (N=10 consecutive turns).

## Concurrency-4 receipt

All four clients began together. Thread scheduling changes which request index wins the
first FIFO slot, so the useful A/B is service rank rather than request number.

| service rank | default request / TTFT s | spec-off request / TTFT s |
|---:|---:|---:|
| 1 | 15 / 15.188 | 14 / 15.173 |
| 2 | 13 / 40.209 | 12 / 40.176 |
| 3 | 12 / 65.294 | 15 / 65.184 |
| 4 | 14 / 90.328 | 13 / 90.153 |

Both arms report exactly 4,659 `admission_vram_defers`, zero session defers, and zero
step-OOM parks for the burst. The gate's captured diagnostic is:

> `[admit-oom] VRAM defer: 1 active, effective free 7978MB (driver + 175MB pool-cached) < cost 11174MB + reserve 11174MB -- queueing (FIFO) [pool res 93382MB used 93207MB; parked spec sessions 0; plain reuse 2; queue 2]`

This is the same driver-free plus async-pool-cached gate used by the Gemma fix. Step does
not bypass that accounting. The defer is real: parked plain entries are live allocations,
not reclaimable CUDA-pool cache, and the current gate does not evict never-matching entries
before it decides to queue.

## Forced-spec diagnostic (additional control)

This was not substituted for the required default/off A/B. It used the identical frozen
workload, binary, model hashes, and box1 thermal block (single run, GPUs P0, 33--60 C across
884 one-second GPU rows).

| signal | forced K=3 result |
|---|---:|
| sequential TTFT, request 0 -> 11 | 9.068 -> 19.860 s |
| TTFT fit versus full prompt | 1.402 ms/token, R2=0.99985 (N=12) |
| cached tokens / prefix entries | 0 / 0 |
| spec hits / misses / affinity rewinds | 0 / 17 / 0 |
| spec evictions / final entries | 15 / 2 |
| c4 TTFT by service rank | 20.972, 52.447, 85.374, 117.531 s |
| VRAM-defer decisions / step-OOM parks | 145 / 0 |

The server recorded 16 explicit decline lines. The sequential form was consistently:

> `[worker] spec-affinity: declined (history diverged at 12322 of checkpoint 12324; 2 parked, 13918 prompt tokens; model step)`

The frozen prompt sequence itself independently has this two-token LCP seam (the plain
prefix cache learned 6,148 from the first 6,150-token prompt). Saving only the exact prompt
end is therefore the wrong rewind boundary for this client/template combination.

The forced arm also exposed two separate serving-hardening gaps which are not used to infer
the latency cause:

- 5/17 responses exceeded requested `max_tokens=768`: four reported 769 and one 770.
  In every case the default response was an exact text prefix and forced spec appended only
  the extra one or two tokens (for example `" legend"`). This is the session-mode overshoot
  contract in `generate_spec_inner2` leaking through the HTTP limit and needs its own
  exact-budget fix/gate.
- `/metrics.tokens_out` remained 0 despite 17 completed speculative responses. Spec usage in
  each response was present, but the process-wide output counter does not account spec bursts.

Raw forced evidence is in [`box1/20260808T225300Z`](raw/box1/20260808T225300Z/).

## Aggregate counter receipt

The default and spec-off box1 arms are byte-for-byte equal on every counter below.

| counter | default | spec off |
|---|---:|---:|
| completed / admitted | 17 / 17 | 17 / 17 |
| prompt / cached tokens | 191,834 / 92,220 | 191,834 / 92,220 |
| prefix hits / misses / inserts / evictions | 15 / 2 / 2 / 0 | 15 / 2 / 2 / 0 |
| prefix entries / bytes | 2 / 1,027,128,960 | 2 / 1,027,128,960 |
| continuation hits / evictions / entries | 0 / 15 / 2 | 0 / 15 / 2 |
| spec hits / misses / affinity rewinds / entries | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| VRAM defers / step-OOM parks | 4,659 / 0 | 4,659 / 0 |

Raw evidence:

- [`box1/20260808T222300Z`](raw/box1/20260808T222300Z/) contains the frozen workload,
  model and binary hashes, build log, both request JSONL files, per-response bodies,
  one-second GPU samples, final metrics, and server logs.
- [`runpod/20260808T221300Z`](raw/runpod/20260808T221300Z/) contains the first live-rig
  reproduction plus the original owner-service pre/post receipts.

## Root-cause anatomy

1. `PrefixCache::lookup` correctly chooses the longest stored token prefix. The observed
   entry is real and usage correctly credits its 6,148 tokens.
2. A prefix hit restores a snapshot into a fresh plain session and primes only the suffix.
   That is why request 2 drops from the cold 9--12 s range to 2.35 s.
3. `maybe_prefix_seed` is deliberately cold-only: it returns when `s.n_cached > 0`, and
   also returns when any existing entry already covers the prompt. A hit therefore cannot
   publish a newer prompt-end checkpoint. The reusable frontier freezes at the LCP learned
   on request 1 while the agent conversation keeps growing.
4. Exact whole-session continuation reuse requires the new prompt to extend the retired
   session's entire prompt-plus-generation token stream. Pi-shaped history rewriting strips
   reasoning from prior assistant turns, so this probe misses 15/15 times after the two
   initial parks.
5. Rewritten-history affinity and prompt-end rewind checkpoints exist only in
   `SpecSession`. PP-2 placement chooses `K=0`, so the deployed arm constructs no spec
   session. When forced on, the mechanism is nominated but its exactness check correctly
   declines at checkpoint minus two tokens; the checkpoint is later than the stable token
   boundary this template needs.
6. Plain retirement still parks two `ctx_cap=262144` caches. They are bounded by
   `MEMRA_REUSE_POOL=2`, but are sized to the full context rather than the request's
   prompt-plus-output need and are not reclaimed before the admission check.

## Prefix budget model

The two snapshots give an exact Step density of 83,520 bytes/token:

- 6,150 tokens = 513,648,000 bytes;
- 6,148 tokens = 513,480,960 bytes;
- a 4,096-token entry is 342,097,920 bytes (342.1 MB / 326.25 MiB);
- 2 GiB holds only 25,712 total cached tokens;
- a 262,144-token entry is 21,894,266,880 bytes (20.39 GiB), over ten times the budget.

No eviction occurred in this 14k-token receipt; budget thrash is therefore not the first
observed term. But the owner's 256k concern is confirmed structurally: once a single entry
exceeds 2 GiB, `insert_with_budget_pins` skips it rather than evicting into usefulness.
Simply raising the device budget to about 21 GiB is not safe because the cache competes with
active-session allocations and the admission reserve.

## Pi engagement audit

The installed `@earendil-works/pi-coding-agent` is 0.83.0. Its bundled pi-ai client has a
real session id in `options.sessionId`, but emits `x-session-id` only when
`compat.sendSessionAffinityHeaders` is true. The installed `local-memra` provider does not
set that compatibility option, and the detected default is false. The exact transient
RunPod provider profile used during the owner's report is no longer present, so this is an
installed-client/config audit rather than a packet capture; the audited default path does
not explicitly name the conversation to memra. The implicit fingerprint tier also cannot
engage on the deployed arm because it is probed only after a speculative session has been
parked, and the deployed policy creates none. See [`pi-affinity-audit.txt`](raw/pi-affinity-audit.txt).

Enabling the pi header alone is not a fix for this receipt: plain sessions currently ignore
affinity, and the forced control's body `session_id` still hit the two-token exactness veto.
It is a necessary client-side half of the recommended plain-affinity design and should be
enabled when that server path exists. The compatible Pi spelling is
`sendSessionAffinityHeaders: true` plus `sessionAffinityFormat: "openrouter"`, which emits
the `x-session-id` header memra accepts; Pi's default `openai` spelling emits other headers
that memra does not consume.

## Hypothesis disposition

| hypothesis | disposition | receipt |
|---|---|---|
| spec decode fails to credit cached tokens | not causal for deployed policy; forced path has a separate alignment miss | default selects K=0 and spec-off is identical; forced K=3 has 0/17 spec hits because checkpoint diff declines |
| session/KV state accumulates | bounded amplifier confirmed, leak not found | plain pool stays at 2, hits 0, evicts 15; it constrains c4 admission |
| Step misses the `free + pool_cached` gate fix | rejected | diagnostic explicitly reports driver plus pool-cached; the remaining live pool is not allocator-cached |
| 2 GiB cache thrashes large entries | not reached at 14k; structurally guaranteed by 25.7k total-token capacity | zero measured evictions, but 256k snapshot is 20.39 GiB and is skipped |

## Recommended design, in order

### P0: plain-session affinity with a prompt-end checkpoint

Port the `SpecSession` ownership idea to the plain PP-2 path, but fix the boundary the forced
control disproved. Retain a checkpoint at a **stable pre-generation token boundary**, not
unconditionally at prompt end. For structured chat, derive it before the template's live
assistant-generation suffix. For raw completions, split-prime a small conservative guard
window (for example 8--32 tokens), checkpoint before it, then prime the tail; the exact diff
still decides whether that earlier boundary is safe. Do not hardcode the observed value 2.

The checkpoint needs only a position for append-only full-attention KV and a real copy of
the hybrid recurrent state, following the existing spec mechanism. Park the session under
`(model, cache namespace, explicit/fingerprint affinity)`. On the next rewritten request,
identity nominates only; an exact token comparison through the stable boundary authorizes
rewind, then the guarded tail, rewritten answer, and new turn are primed. This makes work
proportional to per-turn growth instead of total conversation length and avoids copying a
second 20 GiB device snapshot.

Right-size a plain session to `need = prompt + max_new + safety slack`, as the existing spec
ladder already does, rather than always allocating the request's 262k cap. Grow/re-admit only
when a later turn needs a larger rung. This is also the direct fix for the useless ~11.2 GB
full-cap entries observed in the c4 gate.

Required gate: replay this exact 12-turn workload with affinity on/off, compare every output
token/hash against a cold oracle, require the stable resumed boundary to advance each turn,
require `completion_tokens <= max_tokens`, and require TTFT to track only the new suffix.
Cover multiple chat templates and raw completions so the guard is not Step-specific. Then
run `kernel-check`, `run-gen` argmax, and `run-spec` K=1..8 on the target rig before merge/tag.

### P0: admission must reclaim dormant state before deferring

Publish parked bytes, not just entry counts. When `free + cuda_pool_cached` is below
`cost + reserve`, evict or host-demote nonmatching plain/spec/prefix entries and re-read free
before queueing. Preserve matching affinity entries first. This should allow at least the
hardware-feasible concurrency instead of serializing behind dead continuation slabs; it
must still leave the transient reserve intact.

### P1: a bounded pinned-host checkpoint tier

Device VRAM should hold active sessions and the hottest matching checkpoint; dormant long
conversation checkpoints belong in host RAM. The pod has about 755 GiB host RAM
([host receipt](raw/runpod-host-memory-topology.txt)), but the
tier must be capped from `MemAvailable` rather than pinning it wholesale. Restore should be
asynchronous and PP-stage-local.

A separate lock block ran the existing CUDA `bandwidthTest` with pinned 512 MiB and 1 GiB
buffers on each box1 GPU ([raw screen](raw/box1/h2d-screen-20260808T230800Z.log)). This was one
cold/P0 screen, 33--35 C, and the tool itself warns it is not a performance benchmark:

| transfer | dev0 | dev1 |
|---|---:|---:|
| 512 MiB H2D | 9.44 ms | 9.44 ms |
| 1 GiB H2D | 18.87 ms | 18.85 ms |
| 512 MiB D2H | 9.17 ms | 9.17 ms |
| 1 GiB D2H | 18.34 ms | 18.33 ms |

At the measured 83,520 bytes/token density, 512 MiB represents 6,428 tokens. The controlled
box1 prefill slope prices those tokens at 11.30 s, versus the 9.44 ms raw H2D screen: about
1,198x transfer-only headroom. Linear screening of one 256k snapshot gives 0.385 s H2D or
0.759 s D2H+H2D, versus 460.9 s recompute at the box1 slope. This strongly promotes the host
tier to prototype; it is **not** an end-to-end restore claim. Checkpoint packing/scatter,
thousands of layer/state copies, two-device scheduling, synchronization, and recurrent-state
restore must be included in the implementation gate; raw link bandwidth alone is not a serving
latency result.

### P1: VRAM-aware device budget as a guardrail, not the owner

Replace the fixed standalone prefix budget with a hot-tier budget bounded by
`free-after-weights - active-session target - admission reserve`. This can use otherwise idle
VRAM, but it must shrink/reclaim as admissions arrive. A static 15--25 GB budget merely moves
the failure: one 256k snapshot can occupy it and prevent the destination session from being
admitted.

### P2: capped partial prefixes

Keep a bounded shared head (system/tools/project context) when no conversation checkpoint is
available. This prevents a single long entry from flushing all shared prefixes, but it does
not solve the owner report: after the cap, TTFT again grows linearly with every turn. Treat it
as cross-session fallback, not the growing-conversation owner.

## Why no runtime fix is committed here

The first dominant fix crosses cache ownership, stable-boundary selection, hybrid-state
rollback, admission, client identity, and exactness. The forced arm proves the existing
checkpoint cannot simply be reused unchanged. A rolling prefix insertion is small but would
only mask this 14k receipt until the 25.7k-token budget boundary, add a large D2D snapshot
every turn, and fail to serve the stated 256k workload. Forcing speculative decode globally
would expose the existing affinity path but would reverse the measured PP-2 placement default
and trade away its concurrency winner. Both would be result-chasing rather than a durable
correction.

This lane therefore lands the reproduction harness, raw evidence, and missing observability,
then specifies the implementable plain-affinity/reclaim design. It does not change serving
defaults, merge, tag, or push.
