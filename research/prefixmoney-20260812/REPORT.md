# Prefix-cache on the Step PP-2 money path

Date: 2026-08-12

Source inspected and locally run: `8b2ba8c883152fdbb9f9bbd800a055ad03fe80c4`

Lane: `lane/cx-prefixmoney`

## Verdict

**Native-peer, ring-OFF, plain batched Step PP-2 has a working prefix-cache path, but the
current production intersection is not yet validated.** Step-3.7 PP-2 cache hits and large TTFT
wins were measured on 2026-08-08. The later dual-PP soak and flip batteries explicitly set
`MEMRA_PREFIX_CACHE_MB=0`, so they never established cache-on output identity, hit TTFT, or
90%-hit concurrency under the current naked dual scheduler.

There is no small engine unwiring to repair on the native-peer path. The smallest correct
enablement is a serve configuration and a missing integration battery: native P2P devices 0/1,
SWA ring OFF, host bounce OFF, serving spec OFF, batching ON, and a cache budget sized for the
Step working set. This lane adds that exact deferred battery; it does not change runtime code.

Two architectural combinations remain deliberately unavailable:

1. Step35 SWA-ring sessions refuse flat-history prefix snapshots/restores.
2. Host-bounce PP-2 disables prefix snapshots/restores because those copies do not use the
   bounced activation transport.

Therefore the full `262144`-context ring-ON serve shape cannot also serve from the current prefix
cache. A finite-context, ring-OFF native-peer deployment can, subject to the box1 battery below.
Promotion status is **HOLD until that battery passes**.

## 1. History first: what `MEMRA_PREFIX_CACHE_MB=0` means

The dual-PP soak launches Step-3.7 with this literal configuration:

```text
MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 MEMRA_PREFIX_CACHE_MB=0
```

Source: `research/dualpp1-20260811/box1-soak.sh:119-128`. `git blame` attributes the line to
`673596a00743b6a5f89b6f71db46b0150ba6810c`, whose complete subject is:

```text
test: add dual PP scheduler and slot soak battery
```

The setting, surrounding comments, report, and introducing commit contain no statement that
prefix caching was broken, unsafe, or refused. The setting was copied into the later fixed soak
and serve-stress recipes (`research/dualpp2-20260811/box1-soak-fixed.sh:126` and
`box1-regate-servestress.sh:82`). The only evidence-bounded conclusion is:

> The dual-PP batteries excluded prefix caching and therefore left the dual-PP/cache intersection
> untested. Their `=0` is not evidence of a cache defect, and the historical reason for choosing
> it is not recorded in the inspected files or commit.

This differs from the ring lane, which records an explicit refusal and reason:

```text
[prefix-cache] refused for MEMRA_SWA_RING=1 Step35 session (flat-history snapshots/restores are excluded)
```

Source: `research/ringval-20260810/RESULTS.md:95-111`; the source guards are
`crates/memra-server/src/worker.rs:2480-2488,2537-2544,5588-5595`.

## 2. Exact state map

| State | Step PP-2 combination | Evidence, quoted or measured | Meaning for this lane |
|---|---|---|---|
| **WORKS, before the dual default** | Native peer, ring OFF, plain batched Step-3.7 PP-2 | The 2026-08-08 N=8 receipt records cache OFF as `[0] x 8` and cache/dedup ON as `[0] + [1024] x 7`, with p50 TTFT `22.263 s -> 3.852 s` (`research/prefixdedup-20260808/PROGRESS.md:217-239`). A separate Step PP-2 serve receipt records full 4k hits at p50 12.2 ms (`research/serve-ready-20260808/RESULTS.md:26-34`). | The native-peer cache is not generally unwired from PP-2. These receipts predate the 2026-08-11 naked dual-PP default and do not close the requested current-intersection gate. |
| **UNTESTED** | Current naked dual-PP Auto + native-peer prefix cache | Every dualpp1/dualpp2 soak and serve-stress server boot sets `MEMRA_PREFIX_CACHE_MB=0`; no recorded rationale says why. | This is the actual missing money-path gate. It must not be called working or broken until the deferred battery runs. |
| **CONFIG REFUSAL / MISS** | Step 4k prefixes under the 256 MiB default budget | The server said `skip seed insert: entry 343.0MB > budget 268MB`; the request missed at 6.08 s. `MEMRA_PREFIX_CACHE_MB=2048` made the same shape hit at 12.2 ms (`research/serve-ready-20260808/RESULTS.md:111-126`). | A production Step configuration must size the device-snapshot budget for the intended active prefix working set. Leaving the default is not enablement. |
| **REFUSED** | Step35 + `MEMRA_SWA_RING=1` + prefix cache | Exact refusal above. The ring report calls the stock smoke scoped red in exactly the flat cache-accounting cell (`research/ringval-20260810/RESULTS.md:95-111`). | Architectural: define an exact ring snapshot/restore contract before combining the 262k ring capacity path with prefix reuse. Do not flip a flag around it. |
| **REFUSED / disabled** | `MEMRA_PP_HOST_BOUNCE=1` + prefix cache | “Prefix and affinity cache snapshots … issue device copies through the primary engine rather than the activation boundary. Prefix snapshots and plain-affinity checkpoints are disabled under host bounce” (`research/hostbounce-20260810/RESULTS.md:73-85`). Current code forces the configured cache budget to zero and explains why (`worker.rs:1474-1489`), then logs the safety doors (`worker.rs:3123-3127`). | Architectural: snapshot/restore must copy each stage through its owner/bounce transport before this can open. |
| **REFUSED locally** | Same-device PP-2 (`MEMRA_PP_DEVICES=0,0`) with the production prime pipeline | The 5090 run returned: `prime chunk pipeline refused with 2 stage streams on one device — that concurrent-stream placement remains quarantined by the deferred pp flake record. Use one device per stage or MEMRA_PRIME_PIPE=0 for the serial split.` (`raw/local5090/gate.log`; source `crates/memra-engine/src/hybrid_forward.rs:737-753`). | This refusal occurs before cache insertion and blocks the only local PP-2 placement. The suggested serial-prime rollback was not used because it changes the path the lane was asked to validate. |
| **POLICY BYPASS** | Spec session + cross-request prefix cache | Source says a trunk-only restore would leave draft state unprimed, so spec sessions bypass (`worker.rs:5554-5567,5622-5632`; `docs/FLAGS.md:507`). | Architectural if cache+spec is required. It does not block the plain money path, which explicitly uses `MEMRA_SERVE_SPEC=0`. |
| **POLICY BYPASS** | Legacy `MEMRA_SERVE_BATCH=0` | `docs/FLAGS.md:507` says spec and legacy round-robin bypass. | Keep batching enabled. |
| **UNWIRED performance fallback** | Step35 in-window fanout suffixes carried into the fresh PP batch core | `docs/FLAGS.md:506` says: “Step35 suffixes keep their per-request continuation-prime path because its fresh PP batch core deliberately refuses carried caches.” | This does not remove cross-request hits, but it leaves additional shared-suffix prefill savings on the table. It is separate from the minimal enablement. |

The current native-peer allocation side is PP-aware: a restored session is allocated with
`pp::new_cache`, whose contract places every layer cache on its owning stage
(`worker.rs:5632-5657`; `crates/memra-engine/src/pp.rs:2322-2368`). Snapshot and restore still
take one primary `Engine` (`worker.rs:2477-2583`), which is why native P2P can carry the copies
but host bounce is shut.

## 3. Local RTX 5090 result

### Configuration

- GPU: NVIDIA GeForce RTX 5090 Laptop GPU, 24,463 MiB, driver 595.84.
- Model: local 15,705,920,064-byte Qwen3.6-27B NVFP4/Q4_K_M GGUF, the largest known local
  PP-2-capable artifact.
- Server: release build, CUDA 13.1, auto sm_120a.
- Placement: `MEMRA_PP_STAGES=2`, `MEMRA_PP_DEVICES=0,0`.
- Cache: `MEMRA_PREFIX_CACHE_MB=512`, dedup ON; server logged
  `[prefix-cache] on: budget 537MB`.
- Production doors: dual/overlap/prime-pipe unset, ring explicitly OFF, host bounce unset,
  spec OFF. The run held `/tmp/gpu5090.lock`, used `nice`/`ionice`, and made no clock change.

### Result

The model loaded and the server became ready. The first 528-token request admitted, recorded one
cache miss, and then hit the binding same-device prime-pipeline refusal quoted in the state map.
Metrics ended at one admitted/completed request, zero output tokens, zero inserts, and zero hits.
The server shut down cleanly.

The lane instruction says a refusal and its reason are the deliverable and must not be bypassed.
Accordingly:

- `MEMRA_PRIME_PIPE=0` was **not** set;
- the smaller 9B model was **not** run because the refusal predicate is placement-based, not
  model-size-based;
- there is no false cache exactness or timing claim.

### Requested local hit-path timing delta

**Unavailable: N=0 cache hits.** Prefill refused before the first prefix could be inserted, so
neither repeated-identical nor shared-prefix hit timing exists locally. A numeric delta here would
be fabricated or would describe a bypassed serial-prime path.

Historical box1 context is not substituted for the local result:

| Historical shape | N / regime | Cold | Hit | Delta |
|---|---|---:|---:|---:|
| Step PP-2, simultaneous 1,024+16 fanout | one N=8 burst per arm | p50 22.263 s | p50 3.852 s | -82.7%, 5.78x |
| Step PP-2, repeated 4,107-token prompt | N=3 hit requests | miss 6.08 s at undersized default | p50 12.2 ms at 2,048 MiB | historical config finding |

Those receipts show why the path matters; they do not validate the later dual default, byte
identity on that intersection, or 90%-hit concurrency.

Raw local evidence: `research/prefixmoney-20260812/raw/local5090/`. It contains the complete
driver/gate error, server log, metrics before/after, model/server/harness hashes, one-second GPU
samples, and before/after compute-process snapshots. `raw/build-server.log` is the build receipt.

## 4. Minimal fix list for 90%-hit agentic traffic

### Required now: configuration and validation, not an engine patch

1. **Use native-peer cross-device PP-2.** Set stages/devices to `2` / `0,1`; leave
   `MEMRA_PP_HOST_BOUNCE` unset. Keep the boot peer-integrity probe enabled.
2. **Run the cache-compatible finite-context shape.** Set `MEMRA_SWA_RING=0`. This is the
   current hard tradeoff: the validated 262k ring capacity shape cannot cache prefixes.
3. **Keep the plain batched path.** Set `MEMRA_SERVE_SPEC=0`; leave batched serving on. Do not
   claim cross-request reuse for spec sessions.
4. **Budget the actual Step working set.** The deferred battery uses
   `MEMRA_PREFIX_CACHE_MB=4096`, not the 256 MiB default and not zero. Size production from entry
   bytes, active tenant prefixes, and measured eviction churn. Prefix entries are compact device
   snapshots (`docs/FLAGS.md:507`); they are not free host metadata.
5. **Preserve routing and isolation.** Repeated requests must return to the same server process
   and carry a stable tenant-scoped `cache_salt`; changing the salt is intentionally cold
   (`docs/SERVING.md:925-937`). OpenRouter's current prompt-caching guidance likewise describes
   sticky routing as the mechanism that keeps follow-ups on the cache-owning provider:
   <https://openrouter.ai/docs/guides/best-practices/prompt-caching>.
6. **Promote only on the deferred receipt.** Require cached-token usage, output-byte identity,
   hit TTFT, token-weighted hit rate, eviction rate, dual-slot liveness, and the concurrency
   comparison below at the staged commit.

No runtime source change is justified by the state map. Turning an architectural refusal off or
using the local serial-prime rollback would not be a config fix; it would test a different shape.

### Architectural follow-ups, reported only

1. **Ring-aware prefix state.** Define snapshots in terms of the live logical SWA window,
   absolute positions, recurrent conv/SSM state, and exact restore/lap rules. This is required to
   combine prefix reuse with the 262k ring capacity win.
2. **Transport-aware snapshot/restore.** Allocate/copy every prefix plane through its owning
   stage and support D2H/H2D staging so host-bounce PP-2 no longer depends on primary-engine peer
   copies.
3. **Spec-complete prefix entries.** Capture/restore target and draft state together if spec and
   prefix reuse are ever required simultaneously.
4. **Step suffix batch carry.** Teach the fresh Step PP batch core to accept already-carried cache
   state before claiming the full in-window fanout saving.

### Concurrency effect: what is known now

The number of additional concurrent sessions at a 90% hit rate is **not yet measured**. Existing
receipts measure either a 98.46%-shared N=8 burst (1,024 / 1,040 tokens) or ring capacity while
prefix caching is refused. Combining those numbers would be invalid.

Also, a prefix hit does not eliminate the new session cache: the implementation deep-copies the
prefix entry into a freshly allocated PP-aware cache. The expected gain is avoided prefill work
and lower TTFT/queue occupancy, not a proven reduction in per-session KV allocation. Cache entries
themselves consume the configured device budget. The box1 ladder therefore measures clean carried
concurrency and TTFT rather than estimating it from tok/s or VRAM arithmetic.

## 5. Deferred box1 validation battery

Executable entrypoint: `research/prefixmoney-20260812/run-box1.sh`

Clients: `prefix_gate.py` and `cache_concurrency.py`

The script requires `EXPECTED_SOURCE=<staged-lane-commit>`, refuses a nonmatching checkout,
builds `memra-server` from that checkout (or requires an explicit expected hash for a custom
binary), requires two idle GPUs, holds `/tmp/memra-gpu.lock` (or verifies inherited fd 9), records
artifact hashes and thermal/process state, makes no clock changes, and never overwrites an
existing output directory.

### Server shape

```text
Step-3.7 IQ4_XS trunk + Q8_0 MTP drafter loaded
CUDA_VISIBLE_DEVICES=0,1
MEMRA_PP_STAGES=2
MEMRA_PP_DEVICES=0,1
MEMRA_DUAL_PP unset              # current Auto default
MEMRA_PP_OVERLAP unset           # follows Auto
MEMRA_PRIME_PIPE unset           # production pipeline, no local bypass
MEMRA_PP_HOST_BOUNCE unset       # native peer
MEMRA_SWA_RING=0
MEMRA_SERVE_SPEC=0
MEMRA_PREFIX_CACHE_MB=4096
MEMRA_PREFIX_DEDUP=1
MEMRA_CTX=262144
MEMRA_MAX_SESSIONS=64
```

The drafter is loaded to preserve the serve residency shape, but serving spec is off so the
cross-request prefix cache remains eligible.

### Phase A: byte exactness and hit timing

N=3 independent namespaces. Each repetition performs:

1. `A-cold`: `prefix + suffix-A`, seeding a full-prompt entry;
2. `B-learn`: `prefix + suffix-B` in the same namespace. The full A entry is not a prefix of B,
   so this remains cold while its 4,096-token LCP arms/inserts the exact boundary entry;
3. `B-hit`: repeat B, requiring exactly the 4,096-token boundary from cache and byte identity
   with `B-learn`;
4. `A-hit`: repeat A, requiring the full 4,551 prompt tokens from cache and byte
   identity with `A-cold`;
5. a barrier burst of eight identical A hits, each requiring full cached-token credit and the
   same output bytes as `A-cold`.

Every request is greedy with a fixed seed and 64 generated tokens. The client stores the exact
UTF-8 response bytes as base64 plus SHA-256; timing may change, bytes may not. Aggregate gates:

- at least 30 prefix hits and six misses across the three repetitions;
- `dual_pp.slot_pairs` increases under the concurrent hit waves;
- `dual_pp.slot_collisions` does not increase;
- every expected cached-token count and every byte comparison matches;
- no request/refusal/OOM error.

The summary reports median cold/hit TTFT, absolute delta, and speedup separately for repeated
identical prompts and shared-prefix prompts.

### Phase B: 0%-vs-90%-hit concurrency

The prompt is 4,096 cached tokens plus a 455-token unique suffix:

```text
4096 / (4096 + 455) = 90.0022% requested hit ratio
```

Cold and hit arms run at `c=1,2,4,8,16,24,32`, N=3 cells per arm and concurrency, under one
server boot and one GPU sampler. Arm order is interleaved and reversed on alternating cells. Cold
requests use unique salts and must report zero cached tokens. Before each hit cell, a unique
4,096-token prefix is seeded; every measured request must report exactly 4,096 cached tokens.

Each cell records request-level TTFT/output hashes and:

- request completion count and wall throughput;
- output tok/s;
- p50/p95 TTFT;
- actual token-weighted hit ratio;
- peak sampled active/queued requests;
- admission session/VRAM defers and step-OOM parks;
- cache evictions;
- dual-PP slot pairs and collisions.

The owner-requested carried-concurrency number is defined without inventing a product SLO:

> largest clean tested concurrency whose median-of-cell-p95 TTFT is no worse than the cold c=1
> median-of-cell-p95 TTFT.

The receipt emits `cold_latency_parity_concurrency`, `hit90_latency_parity_concurrency`, and their
multiplier. It also retains the full curves so a later product SLO can be applied without rerunning.
“Clean” for the capacity number also requires zero admission session/VRAM defers at that cell.
Any request error, cached-token mismatch, OOM park, slot collision, or absence of dual-slot pairs
makes the battery fail; admission defers bound the reported capacity rather than failing the
entire ladder.

### Promotion gate after box1

The prefix-specific battery is necessary but does not replace the repository pre-release battery.
Before merge/tag, the same staged commit still requires the designated PRO-pair `kernel-check` ALL
GREEN, `run-gen` argmax MATCH on Step, and `run-spec` K=1..8 self-consistency PASS. Report spill,
decode, cache-hit TTFT, cache-hit ratio, and concurrency as separate quantities.

## 6. What was and was not changed

- Added the exactness/timing client, 90%-hit concurrency client, deferred box1 runner, progress
  ledger, report, and raw local/build receipts.
- Host prefix-cache tests pass 13/13 (`raw/host-prefix-tests.log`). Both clients also pass a
  stateful fake-API control smoke covering miss, LCP learning, hit accounting, byte comparison,
  concurrency, and summary generation; that smoke is harness validation, not model evidence.
- Did not change prefix-cache, PP, scheduler, model, kernel, generated performance-board, or release
  code.
- Did not touch box1.
- Did not merge, tag, push, regenerate boards, format the tree, change clocks, or bypass hooks.
