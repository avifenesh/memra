# Request-shaped 256k admission and reclaim-on-defer

Lane `lane/cx-cap256k`, reconciled against local train tip `abe318dc`. This closes code-audit
findings 6.1 and 6.2 without changing the affinity pool shape or the existing
`SPEC_SHRINK_RESERVE` contract.

## Verdict

PASS on the local RTX 5090 proof rig.

- Bug A is fixed: admission is armed at model load and every request is charged from its own
  effective context cap, not a scalar frozen by the first measurable admission.
- Bug B is fixed: a shortfall with parked sessions evicts the globally oldest entry across
  `reuse` and `spec_reuse`, re-reads effective free memory, and defers only if the request still
  does not fit or both pools are empty.
- The matched after sequence admitted the c=4 128k burst with zero VRAM defers after reclaiming
  one parked 256k spec session. All requests completed and the captured failure scan was empty.

## Implementation

The lane keeps one allocation geometry source rather than duplicating model rules in the server:

- `memra-kv` factors the full-attention per-layer layout used by `Cache::new_inner` and exposes
  its context-linear bytes/token sum. Gemma per-layer head geometry, shared layers, and active KV
  format doors therefore affect allocation and admission identically.
- `memra-engine` exposes plain-session and speculative-session coefficients. The latter adds the
  persistent MTP scratch bytes/token computed by the same helper used by `MtpScratch::new`.
- The worker tokenizes/renders a queued request once, derives the unchanged effective `ctx_cap`,
  and evaluates `bytes_per_token * ctx_cap + fixed_residual`. The residual is a per-model
  high-water measurement of allocation bytes not explained by the linear coefficient; reuse or
  allocator-pool hits cannot lower it.
- `effective_free_bytes` retains the admit-OOM lane's driver-free plus CUDA-pool-cached accounting.
  `SPEC_SHRINK_RESERVE` is unchanged. Spec-capable models still pay that fixed transient reserve;
  the plain-only fallback still reserves one request cost.
- `ReuseEntry` and `SpecReuseEntry` carry `parked_at`. Reclaim scans the existing maps/vectors,
  removes exactly the globally oldest entry, updates the existing eviction counters, and leaves
  affinity checkpoints, keys, selection, and ownership untouched.

## Matched 5090 receipt

Workload in both arms: one 8k calibrator, two sequential 256k requests allowed to park, then a
barrier-released c=4 128k burst. Greedy, short generation, same byte-identical trunk/draft
artifacts, `MEMRA_REUSE_POOL=2`, prefix cache off, and one exclusive
`/tmp/memra-gpu.lock` hold per run.

Raw blocks:

- Before: `raw/20260809T113429Z-before/`
- After: `raw/20260809T123952Z-after/`

| Observation | Before | After |
|---|---:|---:|
| 8k calibration | CUDA-pool hit left cost map unset | 152 MB linear; learned 103 MB fixed residual |
| 256k spec estimate | first measurable admit became a 4,899 MB scalar | 4,968 MB before allocation |
| 128k request estimate | inherited 4,899 MB scalar | 2,536 MB spec; 2,292 MB plain |
| parked reclaim at shortfall | none; 2 spec entries remained | 1 oldest spec entry evicted |
| effective free across reclaim | no reclaim | 3,430 MB -> 8,486 MB |
| admission VRAM defers | 10 | 0 |
| burst completion | 4/4 | 4/4 |
| burst TTFB service order, seconds | 0.038, 0.306, 0.463, 0.853 | 0.040, 0.086, 0.298, 0.298 |
| burst TTFB span | 0.815 s | 0.258 s |
| step-OOM parks / captured failures | 0 / none | 0 / none |

The after log order is the load-bearing proof: it first reports the 128k request-specific charge,
then `reclaim-on-defer` with the 3,430 -> 8,486 MB free-memory change, and contains no subsequent
`VRAM defer` line. Final metrics independently report `admission_vram_defers: 0`.

This is N=1 per arm, not a throughput benchmark or median. The before run recorded seven P0
samples at 68--76 C. The after run recorded five samples (first idle sample P8/56 C, then P0 at
56--65 C). A pre-existing Hermes gateway context used 394 MiB in both arms and is captured in
both compute-app censuses. The TTFB spans are therefore single-run serialization observations;
the estimator values, reclaim ordering, zero-defer counter, and clean completion are the verdict.

## Tests and gates

- `cargo test -p memra-server`: **152 passed**, 0 failed (148 merged baseline + 4 new).
  - request cost distinguishes 128k from 256k and charges spec scratch only on the spec shape;
  - an 8k observation updates only the fixed high-water residual;
  - global LRU selection compares plain and spec candidates together;
  - hoisted request context sizing preserves explicit, floor, bounded, and model-cap behavior.
- `cargo build --release`: PASS on CUDA 13.1, auto-detected sm_120a.
- Local-CI admission slice under `/tmp/memra-gpu.lock`: kernel-check GREEN; c=64 stress **64/64**
  completed with well-formed streams, worker alive, and server log clean. Timing is informational
  because the same 394 MiB Hermes context remained resident.
- Inverted teeth under the same lock: `MEMRA_ADMIT_RESERVE_MB=16` completed only **46/64**;
  18 streams failed and their client rows captured `CUDA_ERROR_OUT_OF_MEMORY`. The gate returned
  `TEETH OK`, proving the ordinary green still depends on the preserved transient reserve.

## Scope and handoff

- No origin push, merge, tag, runtime-default flip, or perf-board update was made.
- This lane changes admission/reclaim only. Full-cap parking and a byte-budgeted pool remain the
  separate code-audit 1.3/6.3 follow-up; eviction here intentionally sacrifices a resume
  opportunity when live admission needs the memory.
- The local 5090 Laptop is the required proof rig, not the final RTX PRO 6000 Blackwell-class
  deployment target. The result is correctness and admission evidence, not a deployment
  performance claim.
