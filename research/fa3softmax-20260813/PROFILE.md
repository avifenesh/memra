# Cached-prefill FA profile gate

Date: 2026-08-13
Verdict: **PROCEED TO MINIMAL CANDIDATE**

All observations below are from the local RTX 5090 Laptop GPU under the owner-imposed
210--1200 MHz thermal cap. They are diagnostic and relative-only. They are not absolute
throughput claims. Every profiler launch held `/tmp/memra-5090.lock`; the preflight checks
showed no other compute application, and no clock setting was changed.

## Frozen serving request

- Prompt: 4,860 exact token ids from `research/sellgate-20260812/workload.lock.json`.
- Canonical prompt-id hash: `eba9dff66d4ebf3f40d6db80298ab9c884fa1bb2e1a6d4ca2c7dadd4f21513fb`.
- Cold-request proof: both responses report `cached_tokens: 0` and 4,860 prompt tokens.
- Completion: 60 tokens, temperature 0, seed 3407, context limit 4,928.
- Path observed in both traces: `fa_dequant_kv_ws_bf16` followed by
  `fa_prefill_qw_db`; this is the requested dequant-once cached-prefill route.
- Baseline server binary SHA-256:
  `9b6ee7d863d82ce7ea1c18dec3cd0554cce283a6052c8b825de4ba077d3e6881`.

The one-request TTFT values in the request logs include profiler overhead and are not scored
performance measurements. They exist only to prove that the captured launches came from the
actual cold serving request.

## `nsys`: serving-path envelope

| Model | `fa_prefill_qw_db` launches | Full-chunk grid | Full-chunk total | Tail grid | Tail total | Share of traced GPU-kernel time |
|---|---:|---|---:|---|---:|---:|
| Q27 | 32 | `(64,24,1)`, N=16 | 197.128 ms | `(12,24,1)`, N=16 | 83.303 ms | 3.7% |
| Q35 | 20 | `(64,16,1)`, N=10 | 84.232 ms | `(12,16,1)`, N=10 | 38.516 ms | 5.8% |

The 64-CTA and 12-CTA x-dimensions are the 4,096-token serving chunk and the remaining
764-token chunk (`ceil(T/64)`). The totals are single profiled requests, not benchmark medians.

## `ncu`: stall and occupancy location

Each row is one replay-profiled launch selected by exact kernel name and grid. CUDA 13.1 NCU
used 19--20 replay passes; profiler reports remain under `/tmp` and are not repository files.

| Model / chunk | Duration | Tensor-pipe active | Warp cycles / issued inst. | Eligible warps / scheduler | Short scoreboard | Wait | Math throttle | Barrier | Occupancy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q27 / 4,096 | 13.50 ms | 34.31% | 8.33 | 0.12 | 3.84 | 1.65 | 1.06 | 0.14 | 8.33% |
| Q27 / 764 | 5.72 ms | 32.90% | 8.22 | 0.12 | 3.94 | 1.65 | 1.09 | 0.14 | 8.33% |
| Q35 / 4,096 | 9.22 ms | 33.41% | 8.33 | 0.12 | 3.85 | 1.65 | 1.06 | 0.14 | 8.33% |
| Q35 / 764 | 4.27 ms | 29.34% | 8.22 | 0.12 | 3.94 | 1.65 | 1.09 | 0.14 | 8.33% |

The kernel uses 255 registers/thread, 70.91 KiB dynamic shared memory as reported by NCU,
one CTA/SM, and one active warp per scheduler. No local-memory spilling was observed. Shared
memory, not the block limit or register allocator, is the reported one-block occupancy limiter.

Shared `ldmatrix` traffic is replay-heavy on every captured shape:

| Model / chunk | Shared wavefronts | Ideal | Excess | Actual / ideal |
|---|---:|---:|---:|---:|
| Q27 / 4,096 | 873,541,632 | 135,966,720 | 737,574,912 | 6.42x |
| Q27 / 764 | 354,302,976 | 54,912,000 | 299,390,976 | 6.45x |
| Q35 / 4,096 | 582,361,088 | 90,644,480 | 491,716,608 | 6.42x |
| Q35 / 764 | 236,201,984 | 36,608,000 | 199,593,984 | 6.45x |

## SASS phase attribution

The extracted Source Counters tables were split at the stable SASS boundaries surrounding the
loop's QK MMA, register softmax/P stores, and PV MMA. This is sampled-stall attribution, not a
wall-time decomposition.

| Model / chunk | QK samples | Softmax + P-store samples | PV samples | QK share of short-scoreboard samples | PV share of short-scoreboard samples |
|---|---:|---:|---:|---:|---:|
| Q27 / 4,096 | 37.25% | 5.42% | 38.56% | 49.36% | 44.71% |
| Q27 / 764 | 39.03% | 5.68% | 40.52% | 49.69% | 45.10% |
| Q35 / 4,096 | 37.23% | 5.42% | 38.61% | 49.66% | 44.47% |
| Q35 / 764 | 39.12% | 5.64% | 40.52% | 49.63% | 45.18% |

The P stores themselves are not the dominant sampled region. The material cost appears after
the mandatory P round-trip: `STS` is followed by `LDSM.16.M88.4`, and dependent PV
`HMMA.16816.F32.BF16` instructions carry the large short-scoreboard counts. QK and PV together
occupy roughly three quarters of sampled cycles and more than 94% of short-scoreboard samples in
the loop. Softmax/P-store is only about 5.4--5.7% of samples, so a softmax-only rewrite is not
justified; overlapping the independent QK and PV phases is the mechanism worth testing.

## Gate decision and numeric boundary

The profile clears the experiment gate because the serialized QK/PV phases are material, tensor
issue is only 29--34%, and the single resident warp per scheduler offers no same-scheduler latency
hiding. The candidate must therefore test cross-warp phase staggering rather than claim literal
WGMMA semantics: sm_120a `mma.sync` has no WGMMA commit/wait group.

The minimal candidate will keep every warp's KV-tile traversal and every per-tile MMA/online-
softmax accumulation in the existing order. It will use two warp cohorts, generation-tagged
shared-stage handshakes, and a second P stage so cohort A can execute QK for tile `i+1` while
cohort B executes PV for tile `i`. Because rows are warp-private, this changes inter-warp timing
only; it does not intentionally create a new numeric class. Any bit difference still stops the
lane.

## Profiler artifact boundary

The non-committed reports are identifiable without exposing them:

- Q27 Nsys: `3a545d87433e6b6998c42aafc6ce980533ce10319085aa3cf90ea4e6ce6becfd`
- Q27 NCU: `7ff1929e662ed862e4b369cd5cf43ba7d4ac9c18aaa8548fd270d04a5709c22c`
- Q35 Nsys: `10363f2670e3c8220d0a6731c95f1329d5683796c6b1b26687036121ba70485a`
- Q35 NCU: `ca0a0d86f33521d949f721c5e0ce9be6ea3c7156ac840a0d3736d6326e373638`

Only extracted CSV tables and console/request logs live under `raw/`.
