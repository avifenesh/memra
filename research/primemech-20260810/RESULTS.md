# Concurrent-prime anatomy — local RTX 5090 Laptop

Date: 2026-08-10

## Verdict

On this single 82-SM RTX 5090 Laptop, simultaneous distinct-prefix 4k primes do
not create concurrent prime compute. They create more serial work on one GPU
worker and one compute stream:

- Production aggregate prefill is flat: **4519 / 4489 / 4480 tok/s** at
  N=1/2/4, or **1.000x / 0.993x / 0.991x** of solo.
- Client TTFT is correspondingly linear: **0.924 / 1.852 / 3.704 s** p50,
  or **1.000x / 2.004x / 4.008x** of solo.
- At N=4, the median burst has 16,512 prompt tokens and spends about **3.685 s**
  in the worker's measured prefill phase versus **3.728 s** to the last first
  token: about **98.9% of burst wall is prime work**, not scheduler idle.
- A typical steady N=4 tick reports four separate calls and no batch:

  ```text
  [tick] act=4 ... prefill_single_calls=4 prefill_single_tokens=4096 \
         prefill_batch_calls=0 prefill_batch_tokens=0 prefill_ms=786.0 ...
  ```

- During N=4 overlap, the 1 Hz samples report **100% NVML GPU-busy p50**, **85%
  GPM SM-busy p50**, and **27% of theoretical DRAM bandwidth p50**. This is not
  a storage or gross DRAM-bandwidth stall with an idle second compute lane.

The named serialization mechanism is a **host-synchronous per-session prime on
the sole CUDA worker, ending in a D2H synchronization on the same in-order
compute stream**. There is no top-level engine mutex responsible for the
measured sequence. The prime-slab and other shared-scratch mutexes are currently
uncontended because the single worker has already serialized entry; they become
correctness constraints if multi-stream entry is attempted.

The live production path has one additional, material detail: chat-template
traffic arms a plain-affinity checkpoint even with `MEMRA_REUSE_POOL=0`. That
boundary blocks the solo-widening predicate and excludes these sessions from
the existing concat-prime path. The 4k queue length also exceeds the 2048-token
batch ceiling. Any cross-request batching proposal must preserve the per-request
checkpoint boundary; merely raising a scheduler or batch limit is not the
mechanism.

## Protocol and provenance

One bounded A/B block held `/tmp/memra-gpu.lock` from
`2026-08-09T23:14:44Z` through `23:16:21Z`:

- Rig: local NVIDIA GeForce RTX 5090 Laptop GPU, GB203, 82 SM, sm_120a.
- Model: `qwen3.5-9b-judge-q8_0.gguf`, 9,527,501,696 bytes,
  SHA-256 `0825505bda37933f5856fd0751273b3bdf7224961d81dad9c4fcc1d47d49210c`.
- Run tree: `415b01c1468717292c9270a1f094554bf01449fa`.
- Release binary: 51,911,736 bytes,
  SHA-256 `fc76bdad1f61056cbf872d782b4a7bdc77f7b15be36015865a2df3189b24dde0`.
  The capped `CARGO_BUILD_JOBS=4`, `nice -n 10` build used CUDA 13.1 and
  auto-detected `120a`.
- Request: real `/v1/chat/completions`, no-think, greedy, eight output tokens;
  a request-specific marker is at the start of each otherwise equal long text.
  Every request has a unique `cache_salt` and reported `cached_tokens=0`.
- Shape: worker-authoritative prompt usage is 4,128 tokens/request in the
  production arm and 4,132 in the control (the arm label accounts for the
  four-token difference). Throughput uses each arm's exact count.
- Schedule: one excluded warmup, then `1,2,4,4,2,1,2,4,1`, two seconds of
  cooldown between bursts. Each concurrency therefore has three measured
  bursts: 3/6/12 request observations at N=1/2/4.
- Production arm: naked prefill geometry. Control: only
  `MEMRA_PREFILL_TICK=8192` is explicit. Both keep the default 2048-token
  concat-prime ceiling and default-on affinity; cross-request KV/prefix reuse is
  disabled.
- GPU sampling: CUPTI-free `nvidia-smi dmon` at 1 Hz, including ordinary
  `sm`/`mem` plus GPM metric 2 (busy SM percentage) and metric 10 (DRAM bandwidth
  versus theoretical maximum). Each printed sample is joined as a trailing
  one-second interval overlapping the client burst.

The block entered at 52 C and 31 MiB. The sampled thermal envelope was 55-82 C;
the production arm came first, so cross-arm percentage deltas are anatomy, not
promotion-grade order-paired evidence. Both clients exited 0, the server logs
have no CUDA error, OOM, panic, actual Xid event, request error, or server death,
and each teardown returned the GPU to 31 MiB. The wrapper's final `rc=1` is only
the original offline formatter attempting to format the string arm name as a
float after both servers were already stopped. The corrected analyzer consumes
the same immutable client/server/GPU logs; no GPU rerun was needed.

## Production anatomy

`server agg` is exact prompt tokens divided by the sum of worker
`prefill_ms` for the burst. `client agg` is exact prompt tokens divided by wall
to the last first visible token. `prime span` is the server TTFT trace from the
request's first `prefill_tick` entry until its prompt queue empties; it can be
shorter than client TTFT for a request first admitted after another synchronous
call. All medians below are three bursts/cell; p95 is over the 3/6/12 individual
requests.

| simultaneous primes | prompt tok/request | server aggregate tok/s median (range) | scale vs N=1 | client aggregate tok/s median | client TTFT p50 / p95 | per-prime span p50 / p95 | worker queue wait p50 / p95 | first-prime wait p50 / p95 | prompt-work scheduler ticks / single calls | batch calls |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,128 | **4,519.4** (4,510.5-4,544.8) | 1.000x | 4,467.0 | **0.924 / 0.926 s** | 913.5 / 915.6 ms | 0.1 / 0.1 ms | 0.3 / 0.5 ms | 12 / 12 | 0 |
| 2 | 4,128 | **4,488.9** (4,483.8-4,522.3) | 0.993x | 4,443.2 | **1.852 / 1.870 s** | 1,733.9 / 1,829.8 ms | 4.5 / 197.4 ms | 98.6 / 195.0 ms | 12 / 24 | 0 |
| 4 | 4,128 | **4,480.3** (4,462.0-4,505.0) | 0.991x | 4,429.0 | **3.704 / 3.739 s** | 3,299.2 / 3,646.6 ms | 105.3 / 585.1 ms | 290.4 / 606.6 ms | 13 / 48 | 0 |

The 12 calls/request are four 1024-token calls, the request-specific
plain-affinity boundary/tail, and the remaining below-prime-floor tokenwise
calls. Arrival skew sometimes adds one scheduler tick at N=2/4, but never
changes the 12 calls/request or forms a batch.

## Larger-quantum control

The explicit 8192 budget does **not** make this production-shaped chat prompt a
single call: the plain-affinity boundary leaves eight calls/request (one
4,125-token call plus seven tokenwise boundary-tail calls). It is still a useful
control because it removes most outer segmentation without changing the engine,
stream, model, or request shape.

| simultaneous primes | prompt tok/request | server aggregate tok/s median (range) | scale vs control N=1 | delta vs production | client TTFT p50 / p95 | per-prime span p50 / p95 | queue-wait p95 | first-prime-wait p95 | prompt-work ticks / single calls | batch calls |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,132 | **4,649.0** (4,616.8-4,689.1) | 1.000x | +2.9% | **0.900 / 0.906 s** | 889.1 / 895.0 ms | 0.1 ms | 0.4 ms | 8 / 8 | 0 |
| 2 | 4,132 | **4,581.2** (4,579.4-4,608.5) | 0.985x | +2.1% | **1.808 / 1.833 s** | 1,396.2 / 1,785.6 ms | 802.5 ms | 787.8 ms | 9 / 16 | 0 |
| 4 | 4,132 | **4,530.6** (4,524.5-4,543.0) | 0.975x | +1.1% | **3.669 / 3.691 s** | 2,408.8 / 3,607.6 ms | 2,498.3 ms | 1,697.9 ms | 9 / 32 | 0 |

The shorter N=4 median `prime span` is not less per-request compute. Later
requests do not mark `prime_start` until earlier synchronous calls return. Their
lost wall moves into queue/first-prime wait: the p95 queue delay jumps from
585 ms in production to **2.50 s**, in approximately one 0.8 s large-call step
per blocked arrival. End-to-end TTFT remains about four times solo. This is the
direct signature of the worker being unable to drain/tokenize arrivals while it
is blocked in a prime call.

## GPU activity during the overlap window

Ordinary `sm` is percent of the preceding device sample period with one or more
kernels executing; ordinary `mem` is percent of that period with global-memory
reads/writes. GPM `SM` is the percentage of SMs busy, while GPM `DRAM` is
bandwidth as a percentage of theoretical maximum. Sub-second N=1 bursts have
only boundary-straddling 1 Hz intervals, so their p50s are cadence-sensitive;
N=4 has 14-15 intervals and is the useful saturation row.

| arm | simultaneous primes | joined 1 Hz samples | NVML GPU busy p50 / p95 | NVML memory-active p50 / p95 | GPM SM busy p50 / p95 | GPM DRAM BW p50 / p95 | sampled temp range | power p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| production | 1 | 5 | 99 / 100% | 41 / 89% | 56 / 80% | 21 / 30% | 63-74 C | 84 W |
| production | 2 | 9 | 100 / 100% | 41 / 62% | 73 / 87% | 26 / 34% | 55-75 C | 86 W |
| production | 4 | 14 | **100 / 100%** | 43 / 92% | **85 / 88%** | **27 / 37%** | 56-77 C | 169.5 W |
| tick8192 | 1 | 6 | 50 / 100% | 21.5 / 46% | 13.5 / 81% | 6.5 / 30% | 65-82 C | 100.5 W |
| tick8192 | 2 | 7 | 94 / 100% | 45 / 81% | 78 / 94% | 25 / 30% | 68-81 C | 162 W |
| tick8192 | 4 | 15 | **100 / 100%** | 41 / 45% | **92 / 94%** | **29 / 37%** | 65-82 C | 173 W |

## Named serialization point in code

The HTTP handlers submit through a channel, but the same sole worker drains
commands, tokenizes/admits them, and performs GPU work. It only polls commands
at the top of the scheduler iteration (`worker.rs:2614-2644`); tokenization is
also on that worker (`worker.rs:4170-4220`). A synchronous prime therefore
blocks both the next GPU submission and the admission of requests arriving
during it.

The active-session loop is explicitly sequential (`crates/memra-server/src/worker.rs:3590-3621`),
and each iteration enters the engine synchronously (`worker.rs:5596-5603`):

```rust
for i in 0..active.len() {
    // ... select one mutable session ...
    match prefill_tick(&engine, &loaded, &mut px, s, budget) {
        // the loop advances only after the call returns
    }
}

let (l, _h, _x) = lm.model.prime_cache(
    engine, &chunk, s.cache.as_mut().unwrap(), s.prefill_queue.len()
)?;
```

All normal engine operations select the same main stream
(`crates/memra-runtime/src/lib.rs:83-92`). The prime epilogue computes one logits
row and downloads it (`crates/memra-engine/src/hybrid_forward.rs:1342-1346`);
that download is an explicit host barrier (`crates/memra-engine/src/lib.rs:3908-3911`):

```rust
let logits = e.matmul(&self.output, &hlast, 1)?;
Ok((e.dtoh(&logits)?, h_seed, ...))

let v = self.gpu.stream().clone_dtoh(d)?;
self.gpu.stream().synchronize()?;
```

Thus there are two related but distinct facts:

1. **Host serialization:** the worker cannot poll/admit/tokenize the next
   request or call the next session until the current `prime_cache` reaches its
   logits D2H fence.
2. **Device serialization:** even if the host fence were deferred, the current
   calls are enqueued on one in-order CUDA stream, so their kernels would still
   execute serially. Removing only the D2H wait can improve admission overlap;
   it cannot create concurrent prime compute.

The trace confirms both. Production N=4 has 48 single calls, zero batch calls,
and flat aggregate throughput while the GPU is continuously busy. The
larger-quantum control moves late arrivals into 0.8 s-spaced queue delays but
does not change the approximately four-times-solo completion wall.

Current NVIDIA documentation independently matches the source reading: a CUDA
stream is an in-order work queue, separate streams only *permit* concurrency
when resources and dependencies allow it, and `cudaStreamSynchronize` blocks
until preceding stream work completes. Sources checked 2026-08-10:

- [CUDA Programming Guide — asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [CUDA Programming Guide — streams and explicit synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html)
- [NVIDIA SMI utilization definitions](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
- [NVML GPM metric definitions](https://docs.nvidia.com/deploy/nvml-api/group__nvmlGpmEnums.html)

## Ranked mechanism candidates

These are first-gate ceilings, not forecasts or promotion claims. None is a
single-card 4x answer; preserving solo-class TTFT at N=4 would require roughly
4x aggregate compute, while the measured card is already busy.

| rank | candidate | required mechanism | evidence-bounded ceiling on this shape | effort |
|---:|---|---|---|---|
| 1 | **Checkpoint-aware cross-request prefill GEMM batching** | Tokenize/prepare outside the blocking CUDA worker so the batch exists before launch. Batch the same layer's projections/FFN at `m=sum(T)` while keeping attention, KV, absolute `seq_end`, and each affinity checkpoint boundary request-local. This needs a continuation-capable chunked batch walker; raising `MEMRA_PRIME_BATCH_MAX_T` alone cannot handle the current `ckpt_at` exclusion. | **First gate: 1.03-1.06x aggregate** (about 4.61-4.75k tok/s; c=4 wall about 3.49-3.60 s if the gain transfers). The existing Step35 PP-2 concat path measured only +2.5% at B=2 and +2.3% at B=4, T=520; integrated batch receipts quoted in `worker.rs:3481-3485` reach +5.9% at T=937 and +3.0% at T=1536. The local c=4 GPM row is SM-busy rather than DRAM-bandwidth-bound, so an N-fold weight-reuse claim is not credible. | **High** — existing exact batch machinery helps, but continuation attention/KV, per-sequence checkpoint stops, pre-admission decoupling, and full exactness gates are new. |
| 2 | **Asynchronous multi-stream request/layer overlap** | Give independent primes non-blocking streams, defer per-request logits D2H to event/final batch fences, and interleave layers or complementary stages. Replicate or partition prime slabs, CUTLASS/FP8 scratch, positions, and any other stream-unsafe resident workspace; use explicit events for dependencies. | The optimistic utilization-only bound is roughly **1/0.85 = 1.18x** at c=4 (about 5.27k tok/s, no better than about 3.15 s c=4 wall). A realistic first gate is **1.00-1.10x** because the large GEMMs already keep the device busy and two copies may not co-reside. This is the only candidate that expresses actual simultaneous kernels, but CUDA does not guarantee useful overlap merely because streams differ. | **Very high** — it breaks the engine's documented single-stream scratch invariant (`memra-engine/src/lib.rs:616-618`) and the prime-slab guard is held through the layer walk (`hybrid_forward.rs:1035-1056`). Exactness, allocator, graph, and stream-order audits are mandatory. |
| 3 | **Larger/adaptive round-robin quanta** | Choose a larger chunk when the queue is shallow and smaller quanta when an admission/decode SLO is live; retain request-level `seq_end` invariance. This is scheduler shaping, not concurrent compute. | Direct observed ceiling in this block: **1.029x / 1.021x / 1.011x** at N=1/2/4. At c=4 it changed TTFT only 3.704 -> 3.669 s while queue-wait p95 worsened 0.585 -> 2.498 s. Treat **<=1.03x** as cleanup, not the revenue fix. | **Low-medium** — the arithmetic invariance infrastructure exists, but QoS regression gates are essential. |

Current vLLM V1 documentation also treats chunk size as a TTFT/ITL tradeoff and
batches pending prefill tokens inside one token budget; it does not imply a free
second compute lane. SGLang's current server surface separately exposes maximum
prefill requests/tokens and CPU-scheduler overlap. These are architecture sanity
checks, not memra performance evidence:

- [vLLM — chunked prefill tuning](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill)
- [SGLang — current server arguments](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md)

## Receipt index and exclusions

- Derived anatomy table:
  `raw/local5090/anatomy-20260809T231431Z-cells.tsv`.
- Per-burst joined receipt:
  `raw/local5090/anatomy-20260809T231431Z-bursts.jsonl`.
- Production client/server/GPU:
  `raw/local5090/default-{client,server,gpu}-20260809T231431Z.*`.
- Large-quantum client/server/GPU:
  `raw/local5090/tick8192-{client,server,gpu}-20260809T231431Z.*`.
- Lock, hashes, entry/teardown state:
  `raw/local5090/run-20260809T231431Z.log`.
- Build: `raw/local5090/build-20260809T230528Z.log` and incremental tip checks.

Two shakeouts are retained but excluded before any measured burst. The first
used arbitrary raw token ids and produced no visible token inside four output
tokens; the second used normal text through raw completions but this
thinking-capable artifact still emitted no visible token inside eight. Both
servers completed their captured prime and stopped cleanly. Their receipts are
the `20260809T230845Z` and `20260809T231256Z` files. The measured harness uses the
real chat template with reasoning disabled.

No mechanism was implemented, no generated perf board was touched, no `nsys`
was run, and no push, merge, tag, or release was performed.
