# Decode-priority global prefill budget — RTX 5090 card 0

## Verdict

**NO-GO.** Moving interactive prefill after decode and capping it with one scheduler-global
token budget is not Sarathi-style stall-free batching in Memra today: decode and prefill still
run as two separate model invocations. On the served Qwen3.8-27B GGUF shape, both tested global
policies reduced throughput and worsened hit-stream ITL. The runtime changes were removed; no
default or release carries them.

The next viable rung is a Memra-owned mixed model pass that co-schedules decode rows and one
prefill chunk inside the same weight walk (including mixed-length attention/state handling), or
prefill/decode disaggregation. Reordering two serial calls cannot consume decode's arithmetic
intensity slack.

## Exact scope

- Hardware: NVIDIA GeForce RTX 5090 Laptop GPU, physical card 0 only; every GPU run held
  `/tmp/memra-5090.lock` and set `CUDA_VISIBLE_DEVICES=0`.
- Parent: `b40bd07c82fdbc5f5c200c8d3b0ab3310629c1f8`.
- Iteration 1: `14a3dcf3dad2bf8fc68a64406182a9715b85a5f6`, one rotating single-session
  prefill action after decode.
- Iteration 2: `4ccfd9075c4a1d307fbfad5b36f980b4a0041c09`, one water-filled same-model
  varlen prefill batch after decode, with per-request single/batch program pinning and B=1 batch
  tails.
- Model: `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`; spec disabled to isolate plain mixed scheduling;
  context 8192.
- Workload: 90% exact-prefix hits / 10% cold, 4,860 prompt tokens + 60 output tokens,
  temperature 0, widths 1/4/8/12/16. The card-fit corpus keeps two hot prefixes in a 2 GiB
  prefix cache; request semantics and cold ratio are unchanged from the frozen mixed90 harness.
- This is an exploratory N=1 mechanism verdict, explicitly not an N=5 promotion receipt. The
  direction is large and consistent enough to kill the arm before spending four more pairs.

## Aggregate iteration results

Strict protocol: 100/100 requests valid on each binary, 0/100 cross-binary output-hash
mismatches, ten clean cells, 250 ms card-0 telemetry, and 40 candidate ticks containing both
decode and prefill. The candidate never issued more than one prefill action per tick and never
exceeded 1,039 tokens while more than one session was active.

| concurrency | output tok/s parent → candidate | delta | hit ITL p95 parent → candidate | delta | hit TTFT p95 parent → candidate |
|---:|---:|---:|---:|---:|---:|
| 1 | 37.502 → 36.374 | -3.01% | 21.554 → 21.780 ms | +1.05% | 5.232 → 3.657 ms |
| 4 | 79.342 → 73.001 | -7.99% | 137.490 → 155.887 ms | +13.38% | 33.144 → 30.827 ms |
| 8 | 88.071 → 80.326 | -8.79% | 155.275 → 175.093 ms | +12.76% | 38.569 → 50.752 ms |
| 12 | 84.041 → 77.440 | -7.85% | 132.136 → 148.254 ms | +12.20% | 64.130 → 68.715 ms |
| 16 | 82.865 → 80.267 | -3.14% | 201.706 → 207.395 ms | +2.82% | 11,939.006 → 12,302.858 ms |

Cold TTFT also regressed at every width (+9% to +19%). Iteration 1 pointed in the same direction:
one full pair measured roughly -5% to -11% output throughput from c=4 through c=16 with worse
hit ITL. It was stopped after the first pair and replaced by the requested aggregate form.

## Why it lost

The parent processes two simultaneous 1,024-token cold chunks in fewer, larger concat-prime
iterations. The global arm preserves the same total prefill tokens but spreads them over more
scheduler ticks. Because Memra executes decode and prefill sequentially, every extra tick pays
another prefill call's fixed costs and the next token interval still contains the preceding
prefill wall. Decode-first ordering changes which side of an emitted token sees the stall; it does
not overlap the work or remove it.

This is the missing distinction from vLLM V1/Sarathi-Serve: their token budget constructs one
mixed execution batch, so decode rows piggyback on the prefill chunk's compute. Memra's current
`decode_step_batch` followed by `prime_cache_batch` is two weight walks and has no such slack
reclamation.

## Correctness and raw receipts

- Exact candidate-1 correctness battery (GPU 0): 411 server tests, kernel-check 107 cells,
  prime gate, run-spec K=1..8, run-gen/verify, graph stress+canary, serve-smoke 0 failed, c=64
  stress, acceptance and spec/cache-hit teeth all green. Log SHA-256:
  `9d61bd5d1d356755f33c9570e38db2b5ff61862d6f5764e599c23b6260cb9e03`.
- Aggregate N=1 root:
  `/home/avifenesh/projects/receipts/global-prefill-budget-20260825/mixed-ab-aggregate-n1-card0/`.
  `aggregate.json` SHA-256:
  `9b0df8630c231c8f0774f0dcbfdbc80dafcdc02b2ee289f14520ffda3f7a0b59`.
  Root manifest SHA-256:
  `290a6cb91dc477f9a69901f717e80074678dffe036f9c7212d8bfb9042e1f05b`.
- The 4 GiB frozen-cache attempt is retained but excluded: parent prime transients OOMed from
  c=8 because the 24 GiB card could not hold the eight-entry cache plus model and transient.
  The two-entry/2 GiB rerun is the valid receipt above.
- The first harness attempt stopped before scoring on a `strings | grep -q` SIGPIPE false failure;
  it is retained as harness history, not evidence.

## Decision

Do not merge either sequential global-budget implementation. Preserve the measurements and fund
the actual mixed-forward seam; re-use this workload as its first structural/performance gate, then
require N=5 before any default.
