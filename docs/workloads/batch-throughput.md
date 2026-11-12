# Batch throughput

For readers who care about aggregate completion rate across many concurrent requests more than about
any single request's latency.

## Request shape

Many independent requests at once, often on batch-class API keys. The scheduler batches decode
across sessions; speculative decoding sheds to plain batching as concurrency rises.

## Recommended model and card

| Model | Card | Current batch receipt |
|---|---|---|
| [DeepSeek-V4-Flash](../models/deepseek-v4-flash.md) | 2x RTX PRO 6000, TP/EP | greedy c16 336-340 tok/s aggregate, 16 lanes (2026-09-28) |
| [Qwen3.8-27B](../models/qwen38-27b.md) | DGX Spark (GB10) | four penalised sampled requests at once, 41.90 tok/s (2026-11-10) |
| [Qwen3.6-35B-A3B](../models/qwen36-35b-a3b.md) | DGX Spark (GB10) | four penalised sampled requests at once, 131.41 tok/s (2026-11-10) |

The fastest single-stream route is not always the fastest batch route; read the receipt for your
concurrency.

## Settings that matter

- Batch-class keys (`MEMRA_API_KEYS`, lane `batch`) or the `x-lane: harvest` header put requests on
  the harvest lane, which sheds under interactive load with a retryable 429.
- `MEMRA_DECODE_BATCH_CAP`: automatic per model class; explicit values clamp.
- `MEMRA_MAX_SESSIONS` and `MEMRA_MAX_QUEUE_DEPTH`: the admission ceiling and the queue bound.
- `MEMRA_SPEC_GATE_LOW` / `_HIGH`: where speculative admission hands over to plain batching.

Flag detail: [FLAGS.md](../FLAGS.md).

## How to validate

Measure the real HTTP surface at the intended concurrency and request mix, not a kernel benchmark.
`tools/serve-smoke.sh` checks the surface; `/yield/metrics` reports per-lane QoS and step p50/p99.

## Read next

[Serving](../SERVING.md), [PERFORMANCE.md](../PERFORMANCE.md), [TESTING.md](../TESTING.md).
