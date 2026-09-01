# Long context and prefix reuse

| | Recommended use |
|---|---|
| **Start with** | RTX PRO 6000 Blackwell and a Qwen3.8 or Ornith 1.5 configuration sized to the workload |
| **Prioritize** | KV capacity, prefix-cache size, session count, and admission behavior |
| **Avoid** | Reserving the model's maximum context when clients never send it |
| **Read next** | [Serving: prompt caching](../SERVING.md#prompt-caching-cross-request-prefix-cache--2026-08-02), [Performance](../PERFORMANCE.md) |

Measure the actual prompt and turn shape. A cache-friendly long-running conversation and a fan-out
batch over unrelated documents have different useful defaults even at the same token count.
