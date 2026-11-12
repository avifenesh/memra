# Long context and prefix reuse

For readers whose prompts run to hundreds of thousands of tokens, or whose conversations reuse a
long shared prefix.

## Request shape

Two shapes with different useful defaults at the same token count: a long-running conversation that
extends its own prefix (cache-friendly), and a fan-out over unrelated long documents (every prompt
cold). On the Gated DeltaNet hybrids the prefix cache serves the first shape only, since the
recurrent state exists only at an entry's end boundary.

## Recommended model and card

| Model | Card | Context measured | Run it |
|---|---|---|---|
| [Qwen3.8-27B](../models/qwen38-27b.md) | RTX PRO 6000 | 262,144 | [cookbook](../COOKBOOK.md#qwen38-27b) |
| [Qwen3.8-27B](../models/qwen38-27b.md) | DGX Spark (GB10) | 262,144 (cold TTFT 313.9 s at 259,951 tokens) | [GB10 card](../rigs/dgx-spark.md) |
| [Ornith-1.5-35B-A3B](../models/ornith15-35b-a3b.md) | RTX PRO 6000 | 262,144 | [cookbook](../COOKBOOK.md#ornith-15-35b-a3b) |
| [DeepSeek-V4-Flash](../models/deepseek-v4-flash.md) | 2x RTX PRO 6000, TP/EP | 1M plain, 500k with DSpark | [model card](../models/deepseek-v4-flash.md) |

## Settings that matter

- `MEMRA_CTX`: defaults to the checkpoint's declared context. Reserving the maximum costs KV for
  clients that never send it; set the workload's ceiling and send `max_tokens`.
- `MEMRA_PREFIX_CACHE_MB`: derived from boot free memory, served context and the session peak when
  unset; `0` disables. Unified-memory devices size it differently ([GB10 card](../rigs/dgx-spark.md)).
- `MEMRA_KV_HOST_MB`: the pinned host tier, off by default, clamped to 60% of MemAvailable.
- `MEMRA_MAX_SESSIONS`: fewer sessions leave more KV per session.

Flag detail: [FLAGS.md](../FLAGS.md).

## How to validate

Measure cold TTFT at the real prompt depth and a warm turn on the real turn shape; the boot lines
`[prefix-cache]` and, on GB10, `[deadline]` state the derived sizes.

## Read next

[Serving: prompt caching](../SERVING.md#prompt-caching-cross-request-prefix-cache--2026-08-02),
[PERFORMANCE.md](../PERFORMANCE.md).
