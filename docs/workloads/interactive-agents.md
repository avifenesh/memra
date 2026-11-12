# Interactive agents

For readers serving coding or tool-using agents: one user per session, multi-turn, long shared
history, tools and reasoning on.

## Request shape

Single-stream latency matters more than aggregate throughput. Turns extend the previous prompt, so
the prefix cache carries most of each prompt. Clients send tools and expect `tool_calls` back.

## Recommended model and card

| Model | Card | Run it |
|---|---|---|
| [Qwen3.8-27B](../models/qwen38-27b.md) | RTX 5090 Laptop (24 GB envelope) or RTX PRO 6000 (262k context) | [cookbook](../COOKBOOK.md#qwen38-27b) |
| [Ornith-1.5-35B-A3B](../models/ornith15-35b-a3b.md) | RTX PRO 6000 | [cookbook](../COOKBOOK.md#ornith-15-35b-a3b) |

Both are the release-roster checkpoints, so every tag runs their battery.

## Settings that matter

- `MEMRA_CTX`: defaults to the checkpoint's declared context; set it to the workload on a 24 GB card.
- `MEMRA_MAX_SESSIONS`: the interactive cap (default 64); size it with the context.
- `MEMRA_PREFIX_CACHE_MB`: derived from free memory when unset; `0` disables it.
- `MEMRA_COMPAT=openai`: OpenAI shapes on `/v1/completions` (chat routes are always OpenAI-shaped).

Speculative decoding is on by default when a drafter is attached; the spec gate sheds to plain
batching under concurrency. Flag detail: [FLAGS.md](../FLAGS.md).

## How to validate

Run `tools/serve-smoke.sh` against the configuration, then a real tool loop in the client you will
use. OpenAI Chat Completions, OpenAI Responses and Anthropic Messages all translate onto the same
chat core ([API-SURFACES.md](../API-SURFACES.md)).

## Read next

[Serving](../SERVING.md), [long context and prefix reuse](long-context.md),
[the cookbook](../COOKBOOK.md).
