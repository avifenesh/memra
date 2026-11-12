# Model cards

One card per model: what it is, its pack state and measured serving route, how to run it, current
numbers, limits and deeper sources. [MODELS.md](../MODELS.md) is the index and defines the two
axes; commands live in the [cookbook](../COOKBOOK.md).

Covered by the tag battery on every release (the two `tools/release-roster.tsv` checkpoints):

- [Qwen3.8-27B](qwen38-27b.md)
- [Ornith-1.5-35B-A3B](ornith15-35b-a3b.md)

Measured serving routes with no standing gate:

- [Qwen3.5-9B](qwen35-9b.md)
- [Qwen3.6-27B](qwen36-27b.md)
- [Qwen3.6-35B-A3B](qwen36-35b-a3b.md)
- [Qwen-AgentWorld-35B-A3B](qwen-agentworld-35b-a3b.md)
- [Ornith-1.0-9B](ornith10-9b.md)
- [Ornith-1.0-35B](ornith10-35b.md)
- [Gemma-4 12B](gemma4-12b.md)
- [Gemma-4 26B-A4B](gemma4-26b-a4b.md)
- [Gemma-4 31B](gemma4-31b.md)
- [Step-3.7-Flash 196B-A11B](step37-flash.md) (two or more cards)
- [DeepSeek-V4-Flash](deepseek-v4-flash.md) (two cards, own serve door)
- [GLM-5.3-Flash](glm53-flash.md): its `glm5_next` pack is NativeReference <!-- support: glm5_next-reference -->
- [Qwen3-Embedding-8B](qwen3-embedding-8b.md) and [Qwen3-Reranker-8B](qwen3-reranker-8b.md) (capture models)

Engine cells or reference executor only, no gated serving route:

- [Gemma-4 E4B](gemma4-e4b.md): engine cells pass, `memra-server` serving is not gated
- [DeepSeek-V4.1-Flash](deepseek-v41-flash.md): NativeReference, reference executor only <!-- support: deepseek_v41-reference -->
- [Hy3](hy3.md): NativeReference for the canonical plan and the W4A16 profile; the 2026-08-30 four-card gates are reported, not tracked <!-- support: hy3-bf16-reference,hy3_nvfp4-modelopt-w4a16 -->
- [Qwen3.8-Flash-Next](qwen38-flash-next.md): bring-up only

A card is specific to its checkpoint, quantization, drafter and hardware. It promises nothing about
an adjacent checkpoint.
