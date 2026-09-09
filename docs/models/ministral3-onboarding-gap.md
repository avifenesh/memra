# Ministral 3: native onboarding scope, 2026-09-09

Status: unsupported. Static audit of Memra `182819614874be818ff8bef0cfaf13cb3b8051e1`. No implementation, build, execution gate or support promotion is included here.

Source: [Ministral-3-8B-Instruct-2512 config](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512/blob/5b26027e7b19eeb4b7352e1fed3926375dd2cb4d/config.json), revision `5b26027e7b19eeb4b7352e1fed3926375dd2cb4d`. The wrapper is `mistral3`, the text decoder `ministral3`. Its 34 layers have hidden size 4096, FFN 14336, GQA 32/8, head dimension 128, RMSNorm epsilon 1e-5, SiLU, no projection biases and full causal attention. `sliding_window` is null. Vision can be excluded for text-only execution, with image input refused explicitly.

This does not match `llama_dense`. That pack accepts only `Arch::Llama` and refuses non-default RoPE scaling. Neither source model type is an alias. Adding an alias or deleting scaling fields would change the program.

## Required native work

1. Add a `ministral3` pack and explicit normalized semantics. Extend the typed plan for position-dependent query scaling. Retain the existing llama window and scaling refusals. Validate supported geometry, activation, biases, scaling keys and text-only tensor ownership.
2. Implement YaRN factor 16, original context 16384, beta fast/slow 32/1, theta 1e6. Both `mscale` values are 1, making the RoPE amplitude ratio exactly 1. Existing `qwen4exp_rope_yarn` rejects these fields and the Qwen runtime derives `1 + 0.1*ln(factor)`. Reusing that amplitude would be wrong even for short prompts.
3. Implement query scaling `1 + 0.1*ln(1 + floor(position/16384))` after RoPE in native execution, prefill and decode, including absolute positions during continuation. At position 16384 this is approximately 1.0693147, versus 1 below the boundary. The operation and rewrite coverage must be explicit. Check boundary positions 16383/16384/32768 as well as ordinary short prompts.
4. Implement the checkpoint's Tekken Unicode splitter. `pre_from_split_regexes` recognizes Qwen35, Qwen2, GLM4 and DeepSeek-V3, but not this case-sensitive Unicode pattern with single-digit splitting. DictaLM's different GLM4 splitter is not transferable. Preserve fail-closed unknown-splitter behavior; never use `MEMRA_ALLOW_UNKNOWN_PRETOKENIZER` to qualify it.
5. Add the canonical Mistral template renderer and streaming parser. The template emits `[AVAILABLE_TOOLS]`, `[INST]`, `[TOOL_CALLS]name[ARGS]JSON`, and `[TOOL_RESULTS]`. It merges adjacent messages and supplies a default system prompt. Bank byte-oracle fixtures for plain, system, Unicode, tools, multiple calls, result continuation and invalid histories. Cover fragmented streaming, malformed JSON, EOS, IDs, capabilities and all supported HTTP wire formats.
6. Qualify the actual artifact. The requested pin is FP8 E4M3 with BF16 scalar `weight_scale_inv` and `activation_scale`, despite its top-level BF16 dtype. Its tensor namespace is `language_model.model.*`, with `language_model.lm_head.weight`. Prefix lookup support exists, but does not establish numerical compatibility. Source-FP32 oracle captures must state whether they expand this FP8 checkpoint or use a separately pinned BF16 sibling. Do not silently change checkpoint or activation semantics.

The source math is visible in [Transformers Ministral3Attention](https://github.com/huggingface/transformers/blob/e8bcd79c4c77a529bc34d6e19f7d718164b7fa6a/src/transformers/models/ministral3/modeling_ministral3.py) and [YaRN parameter construction](https://github.com/huggingface/transformers/blob/e8bcd79c4c77a529bc34d6e19f7d718164b7fa6a/src/transformers/modeling_rope_utils.py). These are offline oracle references, not runtime dependencies.

## Qualification boundary and estimate

Estimate: 6–12 agent-hours for implementation, artifact preparation and fixture work, plus 2–6 rented 5090 GPU-hours for build and qualification, including iteration. This is a planning range, not a measured runtime. The GPU stage follows a complete rental comparison and real CUDA allocation acceptance.

Run config/tensor census, tokenizer/template parity, tiny and checkpoint parity, rewrite parity, real sampled tool round trip, eight-turn continuation and context-boundary checks. Source-FP32 parity tooling exists in `memra-cli`, but requires a working native runner and a pinned reference bundle; its existence is not a Ministral pass. Native eager support must reach the real serving path before admission. No external or reference-only server is a delivery substitute.

This exceeds a bounded config/template/refusal/test change. Stop at the scope until a full implementation lane is authorized. No PR or `NativeReference` claim follows from this document. No gates or builds ran on the local rig. `MEMRA_SKIP_PERF_CI=1` is used when pushing this documentation branch under the current rig restriction; required CONTRIBUTING execution evidence remains absent.
