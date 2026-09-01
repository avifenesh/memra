---
license: mit
base_model:
  - zai-org/GLM-5.3-Flash
base_model_relation: quantized
quantized_by: Avifenesh
pipeline_tag: text-generation
library_name: memra
language:
  - en
  - zh
tags:
  - nvfp4
  - fp4
  - 4-bit
  - modelopt
  - w4a16
  - quantized
  - memra
  - glm5_next
  - moe
  - blackwell
  - conversational
  - tool-calling
---

# GLM-5.3-Flash NVFP4

NVFP4 (4-bit e2m1 weights with an FP8-e4m3 per-16 scale plane, 4.5 bits/element)
weight-only mint of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash),
produced with [NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
0.46.0 and gated against an unfused f32 reference executor before publication.

Built and validated with [**memra**](https://github.com/avifenesh/memra), a from-scratch
Rust and CUDA inference engine for RTX Blackwell, by **tiyuvta**
([inference.tiyuvta.ai](https://inference.tiyuvta.ai)).

> **This is a bring-up artifact, not a hosted offering.** We do not serve
> GLM-5.3-Flash to customers, there is no endpoint and no model id for it, and every
> number below is a dated bench measurement on named hardware rather than a product
> claim. The brands above say who made this and what ran it.

- **190.7 GB** across 20 safetensors shards, 38,770 tensors: 37,338 quantized, 1,432 kept.
- Source is the vendor's **BF16 twin**, not their FP8 release. Never quantize from a
  quant when the full-precision twin ships.
- Upstream technical report: [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763)
  (arXiv 2602.15763).

## How it was made

| | |
|---|---|
| Source | `zai-org/GLM-5.3-Flash-BF16` @ `f12e0fe1f6b2ea274c11a569582edfd99d993c5e` (656 GB) |
| Tool | `nvidia-modelopt` 0.46.0, `W4A16_NVFP4` |
| Scheme | weight-only. e2m1 weights, dynamic per-16 block scales in e4m3, per-tensor f32 macro scale. Activations untouched |
| Calibration | **none, and that is not a shortcut**. W4A16 NVFP4 derives its block scales from the weight tensor itself, so modelopt runs no calibration forward pass for this scheme. No prompts are rendered and no activation statistics enter the checkpoint |
| Packaging | streamed tensor by tensor, because 656 GB of BF16 does not fit the box whole. The quantization math is `NVFP4QTensor.quantize` from 0.46.0, the same function modelopt's own `export_hf_checkpoint` calls, so the emitted weight bytes match an official export. Only the shard writing, index and config are ours, mirroring the 0.46.0 export code |

Per tensor `W [out, in]`:

```
weight_scale_2 (f32 scalar)      = amax(|W|) / (6 * 448)
weight_scale   (e4m3, [out, in/16]) = block_amax / (6 * weight_scale_2), zeros -> 1.0
weight         (u8,   [out, in/2])  = e2m1 codes, element 2i in the low nibble
```

One trap worth repeating: `NVFP4QTensor.quantize(..., try_tensorrt=True)` on a box with
`tensorrt_llm` installed hands back CUTLASS-swizzled scales rather than the modelopt
layout. This mint uses the default `try_tensorrt=False` and asserts the scale shape and
dtype on every tensor.

## Precision split

GLM-5.3-Flash is a hybrid stack: 45 decoder layers plus one MTP layer, 34 of them
KDA linear-attention and 11 DSA (MLA with a sparse indexer), MoE on 42 layers with 288
routed experts plus one shared, and 3 dense layers. The split mirrors the vendor's own
FP8 exclusions and the tensor census, not a guess about which layers "look big".

| group | quantized |
|---|---|
| MoE routed experts, shared experts, dense MLPs | yes |
| MLA projections (`q_a`/`q_b`/`kv_a`, `o_proj`) | yes |
| Every KDA tensor (`b_proj`, `f_a`/`f_b`, `g_a`/`g_b`, q/k/v projections, short convs) | **no** |
| `kv_b_proj` | **no** |
| mHC hyper-connection tensors | **no** |
| Router gates and the `e_score_correction_bias` | **no** |
| Norms, `embed_tokens`, `lm_head` | **no** |
| Vision tower | **no** |

The keep list ships in both dialects in `config.json`: `modules_to_not_convert` and
compressed-tensors `ignore`, 628 entries each. That redundancy is deliberate. Writing it
in only one dialect made it invisible to a loader that reads the other, which silently
re-encoded the large KDA projections to 8-bit against the intent of the split. Stating
your own fact in the dialect the reader speaks is cheaper than teaching every reader a
new dialect.

## The gate this passed

Same engine (memra's unfused f32 streaming reference executor), same tokens, full
last-position vocabulary row of 154,880 logits, all three artifacts run on one box:

| comparison | argmax | top-k rank-identical | max_abs | mean_abs |
|---|---|---|---|---|
| this NVFP4 mint vs its BF16 source | MATCH | top-3 | 3.117 | 0.534 |
| vendor FP8 vs the same BF16 source | MATCH | top-3 | 3.489 | 0.490 |
| this NVFP4 mint vs vendor FP8 | MATCH | top-5 | 4.184 | 0.705 |

**The middle row is the point.** An absolute logit delta has no interpretation without a
same-instrument reference point, and a 4-bit mint compared against an 8-bit artifact
measures two error sources blended together rather than one. Compared against its own
full-precision source, this mint's deviation is comparable to the vendor's own 8-bit
quantization deviation, at half the bit width.

The gate was re-run on the exact bytes published here, not inherited from a run over a
directory with the same name. Publishing a mint that has only a family-level gate is how
you ship a quantization that scrambles rankings while every smoke test stays green.

**What this gate does not claim:** serving accuracy over long generations, long-context
behaviour, sampled decoding quality, or any engine's fused quant kernels. The reference
runner dequantizes to f32; a serving path riding 4-bit matmul kernels is a separate gate.

## Why bring-up found things, and what it found

Every defect in this bring-up was caught by an instrument built to disagree loudly:
memra's unfused f32 reference executor run against the engine on the same input, byte
oracles over rendered prompts and tokenizer output, and refusals written to fire rather
than to cope. Not one was caught by the engine looking unhealthy. A wrong forward pass
on a large language model does not crash and does not obviously degrade. It answers
fluently, at full speed, off distribution, and that is why the instrument has to be a
second implementation rather than a closer look at the output.

Twelve of them landed in this lane, so here they are rather than as a number you have
to take on faith:

| # | defect | how it presented |
|---|---|---|
| 1 | MLA projection names unresolved on the engine's HF loading path | a refusal that fired, in a place where the same class of miss stays silent |
| 2 | 3-D quantized operands accepted where the kernel could not honour them | silently wrong expert math |
| 3 | KDA tensors reaching a quantized operand path they are excluded from | silently wrong linear-attention state |
| 4 | Pre-activation clamped SwiGLU present in the plan, not wired in the forward | wrong activation on every MLP, fluent output |
| 5 | The mint's own keep list invisible to the loader (written in one dialect, read in another) | large KDA projections silently re-encoded to 8-bit |
| 6 | Router `e_score_correction_bias` missing from the ggml-to-HF name map | routing drift, census still green |
| 7 | Shared-expert projections missing from the same map (singular vs plural spelling) | the shared expert dropped from every MoE layer |
| 8 | Only the first declared eos id honoured | generation never stopped |
| 9 | Chat template pattern-matched to ChatML | every surface 200, every prompt off distribution |
| 10 | `reasoning_effort` accepted and ignored | rendered bytes identical with and without it |
| 11 | Effort clamp written for a three-rung family ate this model's top rung | the highest effort level unreachable |
| 12 | Pre-tokenizer split rule not the glm4 one | token boundaries diverged from the checkpoint's own tokenizer |
| 13 | Fast chat-render path dropped `reasoning` from replayed turns | two render paths disagreeing on a dialect that replays it |

That is thirteen rows. Twelve of them changed what the model produced; row 1 refused
loudly instead, which is the outcome you want and the reason it is listed separately.

The bring-up log these come from is public, defect by defect, with the before and after
logit rows:
[`research/glm53-flash-bringup-20260827/BRINGUP.md`](https://github.com/avifenesh/memra/blob/0decc48ca1210205809633cc925febc87dd3094e/research/glm53-flash-bringup-20260827/BRINGUP.md).
The mint receipts and the gate coverage for the exact bytes in this repo are in the
same directory.

Two are worth expanding, because they generalize past this model:

**Two missing rows in a name map.** The engine's ggml-to-HF name map lacked entries for
the router's `e_score_correction_bias` (nested under the router module, not where the
DeepSeek-V3 lineage puts it) and for the shared expert's `ffn_{gate,up,down}_shexp`
weights (the generic map spells the shared expert singular; this checkpoint spells it
plural). Effect: the shared-expert branch was silently dropped from every MoE layer, and
router selection drifted. The tensor census passed the whole time, because a census
exercises the contract's names, and the contract's names were right. The property no
existing gate covered was name resolution on the engine's own loading path.

**The chat template is not ChatML, and a ChatML lookalike returns 200.** Serving this
model through a ChatML renderer produced fluent text at every surface: chat completions,
streaming, the Anthropic and OpenAI translation surfaces, and tool calls. All of it ran
off distribution. The `<|im_start|>` and `<|im_end|>` frame is not in this checkpoint's
special vocabulary at all, so it tokenized as ordinary text. Tool calls only parsed
because the model obeyed the foreign instruction block it was handed; the native wire had
no parser branch and would have surfaced verbatim as content. `reasoning_effort` was
accepted and ignored, provable because a request with `effort: "low"` hit the prefix
cache at 288 of 288 tokens against the identical request without it: zero rendered bytes
differed.

The native dialect, for anyone else serving this checkpoint:

- `[gMASK]<sop>` framing, with `<|system|>` / `<|user|>` / `<|assistant|>` / `<|observation|>` role markers.
- Tool calls: `<tool_call>NAME<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>`.
- Tool results come back in an `<|observation|>` block, with run grouping and id-keyed re-ordering.
- `reasoning_effort` has a real rung above `high`. A clamp written for a three-rung family eats it.

Read the template out of `chat_template.jinja` in this repo and render through it. Do not
pattern-match a template to a family you already support: the failure mode is a 200 with
good prose.

## Running it

This artifact runs on [memra](https://github.com/avifenesh/memra). It is a
modelopt-quantized `glm5_next` checkpoint, so a stock `transformers` load is not the
path, which is why `library_name` here says memra rather than transformers: a
transformers usage snippet on this repo would be a button that does not work.

```bash
MEMRA_MODELS="zai/glm-5.3-flash=/path/to/GLM-5.3-Flash-NVFP4" \
MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=12000 MEMRA_ST_PINNED=1 \
MEMRA_CTX=8192 MEMRA_COMPAT=openai MEMRA_ADDR=127.0.0.1:8000 \
memra-server
```

190.7 GB does not fit a single 96 GB card, so the routed experts stream from host memory
through an SLRU residency cache while the trunk stays resident. `MEMRA_ST_PINNED=1`
selects pinned-DMA staging over the pageable path, which is worth roughly 2.4x on this
model.

### Measured decode, so the offload cost is not a mystery

Bench measurement, 2026-08-28, on one RTX PRO 6000 Blackwell Server Edition (96 GB, 500 W)
of a two-card box, card 0 only, single stream, greedy, `NVIDIA_TF32_OVERRIDE=0`,
12,000 expert slots, trunk resident at 79.6 GB:

**20.3 tok/s**, median of 3 reps, reproduced across six boots.

Greedy is used here because it is byte-deterministic and therefore checkable, not because
it is how anyone should run the model. Where the time goes, solved from measurement
rather than modelled: about 33 ms per token of resident work and the rest expert staging
over PCIe, at roughly 53 GB/s on the pinned path against about 10 GB/s pageable. The
resident-traffic roofline for this configuration is 63 tok/s, so this number has room in
it and is not a property of the artifact. It is a bring-up figure on one configuration.

## Context, and where the practical wall actually is

The upstream checkpoint declares `max_position_embeddings: 1048576`. That is **the
vendor's architecture figure**, and it is repeated here as one, not as something this
mint delivers. **We have not gated or run this artifact anywhere near that context**,
and nothing in this repo should be read as a 1M-context serving claim.

The reason is worth stating, because it is checkable from `config.json` in this repo
rather than something you have to take on trust. The DSA layers score pooled keys:
`index_kpool` is 4, so a call over `t` query tokens against `t_kv` keys allocates a
score plane of `t * (t_kv / 4)` f32 values. Under a monolithic prefill where
`t = t_kv = N`, that plane is **N squared bytes**, per MLA layer, per call:

| context | transient score plane |
|---|---|
| 8,192 | 67 MB |
| 50,000 | 2.5 GB |
| 262,144 | 68.7 GB |
| 1,048,576 | 1.10 TB |

Our bring-up hit CUDA out-of-memory on a 96 GB card in the region that arithmetic
predicts. Note where the wall falls: 262,144 is already past a single 96 GB card, so
it sits well below the headline number rather than at it.

Two things this is not. It is not a quantization artifact, since the plane is f32
scratch whose size depends only on context and the pooling factor, so an FP8 or BF16
copy of this model has the same wall. And it is not a bigger-card problem in any
practical sense: this is per-call transient scratch, not a persistent allocation you
can amortize, so chunked prefill is the shape that moves it, not more VRAM.

If you need long context on this model today, that is unsolved work, not a flag.

## Files

| file | what it is |
|---|---|
| `model-*-of-00020.safetensors` | the weights, 190.7 GB, NVFP4 triples plus kept tensors |
| `model.safetensors.index.json` | 113,446 weight-map entries = 1,432 kept + 3 x 37,338 quantized (`weight`, `weight_scale`, `weight_scale_2`) |
| `config.json` | `glm5_next`, `quantization_config` with `quant_algo: W4A16_NVFP4` and the keep list in both dialects |
| `hf_quant_config.json` | modelopt's own quant config sidecar |
| `chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json` | the checkpoint's own tokenizer and template, unmodified |
| `generation_config.json` | unmodified, and it declares three eos ids. Honour all of them |

The vision tower is present in the source architecture and is not exercised by this
lane's gates. Text is what was gated.

## Attribution and licence

MIT, following upstream [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).
All model capability belongs to the GLM-5 team; this repository contributes a
quantization and the receipts that it is faithful to its source.

```bibtex
@misc{glm5team2026glm5vibecodingagentic,
      title={GLM-5: from Vibe Coding to Agentic Engineering},
      author={GLM-5-Team},
      year={2026},
      eprint={2602.15763},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.15763},
}
```
