# Inkling 2 Mini onboarding (opened 2026-11-21)

Status: open. This directory holds the lane's pinned sources, the header census and the
semantics note. No pack, plan or executor exists yet, so nothing here is a support claim and no
gate has run. No GPU was used.

## Scope

Onboard `thinkingmachines/Inkling-2-Mini` as a native memra family (`inkling2`, aliases
`inkling2_loop` and `inkling-2-mini`) and tune it on one 24 GB sm_120a card.

- Text only first. Image parts are refused until a vision lane; the vision tower's 422 tensors
  are still owned by the tensor contract.
- The weights stay as shipped: NVFP4 experts on layers 0 to 19, MXFP8 experts on layers 20 to 23,
  BF16 and F32 elsewhere. No requantization.
- The receipts of later steps (inspect, tiny, checkpoint, serve) land in this directory.

## Sources

[`sources.lock.json`](sources.lock.json) pins every source:

| Source | Pin | Role |
|---|---|---|
| Checkpoint | `thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7`, 12 files with sha256 and size | the semantic source |
| llama.cpp | `29b52d398b294509746ec0a3f28948a3107c0213` (`src/models/inkling2.cpp`, `inkling2-loop.cpp`) | second implementation of the float-activation trunk, for triage |
| transformers | PR 51050 at head `6173a2f679ff4ed393afc499aefc569da7e68bc1` (open at capture) | oracle source for the float-activation bring-up class |
| vLLM image | `vllm/vllm-openai:inkling-2-mini`, index digest `sha256:c2dc759f...` | only public source of the drafter and of the declared activation-quantized class, the default |

External engines are oracles and baselines. Their results are never memra support evidence.

## Census

[`census.py`](census.py) reads the safetensors headers and `hf_quant_config.json` of a local
checkpoint directory. No weight bytes are read.

```bash
python3 research/modelplan-onboarding-inkling2mini-20261121/census.py <checkpoint-dir> \
  research/modelplan-onboarding-inkling2mini-20261121
# 1298 tensors, 109 classes
```

- [`census.tsv`](census.tsv): one row per tensor (file, name, dtype, shape, bytes, declared
  quant group). Rows sha256 `3752c60c4d5facbc9cbd089e76b710c2f1523c794ffc3776c2628dde9f3aa41c`.
- [`census-classes.tsv`](census-classes.tsv): the same census as 109 name classes. The script
  expands the classes and refuses to write unless they reproduce every row.

| File | Tensors | Data bytes |
|---|---:|---:|
| `model-00001-of-00001.safetensors` (the one indexed shard) | 1,244 | 19,525,870,752 |
| `mtp.safetensors` (drafter, not in the index) | 54 | 635,795,744 |

| Class | Tensors | Bytes | Dtype |
|---|---:|---:|---|
| NVFP4 expert payload, layers 0-19 (`w13 [128,2304,1152]`, `w2 [128,2304,576]`, two codes per byte) | 40 | 10,192,158,720 | U8 |
| NVFP4 expert scales (one e4m3 per 16 inputs per row, `_amax [128]`, `_outer_scale_denominator [1]`) | 120 | 1,274,040,480 | F8_E4M3, F32 |
| MXFP8 expert payload, layers 20-23 (`w13 [128,2304,2304]`, `w2 [128,2304,1152]`) | 8 | 4,076,863,488 | F8_E4M3 |
| MXFP8 expert scales (one u8 exponent per 32 inputs per row) | 8 | 127,401,984 | U8 |
| attention projections (`wq_du`, `wk_dv`, `wv_dv`, `wo_ud`, `gate_attn`, `wr_du`) | 144 | 1,288,175,616 | BF16 |
| embed and unembed `[201024,2304]` | 2 | 1,852,637,184 | BF16 |
| vision tower and projector (owned, not executed) | 422 | 636,512,256 | BF16 |
| loop controllers and the two cross-pass query tensors | 146 | 57,065,472 | BF16, F32 |
| router weight and bias | 48 | 14,168,064 | BF16, F32 |
| norms, clips, sink, relative bias | 154 | 3,873,792 | BF16 |
| short convolutions | 104 | 2,457,600 | BF16 |
| block-attention-residual queries | 48 | 516,096 | F32 |

## Decisions

1. Text only first; image parts fail closed.
2. Weights stay as shipped.
3. The default numeric class, and the only support target, is the program the checkpoint
   declares as required in `hf_quant_config.json`: NVFP4 W4A4 with per-token amax and
   four-over-six scale selection on the layer 0-19 experts, MXFP8 W8A8 on the layer 20-23
   experts. Its only public source is the vLLM image. The `float-activation` class (exact code
   times scale dequantization with f32 or bf16 activations, what the transformers PR and
   llama.cpp run) is a bring-up and oracle step: never the default and never a support claim.
   The two classes are named and never presented as one another. No GGUF-style quant formats
   (Q*_K, IQ*) enter this family's path.
4. The base is `main`. Nothing stacks on unmerged branches.
5. The family is `inkling2`; receipts live here.

## Semantics

[`SEMANTICS.md`](SEMANTICS.md) states the program as read from the config, the two reference
implementations and the bytes, and lists the facts that are still open.

## Next

Config parsing, the typed 48-slot plan, the tensor contract for all 1,298 tensors and the
`inkling2` pack at support none, with `memra model inspect` receipts in `inspect/`.
