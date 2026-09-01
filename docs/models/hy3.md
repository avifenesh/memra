# Hy3

Support state: **NativeReference** for the canonical plan and official BF16 safetensors;
**NativeQualified** for the exact all-expert ModelOpt W4A16 artifact described below. The NVFP4
qualification binds the artifact manifest, serialized plan, Memra runtime binary, and four-card
RTX PRO 6000 Blackwell receipts. `NativeTuned` remains pending: the faster PP-4 wavefront stayed
off after failing its serial-vs-wave logit-identity gate.

## Artifact contract

- Semantic source:
  `tencent/Hy3@a960ebc3da325ba167f069f76c41eb62c9280d22`.
- Primary RTX candidate: Memra's streaming NVIDIA ModelOpt 0.46.0 mint from the pinned source,
  using ModelOpt commit `43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a`.
- Existing comparison controls:
  `kodelow/Hy3-NVFP4-W4A16@4f7e1f02f32f4662bdc47f1237f34995088867a9` and
  `LibertAIDAI/Hy3-NVFP4@8a805d687c4bf02b364af15db5ee85e43ed998bf`.
- Model pack: `hy3_nvfp4` (aliases `hy3-nvfp4`, `hy_v3_nvfp4`).

The NVFP4 contract is deliberately exact: every routed-expert projection in layers 1 through 80 is
NVFP4, including all 576 MTP expert weights. Attention, router, shared MLP, dense layer 0,
embeddings, output head, norms, biases, and non-expert MTP tensors remain checkpoint BF16/F32.
There is no MTP-BF16 profile.

Both accepted storage encodings preserve the same semantic program:

- compressed-tensors `weight_packed` + `weight_scale` + divisor
  `weight_global_scale`;
- ModelOpt packed `weight` + `weight_scale` + multiplier `weight_scale_2`.

For ModelOpt fused MoE GEMM1, each expert's gate and up projections share the larger of their two
per-tensor `weight_scale_2` values; the artifact gate checks all 15,360 pairs bit-for-bit. This is
the pinned NVIDIA ModelOpt fused-MoE recipe, not a deployment-loader repair after quantization.

The header contract requires exactly one macro scale, validates its scalar F32 layout and the
per-16 E4M3 grid, and normalizes the physical packed name to the logical weight. The runtime applies
the compressed divisor inversion once and preserves all deliberate BF16 tensors instead of
re-encoding them to Q8.

The complete tensor payload is 180,826,481,152 bytes (168.408 GiB) across 99 shards. The full MTP
block is 2,295,901,696 bytes (2.138 GiB); removing it is not a supported profile.

The sealed 108-file artifact manifest is
`d63f8c3da9ab144d42dbfc1136d05294e26d5b7b7a40b114bf4b301359f4092c`.
Its generated metadata hashes are:

- `config.json`: `3cb16aa29d0046ffddd2f8a4866e4c7511e4018c6fced8dd913d1a788d787af9`;
- `hf_quant_config.json`: `38e5689cd6847427cc28c26c3cd3ca30568822bf311f479f11d21cf8ab632d2e`;
- `model.safetensors.index.json`:
  `0f22f6fc51ac7e39b7510a77c77098c4fd7c722e9e6cfdb9782247c37f1b6afd`;
- logical tensor census: `566db2975edac5cd1a86061ec6943988ef695cc8ae8c6cda050ad0d354ae2600`.

The ModelOpt deployment declaration is weight-only `W4A16_NVFP4`; `NVFP4` without the W4A16
prefix means W4A4 to NVIDIA consumers and is rejected for this artifact. The pinned HY3 template's
native reasoning and tools branches are implemented directly, including suffixed tool-call,
argument, and tool-response tokens plus streaming conversion to OpenAI tool calls.
Memra carries that declaration into runtime dispatch: routed NVFP4 weights use BF16-rounded
activations, while the q8_1 expert kernels remain available only to activation-quantized artifacts.
Startup records this selection as `[w4a16] ... expert_activations=bf16-rounded
q8_expert_program=disabled` before model allocation.

The embedded MTP head serves through the eager draft chain. HY3's sigmoid router returns the
selected expert rows through a host-visible synchronization, so a resident expert bank alone does
not make its MTP FFN CUDA-graph-capturable. Memra detects that structural condition and logs
`[spec] draft graph unavailable: sigmoid-router MoE MTP head requires host-visible routing; eager
draft chain engaged`; this is the supported exact arm, not an operator-set `MEMRA_SPEC_NOGRAPH`
workaround.

## Reproduce offline gates

```bash
cargo run -p memra-cli --bin memra --release -- model inspect \
  tencent/Hy3@a960ebc3da325ba167f069f76c41eb62c9280d22 \
  --against hy3 \
  --out research/modelplan-onboarding-hy3-20260830/bf16-inspect

cargo run -p memra-cli --bin memra --release -- model verify tiny \
  --against hy3 \
  --out research/modelplan-onboarding-hy3-20260830/tiny

cargo run -p memra-cli --bin memra --release -- model inspect \
  kodelow/Hy3-NVFP4-W4A16@4f7e1f02f32f4662bdc47f1237f34995088867a9 \
  --against hy3_nvfp4 \
  --out research/modelplan-onboarding-hy3-20260830/existing-kodelow-inspect

cargo run -p memra-cli --bin memra --release -- model inspect \
  LibertAIDAI/Hy3-NVFP4@8a805d687c4bf02b364af15db5ee85e43ed998bf \
  --against hy3_nvfp4 \
  --out research/modelplan-onboarding-hy3-20260830/existing-libertaidai-inspect
```

Remote inspect downloads config, tokenizer/template, index, and safetensors headers only. Both
pinned candidates bind all 47,138 logical tensors and pass the tokenizer/template contract.

## Required RTX gate

Use only the exact RTX PRO 6000 Blackwell topology being qualified. Load through the structural
entrypoint:

```bash
MEMRA_PARALLEL=auto \
MEMRA_PARALLEL_DEVICES=0,1,2,3 \
memra-server
```

For this exact artifact, the bound tensor census must select whole-expert EP-4, report 79 routed
trunk layers, retain the MTP expert bank on the root, and preserve the configured per-card reserve.
No HY3 layer list is an input. Then require:

1. immutable artifact and binary manifests;
2. finite native forward and same-artifact external-oracle comparison;
3. plain-vs-MTP greedy identity with nonzero acceptance at K=1..8;
4. explicit eager, batched, PP, MTP-verify, cache, and rewrite engagement receipts;
5. vendor-default sampled serving, tools/reasoning, concurrency, context/admission, stress,
   cache-on multi-turn, and rollback gates.

Qualification outcome on four RTX PRO 6000 Blackwell Server Edition cards:

- automatic placement selected the generic W4A16 whole-expert backend from `ModelPlan` plus the
  artifact census; the checkpoint estimates were about 60.05 GB on the root and 40.26 GB on each
  peer before the 6 GiB reserve;
- same-artifact vLLM pipeline oracle: argmax equal, top-20 overlap 20/20, cosine
  `0.9995285974311506`, RMSE `0.05849238475772628`, mean absolute error
  `0.046138540558894096`, maximum absolute error `0.30402064323425293`; the pack records the
  predeclared elementwise checkpoint bound as absolute error `<=1.0` with equal argmax;
- greedy MTP K=1..8: target-identical at every K, with nonzero acceptance at every K;
- vendor-default sampled Memra serving (temperature 0.9, top-p 1.0), native tools,
  `reasoning_effort` high/none, concurrent sessions, cache reuse, client-abort rollback, and
  explicit MTP engagement all pass;
- PP-4 wavefront: default remains off. The first arm improved four-client sampled throughput but
  failed full-batch serial-vs-wave identity because wave decomposition narrowed the BF16 reduction
  width. `MEMRA_BF16_MMV=1` restores one per-row BF16 program and passes the B=4/B=8 identity
  matrix with live overlap; it remains an explicit numeric-class door pending its own sampled
  quality/performance and default decision. Without it the runtime now fails closed.

No HY3 DFlash2 checkpoint or Memra HY3 DFlash2 consumer existed at the 2026-08-30 live check.
HY3 DFlash and DFly releases are different draft architectures, so embedded MTP remains the current
speculative path. Recheck this volatile decision before promotion.

Detailed pins, negative candidates, recipes, and pending cells live in
[`research/modelplan-onboarding-hy3-20260830/`](../../research/modelplan-onboarding-hy3-20260830/).
