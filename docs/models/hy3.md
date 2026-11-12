# Hy3

Tencent's Hy3 mixture-of-experts as a native ModelPlan pack: the canonical BF16 plan, and an exact
all-expert ModelOpt W4A16 (NVFP4 weights) profile placed on four RTX PRO 6000 Blackwell cards by
automatic whole-expert expert parallelism.

Support state: **NativeReference** for the canonical plan and official BF16 safetensors, <!-- support: hy3-bf16-reference -->
and **NativeReference** for the exact all-expert ModelOpt W4A16 profile `hy3_nvfp4` described below. <!-- support: hy3_nvfp4-modelopt-w4a16 -->
The 2026-08-30 four-card RTX PRO 6000 Blackwell run reported the outcome below, but its oracle and
serve receipts are not tracked in this repository, and the tracked inspect record of the minted
artifact lists checkpoint parity and serve pending. Qualification is pending until those gates are
re-run and their receipts land with a record in `docs/support-records.toml` (memra#551).
`NativeTuned` also remains pending: the faster PP-4 wavefront stayed <!-- support: none; not NativeTuned -->
off after failing its serial-vs-wave logit-identity gate.

| | |
|---|---|
| **Serving route** | the W4A16 mint on 4x RTX PRO 6000 Blackwell Server Edition, automatic whole-expert EP-4. Reported 2026-08-30 at `bfd5a2d88e`, receipts not tracked. Standing gate: none |
| **Packs** | `hy3` (canonical BF16 plan) and `hy3_nvfp4` (aliases `hy3-nvfp4`, `hy_v3_nvfp4`), an onboarding profile that `model inspect --against hy3_nvfp4` selects and automatic pack selection never does |
| **Artifacts** | source `tencent/Hy3@a960ebc3da325ba167f069f76c41eb62c9280d22`; Memra's streaming NVIDIA ModelOpt 0.46.0 mint (ModelOpt commit `43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a`), sealed 108-file manifest `d63f8c3da9ab144d42dbfc1136d05294e26d5b7b7a40b114bf4b301359f4092c`, 180,826,481,152 tensor bytes (168.408 GiB) in 99 shards. Comparison controls: `kodelow/Hy3-NVFP4-W4A16@4f7e1f02f32f4662bdc47f1237f34995088867a9`, `LibertAIDAI/Hy3-NVFP4@8a805d687c4bf02b364af15db5ee85e43ed998bf` |
| **Drafters** | the embedded MTP head, through the eager draft chain |
| **Hardware** | 4x RTX PRO 6000 Blackwell Server Edition for the W4A16 profile; CPU for the reference gates |
| **Input** | text; the pinned HY3 template's own reasoning and tools branches, including its suffixed tool-call, argument and tool-response tokens and the streaming conversion to OpenAI tool calls |

A W4A16 mint is published as [`tiyuvta/Hy3-NVFP4`](https://huggingface.co/tiyuvta/Hy3-NVFP4) (99
safetensors shards and a `SHA256SUMS` file). This repository tracks no byte comparison between that
repo and the manifest above, and its `masked-mtp/` ranks files have no gate here.

## Artifact contract

The NVFP4 contract is exact: every routed-expert projection in layers 1 through 80 is NVFP4,
including all 576 MTP expert weights. Attention, router, shared MLP, dense layer 0, embeddings,
output head, norms, biases and non-expert MTP tensors remain checkpoint BF16/F32. There is no
MTP-BF16 profile, and removing the 2,295,901,696-byte MTP block is not a supported profile.

Both accepted storage encodings preserve the same semantic program:

- compressed-tensors `weight_packed` + `weight_scale` + divisor `weight_global_scale`;
- ModelOpt packed `weight` + `weight_scale` + multiplier `weight_scale_2`.

For ModelOpt fused MoE GEMM1, each expert's gate and up projections share the larger of their two
per-tensor `weight_scale_2` values; the artifact gate checks all 15,360 pairs bit for bit. This is
the pinned NVIDIA ModelOpt fused-MoE recipe, not a loader repair after quantization.

The header contract requires exactly one macro scale, validates its scalar F32 layout and the
per-16 E4M3 grid, and normalizes the physical packed name to the logical weight. The runtime applies
the compressed divisor inversion once and keeps every deliberate BF16 tensor instead of re-encoding
it to Q8.

The ModelOpt deployment declaration is weight-only `W4A16_NVFP4`; `NVFP4` without the W4A16 prefix
means W4A4 to NVIDIA consumers and is rejected for this artifact. Memra carries the declaration into
dispatch: routed NVFP4 weights use BF16-rounded activations, and the q8_1 expert kernels stay
reserved for activation-quantized artifacts. Startup logs `[w4a16] ... expert_activations=bf16-rounded
q8_expert_program=disabled` before model allocation.

The embedded MTP head serves through the eager draft chain. HY3's sigmoid router returns the selected
expert rows through a host-visible synchronization, so a resident expert bank alone does not make
its MTP FFN capturable. Memra detects that and logs `[spec] draft graph unavailable: sigmoid-router
MoE MTP head requires host-visible routing; eager draft chain engaged`; this is the exact arm, not a
`MEMRA_SPEC_NOGRAPH` workaround.

## Run it

The offline gates run from a source checkout:

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

Remote inspect downloads config, tokenizer and template, the index and safetensors headers only.
Both pinned controls bind all 47,138 logical tensors and pass the tokenizer and template contract.

To serve the W4A16 mint, name its directory in `MEMRA_MODELS` and set `MEMRA_PARALLEL=auto` with
`MEMRA_PARALLEL_DEVICES=0,1,2,3`. The bound tensor census must select whole-expert EP-4, report 79
routed trunk layers, keep the MTP expert bank on the root and preserve the configured per-card
reserve; no HY3 layer list is an input. On this placement `MEMRA_PARALLEL=auto` also keeps the
non-expert BF16 rows BF16-resident (`MEMRA_BF16_MMV`, explicit `0` is the logged rollback).

## Measured

Reported for 2026-08-30 on four RTX PRO 6000 Blackwell Server Edition cards, receipts not tracked:
automatic placement picked the generic W4A16 whole-expert backend (about 60.05 GB on the root and
40.26 GB per peer before the 6 GiB reserve); a same-artifact external pipeline oracle gave equal
argmax, top-20 overlap 20/20 and cosine 0.9995; greedy MTP K=1..8 was target-identical with nonzero
acceptance at every K; vendor-default sampled serving, tools, reasoning, concurrency, cache reuse
and client-abort rollback passed. The full list is in the
[history file](../history/MODELS.md#hy3-card-required-rtx-gate-and-the-2026-08-30-outcome).

## Limits

- CheckpointParity and Serve are pending for both records until the gates re-run with tracked
  receipts.
- The PP-4 wavefront stays off: its first arm failed full-batch serial-vs-wave identity because
  wave decomposition narrowed the BF16 reduction width.
- Embedded MTP is the only speculative path. At the 2026-08-30 check, HY3 DFlash and DFly releases
  were different draft architectures with no consumer in memra.

## Deeper sources

- Pins, negative candidates, recipes and pending cells:
  [`research/modelplan-onboarding-hy3-20260830/`](../../research/modelplan-onboarding-hy3-20260830/).
- Placement policy: [PRO6000-MULTICARD](../decisions/PRO6000-MULTICARD.md).
- Packs: `crates/memra-gguf/src/model_packs/hy3/`. Spill work on this family:
  [HY3-SPILL.md](../HY3-SPILL.md).
- Metadata hashes of the mint: [history](../history/MODELS.md#hy3-card-artifact-hashes).
