# Hy4-flash

Tencent's Hy4-flash as a native ModelPlan pack, onboarded on its weight-only ModelOpt NVFP4
artifact: gated MLA on every layer (12 indexed layers with a pooled indexer, 30 sliding-window
layers), 4-stream hyper-connections, a latent MoE of 512 experts and the Engram n-gram memory.

Support state: **NativeReference** on CPU. The reference executor runs the whole text program, <!-- support: hy_v4_flash-reference -->
prefill and decode, and matches the pinned transformers code to the bit on the tiny fixture. No
native runner loads the artifact yet, so CheckpointParity, RewriteParity and Serve are pending
and nothing serves the pack. NativeQualified needs those gates on a pinned topology first. <!-- support: none; not NativeQualified -->

| | |
|---|---|
| **Serving route** | none |
| **Pack** | `hy_v4_flash` (aliases `hy-v4-flash`, `hy4-flash`) |
| **Artifact** | `Avifenesh/Hy4-flash-NVFP4@f9eac06cf022cfd3dc2e8ed423922466b2adcd87`, from `tencent/Hy4-flash@d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959`: NVFP4 routed experts, FP8 128x128 attention projections, FP8 Engram rows, BF16 and F32 elsewhere; 2,429 tensors, 161.08 GB |
| **Drafters** | the checkpoint's DFly2 drafter is contract-owned and not run |
| **Hardware** | CPU for the reference gates |
| **Input** | text; the artifact's chat template and tool-call format; image parts are out of scope |

## Gates

- Config, TokenizerTemplate and TensorCensus pass at `memra model inspect` on the pinned revision,
  over HTTP with no weights downloaded.
- TinyParity: `memra model verify tiny --against hy_v4_flash` on the artifact's config at hidden
  64, past the 512-token window and the indexer budget, with splits below each.

## Limits

- No native runner and no GPU route.
- Vision is out of scope; the tower's tensors are owned by the contract and not run.
- The context the pack would serve is not measured.

## Deeper sources

- The lane: [`research/modelplan-onboarding-hy4flash-20261119/`](../../research/modelplan-onboarding-hy4flash-20261119/)
  (pins, semantics, each step's oracle and gate).
- Pack: `crates/memra-gguf/src/model_packs/hy_v4_flash/`; reference:
  `crates/memra-reference/src/hy_v4_flash.rs` and `hy_v4_flash_trunk.rs`.
