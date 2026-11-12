# RTX PRO 6000 Blackwell

The 96 GB Blackwell workstation and server card, memra's primary optimized target alongside the RTX
5090. This card covers the three editions (full-power Workstation, Max-Q Workstation, Server), which
share a build and a hardware class but not every default.

| | |
|---|---|
| **Build target** | `sm_120a`, the default target; auto-detected from compute capability 12.0 (`crates/memra-engine/build.rs`) |
| **Prebuilt** | yes, the release tarball and `tools/install.sh` |
| **Push gate** | yes: a push that reaches `sm_120a` needs a passing cell record from an `sm_120a` device (`tools/change-impact.toml`); an RTX 5090 counts unless the default is keyed on this card's name |
| **Maintained** | yes, the primary optimized target |
| **Topologies** | one card; 2-4 cards for PP-2, TP/EP and EP-4 routes ([decision](../decisions/PRO6000-MULTICARD.md)) |

## Device-keyed defaults

Each default below is keyed on the device name and states itself at boot. A default measured on one
edition applies to that edition only.

| Default | Here | Elsewhere | Record |
|---|---|---|---|
| On-demand KV planes (`[kv-vmm] door=ON`) | full-power Workstation, one-card workers | pooled planes | [KV-VMM-SERVING-DEFAULT](../decisions/KV-VMM-SERVING-DEFAULT.md) |
| Memory-shaped admission (`[admit-mem] door=ON`) | full-power Workstation, one-card workers | envelope admission | [ADMIT-BY-MEMORY-DEFAULT](../decisions/ADMIT-BY-MEMORY-DEFAULT.md) |
| Batched-chunk OOM retry | full-power Workstation (and the RTX 5090 Laptop) | one OOM ends the chunk | [BATCH-OOM-RECOVER-DEFAULT](../decisions/BATCH-OOM-RECOVER-DEFAULT.md) |
| `O_DIRECT` handoff export | full-power Workstation | page-cache writer | [HANDOFF-DIRECT-EXPORT](../decisions/HANDOFF-DIRECT-EXPORT.md) |
| MoE in-token expert prefetch | full-power Workstation (and the RTX 5090 Laptop) | off | [MOE-INTOKEN-PREFETCH-DEFAULT](../decisions/MOE-INTOKEN-PREFETCH-DEFAULT.md) |
| Cold-long speculative depth K=2, draft-step cost fit | Max-Q (K=2 also on the RTX 5090 Laptop) | K=3 | [SPEC-K-COLD-LONG-K2](../decisions/SPEC-K-COLD-LONG-K2.md) |
| DFlash prefix restore (`MEMRA_DSPARK_PREFIX_RESTORE`) | Max-Q (and GB10) | off | `MEMRA_DSPARK_PREFIX_RESTORE` row of [FLAGS.md](../FLAGS.md) |
| Device penalty arm (`MEMRA_SERVE_DEVPENALTY`), dense Qwen3.5 class | on | on for the RTX 5090; GB10 adds the MoE class | [GB10-DEVICE-PENALTY-DEFAULT](../decisions/GB10-DEVICE-PENALTY-DEFAULT.md) |

The Workstation list with its boot lines: [serving](../SERVING.md#serving-defaults-keyed-to-the-rtx-pro-6000-workstation-edition-2026-11-04).
The Server Edition has no default of its own.

## Models measured here

| Model | Artifact | Pack record | Last measured | Receipt |
|---|---|---|---|---|
| [Qwen3.8-27B](../models/qwen38-27b.md) | NVFP4+Q5_K MTP GGUF | none for this artifact | tag battery v0.141.1, 2026-11-11 (Max-Q) | [release records](../../research/release-qualification/) |
| [Ornith-1.5-35B-A3B](../models/ornith15-35b-a3b.md) | NVFP4+Q5_K MTP GGUF | none for this artifact | tag battery v0.141.1, 2026-11-11 (Max-Q) | [release records](../../research/release-qualification/) |
| [Gemma-4 31B](../models/gemma4-31b.md) | Q6_K trunk, NVFP4 assistant drafter | none | no tracked speed receipt | [cookbook](../COOKBOOK.md#gemma-4-31b) |
| [Step-3.7-Flash 196B-A11B](../models/step37-flash.md) | IQ4_XS GGUF, PP-2; FP8, PP-3 | none (`step35` has no state) | 2026-09-20 (Server Edition) | [record](../../research/modelplan-onboarding-537-20260919/native-20260920/RESULTS.md) |
| [DeepSeek-V4-Flash](../models/deepseek-v4-flash.md) | NVFP4 checkpoint, TP/EP pair | none (`deepseek_v4` has no state) | 2026-09-28; gate cells 2026-11-08 (Max-Q) | [lane](../../research/dsv4f-bringup-20260923/) |
| [GLM-5.3-Flash](../models/glm53-flash.md) | NVFP4 mint, one card and pairs | `glm5_next-reference` (CPU) | 2026-09 | [lane](../../research/glm53-flash-bringup-20260827/BRINGUP.md) |
| [Hy3](../models/hy3.md) | W4A16 mint, EP-4 on four cards | `hy3_nvfp4-modelopt-w4a16` (tiny gate) | reported 2026-08-30, not tracked | [lane](../../research/modelplan-onboarding-hy3-20260830/) |

## Limits

- Do not copy an RTX 5090 default onto this card, or a Workstation default onto the Max-Q or Server
  Edition, without its own measurement.
- Multi-card routes are per model: PP-2 for Step, TP/EP for DeepSeek-V4-Flash, `MEMRA_GLM5_TP`
  TP-2/TP-4 for GLM-5.3-Flash, automatic whole-expert EP-4 for Hy3. Never call PP or replicas TP.
- Both cards of a pair share a PCIe path, so work on the idle card changes a scored campaign's
  thermal and I/O regime.

## Deeper sources

- Measurement conditions per rig: [PERFORMANCE.md](../PERFORMANCE.md).
- Which evidence a change needs: [RELEASE-QUALIFICATION.md](../RELEASE-QUALIFICATION.md) and
  [TESTING.md](../TESTING.md).
