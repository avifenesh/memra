# Execution surfaces — what the engine implements, per plan operation

Generated table: **do not hand-edit** between the markers. The source of truth is
`crates/memra-gguf/src/op_registry.rs` (`surfaces(OperationKind)`); the seven execution manifests
in `crates/memra-gguf/src/execution_manifest.rs` (`CARRIED_PRIME`, `NATIVE_EAGER`, `DECODE_BATCH`,
`DECODE_GRAPH`, `MTP_SPEC`, `GLM5_SPEC`, `PIPELINE`) derive their per-operation support from it,
and `op_registry::tests::rendered_table_matches_docs_execution_surfaces` fails the build when this
file and the registry disagree. Regenerate with

```
cargo test -p memra-gguf -- --ignored print_execution_surfaces_table --nocapture
```

and paste the output between the markers.

## How to read it

- A column is a tuned execution program. `yes` means the engine implements that operation on that
  program **and the program's gate covers it**; a row that is all `—` runs only in the reference
  executor (`crates/memra-reference`). A plan is eligible for a program when every operation in
  it says `yes` in that column — the manifest is an AND over the plan, never a family name.
- `mtp_spec_*` and `glm5_spec_*` are split into draft and verify halves because the manifests gate
  them separately (`OperationSupport::{spec_draft, spec_verify}`); the two programs are disjoint on
  the mixer/residual classes, so no plan is ever eligible for both.
- `chunked_prime` (added by memra#535 P1a): the generic `prime_cache` chunked / continuation
  prime program, with a chunk-invariance receipt per operation (`tools/chunk-invariance-gate.sh`,
  `concat-prime-probe tickinv`). A row flips only on the receipt: `GemmaParallelMoeResidual` was
  first MEASURED chunk-dependent (gemma-4-26B-A4B: prefill logits moved O(1) with the chunk size,
  first divergence at row 0) — the gemma MoE arm routed prefill through the m-dependent cuBLAS
  matmul (memra#562) — and became yes when the router moved to `router_gemv` and the 26B read
  EXACT on both gates. `HyperConnections` says no because glm5 chunks through its own
  `prime_cache_hyper` / walker program, not this one.
- Dedicated per-family arms that bypass the canonical programs are **not** rows here until they
  implement the shared contract. Today that is the `HyperConnections` batched-decode walk
  (`decode_step_batch_hyper`), `prime_cache_hyper`, the glm5 TP walk and the dsv4 serial route;
  `execution_manifest::decode_batch_unconverted` names that gap and the worker routes around it.
  memra#535 phases P1–P4 move those arms behind the contract, at which point their rows flip
  here and the predicate is deleted.
- The registry's `surfaces` match is exhaustive: adding an `OperationKind` without declaring its
  row is a compile error. That is the fail-closed-and-visible property this table exists for.

Provenance: introduced by memra#535 P0 (zero behaviour change — the registry reproduces the seven
pre-existing allowlists for every operation, pinned by
`execution_manifest::tests::registry_reproduces_every_legacy_manifest_table`).

<!-- EXEC-SURFACES:START -->
| operation | carried_prime | decode_eager | decode_batch | decode_graph | mtp_spec_draft | mtp_spec_verify | glm5_spec_draft | glm5_spec_verify | pipeline | chunked_prime |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `AudioLogMel` | — | — | — | — | — | — | — | — | — | — |
| `AudioStridedConv` | — | — | — | — | — | — | — | — | — | — |
| `AudioPositionEmbedding` | — | — | — | — | — | — | — | — | — | — |
| `BiasedLayerNorm` | — | — | — | — | — | — | — | — | — | — |
| `GeluErfActivation` | — | — | — | — | — | — | — | — | — | — |
| `AudioEncoderAttention` | — | — | — | — | — | — | — | — | — | — |
| `AudioDecoderSelfAttention` | — | — | — | — | — | — | — | — | — | — |
| `AudioCrossAttention` | — | — | — | — | — | — | — | — | — | — |
| `AudioCrossKvState` | — | — | — | — | — | — | — | — | — | — |
| `AsrDeterministicDecode` | — | — | — | — | — | — | — | — | — | — |
| `Embedding` | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `VisionPatchEmbedding` | — | — | — | — | — | — | — | — | — | — |
| `VisionBidirectionalAttention` | — | — | — | — | — | — | — | — | — | — |
| `VisionMlp` | — | — | — | — | — | — | — | — | — | — |
| `VisionStandardize` | — | — | — | — | — | — | — | — | — | — |
| `VisionDownsample` | — | — | — | — | — | — | — | — | — | — |
| `VisionProjection` | — | — | — | — | — | — | — | — | — | — |
| `VisionTokenInjection` | — | — | — | — | — | — | — | — | — | — |
| `RmsNorm` | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `FullAttention` | yes | yes | yes | yes | yes | yes | — | — | yes | yes |
| `SlidingWindowAttention` | — | — | yes | — | yes | yes | — | — | yes | yes |
| `MiMoAttentionMath` | — | — | — | — | — | — | — | — | — | — |
| `LatentMlaAttention` | — | — | — | — | — | — | yes | yes | yes | — |
| `CompressedMlaAttention` | — | — | — | — | — | — | — | — | — | — |
| `KvCompressor` | — | — | — | — | — | — | — | — | — | — |
| `SparseIndex` | — | — | — | — | — | — | yes | yes | yes | — |
| `SharedSparseIndex` | — | — | — | — | — | — | yes | yes | — | — |
| `MicroBlockSparseIndex` | — | — | — | — | — | — | — | — | — | — |
| `GatedDeltaNet` | yes | — | yes | yes | yes | yes | — | — | — | yes |
| `KimiDeltaNet` | — | — | — | — | — | — | yes | yes | yes | — |
| `FusedAttentionGate` | yes | — | yes | yes | yes | yes | — | — | — | yes |
| `SeparateAttentionGate` | — | — | yes | — | yes | yes | — | — | yes | yes |
| `DenseMlp` | yes | yes | yes | yes | yes | yes | — | yes | yes | yes |
| `MoeMlp` | — | — | yes | yes | yes | yes | yes | yes | yes | yes |
| `SoftmaxRouter` | — | — | yes | yes | yes | yes | — | — | — | yes |
| `SigmoidRouter` | — | — | yes | yes | yes | yes | yes | yes | yes | yes |
| `SqrtSoftplusRouter` | — | — | — | — | — | — | — | — | — | — |
| `TokenHashRouter` | — | — | — | — | — | — | — | — | — | — |
| `SharedMlp` | — | — | yes | yes | yes | yes | yes | yes | yes | yes |
| `SiluActivation` | yes | yes | yes | yes | yes | yes | — | — | yes | yes |
| `GeluTanhActivation` | — | — | yes | — | — | — | — | — | — | yes |
| `SwiGluOaiActivation` | — | — | — | — | — | — | — | — | — | — |
| `SwiGluClampedActivation` | — | — | yes | — | yes | yes | — | — | yes | yes |
| `SwiGluPreClampedActivation` | — | — | — | — | — | — | yes | yes | yes | — |
| `NamedActivation` | — | — | — | — | — | — | — | — | — | — |
| `SerialResidual` | yes | yes | yes | yes | yes | yes | yes | — | yes | yes |
| `GemmaResidual` | — | — | yes | — | — | — | — | — | — | yes |
| `GemmaParallelMoeResidual` | — | — | yes | — | — | — | — | — | — | yes |
| `HyperConnections` | — | — | — | — | — | — | — | yes | yes | — |
| `GatedResidualConnections` | — | — | — | — | — | — | — | — | — | — |
| `GatedResidualMixer` | — | — | — | — | — | — | — | — | — | — |
| `PleNgramEmbedding` | — | — | — | — | — | — | — | — | — | — |
| `KvState` | yes | yes | yes | yes | yes | yes | — | — | yes | yes |
| `SlidingKvState` | — | — | yes | — | yes | yes | — | — | yes | yes |
| `RecurrentState` | yes | — | yes | yes | yes | yes | yes | yes | yes | yes |
| `LatentKvState` | — | — | — | — | — | — | yes | yes | yes | — |
| `CompressedAttentionState` | — | — | — | — | — | — | — | — | — | — |
| `Mtp` | — | — | — | — | yes | — | yes | — | yes | — |
| `DraftPlan` | — | — | — | — | — | — | — | — | — | — |
| `MtpFusion` | — | — | — | — | yes | — | yes | — | yes | — |
| `MtpHead` | — | — | — | — | yes | — | yes | — | yes | — |
| `DsparkFusion` | — | — | — | — | — | — | — | — | — | — |
| `DsparkMarkovHead` | — | — | — | — | — | — | — | — | — | — |
| `DsparkConfidenceHead` | — | — | — | — | — | — | — | — | — | — |
| `PipelineBoundary` | — | — | — | — | — | — | — | — | yes | — |
| `LogitsSoftcap` | — | yes | yes | — | — | — | — | — | — | yes |
| `LogitsMask` | yes | yes | yes | yes | yes | yes | — | — | yes | yes |
| `OutputProjection` | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
<!-- EXEC-SURFACES:END -->
