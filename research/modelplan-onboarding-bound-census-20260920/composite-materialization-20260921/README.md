# Composite masks, materialization and opened identity

This stage builds on the reviewed metadata catalog and the separate #537 configuration intake
at `b45d065471d8e1c2498f9c5b6bb18b632ec7d95c`. The strict decoder and Cargo feature pair remain
the reviewed `bb637184682216ea38cbb9eb79dab4841368eca1` implementation. No decoder fork or
runtime root activation is included.

## Source contract

Each component inherits the nearest lower declaration of a layer's expert mask unless it
explicitly replaces that declaration. Masks are applied to typed MoE plans and validated against
router width and top-k. Every group records original router IDs. A higher sparse component cannot
resurrect an inherited pruned ID, and a lower full bank cannot implicitly supply a missing split
member. Lower original banks and pruned records remain accounted for in their own physical
inventory; they are not exposed as retained members.

GGUF split-member names are an explicit encoding alternative derived from the canonical typed
expert-bank schema. Both stacked and split forms in one component are ambiguous and refuse.
Missing members, invalid original IDs, conflicting shapes and undeclared tensors still refuse.
Unpruned numerical plans are unchanged; retained programs keep the existing
`RetainedExpertRouting` operation and optimized-rewrite blockers.

`BoundCompositeSource` retains the actual opened components and creates requests only from the
sealed selection. Reads carry the selected physical record, dialect and transform. Group access
requires an original member ID; whole-bank slicing remains refused. The existing tensor and
native-plane materializers preserve their numerical programs. Q2_K and NVFP4 repack bytes remain
in their declared encodings. Native FP8 and NVFP4 planes retain their scale data. Optional native
misses are evaluated only after scale validation.

All component auxiliary values are checked before the source bundle is returned, including
shadowed and unselected planes. Independent scales stay with their own weights; tied output
heads use the embedding's scale as well as its bytes. Selected RoPE factors are materialized
fallibly before entering the existing source-factor preflight, so a read error cannot become
absence. The validated config and its factors contribute to the semantic identity.

Disk reads expose only opaque bounded views. Those views retain their exact backing after the
source is dropped or its pathname is replaced. No whole-file descriptor or mmap is exported.
The explicit bundle does not implement `TensorSource`: ordinary fallback binding, artifact
identity and engine-root entrypoints remain refused until the final runtime integration.

## Identity

The composite opened-source digest frames every component in order, including exact captured
manifest/config declarations, complete opened shard bytes, padding, and repack shard-name
assignment. It never reopens a pathname. A separate semantic digest binds the catalog and
validated config; the final identity composes both domains.

The standalone complete/retained repack identity program is preserved. An identical real-file
probe ran against the frozen pre-change source and the candidate; both identity values matched
exactly. The probe, input hashes, commands and results are retained in `identity-compat/`.

## CPU acceptance

The focused controls exercise inherited masks, explicit retained overlays over uniform banks,
mixed Q2_K/NVFP4 members, original-ID lookup across components, invalid/pruned-member rejection,
ambiguous stacked/split declarations, native FP8/NVFP4 matrices and stacked planes, detached disk
reads, pathname replacement, shadowed payload changes, manifest-byte and padding changes,
independent/tied-head scales, invalid-scale refusal, source-factor preflight and text-only scope.

A real-file F32 retained composite also passes an end-to-end Memra reference test: every loaded
weight and every output logit is bit-identical to the independent deterministic reference fixture.
The fixture explicitly declares tied embeddings and writes its optional q/k norm tensors. Initial
fixture-construction failures are retained as development results, not relabeled as passes.

All ten final commands pass. The composite filter runs 24 tests; the separate reference
integration runs one complete loaded-model comparison. The full GGUF run reports 411 passes
and two declared ignores; inherited artifact-dependent returns retain their original limits.
Compiler/CLI, feature-mode, identity-host and lint checks pass.
Full command statuses, source hashes and lossless logs are in `evidence.json` and `raw/`.
Feature-unified canonical config and manifest controls run with `preserve_order`. Linux
engine/server checks use `DOCS_RS=1` and provide type/lint evidence only. Artifact-dependent
early returns in inherited tests are not checkpoint coverage. The existing #548 full-reference
failure is unchanged and is not claimed fixed by the passing targeted reference test.

## Remaining gates

Independent review of the combined intake and this source slice is required next. Engine-root
adoption, external/trimmed draft contracts, private canonical cache-output boundaries and native
numeric, serving, worker/H2D and I/O qualification remain pending. Unsupported materializer or
independent auxiliary-group forms continue to return errors. No source review, native/GPU/model
support, performance default, public-format or main-merge approval is asserted by these receipts.
No remote job, GPU lease, rental or new issue lane was created for this stage.
