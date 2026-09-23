# Bound self-contained retained repacks

A retained repack with no fallback now declares its mask through `RetainedRepack` interpretation.
The compiler installs retained original IDs on the typed MoE operation before deriving the tensor
contract. It validates mask width, top-k, routed-layer ownership and source/declaration agreement,
then owns a mask snapshot for the runtime. Source callbacks cannot change that mask after binding.
All optimized surfaces remain blocked by `RetainedExpertRouting`.

Runtime expert names and the semantic member API use original router IDs. They translate to compact
group positions before issuing a sealed request. Per-expert group names are not whole-bank aliases;
a whole-bank probe returns absence, while a pruned member is an explicit error. Metadata validation
accepts the structural StackExperts group without changing the member's physical encoding. The
self-contained mask also requires preservation of expert encodings. Loaded masks are included in
artifact identity alongside the already retained manifest/config and named whole-file bytes.

The positive fixture has four original experts and only IDs 1 and 3 retained. Every retained
projection is explicitly Q2_K or NVFP4 with 256-wide block-valid rows. Controls read the original
encoded bytes through the actual bound runtime and semantic APIs, retain a bounded disk window
after unlink/source drop, and pin identity/mask stability across pathname replacement. A mutable
source callback control proves the compiler mask is sealed rather than re-queried. Missing retained,
extra pruned, wrong-shape and shadowing uniform-bank rows refuse before runtime access.

Manifest parsing now rejects duplicate declaration keys, duplicate IDs/layer aliases, unknown
layers, malformed mask objects/integers and too-few retained experts. Required shape/byte values
and present offset/stride values use strict integer parsing; an invalid optional numeric field
cannot become zero or absence. The existing file-backed JSON structural parser rejects truncated
or trailing content before interpretation. No parser dependency was added.

Validation: six focused source controls pass; GGUF374PASS/2ignored, CLI13, Step1, inspector7,
external3 and doctest2 pass. Actual runtime identity/snapshot host18 pass; all-target GGUF/CLI
warnings-denied Clippy and Linux engine/server lib/bin/test Clippy with DOCS_RS=1 documentation
stubs pass, as do formatting and whitespace checks. Initial fixture/compiler failures and the
separate passing logs are preserved losslessly. The earlier full reference result remains
68PASS/1FAIL at existing [#548](https://github.com/avifenesh/memra/issues/548); its numerical program
was not changed or re-qualified in this source-only slice.

Independent source review withheld GO on exact `86751a19`: escaped JSON declaration keys/values
can bypass interpretation. The [repair bank](../repack-json-repair-20260921/README.md) preserves
the actual red probe. Earlier successful controls above are not reclassified as source approval. Fallback overlays still refuse: their full component
provenance and shadowed inventory are not represented by the current effective census. Root loader
activation, model/native/serving/performance qualification, all optimized rewrites, external draft
composites and the final coordinated #542 dependency intake remain separate. This is an internal
repack source boundary, not a new public format or a scored five-arm research result.
