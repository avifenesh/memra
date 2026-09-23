# Bound semantic census — #541

For the current integration state, exact stage heads, review status and remaining gates, see
[PROGRESS.md](PROGRESS.md). The sections below retain the original foundation-stage account.

Status: compiler/access foundation; **runtime loader activation is not complete**.
The first foundation review found four material defects. The [repair and its red/green controls](review-repair/RESULTS.md) are recorded separately; independent rereview passed the exact repair head `60c07ea2a`. The [runtime-access preparation](runtime-access/RESULTS.md) is a subsequent, separately unqualified increment.
Base: `dc598deb47abe1d24572683263ece9dd7a5fd0ca`, which retains the reviewed #537
model-semantics dependency (`f2c6fdcfc3ec4b61ebdaab67f1e5bc76d1b79322`).

The dense and hybrid entrypoints still use their existing source APIs. This increment is
reviewable on its own but does not close #541, qualify a model, or establish a CUDA result.
Integration review and merge remain with the coordinating task.

## Implemented boundary

`BoundTensorSource::compile` owns the immutable config, ModelPlan, tensor contract, complete
binding and census. It binds all metadata and validates storage headers before materializing
weights. The Step factor preflight from #537 remains in force after binding. Read-only accessors
cannot swap the config, plan, target or transform independently.

Typed access uses the bound physical target and transform. GGUF borrows its original mapping;
safetensors materializers now accept resolved targets and also retain their legacy entrypoints.
Native FP8/NVFP4 and disk methods return `Result<Option<_>, String>`: incompatible optional native
representations may be absent, while malformed bound payloads return contextual errors. Missing
source implementations refuse explicitly. Auxiliary metadata includes physical names, dtypes,
shapes and byte counts, even when the owning weight is floating point. Macro, input and AWQ
scale views come from that same bundle; compressed-tensors global scales retain divisor semantics.

Output ownership comes from explicit normalized `tie_word_embeddings` or a declared pack default.
Gemma packs default to tied ownership; the other packs default to separate ownership. Inspection
uses that decision as well. An absent untied head is an error, and a tied artifact with an extra
independent head is not silently accepted. Generic weight constraints reject integer storage, and
binding checks duplicate semantic IDs even for a manually assembled `TensorContract`.

`binding_sha256` is an **interpretation digest only**, not an artifact identity or trusted receipt.
Its versioned, length-delimited canonical encoding includes config, plan, contract, ownership,
effective targets, transform order, shapes/storage and physical auxiliary metadata. Census row
order is normalized. The #542 composition must additionally hash the exact opened artifact's
`artifact_sha256`; underlying bytes and current numeric environment remain separate obligations.

## Validation

See [VALIDATION.md](VALIDATION.md) and the raw logs. The twelve new CPU tests cover malformed
censuses, missing and ambiguous tensors, integer weights, head ownership, GGUF/HF borrowing,
transforms, native payload failures, scale handling, digest changes and opened-path replacement.

## Remaining implementation before #541 acceptance

- Activate the bundle at both dense and hybrid loader boundaries, including parallel placement.
- Route root, layer, auxiliary, native, expert and disk reads through the same bundle and propagate
  errors without converting them to absence. Remove raw GGUF materialization bypasses.
- Complete typed derived MLA planes, fused expert banks, native expert members, standalone MTP and
  manifest/overlay ownership. The current accessor refuses unsupported derived-plane requests.
- Preserve #542's opened/private backing and compose its artifact identity with this digest.
- Qualify supported GGUF and safetensors dialects on intended native paths, after source review
  and coordinated physical-card locking. No GPU work was performed in this increment.
