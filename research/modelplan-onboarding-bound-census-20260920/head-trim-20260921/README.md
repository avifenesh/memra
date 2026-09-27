# Bound target-head self-trim materialization

This slice follows reviewed `b863ab07d1529ec5988b21f4280530ae71686cc4`. Rank intake already
binds consumed IDs and their source identity. This slice binds the next dependency: which
target head supplies the rows, which macro belongs to it, and exactly what reaches the upload
consumer. Independent source review is pending; earlier refs and evidence remain unchanged.

## Selection and materialization

`PreparedHeadTrim` accepts only a compiler-bound target source and a captured `RankArtifact`.
Raw TensorSource implementations and external-draft authority refuse. Selection is explicit:
model output (including its declared tied embedding), first private MTP head or model output,
or a specific declared MTP block. A known block without a private head returns absence for
the existing chain boundary; an unknown block errors. No failed read is retried as another head.
Composite target selection retains the winning physical component, dialect and record.

The selected runtime matrix must have complete, independently encoded rows supported by the
existing resident consumer. Rank bounds are checked before gathering. NVFP4 macro lookup uses
the selected head's own semantic owner, including tied embeddings and each extra private head;
it no longer borrows output.scale for a private head or drops extra-head macros. The reviewed
F32 macro reader supplies the checked value. Unsupported encodings, misaligned quant rows and
short macro encodings refuse before trim upload.

One row buffer captures the source runtime view: the same captured bytes feed the full-head
digest and the rank-ordered output. The producer keeps the gathered output without making a
second full-head copy. Computing that digest scans the selected head, not the complete model.
`Preserve` is byte gathering. `Nvfp4ForEligibleBf16` retains the existing conditional policy:
only BF16 heads with width divisible by64 run the unchanged BF16-expansion/NVFP4 encoder;
other supported types retain their row-gather program. No new format or math fallback is added.

The immutable result owns bytes, rank order, dtype, shape, macro and a materialization receipt.
The receipt binds the compiled source interpretation, semantic/physical head, rank identity,
full captured runtime-head digest, policy/program, source macro bits and exact output digest
and parameters. This is a selected-head receipt, NOT a full opened-model artifact identity.
Other target weights and the running binary must still enter later paired identity composition.

## Actual consumer adoption

All four native trim sites use the same producer and upload consumer: normal MTP self-trim,
MTP-skip stub, extra MTP heads and GLM DFlash2 target-head slab. The consumer accepts only the
immutable result and hands its exact bytes/shape/dtype/macro to the existing GpuTensor path.
It emits the materialization digest after successful construction. Public MtpHead/DflashTrimHead
layouts are unchanged; no paired-identity/admission interface is installed.

The extra-head gate is CPU-testable production code. Already-trimmed standalone drafts with
zero extra heads require no separate rank artifact; a real trimmed chain must use the captured
rank order. Existing first/last-stage engine selection, missing-private-head chain termination,
full-precision gates and the GpuTensor quant packing implementation are preserved.

## Evidence and limits

Three source tests pass for separate/tied HF heads, winning composite head/macros and identity
sensitivity to rank order/unselected source-head rows. Seven consumer tests compile the actual
engine upload/gating modules by path. Terminal GPU operations are recorders, not GPU execution.
Real complete micro-GGUFs prove model/private/extra macro values2/3/5, tied fallback, explicit
missing/unknown cases, raw F32/BF16/Q8 rows, exact known BF16-to-NVFP4 blocks, unsupported dtype
and row alignment, short macros, rank bounds, external-authority refusal and immutable payloads
after source/rank edits. A macro-only edit changes the receipt and upload scale while payload
bytes stay identical. The fourteen existing standalone-draft controls also pass:24 total,
zero ignored. Strict source/harness and Linux DOCS_RS engine lib/test Clippy and formatting pass.

The preservation receipt verifies923 other crate files, public draft layouts and the existing
standalone loaders/rank helpers. Source, commands, raw logs and two CPU executables are pinned
in evidence.json. The initial composite test expected the wrong component path (the root
overlay is[]); that test repair and initial logs are preserved. Existing targets were reused;
no owned cache cleanup or large target copy was performed.

Parent coordination was notified before implementation. Shared RewriteIdentity/admission,
#542 plan_backend/fallback, worker/LRU and support/default policy remain unchanged. Composite
external drafts, whole target/draft identity and native qualification remain separate work.
Eager/DecodeGraph proof does not authorize external Spec. No GPU/remote job, main merge,
rental/topup or native/model/support promotion occurred.
