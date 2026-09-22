# Target/draft source-pair receipts

This bounded slice follows reviewed `ab9dcfc00853d40c76bc842b77696e15d4509f73`.
It composes the complete opened target identity with either an external draft program or a
newly produced target-head host materialization. Independent review of this slice is pending.
The two #542 handoffs were read before implementation, and the concrete API proposal was sent
through the parent. The parent relayed it and confirmed the owned source/host scope.

## Typed source composition

`PreparedDraftTarget::bind` requires compiler-private Single/Composite target authority and a
valid text-root scope. Raw sources and external-draft authority refuse. It captures the actual
sealed target config and complete `BoundArtifactIdentity`; callers cannot replace them with a
config, manifest, hash string, Eager receipt or caller-built preflight tuple.

The opt-in API has three operations:

- `with_external` prepares an opened standalone GGUF against that target config.
- `with_composite_external` uses reviewed ordered GGUF-component intake against the same config.
- `prepare_head_trim` runs the reviewed materializer against that exact target source and returns
  `PreparedPairedHeadTrim`, which owns both the real immutable payload and its source-pair receipt.

`DraftSourcePairIdentity` has private fields and no public constructor. Its tagged, length-framed
`memra-draft-source-pair-v1` digest binds the target identity and a separately typed proposal.

| Field | Meaning |
|---|---|
| `target()` | Complete opened target artifact, selected semantic/scope binding, and their existing composite hash |
| `External` proposal | Separate complete draft artifact; ordered complete component hashes for composite input; selected semantic head role/name and norm; declared NextN blocks with the actual selected first block; exact full/trimmed vocabulary order |
| `TargetHeadTrim` proposal | Existing `HeadTrimIdentity`: chosen head role/component/tensor, selected source-head image, macro presence/bits, captured raw rank identity and normalized order, policy/program, dtype/shape and immutable host output hash |
| `source_pair_sha256()` | Composition of those separate ingredients; not a device upload or runtime admission identity |

An external file-level `output.weight` remains an External proposal, even though a target trunk
head may also have semantic role `OutputProjection`. Student geometry, private NextN head roles,
tied target embeddings and per-head rank order remain explicit. A caller cannot attach a trim
made from another artifact merely because its binding or selected bytes happen to be equal.

Complete target/draft hashes cover the existing opened mappings, including shadowed components
and unused bytes through the reviewed artifact implementations. Selected materialization identity
remains separate: changing unrelated target bytes or a shadowed draft head changes the pair
without requiring a change to the selected output or semantic binding. Equal normalized text,
I32 and I64 rank order retains distinct raw input identity and therefore distinct pair receipts.

## Drift detection and consumer boundary

The target capture is checked before and after each operation. External draft opened identity is
captured before preparation and checked against the prepared binding and again after the callback.
Detected drift returns an error; no replacement pathname is opened or hashed. These checks cover
opened source extents and are explicitly drift detection, not an atomic writer-snapshot proof or
concurrent-writer exclusion. They cannot establish a point-in-time snapshot against every possible
concurrent mutation. Rank and host-trim payloads retain their existing immutable captures.

Hashing complete artifacts is deliberate work in these opt-in proof APIs. Existing default loaders
do not invoke them. The generic source callback may itself return a `Result`; a source receipt is
not a claim of successful device allocation. The engine wrappers propagate allocation errors.

`MtpHead::load_paired_draft` and `load_paired_composite_draft` return `(MtpHead,
DraftSourcePairIdentity)` through the unchanged `load_prepared_draft`. No fields were added to
MtpHead/HybridModel/DflashTrimHead and no existing load path was changed. The target argument is a
sealed source; this slice does not assert that an independently loaded HybridModel is that target.
The future runtime integration must establish that relationship.

The existing trim consumer can consume `pair.materialization()` without a new upload path.
`HeadTrimIdentity::output_sha256` still names the host payload before `GpuTensor::from_quant_bytes`.
That unchanged constructor can conditionally repack NVFP4 into splitA6 layout before H2D. This
receipt does not claim the final uploaded bytes/layout. Their digest, actual loaded model/binary,
numerical program, hardware and execution-surface qualification remain future requirements.

Shared RewriteIdentity/admission, `capture_rewrite_identity`, `frspec_src_sha16` refusal sentinel,
loaded-model identity, `d2t_from_target_head`, #542 fallback, worker/LRU, config/deps and defaults are
unchanged. Target Eager/DecodeGraph proof still cannot qualify an external or trimmed Spec proposal.

## Focused CPU evidence

Nine tests pass with zero failures/ignored controls: seven host consumer tests and two source
controls. The host harness compiles both new engine entry methods verbatim from hybrid.rs.
Only their existing native allocation body is replaced by a recorder. Actual source preparation,
bound reads, full source identities and the production NVFP4 macro reader execute. The real
`head_trim::load` dispatch executes with terminal H2D/quant-constructor recorders; final device
packing and GPU execution do not run.

Controls cover standalone and ordered composite external inputs, private and file-level heads,
natural/student/trimmed programs, exact d2t order, target/raw/external authority refusals, target
contract mismatch, consumer failure propagation, shadowed source identity, tied HF heads,
composite target overlays, distinct trunk/private/extra macros 2/3/5, BF16 conversion policy,
text/I32/I64 rank provenance, same-materialization/different-whole-target identities, pathname
replacement, and detected in-place target/draft drift before or during consumption. Runtime
identity values and host materialization receipts remain in the raw logs.

Source/harness all-target Clippy, Linux engine lib/tests Clippy, formatting and diff checks pass.
The Linux checks use `DOCS_RS=1`; they are typechecks, not native execution evidence. No broad
previous root/draft proof suites were rerun. The redundant harness-cast lint failure, subsequent formatting check, and their
pre-fix source pins/results are retained separately; final harness tests, lint and format pass.

`evidence.json` binds exact tested sources, commands, raw/compressed logs and retained CPU binaries.
The preservation audit checks 930 unchanged crate files and modes. Removing only the two new
engine methods restores hybrid.rs exactly to ab9; removing the module declarations restores the
two touched source/test roots. The extracted methods executed by the harness match source bytes.
Earlier head/rank/composite implementations, receipts, references and native archives are unchanged.

Source-pair review remains pending. No Spec/native/model/support/main/default promotion, GPU or
remote job, rental/topup, main merge, new agent or new issue lane occurred.
