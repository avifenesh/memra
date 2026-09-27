# Typed root trim completion and private capture

This bounded candidate follows reviewed `341188ae5bdb4b57d016e5fd14042a790b17ae99`.
It connects the reviewed source/host/upload owners to the actual HybridModel root, then seals a
private proof after the existing load barrier. Independent source review is pending. No branch
intake or native qualification occurred.

## Early preparation and exact loaded slots

Strict identity loading with a nonempty `MEMRA_FRSPEC_TRIM` enters `RootTrimLoad::begin` before
parallel preparation or any weight upload. It requires the actual sealed text target, embedded
MTP blocks, active MTP/trim mode, and eligible compiler-selected DecodeEager and MtpSpec programs.
The existing external MTP/draft/DFlash selectors remain refused, as do skip/stub/full-precision
substitutions. Ordinary loading and the ordinary capture wrapper remain unchanged.

Preparation captures the rank input through the existing reader and materializes all selected
host heads from the same target. A captured option set drives the cap/policy decisions and is
checked again after preparation and before completion. The existing cap and missing-private-head
truncation rules remain: a missing extra private projection ends the chain; the existing root's
multi-head dense/block-count restriction is preserved, including its refusal of partial multi-head
counts. Complete target identity remains separate from selected source/head/rank/upload identities.

Each embedded MtpHead records its private source block index at construction. During proof-mode
staging, a private reservation identifies the exact head slot and `UploadedHeadTrim` retains its
actual returned GPU tensor. The old full head is dropped at its normal replacement point, leaving
an internal reserved slot; no unfinished model is returned or executed. Block origins and live
reservations reject reordered or substituted slots even if dimensions and token maps are equal.

After constructing the actual HybridModel, completion validates requested/loaded/selected counts,
truncation observation, all reservations, block indices, exact d2t order, head origin, rank sentinel,
source/config/program identity and device placement. It calls `sync_stages_after_load`, repeats
source/model checks, and moves the exact privately owned tensors into those reserved model slots
under an exclusive mutable borrow. Missing/extra/reordered/substituted slots, wrong mappings,
partial uploads or barrier failure cannot produce the complete proof. No portable CUDA pointer
identity is introduced.

The private linear `CompleteTargetTrimProof` belongs to the actual model's ProgramGeneration and
its post-installation mutation count. Cross-model use and later mutable program access refuse;
the existing TrackedProgram execution-snapshot revocation remains authoritative. Source comparisons
are detected-drift checks, not an atomic multi-file or concurrent-writer/ABA snapshot guarantee.

## Shared program selection and capture

The root queries the shared compiler `execution_rewrites(plan)` by typed DecodeEager/MtpSpec
surfaces. It does not assume a generic rewrite ID or duplicate a family-specific Eager selector.
The exact selected compiler descriptors—ID, implementation, plan digest and operation set—are
retained, revalidated against the loaded model, and included in the complete proof and numeric
interpretation. The parent confirmed this is the shared selection route; future #542 composition
must consume its canonical selector changes through that route without a fork.

A separate private `capture_target_trim_rewrite_identity` accepts only the completed proof. It
reuses the existing executable, mapped-library, environment, hardware and loaded-model numeric
capture helpers. The target/source/ordered source-host-upload/program descriptors form a distinct
artifact identity. `capture_rewrite_identity`, its existing refusal behavior, the public identity
schema, qualification matcher and mutation guards remain byte-identical.

Proof-mode loads start StrictPending even for identity capture without a bundle. Capturing an
identity therefore grants no execution surface. Matching Eager/DecodeGraph receipts do not grant
MtpSpec: it requires its own eligible manifest and exact matching receipt. Target-only, different
rank/pair, binary or numeric identities refuse. External/DFlash/stub paths remain outside this
exception, and `frspec_src_sha16` remains present and must match the captured raw rank identity.

## Actual target prerequisite: still refused

Read-only opened-file inspection pinned the existing artifact:

- `/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`
- SHA-256 `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`, 15,705,922,304 bytes.
- Actual metadata/header: Qwen35, 65 layers including one NextN block, hidden 5120, vocab 248320.
- Plan digest `ba7c980ab6201a25c0534493aab7f0fd1d90507f80797c7ad88807a0057464fc`.

At this source, MtpSpec is structurally eligible; DecodeEager is **ineligible** with blockers
`GatedDeltaNet`, `RecurrentState`, and `FusedAttentionGate`. The typed root therefore refuses this
real target before uploads. Its legacy/v3 Spec positives do not supply the missing Eager declaration
or a paired native receipt. The parent relayed this canonical prerequisite to #542; no registry or
selector changes were made here, and no admission claim is made for this model.

The full-file checksum and captured metadata/embedding header came from the same opened hash
stream with stat drift checks. The initial metadata-only witness correctly failed because vocab
comes from the physical embedding header. Both failures and the corrected partial-header witness
recipe are preserved. The local witness is sparse, contains only the embedding header and no
supplied model weights, and fails the complete tensor census. It is solely a config/manifest
inspection witness, never a load/admission artifact. Large header archives remain hash-pinned in
the parent artifact directory; they are not substituted for original model bytes.

## CPU validation and preservation

Eleven focused tests pass, zero failed/ignored. The harness executes the actual root preparation,
staging, slot reservation/completion and private capture code; actual source binding/materialization,
upload logic, executable hashing, numeric framing, qualification parsing and generation/snapshot
guards are used. H2D, the barrier, unavailable native process-library/hardware inspection and the
full native model-state provider are explicit host stand-ins. Actual HybridModel loading is source-
traced and Linux-typechecked; GPU allocation, visibility, readback, inference and native parity are
not executed by these tests.

Controls cover early raw/external/stub/rank refusal; exact three-slot allocation moves after the
barrier; missing/duplicate/reordered/substituted slots; mapping/origin/sentinel errors; partial-copy
and barrier failure; source/options drift; cap/truncation behavior; model-specific proof and mutation
revocation; actual executable identity; pair/binary/numeric receipt mismatches; explicit MtpSpec
qualification; and Eager-eligible targets with no embedded Spec program. Synthetic token receipts
are temporary matcher fixtures, not native evidence. The actual-model eligibility command exits
with its expected refusal and records the blockers above.

Warnings-denied harness and Linux DOCS_RS engine lib/tests Clippy, formatting and diff checks pass.
No broad old suite was rerun. The preservation audit checks 933 unchanged crate files/modes,
byte-equal ordinary capture, unchanged public upload behavior after removing the private assignment
helpers, and unchanged hybrid code outside the root body/private origin fields. It also verifies
begin-before-upload and construct/barrier/install/capture ordering. Earlier compiler/import-order
and witness failures remain preserved separately.

## Composition boundary

The earlier compatibility pins remain #542 `24668af517c83f5bbee64de0a0f0a16c1f4dec74` and main
`653c997f445ed46dade7cf79b3437894114ae49d`. Fresh observation for this freeze found #542
`3c5cb6822d0c0fb22668338ca9de157031062238` and GitHub main
`81d75c457141f1b1c34864fd2a3ba44853675cec`. #542 capture/root/model/snapshot seams have no changes
or WIP relative to246; its registry/manifest work remains separately owned. No broad intake or
native-equivalence claim is made. Later parent-selected composition must recheck main, preserve
RetainedExpertRouting's NONE row/count68/new Eager column false, regenerate tables and run both
registry/compiler suites. Fresh native proof requires exact combined source, binaries and program
under the existing per-UUID protocol.

No model-support/default/Spec/native promotion, GPU job, rental/topup, main merge, unrelated host
write or new actor occurred. Prior reviewed refs and proof banks remain immutable.
