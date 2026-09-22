# Owned final head-upload receipts

This bounded slice follows reviewed `57de5874a94a8a5b520373a5344adbff35e9a280`.
The parent requested freezing the owned upload layer before shared root/admission integration.
The exact #542 seam review is banked under `reviewed_input`; independent review of this new
slice is pending. No broad #542/main intake occurred.

## Source, host and submitted-buffer boundaries

`UploadedHeadTrim::load` accepts only `PreparedPairedHeadTrim`, whose immutable host bytes and
source-pair receipt were produced together by the reviewed compiler/materializer. It performs the
actual backend upload and retains the returned `GpuTensor` privately with its source and upload
identities. There is no public constructor accepting a tensor, callback result, hash or receipt,
and no mutable tensor accessor. The old `head_trim::load` path remains byte-for-byte unchanged.

The recorded quant path and ordinary `GpuTensor::from_quant_bytes` share one constructor and the
existing qtype mapping, NVFP4 repack and `MEMRA_RP` switch. The opt-in path captures the host input
into an owned buffer, validates complete encoded rows and exact checked byte/shape bounds, and
hashes the same final buffer passed to `Engine::htod_bytes`. Ordinary construction keeps its
existing behavior without hashing or a host-input clone. BF16 uses the exact captured byte buffer;
F32 uses the actual native byte representation of the immutable f32 allocation passed to `htod`.
No other conversion, source lookup or format fallback is introduced.

The constructor returns an identity only after the backend copy succeeds and its returned tensor
passes dtype/qtype, shape, row-byte, extent, macro-bit, layout and unexpected-mirror checks. The
opaque owner also checks the returned device. The layout validator verifies source-to-submitted
bytes: exact raw/BF16 equality, F32 little-endian source to native representation, or the exact
NVFP4 splitA6 permutation. A changed layout tag or buffer therefore cannot pass on equal lengths.

| Identity | Coverage |
|---|---|
| Existing source pair | Complete target/artifact identity, selected binding/head/macro/rank and host materialization; retained without reinterpretation |
| `FinalUploadIdentity` | Host payload hash; actual submitted-buffer hash and length; dtype; shape; qtype/row-layout/repack class; macro bits; device ordinal |
| `UploadedHeadTrim::sha256` | Domain-separated, framed source-pair plus final-upload identity |
| `UploadedTrimChain::sha256` | Ordered composition of the privately owned uploaded heads |

No CUDA pointer is serialized. These are receipts for the final host operands submitted through
the backend API, not device readback, native numerical qualification or a complete model-load
proof. Cross-stream visibility/order must still be established at the actual constructed-model
`sync_stages_after_load` barrier before future admission authority can be minted.

## Ordered heads and current limits

`UploadedTrimChain` validates all provided entries before uploading: a nonempty ordered prefix of
first/extra MTP head roles, one complete target identity and one captured rank identity. First
output/tied embedding or private depth-zero head is explicit; extra roles must match their slot
index. Gaps, duplicates, reordering, different targets and different rank provenance refuse.
Each retained head binds its own physical/source materialization and macro. Failed or partial
copies return no opaque chain; previously allocated local objects are dropped normally.

This sequence does not assert the independently loaded HybridModel's selected chain length or
truncation. The future root integration must associate every actual loaded slot with its privately
owned tensor/receipt, preserve loader-selected chain length and exact d2t semantics, and refuse
missing/substituted/extra slots relative to that actual model. Device allocation association must
remain private rather than becoming portable pointer identity.

Shared capture/admission is unchanged and remains a separate parent-selected composition. The
#542 review requires typed early preparation because strict preflight rejects MEMRA_FRSPEC_TRIM
before uploads; a final-capture-only exception is insufficient. All other external/DFlash/stub
refusals remain required. `frspec_src_sha16` stays present and may eventually be accepted only with
a complete matching trim-load proof. TrackedProgram mutation revocation stays authoritative.

Before claiming admission, the parent must select the actual embedded-MTP target and verify both
Eager and MtpSpec manifest eligibility. A complete future proof must also bind the loaded target,
visibility barrier, running executable/libraries, environment, hardware and numerical program.
Matching Eager/DecodeGraph receipts cannot grant MtpSpec. Composition with #542's Gemma Eager
column must retain the `RetainedExpertRouting` NONE row/count 68 with that column false, regenerate
tables and run registry/compiler tests. None of that shared work is included in this upload slice.

## Focused CPU evidence

Eight controls pass in each of two fresh processes, `MEMRA_RP=0` and `MEMRA_RP=1`: 16 passed,
zero failed, zero ignored. The harness compiles the actual current constructor, qtype helper,
repack and switch verbatim. It also executes the exact frozen57de constructor body (only its
method name changes), with the same terminal recorder. All nine existing quant encodings produce
matching old/new ordinary/recorded bytes; no broad frozen root suite is rerun.

Real complete GGUF fixtures provide three ordered heads and distinct macros 3/5/7. Controls check
independently specified splitA6 planes against the actual H2D arguments and returned tensors;
F32/BF16 equality with the existing upload dispatch; zero/short/overflow/misaligned/unsupported
inputs; qtype/shape/macro/row-layout/mirror/buffer mismatches; wrong returned extent/device; failure
on the second copy; target/rank/order mismatches before any copy; and equal submitted bytes with
different macro identities. The actual production upload/identity logic executes with terminal
H2D recorders. There was no GPU allocation, device readback or native execution.

Harness all-target Clippy, Linux DOCS_RS engine lib/tests Clippy, both formatting checks and diff
checks pass. Linux DOCS_RS is typecheck evidence only. Final source hashes, exact commands, raw
logs and one preserved ARM64 CPU executable are recorded in `evidence.json`. Earlier development
logs, including the initial unused import and pre-baseline runs, remain separately preserved.
The preservation audit verifies 934 unchanged crate files/modes and the exact baseline/current
constructor extractions. The old head-trim dispatch, source/rank/pair implementations, root,
RewriteIdentity/admission, worker/LRU, numerical kernels, config/deps and defaults are unchanged.

## Pinned compatibility

Fresh GitHub `ls-remote` checks, including the pre-freeze refresh, resolve main to
`653c997f445ed46dade7cf79b3437894114ae49d`. The reviewed #542 pin is
`24668af517c83f5bbee64de0a0f0a16c1f4dec74`. Original quant constructor, split repack and RP switch
bytes and qtype constants match exactly across those pins and57de. The numeric patch hunks
apply cleanly to all three copies.

The full patch's module-registration context requires #541's reviewed `nvfp4_scale` prerequisite,
which is absent in raw #542/main. Those direct whole-patch dry runs fail at registration and are
recorded honestly; the parent must compose the reviewed #541 series. This is a pinned compatibility
packet, not an intake, successful combined build or native-equivalence claim. Later joint intake
must select one frozen #542 revision and recheck current GitHub main.

No root/native/Spec/model/support/default promotion, main merge, remote dispatch, GPU allocation,
rental/topup or new agent occurred. The complete loaded-target/upload/binary/numeric admission
integration remains future work under the explicit #542 and parent boundaries above.
