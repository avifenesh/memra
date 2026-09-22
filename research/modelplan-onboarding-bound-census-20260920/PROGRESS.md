# #541 progress — 2026-09-21

The reviewed adapter stages through `caf52b441bbf11a5cf4ddf2624ff9c2d99184aea` are now
composed with the actual reviewed #542 `68e5e520810fd105bf64f43048a3a88514f6a064` dependency;
see [the transfer intake receipts](transfer-intake-20260921/README.md). The exact composition
`a49946a30e867697ee5568627bbe040a77d06d65` passed the original finder and independent source
review. A later dependency repair awaits the coordinator's accepted final ref; it has not been
substituted into the current CPU source snapshot.
Root loader activation and whole-issue completion remain pending. No GPU execution, hardware
performance/default decision, positive model support-state change, or main merge occurred here.
The existing native Step vision route and its historical qualification remain intact; its canonical
catalog inventory is not a claim that native vision is absent.

## Frozen stages

| Head | Stage | Source review |
|---|---|---|
| `cbce8e0b4b14adba769cd6ef79c7e5bafa3cdd80` | Exact auxiliary accounting, including floating unselected inventory | Independent PASS; original 1,597-header catalog still complete |
| `c3aaab083469d5cc2b102df6c8ed6779ad4dd407` | Opaque disk readers and engine spill consumers, including `71f4d0a8e` | Independent PASS; native I/O/H2D not run |
| `ce0be1eb9b5186214e84b9f218b3abf5b71b8e7f` | Actual reviewed #542 dependency merge from `8c514a91` | Conflict preservation reviewed as part of composition |
| `1e17fbba5d734ce2eadf1e49441129cb8d36e38f` | Opened-source + semantic/scope identity and source metadata | Initial preflight-hook defect subsequently repaired in `5e3fc6174` |
| `343b3bec0dc145209f7e1d9486f303cf9718134a` | Actual reviewed `222d50405` storage dependency merge | 197 ordinary tier checks pass; SLRU rows unchanged after source-pin regeneration |
| `54cc1e0a7a210bb7323504611c049288e4463e66` | Retained parsed GGUF shard views | Independent PASS |
| `c2bcfe858f0d9aa428224b32fb403189b9f2a9b9` | Handle-free GGUF residency/MTP accounting | Independent PASS with `49b24d331` review |
| `5e3fc617430c473847bd3276f3d44b032f7e2d00` | Compiler-private preflight authority; TC541-IDENTITY-01 | Original finder and independent reviewer both PASS; exact old/new probes retained |
| `49b24d33146eab508b195d6c0e8df1a741de80e9` | Explicit GGUF spill through checked windows; ordinary host policy preserved | Independent PASS; CPU placement/failure-counter controls pass; native gate pending |
| `caf52b441bbf11a5cf4ddf2624ff9c2d99184aea` | Complete internal repack bound access | Independent PASS; strict identity and overlays remain explicitly unsupported by this adapter |
| `a49946a30e867697ee5568627bbe040a77d06d65` | Reviewed transfer/router/KV dependency intake | Original finder and independent source PASS; native qualification remains separate |
| `ed67b9a8403684cc702467e9ba931c03ff5e15fb` | Retained bounded reader C ABI | Independent PASS; C retain/drop/read/layout controls pass |
| `65dae02279d2a10f03f055b6907c421a0fd1f2c2` | Scoped CPU companion and deterministic detached-prefetch ownership | Independent PASS; warning-clean GCC/OpenMP and buffered/direct controls pass |
| `a055f272be593212436dd07788ae2f17f8410f56` | Actual Rust-to-C++ scoped bridge | Independent PASS within executed scope; initial unaligned direct-labelled windows used buffered fallback |
| `af0cf8bcf6d6ceb0b1642a2e0887ed9ccba33135` | Aligned direct Rust reads and production prefetch helper | Independent PASS; retained O_DIRECT descriptors, detached ownership and annex bytes exercised |
| `bdd377a0c120f694dfb809e91fba19cf440e0681` | Verified mirrored scoped readers and original ABI parity | HOLD: ARCH541-CPU-MIRROR-01; earlier 21 zero-exit commands did not detect signed counter underflow |
| `291c1ef5c8a610131748645ac1d1666a121c71ef` | Signed mirror-counter regression without production repair | Expected red reproduced in four controls; exact raw bank retained |
| `8852faac7a88dba2493cc7d784deff06afbb3e0f` | Release one prefetch charge after final projection half | Original finder CLOSED ARCH541-CPU-MIRROR-01; bounded source PASS, all 25 CPU commands pass |
| `82a28b225557f17ff80bb9a8daf08dec33daa752` | Complete internal repack opened-artifact identity | Independent source PASS; mask/fallback refusal remains; no root/native promotion |
| `2e7ce065aea400aa512dc5b587ff2c776344b639` | Typed retained original IDs, tensor groups and compact reference banks | Independent source PASS; six new controls pass; every optimized surface blocked |
| `86751a19cd577058453cfe70024484d7d28f0883` | Self-contained retained repack binding | HOLD: escaped JSON declarations bypass legacy interpretation; original red probe preserved |
| `625aad049d5b5c87689efc047452364eb726c979` | One decoded JSON tree for repack declarations | Original finder CLOSED ARCH541-REPACK-JSON-01; bounded retained-repack source PASS |
| `c3fba6b25acfeb640342c5aebf0e1d25fc3ce2cc` | Physical component inventory and bounded iterative opening | Carver source PASS; six independent controls and 933 blobs verified |

Each stage's neighboring receipt directory retains source hashes and losslessly compressed raw
logs, including intermediate failures. Historical review passes are scoped to their immutable
heads. The original `343b` tuple-hook finding is preserved, not overwritten by its repair. Its
failed lint log and separately successful repair lint log also remain unchanged.

## Current checks and limits

Reviewed complete-repack identity stage at `82a28b225`: 365 GGUF tests pass (2 ignored), 13 CLI, 1 Step integration,
7 inspector, 3 external-source authority controls, and 2 compile-fail doctests pass. Host
GGUF/CLI all-target warnings-denied Clippy, Linux-target engine/server typechecks and formatting
pass. Linux engine checks use `DOCS_RS=1` documentation stubs; they are not executed CUDA evidence.

Identity/snapshot host controls and native-repack host controls passed when composing the reviewed
identity dependency. The scoped-GGUF CUDA gate for zero/partial/full pin budgets is implemented
and typechecked but ignored until hardware admission. Existing raw numeric/serving and disk
worker gates remain required. The known broader reference numeric caveat remains recorded in
[the original validation bank](VALIDATION.md).

The retained-expert candidate passes 368 GGUF tests (2 ignored) and the compiler/CLI gates.
Its full reference suite is 68 pass / 1 fail: the existing [#548](https://github.com/avifenesh/memra/issues/548)
Qwen3.5 macOS bit pin reproduces identically at frozen pre-change `82a28b225`. Both raw failures
and the unchanged unpruned plan/manifest hash comparison are preserved in the
[retained-expert bank](retained-expert-plan-20260921/README.md).

## Remaining acceptance work

1. The reviewed #537 canonical decoder/config change `bb637184` is now intaken at `b45d06547`,
   including numeric lexemes, the atomic Cargo feature pair, typed head ownership and captured
   raw bytes. Combined CPU checks pass; the complete intake/materialization source awaits review.
   Root activation remains held.
2. Intake the coordinator's accepted final #542 dependency ref when supplied, preserving the
   private repack, execution snapshot, source identity and storage contracts. The existing
   `68e5e520` composition has source approval; native qualification remains pending on this tree.
3. The [mirrored-prefetch counter finding](cpu-mirror-20260921/COUNTER-REGRESSION.md),
   issue #586, is closed on exact `8852faac7` with bounded source approval. The reader ABI, actual Rust bridge,
   aligned direct reads and production prefetch helper have bounded source approval and CPU
   receipts. Positive predictor routing itself remains unexecuted. The bdd mirror result bank
   remains explicitly source_GO=false; do not reinterpret its clamped statistics as correctness.
4. The new composite source stage implements typed overlay masks, original-ID member selection,
   materialization and opened identity in an explicit source-bound API, with CPU/reference controls;
   independent review is pending. Ordinary root fallback entrypoints still refuse. Standalone
   MTP/student/trimmed-draft contracts, remaining derived/native coverage, independent auxiliary
   groups, and private canonical repack output/cache-target boundaries remain unfinished.
5. After those seams and the latest dependency pass source review, activate dense/hybrid roots on
   one sealed bundle, then run the target-rig numeric, serving-shape, worker/H2D and I/O qualification
   under the coordinator's physical card locks. No support promotion from synthetic fixtures.

The original frozen source checkout is clean and retains `c3aaab083`. Its saved GGUF patch was
fully preserved in the integration commit and raw bank before clearing that owned working change.
The integration branch retains every later stage; no unfinished stash remains.

## Composite catalog continuation

The [composite catalog](composite-catalog-20260921/README.md) validates every physical
fragment before precedence and seals selected member ownership, dialect and transform as
metadata. Eleven focused controls, GGUF395/2ignored, compiler/CLI, identity18 and host/Linux
stub lint checks pass. Independent source review is pending. Runtime fallback and identity
refusals remain; split GGUF member/mask contracts and actual composite materialization are
still outstanding. The shared #537 numeric/config dependency `bb637184` remains disjoint and
pending review; it will be integrated atomically with its features and consumers.

## Composite source continuation

The reviewed #537 decoder/config stack is integrated at
`b45d065471d8e1c2498f9c5b6bb18b632ec7d95c`; typed head policy and captured config bytes are
preserved. The [composite source stage](composite-materialization-20260921/README.md) adds
inherited typed masks, original-ID member selection, selected-component tensor/native-plane
materialization, bounded disk views, source-factor validation and complete opened identity.
Twenty-four focused checks and a full loaded-weight/logit reference comparison pass, together
with compiler/CLI, feature-unification, identity and lint checks. Legacy standalone identities
match the frozen baseline. This source awaits independent review; no root/native/main promotion.
The ordinary fallback TensorSource boundary still refuses, while the explicit source-bound API
is exercised by CPU controls. Final runtime adoption and the remaining draft/cache/native gates
remain outstanding.

## Canonical-output prerequisite

Carver approved exact `5a805c39` for the combined config/composite source slice. The next
[canonical-output boundary](canonical-output-20260921/README.md) captures directory authority
at source open and derives private output only from bound operands. Nine unique CPU controls,
compiler/CLI, reference-composite, identity and lint checks pass. Engine consumers, root adoption,
draft contracts and native qualification remain pending; no #542 correction WIP was imported.

## Canonical-output engine consumer

Carver approved exact `853099a1` for retained canonical output. The
[consumer delta](canonical-consumer-20260921/README.md) routes stacked and gathered NVFP4 disk
loads through bound canonical views before named caches, preserving macros, pinned prefixes
and whole-slab policy. Production CPU harness10 and GGUF420/2ignored pass; engine-library
DOCS_RS strict lint passes. Existing targets were reused with incremental caching disabled.
The consumer source awaits review. Root entrypoints, fallback, worker and config seams are
unchanged; full root/draft/native acceptance remains outstanding.

## Actual root adapter integration

Reviewed consumer `725551d7` is now the base for [root adoption](root-integration-20260921/README.md).
Dense and hybrid roots invoke PreparedModelSource before their existing load bodies. Ordinary
and composite semantic adapters, private preflight authority and compiled placement charges
are integrated. Five root controls, four real production placement controls and complete
root-ABI weight/logit parity pass; all eight final checks are green. The source awaits independent
review. No native/model/main promotion is claimed; external/trimmed draft contracts and native
qualification remain outstanding. Parent-owned fallback, worker and device-selection logic
are preserved.

## Standalone external draft integration

Root source is frozen at `3a0f1ed71c804410498c46a2420617fbb380ede6` for independent review.
The separate [external draft candidate](external-draft-20260921/README.md) adds typed
natural/student/trimmed GGUF binding and production `MtpHead::load_draft` adoption. Twelve
focused real-file CPU tests, source Clippy and Linux DOCS_RS engine lib/test Clippy pass.
Private head precedence, map order/range, unselected inventory and opened identity are bound;
draft-only authority refuses model-root reentry. The complete target/draft rewrite-identity
composition and separate self-trim/ranks surface remain pending. Source review is pending;
no native/model/default promotion or remote launch occurred. The frozen root suite was not rerun.

## Mandatory Gemma root review repair

Carver found `ARCH541-ROOT-01/02` on immutable root3a; the1254 draft review is checkpoint-only,
not GO. The [separate repair](root-repair-20260921/README.md) restores registered Gemma-MoE
callback entry and dense optional-probe absence using semantic aliases, while preserving the
physical GGUF refusal. Both original real-file repros fail before and pass after; new Gemma3,
existing runtime4/composite2/draft12 and strict source/engine typechecks pass. All919 other crate
files are unchanged from1254. Independent finding closure and draft-review completion remain
pending in Carver's original task. Further acceptance work and all remote/native work stay held.

## Mandatory draft consumer review repair

Carver closed ROOT01/02 at804. Completed1254 review then found `ARCH541-DRAFT-01/02`:
target-driven FFN activation mismatch and short NVFP4 macros reaching an F32 reader. The
[bounded repair](draft-repair-20260921/README.md) shares activation selection with the actual
consumer and rejects selected non-F32 macros before allocation. Carver's original probe
fails before/passes after; draft14 and production-consumer harness6 pass, with strict source,
harness and engine typechecks. Source/consumer call-path and916 protected-file hashes are
banked. Independent closure remains pending; other acceptance work and all remote/native,
default/main/rental actions remain held.

## Opened rank intake dependency

Carver closed DRAFT01/02 atf870. The next [bounded rank-intake slice](rank-intake-20260921/README.md)
captures rank order/encoding and full source identity from the same consumed buffers, then
shares one captured input across the three existing self-trim intake sites. Path replacement,
in-place edits, raw/normalized identity separation and split-source controls pass. Rank8,
production consumer3 and existing draft14 tests pass with strict source/harness/engine typechecks.
The parent relayed future identity coordination to542; shared rewrite/admission interfaces and
all Spec/native/support gates remain unchanged. This source slice awaits independent review;
target-head materialization and paired identity remain separate work.

## Bound target-head materialization

Carver passed rank intake atb863. The [next slice](head-trim-20260921/README.md) binds target
head ownership, its own macro and captured ranks into an immutable host payload and receipt.
Normal, skip-stub, extra-head and GLM trim uploads consume that object; private/extra macros
are preserved. Source3, actual consumer7 and draft compatibility14 pass, plus strict source,
harness and engine typechecks. The receipt names the selected runtime-head image and produced
payload; full target/draft identity and native qualification remain separate. Shared rewrite
interfaces and all support/default gates are unchanged. This slice awaits independent review.

## Ordered composite external drafts after reviewed 231a5a12

The next [bounded composite draft slice](composite-draft-20260921/README.md) adds an explicit
ordered GGUF input, common-program inventory validation, component-owned weights/scales/maps,
and complete opened component/aggregate identities through the existing prepared MtpHead consumer.
Seven real-source consumer controls and fourteen affected standalone controls pass (21/0/0), with
source/harness and Linux engine warnings-denied checks. Source review is pending for this new
slice. All previous refs and banks remain immutable; head/rank receipts remain separate from whole
source identity. Shared RewriteIdentity/admission, #542 fallback, worker/LRU, native/model/Spec
qualification, parent main integration and capacity ownership remain unchanged.

## Source-pair receipts after reviewed ab9dcfc0

The next [owned source-pair slice](draft-pair-20260921/README.md) composes sealed complete target
identity with separately typed external-draft identity or newly prepared immutable target-head
materialization. Ordered raw component hashes, selected binding/head/macro/rank receipts and
host output identity remain distinct. Two opt-in engine entry methods return the source receipt
through the unchanged prepared consumer. Nine focused CPU controls pass (9/0/0), including actual
entry-method execution with native recorders, composite/tied targets and detected opened-source
drift. Independent source review is pending. Before/after checks are not atomic writer snapshots.
Final uploaded bytes/layout, binary/numerical identity and runtime admission remain future work;
#542 interfaces/sentinels, existing defaults and parent integration/native/capacity ownership remain
unchanged. The concrete API proposal was relayed by the parent before implementation.

## Owned final uploads after reviewed 57de5874

The [final-upload candidate](final-upload-20260921/README.md) retains actual returned GPU tensors
privately with source/host and same-buffer submitted identities, including NVFP4 splitA6 layout,
shape/byte/qtype/macro/device bounds and ordered extra heads. Sixteen focused CPU controls pass
(8 each under RP0/RP1), comparing the exact old/new constructors and real upload arguments through
terminal recorders. Source review is pending; no native proof is claimed. Pinned compatibility
records exact54224668 and live GitHub main653, including the reviewed541 module prerequisite;
no broad intake occurred. Per the parent and542 seam review, complete actual-model slot/length/
truncation association, the load barrier, typed early strict preparation, manifest eligibility,
binary/numeric binding and shared admission remain a separate future composition. All prior refs
and proof banks remain immutable.

## Root trim completion after reviewed 341188ae

The [root-trim candidate](root-trim-20260922/README.md) adds typed early preparation, private
block/slot reservations, exact owned-tensor moves after the actual model load barrier, and a
separate complete-proof capture path. Ordinary capture remains byte-identical; proof-mode loads
start StrictPending and require explicit matching Spec receipts. Eleven focused CPU protocol
controls pass; native operations are stand-ins. The actual hash-pinned Qwen3.8-27B MTP metadata
is MtpSpec-eligible but Eager-blocked by GatedDeltaNet/RecurrentState/FusedAttentionGate, so real
target admission remains refused and the parent/#542 own that prerequisite. Shared compiler
selection is consumed by typed surfaces with exact descriptors retained; no registry fork/intake.
Fresh observed #5423c5cb and main81d75 are recorded separately from compatibility246/653. Source
review and later exact combined native qualification remain pending; prior refs/evidence unchanged.
