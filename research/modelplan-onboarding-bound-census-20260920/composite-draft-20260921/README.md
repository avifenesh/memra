# Ordered composite external-draft intake

This bounded slice follows reviewed `231a5a12ee8e161515ec9683bcefc57b893f0eb4`.
It binds an explicitly ordered set of opened GGUF components to the existing external NextN
contract and prepared engine consumer. Independent source review is pending.

## Input and selection contract

`CompositeExternalDraftInput::open(&paths)` accepts 2–64 existing GGUF components, highest
priority first. Each component may itself be a split GGUF. Every component must declare the
same complete normalized `ModelConfig`, including diagnostic fields, and fit one common
external-draft geometry. There is no config inheritance, new manifest/environment syntax,
mixed HF/repack intake, tensor-format conversion, or failed-read retry through a raw source.

Selection uses the first occurrence of each physical name. Existing private-head precedence
and alias ambiguity rules remain authoritative. Each weight's optional macro/input scale must
come from that weight's component; replacing a weight without a macro uses the existing absent
macro value of one and cannot borrow a lower component's scale. A scale-only component refuses.
The selected head owns its `d2t`; lower maps cannot supply the row interpretation of a replacement
head. A map without a component-owned head refuses. Cross-component private-head aliases and
components with incompatible full/trimmed/student geometry refuse rather than being reinterpreted.

Each physical component is checked against the common tensor contract, including shadowed rows,
before selected payload reads. Missing required union operands, extra/unknown tensors, malformed
shapes and ambiguous aliases refuse before the runtime callback. This is metadata validation of
inventory, not numerical qualification of shadowed payloads. Selected NVFP4 macros retain the
reviewed exact F32/width validation. Existing selected map validation and target-driven FFN
compatibility checks are reused unchanged. Only the first declared NextN block is executable,
as in the existing standalone consumer; later blocks and optional trunk copies remain inventory.

All operand reads use the selected bound record and its opened component. Required RoPE factors
now also pass through a bound tensor view before the existing numerical preparation helper.
The private layered source has no raw lookup implementation. Retained GGUF disk views keep their
selected opened storage alive after the input object is dropped.

## Identity ingredients and consumer seam

| Ingredient | Coverage |
|---|---|
| `CompositeDraftSourceIdentity::components()` | Ordered whole-component digests from existing `GgufSource::artifact_sha256`, including every opened split shard, shadowed payload and unused byte |
| `opened_source_sha256()` | Domain-separated, length-framed ordered aggregation of those component digests |
| `BoundArtifactIdentity::semantic_scope_sha256` | Existing prepared draft contract, target compatibility program, normalized token order, plus sorted component censuses and exact selected component ownership |
| `BoundArtifactIdentity::artifact_sha256` | Existing combination of complete opened-source and semantic/scope identities |
| Earlier head/rank materialization receipts | Separate selected runtime-head bytes, macro, captured rank source/order, and output/program evidence; unchanged by this slice |

Whole-source identity is not a selected-head/materialization receipt. Changing a shadowed payload
or unused shard annex changes the whole-source digest while selected bytes and semantic binding
can remain equal. Tensor header order is normalized for semantic metadata; full artifact bytes
still bind that order. Changing component priority changes the whole-source identity. Hashes use
the opened mappings, not a second pathname read; path replacement leaves an existing input intact.
This is not a new concurrent-writer lock or atomic filesystem snapshot guarantee.

`MtpHead::load_composite_draft` calls the typed input's `with_runtime`, then the existing
`load_prepared_draft`. That unchanged consumer records `prepared.artifact_identity()` before its
first `load_t` allocation and retains it through `external_source_identity()`. Existing standalone
`load_draft`, prepared loading, public head layouts and all four trim upload sites are byte-equal
to the reviewed base after removing only the new method. External-draft authority still refuses
model-root and target-head-trim use. Shared `RewriteIdentity`, admission, #542 fallback, worker/LRU,
config, dependencies and defaults are unchanged. Parent was sent the owned API seam before freeze.

## Focused CPU evidence

Seven real-source harness tests and fourteen affected standalone draft controls pass: 21 passed,
zero failed, zero ignored. The new harness opens independently specified real GGUF fixtures and
executes the production NVFP4 macro reader. It checks selected encoded bytes, distinct upper/lower
macros, lower attention operands, student output-up geometry, same-component trimmed maps,
shadowed metadata refusals, no inherited macros, unknown/out-of-plan refusal, root-authority
separation and retained disk-view lifetime. Whole-identity controls cover priority, path replacement,
shadowed weight mutation, split entry-point equivalence and unused bytes in a shadowed second shard.
Actual runtime/component hashes are retained in the raw test log.

The fourteen standalone controls exercise natural/student/Step geometry, RoPE factors, activation
compatibility, macro encoding, trimmed I32/I64 maps and refusal boundaries after the common
compiler's bound-read change. No broad frozen root suite was rerun. Source and harness all-target
Clippy, Linux engine lib/tests Clippy, both format checks and diff checks pass. Linux engine checks
use `DOCS_RS=1` stubs: the new MtpHead entry is typechecked and its consumer path inspected, not
executed on a GPU. The CPU harness executes source preparation/materialization and the actual macro
reader; it does not execute the whole GPU loader.

`evidence.json` binds the tested source files, exact commands, compressed/raw logs and two retained
CPU executables. `raw/preservation.json.gz` records 930 unchanged crate-file hashes/modes, the exact
engine insertion audit and harness registry versions/checksums matching the root lock. Development
failures remain separate: initial borrowed-view lifetime error; incorrect expert-disk test probe;
unsorted metadata identity; and two audit-script assumptions (optional Cargo dependency graph and
an incorrect allocation symbol). Final code and final checks pass.

This is a source-review candidate. Paired target/draft rewrite identity, external Spec admission,
native numeric/serving/worker/H2D/I/O qualification, model support and main integration remain
separate. No GPU, remote job, rental/topup, main merge or default change occurred.
