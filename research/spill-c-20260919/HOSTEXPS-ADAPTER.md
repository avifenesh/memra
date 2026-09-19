# WP-C day-2 HostExps and PLE adapters

Repository **avifenesh/memra**, branch `lane/spill-c-20260919`. Existing engine
citations below are unchanged from `c5a33b14ff7b6c75a3cd808dbc0ee4f10aa8b33f`.
Frozen interfaces merged from `259ff819` by `0f52f63c`; no conflicts.
This is CPU adapter/control-plane evidence, **not native runtime qualification**.

## HostExps mapping

Portable code: `crates/memra-tier/src/bank/adapters.rs`.
Native bridge source: `crates/memra-engine/src/banked_residency.rs`.
The bridge is intentionally **not exported**: lead owns engine dependency/export wiring.
`cargo test -p memra-tier` compiles the actual bridge against an API-shaped CPU
HostExps fixture; it does **not** compile the CUDA engine or prove ABI integration.

- `HostExpsView` delegates to `is_uniform_layout()`, `expert_layout(original_id)`
  and `max_expert_bytes()`. The bridge checks metadata/tiers/macros vector lengths
  before indexing. Projection-wide qtype/row_bytes/stride never override `Some(layouts)`.
- `None` maps to `LayoutClass::Uniform`; `Some`, including a homogeneous subset,
  maps to `PerRecord`. The `Catalog` validates every full layout identity against
  the frozen canonical `RecordLayout::identity()` and preserves original IDs.
- Each `EncodingId` binds the exact runtime qtype under the domain-separated
  `host-exps-qtype-v1` digest, plus authoritative `row_bytes`; exact length and
  source extent stay explicit. Tests pin native Q2_K=13, Q3_K=4, NVFP4=7 against
  `memra-engine/src/lib.rs`, without executing/promoting any quantized kernel.
- Payload, macro scales and block-FP8 scales are independent typed byte segments
  in one atomic record. The bridge refuses omitted/extra scale classes and checks
  supplied scale hashes against the existing `HostExps` float bits. `macros=None`
  remains the existing implicit-one program, not fabricated scale storage.
- `ExpertSource` is loader-supplied immutable tensor/extent/checksum metadata.
  Slabs use the authoritative expert offset; split per-expert tiers use offset 0
  and their own original tensor identity. No double offset. The bridge verifies
  that this source class agrees with `HostExps.tiers`.
- Active masks retain router positions. A masked original ID has no source or
  backing assignment; a retained ID without its source/checksums refuses.
  Storage neither computes top-k nor reconstructs a missing expert.
- `with_uniform_experts(&UniformLease, ...)` is the new compile-time entry boundary.
  A compile-fail doctest rejects `BankLease`. Shared `UniformLease::try_new`
  additionally checks source class and actual leased layout geometry, returning
  owned inputs on refusal. This guarantee applies to the **new adapter API**;
  existing runtime kernels have not yet been changed to consume that proof.

### Exact existing dispatch sites and planned changes — NOT applied

| Existing source | Planned change after lead/native wiring |
|---|---|
| `model.rs:3158-3223` (`is_uniform_layout`, `expert_layout`, `max_expert_bytes`, `expert_source`) | Construct immutable original-ID catalog and exact source extents; retain existing source keepalives. Use max expert bytes for scratch, not projection stride. |
| `hybrid.rs:2053-2065` (`build_dev_exps`) and `hybrid.rs:2835-2838` (`has_uniform_expert_layout`) | Preserve all three gate/up/down uniform guards; resident slab construction must retain a checked uniform lease, not derive one from a selected subset. |
| `hybrid_forward.rs:13676-13677` | Replace detached uniform eligibility with checked source/lease proof while retaining all numeric, routing, shape and macro predicates. |
| `hybrid_forward.rs:13776`, `17892` (`moe_ffn_pairs`) | Require `UniformLease` in the new native wrapper; preserve real-prefill-only, no-macro, routing and clamp gates. Do not make pairs a mixed fallback. |
| `hybrid_forward.rs:13796-13825`, `18342` (`dev_ok`, `moe_ffn_dev`) | Same proof boundary for resident/pointer dispatch; retain decode/spec numeric-class restrictions. |
| `hybrid_forward.rs:13834-13837`, `13929-13937` (local slab / device pointer tables) | Retain owner-device locality and uniform proof alongside pointers through all last-use fences. No remote pointer substitution. |
| `hybrid_forward.rs:14131-14136`, `14524-14573` (grouped-decode dispatch) | Require uniform proof for fused kernels; mixed/Q2_K demand stays metadata-aware staged f32-dequant unless separately qualified. |
| `hybrid_forward.rs:14211-14234` (fused epilogue / verify-row arms) | Preserve uniform, shape, macro/clamp and numerical gates; tie pointer-table ownership to native leases. |
| `hybrid_forward.rs:14831-14846`, `19863` (`moe_cached_gemm`) | Demand original projection IDs from bank service, insert A's owner-ready wait, feed the same qmatvec byte range/encoding; keep macro folding unchanged. |
| `hybrid_forward.rs:14917-14965` (scratch staging) | Stage exact layout lengths into max-sized admitted scratch; preserve existing `qmatvec_view`, activation, and accumulation order. |
| `hybrid_forward.rs:21146`, `21159-21180`, `20063` | Keep active mask before grouped top-k; uniform resident grouped wrapper consumes proof. Do not confuse resident grouped with metadata-aware staging. |
| `hybrid_forward.rs:21212-21243` (grouped scratch/cache) | Preserve maximum per-record lengths and existing q8-vs-f32 numeric choice; change storage provenance only. |
| `moe_cache.rs:1100-1155` (`dispatch_source`) | Replace local byte identities with full BankId; adopt A transfer ticket/ReadyView; preserve source retain-before-submit and compute wait before publication. |
| `moe_cache.rs:1197-1237` (`prefetch_bytes`) | Retain pending-copy, slot and source lifetime; preserve unknown-copy quarantine and original SLRU policy. Prefetch never substitutes for exact demand. |

No changes to these dispatch sites, CUDA files, FFI, Hy3 routing, expert math, or
runtime defaults were made. Existing external CPU-expert code encountered in the
source is untouched and is not used by this implementation.

## PLE adapter and recorded n-gram trace

`PleTable` maps the existing `NgramTable::{F32,Bf16}` host row representation
(`qwen4exp_gpu.rs:276-303`) to canonical original-row BankIds. `last_chunk` consumes
full-history or cached IDs in exactly the last-t, token-major/head-major order used
at `qwen4exp_gpu.rs:13735-13760`, rejecting negative/out-of-table IDs and malformed
chunk lengths. It **does not** implement token hashing or mutate accepted history.

`fixtures/ple-ngram-synthetic.json` is explicitly **synthetic**, not a checkpoint
or GPU capture. Six steps cover decode, ragged growth, EOS, speculative proposals,
diverging rewind, and shorter request reuse. Geometry: max_ngram=3, one head per
ngram, multipliers [3,5,7], sizes [17,19], offsets [0,17], EOS=99.

The test-only `tests/bank/ple_oracle.rs` pins the existing native full and cached
host-ID functions plus their EOS helper. A source-body equality test detects
upstream drift (only test visibility/lint annotations and whitespace differ).
Both native oracles reproduce the recorded IDs before replay through RowService.
This is native **host-ID CPU** evidence at small synthetic geometry, not model
quality or hardware evidence.

Row replay checks native F32 bit preservation and BF16 expansion
`f32::from_bits(u32::from(bits) << 16)`, including duplicates and reordered rows.
No new row encoding, floating-point arithmetic, or projection program is introduced.
`expand_row` checks the exact BankId/layout before reading published host bytes.
The real `ple_block` replacement will gather these same raw rows, preserve the
existing expansion and `e.htod`/projection order, then retire after projection's
actual consumer fence. The existing native n-gram history remains authoritative.

`PleTable` itself is O(1) in logical table size. Day-2 services materialize only a
bounded requested catalog/trace window. A loader-backed lazy catalog/checksum
index and real persistent hot-row cache still need integration; this is not a
million-row loaded/pinned table or a model-scale cache claim.

## Coalescing decision (portable host policy, no runtime default)

`CoalescingPolicy::default()` selects **512-byte requested granularity, one 4KiB
slot**, based on day-1 arithmetic: sparse 264B×48 gives 1.9394× amplification at
512B, 15.5152× at 4KiB, 62.0606× at 16KiB. Packed adjacency gives
13,312/16,384/16,384 submitted bytes. The smaller granularity avoids needless
sparse overfetch; adjacent/overlapping extents of the **same original tensor**
coalesce. Different scale tensors never merge across namespaces.

This is a portable host-baseline policy, **not an SSD/O_DIRECT or hardware
performance promotion**. Native backends must supply their actual alignment
(e.g. 4KiB); tail padding must be explicitly readable or the plan refuses. Each
merged extent can exceed the slot; scatter progresses chunkwise with duplicate
logical outputs aliasing one allocation. `RowReadPlan.io_bytes` counts aligned
requested reads; actual physical SSD traffic remains unmeasured.

## Remaining native boundaries

- Current reads are synchronous bounded host work inside stage/gather. Tickets,
  Completion validation, publication, cancellation and last-use retirement are
  separate; producer-done is never labeled device-ready. Completion's per-segment
  copy lengths cover padded logical output; aligned read overhead is in RowReadPlan.
- A's asynchronous ObjectStore/TransferEngine, PinnedLease and owner ReadyView,
  real CUDA/graph fences, native SLRU reuse and real NVMe/pinned/UVA are pending.
  The host-only backend rejects requests asking it for device or pinned allocations.
- One injected frozen BudgetGovernor serves all allocations; no production C
  governor exists. B's production queues/fairness and actual allocator/RSS overhead
  calibration remain pending. Payload, slots, in-flight/tombstone metadata and
  resident metadata estimates are charged; no quota credit occurs via Drop.
- The catalog/source manifest is caller-owned immutable loader metadata, not an
  uncharged dynamically growing cache. Native loader budgeting must include it.
- All native Hy3/PLE byte/logit/token and performance cells remain required.


## Day-3 boundary update

The day-2 synchronous-I/O/governor-pending descriptions above are historical.
`BankService::stage`/`RowService::gather` now enqueue host work only; explicit
`progress` drives a bounded chunk. `ObjectReader` uses A's ObjectStore and
CpuTransfers, validates outcomes and calls `retired` before acknowledge. A
request-identity forwarding wrapper preserves current priority/deadline/tenant
while retaining the configured transfer byte budget; the same B Governor
instance now has real C/A contention tests. It is not a second governor.

This is still a **CPU drive-pump**, not an asynchronous OS worker or native CUDA
owner. A store lookup/lease currently verifies whole objects, and the table
catalog is eager metadata; lazy model-scale source indexing and bounded physical
validation I/O remain unresolved. Between-chunk cancel does not prove mid-DMA
retirement. Native dispatch still has no bank service installed.

`HY3-DISPATCH-PATCH.diff` is a partial mask-guard proposal, unapplied and not
native-typechecked; `PATCH-REVIEW.md` explicitly records NO-GO for the complete
migration. In particular, existing fused kernels still do **not** take
UniformLease. Do not promote the CPU compile-fail boundary to a runtime claim.
Trace search and new synthetic amplification table: `TRACE-AUDIT.md`.
