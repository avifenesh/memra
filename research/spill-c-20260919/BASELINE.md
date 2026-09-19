# WP-C day-1 baseline — 2026-09-19

Repository `avifenesh/memra`, inspected baseline
`c5a33b14ff7b6c75a3cd808dbc0ee4f10aa8b33f`.
Branch `lane/spill-c-20260919`; isolated worktree `../wt-spill-c`.
This is CPU prototype work, not a model-support, GPU, performance or release receipt.

## Ownership and current implementation

Existing reserved adapter surfaces are **read-only this milestone**:
`crates/memra-engine/src/{model.rs,moe_cache.rs,spill.rs,hybrid.rs,hybrid_forward.rs,qwen4exp_gpu.rs}`.
Future edits must stay narrowly within expert/PLE residency. A's `spill_pread.rs` and
`pinned_host.rs`, B's KV/server paths and D's PP/transport paths are not edited.
New implementation/tests are entirely under `crates/memra-tier/src/bank/` and
`crates/memra-tier/tests/bank/`. Research harness avoids shared Cargo/lib/contracts edits.
The lead supplied these reservations; mesh reservation tools are not exposed in this worker.

| Surface independently inspected at baseline | What exists / what generalizes |
|---|---|
| `model.rs:1845-1898` | `ExpertLayout {offset,len,qtype,row_bytes}` and strict disk-source lookup. Generic descriptor must preserve all four, not infer dtype from the projection. |
| `model.rs:1900-1932` | `HostExps` has raw host bytes, optional per-expert tiers, `layouts`, per-expert macro scales and block-FP8 scale plane. Those scale surfaces cannot disappear in a bank abstraction. |
| `model.rs:3158-3223` | `layouts == None` is uniform proof. `expert_layout()` and `max_expert_bytes()` govern mixed scratch/dispatch. Per-expert tiers are already positioned: `expert_source` adds offset ONLY for a slab, avoiding double offsets. |
| `moe_cache.rs:50-65,250-287` | `(layer,projection,expert)` keys, fixed slots, probation/protected SLRU, optional size classes, pending DMA excluded from ordinary eviction; source owners retained/quarantined. Generalized identity adds immutable artifact/tensor/layout, not prefix hashes. |
| `moe_cache.rs:1081-1155` | Hits are bytes already resident; pending prefetch inserts compute wait before publication; misses stage exact source. Copy completion and source lifetime are separate. |
| `moe_cache.rs:1160-1237` | Known-next prefetch reserves a slot, fences overwrite, retains source and publishes only after an owner-side event wait. Unknown completion quarantines; it must never become a reusable slot. |
| `hybrid.rs:2049-2065,2836-2846` | Resident banks refuse mixed layouts. Gate/up/down all must be uniform. Active expert count retains original mask positions. |
| `hybrid_forward.rs:13675-13677,13796-13807,13932,14131-14146,14211-14234` | Device/pointer/pairs/slab/grouped-decode eligibility is uniform-only plus numerical/geometry restrictions. Generic metadata must not weaken any predicate. |
| `hybrid_forward.rs:14826-14849,14905-14965` | Cached and scratch staging use exact selected expert bytes, `expert_layout(ex)`, native qmatvec, gate/up and down macros. Preserve existing numerical program; no new CPU expert execution. |
| `hybrid_forward.rs:21146-21243` | Grouped prefill masks before selection; uniform resident twin is distinct from metadata-aware grouped staging, whose scratch uses maximum per-expert sizes. Q2_K stays existing staged f32 dequant. |
| `qwen4exp_gpu.rs:276-303` | PLE `NgramTable::F32(Vec<f32>)` / `Bf16(Vec<u8>)` ordinary host table; row gather converts BF16 via exact high-half bits. No generic bounded table service here. |
| `qwen4exp_gpu.rs:4228,4312,8629-8789` | Incremental and full-history n-gram IDs plus existing exact oracle: ragged chunks, EOS, divergent rewind, shorter reused state. Keep hashing/history/rollback in this adapter. |
| `qwen4exp_gpu.rs:13644-13760` | PLE gathers last t logical rows with bounds checks, then H2D and existing projections. Service should return same raw row bytes/order before existing F32/BF16 expansion. No FP8 PLE or changed projection program. |

## Design interpretation

- Reuse existing SLRU semantics in later adapter work. Day-1 hot-cache is a small
  injected-policy skeleton, **not** a replacement or performance claim for SLRU.
- `UniformLease` binds layout proof to actual owned bytes, unlike a detached boolean.
  `PerRecord` cannot convert even when a selected subset happens to look homogeneous.
- Immutable row keys and original expert/projection keys are different from KV identities.
  Row and expert policies use different marker domains; no shared heat scores.
- CPU `ExactReader` and `ConsumerFence` are stand-ins for A's owned read/transfer handles;
  the latter's fake readiness is **not** a CUDA fence. Existing engine adapters are untouched.
- One injected budget interface governs both bank and row allocations; only tests implement
  a fake governor. Real all-tier accounting, deadlines/fairness and mandatory headroom belong to B.
- Row coalescing measures **requested aligned bytes** versus unique useful bytes; actual SSD
  physical traffic still requires device counters/cache-state evidence. Sparse 264×48 and packed
  adjacency are different distributions, not conflicting results.
- Learned source warning: Qwen4Exp's earlier n-gram bottleneck was O(history) ID recomputation,
  not its microsecond host gather. Bounded NVMe capacity is the row-service purpose; no assumed
  latency win from asynchronous H2D alone.

## Baseline/environment limits

Read AGENTS.md completely, ROUTER, INDEX, benchmarks and the GPU lesson index/relevant
measurement/gate/PLE entries. Read the supplied 08 plan and D/E reference inventory.
External systems remain design references only. No runtime dependency added.
No SSH retry or remote action in this worker: lead's two failed launch attempts stand as
reported access evidence; no GPU exists on this Mac. No secrets/env files read.
DSv4 Flash 0731 remains unrelated and paused; no execution or adapter edit in that lane.

Initial `git worktree add` exceeded the 20-second tool timeout and left an incomplete
checkout/index lock. A process check showed no surviving checkout process. Removed only
this worktree's stale lock, reset the **new, unedited** tree to the assigned baseline and
unlocked it. `git status --short` was empty before implementation.
