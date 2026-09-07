# Bounded recent-row C4 GPU cache, 2026-09-06

Implemented, API-only and default OFF. The new sidecar targets the measured
host-C4 gather cost without changing KV values, index selection or history.
Hardware correctness and performance remain pending.

## Storage contract

Host C4 stays authoritative. A bounded GPU array holds recently written f32
compressed rows, with one absolute-row tag per cache slot. Gather first checks
the normal live-history bound, then requires an exact tag match before reading
the GPU sidecar. Otherwise it reads the same pinned host row as before. SWA,
transient rows, C128 and indexer storage are unchanged. Copies preserve all
bits, including signed zero; no precision conversion or approximate selection
is introduced.

Absolute tags are essential: a speculative future write can overwrite a ring
slot containing an older live row. After rollback, that older row must miss the
tag and read canonical host history. Re-emitting the same absolute row writes
both the host and cached value again. Bulk writes cache only their last bounded
suffix, so no two blocks race to fill one slot. Same-stream ordering ensures a
published tag and its row contents are complete before a later gather runs.

Canonical snapshots exclude this derived cache. Restore first copies the live
host history, then resets all tags and seeds only the latest live rows. Short
and empty restores invalidate old tags. Runtime and driver device binding are
explicit for recent writes/gathers, including a restore crossing GPU owners.

This is a recent-write cache, not a general LRU or a claim about measured hit
rate. The broader exact hierarchy is motivated by the current primary
[HiSparse paper](https://arxiv.org/abs/2608.07009); its performance numbers are
not inherited on this hardware. No third-party runtime/kernel was introduced.

## API and accounting

- `alloc_decode_state_host_c4_recent(capacity, transient_rows, recent_rows)`
  creates a fresh matrix-program state with explicit sidecar capacity.
- `restore_decode_state_host_c4_recent` accepts the same canonical snapshots,
  retaining the existing numerical-program tag checks.
- Existing allocation/restoration APIs pass zero recent rows, so their behavior
  stays unchanged. No new runtime environment flag or serving default is added.
- Requested recent rows are bounded at 8192 and clipped to the layer's host
  capacity. Each cached row costs 512 f32 values plus one i32 tag: 2052 bytes.
  A 512-row cache on all 21 C4 layers is 22063104 bytes (about 21.04 MiB), not
  a second full history. All additional GPU bytes enter `state.cache_bytes`;
  `c4_recent_gpu_bytes` reports them separately. Host-byte budgets are unchanged.
- `c4_recent_gather_calls` records cache-aware submissions, not hits or graph
  replays. Graphs remain tied to the state/sidecar addresses they captured.

## Gates and current status

412 engine CPU tests pass (12 GPU-dependent tests ignored), 604 server tests
pass, and strict clippy/format checks pass. These are not hardware qualification.
The compiled GPU unit covers cache sizes 1/7/32/65, misses/hits, ring wrap,
future-write rollback, same-index re-emission, a host-poison control proving
the cache-hit branch, short/empty restoration and fixed-shape graph replays
with changing history/tags. Test binary:
`90c09bdb20c2f52a32bb9d53cb4682b3f47005b621ab011f551c8ebd084c8944`.

Full-model gate `dsv4_c4_direct_gate ... recent-512` checks fresh position zero,
all live state/rings, suffixes, sampled plain/DSpark output, snapshot size,
cache accounting and resized restores at lengths 1/160/1025/4097. Binary:
`5dc63933d62bc88ab5fb22e23c501548e682b1cf3f7c40c5a8c8716aab25fe79`.

Only passing per-GPU component/memcheck/synccheck and model gates permit the
one-load `recent-ab` timing comparison. That fixes radix sampling and full
MoE grids, compares zero/512 recent rows across plain/DSpark, and uses six
interleaved rows per combination at 256/8192 context, with frozen sampled
outputs, no-loop filters, byte accounting and submission counters. Binary:
`90211dd805923d5a6756449b7635748788588d8f591cc36c681b1c8e025e5ca6`.

Target work waits behind the near-1M run and grouped-grid qualification. No
cache speedup, long HTTP support, concurrency/fairness or default promotion
is claimed from the compile/CPU results.
