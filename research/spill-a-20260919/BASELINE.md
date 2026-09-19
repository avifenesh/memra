# WP-A day-1 baseline — 2026-09-19

Repository: avifenesh/memra. Inspected base: `c5a33b14ff7b6c75a3cd808dbc0ee4f10aa8b33f`.
Main was clean at worktree creation. Worktree `../wt-spill-a`, branch
`lane/spill-a-20260919`, is the only implementation tree used. This is a CPU
substrate milestone, not generic tier qualification, a merge, or model support.

## Independently read reuse targets (base file:line)

- `crates/memra-engine/src/spill_pread.rs:116-143`: exact `FileExt::read_at`
  loop retries EINTR, refuses EOF, checks offset addition. Buffered unaligned reads
  work; direct eligibility at `:30-32` requires 4096-byte offset/length alignment.
- `spill_pread.rs:300-390`: bounded request channel and fixed CPU thread pool;
  a worker owns the pinned buffer while reading and returns it in its completion.
  Workers do not issue CUDA calls. Worker shutdown joins before draining results.
- `spill_pread.rs:421-470`: startup pinned allocations are bounded by configured
  depth; partial allocation reduces effective depth, loudly. `:490-532` chooses
  file advice or O_DIRECT reopened by inode; direct failures are not direct success.
- `spill_pread.rs:624-772`: one expert extent must fit one slot; optional prefetch
  keeps one demand slot free. Admission returns `None` on saturation, permitting
  the existing mmap fallback. This is not admission for sole active KV backing.
- `spill_pread.rs:807-959`: cancellation retains worker-owned buffers until return;
  H2D events gate recycling. Missing event/failed stream drain retains/leaks the
  bounded backing rather than frees potentially DMA-visible memory.
- `crates/memra-engine/src/pinned_host.rs:20-60`: best-fit/coalescing allocator;
  plane reservations are atomic via cloned extent state. Alignment is four bytes,
  not a general NVMe/DMA alignment contract.
- `pinned_host.rs:85-141`: one portable CUDA-pinned arena allocated at startup;
  leases have fixed backing. `:178-194,244-296` enforce initialization before slice
  reads; full copies/fill initialize the logical range. Legacy per-image allocation
  remains a distinct path. CUDA copies here synchronize, not scheduler tickets.
- `crates/memra-gguf/src/source.rs:314-323`: `DiskExtent` retains `Arc<File>` and
  `Arc<Mmap>`, absolute file offset and exact len. Reuse this source owner, not a
  new loader or copied model bytes.
- `crates/memra-engine/src/moe_cache.rs:1100-1125,1197-1237`: pending source
  keepalive and compute-wait precede publication; unknown copies quarantine slots
  and sources. This file remains C-owned and unchanged.

## Generalization started

New A-owned modules under `crates/memra-tier/src/` implement:

1. 4096-byte, version-1 little-endian extent headers, SHA256 of the valid payload
   and header, zero padding, bounded chunk/root parsing, immutable chunk-first
   publication. `FileBackend` is **buffered ephemeral filesystem**, not O_DIRECT,
   measured NVMe, or durable restart. Immutable inserts use hard links and refuse
   overwrites; temporary names are removed. fsync/directory durability is pending.
2. `ReadAt` plus fixed-thread `BoundedReader`, preserving the existing exact-loop
   and owned-buffer discipline. Explicit in-flight bound includes unread
   completions. This is not yet wired into/replacing the expert `PreadPool`.
3. `FakePinnedPool` preallocates aligned CPU backing and accounts whole physical
   slots including alignment overhead. Demand headroom, initialization, explicit
   release, quarantine, and fake disk/DMA/consumer retirement are tested. **No
   memory is CUDA-pinned**; no GPU pointer API or live transfer is implemented.
4. `storage-bench cpu-fixture` emits schema-v1 JSONL for a deterministic byte
   fixture. Unknown physical disk bytes/GPU timing/RSS are null, not fabricated.

Large objects progress chunk-by-chunk through a pool smaller than the object in
CPU fixtures. Automatic multi-chunk transfer scheduling, global governor permits,
NUMA placement, real pinned arenas, producer/consumer CUDA fences, D PeerBackend,
io_uring/direct comparators and B/C model adapters remain subsequent milestones.
The store currently returns a metadata snapshot named ObjectLease and has no GC;
production governor-backed ownership must be frozen with B before integration.

## Reading / historical scope

Read AGENTS.md fully; docs/ROUTER.md; research/INDEX.md fully; research/benchmarks.md;
08 pre-work plan sections A-C/F, WP-A and Session A; inventories D/E (including
P1/M6/D2/D3/N8/T4); sibling private GPU corpus README and relevant LAW/TRAP/GATE
entries. Historical measurements were not rerun or promoted. No paused model path
was executed or adapted; no model/kernel/numerical program changed.

## Environment and scope boundaries

Mac CPU only. Two bounded SSH attempts using
`ssh -o ConnectTimeout=10 -o BatchMode=yes home hostname` each exited 255 with
`Connection closed by UNKNOWN port 65535`. No remote inventory or GPU work ran.
No further SSH attempts in this round. Parent owns rig rebooking and artifact pins.

Only basic worker tools were exposed (read/write/edit/bash/parent communication).
Mesh reservation and receipt tools were unavailable: the lead's explicit reserved
path list is the ownership record. No live config, credentials, environment files,
serving instances, drivers or kernel modules were inspected or changed.
The scaffold lives under this receipt directory instead of modifying lead-owned
root Cargo files or `crates/memra-tier/{Cargo.toml,src/lib.rs,src/contracts.rs}`.
It imports the actual A modules and the actual engine-local CLI by path.
