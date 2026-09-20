# Native KV physical reclaim — WP-B day 9

## Diagnosis

The initial subrange-ownership explanation is **withdrawn**. Source inspection at
`3febcd43c6201e863aea148b447a340d71e50dda` shows:

- `memra-kv::Cache::new_inner` allocates an independent `CudaSlice<u8>` for each
  layer's K and V plane, with an 8-byte kernel tail pad.
- The active gate moves each **whole** allocation out of its detached `KvLayer`
  into `CudaTransfers`; only the initialized prefix is copied to host.
- `retire_source` drops the ticket's source lease; `release_device` removes the
  owner's resource and quota. `CudaSlice::drop` calls CUDA free asynchronously
  when the context supports stream-ordered allocation. This does not guarantee
  driver-visible physical reclamation.
- Day 8 already called the engine's best-effort pool trim, but did not record
  trim API errors or source destruction. Its zero driver delta does not name a cause.

The day-9 native diagnosis records a **weak** reference to each backend backing
Rc across `release_device`, without extending its lifetime. Before=1/after=0
proves the sole `RefCell<CudaSlice>` was destroyed (and cudarc's free path reached);
it does not claim the driver returned memory. Registry occupancy is recorded
separately. The detached layer contains a one-byte placeholder; neither the gate
nor materializer retains the source plane. Full free-VRAM and async-pool
reserved/used counters are captured before demote, before trim and after checked
`cuDeviceGetDefaultMemPool` + `cuMemPoolTrimTo(pool, 0)` calls.

**Native outcome: `RECLAIM-DIAG: freed but not observable`.** Source
`ef0ebe8c6`, collector `rented-5090-20260919/day9-diag-8192/`, exit 0.
All 32 source backing Rcs go 1→0; registry occupancy goes to zero. Pool used
falls from 15,191,180,692 to 14,947,910,836 B: 243,269,856 B, exactly the
243,269,888 B source planes less the 32 one-byte placeholders. Pool reserved
stays 15,200,157,696 B. The checked trim succeeds, but driver-free bytes remain
17,934,516,224 B before demote, before trim, after trim and after restore.
Seven frozen baseline byte surfaces (including all continuation logits/token ids)
match. This is the door-OFF control:
`ACTIVE-8K copy/restore bit-identical, no reclaim — not G1 PASS`.
The source-ownership explanation is disproved; the physical pool reservation
remains despite source destruction and successful trim. The reason the driver
cannot trim this reservation is not inferred (fragmentation is a hypothesis).
This result selects the VMM candidate, not another trim-only fix.
A failed trim would be an error, not a zero-byte success.
One collector cell, same frozen 8k continuation comparison, selects the next step:

1. `RECLAIM-DIAG: async-pool retention`: deterministic trim after demote (or a
   supported non-pooled allocation) is the candidate fix.
2. `RECLAIM-DIAG: source still owned by <owner>`: fix the identified ownership seam.
3. `RECLAIM-DIAG: freed but not observable`: implement VMM below.

No allocator door has been introduced by the diagnosis.

## Candidate physical-reclaim designs

### Fixed-VA CUDA VMM (selected by diagnosis; implemented at f3a3247a)

Reserve each plane's virtual range once using `cuMemAddressReserve`. Query device
VMM support and allocation granularity; refuse explicitly if either cannot be
established. Create/map chunks with `cuMemCreate`/`cuMemMap`/`cuMemSetAccess`.
Demote only after successful D2H, checksum and producer retirement; unmap and
release chunks **wholly contained** in the demoted prefix. Preserve edge chunks.
Restore by creating/mapping physical storage into the **same VA**, then H2D and
checksum verification before publication. No token or graph may execute while
any required range is suspended. Unknown completion quarantines ownership.

The numerical kernels, strides, encodings and request program do not change.
The transfer governor accounts rounded physical bytes, not merely logical bytes.
The gate replaces freshly allocated empty cache planes before any token executes;
unused pooled bootstrap reservation can remain. Direct VMM allocation inside
Cache construction and serving admission are not yet integrated. No VMM
allocation is freed with `cudaFree[Async]`. A private `KvPlane` owns the mapping
and its cudarc operand; Drop consumes the operand with `CudaSlice::leak` before
unmap/release/address-free. The lead authorizes the single documented unsafe
`upgrade_device_ptr` constructor in that owner, in addition to CUDA FFI calls.
The VMM slice must never escape as an independently owned `CudaSlice`.

The implemented gate-only typed door is `--kv-allocator vmm`, default
`pooled`, **decide-by: 2026-10-04**. No `MEMRA_*` read or shared FLAGS edit.
Absence of proof leaves it unqualified; a negative/no-go receipt deletes the door.

### Paged KV block pool (alternative, not implemented)

A pool may reclaim slots for other requests without returning VRAM to the driver.
Its acceptance metric is pool-available bytes, not `cuMemGetInfo`. It requires
kernel-side page indirection and explicit validation of the resulting execution
program; unlike fixed-VA VMM, existing contiguous-address kernels cannot consume
it unchanged. This alternative does not satisfy today's driver-free G1 criterion.

## Chunk arithmetic (conditional on a queried 2 MiB granularity)

Qwen3.8-27B's frozen native format is K q8_0 / V q5_1. The day-8 bundle census
contains 16 full-history layers, K=1088 and V=768 bytes per token. No format change.
For prefix interval `[start,end)` and chunk size G, releasable chunks occupy
`[ceil(start/G)*G, floor(end/G)*G)`, or none if the interval is empty. Both edges
remain resident. Each plane's full capacity also includes its existing 8-byte pad.

| Prefix | K bytes / 2 MiB | V bytes / 2 MiB | Whole chunks K+V / layer | Releasable, 16 layers |
| --- | --- | --- | --- | --- |
| 8192 tokens | 4.25 | 3 | 4+3 | 234,881,024 B |
| 8064 tokens (8k gate suspension) | 4.18359375 | 2.953125 | 4+2 | 201,326,592 B |
| 32768 tokens | 17 | 12 | 17+12 | 973,078,528 B |
| 32640 tokens (32k gate suspension) | 16.93359375 | 11.953125 | 16+11 | 905,969,664 B |

The gate suspends **128 tokens before** the requested final context; this is why
nominal 8k/32k prefix arithmetic is not its expected physical-release count.
A different queried G must produce a newly recorded calculation, never silently
pretend that 2 MiB is the device contract.

## Native 8k VMM evidence

At `f3a3247a027ca2f36bff2cf194669712f564c76b`, the native query returns
G=2,097,152 B. The 8064-token suspension releases exactly **201,326,592 B**:
free VRAM **17,613,651,968 → 17,814,978,560 → 17,613,651,968 B**.
All 32 planes restore at their original VA; all frozen continuation bytes match.
Verdict: **ACTIVE-8K G1 PASS**. Raw receipt: `rented-5090-20260919/day9-vmm-8192/`.
Per-plane physical capacity, valid bytes, granularity and VA are in `vmm-planes.tsv`.
Rounded capacity is 301,989,888 B, including 100,663,296 B of retained edge/unused
capacity chunks. No total-footprint or performance improvement is inferred.
