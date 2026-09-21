# Pinned host allocation flags on the contract path: census (WP-A day 13)

Status: a census written before any code, for the `MEMRA_KV_HOST_CONTRACTS` decide-by review
(2026-10-05). Tree `c2f13f6cc` (`lane/spill-a-20260919` with `origin/main` `1b354be59` merged);
`core.rs`, `result.rs` and `sys/mod.rs` are cudarc 0.19.8 (`crates/memra-engine/Cargo.toml:22`,
`default-features = false`, `driver` feature). Lane C's finding that motivates it:
`research/spill-c-20260919/WC-DESTINATIONS.md` and `DAY16.md` "The WC pair" (read only; C's day 16 is
not on `main`).

## Headline

The contract path has exactly one pinned host allocation site, `CudaTransfers::alloc_host`
(`crates/memra-engine/src/tier_transfer.rs:210-245`, the driver call at `:225`), and it takes
cudarc's `CudaContext::alloc_pinned`, which hard-codes `cuMemHostAlloc(bytes,
CU_MEMHOSTALLOC_WRITECOMBINED)` (`core.rs:1412-1424`, flag value 4; `PORTABLE` = 1 and `DEVICEMAP` =
2 are not set, `sys/mod.rs:256-258`). Every other pinned host allocation in the engine chooses its
flags explicitly through the raw `result::malloc_host(bytes, flags)` (the same `cuMemHostAlloc`
FFI, `result.rs:863-867`): the OFF tier's planes and the router readback staging are cached
(flags 0), the startup arena and the PP bounce are `PORTABLE`. The contract path is therefore the
only pinned surface in the engine whose host reads run write-combined without an explicit choice,
and under the door every demoted plane is read by the CPU twice at demote (the engine's
completion hash, the bind hash) and, under C's Option C, once more at promote (the H2D source
hash). The verify digest is not one of those reads (section 2).

## 1. Every pinned host allocation site in the engine

| Site | Call | Flags | Who uses the bytes |
|---|---|---|---|
| `tier_transfer.rs:225` in `CudaTransfers::alloc_host` (`:210-245`) | `self.stream.context().alloc_pinned::<u8>(bytes)` then a zero fill through `as_mut_ptr().write_bytes(0, bytes)` (`:227`) | `CU_MEMHOSTALLOC_WRITECOMBINED` (4), chosen by cudarc, not by the engine | every `CudaPinnedLease`: the contract's D2H destinations (`worker.rs:9242`, one lease per K and V plane and per draft plane, C's Option B), the gates' leases (`kv_tier_gate/active.rs:255`, `fault.rs:380,434`, `tier_transfer_gate.rs:62,148,247,392,480,633`), and under C's Option C (not on `main`) the H2D sources, which are twins of the same leases (`retain_host`) |
| `pinned_host.rs:190` in `PinnedHostBuf::new` (`:186-201`) | `result::malloc_host(len.max(1), 0)` then `write_bytes(0, len)` | 0: cached, page-locked | every OFF-arm host plane of the pageable tier, every handoff import, `HostF32::down` |
| `pinned_host.rs:102-105` in `PinnedHostArena::reserve` (`:95-117`) | `result::malloc_host(bytes, CU_MEMHOSTALLOC_PORTABLE)`, uninitialized by design | `PORTABLE` (1): cached, recognized by every context | the fixed arena of `MEMRA_GLM5_TP_KV_HOST` (refused with the door at boot) |
| `lib.rs:2074` in `PinnedStage::new` (`:2064-2083`) | `result::malloc_host(cap, 0)` | 0: cached; the comment at `:2064-2067` says why: cudarc's `alloc_pinned` is "pathologically slow for host READS" and the router readback is read-heavy | DtoH staging of the router readback |
| `lib.rs:7615-7625` `moe_router_topk_host` | the same `PinnedStage` | 0: cached ("NOT cudarc's WRITECOMBINED default, so the host-side reads of sel/w stay cached") | router top-k readback |
| `pp.rs:1450-1456` (`:1432` doc) | `result::malloc_host(.., CU_MEMHOSTALLOC_PORTABLE)` | `PORTABLE` (1) | the bidirectional PP bounce (must be visible to both contexts) |
| `model.rs:1989`, `:2376`, `:2666`, `:2781`, `:2866`; `spill.rs:175`; `spill_pread.rs:430` | `e.ctx().alloc_pinned::<u8>(..)` | `WRITECOMBINED` (4) | H2D-only expert and weight staging; `model.rs:1679-1681` records the caveat: "write-combined memory is SLOW for CPU reads. A future CPU-VNNI cold-expert fallback must NOT read from this buffer" |
| `dsv4_c4.rs:144` | comment only | "cacheable `cuMemHostAlloc(flags=0)`" is the UVA assumption that code relies on | (no allocation at that line) |

So the engine already knows the rule (cached for CPU reads, write-combined for H2D-only streams)
and applies it everywhere it allocates by hand. The contract path inherited cudarc's default
because `alloc_pinned` has no flag parameter.

## 2. Every host read of a contract plane over its lifetime (door ON)

| When | Read | Site | Write-combined? |
|---|---|---|---|
| Demote, first observation of the D2H completion | the engine's completion checksum, `checksum(item.host.bytes())` (SHA-256 over the whole destination) | `tier_transfer.rs:722-724` in `progress` (`:688-750`); `progress` runs from `synchronize` (`:640-647`), `poll` (`:983`), `publishable` (`:740-750`, so from `take_destination`, `ready_view`, `with_destination`), `retire_source` (`:388`), `retire` (`:1071`) and `recover_source` (`:452`) | yes: a CPU read of the WC destination, every item, once (the `producer_done` guard) |
| Demote, at bind | `bind_tier_image`'s bundle checksum, `checksum(bytes)` per `Role::Key`, `Value` and `Draft` segment, compared with the completion receipt | `crates/memra-server/src/worker.rs:8739-8800` (`:8787`) | yes: the second CPU read of the same bytes |
| Promote, C's Option C (`wt-spill-c`, not on `main`) | the engine's completion checksum of the H2D SOURCE, the same `progress` code; the source twin is the WC lease | `tier_transfer.rs:722-724` | yes: a third read, at promote |
| Promote, the OFF program under ON (`main` today) | `htod_u8_into(&bytes[..kb])`: `cuMemcpyHtoDAsync` reads the pinned source by DMA | `worker.rs plane_up` | no: a DMA read; write-combined is the documented right attribute for this direction |
| `MEMRA_KV_HOST_VERIFY` digest, demote and promote | `host_roundtrip_digest` (`worker.rs:10063`, called at `:10201` and `:10708`) is `prefix_entry_state_digest` (`worker.rs:12590`), which reads the DEVICE entry through `engine.dtoh_u8_view(&plane.k.slice(..))` into pageable memory and hashes that | `worker.rs:12590-12640` | no: the digest never touches the pinned image. Correction to the day-13 brief, which listed the verify digest among the WC reads: its cost is a D2H readback plus a hash in both arms, flag-independent |
| `MEMRA_KV_HOST_FAULT=flip-demote` | `flip_first_byte`: `lease.bytes()?.to_vec()` (a full WC read), flip, `write` | `worker.rs:7789-7805` | yes, diagnostic only (gate box) |
| Gate readers | `checksum(host.bytes()?)` after the D2H (`kv_tier_gate/active.rs:257`); `bundle.verify(&[host.bytes()?.to_vec()])` before the H2D (`:306`, a WC memcpy into a `Vec` then the hash); `host.bytes() == bytes` and `checksum(host.bytes())` (`tier_transfer_gate.rs:659-660`) | gate binaries | yes |
| Host WRITES into the lease (the WC-friendly direction) | the zero fill at allocation (`tier_transfer.rs:227`, 160 MB of CPU stores per plane class allocation), `CudaPinnedLease::write` (`:74-88`; `tier_transfer_gate.rs:63`, `fault.rs`) | | stores combine and post: the attribute helps here |

Under OFF none of the contract reads exist (no receipt, no bind hash: `bind_tier_image` returns at
`:8741` without a tier), and the OFF tier's `PinnedHostBuf` is cached anyway. The door's byte
attestation costs CPU reads by design; the flag decides whether they run at cached-DRAM or at
write-combined (uncached, unaggregated) read speed. C measured the whole delta on the target card
(demote median 37.8 ms OFF against 169.2 ms ON, steady state 6 to 8 against 136 to 140 ms, N=5
per arm per order, one lock hold; `WC-DESTINATIONS.md` "Results"); the split between the two WC
hashes and the ticket lifecycle is what day 13's cell measures directly.

## 3. What cudarc 0.19.8 offers

- `CudaContext::alloc_pinned<T>(len) -> PinnedHostSlice<T>` (`core.rs:1406-1424`): write-combined
  only, no flag parameter. `PinnedHostSlice { ptr, len, event }` fields are `pub(crate)`
  (`:1388-1392`), so a slice with other flags cannot be constructed from outside the crate. Its
  `Drop` synchronizes the tracking event and calls `free_host` (`:1397-1403`); `as_ptr`,
  `as_mut_ptr`, `as_slice`, `as_mut_slice` synchronize the event first (`:1447-1479`).
- `result::malloc_host(num_bytes, flags: c_uint) -> *mut c_void` (`cuMemHostAlloc`,
  `result.rs:863-867`) and `result::free_host(ptr)` (`cuMemFreeHost`, `:873-875`): the raw FFI the
  engine already uses at `pinned_host.rs:102,190`, `lib.rs:2074`, `pp.rs:1450`.
- `sys::CU_MEMHOSTALLOC_PORTABLE = 1`, `CU_MEMHOSTALLOC_DEVICEMAP = 2`,
  `CU_MEMHOSTALLOC_WRITECOMBINED = 4` (`sys/mod.rs:256-258`).
- `sys::cuMemHostGetFlags(&mut flags, ptr)` (`sys/mod.rs:12855`): the driver's own record of the
  flags of a pinned allocation, the receipt that an arm was honoured.
- `HostSlice<T>` is a public, unsealed trait (`core.rs:1304-1326`: `len`, `stream_synced_slice`,
  `stream_synced_mut_slice`) and `SyncOnDrop::{Record, Sync}` are public variants
  (`:1112-1117`), `CudaEvent::context()` / `synchronize()` (`:575`, `:596`),
  `CudaContext::new_event` (`:551`), `record_err` (`:514`), `CudaStream::wait` (`:766`) are public:
  an engine-owned pinned backing can present exactly the `HostSlice` contract `PinnedHostSlice`
  presents (wait the stream on the tracking event before the copy, record the event on the stream
  after it; `:1481-1507`).
- No `cuMemHostRegister` wrapper (the route a hugepage-backed or externally owned buffer would
  take); out of scope here.

## 4. The copy program, which must not change

D2H: `self.stream.memcpy_dtoh(&device.slice(..bytes), backing)` (`tier_transfer.rs:948-951`), which
is `cuMemcpyDtoHAsync(dst, src, owner_stream)` after `stream_synced_mut_slice` has the stream wait
on the backing's tracking event (`core.rs:1644-1654`), then `record_event`, `stream.wait(event)`
(`:953-955`). H2D: `self.stream.memcpy_htod(backing, &mut device.slice_mut(..bytes))`
(`:937-944`; `core.rs:1602-1612`). The allocation flag is an attribute of the host pages, not of
the copy call: the same driver calls, in the same order, on the same owner stream run in both arms.
Byte exactness is checked on every roundtrip (the engine's completion checksum against
`SegmentExpectation`, `Completion::require`; the gate's compare).

## 5. The seam (task 2, code lands after this file)

- `PinnedKind { WriteCombined, Cached }` in `tier_transfer.rs`, `host_alloc_flags()` returning
  `sys::CU_MEMHOSTALLOC_WRITECOMBINED` (4) and `0`. `Default` is `WriteCombined`: today's flag bits.
- `CudaTransfers::alloc_host` is unchanged in signature and behaviour and delegates to
  `alloc_host_kind(bytes, request, PinnedKind::default())`; the gate calls `alloc_host_kind` with
  the arm it selects. No `MEMRA_*` read: the gate passes the arm; a server door, if one is ever
  wanted, is lane C's call.
- The backing becomes engine-owned (`PinnedBacking`: pointer, length, kind, the same
  `CU_EVENT_BLOCKING_SYNC` tracking event) allocated through the existing
  `result::malloc_host(bytes, kind.host_alloc_flags())` FFI and freed through `result::free_host`
  after an event synchronize, with the same `HostSlice` implementation cudarc gives
  `PinnedHostSlice`. The two arms then differ in the flag bits and in nothing else.
  `CudaPinnedLease::pinned_kind()` reports the arm; `validate` keeps its context check through
  `backing.context()`.
- Tests: a CPU cell that the default's flag bits equal cudarc's constant and that `Cached` is 0;
  a GPU cell (ignored without a device) that allocates under each arm and reads back
  `cuMemHostGetFlags`, then D2H-H2D roundtrips bytes exactly under each.

## 6. The measurement (task 3, pre-registered in `DAY13.md` before the runs)

`tier-transfer-gate pinned-ab`: in one process, one CUDA context, one collector lock hold on the
target card, the same D2H and H2D roundtrip per arm over a 160 MiB buffer (the entry class C
measured): allocation wall, D2H wall (submit to owner-stream synchronize), the engine's
completion-hash wall (`synchronize(&ticket)` on an already complete copy), the bind-hash wall
(`checksum(host.bytes())` over the taken destination, the read the bind does), H2D wall, the
H2D-source hash wall (Option C's read), and `byte_exact` on both legs. Interleaved A B A B and
B A B A, N=5 pairs per order, one untimed warm-up roundtrip per arm, the collector's 250 ms
telemetry, medians with N and regime. The rule and the verdict are in `DAY13.md`.
