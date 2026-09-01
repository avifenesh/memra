# Boot-time PP peer byte-integrity probe

Date: 2026-08-11
Lane: `lane/cx-peerprobe`

## Contract

CUDA peer capability and a successful asynchronous copy are necessary but not sufficient for
memra PP-N. Before model weights upload, each cross-device PP boundary must prove that the exact
peer-copy API used by serving preserves every byte.

Bring-up performs two deterministic probe classes in both directions across every adjacent
cross-device stage pair:

1. a fixed 16 KiB payload, immediately after `cuCtxEnablePeerAccess` and before default-pool
   grants, using legacy `cuMemAlloc` as a fast simpleP2P-class preflight;
2. production `BoundarySlot` payloads at 1, 8, and 16 `[n_embd] f32` rows, using the authoritative
   model width at loader entry and running after default-pool grants.

The preflight allocates legacy device buffers in the source and destination contexts, fills the
source with a deterministic xorshift byte pattern and the destination with its bitwise inverse,
issues `cuMemcpyPeerAsync` on the source stage stream, synchronizes that stream, reads the
destination back, and compares every byte. Legacy `cuMemAlloc` deliberately makes this first gate
independent of the stream-ordered default-pool grants that follow it.

The production gate then constructs real `BoundarySlot` objects backed by stream-ordered
`CudaSlice<f32>` allocations and invokes the same shared TX/RX functions used by decode. For each
direction it exercises `n = B * n_embd` at `B={1,8,16}`, including the RX-side local copy and event
choreography. Each destination slot is first filled with the bitwise inverse pattern so a missing
or partial write cannot agree with source zero bytes, then the host readback is compared
exhaustively byte-for-byte. The direction and total summaries report the largest clean payload in
bytes. These temporary slots are dropped and their source/destination streams synchronized before
bring-up continues.

Any native-path API error or mismatch aborts PP bring-up. A mismatch error includes the exact
number of differing bytes, boundary index, direction, probe class or token width, and payload
size. If
`MEMRA_PP_HOST_BOUNCE=1` was selected before runtime construction, the same diagnostic logs the
corruption count and proceeds on the existing pinned D2H/event/H2D path; a peer capability or API
failure also cannot disable that fallback. Bounce diagnostics temporarily grant only the access
needed for each probe class. They set temporary pool access back to `PROT_NONE` and call
`cuCtxDisablePeerAccess` for every enabled direction before host-staged serving; teardown failure
refuses startup. NVIDIA's current Driver API defines `cuMemPoolSetAccess` as the visibility control
for pools and `CU_MEM_ACCESS_FLAGS_PROT_NONE` as making the mapped range inaccessible:
[stream-ordered memory-pool API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html)
and [memory access flags](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html).
`MEMRA_PEER_PROBE=0` skips the byte gate only as a loud, diagnostics-class escape hatch. The
runtime never changes transport after model loading begins.

The pure readback classifier is separately unit-tested by corrupting three offsets in a 16 KiB
oracle. It must return an exact `3 mismatched byte(s)` failure for native P2P and a
`ProceedWithHostBounce { mismatches: 3 }` decision for the host-bounce policy. This test allocates
no CUDA context and needs no peer hardware.

## NVIDIA R570 basis

NVIDIA's R570 Data Center driver notes document why a capability bit cannot be the correctness
gate. For GPUDirect P2P over PCIe on Ada and older architectures, some hosts do not preserve the
required ordering of GPU-initiated posted transactions under Relaxed Ordering, which can cause
run-time silent data corruption. NVIDIA's included `simpleP2P` example reports peer access as
available and prints copy bandwidth before its byte validation fails. The notes also state that
drivers 525 and newer choose an RO-disabling mitigation from PCIe host-bridge IDs, but a guest OS
may not expose the exact topology, so the mitigation may not be applied when needed. Source:
[NVIDIA Data Center GPU Driver R570, version 570.133.20, known issue “Disable GPU initiated RO
traffic on Ada Lovelace and older GPUs”](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-570-133-20/index.html#known-issues).

That vendor note is architecture- and topology-scoped; it does not establish the root cause of
memra's separate Blackwell Vast failure. The committed Vast receipt independently showed
`cuDeviceCanAccessPeer=1`, successful CUDA returns, and apparent throughput while 16,320 of 16,384
bytes were wrong (`research/p2pvast-20260810/RESULTS.md`). The shared engineering conclusion is
narrower: only an end-to-end byte comparison proves a usable PP peer path.

## Validation boundary

This DEV/BUILD lane proves the injected fail path, local release build, and local RTX 5090
`kernel-check`. A native peer clean pass cannot be produced on the single-card development rig.
The pending Vast window must boot the real PP-2 serving shape, observe a clean legacy 16 KiB line
and clean production-slot 1/8/16-token lines in both directions (including the largest-clean
payload metric), and confirm the established golden output hash before merge or release.
