# WP-A io_uring decision proposal — day 3

**Proposal only; no dependency, feature, ring, kernel module or default added.**
Current byte-worker baseline remains `io::BoundedReader`; explicit Linux direct
read/write is a comparator, not a scored winner. No Linux execution has run here.

## Lead-owned dependency fragment

```toml
# crates/memra-engine/Cargo.toml — DO NOT paste the placeholder as a real version.
[features]
tier-uring = ["dep:io-uring"]

[target.'cfg(target_os = "linux")'.dependencies]
io-uring = { version = "=<reviewed-exact-version>", optional = true }
```

Select and lock an audited exact release on the rig/toolchain before implementation;
no network/catalog evidence was available in this round and no version is invented.
The adapter belongs on the native side of the unsafe-free memra-tier boundary.
Avoid a second executor/runtime dependency. Feature is default OFF during its
bounded admission/measurement window; landing requires a dated decision record
and removal if no-go. This is not an environment read or a new FLAGS row today.

## Interface and authority

Use the same bounded-worker ownership protocol (`ReadRequest`, `ReadCompletion`,
`BoundedReader::submit/poll`, and `CpuTransfers::enable_background/progress`).
A native `RingReader` can implement the same submit/poll facade, using one ring
per I/O owner and a fixed SQ/CQ/in-flight bound that INCLUDES unread completions.
Do not change the frozen TransferEngine/ObjectStore traits to smuggle a ring
pointer, borrowed buffer or device lease across threads.

1. Owner validates full metadata, reserves selected extent + aligned scratch via
   the **same B governor**, and retains the pending reservation.
2. Submit only owned file descriptors, aligned preallocated CPU slots, checked
   offset/length and full `(issuer,sequence,state,src_gen,dst_gen)` identity.
   Slot registration is startup work, never per-request pin/register/free.
3. Zero submitted SQEs returns every owned input. Partial/uncertain submission
   is an accepted ticket with one outcome per original item, never zero-accept Err.
4. Handle short CQEs, EINTR, CQ overflow, timeout and kernel cancellation explicitly.
   Cancel-SQE completion is NOT proof the target read retired; wait for target CQE
   or quarantine backing + governor pin through unknown shutdown.
5. Verify header/padding/content address/valid checksum before host publication.
   H2D and GPU publication remain on the designated CUDA owner; CQE means host I/O
   completed, never device-ready or consumer-retired. Tombstones persist to ack.
6. Unsupported syscall/filesystem/registration is `Unsupported` or a **separate,
   explicitly selected buffered/worker row**, never direct/uring success.

The day-3 background mode is only byte-I/O asynchronous: bounded root lookup,
metadata decode, file open and governor admission still execute on the owner.
Moving those off a serving thread requires a prepared immutable catalog/descriptor
cache plus owner admission response channel, not making Rc/device authority Send.

## Rig decision cell (M1, booking date 2026-09-23)

Non-serving rented Linux RTX 5090 first, then designated PRO target; canonical
whole-rig lock, local NVMe ancestry and materialized (non-hole) extents. Capture
binary/source/artifact/fixture hashes, submitted versus physical bytes, actual
fallbacks, 250 ms GPU/host/SSD telemetry, queue depth and thermal/cache regime.
No network-volume faults, sparse-hole speed or Mac timing enters a scored row.

Compare measured positioned-worker baseline with io_uring on opaque 264-byte rows,
4096-straddling rows, 116654080/933232640-byte restores and read/write interference
from CELLS.md. Include mmap/pread/direct comparators where supported, preserving
identical valid bytes. Force correctness ON/OFF first; five AB AND five BA pairs,
N=10/arm, same window/binary. B/C supply native consumers for end-to-end measurement.

Hard fail: corruption/omitted item, early reuse, dropped charge, unknown fallback,
unbounded backlog. Keep only >=5% end-to-end pipeline gain with <=2% serving-tail
regression and both-order consistent direction (CELLS.md); require actual 70%
measured route/SSD headroom. Flat/negative/no-go deletes feature, dispatch and
exclusive code in the deciding lane; raw results/verdict remain in the corpus.

## Missing gates, not passes

No ring implementation or Linux run; no native pin/DMA, GPU consumer, physical-I/O
counter or full trace/250 ms collector. `rig-cells-a.sh` logs unsupported A2/M1
probes and exits nonzero; its five-second probes are NOT the 30-minute scored
windows. Cross-target `cargo check` cannot qualify syscalls, filesystem alignment
or this proposed dependency.
