# WP-A positioned-read baseline — day 8

## Decision and scope

**io_uring remains DEFERRED.** No dependency, runtime adapter, environment flag,
new numerical program, or default is introduced. This baseline informs the
existing [proposal](IO-URING-PROPOSAL.md); it cannot admit an accelerator.

Storage class: **block-device ext4 (virtio; NVMe ancestry provider-claimed, not proven)**.
`day8/native/storage/filesystem.log` records `stat -f`, `findmnt` and
`/sys/block/vda/device`. This supersedes overlay-only filesystem evidence for
these particular cells, not for earlier measurements. No cross-machine timing
comparison is valid.

## Protocol

`pread_baseline.py` uses the existing Python standard library and libc `pread`;
there is no external dependency. One fully materialized 64 MiB fixture, fsynced
before use, is read sequentially at queue depth one. Requests use 1 MiB or 16 MiB
chunks in buffered or explicit `O_DIRECT` mode. Direct-read buffers are
4096-byte aligned, initialized before timing, and reused. No direct-I/O fallback
is accepted. Each arm reads 64 MiB once: **N=1 development plumbing, no medians,
not spill speed**. It is not a timed comparison of engines or a served workload.

Each cold preparation uses file-scoped `POSIX_FADV_DONTNEED`, never global cache
dropping. Each warm preparation reads the complete fixture through a buffered
file descriptor. `mincore` must then show 0/all pages resident respectively or
the cell fails. These are **host page-cache states**, not storage-controller cache
states. O_DIRECT bypasses page cache in both preparations; warm does not imply an
O_DIRECT cache hit. `/proc/self/io` read-byte deltas are recorded separately.

`read_ns` sums timed libc calls; `wall_with_hash_ns` includes the per-chunk copy
and SHA-256 verification. SHA-256 is compared with the independently generated
fixture digest. The rates are syscall-plumbing rates, excluding preparation,
open, allocation, publication, H2D and serving latency. The collector samples
250 ms GPU telemetry for the whole batch under `/tmp/memra-gpu.lock`; short
subcells are not independently thermally qualified.

## Measured N=1 development rows

Batch `day8/native/storage-batch/attempt-04` completed with collector verdict
`executed-not-qualified`, `qualification: false`. Four prior attempts refused the
canonical lock; no work ran in them. Source `7eba2ebb8`; native storage binary
built at `e4f7e633b27fc2caf85d553d6e0659449af1b3b2` and post-run SHA-256 unchanged.
Power cap/max **600/600 W**. Raw rows: `pread-baseline.log`, with per-subcell hashes
in `subcells.jsonl`; offline replay in `day8/receipt-replay.json`.

Each row is one 64 MiB pass, N=1; rates below are **not medians**. Cache and thermal
order is fixed (1 MiB then 16 MiB; buffered then direct; cold then warm), not
interleaved/balanced or steady-state. Do not infer an arm winner from these rows.

| Chunk | Read mode | Host cache before | Read-call sum, ms | MiB/s, N=1 | Process physical read bytes |
| --- | --- | --- | ---: | ---: | ---: |
| 1 MiB | buffered | cold | 65.420 | 978.294 | 67108864 |
| 1 MiB | buffered | warm | 4.680 | 13676.184 | 0 |
| 1 MiB | O_DIRECT | cold | 52.642 | 1215.751 | 67108864 |
| 1 MiB | O_DIRECT | warm | 49.870 | 1283.325 | 67108864 |
| 16 MiB | buffered | cold | 30.994 | 2064.930 | 67108864 |
| 16 MiB | buffered | warm | 5.000 | 12798.828 | 0 |
| 16 MiB | O_DIRECT | cold | 19.187 | 3335.625 | 67108864 |
| 16 MiB | O_DIRECT | warm | 15.277 | 4189.292 | 67108864 |

All eight rows were byte-exact. Cold/warm preparation was confirmed as
0/16384 versus 16384/16384 resident pages. Buffered warm rows charged zero process
physical read bytes; cold buffered and both direct preparations charged
67108864 bytes. That is process I/O accounting, not proof of physical NVMe media
reads or a cold device cache. The 16 MiB direct rows consist of only four syscalls;
these remain plumbing observations with high N=1 uncertainty.

## What io_uring must beat

A later candidate must first reproduce exact bytes and the bounded ownership,
partial-acceptance, cancellation, unknown-completion and governor schedules.
Submission or CQE completion cannot substitute for CUDA producer/consumer fences.

The numeric admission criterion remains the proposal's: **at least 5% end-to-end
storage-to-compute pipeline gain**, at most **2% serving-tail regression**,
consistent direction in five AB and five BA pairs on the same hardware/window,
and actual **70% measured route/SSD headroom**. The comparison must include the
bounded positioned-reader worker, not just this isolated synchronous syscall
probe. Preserve the same bytes, chunk sizes, cache states, binary and device
conditions. Correctness ON/OFF and raw failures are mandatory before timing.

### Numerical screening targets (decision inputs, not scored thresholds)

Using the observed **O_DIRECT cold** N=1 rows only to size the experiment, a
5% throughput improvement on the same 64 MiB read pass would mean:

| Chunk | Observed pread MiB/s (N=1) | Illustrative io_uring minimum MiB/s (+5%) | Equivalent maximum read-call sum, ms |
| --- | ---: | ---: | ---: |
| 1 MiB | 1215.751 | 1276.539 | 50.135 |
| 16 MiB | 3335.625 | 3502.406 | 18.273 |

These targets are arithmetic projections (rate × 1.05; time ÷ 1.05), **not an
io_uring measurement or a go verdict**. Re-measure both pread and io_uring on the
same box/window at each chunk size with cold host-cache preparation and exact
bytes, **five AB plus five BA pairs (N≥5 per order)**, raw failures/hashes and
250 ms telemetry. The real threshold is ≥1.05× the fresh pread control, not the
historical absolute number above. Require consistent direction in both orders;
a failure at either chunk size is no-go for a generic replacement. Passing the
syscall screen alone is insufficient: the ≥5% composed storage-to-compute gain,
≤2% serving-tail regression, correctness, and measured headroom gates above still
apply. A flat/negative/unsafe candidate loses its exclusive door in that lane.

N=1 syscall numbers alone do not establish headroom or justify a dependency.
Still required: native worker + CUDA consumer baseline, verified storage-route
ancestry for an NVMe claim, same-window candidate measurements, queue/CPU/storage
telemetry, and serving-tail evidence. If the measured candidate is flat, negative
or unsafe, delete its admission door and exclusive arm in the deciding lane;
retain its raw evidence and verdict.
