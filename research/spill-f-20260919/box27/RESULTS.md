# M1 on BOX27: proven local NVMe, then memra's spill paths on it (2026-09-25)

Box shape (hardware only; identity in the private notes): one RTX PRO 6000 Blackwell
Workstation Edition, enforced cap 600 W of 600 W, a whole machine (no co-tenant GPU), a
16-core Ryzen 9 9950X host with 123 GB RAM, one 8 TB PCIe 4.0 x4 NVMe drive. The container's
cgroup: `memory.max` 127,295,029,248 bytes, `memory.swap.max` 0, `RLIMIT_MEMLOCK` 8 MiB and not
raisable, io_uring refused by the container's seccomp profile. Every cell ran under the
collector with the canonical `/tmp/memra-gpu.lock`; nothing else ran on the box.

Code: binaries built once at `ffff2d89a` (`build/commit.txt`, hashes in
`build/binaries.sha256`, frozen copies in `build/frozen-binaries.sha256`); the scripts moved
forward during the window, and every cell's checkout was verified to carry the build's engine
source unchanged (`git diff --quiet ffff2d89a HEAD -- crates Cargo.toml Cargo.lock`).
Artifact: `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` downloaded from its pinned revision straight onto
`/scratch/spill-f` and re-hashed there, SHA-256 `df27a780...7adf`, 18,209,036,576 bytes
(`stage/stage.log`). Receipts: this directory, mirrored with hash manifests
(`MANIFEST-part*.sha256`); files that carried the rented volume id are sanitized and listed in
`EXPORT-MANIFEST.json` with their original hashes (originals kept privately).

## A. The proof (run by the lead, mirrored 11 of 11 by hash)

```text
M1-PROOF verdict=PASS class=nvme-local-direct reasons=0
```

`/scratch` is an xfs bind (`/volumes/V.<volume-id>/_data`) of partition `nvme0n1p3` on
namespace `nvme0n1`, a PCIe 4.0 x4 controller (vendor 0x15b7, an 8 TB consumer drive class);
the kernel is not a guest (no `hypervisor` flag, 58 PCI functions, none emulated). The 1 GiB
O_DIRECT binding covered its payload on the leaf (2,125,160 written and 2,097,152 read
sectors). `m1-proof/`.

## OWED 7 GPU gate (pinned-pool test on the target card)

```text
test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 608 filtered out; finished in 1.04s
```

The frozen test binary ran from `/scratch/spill-f/bin`, so its files sat on the proven xfs: the
CPU window tests, the xfs `EINVAL` red control (enforcing branch, not skipped), and both CUDA
tests (`worker_positioned_reads_...` and `direct_worker_overread_preserves_exact_bytes`: pinned
bytes equal the file, zero errors and fallbacks, over-read equal to the prediction).
`owed7-gpu/`. `direct16` was admitted to B3 on this receipt.

## B0: device envelope (fio 3.36, 16 GiB lane file, direct I/O)

Random reads, 10 s per cell after a 2 s ramp; N=1 per grid cell (descriptive):

| Block bytes | Depth | psync threads MB/s | libaio MB/s | io_uring | psync p99 us | psync CPU s/GiB |
|---|---|---|---|---|---|---|
| 4,096 | 1 | 78 | 77 | refused | 59 | 0.957 |
| 4,096 | 2 | 160 | 148 | refused | 79 | 0.774 |
| 4,096 | 16 | 958 | 997 | refused | 132 | 0.789 |
| 450,560 | 1 | 1,588 | 1,586 | refused | 354 | 0.051 |
| 450,560 | 2 | 3,090 | 3,138 | refused | 453 | 0.041 |
| 450,560 | 16 | 6,723 | 6,726 | refused | 2,605 | 0.041 |
| 557,056 | 1 | 1,458 | 1,457 | refused | 408 | 0.053 |
| 557,056 | 2 | 5,384 | 5,383 | refused | 281 | 0.038 |
| 557,056 | 16 | 6,896 | 6,903 | refused | 3,391 | 0.039 |
| 860,160 | 1 | 1,761 | 1,760 | refused | 537 | 0.047 |
| 860,160 | 2 | 5,391 | 5,409 | refused | 449 | 0.037 |
| 860,160 | 16 | 6,854 | 6,853 | refused | 5,800 | 0.037 |
| 1,048,576 | 1 | 2,050 | 2,049 | refused | 578 | 0.044 |
| 1,048,576 | 2 | 6,144 | 6,163 | refused | 494 | 0.034 |
| 1,048,576 | 16 | 7,029 | 7,030 | refused | 7,700 | 0.035 |

Sequential writes, 4 MiB, 30 s: 6,445 MB/s direct, 5,100 MB/s buffered plus fsync (the
16 GiB prepare ran at 6,505 MB/s). Sustained read maximum 7.03 GB/s; the 70% headroom figure
is 4.92 GB/s. The scored io_uring screen (557,056-byte reads, 5 AB plus 5 BA) ran its
psync half (depth 2: median 5,381 MB/s, range 5,379 to 5,387; depth 16: median 6,899, range
6,055 to 6,902) and its io_uring half was refused on every visit:

```text
M1-B5-SCREEN depth2: refused-io_uring-unavailable
M1-B5-SCREEN depth16: refused-io_uring-unavailable
fio: pid=7711, err=1/file:engines/io_uring.c:1049, func=io_queue_init, error=Operation not permitted
```

The host kernel allows io_uring (`/proc/sys/kernel/io_uring_disabled` = 0); the container's
seccomp profile refuses `io_uring_setup`. On this route a memra io_uring backend could not run
at all, and at matched concurrency `psync` threads already equal `libaio` everywhere. B5's
input is therefore "refused here", not "not justified". Two earlier B0 attempts are kept: both
aborted at the first io_uring cell because the runner stopped instead of recording the
refusal (`b0-attempt1-uring-abort`, `b0-attempt2-uring-abort`; the second because fio wrote the
error into its `--output` file).

## B1: KV ObjectStore storage (`storage-bench`, 360 per-visit collector cells)

Six sizes x three modes x two phases x ten rounds, restore visits cold (`mincore` 0 before
each). Scored with the B1 amendment's read gate (`m1-b1-resummarize.py`, recorded inputs only):

```text
M1-B1-RESUMMARY visits=360 scored=360 failed=0
```

Median MB/s (N=10 each; verdicts are the registered rule against buffered):

| Size bytes | Phase, metric | buffered | uncached (O_DIRECT read) | direct (O_DIRECT read and write) |
|---|---|---|---|---|
| 264 | restore, read | 6.7 | 1.3 (loser) | 1.3 (loser) |
| 4,096 | restore, read | 104.5 | 19.1 (loser) | 18.9 (loser) |
| 4,097 | restore, read | 96.8 | 29.4 (loser) | 30.0 (loser) |
| 1,048,576 | restore, read | 533.9 | 453.9 (loser) | 455.4 (loser) |
| 116,654,080 | restore, read | 675.0 | 605.5 (loser) | 605.5 (loser) |
| 933,232,640 | restore, read | 500.9 | 406.3 (loser) | 406.3 (loser) |
| 116,654,080 | roundtrip, write | 208.8 | 183.1 (loser) | 184.1 (loser) |
| 933,232,640 | roundtrip, write | 221.0 | 179.6 (loser) | 182.1 (loser) |

(Full table with roundtrip reads and the small-size writes: `b1/summary-amended-gate.json`.)
Every visit was byte-exact with zero fallbacks. Two facts stand out: buffered beats both
O_DIRECT modes at every size and phase, and the store's best read (799 MB/s at 116 MB on a
roundtrip) is about a ninth of the drive's 7.03 GB/s. So the ObjectStore read path, whose
`io_ns` includes the store's own per-chunk integrity checks, is the ceiling on this rig, not the
device. The run's in-process scoring used the pre-amendment gate (the amendment's first commit
was stopped by a failed shell chain); its `summary.json` is kept and superseded by
`summary-amended-gate.json`.
