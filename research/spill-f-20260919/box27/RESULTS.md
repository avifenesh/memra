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

## Verdicts in one place

| Cell | Registered result | Notes |
|---|---|---|
| A proof | `M1-PROOF verdict=PASS class=nvme-local-direct reasons=0` | xfs bind, partition of a PCIe 4.0 x4 NVMe, bare-metal kernel |
| OWED 7 GPU gate | 12/12 on the target card | `direct16` admitted |
| B0 | envelope recorded; B5 screen `refused-io_uring-unavailable` at depth 2 and 16 | container seccomp refuses io_uring; psync threads equal libaio |
| B1 | 360/360 scored (amended read gate) | buffered beats O_DIRECT at every size and phase; the store's read path, not the drive, is the ceiling |
| B3 cold | window 1 unscored (host-level foreign I/O); **window 2 scored**: `mmap-normal` winner 1.191x, every other arm loser | readahead mmap beats the worker when the bank fits the page cache after one pass |
| B3 warm | unscored (ill-posed gate); read-gate rescoring, post-hoc: both mmap arms 1.19x winners | reported only as post-hoc |
| B3 bounded | **scored**: `worker16` beats every challenger (`mmap-random` 0.093, `mmap-normal` 0.435, `pread16` 0.407, `direct16` 0.893; `worker2` insufficient at 0.460) | the worker is the right path under memory pressure |
| run-spec | `=== SELF-CONSISTENCY PASS ===` for `worker16` and `direct16` | K=1..8 identical to plain |
| B2 | 1 GiB 5/5, 8 GiB 5/5 | export about 1.5 to 1.7 GB/s, import about 1.4 GB/s; engine path, not the drive |
| B4 | `worker16` registered descriptive row; post-hoc mapped challengers 1.17x (c=1) and 1.18x (c=4) | lower TTFT, TPOT and ITL too |
| B6 | 200 samples; pinned beats pageable at every size and direction | completes the ten-size G2 matrix on the target class |

> **Correction, 2026-09-26 (found on the 5090 half, checked here on the mirrored logs): every
> `worker2` visit in every B3 regime fell back to mmap hundreds of times** (mean fallbacks per
> visit from the `[spill-pread]` totals line: warm 455, cold 591, cold window 2 737, bounded 1,404;
> the smoke 517). The engine quotes the first three reasons per visit; all 123 quoted lines read
> `[spill-pread] falling back to mmap: worker read ring is busy`: with two buffers, a demand read
> that finds none free takes the mmap path instead of waiting. The registered fallback gate covered only the direct arm, so these visits passed.
> The `worker2` rows below therefore measure a mixed worker-plus-mmap program, not the depth-2
> worker. `worker16`, `pread16` and `direct16` had zero fallbacks on this box. The rows stay as
> recorded; the mechanism and the fix candidate are OWED 26.

No default changes from this box alone: per CLAUDE.md a default needs both rigs. The regime-shaped
result (mapped access wins while the bank fits in RAM, the positioned-read worker wins under memory
pressure) is the input for the 5090 half and for any per-regime policy decision.

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

## B3: expert-bank spill arms (`run-gen`, 128 greedy tokens, 8 slots, every expert on the disk tier)

Every visit passed correctness: gate `MATCH`, 128 token ids identical to `mmap-random`'s,
placement `0 pinned / 30720 mmap'd`, no config fallback, zero read errors and short reads;
`direct16` zero fallbacks and `overread_bytes == 4096 x reads` on every visit (for example
813,555 reads and 3,332,321,280 over-read bytes). No arm was refused in the cold or warm regime.

Two smokes preceded the scored runs (one round, never scored): the first used the registered
`MEMRA_SPILL_PINNED_FRAC=0`, which the engine rejects (`using 0.6`), so every expert was
pinned in host RAM and all six arms ran 36.7 to 37.1 tok/s (the pinned-host tier, recorded as a
diagnostic); the second, after B3 amendment 2, is the one that sized the regimes.

### Cold regime (the artifact out of the page cache at each visit start)

Registered verdict: **regime unscored**. Five of 60 visits carried host-level foreign I/O on
the drive (6.6 to 14.4 GB read, up to 2.2 GB written, by something outside this container while
nothing else of this lane ran), and `mmap-random` had three of them, above the registered two.
The read gate agrees (those were real foreign reads), so cold is unscored either way. Descriptive
numbers:

| Arm | Decode tok/s median (range), all visits | Device read GB per visit (median) | Artifact resident at end (median) | Read-gate verdict vs `worker16` (median ratio, scored pairs) |
|---|---|---|---|---|
| `worker16` | 22.48 (18.77 to 22.57), N=10 | 12.8 | 0.702 | baseline |
| `mmap-random` | 7.75 (7.36 to 7.77), N=10 | 12.8 | 0.701 | unscored-regime (0.344, 6 pairs) |
| `mmap-normal` | 26.79 (26.53 to 26.89), N=10 | 16.9 | 0.931 | unscored-regime (1.189, 9 pairs) |
| `pread16` | 16.35 (16.25 to 16.44), N=10 | 14.5 | 0.797 | unscored-regime (0.728, 9 pairs) |
| `worker2` | 14.09 (14.02 to 14.14), N=10 | 12.8 | 0.702 | unscored-regime (0.627, 9 pairs) |
| `direct16` | 9.59 (9.02 to 10.17), N=10 | 409.1 | 0.147 | unscored-regime (0.406, 9 pairs) |

"Cold" is cold only at the visit's start: the buffered arms re-read their first pass from the
drive (12.8 GB) and then run from the page cache (70% of the file resident at the end), which is
why `worker16` outruns the drive (22 tok/s x 454 MB per token). `direct16` reads every expert
from the drive on every token (409 GB per visit) and is the drive-bound rate: about 10 tok/s,
4.6 GB/s during decode.

### Cold regime, second window (registered gate; the scored cold result)

Rerun after B6 in a new window (`b3-cold-w2`), same binary, lock, arms and order: all 60 visits
clean (no foreign I/O this time: the largest foreign share was 0.26%), all correctness gates
passed, and the regime scores under the registered gate (the read gate agrees):

| Arm | Scored | Decode tok/s median (min to max) | Device read GB per visit | Artifact resident at end | Read s (worker or blocking) | Owner wait s | Verdict vs worker16 |
|---|---|---|---|---|---|---|---|
| `worker16` | 10/10 | 22.35 (22.23 to 22.50) | 12.8 | 0.702 | 115.85 | 19.84 | baseline |
| `mmap-random` | 10/10 | 7.71 (7.68 to 7.72) | 12.8 | 0.701 | n/a | n/a | loser (0.344) |
| `mmap-normal` | 10/10 | 26.66 (26.61 to 26.85) | 16.9 | 0.931 | n/a | n/a | winner (1.191) |
| `pread16` | 10/10 | 16.04 (15.84 to 16.13) | 14.5 | 0.797 | 40.01 | 0.00 | loser (0.717) |
| `worker2` | 10/10 | 13.73 (13.58 to 13.86) | 12.8 | 0.702 | 42.68 | 45.22 | loser (0.613) |
| `direct16` | 10/10 | 10.25 (10.21 to 10.32) | 409.1 | 0.147 | 534.72 | 83.55 | loser (0.459) |

```text
M1-VERDICT regime=cold arm=mmap-random vs worker16: loser median_ratio=0.3444 pairs=10
M1-VERDICT regime=cold arm=mmap-normal vs worker16: winner median_ratio=1.1913 pairs=10
M1-VERDICT regime=cold arm=pread16 vs worker16: loser median_ratio=0.7168 pairs=10
M1-VERDICT regime=cold arm=worker2 vs worker16: loser median_ratio=0.6126 pairs=10
M1-VERDICT regime=cold arm=direct16 vs worker16: loser median_ratio=0.4585 pairs=10
```

Window two's medians against window one's (all visits): `worker16` -0.6%, `mmap-random` -0.5%,
`mmap-normal` -0.5%, `pread16` -1.9%, `worker2` -2.6%, `direct16` +6.9% (window one's median
includes its contaminated visit, 9.02 tok/s). The ranking and every verdict direction are the
same in both windows.

### Warm regime (the whole artifact in the page cache before each visit)

Registered verdict: **regime unscored**, but for a different reason than cold: a warm buffered
visit reads nothing from the drive, so the registered foreign-share gate divides about 21 MB of
log and journal writes (kernel-thread I/O no process accounting sees) by zero own device bytes
and marks every buffered visit contaminated. Rescored with the B1 read gate (an amendment made
after seeing this data, reported as such, `summary-read-gate.json`; no visit had a foreign read
except one `direct16` visit with 6.2 GB, 1.5% of its 406 GB):

| Arm | Decode tok/s median (range), all visits | Device read GB per visit (median) | Artifact resident at end (median) | Read-gate verdict vs `worker16` (median ratio, scored pairs) |
|---|---|---|---|---|
| `worker16` | 26.27 (25.85 to 26.61), N=10 | 0.0 | 1.000 | baseline |
| `mmap-random` | 31.09 (31.00 to 31.42), N=10 | 0.0 | 1.000 | winner (1.188, 10 pairs) |
| `mmap-normal` | 31.11 (31.02 to 31.48), N=10 | 0.0 | 1.000 | winner (1.186, 10 pairs) |
| `pread16` | 20.88 (20.62 to 21.21), N=10 | 0.0 | 1.000 | loser (0.796, 10 pairs) |
| `worker2` | 17.71 (17.66 to 17.81), N=10 | 0.0 | 1.000 | loser (0.674, 10 pairs) |
| `direct16` | 9.71 (8.89 to 10.12), N=10 | 406.4 | 1.000 | loser (0.369, 10 pairs) |

With the bank in the page cache the mapped arms beat the worker by 19% (31.1 vs 26.3 tok/s), and
the positioned-read arms rank by concurrency: `worker16` over `pread16` over `worker2`.
`direct16` stays drive-bound.

### Bounded page cache regime (touched balloon, B3 regime (iii) amendment)

The balloon touched 120,240,163,328 bytes in the swapless cgroup, leaving 6,992,789,504 bytes of
headroom; it held for all 60 visits and released cleanly. The bound held on every scored visit:
the artifact was about 30% resident at each visit's end (`direct16` 15%), so the buffered arms
read the bank from the drive (95 to 181 GB per visit).

Registered verdict: **the regime scores**, and the read-gate rescoring agrees. `worker16` beats
every challenger:

| Arm | Scored | Decode tok/s median (min to max) | Device read GB per visit | Artifact resident at end | Read s (worker or blocking) | Owner wait s | Verdict vs worker16 |
|---|---|---|---|---|---|---|---|
| `worker16` | 10/10 | 10.86 (10.49 to 16.65) | 95.1 | 0.297 | 236.87 | 48.00 | baseline |
| `mmap-random` | 10/10 | 1.00 (0.99 to 1.01) | 116.1 | 0.296 | n/a | n/a | loser (0.093) |
| `mmap-normal` | 9/10 | 4.71 (4.61 to 4.86) | 180.5 | 0.312 | n/a | n/a | loser (0.435) |
| `pread16` | 10/10 | 4.42 (4.29 to 8.21) | 150.8 | 0.299 | 121.82 | 0.00 | loser (0.407) |
| `worker2` | 8/10 | 4.94 (4.91 to 7.63) | 95.1 | 0.298 | 111.26 | 114.20 | insufficient (0.460) |
| `direct16` | 8/10 | 9.73 (9.49 to 10.03) | 409.1 | 0.147 | 529.26 | 86.53 | loser (0.893) |

```text
M1-VERDICT regime=bounded arm=mmap-random vs worker16: loser median_ratio=0.0926 pairs=10
M1-VERDICT regime=bounded arm=mmap-normal vs worker16: loser median_ratio=0.4353 pairs=9
M1-VERDICT regime=bounded arm=pread16 vs worker16: loser median_ratio=0.4074 pairs=10
M1-VERDICT regime=bounded arm=worker2 vs worker16: insufficient median_ratio=0.4597 pairs=8
M1-VERDICT regime=bounded arm=direct16 vs worker16: loser median_ratio=0.8931 pairs=8
```

`worker2` is `insufficient` only because two of its visits carried foreign I/O and left 8 pairs
(4 and 4 are needed per order; its median ratio is 0.46). Under memory pressure the 4 KiB fault
path is ten times slower than the worker (1.0 vs 10.9 tok/s), readahead recovers part of it
(4.7), and `worker16` stays above `direct16` (10.9 vs 9.7) because its buffered reads still hit
the 5.7 GB of page cache the balloon left.

### run-spec self-consistency (B3 step 1 for the two speculative-capable arms)

```text
worker16: === SELF-CONSISTENCY PASS ===   K=1..8 each "self-consistency: PASS (identical to plain target)"
direct16: === SELF-CONSISTENCY PASS ===   [spill-pread] reads=2379237 ... fallbacks=0 ... overread_bytes=9745354752 (= 4096 x reads)
```

`spec-worker16/`, `spec-direct16/`. On this 8-slot spill shape speculation does not pay
(`worker16`: 22.72 tok/s at K=4, 1.04x plain; 14.18 at K=8, 0.65x), which is expected: every
drafted token multiplies the expert reads.

## B2: KV host-tier handoff (`kv-handoff-gate`, resident model, spec off)

Two recorded failures preceded the passing cells, each fixed by a registered amendment before any
passing cycle: attempt 1 at 1 GiB never filled the host tier (the spec route does not publish the
prompt-end seed; amendment 2 sets `MEMRA_SERVE_SPEC=0`), and attempt 1 at 8 GiB plateaued at
8,520,110,208 host bytes because the default 50% per-tenant share cap evicted the oldest entries,
the probe prompts (amendment 3 sets the single-tenant cell to 100%). That attempt still exported
8,520 MB in 5.2 s and imported it in 6.1 s; its receipts are kept.

| Cell | Cycles passed | Entries / MB exported | Export ms (write / fsync) | Import s | Probes |
|---|---|---|---|---|---|
| 1 GiB | 5 / 5 | 17 / 2,161.8 | 1,456 to 1,460 (949 / 326) | 1.5 each | 6,496 cached tokens, text identical to cold, 20 of 20 |
| 8 GiB (share 100%) | 5 / 5 | 76 / 9,664.6 | 5,772 to 5,898 (about 4,245 / 1,331 to 1,467) | 6.9 each | 6,496 cached tokens, text identical to cold, 20 of 20 |

The export includes the drain-demote of the device entries (8 per cycle) before the write. The
storage rates are about 1.5 GB/s (1 GiB) and 1.7 GB/s (8 GiB) for export end to end (serialize, buffered write and
fsync) and 1.4 GB/s for import (a validated, frame-by-frame buffered read with re-materialization,
cold file): both are well under the drive's 5.1 to 6.4 GB/s write and 7 GB/s read, so the
handoff's cost is the engine path, not the storage. Its O_DIRECT arm stays OWED 18.

## B4: serving shape (`memra-server`, warm regime, the server's default spec route)

No challenger won under the registered B3 gate, so `worker16` is the registered descriptive
row; `mmap-random` and `mmap-normal` ran beside it as the warm regime's post-hoc read-gate
winners (`M1-PREREG.md` "B4 arms"), in one round-robin schedule: 60 visits, every visit scored
(zero request errors, valid telemetry, warm regime confirmed). Medians over N visits, each visit
32 streamed requests of 128 tokens:

| c | Arm | N | tok/s | req/s | TTFT s p50 / p95 / p99 | E2E s p50 / p99 | TPOT ms p50 / p99 | ITL ms p50 / p95 / p99 |
|---|---|---|---|---|---|---|---|---|
| 1 | `worker16` | 10 | 19.14 | 0.150 | 0.70 / 0.72 / 1.99 | 6.57 / 8.11 | 46 / 50 | 0 / 139 / 141 |
| 1 | `mmap-random` | 10 | 22.44 | 0.175 | 0.60 / 0.60 / 1.75 | 5.60 / 6.97 | 39 / 43 | 0 / 118 / 119 |
| 1 | `mmap-normal` | 10 | 22.42 | 0.175 | 0.60 / 0.60 / 1.75 | 5.60 / 6.97 | 39 / 43 | 0 / 118 / 119 |
| 4 | `worker16` | 10 | 27.70 | 0.216 | 1.70 / 3.76 / 3.76 | 18.28 / 20.25 | 130 / 131 | 129 / 133 / 135 |
| 4 | `mmap-random` | 10 | 32.70 | 0.255 | 1.43 / 3.25 / 3.25 | 15.46 / 17.21 | 110 / 111 | 109 / 113 / 114 |
| 4 | `mmap-normal` | 10 | 32.69 | 0.255 | 1.43 / 3.24 / 3.24 | 15.45 / 17.27 | 110 / 111 | 109 / 112 / 114 |

```text
M1-B4-VERDICT c1/mmap-random vs worker16: winner median_tok_ratio=1.1727
M1-B4-VERDICT c1/mmap-normal vs worker16: winner median_tok_ratio=1.1720
M1-B4-VERDICT c4/mmap-random vs worker16: winner median_tok_ratio=1.1802
M1-B4-VERDICT c4/mmap-normal vs worker16: winner median_tok_ratio=1.1803
```

Post-hoc challengers, reported as such: in serving shape the mapped arms beat `worker16` by 17%
at c=1 and 18% at c=4, with lower TTFT, TPOT and ITL at every percentile, reproducing the warm
B3 result (1.19x) through a different binary and the spec route. ITL p50 is 0 at c=1 because the
spec route delivers accepted bursts as consecutive frames.

## B6: the five unrun G2 sizes (D's worker, pinned vs pageable, 5 AB plus 5 BA each)

`RESULT {"campaign": "G2", "status": "all-visits-complete", "samples": 200, ...}`; D's
summarizer validated the capture (250 ms telemetry, 600/600 W on every sample; GPU 30 to 40 C):

| Bytes | Direction | Arm | N | Median GiB/s | Median us per copy |
|---|---|---|---|---|---|
| 16,384 | d2h | pageable | 10 | 1.49 | 10.26 |
| 16,384 | d2h | pinned-cacheable | 10 | 1.75 | 8.70 |
| 16,384 | h2d | pageable | 10 | 1.17 | 13.09 |
| 16,384 | h2d | pinned-cacheable | 10 | 1.58 | 9.69 |
| 262,144 | d2h | pageable | 10 | 10.18 | 23.98 |
| 262,144 | d2h | pinned-cacheable | 10 | 19.06 | 12.81 |
| 262,144 | h2d | pageable | 10 | 10.13 | 24.11 |
| 262,144 | h2d | pinned-cacheable | 10 | 13.52 | 18.06 |
| 4,194,304 | d2h | pageable | 10 | 15.99 | 244.30 |
| 4,194,304 | d2h | pinned-cacheable | 10 | 37.99 | 102.81 |
| 4,194,304 | h2d | pageable | 10 | 19.18 | 203.67 |
| 4,194,304 | h2d | pinned-cacheable | 10 | 22.17 | 176.21 |
| 67,108,864 | d2h | pageable | 10 | 18.13 | 3448.19 |
| 67,108,864 | d2h | pinned-cacheable | 10 | 49.74 | 1256.54 |
| 67,108,864 | h2d | pageable | 10 | 19.53 | 3200.70 |
| 67,108,864 | h2d | pinned-cacheable | 10 | 49.23 | 1269.45 |
| 1,073,741,824 | d2h | pageable | 10 | 18.37 | 54434.15 |
| 1,073,741,824 | d2h | pinned-cacheable | 10 | 51.10 | 19568.04 |
| 1,073,741,824 | h2d | pageable | 10 | 19.30 | 51810.55 |
| 1,073,741,824 | h2d | pinned-cacheable | 10 | 50.25 | 19898.92 |

Pinned (cacheable) beats pageable at every size in both directions, from 16 KiB up; at 64 MiB and
1 GiB pinned reaches about 50 GiB/s each way against 18 to 20 GiB/s pageable. Together with D's
day-10 sizes this completes the registered ten-size G2 matrix on the target class.
