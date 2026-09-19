# WP-A queued cells / pre-registration — 2026-09-19

**No GPU/NVMe measurement has run.** CPU smoke is not NVMe evidence. Booking M1:
2026-09-23; M3 feasibility: 2026-09-24. Rig booking/artifact pins are lead-owned.
Two bounded connectivity attempts failed; remote inventory is unavailable.
Paused model path stays paused and is absent from every command below.

## Preconditions for EVERY scored hardware window

- Non-serving rig explicitly released by its owner; canonical lock held across the
  whole window, both cards included. 5090 `/tmp/memra-5090.lock`; PRO pair/target
  `/tmp/memra-gpu.lock`. No second card campaign, no new lock name.
- Pin git commit, binary SHA256, artifact byte manifest, fixture/trace hash,
  filesystem/mount/local-NVMe device, NUMA, driver and actual resolved backend.
  Stage byte-identical artifacts from /data onto /scratch. No cache-dropping,
  driver changes, kernel modules, or GDS installation without approval.
- Sparse logical tables qualify capacity accounting only. Scored cold misses must
  read physically materialized non-hole extents, proven by device I/O counters;
  sparse-hole reads/page-cache hits cannot be reported as NVMe throughput.
- Record nvidia-smi topology/compute-apps before/after and at failure; capture
  clocks, temperature, power, memory and PCIe at 250 ms with host/SSD/queue
  counters. Retain thermal/cache regime and warmups. No telemetry means no score.
- Forced exactness first. Candidate comparisons use five AB AND five BA pairs,
  same binary/window, N=10 observations per arm. Raw stdout/stderr first through
  tee; parse saved files second. Preserve exit statuses with `set -o pipefail`.
- Serving consumers record TTFT/E2E/TPOT/ITL p50/p95/p99, request/token throughput,
  actual cache misses and valid/submitted/physical/H2D bytes. Microbench timing
  cannot substitute for B/C's model-consumption cells.

## Runnable now (GPU test command queued, not executed)

A1-CPU runs in the scaffold. No GPU lock is needed:

```sh
cargo test --manifest-path research/spill-a-20260919/scaffold/Cargo.toml \
  -p memra-tier-spill-a-scaffold --offline
```

A2-existing-worker-pinned-I/O, fitting 5090 then non-serving PRO pair. This
existing ignored test checks pinned worker exact reads/reuse, NOT actual H2D/D2H
roundtrip. Budget 5 min per rig including startup, max pread depth 2 and 64-byte
buffers in test; zero mismatches/errors except injected EOF. After build outside
the scored window:

```sh
set -o pipefail
flock -x /tmp/memra-5090.lock cargo test -p memra-engine --lib \
  spill_pread::tests::worker_positioned_reads_preserve_exact_bytes_and_reuse_after_short_read \
  -- --ignored --exact --nocapture 2>&1 | tee research/spill-a-20260919/raw/a2-worker-5090.log

flock -x /tmp/memra-gpu.lock cargo test -p memra-engine --lib \
  spill_pread::tests::worker_positioned_reads_preserve_exact_bytes_and_reuse_after_short_read \
  -- --ignored --exact --nocapture 2>&1 | tee research/spill-a-20260919/raw/a2-worker-pro.log
```

## Proposed CLI commands queued for later implementation (NOT runnable day-1)

The day-1 CLI accepts only `cpu-fixture`; every command here is a precise
pre-registered intended interface and currently fails closed. Missing real CUDA
adapter/direct/io_uring/trace engine/telemetry is an implementation blocker, not a
skip/pass. Lead may adjust command spelling at freeze but must retain budgets.
Run from pinned checkout. `$RECEIPTS` is a caller-owned dated mechanism directory;
its creation and cleanup of scratch belong to the measurement task.

| Cell | Exact intended command (wrap with correct rig lock below) | Pre-registered budget / pass condition |
|---|---|---|
| A2-byte-roundtrip | `target/release/storage-bench roundtrip --directions h2d,d2h --sizes 1,264,288,4095,4096,4097,1048576,933232640 --slot-bytes 1048576 --slots 4 --reserved-demand-slots 1 --repeats 100 --faults cancel,late-fence,lost-fence --telemetry-ms 250 --out "$RECEIPTS"` | 5090 then PRO pair; 15 min each; at most 4 MiB pinned staging, object may exceed pool; zero byte mismatches, no request-path pin/free; quarantine until disk+DMA+consumer retirement. |
| M1-row-amplification | `target/release/storage-bench trace --trace opaque-row264x48-v1 --logical-table-bytes 202758032400 --requests-per-second 800 --seconds 1800 --backends worker,pread,mmap,direct,uring --slot-bytes 1048576 --slots 8 --reserved-demand-slots 2 --scratch /scratch/spill-a-row --order ab5,ba5 --telemetry-ms 250 --out "$RECEIPTS"` | 5090 and PRO, 8 MiB pinned staging, bounded sparse/streamed fixture not 203 GB allocation; 38,400 row lookups/s and measured straddles/coalescing; row-batch p99 <=10 ms, zero growing queue in final 20 min. Full 30min per arm/order is 10h per pairwise comparison; book one mechanism window, not parallel cards. |
| M1-bulk-restore | `target/release/storage-bench trace --trace opaque-bulk-v1 --sizes 116654080,933232640 --restores-per-second 1 --seconds 1800 --backends worker,direct,uring --slot-bytes 1048576 --slots 8 --reserved-demand-slots 2 --scratch /scratch/spill-a-bulk --order ab5,ba5 --telemetry-ms 250 --out "$RECEIPTS"` | 5090 then PRO; at most 8 MiB pinned; 116.654 MB restore p99 <=100 ms, 933.233 MB restore p99 <=500 ms; exact valid byte hash; sustained backlog bounded. 933MB/s demand remains conditional on measured <=70% route capacity. |
| M1-read-write-interference | `target/release/storage-bench trace --trace opaque-mixed-v1 --row-batches-per-second 800 --rows-per-batch 48 --row-bytes 264 --restore-bytes 116654080 --restores-per-second 1 --backup-bytes-per-second 712000 --seconds 1800 --backends worker,direct,uring --slot-bytes 1048576 --slots 8 --reserved-demand-slots 2 --scratch /scratch/spill-a-mixed --order ab5,ba5 --telemetry-ms 250 --out "$RECEIPTS"` | PRO pair then target four-card whole-fabric window with D; same row/restore latency caps; demand <=70% measured sustained SSD and route capacities; no growing queue/dirty backlog over 30 min; explicit admission reduction requires owner approval. |
| A3-consumer-pipeline | `target/release/storage-bench replay --trace "$PINNED_NATIVE_TRACE" --manifest "$ARTIFACT_MANIFEST" --backends worker,pread,mmap,direct,uring --order ab5,ba5 --telemetry-ms 250 --out "$RECEIPTS"` | Hy3/PLE/Qwen adapter traces provided by C/B; no numeric changes. 5090 fitting cases + PRO full-size; pre-register artifact/trace and actual memory caps before run. Compare storage through GPU consumer, not disk-only. |

Lock invocation for each proposed command: `flock -x /tmp/memra-5090.lock ...`
on the 5090; `flock -x /tmp/memra-gpu.lock ...` on PRO pair/four-card target.
Use `... 2>&1 | tee "$RECEIPTS/raw.log"` with pipefail, then parse raw log.
Locks must cover the telemetry and compute interference generator too, not just
storage subprocesses. One orchestrated campaign owns all child activity.

## Decision budgets (engineering targets, not observed performance)

- Byte/integrity failures, stale completion publication, reuse before retirement,
  accepted item omission, hidden direct fallback or quota overflow are hard fails.
- Throughput admission <=70% of measured sustained route AND SSD bandwidth/IOPS;
  a nominal 7/14 GB/s SSD label or 64 GB/s PCIe ceiling cannot set this budget.
- Candidate default requires >=5% end-to-end pipeline improvement, both-order
  consistent direction, and <=2% p95/p99 serving-latency regression on every
  deciding workload; tails need long stationary samples, not N=10 percentile
  confidence. A one-rig win selects at most a one-rig default. Lead freezes final
  policy after B/C workload budgets are known; no default decision today.
- A flat/negative/no-go arm is removed in the deciding lane, not retained as a
  new environment door. Existing worker/default behavior is unchanged today.
- M3 GDS is feasibility only, optional. Read-only check actual GPU/toolkit/driver,
  mounted filesystem/direct-path prerequisites with A+D on 2026-09-24. No driver
  installation or compatibility-mode success counted as direct transfer.

## Remaining booking/fixture dependencies

B/C must provide immutable existing-model artifact and real native-consumer trace
pins; D supplies peer/fabric inventory and fake/real device fences. Lead confirms
non-serving rig permission, available RAM/NVMe and final run duration bookings.
The owner-approved paused path is not a fallback for missing artifacts.
