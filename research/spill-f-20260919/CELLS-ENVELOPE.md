# G2: bounded host/device transfer pre-registration

Date: 2026-09-19. **Protocol only; no G2 GPU samples have run.** This is a
rented-development RTX 5090 envelope, not a serving gate, storage score,
runtime-default decision, or performance-board update. M1 storage provenance
is independent of these host/device copies and remains unproven.

## Question and fixed matrix

Measure completed H2D and D2H copies for **A = ordinary pageable allocation**
and **B = CUDA page-locked allocation**, with the same bytes, device, stream,
copy count, and binary. CUDA's internal pageable staging is part of A, not a
reason to relabel A as pinned. Use existing cudarc driver types; do not add a
kernel, model path, fallback format, or environment-variable door.

| Dimension | Pre-registered values |
|---|---|
| Transfer bytes per operation | 4 KiB, 16 KiB, 64 KiB, 256 KiB, 1 MiB, 4 MiB, 16 MiB, 64 MiB, 256 MiB, 1 GiB |
| Direction | H2D and D2H, separately; never report round-trip bandwidth as one-way bandwidth |
| Host allocation | A pageable; B page-locked via CUDA, with allocation failure fatal (no substitution) |
| In-flight depth | 1; one stream, synchronize to completion per operation |
| Outer order per size/direction | For rounds 0..4: AB then BA (five AB and five BA pairs) |
| Independent observations | 10 arm visits each, not the thousands of copies inside a visit |
| Thermal class | Short-cell, allocation-warm; **not** thermal steady-state or full-power |
| Power condition | Read and retain enforced cap per arm visit; current inventory is 400 W, maximum 600 W |

Each size/direction is one bounded cell. Default sequence is ascending size,
H2D then D2H. Comparisons are within a cell only; the sequence is not evidence
that two different sizes ran in the same thermal state. Keep AB and BA paired
ratios and per-round signs beside any pooled median. No best-of-N selection.

## Correctness before timing

1. Allocate at most one device buffer, one pageable buffer, and one pinned
   buffer for the current size. Fill host input with a fixed, versioned,
   nonuniform byte pattern. Initialize/touch all host pages before timing.
2. Before timed visits, run both allocation arms in both directions. Verify
   every returned byte and a SHA-256 against the expected pattern, after an
   explicit stream synchronization. Readback for H2D verification is outside
   its timed interval; D2H device seeding is outside its timed interval.
3. Every visit starts from an explicitly seeded source and poisoned destination;
   verify the complete destination after the visit. Keep inputs/outputs alive
   until completion. Reusing an allocation is allowed; using a previous
   successful readback as evidence for the next visit is not.
4. Correctness controls use at least two different patterns, plus a comparator
   red control with one corrupted returned byte. The red control must refuse,
   without changing GPU contents or the timing path. A mismatch, asynchronous
   CUDA error, failed allocation, or missed synchronization aborts the cell.
5. Retain expected and returned hashes, verified byte count, and errors. A
   completed copy is opaque-byte identity only; it does not qualify KV state,
   eviction, cancellation, stream-crossing ownership, or model generation.

## Timing and budgets

- **Five-minute hard cap per collector invocation**, including setup, warmup,
  correctness, timing and final checks. A timeout is incomplete, not a lower-N
  result. At most one five-minute execution in this lane's initial probe
  milestone; the full 20-cell sweep requires a separately scheduled window.
- Per cell, perform fixed untimed warmups in each arm, then a discarded paired
  calibration. Choose ONE inner copy count for both arms and all visits from
  the slower calibration arm, targeting approximately 3 seconds per visit,
  clamped to 1..100,000 copies. Record the count and calibration measurements.
  Do not retune counts after seeing the scored arm results. If the 300-second
  envelope cannot fit the registered count, mark incomplete and reschedule.
- Twenty scored visits target at least 60 seconds of aggregate timed work per
  cell. This is still short-cell evidence, not a sustained thermal soak. If
  actual aggregate timed work is below 60 seconds, retain it as diagnostic
  and rerun with a larger shared count; do not silently combine attempts.
- Record host monotonic wall time around submission **and completion**, and
  CUDA-event elapsed time on the same stream. Wall includes host API/staging
  overhead; event time is device-stream elapsed, not assumed DMA-only time.
  Allocation, initialization, hashing and verification are timed separately.
  Small-copy event quantization/launch overhead must remain visible.
- Peak explicit payload allocations: pinned <=1 GiB, pageable <=1 GiB,
  device <=1 GiB. Allow <=512 MiB extra host memory for context/runtime/logging;
  do not allocate a second full-size verification copy. Preflight requires
  >=4 GiB available host memory and >=2 GiB free VRAM. Driver-internal staging
  is additional/unknown; observe RSS and VRAM rather than claiming the explicit
  budgets bound CUDA's internals. Allocation refusal never falls back.
- Disk writes <=100 MiB per cell for raw logs, hashes and telemetry; no model
  or storage payload is written. Root available space must remain >=20 GiB.
  No cache drops, power/clock changes, raw-device access or other-lane cleanup.

## Collector binding and exclusion

Builds happen before the measured window. Use an exact source revision and
binary SHA-256. A future `h2d_probe.rs` should expose explicit CLI arguments
(size, direction, rounds, inner count), write one sample record per arm visit,
and end with one `RESULT` object. It is **not implemented by this document**;
no placeholder binary or untested runtime support is introduced.

The command shape, once that binary exists, is:

```sh
python3 tools/tier-battery.py --rig rtx5090 --timeout 300 \
  --out /root/wt-f/receipts/g2-size-direction-attempt \
  --execute /root/wt-f/target/release/h2d-probe <registered-arguments>
```

This is a command template, not a run receipt. Resolve arguments before launch;
never run the GPU binary outside the collector. The collector takes the
canonical `/tmp/memra-5090.lock`. No third lock filename and no per-card
parallel campaigns. Check compute applications and competing benchmark
processes before launch; a lock alone cannot exclude nonparticipating jobs.
If any peer is busy, do not start. Coordinate the whole timed window with the
lead, including CPU/I/O-heavy builds; a previously idle snapshot is not an
ongoing reservation. Retain before/after compute-app snapshots and the
failure-time snapshot. A discovered co-tenant invalidates timing, not the logs.

The collector's existing 250 ms `nvidia-smi` CSV records clocks, power draw,
temperature, VRAM, utilization and negotiated PCIe link. It does **not** include
`power.limit`, host allocation counters, or actual transfer-byte counters.
The runner/wrapper therefore must additionally:

- Query `power.limit,power.max_limit` at each arm visit's start and end, outside
  timing, plus record `power.limit` at 250 ms if checking for mid-visit changes.
  A missing, changed, or unparseable enforced limit makes the comparison
  unqualified. Do not infer a cap from power draw or the advertised maximum.
- Record monotonic visit boundaries and operation counts/bytes so sample
  windows can be matched to telemetry. Log measured pinned/pageable/device
  allocation sizes, not fabricated tier counters.
- Verify actual telemetry spacing and full window coverage (nominal 250 ms,
  no gap >500 ms); missing/truncated telemetry invalidates scoring. A raw
  collector status of `executed-not-qualified` is not a G2 pass.
- Retain full stdout/stderr before parsing, timeout and exit status, binary
  and collector hashes, source revision, driver/toolkit versions, power
  snapshots, and payload hashes. Before public sync, remove deployment
  identity metadata from a separate sanitized export; keep original private
  captures outside git. Do not publish collector instance/price fields.

## Publication contract and remaining gates

Publish a dated research condition table only: size, direction, allocation arm,
completed bytes/count, wall and event latency, GiB/s with explicit denominator,
N=10 visits (AB=5, BA=5), median/range and paired ratios, thermal class, enforced
power cap, measured clocks/temperature/link, identity result, failures, raw-file
hashes and exact revision. Descriptive p95/p99 over ten visit aggregates are
not request tail-latency evidence. No TTFT/TPOT/ITL or serving-throughput claim
exists in this copy-only experiment. Do not change generated performance boards.

**Pending:** native transfer runner and correctness controls; CUDA build and
Mac compile check for that runner; per-visit power/byte binding; collector-only
GPU execution; raw receipt sync and hash verification. No probe result is
claimed. These are G2 implementation/execution gaps, not an NVMe blocker and
not a reason to run a bare GPU test. This pre-registration freezes the matrix
before measurements; changing it requires a dated amendment, not retroactive
selection of the faster cells.
