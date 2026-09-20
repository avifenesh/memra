# G2: bounded host/device transfer pre-registration

Date: 2026-09-19; amended 2026-09-20 for the native N=1 plumbing probe and
D's additive capture fields. **The full scored G2 sweep has not run.** This is a
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
  CUDA-event elapsed time on the same stream, summed from one independently
  synchronized event pair per operation. Wall includes host API/staging
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
binary SHA-256. `crates/memra-engine/src/bin/h2d_probe.rs` now implements the
**N=1 plumbing subset**, not the full balanced/calibrated sweep above. See
`H2D-BIN-FRAGMENT.md` for the lead-owned manifest fragment, allocation type,
Mac dry-run and timing interpretation. It accepts bytes/direction/order and
`--repeats 1 --copies C`, where `C` is in `1..100000` (default `1`).
Other repeat counts and out-of-range copy counts fail closed. Copies are
operations inside a visit, not independent observations. N=1 output has no
medians. D's `tools/tier-envelope.py` is the G2 calibration/balancing executor;
the probe does not implement that outer protocol. Cacheable pinned allocation (CUDA flags=0) is the concrete B arm;
write-combined host memory is not substituted for it.

The current auto-discovered binary command shape is:

```sh
python3 tools/tier-battery.py --rig rtx5090 --timeout 300 \
  --out /root/wt-f/receipts/h2d-n1-attempt \
  --execute /root/wt-f/target/release/h2d_probe --repeats 1 --copies 1
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

The collector's 250 ms `nvidia-smi` CSV now includes `power.limit` and
`power.max_limit`, plus clocks, draw, temperature, VRAM, utilization and
negotiated PCIe link. Bind to
`research/spill-d-20260919/CAPTURE-CONTRACT.md`: both `command.capture.json`
and completed `CELL.jsonl` mirror `gpu_power_limits`, the distinct raw
`{device, power.limit, power.max_limit}` triples derived from the hash-checked
CSV. These are strings with units, not inferred numeric caps. Older captures
may omit them and cannot establish the new power envelope. No collector
host-allocation or transfer-byte counter is invented.

The runner/wrapper therefore must additionally:

- Bind each sample's `power_before` and `power_after` raw objects to that
  capture's `gpu_power_limits` for the matching integer device index. The
  probe selects NVML using the actual CUDA UUID without publishing it. Parse
  watts explicitly; unknown, empty, changed or unparseable limits cannot
  qualify a fixed-envelope comparison. A fixed 400/600 W pair is restricted
  power, not a full-power baseline. Do not infer the cap from power draw.
- Bind sample `unix_start_ns`/`unix_end_ns` to raw sampler wall timestamps;
  `mono_start_ns`/`mono_end_ns` are nanoseconds since this probe process's
  start, not a system-wide clock. Require monotonic boundaries and plausible
  wall-clock alignment. `bytes`, `copies`, `completed_bytes`, `verified_bytes`
  and `allocation` rows give explicit payload allocations/copy counts; they
  do not claim driver-internal staging bytes or fabricated tier counters.
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

## N=1 amendment and receipt shape (2026-09-20)

The plumbing run emits **bare JSONL** version 1: `allocation`, two `control` records
per sample (distinct patterns), `sample`, and a final `RESULT`. Sample rows
carry `arm`, `direction`, `bytes`, `n=1`, `copies=C`, `order=ab|ba`, complete-byte identity and
expected/actual SHA256, setup/verification duration, both clock boundaries,
`completed_bytes=bytes*C`, wall completion duration and summed per-operation
CUDA event elapsed (`event_timing=sum-per-operation-owner-stream`). The RESULT explicitly says
`qualified=false`; a missing RESULT, nonzero exit or incomplete matrix means
incomplete/refused. `--dry-run` emits all 40 sample shapes with null identity,
hashes, durations and power, zero completed bytes, and `dry-run-no-cuda`;
it cannot be mistaken for a GPU observation. CPU tests also reject corrupted
comparator input and unregistered CLI values.

CUDA event intervals include stream submission gaps; the current synchronous
single-copy measurement is not a DMA-only event benchmark. The N=1 probe
uses neither calibrated repeats nor full AB/BA balancing; no performance
comparison, pooled median, tail latency or default decision follows from it.
Full G2 calibration, balanced orders, sustained cells and scoring remain
pending. Native build/collector receipt status lives in `H2D-RESULTS.md` once
captured; Mac standalone source compilation is not a package CUDA build.

These are G2 implementation/execution gates, not an NVMe blocker and not a
reason to run a bare GPU test. This dated amendment narrows the initial run
to plumbing before any GPU observations, without changing the full scored
protocol above or retroactively selecting faster cells.


## Copies extension and D token binding (2026-09-20)

Final auto-discovered CLI (underscore; no shared manifest edit):

```sh
h2d_probe [--dry-run] [--bytes BYTES] [--direction h2d|d2h|both] \
  [--order ab|ba] [--repeats 1] [--copies 1..100000]
```

D revision `7f7bf547` is the compatibility authority. Its
`tools/tier-envelope.py::visit` consumes bare JSON rows beginning with `{`,
including one final `record=RESULT` summary. Its collector rejects multiple
line-start `RESULT ` tokens. Consequently **individual probe visits are not
`RESULT `-prefixed**; the outer worker emits that token exactly once. This is
an explicit correction of the initial prefix request to match the actual runner.
The CAPTURE-CONTRACT power fields and refusal token rules are unchanged.

A single plumbing-only collector invocation exercises 4 KiB and 16 MiB,
`copies=1` and `1000`, H2D and D2H, AB then BA: 32 visits, N=1 per
size/count/direction/arm/order, no calibration, aggregation or medians. Run:

```sh
python3 tools/tier-battery.py --rig rtx5090 --timeout 300 \
  --out /root/wt-f/receipts/h2d-copies-n1/collector --external-lock \
  --execute python3 research/spill-f-20260919/run-h2d-copies-plumbing.py \
  --probe /root/wt-f/target/release/h2d_probe \
  --out /root/wt-f/receipts/h2d-copies-n1/visits \
  --lock-fd @COLLECTOR_LOCK_FD@
```

The worker verifies the inherited canonical lock descriptor before executing
anything on CUDA; invoking it without that proof refuses. Nested probe children
remain in the collector worker process group so its timeout kills them before
releasing the lock. A CPU-only test exercises this and detects a deliberately
detached-child red control. Every probe invocation
retains raw output before parsing. `check-h2d-output.py` validates matrix shape,
copy accounting, order, flags=0, identity and the summary. This one-cell worker
is only CLI plumbing proof; **D's runner remains the G2 executor**. D's full
20-cell calibration/balancing/telemetry acceptance remains an independent gate.

Local CPU checks: three Rust unit tests, 12 dry-run invocations (including
100000-copy boundaries), and Linux-target engine/bin typecheck passed. The
`DOCS_RS=1` first attempt failed because `MEMRA_MMQ_ARCHIVE_HASH` is required at
compile time; rerunning with the explicit compile-only
`MEMRA_MMQ_ARCHIVE_HASH=docs-rs-typecheck-only` sentinel passed. Neither docs
stubs nor a cross-target check constitute native CUDA execution. Raw CPU logs
live in `h2d-copies/cpu/`; native build and single-cell status are in
`H2D-RESULTS.md`.
