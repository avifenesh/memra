# H2D copies plumbing receipt (2026-09-20)

**One native N=1 matrix executed, 32 visits: `executed-not-qualified`.** All
visits passed complete-byte identity and D's visit-field checks. No G2
performance result, median, runtime-default decision, NVMe proof or serving
qualification follows from this milestone.

## Exact code and conditions

- Engine/bin build source: memra `e8b4e642191b283231116f7a16ff45bb6f4713c5`.
- Collector worker checkout: `27b900b00001472dcd27ff6d8cd0a7dd27d55f9b`.
  The probe source is unchanged between these commits; source, collector and
  worker SHA-256 values match the captured `visits/identity.json`.
- Binary SHA-256: `d59b7d265a36e74e0d283255741142241b6a6edbbf831569d34e27293622d330`.
- Native build: `cargo build --release -j 16 -p memra-engine --bin h2d_probe`,
  CUDA 13.1, sm_120a. Raw build log, driver/toolchain and source/binary hashes:
  `h2d-copies/native-build/build.log` and `build-identity.log`.
- Hardware: rented-development RTX 5090, enforced cap **400 W**, maximum
  **600 W**. Restricted-power, allocation-warm short cell; not full-power or
  steady-state evidence.
- Matrix: 4 KiB and 16 MiB, copies 1 and 1000, H2D and D2H, AB then BA.
  **N=1 per size/count/direction/arm/order**, 32 visits total. Inner copies do
  not increase the independent observation count. No medians computed.
- Collector: canonical `/tmp/memra-5090.lock`, inherited descriptor verified
  by the worker, 300-second timeout. Execution completed in **25.306 seconds**,
  exit 0, not timed out. GPU compute-app snapshots were empty before/after.
- Owner's N=1-only ruling permits concurrent CPU builds, unlike scored G2.
  `concurrent_builds: []` at both recorded snapshots; these are observations,
  not continuous process surveillance. The capture-bound note is
  `h2d-copies/native-n1-attempt2/CELL-NOTE.json`.

## Receipts and checks

Successful cell: **`h2d-copies/native-n1-attempt2/`**. It contains collector
`CELL.jsonl`, hash-bound `command.capture.json`, complete stdout/stderr,
250 ms raw GPU telemetry, before/after compute-app snapshots, canonical lock
proof, eight per-invocation logs and exact code/binary identity.

Offline checks using D revision `7f7bf547` checked both successful and failed
capture integrity with `validate_cell`, and all 32 native visits with
`tier-envelope.py::validate_visit`. Full matrix uniqueness, copy accounting,
order, flags=0, identity, and source/worker/collector hashes also passed.
Machine-readable outcome: `h2d-copies/validation.json`.

CPU checks: three Rust tests, 12 dry-run invocations including copies 100000,
Linux-target engine/bin typecheck, rustfmt, and the timeout containment
positive/red control passed. The first DOCS_RS attempt failed on a missing
compile-time archive hash; the explicit docs-only sentinel retry passed.
Both logs remain in `h2d-copies/cpu/`. These checks do not replace CUDA evidence.

Before public export, optional deployment/cost fields in CELL rows were set
to null; command logs and telemetry are unchanged. Original and export hashes
are in `h2d-copies/export-manifest.json`; private originals stay outside git.

## Failed launch retained, not counted as a GPU observation

Initial preflights deferred for a peer's GPU process and then a CPU build;
those deferrals preceded the owner's explicit N=1 concurrent-build exception.
The first actual collector invocation exited 2 after 0.459 seconds, **before
CUDA execution**, with the raw diagnostic:

> python3: can't open file '/root/wt-f/research/spill-f-20260919/run-h2d-copies-plumbing.py': [Errno 2] No such file or directory

The F-only worktree was sparse: the committed worker had `S`/skip-worktree
status and was not materialized. Adding only `research/spill-f-20260919` to
that worktree's sparse checkout restored the exact committed script; no code
change was needed. Failed launch evidence remains in
`h2d-copies/native-launch-failed/`. There was **one actual GPU matrix cell**,
not a rerun selected for better timings.

## Note to D: nested timeout containment

**To D (`tools/tier-envelope.py::visit` at `7f7bf547`):** its nested
`B.tee_run` starts the probe in a fresh session, so the collector's outer
worker-process-group kill can leave the probe alive after the worker exits
and releases its inherited canonical lock. F's wrapper keeps probe children
in the collector worker group. The CPU-only regression
`test-h2d-worker-containment.py` verifies timeout kills that group and its
intentional detached-child red control reproduces the escape. Apply the
same ownership/containment correction to D's runner before full G2; this
finding is independent of the successful N=1 byte-identity cell.

Exact token compatibility: D consumes bare JSON sample and summary rows;
only the outer worker emits one `RESULT ` prefix. Per-visit prefixing would
break D's parser and the collector's singleton RESULT contract.

Full G2 calibration, balanced N>=5, sustained-duration and telemetry scoring
gates remain unrun. D's runner is the G2 executor. See `CELLS-ENVELOPE.md`
for the final CLI and publication boundaries.
