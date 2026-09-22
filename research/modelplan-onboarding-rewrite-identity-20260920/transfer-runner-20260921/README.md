# Transfer runner preparation, 2026-09-21

CPU-only preparation on the frozen 222d5040 source derivative. The new phase is
intended for the final reviewed dependency composition and a fresh owned build;
no new native build, GPU execution or qualification is recorded here.

The exact owned executable inventory now includes `tier-transfer-gate`. The
transfer phase runs conformance and roundtrip under the existing per-card lease,
supervisor, telemetry and finalization code. It requires all eleven conformance
markers and all six unique roundtrip sizes, equal hashes, N=1 and every lifecycle
assertion. The source-contract census was checked against reviewed ed5a4f67;
`gate-contract.json` records its immutable gate source hash. This does not import
that dependency or treat source inspection as a native result.

Validation:

- `test_qualify_callers.py`: 17 tests pass, including 66 complete-run transfer
  scheduling/transcript/exit controls and finalization faults in all three phases.
- `test_qualify_native.py`: 33 tests pass, including old manifests and missing,
  stale or non-executable transfer binaries refusing before GPU work.
- `test_finalization_signals.py`: three tests pass, covering 12 actual SIGTERM
  scenarios across baseline, callers, transfer and battery. Native commands are
  mocked; the signals and publication checks are real.

Commands use `python3 -B` and the script path in this research directory. Logs
retain their actual test summaries. The initial caller run failed its test-only
path equality because macOS resolves `/var` through `/private/var`; the fixture
now resolves its temporary root before comparison. Its truncated tool output is
retained as `caller-initial-path-failure-excerpt.log`, explicitly an excerpt. The
successful caller log was captured directly to disk. Provenance and signal logs
were assembled in order from the untruncated command-output chunks.

`evidence.json` seals the runner/test source and evidence hashes. Earlier build
records and native evidence remain bound to their original source and binaries.
