# cx-probehitch progress

## Scope

- Keep expensive native peer re-probes off the interactive scheduler path while preserving the
  existing fail-closed mismatch semantics.
- On a runtime native peer re-probe failure, latch native P2P off and continue through validated
  host-bounce staging; panic only when bounce staging cannot be armed.
- Cover rung scheduling and worker continuity, expose degraded transport in metrics, update flag
  documentation, and retain raw measurement and validation logs in this directory.

## Checklist

- [x] Confirm dedicated worktree and branch at `09900dcaa`.
- [x] Map the peer-probe scheduler, worker fallback, transport state, metrics, and test seams.
- [x] Add focused failing tests.
- [x] Implement the scheduling and fallback changes.
- [x] Update `docs/FLAGS.md`.
- [x] Run focused tests, `cargo test`, and capped 5090 `local-ci`.
- [x] Measure an interleaved x5 warm-hit TTFT A/B on the steered cloudbox PP-2 pair; retain the
      synthetic owner-thread stall receipt as supporting scheduler evidence.
- [x] Write `RESULTS.md` and retain the complete raw evidence tree.
- [x] Review the exact diff and commit the complete lane.

## Constraints

- No origin push, merge, tag, board update, formatting sweep, rustup, nsys artifact, or
  `--no-verify`.
- Before any local 5090 GPU run, capture concurrent compute applications with `nvidia-smi` and
  keep the load capped. Record the locked 210--1200 MHz thermal regime in the receipt.
- Remote PP-2 work takes `flock /tmp/memra-gpu.lock`; do not disturb an incumbent owner.

## Design invariants

- Track one copy-count deadline per width so an overdue idle-only rung cannot block later cheap
  rungs. The maximum-production rung is idle-only before its first measurement; any other rung
  becomes idle-only after a measured wall cost above the fixed 5 ms owner-thread budget.
- A deferred rung does not consume its deadline or increment the completed-probe counter. A late
  idle run advances to the first future cycle instead of replaying a burst of missed probes.
- Runtime degradation is one-way. Native transport is failure-latched first; pinned staging is
  allocated and exercised through an actual host-bounce copy/readback; only then is the live
  transport atomically published as host bounce. Failure to arm or validate staging remains the
  worker-panic case.
