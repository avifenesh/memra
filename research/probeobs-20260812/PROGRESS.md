# cx-probeobs progress

## Objective

Bound and expose peer-probe deferral during continuous speculative serving
without weakening the CUDA-owner-thread safety contract that caused the
deferral.

## Success criteria

- Export `peer_probe_deferred_total` beside the existing peer-probe metrics.
- Choose and document a defensible consecutive-deferral bound after tracing the
  live-spec/UVA safety constraint.
- At the bound, either run only demonstrably safe cheap rungs or surface a loud
  degraded-integrity state while leaving expensive and unsafe work idle-only.
- Test the deferral counter, the bounded policy, and unchanged non-spec serving.
- Update the peer-probe entry in `docs/FLAGS.md`.
- Pass `cargo test -p memra-server -p memra-engine` and the local-ci correctness
  stage, retaining raw receipts under `research/probeobs-20260812/`.
- Commit only the intended lane changes; do not push, tag, merge, update boards,
  format broadly, install Rust, create nsys artifacts, or bypass hooks.

## Status

- [x] Lane inbox, repository instructions, branch, and worktree state read.
- [x] Peer-probe deferral and speculative-session safety contract traced.
- [x] Bounded policy selected and implemented.
- [x] Metrics, tests, and operator documentation completed.
- [x] Required test gates captured and summarized in `RESULTS.md`.
- [x] Final diff audited and committed.

## Initial state and assumptions

- Branch: `lane/cx-probeobs`, based on `main` at `d2fba6200`.
- The pre-existing `PROGRESS.md` described the already-merged `cx-kneeraise`
  lane; replacing it is this lane's required first file edit.
- Safety is unresolved at kickoff. No forced probe will be implemented until
  the shared UVA token/position reads and CUDA-owner sequencing are traced.

## Safety decision

- Choose the permitted alarm-only policy. A runtime mismatch first latches
  native P2P off and validates host bounce, but host bounce covers boundary
  activations only. A live speculative session still has primary-device
  token/position and verification buffers dereferenced from another stage via
  UVA, so forcing even a cheap probe could discover a mismatch and revoke the
  access those in-flight sessions require.
- Bound consecutive cheap-rung deferral at four 8,192-boundary-copy intervals.
  Four intervals are one complete 32,768-copy width rotation, so the operator
  alarm cannot remain silent for more than one normal integrity cycle after a
  runnable rung first becomes blocked.
- Count at most once per boundary-copy interval, not once per scheduler poll.
  At the fourth consecutive interval publish a sticky-until-recovery
  `peer_probe_integrity_degraded` state and a loud log. Clear it only when a
  peer probe safely completes or validated host bounce takes over.
