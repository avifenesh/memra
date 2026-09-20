# H2D copies plumbing status (2026-09-20)

**Native build passed; collector matrix pending an idle GPU window.** No G2
performance result, median, runtime-default decision, NVMe proof or serving
qualification follows from this milestone.

- Engine/bin source: memra `e8b4e642191b283231116f7a16ff45bb6f4713c5`.
- Native release build: `cargo build --release -j 16 -p memra-engine --bin h2d_probe`,
  CUDA 13.1, sm_120a. Raw build output and source/binary hashes:
  `h2d-copies/native-build/build.log` and `build-identity.log`.
- Hardware condition: rented-development RTX 5090, enforced cap 400 W,
  maximum 600 W. Restricted-power, not a full-power baseline.
- Fresh preflight refused **before invoking the collector** because a peer's
  CUDA process was active. No GPU matrix cell has been consumed by this attempt.
  Sanitized evidence: `h2d-copies/native-build/preflight-refused.json`; private
  original retains the process id/worktree path outside git.
- CPU checks: three Rust tests, 12 dry-run invocations including `copies=100000`,
  Linux-target engine/bin typecheck and rustfmt passed. Initial DOCS_RS check
  failed on an absent compile-time archive hash; the explicit docs-only sentinel
  retry passed. Both logs are retained under `h2d-copies/cpu/`.
- Timeout containment: CPU-only nested-child test passed; the detached-child red
  control reproduced escape from the outer timeout. F's plumbing worker keeps
  probe children in the collector process group. No GPU/rig lock used by this test.
- Exact token compatibility: D `7f7bf547` consumes bare JSON sample and summary
  rows. Only the outer worker emits one `RESULT ` prefix. Per-visit prefixing
  would break both D's parser and the collector's singleton RESULT contract.

The requested single <=300-second collector cell remains pending; full G2
calibration, balanced N>=5, duration and telemetry gates remain unrun. See
`CELLS-ENVELOPE.md` for the final CLI and D executor ownership.
