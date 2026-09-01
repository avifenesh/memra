# cx-sec8 progress — peer-probe hardening pair

Date: 2026-08-12
Branch: `lane/cx-sec8`
Baseline: `main @ 8b2ba8c88+`

## Scope

- Fail closed when `MEMRA_PEER_PROBE=0` would otherwise permit sharded cross-device native peer transport.
- Preserve probe-off diagnostics only for single-device/non-sharded placement or explicit `MEMRA_PP_HOST_BOUNCE=1`.
- Add low-frequency owner-thread runtime peer re-probing with loud failure or host-bounce fallback, plus metrics.
- Wire the refusal policy into the kernel-check manifest.
- Keep model bytes and golden-output expectations unchanged.

## Required gates

- [x] Refusal-matrix unit tests (`probe off` x `sharded` x `host bounce`).
- [x] `cargo test -p memra-engine -p memra-server`.
- [x] Kernel-check manifest cell for the refusal policy.
- [x] Local RTX 5090 `run-gen` smoke.
- [x] Local RTX 5090 `run-spec` K=1..8 smoke.
- [x] Review intended diff and commit the lane; no merge, tag, push, board update, or formatting sweep.

## Implemented

- Startup applies a pure fail-closed policy before native transport can serve a sharded
  cross-device placement. The refusal quotes both `MEMRA_PEER_PROBE=0` and
  `MEMRA_PP_HOST_BOUNCE!=1`; the full 2 x 2 x 2 policy matrix is pinned by a unit test.
- Probe-off plus explicit host bounce remains a diagnostics door, but emits a
  `SECURITY RED` log and increments operator-only `peer_probe_bypassed` telemetry.
- Successful native boundary TX copies increment a relaxed process counter. At the first
  scheduler boundary after every 8,192 copies, the CUDA owner thread repeats the existing
  deterministic bidirectional 16 KiB peer-integrity pass. No background thread touches CUDA.
- Runtime probe error or corruption increments `peer_probe_runtime_failures`, latches native
  P2P off for the process, and fails the worker loudly. The latch makes the normal worker
  respawn refuse PP initialization, so serving cannot silently resume on the suspect path.
- Operator metrics expose bypass count, native boundary-copy count, runtime re-probe count,
  and runtime failure count. Completion and tenant scopes cannot observe them.
- `peer-probe-off-refusal` is required by `tools/kernel-check-step35.cells`.

## Evidence log

- 2026-08-12: steering and queued item B read; clean isolated branch confirmed before edits.
- 2026-08-12: focused refusal-matrix and runtime-cadence tests each passed one exact test;
  receipts: `raw/refusal-matrix-test.log`, `raw/runtime-cadence-test.log`.
- 2026-08-12: final-tree `cargo test -p memra-engine -p memra-server` exited 0; engine library
  76 passed / 1 CUDA-specific ignored, server 192 passed; receipt:
  `raw/cargo-test-engine-server.log`.
- 2026-08-12: local RTX 5090 Laptop GPU kernel-check completed `ALL GREEN (106 cells,
  1 skipped)` and printed `peer-probe-off-refusal fail-closed policy OK`; receipt:
  `raw/kernel-check.log`.
- 2026-08-12: local 5090 `run-gen` generated 64 tokens with prefill/decode and
  batched-prime/tokenwise argmax `MATCH`; receipt: `raw/run-gen.log`.
- 2026-08-12: local 5090 `run-spec` generated 32 tokens and passed self-consistency for every
  K=1..8; receipt: `raw/run-spec.log`.
- 2026-08-12: no multi-GPU runtime probe was run in this lane: the permitted local development
  rig exposes one GPU, and box1 was explicitly excluded. This commit therefore carries the
  policy/cadence tests and owner-thread wiring evidence; the required PRO-pair pre-release
  battery remains the release gate.
