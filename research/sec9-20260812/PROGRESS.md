# cx-sec9 progress

Branch: `lane/cx-sec9`
Base: `201508b93b1686113eccdb6538b4b155e83285e8`
Date: 2026-08-12

## Goal

Rotate the owner-thread runtime peer re-probe through a deterministic sampled width ladder,
including the 16-token production rung and the maximum-chunk rung, while preserving sec8's
fail-closed panic/host-bounce behavior.

## Gates

- [x] Inspect sec8 probe, cadence, tests, and kernel-check coverage.
- [x] Add deterministic rotation and cadence tests; prove every selected width is reached.
- [x] Run engine and server unit tests.
- [x] Run the applicable kernel-check cell or battery.
- [x] Capture per-rung probe cost and 5090 `run-gen` argmax MATCH evidence.
- [x] Review the exact diff and commit the complete lane receipt.

## Implementation

- Each due owner-thread runtime probe now selects the next boot-ladder width in the exact
  deterministic order `1, 8, 16, 4096` tokens and wraps after the fourth rung. Probe bytes come
  from the authoritative initialized boundary-row size instead of the old fixed 16 KiB payload.
- The cadence remains one probe every 8,192 successful native boundary copies. A full rotation
  takes 32,768 copies, so the measured-expensive maximum-chunk rung runs only every fourth tick.
- The existing bidirectional probe, CUDA-context restoration, process failure latch, and worker
  panic/host-bounce recovery instructions remain the failure path. Runtime logs now identify the
  rung, token width, byte width, and zero-based probe index.
- `pp-transport-smoke --runtime-probe-cycle` is an opt-in, model-free receipt mode. It drives one
  exact cadence cycle and services probes between completed boundary copies on the CUDA owner
  thread; normal smoke behavior is unchanged.
- The existing sec8 `peer-probe-off-refusal` kernel-check cell pins only the probe-off startup
  policy, not the old fixed runtime payload. No manifest or kernel-check cell update was needed;
  rotation order and cadence are pinned by unit tests and exercised on the two-GPU transport.

## Sbox probe-cost receipt

Exact modified-source SHA-256 values in `raw/sbox/SHA256SUMS` match this worktree. Measurement ran
under an exclusive GPU lock on the provisioned Sbox pair: 2 x RTX PRO 6000 Blackwell Server
Edition, PIX topology, CUDA 13.2/sm_120a release build. The rig began idle in P8 and ended active
in P0. These are single sequential observations (`N=1` per rung), not throughput medians. Each
time is the complete existing integrity pass: pattern preparation, allocation, two directional
peer copies, device-to-host readback, and byte comparison.

| Rung | Tokens | Bytes | Boundary copies | Complete probe cost |
| ---: | ---: | ---: | ---: | ---: |
| 1/4 | 1 | 16 KiB | 8,192 | 0.724 ms |
| 2/4 | 8 | 128 KiB | 16,384 | 1.161 ms |
| 3/4 | 16 | 256 KiB | 24,576 | 1.909 ms |
| 4/4 | 4,096 | 64 MiB | 32,768 | 431.188 ms |

The cycle completed with `serviced=4/4`, zero runtime failures, and
`pp-transport-smoke PASS`; total process wall time, including boot probes and cadence driving,
was 1.99 seconds. The 431.188 ms maximum-rung cost is why that rung stays at the lower one-in-four
frequency. Raw build, rig, source, and run evidence is under `raw/sbox/`.

## Receipt

- Worktree began clean at the base above.
- Focused final-tree rotation/cadence tests: 2 passed, 0 failed
  (`raw/runtime-rotation-tests.log`).
- Final-tree `cargo test -p memra-engine -p memra-server`: engine library 78 passed / 1
  CUDA-specific ignored; server 194 passed; no failures (`raw/cargo-test-engine-server.log`).
- Local RTX 5090 Laptop GPU full required manifests: `ALL GREEN (106 cells, 1 skipped)`;
  `peer-probe-off-refusal` passed unchanged (`raw/kernel-check.log`). The one skip is the existing
  optional sigrouter replay cell, which requires an external capture.
- Local 5090 `run-gen` generated 64 tokens with prefill/decode and
  batched-prime/tokenwise argmax `MATCH` (`raw/run-gen.log`).
- Release binaries were rebuilt from the final source before the last kernel-check and run-gen
  pass (`raw/release-build.log`); local device snapshots bracket that locked pass.
- `raw/SHA256SUMS` verifies all 21 raw receipt files, including the nested Sbox source and binary
  manifest.
- Reviewed scope contains the runtime rotation, its tests, the opt-in two-GPU receipt driver, and
  this evidence tree only. No merge, tag, push, board change, formatting sweep, or hook bypass.
