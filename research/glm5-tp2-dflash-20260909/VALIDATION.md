# Remote validation

Source commit: `436545c94` on base
`dcfeab7c738912a150ebbfea277112724bb99de4`.
`receipts/source-manifest.json` pins every changed Rust file; the independent
remote readback is `receipts/source-match.log`. No cargo, test, benchmark, smoke
server or CI command ran on the local rig.

The owner's single-card tune host ran the commands below with
`MEMRA_CUDA_ARCH=100a`, `MEMRA_GPU_LOCK=/tmp/memra-gpu.lock`, and the exclusive
`flock /tmp/memra-gpu.lock`. Hardware inventory: one B200, driver 595.91.07,
CUDA 13.1, rustc 1.97.1. This is compile/CPU evidence, not a two-rank GPU gate.

| Command | Result |
| --- | --- |
| `cargo clippy -p memra-engine -p memra-server --lib -- -D warnings` | Passed |
| `cargo test -p memra-engine --lib glm5_tp -- --test-threads=1` | 8 passed, 0 failed, 1 ignored |
| `cargo test -p memra-server --lib glm5_ -- --test-threads=1` | 28 passed, 0 failed |
| `cargo build --release -p memra-server --bin memra-server` | Passed |
| `cargo test --release -p memra-engine --lib glm5_tp_spec::tests -- --test-threads=1` | 4 passed, 0 failed |
| `cargo test --release -p memra-engine --lib --no-run` | Ignored pair harness compiled |
| `cargo fmt --all -- --check` | Passed |
| Private cell `bash -n` and `python3 -B -m unittest` | Syntax passed; 5 parser/config tests passed |
| `tools/check-flags.sh` | 842 runtime reads, no uncovered names |

The 8 engine tests include the four new admission/bookkeeping tests. The fake
executor exercises 84 boundary/width/keep combinations across distinct root and
peer states, then checks continuation after rejection. The ignored test is the
actual two-device harness, not a skipped CPU test.

The final combined validation log is `receipts/final-validation-r4.log`. Its
last command initially stopped because `rg` was missing from the container,
after all builds and tests had passed. Ripgrep was installed, and the successful
artifact freeze plus flag census is `receipts/artifact-freeze.log`. Earlier
compile and validation logs remain as `stage2-check.log` and
`stage3-validation-r2.log.gz`; they do not supersede the final source manifest.

## Frozen artifacts

Both files were copied out of the shared Cargo target while holding the lock.
They remain in the tune checkout's lane-owned
`research/glm5-tp2-dflash-20260909/artifacts/` for the scheduled pair cell.

| File | SHA256 |
| --- | --- |
| `memra-server` | `d52fe67cb6a41bd1d90a2e27d46910c3f81ef88afd4cbad38ee19ce07a379e45` |
| `memra-engine-tests` | `ceaec81208a94d3560768d34b5bd0ed0b297ed4edd52c08effe6c108a1ba12de` |

## Pending pair evidence

O1 and O3 could not run on the single-card tune host. O2 also requires the pair.
No real-artifact output, accepted-token sequence, per-row band, or bit equality
is reported as measured here. No speed, TTFT, 128k or 1M improvement is claimed.
The door remains OFF and verify graph capture remains deferred.

The executable cell, pinned input schema and exact recipe templates are in
private Darklanes, `research/glm5-tp2-dflash-20260909/`. Its entry point is
`run-pair-cell.sh`; the parent source is
`dd5b04cd1188fb2aada272c32a6465e569d63fff`. It gates performance on O1-O3,
then schedules c1 TP plain/TP spec/PP spec interleaved x3 at 32k and 128k.

The owner explicitly requested a draft PR before that scheduled pair cell.
Push uses `MEMRA_SKIP_PERF_CI=1`. Pre-commit formatting and pre-push checks run
remotely because the local rig must not run gates. Neither draft status nor
compile/CPU success is permission to enable the door in serving.
