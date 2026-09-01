# cx-probeready results

Date: 2026-08-12

Base: `584ed0af05e5f8a29b318a410a8e76ed8a08292f`

Verdict: **PASS**

## Result

Deferred runtime peer-integrity probes now create their own safe recovery window without forcing
a probe through live speculative UVA state.

- `/readyz` always includes `peer_probe_integrity`: `ok`, `deferred_<n>`, or `degraded`.
  The field is advisory. A peer-integrity degradation does not change an otherwise-ready HTTP 200
  because the worker can still serve plain sessions safely.
- `MEMRA_PEER_PROBE_DEFERRAL_BOUND` controls the consecutive deferred-interval bound. The default
  remains the probeobs value `4` (one 32,768-boundary-copy width rotation), the minimum is `1`, and
  invalid values fall back to `4`. The value is resolved once at worker startup.
- At the bound, new speculative candidates take the existing plain session path. The same
  per-admission decision drives memory estimation, transient reserve selection, and actual session
  construction, so accounting cannot reserve for one path and construct the other.
- Existing speculative sessions are not demoted or interrupted, and plain candidates remain plain.
  Once the live speculative sessions finish, the unchanged scheduler can safely complete the due
  re-probe; that completion clears the local streak, the engine's degraded metric, the readyz
  advisory, and the speculative-admission hold.
- A single-device worker receives `RuntimePeerProbeStatus::NotRun`, never publishes a deferral, and
  retains normal speculative-admission behavior.

The no-force-probe law is unchanged: a live speculative session still makes the runnable probe
return `Deferred` before a deadline or completed-probe counter is consumed. This lane changes
admission and observability only; it does not add a CUDA probe path or alter transport.

## Test receipts

- Focused worker coverage: 5/5 runtime re-probe tests passed, including configurable-bound parsing,
  bound-triggered soft refusal, plain-path preservation, and completed-probe recovery
  ([`raw/focused-runtime-reprobe-tests.log`](raw/focused-runtime-reprobe-tests.log)).
- Focused readyz coverage: 1/1 passed. `ok`, `deferred_2`, and `degraded` were present; degraded
  remained HTTP 200 while worker readiness was healthy, and the field remained present during an
  unrelated 503 ([`raw/focused-readyz-test.log`](raw/focused-readyz-test.log)).
- Focused single-device coverage: 1/1 passed; `NotRun` left peer health `ok` and did not close spec
  admission ([`raw/focused-single-device-test.log`](raw/focused-single-device-test.log)).
- Full `cargo test`: exit 0, **439 passed / 0 failed / 2 ignored**. The ignored tests declare their
  CUDA-only requirements in the log ([`raw/cargo-test.log`](raw/cargo-test.log)).
- Locked local RTX 5090 battery: `flock -w 7200 /tmp/gpu5090.lock tools/local-ci.sh`, exit 0
  ([`raw/local-ci.log`](raw/local-ci.log)):
  - flag audit found no new drift;
  - `kernel-check: ALL GREEN (106 cells, 1 skipped)` and prime gate 8/8 green;
  - Qwen 35B `run-spec` K=1..8 passed 8/8; 31B and 12B argmax/depth gates passed;
  - NVFP4 and Q8_0 decode-batch config and strict gates were all green;
  - graph warmup stress passed 10 cycles x 4 arms plus overlap, and caught its injected corruption;
  - `correctness stage: GREEN`, serve smoke 0 failed, c=64 stress completed 64/64 with a clean
    worker, and the acceptance cell passed (42 rounds, 126 drafted, 85 accepted, 128-token text
    SHA-identical).
- The GPU had no compute applications before or after the locked run. It was P8, 0% utilization,
  and 58 C before; P8, 0% utilization, and 62 C after
  ([pre](raw/pre-local-ci-nvidia-smi.log), [post](raw/post-local-ci-nvidia-smi.log)). These are
  correctness receipts, not performance evidence.

The current development rig is single-device, so this lane did not claim a live PP-2 deferral
reproduction. The transition/admission behavior is pinned by GPU-free unit tests, while the final
source passed the repository's real local GPU correctness battery.

Raw receipt hashes are frozen in [`RAW-MANIFEST.sha256`](RAW-MANIFEST.sha256).

## Scope discipline

No merge, tag, origin push, board change, formatting sweep, Rust toolchain change, nsys artifact,
other-worktree operation, or hook bypass was used.
