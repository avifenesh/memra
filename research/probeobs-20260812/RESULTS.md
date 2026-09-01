# cx-probeobs results

Date: 2026-08-12

Base: `d2fba6200`

Verdict: **PASS**

## Result

Peer-probe starvation under continuous speculative serving is now observable
and bounded by an operator alarm without weakening the live-session safety
contract.

- A due runnable probe returns `Deferred` before consuming its copy-count
  deadline or incrementing the completed-probe counter when a live speculative
  session blocks safe failover.
- `peer_probe_deferred_total` counts deferred copy-count intervals rather than
  scheduler polls. Repeated polls inside one 8,192-copy interval do not inflate
  it.
- Four consecutive deferred intervals set
  `peer_probe_integrity_degraded=true` and emit `SECURITY RED`. Four is one
  complete 32,768-copy rotation of the 1/8/16/4096-token probe ladder.
- The degraded-integrity state clears only after a probe completes at a safe
  boundary or validated host bounce takes over. Ordinary non-spec serving keeps
  the existing allowed execution path, and expensive rungs remain idle-only.
- Both fields are operator-only beside the existing peer-probe counters.

## Safety decision

This lane uses the inbox-authorized alarm-only policy rather than forcing a
cheap rung through a live spec session. The current CUDA Runtime API says peer
allocations remain directly accessible until peer access is explicitly
disabled, and the CUDA Programming Guide describes peer access as allowing a
kernel on one device to dereference another device's pointer:

- <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html>
- <https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#peer-to-peer-memory-access>

Memra's runtime mismatch path deliberately latches native peer access off
before it validates and publishes host bounce. Host bounce covers boundary
activations, but a live speculative session still exposes token/position and
verify state through cross-device UVA. Forcing even a cheap probe could
therefore discover a mismatch and revoke access required by the in-flight
session. The alarm keeps the transport unchanged until a safe boundary exists.

The documented operational response is to drain speculative sessions or
restart with `MEMRA_SERVE_SPEC=0`, then require
`peer_probe_runtime_reprobes` to increase and
`peer_probe_integrity_degraded` to clear before restoring speculative traffic.
Disabling the probe is explicitly not remediation.

## Test receipts

- Focused runtime-reprobe tests: engine 4 passed; server 3 passed, including
  interval coalescing, the four-interval alarm, counter publication, recovery,
  mismatch continuity, and the plain-serving permission
  (`raw/focused-runtime-reprobe-tests.log`).
- Operator metrics surface: 1 passed; both new fields are present only for the
  operator scope (`raw/focused-peer-probe-metrics-test.log`).
- `cargo test -p memra-server -p memra-engine`: exit 0. The engine library ran
  82 passed / 1 CUDA-only ignored; the server ran 213 passed; auxiliary binary,
  fixture, and doc tests were green (`raw/cargo-test-engine-server.log`).
- Locked local RTX 5090 correctness command:
  `flock -w 7200 /tmp/gpu5090.lock tools/local-ci.sh`, exit 0
  (`raw/local-ci.log`). Highlights:
  - release build completed; flag check found no new drift;
  - `kernel-check: ALL GREEN (106 cells, 1 skipped)`; the skip is the optional
    sigrouter served-replay capture;
  - prime gate 8/8 matches, Qwen 35B `run-spec` K=1..8 PASS, 31B and 12B
    `run-gen` argmax MATCH, both depth VERIFY-GATE cells PASS, and 31B stream
    agreement 64/64;
  - NVFP4 and Q8_0 decode-batch config/strict gates ALL GREEN;
  - graph warmup stress 10 cycles x 4 arms plus overlap was bit-identical, and
    the injected-corruption canary was caught;
  - serve smoke reported 0 failed, c=64 stress completed 64/64 with a clean
    worker, and the acceptance smoke passed (42 rounds, 126 drafted, 85
    accepted, rate 0.6746, 128-token text SHA identical).
- The GPU was compute-idle before and after the locked correctness run. It was
  P8 at 58 C before and P8 at 63 C after; only display processes were present
  (`raw/pre-local-ci-nvidia-smi.log`, `raw/post-local-ci-nvidia-smi.log`). These
  are correctness receipts, not performance evidence.

No merge, tag, origin push, board change, formatting sweep, Rust toolchain
change, nsys artifact, other-worktree operation, or hook bypass was used.
