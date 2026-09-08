# CPU checkpoint, 2026-09-08 UTC

Code commit `f604518caf15db72a80d52b10941f9fd243b2972`, after shared seam
`534040262e86d3009ba298ebdfbf79acb48b93d1`. Source hashes were read back from the
remote checkout and matched the local files before commit. The development
binaries were then rebuilt from the exact code commit. No GPU process launched.

All execution used the authorized non-production box CPU, nice 19, two jobs,
`MEMRA_CUDA_ARCH=120a`, `CARGO_TARGET_DIR=/root/target-prefill`. Toolchain:
Rust/Cargo 1.97.1, CUDA compiler 13.0. Raw output is in `cpu/`; input and binary
hashes are in `cpu-adapters.json`.

| Check | Result |
|---|---|
| `cargo fmt --all -- --check` | PASS |
| `cargo clippy --release --all-targets -- -D warnings` | PASS |
| Engine shared walker tests | 3 passed |
| MTP frozen schedule tests | 2 passed |
| DFlash tests, including production carry oracle | 56 passed |
| Full server library suite | 638 passed, 0 failed, 1 existing ignored fixture |
| Release `memra-server` and `run-spec` builds | PASS |
| Mixed-load driver's Python compilation | PASS |

The ignored `prefill_proxy_fixture` is the existing manual loopback fixture for
external proxy qualification. The new failure test covers both a failed chunk
and failed final ingestion remaining ineligible for parking.

Remote binaries for handoff:

- `/root/target-prefill/release/memra-server`, SHA-256
  `f28dc7fec215c7335227600e01168720f4f2f7393910deda2928281341561f4a`.
- `/root/target-prefill/release/run-spec`, SHA-256
  `23b74c97675adc3ed74fd1c878d8ad136b220118924eeafa56a78ccbaf5e8d95`.

Review fixes included in this source: deferred MTP restoration queues its suffix;
pending primes cannot demote, publish or park; finalization failures remain
unparkable; the prepared MTP graph context, init feed and grammar boundary are
consumed once; request identity is checked before consuming prepared state.

`mixed_load.py` preserves the archived fixed-arrival design with explicit binary
hash and profile inputs. Prefill mode schedules one long request at time zero and
20 small requests at 5..100 seconds, so small offered rate is 0.20 req/s over
`(0,100]`. It records total offered rate separately. The client ceiling is 16,
drain is explicit, sampling fields are omitted, and each boot records a nonce,
PID and start ticks. It holds `/tmp/memra-gpu.lock` and requires an empty
`nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` before
launch. The pinned Qwen long request can be supplied through `--long-request`.

GPU c1/c2 greedy bytes, capture equivalence, 1024/4096 chunk wall and interleaved
sampled TTFT/long-request penalty remain pending explicit GPU handoff. The door
stays OFF and the PR stays draft. No merge, release or deployment at this checkpoint.
