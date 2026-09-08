# Startup canary retries (#340)

The candidate recovered after two startup hangs and returned HTTP 200; the old binary stayed HTTP 503 after probes resumed.

Startup uses six consecutive probes with the existing per-probe deadline, with no extra sleeps and no new flag. The default deadline is 10 seconds, giving about 60 seconds before all hangs latch a fault. An answer ends startup immediately: success retains rich fields, a nonzero exit selects the minimal query, and a spawn error leaves Xid-only monitoring. The steady-state loop is byte-identical to the base and still latches one hang.

## Box receipt

Single fault-injection run per binary on an RTX 5090 with `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, context 2048. The PATH-local Python `nvidia-smi` wrapper counts calls in a file, sleeps 15 seconds on each of the first two calls, then executes `/usr/bin/nvidia-smi`. Sleeping happens in the wrapper process itself, so the watcher's kill leaves no sleeping descendant. The probe deadline stays at its 10-second default. Both arms use `MEMRA_GPU_WATCH_S=1` so subsequent real probes arrive quickly. No sampling settings are overridden.

- Old binary at `2f12d82fb16915bc58f645aba070d919092863f3`: at 24.03 seconds, after four probe calls, `/health` was HTTP 503 with `status: unhealthy` and the first startup timeout reason still latched.
- Candidate: `[gpu-watch] startup canary recovered after 2 hangs`; at 22.03 seconds, after four probe calls, `/health` was HTTP 200 with `status: ok`.
- Raw observations, probe arguments/times, full server logs, and binary hashes: [old.json](old.json), [candidate.json](candidate.json). JSON escapes preserve the original logs' punctuation.
- Source provenance: [source.json](source.json). The candidate was built from the rsynced worktree on base `f80553700` plus this patch. The remote `.git` was excluded from rsync, and both binaries report `git: unknown`; their source-tree build IDs are `memra-0.133.0-1d03d97403d8` (old) and `memra-0.133.0-4d4bfd4a283a` (candidate). Use the recorded source and binary hashes for this receipt.

The harness held `/tmp/memra-gpu.lock`, chose an unused loopback port for each arm, and terminated only its own server process group in a `finally` block. Both servers drained and exited. This is a startup health regression receipt, not model qualification or performance evidence.

## Verification

The server depends on memra-engine and required a CUDA rebuild. Compilation, the CPU-only health suite, and clippy ran on the authorized RTX 5090 box with `CARGO_TARGET_DIR=/root/target`, `MEMRA_CUDA_ARCH=120a`, `MEMRA_GPU_LOCK=/tmp/memra-gpu.lock`, `nice -n 15`, and `CARGO_BUILD_JOBS=8`. Release profile reused the box's existing build cache.

- `cargo test --release -p memra-server health`: `test result: ok. 18 passed; 0 failed; 0 ignored; 0 measured; 612 filtered out; finished in 0.50s`.
- New tests: `startup_canary_hang_then_ok_recovers`, `startup_canary_hang_then_exit_degrades`, `startup_canary_all_hangs_latch_with_count_and_window`, `startup_canary_spawn_stops_immediately`, `startup_canary_ok_on_first_or_last_probe_keeps_rich`. All passed without spawning `nvidia-smi`.
- `cargo build --release -p memra-server`: passed.
- `cargo clippy --release -p memra-server --all-targets -- -D warnings`: passed. Raw build, test, and clippy output: [verification.json](verification.json).
- `cargo fmt --check`: passed on the rig under the same CPU cap.
- `git diff --check`: passed; added lines contain no em dashes.
