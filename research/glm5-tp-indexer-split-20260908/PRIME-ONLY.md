# Prime-only indexer split, 2026-09-09

Baseline: `b4a719ed48cac74c032e2cb649607c1833a83152`, memra draft #395.
Prior pair receipt: [darklanes #527](https://github.com/avifenesh/darklanes/pull/527),
[RESULTS.md](RESULTS.md). This follow-up changes dispatch, not CUDA arithmetic.

`MEMRA_GLM5_TP_INDEXER_SPLIT_PRIME` is default OFF, decide-by 2026-09-22.
Only grouped-prime chunks (`t>1`) can enter the pool split. The shared admission helper
returns false for `t<=1` before reading the flag. Both symmetric decode split dispatches
and their scalar PRE workspace code are deleted. The old flag is no longer read; no
future decode door is retained. CHECK remains a prime-only diagnostic.

The previous pair's prime IDs were identical. Its decode arm was negative at 128k and flat
at 1M: merge cost was about 128 ms/GPU over 159 steps plus 36-40 ms exchange. Shared
candidate/exchange/merge kernels remain because grouped prime uses them.

## Validation

All cargo commands ran remotely on one non-serving B200, using a dedicated source and
target directory, `nohup`, `MEMRA_CUDA_ARCH=100a`, and `flock /tmp/memra-gpu.lock`.
The local rig ran no cargo, tests, benchmark or smoke server.

- Formatting: PASS, `cargo fmt --all -- --check`.
- Release build: PASS in 3m47s, `cargo build --release -p memra-engine --bin glm5-tp2-box-probe -j 16`.
- Clippy: PASS in 27s, `cargo clippy --release --all-targets -j 16 -- -D warnings`.
- CPU target: PASS, 4 passed / 0 failed / 3 GPU tests ignored, `cargo test --release -p memra-engine --test glm5_tp_indexer_split -j 16`.
- Engine library: PASS, 463 passed / 0 failed / 19 ignored, `cargo test --release -p memra-engine --lib -j 16`.
- Single-device GPU merge: PASS, 1 passed / 0 failed.
- GPU range bit-identity: PASS at RP=1 and RP=2, 1 passed / 0 failed in each process.

Raw logs, the release binary SHA-256, changed Rust source hashes, job timestamps and exit 0
are in [receipts/prime-only-20260909](receipts/prime-only-20260909/). The local changed Rust
files match the remotely formatted, built and tested source hashes.

The first clippy attempt failed because the isolated source copy omitted tokenizer test
fixtures under `research/reasoning-schema-20260823`. The missing tracked fixtures were
copied from the same lane before rerunning; no source patch was needed for that failure.

The CPU regression runs the production flag helper in child processes with unset/OFF/ON
prime flag values and the retired flag ON, testing decode before and after the prime latch.
The single-device merge gate stages exact rank-major candidate words and compares every
index against both the replicated GPU selector and the CPU oracle, including ties,
nonfinite values, causal tails, odd pools, ragged k, and 1M pool sizes. Transport is not
simulated as a pass: the existing two-device exchange test remains a separate pending gate
on this follow-up binary, since only one B200 was available.

Required post-deploy pair cell: 1M prime OFF/ON, 1M decode OFF/ON interleaved x3, and
byte-identical IDs in every arm. Confirm retained prime improvement and no decode regression.
Vendor-default sampled requests and eight-turn cache-on continuation remain serving gates.
No full-model performance or default-promotion claim is made for this follow-up binary.

The pre-commit fmt hook uses the matching remote receipt for this commit. Push uses
`MEMRA_SKIP_PERF_CI=1`; hosted CI remains required. The existing lane worktree stays open
with the draft PR; this task's remote build directory and local scratch are removed after
receipt collection.
