# GB10 coverage checks (2026-11-06)

Verdict: both checks pass on one DGX Spark (GB10, compute capability 12.1). A streaming request whose prefill runs 289 s gets its headers at 60 s and all 16 tokens, with no HTTP 408. A clean build with `MEMRA_CUDA_ARCH` unset auto-detects `121a`, and memra-probe built the same way runs its kernels on the device.

Driver: `gb10-coverage.sh`, run on the unit under its sparklock lease after the cold-drafter capture runs; log `chain.log`.

## Check 1: streaming prefill past 90 s

Binary: memra-server built from 3bbdf42ff9 (the code of main 5d70118b99; `[server] build ... git: 3bbdf42ff992`), sha256 `24fc6045841f`. Serving arm `gguf-mtpmask`: the owner GGUF Qwen3.8-27B (NVFP4 + Q5_K trunk) with its MTP head, `MEMRA_CTX=262144`; the full boot environment is in `check1-guard.log`. Client: `longprefill_client.py`, one streaming `/v1/completions` request, temperature 0, `max_tokens` 16, on the unit.

| prompt tokens | status | headers | first token | total | data chunks | completion tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 150,222 | 200 | 60.0 s | 289.04 s | 290.23 s | 17 | 16 |

Raw: `check1-result.json`, `check1-client.out`.

Server lines (`check1-server.log`, full log):

- `[deadline] unified memory (48 SMs, served ctx 262144): streaming TTFT default/max 997311 ms (derived), SSE prefill commit 60000 ms`. The headers went out when the 60 s SSE prefill commit fired, while prefill continued.
- `[admission] request cost: model="q38" ctx=150302 path=spec = 31552 B/token x ctx + 4956MB prefill-workspace + 0MB fixed = 9698MB`.
- 74 `[prime-yield]` lines. Chunks 1-37 take 4.9 s to 10.6 s each, growing with depth; chunks 38-74 take about 72 ms each.
- `[prefix-cache] insert (spec-boundary): 150176 tokens, 4895.3MB`, then `[spec-acc] ctx=150223 burst=9/15`.
- No 408, timeout or error line after the request was admitted. The server drained with 0 requests in flight.

## Check 6: clean build with `MEMRA_CUDA_ARCH` unset

Tree c28fb5ee0e: the cold-drafter capture fix, one commit on main 5d70118b99; it touches only `crates/memra-server/src/worker.rs` and gate tooling, none of the build scripts. The build used a fresh, empty `CARGO_TARGET_DIR` (deleted afterwards) and `env -u MEMRA_CUDA_ARCH`, with CUDA from `/usr/local/cuda`.

- `cargo build --release --locked -p memra-engine --lib`: rc 0, 3 m 34 s (`check6-engine-build.log`). The build.rs warning line:

  `warning: memra-engine@0.138.0: MEMRA_CUDA_ARCH auto-detected 121a (compute_cap 12.1); set MEMRA_CUDA_ARCH to override`

- `cargo build --release --locked -p memra-probe --bin memra-probe`: rc 0 (`check6-probe-build.log`). The probe's build.rs also reads `MEMRA_CUDA_ARCH`, and with it unset detects the sm_121a twin from `compute_cap`.
- `memra-probe` run: rc 0 (`check6-probe-run.log`): `device: NVIDIA GB10`, `compute capability: 12.1  SMs: 48`, `vec_add correct: true`, `mma_fp16_smoke: launched + read back OK`, `mma_fp4_blockscale present: true`. Its last line names the sm_120 family; that text is fixed in `crates/memra-probe/src/main.rs`.

GPU UUIDs are hashed and home paths shown as `~`.
