## Build (2 jobs, nice 15, no sccache, balanced platform profile)

| what | N per arm | default wall s (median) | fat LTO wall s (median) | delta | default peak rustc RSS MB | fat LTO peak rustc RSS MB |
|---|---|---|---|---|---|---|
| cold `-p memra-server` (deps + nvcc kernels + server) | 3/3 | 499.4 (499.4, 500.6, 493.5) | 508.0 (486.0, 578.1, 508.0) | +1.7% | 2198 | 3137 |
| rebuild after touching memra-server/src/main.rs | 3/3 | 61.2 (60.3, 62.2, 61.2) | 168.9 (168.9, 168.1, 169.6) | +176.2% | 1309 | 2243 |
| rebuild after touching memra-engine/src/lib.rs | 3/3 | 131.6 (130.6, 131.8, 131.6) | 202.6 (201.8, 202.8, 202.6) | +54.0% | 2197 | 3137 |
| qualified gate binaries (kernel-check, run-gen, run-spec, argmax-margin-probe, tok-parity), incremental | 1/1 | 102.9 (102.9) | 184.3 (184.3) | +79.1% | 2262 | 2501 |
| `cargo test --release --no-run` CPU crates (reference, sampling, tokenizer, gguf, validate) | 1/1 | 61.4 (61.4) | 65.7 (65.7) | +7.1% | 696 | 1002 |
| `cargo test --release --no-run -p memra-engine --lib` | 1/1 | 98.8 (98.8) | 131.1 (131.1) | +32.7% | 2175 | 2300 |
| run those CPU-crate tests | 1/1 | 4.8 (4.8) | 5.5 (5.5) | +13.6% | 2590 | 2592 |
| run the engine lib tests | 1/1 | 5.4 (5.4) | 5.7 (5.7) | +4.4% | 825 | 821 |
| `cpu_native_check` bin, incremental | 1/1 | 1.2 (1.2) | 4.7 (4.7) | +300.8% | 174 | 258 |
| `--workspace --bins` (CI full-mode build, INSTALLATION.md), from the server-built state | 1/1 | 318.1 (318.1) | 2140.5 (2140.5) | +572.8% | 513 | 1233 |

memra-server binary bytes: default [74713920], fat LTO [62011040] (-17.0%)
max coretemp during builds: 102 C

## Serving host paths, RTX 5090 Laptop, Qwen3.5-9B NVFP4 MTP GGUF (rounds interleaved ABBAAB)

| metric | default (per round) | fat LTO (per round) | median delta |
|---|---|---|---|
| boot to /health 200 (s) | 5.252, 4.694, 4.59 | 4.425, 4.219, 4.526 | -5.7% |
| /v1/tokenize, 165525-token chat, median ms | 228.97, 263.71, 251.46 | 248.98, 241.85, 248.12 | -1.3% |
| streamed TTFT, short cached prompt, p50 ms (30/round) | 57.51, 58.07, 58.61 | 58.05, 58.25, 58.5 | +0.3% |
| streamed TTFT p95 ms | 59.44, 59.61, 60.72 | 60.79, 61.63, 61.07 | +2.4% |
| short request E2E p50 ms | 101.34, 102.35, 102.85 | 102.09, 103.07, 102.91 | +0.5% |
| greedy 512-token stream wall s (median of 3) | 3.046, 3.084, 3.19 | 3.064, 3.089, 3.293 | +0.2% |
| tokenize-only rounds: /v1/tokenize median ms (60/round) | 254.49, 256.81, 260.44 | 240.84, 240.55, 238.9 | -6.3% |
| tokenize-only rounds: server CPU ms per tokenize (10 ms clock ticks) | 250.0, 250.0, 260.0 | 240.0, 240.0, 235.0 | -4.0% |
| tokenize-only rounds: boot to /health 200 (s) | 7.754, 4.307, 4.34 | 4.131, 4.118, 4.473 | -4.8% |

Boot includes model load; the first round of each campaign (A) ran on a colder page cache.

## Correctness

- cpu_native_check: 51 lines per arm, differing lines: ['[memra-cpu-profile]'] (timing profile only)
- tokenize_ids_sha256: 1 distinct value(s) across both arms and all rounds: ['28237cad8122be22d93c152adcce9380145a2e3b430187cbd476f6f74ea05967']
- greedy_sha256: 1 distinct value(s) across both arms and all rounds: ['7daadd12a1d5999ee608e4d623f47b57b124f75e47585ccc53d71ff011be2f46']
- sampled_sha256: 1 distinct value(s) across both arms and all rounds: ['ab0756da5563c65145777ff93cd4248aa099786e464f31e4bce836fda24b01e1']
- greedy identical within every round: True; sampled (seed 1234) identical within every round: True
- platform profile on every serving round: ['balanced']
- server binary sha256: default ['d3ef5e5d2894efc579f877f2cc2012045f39c1422bbf6e31558eb320b715dd8a'], fat LTO ['5c6652e579af5d289d819c2d05f50478f8e845ed7ec77d118bc8857420ea372b']
