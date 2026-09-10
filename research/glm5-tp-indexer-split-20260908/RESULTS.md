# GLM TP-2 pool-split indexer results

Decode verdict: NEGATIVE at 128k and FLAT at 1M. Merge cost is the named blocker: about 128 ms per GPU plus 36 to 40 ms exchange across 159 decode steps consumes most score/select savings. Prime wins at both contexts. The owner explicitly retains the code for the prime benefit. The decode half was then deleted and the retained prime-only door was measured on a 2x B200 dev pair on 2026-09-10: prime -34.25% at 1M and -10.36% at 128k with decode FLAT at both (-0.21%, +0.29%) and byte-identical output and merged planes, so the door's default flipped **ON** that day, ahead of its decide-by 2026-09-22. See [PRIME-ONLY.md](PRIME-ONLY.md) section "The pair cell that flipped the default". rev:2026-12-10.

Follow-up: the decode split is deleted and the retained prime-only door is
`MEMRA_GLM5_TP_INDEXER_SPLIT_PRIME`. See [PRIME-ONLY.md](PRIME-ONLY.md) for the new
source/test receipt; the rows below remain the original combined-door measurements.

## Timed rows

One 2x B200 pair, one GPU process at a time, fresh process per row, f32 indexer (`MEMRA_DSA_SCORE_TC=0`, RP=1), TP-2 expert split, symmetric graphs with eager MLA middles, grouped prime and host diet. Plain route, 160-token greedy identity instrument, NVFP4 latent cache OFF. All six pairs are interleaved OFF then ON. Compiler processes were absent during scoring. `MEMRA_ST_REPACK_DISK=0` matches the served loader and avoids a second on-disk repack copy. Source manifest and per-row binary/prompt/env/GPU receipts are retained.

| Context | Pair | OFF prime s | ON prime s | OFF decode tok/s | ON decode tok/s | IDs |
|---|---:|---:|---:|---:|---:|---|
| 128k | 1 | 36.6011 | 32.8148 | 69.812 | 66.947 | identical |
| 128k | 2 | 36.5982 | 32.8214 | 70.036 | 66.574 | identical |
| 128k | 3 | 36.5895 | 32.8982 | 69.852 | 66.968 | identical |
| 1m | 1 | 719.6012 | 470.6716 | 59.625 | 59.670 | identical |
| 1m | 2 | 719.8124 | 470.6359 | 59.824 | 59.860 | identical |
| 1m | 3 | 719.4553 | 470.6511 | 59.554 | 59.872 | identical |

Three-row medians: 128k prime 36.5982 to 32.8214 s (-10.3%), decode 69.852 to 66.947 tok/s (-4.2%). 1M prime 719.6012 to 470.6511 s (-34.6%), decode 59.625 to 59.860 tok/s (+0.4%, FLAT). The saved main 128k control was 36.9209 s / 67.457 tok/s. Earlier 1M controls were 782.3328 s prime and roughly 65 to 68 decode tok/s; these dated controls differ from the current same-binary OFF baseline.

Prompt lengths are 128,847 and 1,001,928 tokens. Each timed output has 160 IDs; none repeats a 16-word span four times. All paired files are byte-identical. All 128k rows also match the saved main control.

- 128k IDs SHA256: `308075f01e83ff9fce016c5ff5976e9b0578c287de7419edca941852edf7c087`.
- 1M IDs SHA256: `2eda121a69165e270d2daaa1ed6138f5c2aa2df6ce9ec308962d35f89dfd479c`. Both profiler outputs match too.
- Probe SHA256: `a0181695d5fdb50c6f52ba80914f68f7bfea8f171dc2e3bf3d3928c2975c2a79`.

## Nsight decode tally

Nsight Systems 2025.5.2, CUDA API capture limited to 159 decode steps, graph node tracing. Totals are summed kernel durations per GPU, not elapsed wall time. Profile rows are excluded from timing medians.

| Arm | GPU | Score ms | Select ms | Pack ms | Exchange ms | Merge ms | Total indexer ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| OFF | 0 | 213.891 | 355.490 | 0.000 | 0.000 | 0.000 | 569.381 |
| OFF | 1 | 213.227 | 346.486 | 0.000 | 0.000 | 0.000 | 559.713 |
| ON | 0 | 112.979 | 265.728 | 4.899 | 39.676 | 128.382 | 551.664 |
| ON | 1 | 112.633 | 273.619 | 3.985 | 35.710 | 128.001 | 553.948 |

Score launches: 1,749 per GPU per arm. Select launches: 10,494 per GPU per arm. ON adds 1,749 each of pack, exchange and merge per GPU. Exact kernel names, counts and nanoseconds are in each profile directory under `tally/kernels.csv`.

## Validation and custody

Remote build PASS (4m 30s), formatting PASS, CPU target 3 passed / 2 GPU tests ignored. The earlier lane GPU range/merge gates and full-model CHECK receipts remain under `receipts/`; CHECK timings are excluded. All 12 timed rows and both profiler rows exited zero and logged symmetric graph engagement. Pool splitting logged `indexer=pool-split`, `exchange=device-signal`, `capture=eager-middle` on ON.

Two setup failures produced no scored rows: GNU time was missing before launch, and the first loader attempt exhausted disk while writing 123,664,859,136 bytes of repack cache. The utility was installed, only the 91 newly created cache files were removed after a handle audit, and both arms used the in-memory loader afterward. Failed receipts remain separate.

Every completed row was copied to the rig before the next cell. One completed 1M ON row was recovered after an account interruption and was not rerun. Subsequent long processes used nohup/setsid; the receipt mirror ran independently as a task-scoped service.

This is a greedy mechanism and identity receipt. Vendor-default sampled serving and TC composition are outside this matrix; there is no default promotion. No cargo, benchmark, smoke server or test ran on the local rig. Push uses `MEMRA_SKIP_PERF_CI=1`; hosted CI remains required.

Public log copies normalize the deployment model-path prefix. Original logs and operation receipts are retained in the private lane; `receipts/pair-box/path-normalization.json` binds original and public file hashes. Numeric fields and token IDs are unchanged.

The measured source snapshot was banked in `5c73a12f967ec0f26afa53454e9e3aeae74309f6` before integration with current main. Binary hashes and source manifests above bind the measurements; hosted CI checks the integrated branch. No new performance measurement is inferred from that merge.
