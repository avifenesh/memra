# Boot wave probe: pool residue and device sizing (2026-11-06)

Verdict: the boot wave probe (`decode_wave_probe`, from 80b0d5710a) left its freed pool blocks mapped, so the prefix-cache budget and the admission calibration read a full card. On an RTX 5090 Laptop (24 GB) a naked Qwen3.8-27B boot had no prefix cache, and with DSpark it refused a 161-token request on an idle server. The probe now hands its pool back to the driver on both arms and refuses early when 32 sessions cannot fit. Q38 boots get their prefix cache back, the DSpark floor returns from 4680 MB to 1544 MB, the failed probe drops from 48.5-51.5 s to 5.3-8.0 s, and the 9B probe decides the same cap. The spec-cost calibration gets the same fit check: on the Q38 plain route it is skipped after 9.7 s instead of running out of memory after 45 s, so boot is ready in 42 s instead of 73-79 s; the 9B calibrates the same tables.

## Mechanism

- `measured_wave_cap` runs `decode_wave_probe` at boot: 32 sessions primed at depth 2048 plus 32 snapshots, then timed waves. It frees every buffer on return, but the async pool keeps freed blocks mapped (release threshold `u64::MAX`), so `cuMemGetInfo` still counts them as used.
- Right after it, `init_prefix_cache_budget` reads driver free memory. With the residue mapped, the post-reserve clamp was 0 and the derived budget 0 B.
- The admission calibration resets the pool high-water mark to the current reservation, which still held the residue, and trims only after its probe. The residue counted as calibration transient: peak-mapped 23712 MB against 20480 MB without it.
- On the MTP route the spec-cost calibration trims before the admission calibration, so the floor stays right there, but the prefix budget is still derived with the residue mapped.

## Change

1. `fix(server)`: `measured_wave_cap` fences the model devices and calls `trim_model_device_pools(engine, loaded, "wave-probe")` after the probe on both arms, before it decides the cap, the way the spec-cost calibration does. It logs the probe time and the bytes released. `decode_chunk_policy` takes `loaded` for that. A source-order test (`wave_probe_releases_its_pool_before_boot_reads_free_memory`) pins probe, fence, trim, decision, and boot's order of caps, prefix budget and admission calibration. Replayed against main's `worker.rs`, its anchors for the fence, the trim and the new call are missing; it passes on the fix.
2. `perf(engine)`: the probe primes and snapshots one session at a time. After the second session it reads that session's pool footprint (pool USED delta after a stream sync; the first session also carries one-time scratch) and refuses when the other 30 need more than is allocatable (`allocatable_mem_info` free plus the pool's reusable cache). It refuses only when the measured per-session footprint says priming cannot complete. The timed waves are unchanged.
3. `perf(engine)`: the spec-cost calibration's plain-batch table (16 sessions, 32 on the b32 tier) gets the same check. Both probes call one helper, `probe_session_fit`, after each primed and snapshotted session; the decision is the pure `sessions_fit_refusal`. A source-order test (`boot_probes_check_their_fit_before_timing`) pins the check between each probe's last prime and its first timed batch step, with the snapshot inside the measured footprint. It failed before this commit and passes after: `raw/cpu/spec-cost-test-red.log` is the test alone on the previous tree (9e3c052a88, then named fd40f2de39), `spec-cost-test-green.log` the commit, `spec-cost-test-head.log` the lane head.
4. Review fixes: the trim already fences every owning stream, so the extra fence is gone. A refusal is a typed `ProbeDoesNotFit`, logged as `[wave-probe] skipped (...)` and `[spec-cost] calibration skipped`, not as a failure. An unreadable free reading never refuses. The per-session footprint prints with one decimal.

## Conditions

RTX 5090 Laptop GPU, 24463 MiB, driver 615.71.09, local rig, 2026-11-06. Boots ran under the 5090 lease with no other compute app (`apps_before` empty in every row). Each boot is a single run; the probe's own timings are medians of 3 reps inside each boot. 250 ms `nvidia-smi` telemetry is beside every log.

| arm | source | build | memra-server sha256 |
|---|---|---|---|
| base | main 5d70118b99 (source-identical to 3bbdf42ff9; source id 19b572b06fea) | cell-evidence build, fat LTO, 1 CGU, `MEMRA_CUDA_ARCH=120a` | `4d5033f3adf7d93a624684d289a27e594f57565d4c7e412bff892b4236d23b76` |
| fix | 2cf3ad3bbe (source id 91e4dc934ad6; built as c3cd67c93b, the same tree before a message-only rewrite) | same settings | `7698cdea2597bd559eda9b7a56398946998ff2cff161f11ab47d3335fe887b49` |
| head (`raw/p4` only) | 62f77cdf1b (source id f738908d621d) | same settings | `a4d2c2904ee2999aa2d30662841f10346b9e11385742bc8bc2763e2c442f2551` |

Environment per cell, nothing else (`boot_cells.py` strips inherited `MEMRA_*`): `MEMRA_COMPAT=openai`, `MEMRA_ADDR`, and

- `q38-dspark`: Qwen3.8-27B NVFP4-Q5K GGUF + DFlash2 drafter, `MEMRA_DSPARK_SPEC=1`, `MEMRA_CTX=8192`, default session cap.
- `q38-dspark-s4`: the same with `MEMRA_MAX_SESSIONS=4`; the probe is skipped (no-probe control).
- `q38-plain`: Qwen3.8-27B GGUF, `MEMRA_CTX=8192`, default session cap.
- `q9`: Qwen3.5-9B NVFP4 MTP GGUF, `MEMRA_MAX_SESSIONS=32`, `MEMRA_CTX=32768` (the env of the probe's original 5090 receipt).

Each boot sends one greedy chat request (65 prompt tokens on Q38, 32 max tokens; the admission context is 161).

## Results

### Q38 + DSpark (`q38-dspark`)

| run | probe | prefix cache | boot free at budget | transient floor (peak / rest mapped) | request | ready |
|---|---|---|---|---|---|---|
| base p1 | OOM after 48.5 s | 0 B, disabled | (not logged) | 4680 MB (23712 / 18912) | HTTP 400, does not fit an idle server | 75.1 s |
| base p2 | OOM after 51.5 s | 0 B, disabled | (not logged) | 4680 MB (23712 / 18912) | HTTP 400 | 84.1 s |
| fix p2 #1 | refused after 8.0 s, released 1696 MB | 800 MB | 7028 MiB | 1544 MB (20480 / 18816) | admitted | 39.1 s |
| fix p2 #2 | refused after 7.8 s, released 1696 MB | 800 MB | 7028 MiB | 1544 MB (20480 / 18816) | admitted | 37.0 s |
| fix p2 #3 | refused after 5.3 s, released 1696 MB | 800 MB | 7028 MiB | 1544 MB (20480 / 18816) | admitted | 32.0 s |
| no-probe control (s4, base) | skipped | 800 MB | 8628 MiB | 1608 MB (20480 / 18752) | admitted | 28.0 s |

The refusal reads `32 sessions at depth 2048 do not fit: the other 30 need 15175MB (505MB each), 7016MB allocatable`. The base OOM is a captured `CUDA_ERROR_OUT_OF_MEMORY` with telemetry at 24024 MiB used and no other compute app. The fix floor of 1544 MB is 64 MB under the no-probe control in this lane; a binary from before 80b0d5710a measured 1640 MB on the same cell.

The fix boot reads 1.6 GB less driver free at the budget than the no-probe control. That is the first prime's working set outside the pool, which the control allocates later in its own calibration: at the end of calibration `still_owned_bytes` is identical (19560811392 B) and driver free is within 64 MiB of the control (base: 160 MiB below). The budget is 800 MB either way (requested 2 x 400 MB, well under the clamp).

### Q38 plain route (`q38-plain`)

| run | probe | prefix cache | boot free at budget | floor | request | ready |
|---|---|---|---|---|---|---|
| base p1 | OOM after 51.5 s | 0 B, disabled | (not logged) | 2138 MB | admitted | 119.1 s |
| fix p2 | refused after 5.4 s, released 1696 MB | 800 MB | 7956 MiB | 2202 MB | admitted | 75.1 s |

The spec-cost calibration that follows OOMs on both of these arms (`[spec-cost] calibration FAILED ... CUDA_ERROR_OUT_OF_MEMORY; the band serves`, 46 s on base, 45.6 s on fix). It primes 16 sessions at the same depth (16 x 505 MB against about 8.3 GB free), a sizing limit of its own on a 24 GB card, not the residue. Commit 3 gives it the fit check; `raw/p4` measures it.

### Spec-cost fit check, same window, A B B A (`raw/p4`)

prev = 2cf3ad3bbe (the wave-probe fix only), new = the lane head 62f77cdf1b.

Q38 plain route (`q38-plain`):

| run | spec-cost calibration | time in it | prefix cache | floor | request | ready |
|---|---|---|---|---|---|---|
| prev #1 | FAILED, `CUDA_ERROR_OUT_OF_MEMORY`; the band serves | 45.2 s | 800 MB | 2202 MB | admitted | 79.1 s |
| new #1 | skipped: the other 14 need 7081MB (505.8MB each), 6992MB allocatable; the band serves | 9.7 s | 800 MB | 2202 MB | admitted | 42.1 s |
| new #2 | skipped, same numbers | 9.7 s | 800 MB | 2202 MB | admitted | 42.0 s |
| prev #2 | FAILED, `CUDA_ERROR_OUT_OF_MEMORY` | 45.4 s | 800 MB | 2202 MB | admitted | 73.1 s |

The verdict is the same on both arms (the band serves); the new arm reaches it about 36 s sooner. The 9.7 s covers the verify table and the two primed sessions before the refusal. The wave probe line on the new arm reads `[wave-probe] skipped (... 505.8MB each ...)`.

Qwen3.5-9B (`q9`), where the calibration fits (plain cap 32, so 32 sessions primed):

| run | verify_ms R=1 / 8 / 16 | plain_ms B=1 / 16 / 32 | round overhead | pooled overhead (4) | calibration | wave probe |
|---|---|---|---|---|---|---|
| prev #1 | 13.8 / 27.4 / 48.5 | 13.6 / 51.5 / 97.1 | 9.07 ms | 2.72 ms | 34.5 s | cap 32, b32 on |
| new #1 | 13.8 / 27.4 / 48.6 | 13.5 / 51.9 / 97.0 | 9.21 ms | 2.73 ms | 34.5 s | cap 32, b32 on |
| new #2 | 13.8 / 27.4 / 48.5 | 13.6 / 51.8 / 96.6 | 9.25 ms | 2.68 ms | 34.4 s | cap 32, b32 on |
| prev #2 | 13.8 / 27.6 / 48.6 | 13.6 / 51.6 / 96.9 | 9.09 ms | 2.64 ms | 34.4 s | cap 32, b32 on |

The fit check passes on the 9B and the tables agree within run-to-run spread; full rows are in the logs. Round overhead reads 0.12-0.18 ms higher on both new runs; with two runs per arm that is not separable from noise, and the commit changes nothing in that section (it runs after the plain table, whose caches are dropped). A Q38 DSpark boot on the head (`q38-dspark-new-1`) skips the wave probe the same way, keeps the 800 MB prefix cache and the 1544 MB floor, and is ready in 36.0 s.

### Q9, same window, A B B A (`q9`)

| run | two 16-row waves | one 32-row wave | tensor-core tier | decision | probe | released | boot free at budget | prefix | floor |
|---|---|---|---|---|---|---|---|---|---|
| base #1 | 103.4 ms | 100.3 ms | 98.6 ms | cap 32, tensor-core b32 on | 25.7 s | (kept mapped) | 10420 MiB | 1078 MB | 1536 MB |
| fix #1 | 103.5 ms | 101.3 ms | 96.9 ms | cap 32, tensor-core b32 on | 25.5 s | 6144 MB | 16180 MiB | 1078 MB | 1536 MB |
| fix #2 | 103.0 ms | 100.7 ms | 96.8 ms | cap 32, tensor-core b32 on | 25.5 s | 6144 MB | 16180 MiB | 1078 MB | 1536 MB |
| base #2 | 102.9 ms | 100.4 ms | 96.6 ms | cap 32, tensor-core b32 on | 25.5 s | (kept mapped) | 10420 MiB | 1078 MB | 1536 MB |

The successful probe decides the same cap and tier and passes the fit check. It leaves 6144 MB mapped on main; the fix returns it before the budget. The 9B budget does not move (requested under the clamp), and its floor is unchanged because its spec-cost calibration trims before the admission calibration.

## GB10 (unified memory)

Same code path. `trim_model_device_pools` already runs on GB10 after the spec-cost and admission calibrations, and the fit check reads `allocatable_mem_info`, the page-cache aware sizing read GB10's prefix budget already uses (the larger of the driver reading and `MemAvailable`). On GB10 the 27B probes fit (main ran the wave probe's 32 sessions in 80 s and the spec-cost calibration in 108 s), so both checks pass and the cap decision and tables are unchanged.

Two GB10 values were expected to move (prediction as written before the GB10 run; the measurement follows):

- The unified-memory prefix budget (the larger of two entries and half of allocatable free after the transient reserve) is now derived after the wave probe's pool is returned, so it should rise by about half the residue. The residue should be about 16 GB: 32 sessions x 505 MB (the per-session footprint measured on the 5090 for the same checkpoint and depth) plus wave scratch. Main's GB10 boot derived 42791 MB from 89.3 GB free, before a spec-cost trim that released 16.9 GB of probe pool. Expected: roughly +8 GB of prefix budget.
- A GB10 DSpark boot skips the spec-cost calibration, so on main its admission calibration reset the high-water mark with that residue mapped and charged it as transient, the same mechanism as the 5090 DSpark cell. Expected: the floor drops from about 17-18 GB to the real transient, about 1.5-2.2 GB, which raises admission capacity on that route.

### GB10 measured (combined landing run, 2026-11-06)

One DGX Spark (GB10, driver 580.178.04), Qwen3.8-27B GGUF (NVFP4+Q5_K trunk), `MEMRA_CTX=262144`, `MEMRA_MAX_SESSIONS=32`, `MEMRA_KV_HOST_MB=0`, derived unified-memory defaults, one fresh boot per cell, order main / fix / fix / main per route, one 32-token greedy request per boot. main = `5d70118b99`; fix = `b645e05e0b` (this PR on main with memra#1073 and memra#1074 below it; this PR's code is byte-identical to `aa309313d4`). Both binaries are the cell-evidence build (two jobs, one codegen unit, fat LTO, `121a`). Harness: `gb10-boot-ab.sh`; raw logs, 250 ms telemetry and per-cell env in `raw/gb10/`.

MTP route (masked sxc32768 MTP draft, so the spec-cost calibration runs):

| cell | wave probe | prefix budget | boot driver free | transient floor (peak / rest mapped) | request | ready |
|---|---|---|---|---|---|---|
| main 1 | cap 32 in 81.0 s, nothing released | 42445 MB | 88.65 GB | 2240 MB (19200 / 16800) | 200 | 207.6 s |
| fix 1 | cap 32 in 69.0 s, 16736 MB released | 51086 MB | 105.93 GB | 2078 MB (19040 / 16800) | 200 | 181.4 s |
| fix 2 | cap 32 in 69.1 s, 16736 MB released | 51086 MB | 105.93 GB | 2078 MB (19040 / 16800) | 200 | 181.4 s |
| main 2 | cap 32 in 80.9 s, nothing released | 42448 MB | 88.65 GB | 2238 MB (19200 / 16800) | 200 | 207.5 s |

DSpark route (DFlash2 drafter, FR-Spec trim, `MEMRA_MTP_SKIP=1`, restore doors on, so no spec-cost calibration):

| cell | wave probe | prefix budget | boot driver free | transient floor (peak / rest mapped) | request | ready |
|---|---|---|---|---|---|---|
| main 1 | cap 32 in 80.9 s, nothing released | 41861 MB | 87.48 GB | 13064 MB (32000 / 18816) | 200 | 104.8 s |
| fix 1 | cap 32 in 69.1 s, 16672 MB released | 50314 MB | 104.39 GB | 1576 MB (20384 / 18688) | 200 | 90.6 s |
| fix 2 | cap 32 in 69.2 s, 16672 MB released | 50300 MB | 104.36 GB | 1576 MB (20384 / 18688) | 200 | 90.7 s |
| main 2 | cap 32 in 81.2 s, nothing released | 42162 MB | 88.08 GB | 13064 MB (32000 / 18816) | 200 | 104.8 s |

- Prefix budget: +8.6 GB on the MTP route (42445-42448 to 51086 MB) and +8.3 GB on the DSpark route (41861-42162 to 50300-50314 MB), as predicted (about half the 16.7 GB residue the trim now returns before the budget is derived).
- DSpark transient floor: 13064 MB on main to 1576 MB, inside the predicted 1.5-2.2 GB. The prediction for main (17-18 GB) was high: main measured 13.1 GB. The MTP route's floor moves 2239 to 2078 MB, because its spec-cost calibration already trimmed before the admission calibration.
- The probes decide the same thing on both binaries: every boot measured two 16-row waves at 262.5-264.2 ms, one 32-row wave at 263.6-265.0 ms and the tensor-core tier at 229.1-230.7 ms, so wave cap 32 with tensor-core b32 on. The spec-cost tables agree within run-to-run spread (round overhead 13.26 / 12.65 ms main, 13.04 / 13.00 ms fix). The wave probe itself takes 12 s less, and boots are ready 26 s sooner on the MTP route and 14 s sooner on the DSpark route.
- Single boots per cell, two per arm, interleaved; no request failed and no boot logged an out-of-memory line.

## Files

- `boot_cells.py`: the harness (`MODELS=<dir>` names the model root).
- `raw/p1/`: base Q38 DSpark, s4 control and plain boots. `raw/p2/`: fix and base in one window, Q38 and the Q9 A B B A. `raw/p4/`: the spec-cost fit check, prev vs head, Q38 plain and Q9 A B B A plus one Q38 DSpark boot. `raw/cpu/`: the source-order test red on the previous tree, green on its commit and on the head. Each boot has a log with seconds since spawn on every line, a 250 ms `nvidia-smi` csv, and a row in `rows.jsonl`. Local paths are replaced with `~`, `<models>` and `<bins>`.
- `gb10-boot-ab.sh` and `raw/gb10/`: the GB10 boot cells above (`chain.log` is the run's own summary; each cell has `env.txt`, `server.log`, `gpu.csv`, `ready.txt` and the request's reply). The home directory reads `~`.
- Harness bugs, kept as recorded: `raw/p1/cells.txt` labels the s4 control FAIL because its prefix check missed the `[prefix-cache]` prefix (the row shows 800 MB and an admitted request). In `raw/p2` the first fix Q38 run's log and csv were overwritten by the second, from a per-invocation counter; its row stays in `rows.jsonl` (8.0 s), `q38-dspark-fix-1.log` is the 7.8 s run and `q38-dspark-fix-2.log` the 5.3 s run.
