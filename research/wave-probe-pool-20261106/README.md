# Boot wave probe: pool residue and device sizing (2026-11-06)

Verdict: the boot wave probe (`decode_wave_probe`, from 80b0d5710a) left its freed pool blocks mapped, so the prefix-cache budget and the admission calibration read a full card. On an RTX 5090 Laptop (24 GB) a naked Qwen3.8-27B boot had no prefix cache, and with DSpark it refused a 161-token request on an idle server. The probe now hands its pool back to the driver on both arms and refuses early when 32 sessions cannot fit. Q38 boots get their prefix cache back, the DSpark floor returns from 4680 MB to 1544 MB, the failed probe drops from 48.5-51.5 s to 5.3-8.0 s, and the 9B probe decides the same cap.

## Mechanism

- `measured_wave_cap` runs `decode_wave_probe` at boot: 32 sessions primed at depth 2048 plus 32 snapshots, then timed waves. It frees every buffer on return, but the async pool keeps freed blocks mapped (release threshold `u64::MAX`), so `cuMemGetInfo` still counts them as used.
- Right after it, `init_prefix_cache_budget` reads driver free memory. With the residue mapped, the post-reserve clamp was 0 and the derived budget 0 B.
- The admission calibration resets the pool high-water mark to the current reservation, which still held the residue, and trims only after its probe. The residue counted as calibration transient: peak-mapped 23712 MB against 20480 MB without it.
- On the MTP route the spec-cost calibration trims before the admission calibration, so the floor stays right there, but the prefix budget is still derived with the residue mapped.

## Change

1. `fix(server)`: `measured_wave_cap` fences the model devices and calls `trim_model_device_pools(engine, loaded, "wave-probe")` after the probe on both arms, before it decides the cap, the way the spec-cost calibration does. It logs the probe time and the bytes released. `decode_chunk_policy` takes `loaded` for that. A source-order test (`wave_probe_releases_its_pool_before_boot_reads_free_memory`) pins probe, fence, trim, decision, and boot's order of caps, prefix budget and admission calibration. Replayed against main's `worker.rs`, its anchors for the fence, the trim and the new call are missing; it passes on the fix.
2. `perf(engine)`: the probe primes and snapshots one session at a time. After the second session it reads that session's pool footprint (pool USED delta after a stream sync; the first session also carries one-time scratch) and refuses when the other 30 need more than is allocatable (`allocatable_mem_info` free plus the pool's reusable cache). It only stops a probe whose priming could not complete. The timed waves are unchanged.

## Conditions

RTX 5090 Laptop GPU, 24463 MiB, driver 615.71.09, local rig, 2026-11-06. Boots ran under the 5090 lease with no other compute app (`apps_before` empty in every row). Each boot is a single run; the probe's own timings are medians of 3 reps inside each boot. 250 ms `nvidia-smi` telemetry is beside every log.

| arm | source | build | memra-server sha256 |
|---|---|---|---|
| base | main 5d70118b99 (source-identical to 3bbdf42ff9; source id 19b572b06fea) | cell-evidence build, fat LTO, 1 CGU, `MEMRA_CUDA_ARCH=120a` | `4d5033f3adf7d93a624684d289a27e594f57565d4c7e412bff892b4236d23b76` |
| fix | c3cd67c93b (source id 91e4dc934ad6) | same settings | `7698cdea2597bd559eda9b7a56398946998ff2cff161f11ab47d3335fe887b49` |

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

Not changed here: the spec-cost calibration that follows OOMs on both arms (`spec-cost] calibration FAILED ... CUDA_ERROR_OUT_OF_MEMORY; the band serves`, 46 s on base, 45.6 s on fix). It primes 16 sessions at the same depth (16 x 505 MB against about 8.3 GB free), so it is a sizing limit of its own on a 24 GB card, not the residue. Follow-up: the same fit check would make it fail in a few seconds.

### Q9, same window, A B B A (`q9`)

| run | two 16-row waves | one 32-row wave | tensor-core tier | decision | probe | released | boot free at budget | prefix | floor |
|---|---|---|---|---|---|---|---|---|---|
| base #1 | 103.4 ms | 100.3 ms | 98.6 ms | cap 32, tensor-core b32 on | 25.7 s | (kept mapped) | 10420 MiB | 1078 MB | 1536 MB |
| fix #1 | 103.5 ms | 101.3 ms | 96.9 ms | cap 32, tensor-core b32 on | 25.5 s | 6144 MB | 16180 MiB | 1078 MB | 1536 MB |
| fix #2 | 103.0 ms | 100.7 ms | 96.8 ms | cap 32, tensor-core b32 on | 25.5 s | 6144 MB | 16180 MiB | 1078 MB | 1536 MB |
| base #2 | 102.9 ms | 100.4 ms | 96.6 ms | cap 32, tensor-core b32 on | 25.5 s | (kept mapped) | 10420 MiB | 1078 MB | 1536 MB |

The successful probe decides the same cap and tier and passes the fit check. It leaves 6144 MB mapped on main; the fix returns it before the budget. The 9B budget does not move (requested under the clamp), and its floor is unchanged because its spec-cost calibration trims before the admission calibration.

## GB10 (unified memory)

Same code path. `trim_model_device_pools` already runs on GB10 after the spec-cost and admission calibrations, and the fit check reads `allocatable_mem_info`, the page-cache aware sizing read GB10's prefix budget already uses (the larger of the driver reading and `MemAvailable`). On GB10 the 27B probe fits (it ran 32 sessions at 80 s on main), so the check passes and the cap decision is unchanged. Two values move on GB10 and are not measured here:

- The unified-memory prefix budget (the larger of two entries and half of allocatable free after the transient reserve) is now derived after the probe's pool is returned, so it rises by about half the residue. Main's GB10 boot derived 42791 MB from 89.3 GB free, before a spec-cost trim that released 16.9 GB of probe pool.
- A GB10 DSpark boot skips the spec-cost calibration, so on main its admission floor included the probe residue the same way as on the 5090. The fix removes that.

## Files

- `boot_cells.py`: the harness (`MODELS=<dir>` names the model root).
- `raw/p1/`: base Q38 DSpark, s4 control and plain boots. `raw/p2/`: fix and base in one window, Q38 and the Q9 A B B A. Each boot has a log with seconds since spawn on every line, a 250 ms `nvidia-smi` csv, and a row in `rows.jsonl`. Local paths are replaced with `~`, `<models>` and `<bins>`.
- Harness bugs, kept as recorded: `raw/p1/cells.txt` labels the s4 control FAIL because its prefix check missed the `[prefix-cache]` prefix (the row shows 800 MB and an admitted request). In `raw/p2` the first fix Q38 run's log and csv were overwritten by the second, from a per-invocation counter; its row stays in `rows.jsonl` (8.0 s), `q38-dspark-fix-1.log` is the 7.8 s run and `q38-dspark-fix-2.log` the 5.3 s run.
