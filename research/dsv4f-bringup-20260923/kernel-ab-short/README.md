# Shared-control served campaign: sink, HC2, fused2 (2026-09-23/24)

One served campaign scored three kernel lanes against one control on 2x RTX PRO 6000 Blackwell
Workstation Edition (450 W limit, driver 580.159.03), `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`,
PP-2, matrix expert program, host sampler. Each lane PR carries this same directory and its own
write-up reads its rows from here.

| arm | tree | server sha256 (16) | write-up |
|---|---|---|---|
| base | main `6978f5fac` | `aa634173e6d69230` | control |
| sink | `lane/dsv4-sink-attn-20260923` `764aca9f1` | `fe2dbd53a762043f` | `../sink-attn/RESULTS.md` |
| hc2 | `lane/dsv4-mhc-20260923` `d9df672be` | `48c2204b9991aab8` | `../hc-finish/RESULTS.md` |
| fused2 | `lane/dsv4-moe-fused-20260923` `899aac4d0` | `7265733070b2fe7b` | `../moe-fused/RESULTS.md` |

Protocol (`q-pair.sh`): one server boot per row (`cell.sh`), under `/tmp/memra-gpu.lock`, 250 ms
`nvidia-smi` telemetry per row. Four arms, five rows each, in a Williams order (every arm follows
every other arm once), DSpark (`MEMRA_DSV4_DRAFTER=dspark`) then plain. Cells
(`cells-spec.txt`, `bench.py`): greedy c1 (8 requests, text kept and hashed), sampled c1 (8), greedy
c1 with `ignore_eos` (4), 256 output tokens each, after a 32-token warmup. The checkpoint bytes were
verified before the first row (`summary.txt`, `hashchk files=77 lfs=48 bad=0`).

`lane_tab.py short <arm>` prints one arm's rows, medians and deltas against base, per-cell hash
identity across all rows of both arms, and the thermal regime.

## Medians (N=5 per arm, decode tok/s p50)

| arm | DSpark greedy | DSpark sampled | plain greedy | plain sampled |
|---|---|---|---|---|
| base | 66.87 | 54.09 | 50.93 | 46.04 |
| sink | 67.67 (+1.2%) | 54.70 (+1.1%) | 53.53 (+5.1%) | 48.07 (+4.4%) |
| hc2 | 68.32 (+2.2%) | 55.24 (+2.1%) | 53.87 (+5.8%) | 48.45 (+5.3%) |
| fused2 | 67.19 (+0.5%) | 54.33 (+0.4%) | 56.62 (+11.2%) | 50.48 (+9.7%) |

Every completed request of every arm hashed the same as base in every cell: greedy
`aea6e69e 26ac8df7 850f75ed a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8`, sampled
`53944095 73fe7f91 6329ee94 827c0b56 437a4427 07de3a97 8a59518d b424bd80`.

## The one incomplete row

`short/plain/r13-fused2` hung in the fourth `greedy-c1-ignore-eos` request. It streamed 181
tokens at the normal cadence, an exact prefix of base's text for that request, then stopped. The
row has no deadline; after about 440 s of silence the thread states were captured and the server
was stopped by hand (row rc 143, `serve.log` ends in the SIGTERM drain). While hung, the `dsv4-serve` thread was running on the host (state R, 421 s of user time,
`short/plain/r13-fused2/hang-threads.txt`) and both GPUs sat idle at 0% utilization and 180 MHz
(`telemetry-250ms.csv`). The pod refused ptrace, so no stack exists (`hang-bt.txt`). The cut
text hashes differently (`96bbce86`); the three requests before it match base, and the row's
greedy and sampled cells completed and are in the medians.

The cause is not attributed. Only fused2's plain rows run the fused MoE kernels, and none of the
other 39 rows hung, but one event in five fused rows is weak evidence, and an idle device with a
spinning host thread fits a blocked CUDA wait as well as a host loop. The fused2 lane reproduces
it on a box that allows ptrace before that lane lands.
