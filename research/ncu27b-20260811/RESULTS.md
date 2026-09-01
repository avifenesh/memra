# Qwen3.6-27B c=1 speculative-decode anatomy on sm_120a

Date: 2026-08-11

Rig: local NVIDIA GeForce RTX 5090 Laptop GPU, 82 SMs, 24 GiB, driver 595.84, CUDA 13.1

## Verdict

The K=3, c=1 second-listing path spends **30.413 ms per speculative round** in this bounded
64-token window. Target verification matvecs own **20.924 ms/round (68.80%)**; all FA decode owns
**3.158 ms (10.38%)**; the own-trim drafter head itself is only **0.336 ms (1.10%)**. GPU argmax
sampling is **0.030 ms (0.10%)**. This is still a target-verify problem, not a drafter-head or
sampling problem.

The top three unique kernel symbols below both requested SOL thresholds (SM < 60% and memory <
70%), ranked by their unperturbed time weight, are:

1. `qmatvec_nvfp4_mmvq_b4_rpr2w8` — **7.785 ms/round, 25.60%**; exposed DRAM latency at only
   0.98/1.95 waves per SM.
2. `fa_decode_f32` — **3.048 ms/round, 10.02%**; 24-block launch underfills an 82-SM GPU.
3. `qmatvec_nvfp4_mmvq_b4_rp` — **2.730 ms/round, 8.98%**; two tiny shapes are occupancy/launch
   limited and the 1,536-block shape is DRAM-latency limited.

No top-three kernel has shared-memory bank conflicts. The only high-conflict selected shape is
the 0.034%-of-round `fa_decode_vec_q_rows_v4`, so bank-conflict work is not a useful first lever.

## Exact run contract

| Item | Value |
|---|---|
| Trunk | `Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf` |
| Trunk SHA-256 | `d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517` |
| External drafter | `draft-daily-owntrim-nvfp4head-q4blk.gguf` |
| Drafter SHA-256 | `b445fbb139e72f9869df06f2f0f91bcaf57527ec34a24bec74d3febd719f3581` |
| `run-spec` SHA-256 | `220d527beeb4bd2306cfd017ababdf2c501326bff9fcf87bcf4015e38c84561c` |
| Prompt | chat-templated `research/e2e/prompts/p1-code-short.txt` (37 tokens) |
| Shape | c=1, `MEMRA_SPEC_K=3`, `MEMRA_NGEN=64`, `MEMRA_CHAT=1` |
| Result | 21 rounds; 42/63 accepted = 66.7%; self-consistency PASS |
| Timing profiler | Nsight Systems 2025.5.2, one profiler-isolated round-loop window |
| Counter profiler | Nsight Compute 2025.4.1, 79 launch shapes, one launch per unique config |
| Platform regime | `balanced`; timing boundary 60 C -> 62 C |
| NCU regime | explicit base clocks, cache-control `all`; end boundary 66 C |

Both captures used the same release binary. Research-only capture commits changed between the
timing and counter runs; engine code and the binary hash did not.

## Method and trust boundary

`MEMRA_PROFILE_SPEC=2` starts profiling after the speculative prime. The timing receipt is a
separate Nsight Systems capture because NCU replays each selected launch; the NCU log's 339-second
instrumented phase is deliberately not used as elapsed-time evidence. This follows NVIDIA's
documented replay model and CUDA-profiler-API filtering behavior in the
[Nsight Compute CLI guide](https://docs.nvidia.com/nsight-compute/2025.4/NsightComputeCli/index.html).

The NCU pass used `--profile-from-start off`, node-level CUDA graph profiling,
`--filter-mode per-launch-config`, `--launch-count 1`, base clocks, and the requested SOL,
memory, occupancy, launch, scheduler, and warp-state sections. Explicit counters captured shared
load/store wavefronts and bank conflicts. NVIDIA defines the SpeedOfLight percentages as achieved
throughput relative to the GPU's sustained peak; occupancy and warp-stall interpretation follow
the [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html).
The archive [release notes](https://docs.nvidia.com/nsight-compute/2025.4/ReleaseNotes/index.html)
record improved Blackwell support and individual CUDA-graph-node profiling in 2025.3, preceding
the installed 2025.4.1 build.

Time weights below come only from the normal-clock Nsys trace. NCU supplies one counter sample per
unique `(kernel, grid, block, shared-memory)` launch shape. The join covers **99.50% of all GPU
kernel time**. When a symbol has multiple launch shapes, its SOL/occupancy columns are weighted by
that shape's Nsys time. NCU's base-clock, cache-flushed durations are never mixed into the time
column.

The Nsys range spans 638.676 ms. GPU busy-time union is 609.850 ms, with only 0.099 ms of kernel
overlap and 28.826 ms of gaps. The in-engine phase accumulator totals 637.4 ms, within 0.2% of the
trace span. This is a single capture (N=1), as requested, not a benchmark median.

Scalar arguments are not part of NCU's launch-config identity. `fa_decode_f32` keeps the same
24x1x1 grid while context length grows, so its counter row is the first, earliest-context launch;
its 3.048 ms/round time weight still averages every launch in the 64-token Nsys window. The
24-block occupancy diagnosis is invariant, while its exact scoreboard ratios may drift with
context depth.

## Where one round goes

| Bucket | ms/round | Round share |
|---|---:|---:|
| Target verify qmatvec/mmvq | 20.924 | 68.80% |
| FA decode, all symbols | 3.158 | 10.38% |
| Host/API/launch gaps, net of 0.10 ms overlap | 1.368 | 4.50% |
| GDN/SSM glue | 1.149 | 3.78% |
| RMS/RoPE glue | 1.125 | 3.70% |
| Drafter-body qmatvec/mmvq | 1.028 | 3.38% |
| CUDA memcpy/memset | 0.745 | 2.45% |
| Other GPU kernels | 0.551 | 1.81% |
| Own-trim drafter head | 0.336 | 1.10% |
| GPU sampling/argmax | 0.030 | 0.10% |

Classification is launch-contract based: batched `_b2`/`_b4` qmatvecs are target verification;
generic Q4_K/Q6_K qmatvecs are the external drafter body; and
`qmatvec_nvfp4_mmvq_mr2_rp` is the 32,768-row trimmed drafter head. The target counts provide a
cross-check: 1,344 dual launches = 64 layers x 21 rounds, both main single-launch families have
3,696 launches = 176 x 21, and the Q5_K target head has 21 launches.

The engine's separately synchronized `commit-host` phase is 14.0 ms total, or 0.667 ms/round. It
contains acceptance/rollback/control work and overlaps categories already represented above, so it
is diagnostic rather than an extra additive row. Likewise, the Nsys CUDA API sum shows pageable
`cuMemcpyDtoHAsync_v2` calls waiting for 399.1 ms in aggregate; that wait overlaps target GPU work
and corresponds to the engine's verify-wait phase, not 399 ms of independent overhead.

## Selected per-kernel counters

“Memory” is `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`; occupancy is
`sm__warps_active.avg.pct_of_peak_sustained_active`. Rows include every symbol at or above 0.10%
of round time plus the requested RoPE and sampling symbols. The complete 79-shape join is in
[`summary.json`](summary.json).

| Kernel | Shapes | ms/round | Round share | SM | Memory | Achieved occupancy | Suspicious? |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `qmatvec_nvfp4_mmvq_dual_b4_rpr2` | 1 | 8.343 | 27.43% | 49.5% | 86.8% | 54.3% | no |
| `qmatvec_nvfp4_mmvq_b4_rpr2w8` | 2 | 7.785 | 25.60% | 43.3% | 56.8% | 59.2% | yes |
| `fa_decode_f32` | 1 | 3.048 | 10.02% | 3.5% | 3.5% | 16.6% | yes |
| `qmatvec_nvfp4_mmvq_b4_rp` | 3 | 2.730 | 8.98% | 20.2% | 28.5% | 44.3% | yes |
| `qmatvec_q5_K_mmvq_b4_r2` | 1 | 1.206 | 3.97% | 70.0% | 85.3% | 56.3% | no |
| `qmatvec_nvfp4_mmvq_b4_rpr2` | 1 | 0.795 | 2.61% | 50.2% | 66.5% | 51.7% | yes |
| `qmatvec_q4_K_mmvq` | 4 | 0.714 | 2.35% | 44.7% | 75.3% | 85.3% | no |
| `add_rms_norm_q8_1` | 2 | 0.664 | 2.18% | 0.9% | 3.7% | 58.5% | yes |
| `gdn_scan_s128` | 1 | 0.633 | 2.08% | 63.4% | 63.4% | 80.7% | no |
| `qmatvec_nvfp4_mmvq_mr2_rp` | 1 | 0.336 | 1.10% | 48.9% | 90.1% | 76.0% | no |
| `qmatvec_q6_K_mmvq` | 2 | 0.314 | 1.03% | 56.5% | 73.5% | 66.4% | no |
| `rms_norm_f32` | 12 | 0.313 | 1.03% | 0.6% | 2.6% | 17.6% | yes |
| `l2_norm_f32` | 2 | 0.176 | 0.58% | 1.5% | 4.0% | 4.8% | yes |
| `silu_mul_scaled_q8_1` | 2 | 0.151 | 0.50% | 10.1% | 18.7% | 61.3% | yes |
| `ssm_conv1d_tm_state_f32` | 2 | 0.119 | 0.39% | 5.6% | 13.0% | 27.7% | yes |
| `qkv_to_gdn_repack_f32` | 2 | 0.107 | 0.35% | 5.8% | 11.6% | 58.6% | yes |
| `append_quantize_kv_q8_0_q5_1` | 1 | 0.105 | 0.34% | 0.3% | 5.8% | 2.1% | yes |
| `gated_rmsnorm_q8_1` | 2 | 0.090 | 0.30% | 6.7% | 6.9% | 18.8% | yes |
| `fa_decode_combine_f32` | 1 | 0.081 | 0.27% | 3.3% | 4.2% | 16.5% | yes |
| `sigmoid_f32` | 4 | 0.073 | 0.24% | 0.6% | 6.4% | 32.9% | yes |
| `ssm_conv_ring_update_f32` | 1 | 0.072 | 0.24% | 3.0% | 6.0% | 61.3% | yes |
| `gdn_glog_f32` | 1 | 0.063 | 0.21% | 0.1% | 7.3% | 27.2% | yes |
| `quantize_q8_1` | 10 | 0.057 | 0.19% | 3.2% | 6.2% | 63.0% | yes |
| `rope_neox_f32` | 7 | 0.054 | 0.18% | 0.9% | 6.0% | 3.7% | yes |
| `qmatvec_q4_K_mmvq_b4_r2` | 1 | 0.031 | 0.10% | 31.8% | 51.6% | 51.6% | yes |
| `argmax_partial_f32` | 1 | 0.019 | 0.06% | 13.0% | 13.0% | 30.5% | yes |
| `argmax_final_f32` | 1 | 0.011 | 0.04% | 0.1% | 6.6% | 13.1% | yes |

## Top-three limiters and tune candidates

The ranking collapses duplicate launch shapes by unique kernel symbol. Shape-level counters and
time weights remain separate in `summary.json`.

### 1. `qmatvec_nvfp4_mmvq_b4_rpr2w8`: exposed DRAM latency

The 640-block shape carries 5.816 ms/round: SM 41.6%, memory/DRAM 54.4%, occupancy 58.9% versus
66.7% theoretical, 0.98 waves/SM, and 6.97 long-scoreboard warps per issued instruction. The
1,280-block shape carries another 1.969 ms: SM 48.2%, memory/DRAM 63.9%, occupancy 60.2%, 1.95
waves/SM, and 6.65 long-scoreboard stalls. Both report zero bank conflicts. By contrast, the
2,176x2 dual qmatvec reaches 86.8% memory SOL in the same counter window. The limiter is not peak
bandwidth in the abstract; it is too little independent work to hide the weight-load latency in
these smaller row-paired grids.

**Tune candidate:** prototype an eight-resident async-prefetch/scale-pipeline twin for precisely
the 640/1,280 grids, preserving the current rpr2w8 row mapping and reduction order. Do not reuse a
variant that drops the eighth resident block and restores the straggler wave.

**Expected ceiling:** reaching the 70% non-suspicious threshold at fixed work saves **1.47
ms/round (4.83%)**. Matching the measured 86.8% sibling is the stretch ceiling: **2.69 ms (8.84%)**.

### 2. `fa_decode_f32`: occupancy/launch geometry

This kernel launches only 24 blocks on an 82-SM GPU. NCU reports 0.05 waves/SM, 16.6% achieved
occupancy versus 100% theoretical, SM/memory 3.5%, DRAM 0.27%, and zero bank conflicts. Long
scoreboard (11.28) and short scoreboard (2.61) are exposed because most SMs have no block; this is
not a DRAM-bandwidth ceiling.

**Tune candidate:** sweep an early-context, context-aware split-K ladder that emits four
partitions per head (96 blocks) and uses the existing deterministic combine contract. Gate every
candidate with the standing argmax and K=1..8 self-consistency tests because changing the key
partition changes FP reduction order.

**Expected ceiling:** perfect 24-to-82 block scaling before extra split/combine cost is **2.16
ms/round (7.09%)**. A practical first target is 1.0-1.5 ms/round; the ideal bound is not a promise.

### 3. `qmatvec_nvfp4_mmvq_b4_rp`: mixed launch occupancy and DRAM latency

The symbol has three materially different shapes:

| Grid | ms/round | SM | Memory/DRAM | Occupancy | Waves/SM | Primary limiter |
|---:|---:|---:|---:|---:|---:|---|
| 12 | 1.041 | 0.6% | 1.3% | 8.3% | 0.01 | launch/occupancy |
| 256 | 0.367 | 12.8% | 22.4% | 25.9% | 0.31 | occupancy |
| 1,536 | 1.322 | 37.7% | 51.5% | 77.8% | 1.87 | DRAM latency |

All three have zero bank conflicts. The 12-block shape fires 2,016 times in the window (96 per
round), so it is a real accumulated tax despite each launch being small.

**Tune candidate:** group the independent 12/256-block auxiliary projections into a multi-tensor
launch so one grid fills the GPU, then test a pipelined 1,536-block path separately. Keep the
projection math and per-output reduction order unchanged.

**Expected ceiling:** halving the two underfilled-shape time and bringing the 1,536-block shape to
70% memory SOL saves **1.05 ms/round (3.46%)**. Eliminating all small-shape time plus the same 70%
large-shape target is the hard bound: **1.76 ms (5.78%)**.

## Secondary findings

- The own-trim drafter head is already at 90.1% memory SOL and costs only 0.336 ms/round. It is not
  a first-order kernel target.
- The target Q5_K head reaches 85.3% memory SOL; it is also not suspicious.
- `gdn_scan_s128` clears the SM threshold at 63.4% and costs 0.633 ms/round. Its MIO/short-scoreboard
  stalls are worth retaining as a later candidate, but it ranks below the three requested finds.
- RMS/RoPE glue totals 1.125 ms/round. `add_rms_norm_q8_1` alone is 0.664 ms and launch-underfilled,
  while all RoPE shapes together are only 0.054 ms. A future glue-fusion pass has a small ceiling;
  RoPE alone does not.
- GPU sampling is 0.030 ms/round. Even deleting both argmax kernels would save under 0.10% of the
  round.
- `fa_decode_vec_q_rows_v4` reports 65.8% shared-bank conflicts, but it contributes 0.010 ms/round
  (0.034%). Fixing it before the three carriers above would be measurement-inverted prioritization.

## Evidence and reproducibility

- Capture harness: [`capture.sh`](capture.sh)
- Deterministic join/aggregation: [`analyze.py`](analyze.py)
- Machine-readable full join: [`summary.json`](summary.json)
- Timing log and raw exports:
  [`nsys-spec-k3-n64.log`](raw/nsys-spec-k3-n64.log),
  [`kernel summary`](raw/nsys-spec-k3-n64-kern-sum.csv),
  [`full GPU trace`](raw/nsys-spec-k3-n64-trace.csv),
  [`CUDA API summary`](raw/nsys-spec-k3-n64-api-sum.csv), and
  [`memory-operation summary`](raw/nsys-spec-k3-n64-mem-sum.csv).
- Counter log and raw exports:
  [`ncu-spec-k3-n64.log`](raw/ncu-spec-k3-n64.log),
  [`raw metrics CSV`](raw/ncu-spec-k3-n64-raw.csv),
  [`details export`](raw/ncu-spec-k3-n64-details.txt), and
  [`selected-kernel regex`](raw/ncu-kernel-regex.txt).
- Baseline metadata: [`environment.log`](raw/environment.log).
- Bounded release build receipt: [`build-run-spec.log`](raw/build-run-spec.log).

No engine or kernel source changed. This is lane-local research evidence, not a published perf
board move; README, `docs/PERFORMANCE.md`, runtime defaults, tags, and remotes are untouched.
