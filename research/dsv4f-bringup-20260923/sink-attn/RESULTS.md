# DSv4 two-launch sink attention (memra #683)

Status: in progress. The kernel-boundary bit gate and the component timing on one RTX PRO 6000
Blackwell and on the local RTX 5090 are banked here. The served A/B on 2x RTX PRO 6000 is queued;
this file gets its rows when it lands.

Scope: `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`, device
f32x chains, sink attention at 64 heads x 512 (main attention) and 32 x 512 geometry. Lane
`lane/dsv4-sink-attn-20260923`, base main `649d96210`.

## What changed

The f32x sink attention was three launches per layer, plus a q transpose on the batched path:
scores (`dsv4_sink_scores_mq_f32acc_kernel`, q `[nq][hd][heads]`), soft (max, `ev`, den into a
workspace) and out. It is now two launches of `memra_dsv4_sink_attn_st_f32acc`:

- `dsv4_sink_scores_st_f32acc_kernel`: one 64-thread block per (8-slot tile, 8-head tile, query).
  The block stages its q rows and its selected kv rows once by cp.async; each thread is one score,
  a single f32 accumulator over `x` ascending, then `* scale`, `-INF` for a `-1` slot.
- `dsv4_sink_softout_st_f32acc_kernel`: one 256-thread block per (16-column tile, 16-head tile,
  query). Max (`fmaxf`, floored at `-1e30`), `ev = expf(s - m)` or 0 for `-INF`, den summed in
  ascending slot order then `+ expf(sink - m)`, and `o = (ascending sum of ev * kv over ev != 0) /
  den` over 128-slot kv tiles, with no workspace round trip.

Replay (graph) launches read the position on device: live slots are
`win + (ratio ? min((pos + 1) / ratio, topk) : 0)`, the score row stride is the live count, and
grid blocks past it exit.

Prior art: the tile and grid split follow FlashInfer's `sparse_mla_sm120` decode kernels (a score
grid whose operands one CTA stages once, and a soft/out CTA that finishes in shared memory).
FlashInfer splits the slots across CTAs and merges partial softmaxes in the exp2 domain; both
change the order of the sums, so neither is taken. The numeric program is the three-kernel one.

`MEMRA_DSV4_SINK_SCORE` and the 2026-09-06 tiled scorer are deleted (`docs/FLAGS.md`, "Removed
doors, 2026-09-23 (the tiled sink scorer ...)").

## Correctness

`crates/memra-engine/tests/dsv4_sink_attn_st_gpu.rs` (ignored GPU tests) compares every output
bit of the two-launch program against the three former entry points on the same inputs:

| test | coverage |
|---|---|
| `sink_attn_st_matches_three_kernel_batched` | 208 cases: shapes 64x512, 32x512, 16x256, 48x64; nq 1 and 3; slots 1, 7, 8, 9, 127, 128, 129, 187, 255, 256, 257, 640, 1100; magnitudes (-2, 1) and (-1, 4); 0% or 30% padded indices, one all-pad query at nq 3 |
| `sink_attn_st_matches_three_kernel_replay` | 86 cases against `memra_dsv4_replay_attention`: heads 64 and 32, (ratio, topk) in (0, max), (4, 512) at limits 1024 and 4096, (128, max) at 65536, positions across each boundary |
| `sink_attn_st_matches_three_kernel_single_query` | 26 cases against the eager single-query entry |
| `sink_attn_st_red_arm` | a 2^-10 change to one live kv element is caught |

- One RTX PRO 6000 Blackwell Workstation Edition, 600 W limit, committed tree `2f8e23c22`, test
  binary `5eae971f380b3417` (`raw/pro6000/summary`, `raw/pro6000/gate-r{1,2,3}.log`, script
  `raw/pro6000/sink-pro.sh`): **5 passed in each of 3 runs, 320 cases bit-identical, red arm
  caught.**
- Local RTX 5090 Laptop (`raw/rtx5090/gate-worktree.log`, the working tree before the kernels
  were committed, st kernels as committed): 5 passed, 320 cases bit-identical, red arm caught.

## Component timing

`sink_attn_st_timing`: one layer's sink attention on the graph replay entry, old program (q
transpose plus the three kernels) against the two launches, 2000 launches per rep after 64 warmup
launches, us per layer.

RTX PRO 6000, 3 runs x 5 reps, medians of 15 (two-launch range across the 15 in brackets):

| shape | old | two-launch | change |
|---|---|---|---|
| 64 heads, ratio 0, 128 live | 26.578 | 8.639 (8.601..8.641) | -67.5% |
| 64 heads, ratio 4, 187 live | 32.250 | 11.753 (11.720..11.760) | -63.6% |
| 64 heads, ratio 4, 640 live | 65.260 | 28.785 (28.776..28.816) | -55.9% |
| 32 heads, ratio 0, 128 live | 26.115 | 8.355 (8.353..8.357) | -68.0% |
| 32 heads, ratio 4, 187 live | 31.176 | 11.006 (11.002..11.012) | -64.7% |
| 32 heads, ratio 4, 640 live | 64.600 | 23.777 (23.771..23.783) | -63.2% |

Local RTX 5090 Laptop, 5 reps, medians of 5:

| shape | old | two-launch | change |
|---|---|---|---|
| 64 heads, ratio 0, 128 live | 27.235 | 11.930 | -56.2% |
| 64 heads, ratio 4, 187 live | 31.960 | 19.768 | -38.1% |
| 64 heads, ratio 4, 640 live | 70.396 | 55.262 | -21.5% |
| 32 heads, ratio 0, 128 live | 25.806 | 8.787 | -65.9% |
| 32 heads, ratio 4, 187 live | 30.489 | 11.759 | -61.4% |
| 32 heads, ratio 4, 640 live | 65.425 | 30.342 | -53.6% |

These are isolated launches in a loop. They bound the change per layer; they are not the served
gain.

## Pending

- `dsv4-gpu-dspark-gate --served` on the lane binary.
- Served plain and DSpark A/B on 2x RTX PRO 6000, one boot per row, order `l b b l l b b l l b`,
  N=5 per arm, the fixed prompt hashes of `../latency/RESULTS.md`.
