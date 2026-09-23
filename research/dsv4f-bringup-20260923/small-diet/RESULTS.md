# DSv4 small-kernel diet on the served PP-2 program (memra #339): the door becomes the code

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Workstation Edition (the healthy pair of `../REBASELINE.md`), 500 W
limit, 2026-09-23. Served program: PP-2, matrix expert program, host sampler, memra-server
otherwise naked. One scored campaign at a time under `/tmp/memra-gpu.lock`, 250 ms telemetry per
row. Lane `lane/dsv4-small-diet-20260923`, commit `8d423ced3` on main `d544c6b82`.

## What changed

The diet (#339, 2026-09-07) fuses two small chains on the plain t=1 step:

- HC finish: `rowsq_scale_f32acc` + `hc_sinkhorn_m` + `hc_collapse` becomes
  `dsv4_small_hc_f32_fixed_order_kernel` (one 128-thread CTA, the same rowsq tree, register
  Sinkhorn with ascending sums and the configured 20 iterations, ascending collapse).
- Q-LoRA norm/pack: in-place `rmsnorm_f32acc` + `cvt_bf16` becomes
  `dsv4_small_norm_pack_f32_fixed_order_kernel` (the same 128-thread tree, the normalized f32 row
  kept, bf16 RNE).

It landed as `MEMRA_DSV4_SMALL_KERNEL_DIET`, default 0, admitted only on TP/EP, with a +4.09%
sampled ABBA receipt there (`../../dsv4f-small-kernel-diet-20260907/README.md`). Its decide-by
(2026-09-21) passed with no decision. PP-2, the served program, never took it: on that program the
served plain profile spends 0.69 ms per step per card in 44 one-warp Sinkhorn launches, 0.73 ms in
94 rmsnorm launches, 0.13 ms in rowsq and 0.14 ms in 165 bf16 packs
(`../m1-stream-664/raw/prof/served-plain-m1/ana.txt`).

This lane removes the door. `Dsv4Gpu::small_kernel_diet_shape` (device decode path, f32x chains,
HC4, hidden 4096) decides it at load, on PP-2 and TP/EP alike. Multi-row verify and prefill rows
keep the unfused kernels. The fused kernels match them bit for bit, so a request that alternates
plain steps and verify transactions stays one numeric program.

## Kernel boundary

`crates/memra-engine/tests/dsv4_small_diet_gpu.rs`, command in `docs/TESTING.md`.

| rig | HC finish | Q norm/pack | log |
|---|---|---|---|
| local RTX 5090 | `EXACT cases=96 outputs=5 red_arms=3` | `EXACT cases=128 outputs=2 red_arms=1` | `raw/component-5090.log` |
| RTX PRO 6000 pair, card 0 | `EXACT cases=96 outputs=5 red_arms=3` | `EXACT cases=128 outputs=2 red_arms=1` | `raw/component/gate.log`, test binary `a685e51b...` |

## Served A/B

Lane binary `80de7038...` (tree `8d423ced3` plus `raw/diet-lane.patch`, a lane-only
`MEMRA_DSV4_DIET_AB` read that was never merged: `0` keeps the unfused kernels, unset is the
shipped code). One boot per row, order `on off off on on off off on on off`, cells
`raw/ab-cells.txt` (a 32-token warmup, then greedy c1 on 8 prompts, 256 max tokens, the fixed
prompts of `../raw/bench.py`). Queue `raw/q-diet.sh`, summary `raw/q-diet.summary`, receipts
`raw/ab/r<N>-<arm>/` (cells.jsonl, controller.log, serve.log, /metrics, /healthz, /readyz, 250 ms
telemetry, binary sha256, env). The tree predates the one-token MoE stream visitor (#672), so the
unfused arm is the 38.8 tok/s program of `../m1-stream-664/RESULTS.md`.

| row | arm | decode tok/s (1/TPOT p50) | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | agg tok/s |
|---|---|---|---|---|---|---|---|
| r1 | diet | 40.12 | 24.93 / 25.13 / 25.20 | 24.81 / 25.85 | 206 / 225 | 6,567 | 39.08 |
| r2 | unfused | 38.77 | 25.79 / 25.98 / 26.06 | 25.66 / 26.65 | 206 / 224 | 6,783 | 37.82 |
| r3 | unfused | 38.79 | 25.78 / 25.98 / 26.05 | 25.65 / 26.65 | 207 / 223 | 6,784 | 37.82 |
| r4 | diet | 40.08 | 24.95 / 25.14 / 25.22 | 24.82 / 25.83 | 205 / 222 | 6,569 | 39.07 |
| r5 | diet | 40.08 | 24.95 / 25.13 / 25.20 | 24.82 / 25.79 | 205 / 223 | 6,569 | 39.07 |
| r6 | unfused | 38.78 | 25.79 / 25.99 / 26.06 | 25.66 / 26.67 | 206 / 223 | 6,784 | 37.82 |
| r7 | unfused | 38.82 | 25.76 / 25.98 / 26.05 | 25.63 / 26.68 | 206 / 224 | 6,775 | 37.85 |
| r8 | diet | 40.11 | 24.93 / 25.13 / 25.21 | 24.81 / 25.86 | 205 / 222 | 6,563 | 39.10 |
| r9 | diet | 40.11 | 24.93 / 25.14 / 25.23 | 24.80 / 25.86 | 205 / 222 | 6,561 | 39.09 |
| r10 | unfused | 38.78 | 25.78 / 26.00 / 26.07 | 25.66 / 26.66 | 206 / 224 | 6,785 | 37.82 |

**Diet 40.11 tok/s median (N=5, 40.08..40.12) against unfused 38.78 (N=5, 38.77..38.82), +3.4%,
TPOT p50 25.78 -> 24.93 ms (-0.85 ms per token).** The arms are disjoint: a 1.3 tok/s gap
against a within-arm spread under 0.05 tok/s. TTFT does not move (205..207 ms in both arms): the
prime's rows are multi-row and keep the unfused kernels.

**Identity.** All ten rows returned the same text on all 8 prompts: `aea6e69e 26ac8df7 850f75ed
a7784b9a 7568f9b3 9dd1aedd 4fd72390 937f04d8`, warmup `e78fb458`, the plain hashes of
`../REBASELINE.md`.

**Thermal regime.** Pooled over both cards, samples at 50% utilization or more (a plain step keeps
each card under 50% most of the time, so these are 5..20 samples per row): power median 176..230 W,
peak 246 W against the 500 W limit, SM clock 2595..2850 MHz, GPU temperature at most 62 C.

## DSpark on top

Same lane binary with `MEMRA_DSV4_DRAFTER=dspark`, order `on off off on`, cells
`../mrow-stream/raw/cells-spec.txt` (warmup, greedy c1 on 8 prompts, sampled c1 on 8 prompts,
greedy c1 ignore-eos on 4 prompts), receipts `raw/spec/s<N>-<arm>/`.

| row | arm | greedy c1 tok/s | TPOT p50 ms | sampled c1 tok/s | ignore-eos tok/s | TTFT p50 ms |
|---|---|---|---|---|---|---|
| s1 | diet | 55.97 | 17.87 | 47.36 | 55.77 | 267 |
| s2 | unfused | 56.04 | 17.84 | 47.43 | 55.89 | 267 |
| s3 | unfused | 56.00 | 17.86 | 47.43 | 55.90 | 267 |
| s4 | diet | 55.96 | 17.87 | 47.39 | 55.89 | 267 |

**Flat: diet 55.97 against 56.02 greedy (-0.1%), 47.38 against 47.43 sampled (-0.1%), N=2 each.**
A DSpark server runs its target on T=6 verify rows, which keep the unfused kernels by design, so
the diet has almost no one-token steps to take there. The 0.05 tok/s gap is under the size this
N=2 can resolve; the lane does not claim a cause for it. All four rows returned the plain greedy
hashes above and the sampled hashes `53944095 73fe7f91 6329ee94 827c0b56 437a4427 07de3a97
8a59518d b424bd80` of `../mrow-stream/RESULTS.md`, so spec output still equals plain output.
Power median 266..268 W, peak 299 W, SM clock 2730..2850 MHz, at most 66 C.

## Identity gates

- Kernel boundary: the two component tests above, EXACT on both rigs.
- `dsv4-gpu-dspark-gate --served` (`MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DECODE_PATH=device`, the
  shipped diet dispatch, binary `63564fb6...`): PASS (`raw/dspark-served/gate.log`).
