# Lever ledger: giving the GB10 h2p prime attention its registers back, 2026-11-08

One row per arm tried: what changed, hardware, shape, n, baseline, result, commit and receipt.
Continues `research/gb10-prefill-attn-20261107/LEVER-LEDGER.md` (its rows A0-A23).

Conventions:

- **Probe:** `probe/probe3.cu` (sha256 `5abf338a`) with `probe/attn3.cuh` (`207f61f5`) and
  `probe/attn3b.cuh` (`888826fe`), built with nvcc 13.0 on the unit (binary `42b1a93c`), main's
  `flash_attn.cu` (`e759f4ac`, main `7781d4ed66`) included verbatim. DGX Spark unit b23d (GB10,
  sm_121a, 48 SMs, driver 580.178.04). Qwen3.8-27B attention shape (24 / 4 heads x 256, causal, f32
  Q, BF16 K/V). T = 4,096 rows; every cell is a median of 3 after one warmup; three rounds, round 2
  in reverse order; the table value is the median over rounds. Baseline h2p = the shipped
  `fa_prefill_qw_h2p` in the same rounds.
- Every arm is exact: 0 mismatching and 0 non-finite floats against h2p at every depth.
- Registers / stack / loop spills / loop NOPs are ptxas 13.0's static counts (`ptxas/`).
- The commit is this lane's research commit (the lane directory); the dump hook is
  `6c739dc807` on `research/gb10-h2p-regs-dump` (`probe/research-dump-hook.patch`).

| # | arm | change | regs / stack | baseline h2p at 131k | result at 131k (TFLOPS, ms) | 4k / 16k / 32k / 64k | receipt | verdict |
|---:|---|---|---|---|---|---|---|---|
| R0 | h2p | shipped (control) | 255 / 8 | | 85.00, 152.805 ms (152.571-152.889) | 62.43 / 78.45 / 82.44 / 84.03 | probe-r1 | control |
| R1 | h2 | main's previous kernel (control) | 255 / 8 | 85.00 | 67.70 (-20.4%) | -16.2 / -18.7 / -19.8 / -19.9% | probe-r1 | control |
| R2 | r32e | K3V2 ring (3 K + 2 V planes, per-plane barriers, mid-tile V wait), Q in registers | 255 / 48 | 85.00 | 72.99 (-14.1%), 177.943 ms | -9.6 / -13.1 / -13.7 / -13.9% | probe-r1 | rejected |
| R3 | qa12e | (a) on K3V2: Q k-tiles 12..15 in shared memory | 255 / 80 | 85.00 | 72.43 (-14.8%), 179.307 ms | -9.6 / -13.5 / -14.2 / -14.4% | probe-r1 | rejected |
| R4 | rb | K2V3 ring (2 K + 3 V planes, one full barrier per tile, waits at the top), Q in registers | 255 / 24 | 85.00 | 81.19 (-4.5%), 159.975 ms | -3.5 / -4.1 / -4.5 / -4.1% | probe-r1 | rejected (the 80 KB ring both levers need) |
| R5 | qb12 | (a): rb + Q k-tiles 12..15 as per-lane fragments in shared memory (16 registers, 0 spill) | 255 / 0 | 85.00 | 78.77 (-7.3%), 164.885 ms | -7.3 / -6.9 / -7.8 / -7.4% | probe-r1 | **rejected** |
| R6 | qb13 | rb + Q k-tiles 13..15 in shared memory | 255 / 0 | 85.00 | 78.27 (-7.9%), 165.943 ms | -0.3 / -2.6 / -6.0 / -7.3% | probe-r1 | rejected |
| R7 | qb14 | rb + Q k-tiles 14..15 in shared memory (best (a): +0.8% against rb) | 255 / 16 | 85.00 | 81.84 (-3.7%), 158.703 ms | -3.3 / -3.6 / -4.1 / -3.4% | probe-r1 | rejected |
| R8 | pw0e | (b) on K3V2, no skew | 168 / 48 | 85.00 | 58.42 (-31.3%), 222.306 ms | -29.1 / -30.7 / -31.3 / -31.5% | probe-r1 | rejected |
| R9 | pw1q15e | (b) on K3V2, groups one QK apart, Q k-tile 15 in shared memory | 168 / 56 | 85.00 | 56.25 (-33.8%), 230.892 ms | -30.8 / -33.0 / -32.9 / -33.9% | probe-r1 | rejected |
| R10 | pb0 | (b): warp-pair split, 12 warps (2 heads x 3 pairs x 16 rows, BQ 48), K2V3, no skew | 168 / 16 | 85.00 | 64.80 (-23.8%), 200.441 ms | -30.3 / -21.9 / -23.0 / -23.7% | probe-r1 | rejected |
| R11 | pb0q15 | pb0 + Q k-tile 15 in shared memory (best (b)) | 168 / 8 | 85.00 | 65.06 (-23.5%), 199.642 ms | -30.1 / -22.3 / -23.1 / -23.0% | probe-r1 | **rejected** |
| R12 | pb1 | pb0 + warp groups released one QK apart | 168 / 32 | 85.00 | 60.23 (-29.1%), 215.644 ms | -33.6 / -27.9 / -28.5 / -28.3% | probe-r1 | rejected |
| R13 | pb1q15 | pb1 + Q k-tile 15 in shared memory | 168 / 32 | 85.00 | 62.91 (-26.0%), 206.461 ms | -31.3 / -25.0 / -25.3 / -25.7% | probe-r1 | rejected |
| R14 | pb2 | pb0 + group 1 after group 0's QK(0), group 2 after its softmax(0) | 168 / 32 | 85.00 | 61.00 (-28.2%), 212.906 ms | -33.2 / -25.9 / -27.8 / -27.8% | probe-r1 | rejected |
| R15 | xh2p | h2p's 13.0 PTX assembled by ptxas 13.4 (3 loop NOPs instead of 70; same 255 registers) | 255 / 8 | 85.14 | 77.32 (-9.2%), 167.981 ms | -2.9 / -6.8 / -8.1 / -8.5% | probe-x134 | rejected: 13.0's NOP pacing is faster on GB10 |
| R16 | xrb / xqb12 / xpb0q15 | the K2V3 arms assembled by ptxas 13.4 | | 85.14 | 81.04 (-4.8%) / 74.19 (-12.9%) / 64.70 (-24.0%) | | probe-x134 | rejected |
| R17 | r32w, qa12w | explicit four-accumulator QK loop | 255 | | ptxas 13.0 emits the same loop as r32 / qa12 (1,037 against 1,038, 1,002 against 1,003 instructions): ptxas already interleaves both QK halves; not timed | | ptxas/sass-loop-census.txt | no-op |
