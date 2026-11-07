# Lever ledger: GB10 prime GDN, activation quantize and glue, 2026-11-07

One row per arm: config, hardware, n, baseline, candidate, commit and receipt. Model: the owner
GGUF Qwen3.8-27B (NVFP4 + Q5_K) with the masked MTP draft. Hardware: DGX Spark unit 2 (GB10,
sm_121a, 48 SMs), clock cap 2,200 MHz. Base: main `9805016bb1`.

Conventions:

- **Prime bench:** `run-gen` with `MEMRA_PP_ONLY=1 MEMRA_PP_PRIME=1` (the served prime path,
  4,096-row chunks) on the 1,240-record prompt (33,890 tokens), one logits pass then 3 timed primes
  per process, median. One process per cell, cells interleaved in the order written. Arms of one
  binary are selected by the lane's seams; M is the main binary.
- **Exactness:** every arm is exact by construction (same operands, same operation order per
  output element). Gates: the prime-path last-row logits byte compare (sha256 per cell), the
  kernel-check bit cells named per row, and the full battery on the PR head.

| # | lever | config / flags | n | baseline | candidate | sha | receipt | verdict |
|---:|---|---|---|---|---|---|---|---|
| 0 | baseline profile (main) | defaults; one cold 131,051-token served prefill under nsys | 1 | | TTFT 157.553 s; kernels 156.114 s: F8F4 GEMM 59.72, attention 40.76, GDN chunk scan 13.85, e4m3 quantize 9.12, glue 8.63, GDN conv/l2/gates 6.15, GEMM image 5.91, dp4a/matvec 4.98 | `9805016bb1` | profile-main-131k/ | reference |
| 1 | GDN compact q/k at a 3:1 value:key head ratio (16 key / 48 value heads were broadcast to 48: the predicate wanted num_v == 2 num_k) | lane arm D (`MEMRA_GDN_HFAST=0 MEMRA_GDN_W32=1` on `48265176f0`) | 2 cells, both orders | B 32.277, 32.301 s (M 32.196, 32.189) | 31.357, 31.323 s (-2.9%); logits `758739be` in every cell | `48265176f0` | ab-gdn1/ | **adopted** (`MEMRA_GDN_TRAFFIC`) |
| 2 | #1 + no f32 W store from K3 on the mma path (only W's bf16 twin has a reader) | lane arm W | 2 | 31.357, 31.323 s | 31.092, 31.126 s (-0.7%) | `48265176f0` | ab-gdn1/ | **adopted** (same seam) |
| 3 | #2 + heads-fastest grid for K2/K3/K5 (the value heads of one key head share its q/k rows in L2) | lane arm F | 2 | 31.092, 31.126 s | 31.062, 30.715 s (-0.7%) | `48265176f0` | ab-gdn1/ | **adopted** (same seam) |
| 4 | one `add_rms_norm` kernel at both residual sites of the non-f16 eager prime (was `add` then `rms_norm`) | arm G1 (`MEMRA_PRIME_NORM_FUSE=1` on `c02f90930d`, the research build's level seam) | 2 | G 30.633, 30.705 s | 30.406, 30.399 s (-0.9%) | `c02f90930d` | ab-gdn2/ | **adopted** (`MEMRA_PRIME_NORM_FUSE`) |
| 5 | #4 + the norm kernel writes the F8F4 v2 activation scratch and seeds the next quantize-once scope; the dense FFN's f32 norm output is not written | arm G3 | 2 | 30.406, 30.399 s | 29.794, 29.850 s (-1.9%) | `c02f90930d` | ab-gdn2/ | **adopted** (same seam) |
| 6 | q8_1 quantize-once in GEMM groups (beta and alpha shared one activation, quantized twice), the pair seeded by the norm before a GDN mixer | arm GQ | 2 | 29.794, 29.850 s | 29.606, 29.553 s (-0.8%) | `c02f90930d` | ab-gdn2/ | **adopted** (no seam: the same bytes, the F8F4 quantize-once precedent) |
| 7 | token-tiled `qmatvec_nvfp4_dp4a_rp_t4`: one block per four tokens over every output row (was one block per (row, token)) | arm A | 2 | 29.606, 29.553 s | 29.098, 29.140 s (-1.6%) | `c02f90930d` | ab-gdn2/ | **adopted** (`MEMRA_DP4A_TILED`) |
| 8 | K4 rewrites (probe): 64 state columns per CTA (one wave) 1.14x; register prefetch of U / gate / beta 0.94x; K5 heads fastest 1.20x | `probe/probe_gdn.cu`, T 4,096 and 8,192 | 7 launches per cell | K4 2.429, K5 2.135 ms | see README | probe | ab-gdn2/probe-gdn.log | K5 grid adopted (#3); the K4 arms are not built: the stores (1.1 ms) bound K4, the fused K4+K5 is next |
| 9 | the GDN out-norm writes the ssm_out GEMM's F8F4 v2 activation (no f32 output); a single-weight quantize-once scope adopts it | arm GU0 on unit 1: `ebe9cf2a8e` with `MEMRA_FFN_GATEUP_FUSE=0`, against M = `5650c30d03` (PR 1) | 2, both orders | M 29.112, 29.162 s | 28.783, 28.823 s (-1.2%; this build also wrote the norm's q8_1 pair without adopting it, fixed in `ffba9c1669`) | `ebe9cf2a8e` | ab-unit1-stack/ | **adopted** (`MEMRA_PRIME_NORM_FUSE`) |
| 10 | the dense FFN's gate and up in one F8F4 v2 launch over a paired weight image, writing silu(gate) * up as the down GEMM's v2 activation (the two [T, 17,408] f32 outputs and the silu quantize pass are gone) | arm A on unit 1 (`ebe9cf2a8e` defaults) | 2 | GU0 28.783, 28.823 s | 26.979, 26.993 s (-6.3%) | `ebe9cf2a8e` | ab-unit1-stack/ | **adopted** (`MEMRA_FFN_GATEUP_FUSE`) |
| 11 | fused K4+K5: the GDN state pass computes the chunk output, so Y and the chunk-start states never reach memory (`gdn_chunk_state_output_mma`); probe 4.288 -> 1.703 ms per call at T 4,096 | unit 1, `ffba9c1669`: K0 = `MEMRA_GDN_K45=0`, A = defaults (with the q8_1 seed adoption fix) | 2, both orders | K0 26.765, 26.823 s | 25.709, 25.697 s (-4.0%) | `ffba9c1669` | ab-unit1-k45/, probe-runs/ | **adopted** (`MEMRA_GDN_K45`, GB10-keyed) |
| 12 | the full set, served: cold TTFT, fresh boot per cell, B L L B B L, unit 2 | defaults; B = main `6f5e03e74a`, L = `189a887d27` | 3 per arm per length | 31.848, 31.887, 31.895 / 158.502, 157.694, 157.706 / 392.298, 391.092, 391.238 s | 25.361, 25.394, 25.382 / 131.568, 131.689, 132.558 / 340.886, 340.477, 340.951 s (-20.40% / -16.50% / -12.87%, medians); answers identical | `189a887d27` | served-unit2-final/ | **adopted** |
| 13 | #12 confirmed on unit 1: B L L B | the same binaries | 2 per arm per length | 31.900, 32.013 / 158.173, 158.250 / 393.122, 394.167 s | 25.420, 25.421 / 132.264, 132.260 / 342.529, 343.602 s (-20.45% / -16.40% / -12.85%); the same answers as unit 2 | `189a887d27` | served-unit1-final/ | confirmed |
