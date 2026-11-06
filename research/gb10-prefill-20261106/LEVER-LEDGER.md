# Lever ledger: GB10 long-context prefill, 2026-11-06

One row per lever tried: config, hardware, n, baseline, candidate, commit and receipt. Model: the
owner GGUF Qwen3.8-27B (NVFP4 + Q5_K) with the masked MTP draft, served at 262,144 context.

Hardware: one DGX Spark (GB10, sm_121a, 48 SMs), SM clock 2,177 MHz under load.

Conventions:

- **Served:** cold TTFT of one fresh prompt per target per boot (`harness/pf_long_memra.py`),
  fresh boot per cell, cells interleaved in the order written (A B C C B A = both orders).
- **GEMM bench:** `run-gen` prefill-only, one ~4.1k-token record prompt, 1 warmup + 5 timed
  monolithic forwards per process, median; last-row logits dumped and compared byte for byte.
- **Exactness:** every lever is exact by construction (same operands, same operation order); the
  gate is the served text fingerprint (sha256 per target/rep identical across arms), the logits
  byte-compare for GEMM levers, and kernel-check bit-identity cells for touched kernels.

Rejected arms keep their commits on `research/*` branches.

| # | lever | config / flags | n | baseline | candidate | sha | receipt | verdict |
|---:|---|---|---|---|---|---|---|---|
| 0 | baseline (main) | none; the db prime attention twin, 4,096-token chunks | 3/2/1 served | | 32k 44.289, 131k 239.104, 260k 636.901 s | 5d70118b99 | baseline-served/ | reference |
| 1 | three-plane prime attention (t3, existing door) | `MEMRA_PRIME_KV_T3=1` | 2+2 boots, both orders | 32k 44.215, 131k 238.971 s | 42.362 / 209.981 s (-4.19% / -12.13%), text identical | door on main; run on b795c0a604 | attention-ab/ | exact win, superseded on GB10 by #2 |
| 2 | two-head prime attention (h2): two Q heads of one KV head per 8-warp CTA, double-buffered K/V, register P | `MEMRA_PRIME_KV_H2=1`; GB10 default in 942e26ad8d | 2+2 boots, both orders | 44.215 / 238.971 s | 42.113 / 205.702 s (-4.75% / -13.92%), text identical | b795c0a604, af5f356c97, 942e26ad8d | attention-ab/ | **adopted** (GB10 default) |
| 3 | F8F4 tile: B fragments for all four k32 steps by two ldmatrix.x4, per-token scales by one 16-byte load, 16-byte dequant stores | build (no flag) | GEMM bench 2+2, both orders | 4.2286 s (911.7 tok/s, pp3855) | 5.8566 s (658.2 tok/s), +38.5% slower; logits byte-identical | 82e583e842 (research/gb10-prefill-mmqld) | gemm-ab/ | **rejected**: register cap 254 with 72-112 B spill per thread (base 187-200, no stack) |
| 4 | #3 + one ratio LUT per NVFP4 scale pair (stacked) | build | 2+2 | 5.8566 s (#3) | 5.8082 s (-0.8% vs #3); logits byte-identical | 5822e33b71 (research/gb10-prefill-mmqdq) | gemm-ab/ | measured only as a step; see #5 for the trim on main |
| 5 | one ratio LUT per NVFP4 scale pair, alone on main | build | GEMM bench 2+2, both orders | 4.2058 / 4.2228 s (median 4.2143) | 4.2076 / 4.2106 s (median 4.2091, -0.12%); logits byte-identical | 08d1324b4f (research/gb10-prefill-dq) | gemm-ab-2/ | **rejected** (flat): the dequant ALU is not what limits the tile |
| 6 | F8F4 tile: one ldmatrix.x2 per k32 step for B, 16-byte dequant stores (register-neutral intent) | build | 2+2 | 4.2143 s (median) | 5.3381 / 5.3364 s (+26.6%); logits byte-identical | b5c2b6a5bd (research/gb10-prefill-ldx2) | gemm-ab-2/ | **rejected**: still at the cap, 255 registers and 136-152 B spill per thread |
| 7 | build the f16 activation operand only when an f16 mirror can take it (dead conversion on this GGUF) | code (no flag) | 2+2 boots, both orders, unit 2 with its own base | 32k 44.156, 131k 238.797 s | 43.373 / 234.527 s (-1.77% / -1.79%), text identical | b5eee93d96 | copies-ab-unit2/ | **adopted** |
| 8 | #7 + dense FFN GEMM outputs fed forward without slab copies (eager path; the captured segment keeps its copy) | code (no flag) | 2+2, both orders, unit 2 | 44.156 / 238.797 s | 40.157 / 222.706 s (-9.05% / -6.74%; -7.41% / -5.04% vs #7), text identical | 022f93ce30 | copies-ab-unit2/ | **adopted** |
| 9 | final stack: #2 (GB10 default) + #7 + #8 | none (defaults) | B S S B, unit 1, 32k/131k/260k per boot | 44.263 / 239.010 / 637.524 s | 38.109 / 189.728 / 475.889 s (-13.90% / -20.62% / -25.35%), text identical, MemAvailable floor unchanged (66.8 GiB) | 5fd85d3513 | final-ab/, profile-stack2-128k/, gates-stack2-unit1/, gates-stack2-unit2/ | **adopted** (draft PR) |
| 10 | prime chunk 8192 rows (vs 4096); 16384 measured as a second arm | `MEMRA_PRIME_CHUNK`; GB10 default 8192 in d5f13e6906 | 2+2+2 boots, S C8 C16 C16 C8 S, on the PR binary | 32k 38.130, 131k 189.995 s | 8192: 37.998 / 188.788 s (-0.35% / -0.64%); 16384: 38.031 / 187.702 s (-0.26% / -1.21%); text identical; admission workspace 4.6 / 6.4 / 10.2 GB per 131k prime, MemAvailable floor 81.4 / 81.3 / 74.5 GiB | ef37ca487b (env), d5f13e6906 | round2-chunk-ab/ | **adopted** at 8192 (16384's extra 0.57% costs 3.8 GB booking and 7 GiB per concurrent 131k prime) |
| 11 | silu(gate)*up fused into the down projection's F8F4 e4m3 quantize (no f32 [t, n_ff] activation pass) | code; `f8f4_prefill_direct` mirrors the dispatch | 2+2, P A B C C B A P | P 32k 38.133, 131k 190.043 s | 36.828 / 184.879 s (-3.42% / -2.72%), text identical; kernel-check `MMQ-F8F4-SILU-FUSED` bit cell | 9d6a5e683b (measured), 68a8a05730 (PR) | round2-levers-ab/ | **adopted** |
| 12 | #11 + each shared activation quantized once per GEMM group (gate/up, GDN qkv/z, attention q/k/v) | code (no flag) | 2+2 | 36.828 / 184.879 s (#11) | 36.256 / 182.654 s (-1.55% / -1.20% vs #11), text identical | 277995d656 (measured), 873634c010 (PR) | round2-levers-ab/ | **adopted** |
| 13 | #12 + GDN L2 norm four rows per warp, loads issued first | code (no flag) | 2+2 | 36.256 / 182.654 s (#12) | 36.252 / 182.588 s (-0.01% / -0.04%), text identical; kernel-check v2r4 vs v2 byte cell | 253a52c36a (research/gb10-prefill-round2-levers) | round2-levers-ab/ | **rejected** (flat) |
| 14 | round-2 stack: #10 (8192) + #11 + #12 | defaults | P S S P, unit 1, 32k/131k/260k per boot | 38.151 / 189.940 / 475.649 s (round-1 stack) | 36.163 / 181.130 / 457.177 s (-5.21% / -4.64% / -3.88%), text identical; unit 2 256k 474.442 -> 456.213 s | 0cb9642b98 | round2-final-ab/, round2-final-256k-unit2/, round2-gates-unit1/, round2-gates-unit2/ | **adopted** (pushed to the draft PR) |
