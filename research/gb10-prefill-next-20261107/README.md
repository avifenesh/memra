# Qwen3.8-27B prime on GB10: GDN traffic, the activation quantize and glue (2026-11-07)

Device: two DGX Spark units (GB10, compute capability 12.1, 48 SMs, unified LPDDR5x), SM clock
cap 2,200 MHz, CUDA 13.0 on the device, memra built `121a` on the device (unit 1 and unit 2 keep
the earlier lanes' names). Model: the Qwen3.8-27B
NVFP4+Q5_K GGUF with its masked MTP draft, served at 262,144 context; the files, prompts and
harness of `research/gb10-prefill-20261106/` and `research/gb10-prefill-attn-20261107/`. Base:
main `9805016bb1` (F8F4 GEMM v2 and the h2p prime attention landed).

## Where the 131k prime goes now (`profile-main-131k/`)

One cold 131,051-token served prefill under Nsight Systems: TTFT 157.553 s, kernels 156.114 s
inside it (the window ends at the last prime attention launch; `box/parts_last.py`).

| part | s | share | main kernels |
|---|---:|---:|---|
| F8F4 v2 GEMM tile | 59.72 | 38.3% | 12,800 x 4.67 ms |
| prime attention (h2p) | 40.76 | 26.1% | 512 x 79.3 ms |
| GDN chunk scan | 13.85 | 8.9% | state 4.24, solve 3.54, output 3.39, attn 2.61 |
| e4m3 activation quantize | 9.12 | 5.8% | silu-fused 5.90 (2,111 x 2.80 ms), plain 3.22 (6,334 x 0.51 ms) |
| glue | 8.63 | 5.5% | add 4.27, rms_norm 3.65, mul 0.65 |
| GDN conv / l2 / gates | 6.15 | 3.9% | conv 3.18, l2 norm 2.95 |
| F8F4 v2 weight image | 5.91 | 3.8% | |
| other matvec | 4.98 | 3.2% | `qmatvec_nvfp4_dp4a_rp` 3.23 (the GDN beta/alpha projections) |
| other | 4.40 | 2.8% | gated_rmsnorm 1.95, quantize_q8_1 1.90 |

Launch shapes per kernel are in `profile-main-131k/shapes.txt`. (`box/` in the receipts is this
directory's `harness/`.)

## Architecture and bytes (one 4,096-row prime chunk)

Shapes from the GGUF's qwen35 config: n_embd 5,120, n_ff 17,408; 48 Gated DeltaNet layers with 16
key heads and 48 value heads of 128 (conv_dim 10,240, value_dim 6,144, conv kernel 4), chunk C =
32 (128 chunks per call); 16 full-attention layers.

Every block measured here is a stream over [4,096 x width] f32 tensors that no 24 MB L2 holds, so
its floor is its DRAM bytes at the device's measured stream rates (read 232.5 GB/s, copy
215.3 GB/s; `research/gb10-prefill-20261106/ceilings/`). The profile's kernels already sit at
those rates: `add_f32` moves 252 MB in 1.04 ms (242 GB/s), `rms_norm_f32_v2` 168 MB in 0.75 ms,
the plain e4m3 quantize 107 MB in 0.51 ms, the silu-fused one 641 MB in 2.80 ms, the l2 norm at
the broadcast 48-head layout 250 MB in 0.96 ms. None of them has kernel headroom; the headroom is
in bytes that need not move:

- **GDN q/k broadcast.** The compact q/k layout (`[T, num_k, 128]`, read as head `h % hk`) was
  enabled only for num_v == 2 num_k; this model has 16 key and 48 value heads, so the conv wrote
  q and k three times over, the two l2 norms ran on 48 heads, and K2/K3/K4/K5 read the copies.
- **K3's f32 W.** On the mma path K4 reads only W's bf16 twin; K3 still stored the f32 W
  (100 MB per call).
- **Residual add + RMSNorm + activation quantize.** Each layer has two sites where `add_f32`
  writes the residual, `rms_norm` re-reads it and writes the f32 norm output, and the GEMM group
  re-reads that to quantize it to e4m3 (and, in a GDN layer, the beta and alpha projections each
  quantize it to q8_1 again).
- **beta/alpha matvec.** `qmatvec_nvfp4_dp4a_rp` runs one 128-thread block per (output row,
  token): 196,608 blocks for a 5,120 -> 48 projection, each re-reading its token's activation.
- **gate/up f32 outputs.** The dense FFN's gate and up GEMMs write two [4,096 x 17,408] f32
  tensors (570 MB) that the down projection's fused silu quantize reads back (2.80 ms per layer).
- **GDN out-norm.** `gated_rmsnorm` writes a [4,096 x 6,144] f32 tensor the ssm_out GEMM's
  quantize reads back.

`derive_next.py` (`derive_next.txt`) prices the byte removals: 135.7 -> 59.5 GB per chunk, 354 ms
per chunk at 215.3 GB/s, 2.8 / 11.3 / 22.5 s at 32k / 131k / 260k before the dp4a rewrite.

## GDN K4/K5 ceiling probe (`probe/`, `ab-gdn2/probe-gdn.log`)

`probe/probe_gdn.cu` includes the tree's `hybrid.cu` verbatim and times the shipped state pass
(K4) and output pass (K5) at the model's shape (48 heads, 16 compact q/k heads, T 4,096 and 8,192)
against exact rewrites, comparing Y, Ssnap, the final state and K5's output bit for bit:

| kernel (T 4,096) | ms | vs shipped | exact |
|---|---:|---:|---|
| K4 as shipped (32 state columns per CTA, 192 CTAs, two waves at 2 CTAs/SM) | 2.429 | | |
| K4, 64 columns per CTA (one wave of 96 CTAs) | 2.123 | 1.14x | yes |
| K4 with the next chunk's U / gate / beta prefetched into registers | 2.588 | 0.94x | yes |
| K4 without the Y / Ssnap stores (diagnostic) | 1.287 | 1.89x | n/a |
| K4, 64 columns, prefetch, no stores (diagnostic) | 0.920 | 2.64x | n/a |
| K5 chunks fastest (shipped) | 2.135 | | |
| K5 heads fastest (this lane) | 1.786 | 1.20x | yes |

K4 is bound by writing Ssnap (201 MB of f16 chunk-start states) and Y (50 MB) for K5: the stores
alone are 1.1 ms, about their DRAM time. Only computing K5's output inside the state pass removes
them, together with K5's 2.1 ms read of the same bytes.

**The fused state + output pass** (`gdn_chunk_state_output_mma`; `probe-runs/k45-r1-unit1.log`)
keeps K4's 64-column, one-wave layout and computes o in the same CTA. The B fragments of the state
come straight from the accumulator registers, packed to bf16 for W.S and to f16 for q.S exactly as
K4's Mb and Ssnap stores rounded them. Each k16 chain of W.S and q.S runs blocks 0..3 on one warp
and continues 4..7 on its partner through 8 KB of shared memory, so every element keeps K4's and
K5's order. Y's fragments are transposed in registers (`movmatrix`) for P.Y. That takes 40,704 B of
shared memory, still 2 CTAs per SM, with 128 registers and 16 B of spill:

| T | K4 + K5 shipped | K4 + K5 heads fastest | fused | state / o |
|---:|---:|---:|---:|---|
| 4,096 | 4.744 ms | 4.288 ms | 1.703 ms (2.52x) | bit-identical |
| 8,192 | 9.379 ms | 8.523 ms | 3.423 ms (2.49x) | bit-identical |

## Levers (`LEVER-LEDGER.md`)

All exact by construction (the same operands and the same per-element operation order); run-gen
prime bench (`MEMRA_PP_PRIME=1`, the served prime path), 33,890-token record prompt, 3 primes per
process, cells interleaved in both orders, last-row prime logits `758739be...` in every cell.

Unit 2, one research build with every lever behind a seam (`c02f90930d`, `ab-gdn2/`):

| arm | cells | mean s | step |
|---|---|---:|---:|
| main `9805016bb1` (`ab-gdn1/`) | 32.196, 32.189 | 32.19 | |
| all off: every seam rolled back (Z) | 32.216, 32.255 | 32.24 | |
| GDN: compact q/k at 3:1, no f32 W store, heads-fastest K2/K3/K5 (G) | 30.633, 30.705 | 30.67 | -4.9% |
| + one `add_rms_norm` kernel per residual site (G1) | 30.406, 30.399 | 30.40 | -0.9% |
| + e4m3 v2 activation from that kernel, dense FFN f32 norm output skipped (G3) | 29.794, 29.850 | 29.82 | -1.9% |
| + q8_1 pair quantized once per group (GQ) | 29.606, 29.553 | 29.58 | -0.8% |
| + token-tiled dp4a matvec for beta/alpha (A) | 29.098, 29.140 | 29.12 | -1.6% |

The three GDN parts one at a time (`48265176f0`, `ab-gdn1/`, against its all-off arm 32.28 s):
the compact layout -2.9%, the dropped f32 W store -0.7%, the heads-fastest grid -0.7%.

Unit 1, the stacked levers (`ab-unit1-stack/`, `ab-unit1-k45/`):

| arm | cells | mean s | step |
|---|---|---:|---:|
| PR-1 set above (`5650c30d03`) | 29.112, 29.162 | 29.14 | |
| + GDN out-norm writes the ssm_out GEMM's v2 activation (`ebe9cf2a8e`, gate/up seam off) | 28.783, 28.823 | 28.80 | -1.2% |
| + gate and up in one F8F4 v2 launch writing the down GEMM's silu activation | 26.979, 26.993 | 26.99 | -6.3% |
| final head `ffba9c1669` with the fused K4+K5 off (adds the q8_1 seed adoption) | 26.765, 26.823 | 26.79 | -0.7% |
| + fused K4+K5 (the default) | 25.709, 25.697 | 25.70 | -4.1% |

Main to the final head: 32.19 -> 25.70 s at 33,890 tokens (-20.2%), the same logits.

The gate/up launch moves more than the silu quantize's 2.8 ms per layer because the two GEMMs no
longer write 570 MB of f32: one launch over a paired weight image (16 gate rows then the same 16
up rows in every 32-row warp span) leaves each thread holding one feature's gate and up
accumulators, and the epilogue writes e4m3 bytes and per-32 scales instead. The norm seeds are
take-once and keyed by the norm output's buffer; a guard clears them on entry to and exit from the
prime walk, and every norm site that does not seed clears them, so no later write of the buffer
can meet a stale activation.

## Served A/B (`served-unit2-final/`)

Cold TTFT, the Qwen3.8-27B NVFP4+Q5_K GGUF with the masked MTP draft at 262,144 context, one fresh
boot per cell in the order B L L B B L, all three lengths per boot, on unit 2. B = main `6f5e03e74a`,
L = the lane head `189a887d27` (the same code as the landing commit before its rebase onto main's
docs/receipts commits). `harness/served_ab.sh`, `harness/pf_long_memra.py`:

| prompt tokens | main | lane | change (medians) | answer sha256 |
|---:|---|---|---:|---|
| 32,784 | 31.848, 31.887, 31.895 s | 25.361, 25.394, 25.382 s | -20.40% | cb9da8ae3cec |
| 131,051 | 158.502, 157.694, 157.706 s | 131.568, 131.689, 132.558 s | -16.50% | 84980efddfee |
| 259,951 | 392.298, 391.092, 391.238 s | 340.886, 340.477, 340.951 s | -12.87% | 3759709dfbe4 |

MemAvailable floor (GiB): 01-B 65.53; 02-L 66.22; 03-L 66.30; 04-B 66.33; 05-B 66.29; 06-L 66.31


Every lane boot is faster than every main boot at every length. The answer text is identical in all
six boots at each length, needles found, no request error.

Unit 1 confirmation, the same binaries, B L L B (`served-unit1-final/`):

| prompt tokens | main | lane | change (medians) | answer sha256 |
|---:|---|---|---:|---|
| 32,784 | 31.900, 32.013 s | 25.420, 25.421 s | -20.45% | cb9da8ae3cec |
| 131,051 | 158.173, 158.250 s | 132.264, 132.260 s | -16.40% | 84980efddfee |
| 259,951 | 393.122, 394.167 s | 342.529, 343.602 s | -12.85% | 3759709dfbe4 |

MemAvailable floor (GiB): 01-B 66.77; 02-L 66.79; 03-L 66.74; 04-B 66.71


The same answers as unit 2 at every length.

## Exactness battery

On the lane binaries, with every lever on and with every lever rolled back as the off arm
(`MEMRA_GDN_TRAFFIC=0 MEMRA_GDN_K45=0 MEMRA_PRIME_NORM_FUSE=0 MEMRA_DP4A_TILED=0
MEMRA_FFN_GATEUP_FUSE=0`; `harness/gates_next2.sh`):

- unit 1, final code `189a887d27` (`gates-unit1-final/`): kernel-check ALL GREEN (121 cells, 5
  skipped) in both modes, including GDN-DB-GQA (compact vs broadcast, fused K4+K5 vs the pair, T
  200 / 64 / 20), ADD-RMS-NORM-E4M3V2, GATED-RMSNORM-E4M3V2, NVFP4-DP4A-RP-T4 and
  MMQ-F8F4-V2-GATEUP (GGUF and split-plane layouts, distinct macro-scales, T 130 / 1,024 / 4,096)
  at 0 mismatches; run-gen argmax-margin flips=0 bad=0 PASS; run-spec K=1..8 self-consistency PASS;
  prime-path last-row logits byte-identical for main, the lane and the lane rolled back at 16,219
  and 33,890 tokens (`d1b25840...`, `758739be...`); the monolithic forward's logits identical at
  16,219; `prime-batch-gate --exact` (the batched varlen prime, where the compact layout at 3:1 now
  also applies) ALL GREEN with 0 logit / state / hidden differences, its output identical with the
  levers rolled back.
- unit 2, the first set (`5650c30d03`, `gates-unit2-pr1/`): the same battery, ALL GREEN.

## Second pass (glue 2)

### Where the 131k prime went after the first pass (`profile-glue2-131k/`)

One cold 131,051-token served prefill of the first pass plus the conv + L2 fusion and the out-gate
seed (`5ea9cc8aa9`): TTFT 130.852 s.

| part | s | main kernels |
|---|---:|---|
| F8F4 v2 GEMM tile | 58.08 | 10,752 x 5.40 ms |
| prime attention (h2p) | 40.73 | 512 x 79.2 ms |
| glue | 6.80 | `add_rms_norm_e4m3v2_f32` 6.18 (4,064 x 1.52 ms) |
| GDN chunk scan | 6.67 | fused K4+K5 2.54, solve 2.17, K2 1.93 |
| F8F4 v2 weight image | 5.98 | |
| GDN conv + L2 | 3.28 | `ssm_conv1d_gdn_state_l2_f32` 3.26 (1,584 x 2.06 ms) |
| attention q path and gate | 1.49 | q split 0.85, sig_mul 0.49, rope 0.15 |
| beta/alpha dp4a | 1.33 | 3,168 x 0.42 ms |
| e4m3 activation quantize | 0.02 | every GEMM activation but the chunk tails is seeded |

The fused residual + norm kernel moves 273 MB (before a dense FFN) or 378 MB (before a GDN mixer,
with the f32 norm output and the q8_1 pair) per 4,096-row call in 1.52 ms, about 214 GB/s: the
floor of its bytes. Its add reads the projection output the GEMM just wrote, so the remaining
bytes are the projection's f32 round trip. K2 at C = 32 moves about 115 MB per call in 1.22 ms,
well under the stream rate; its 67,072-byte shared layout fits one block per SM. The fused conv
moves 336 MB in 2.06 ms (163 GB/s) with 4-byte loads.

### Levers (prime bench, unit 2, 33,890 tokens, interleaved)

| arm | cells | medians | change |
|---|---|---:|---:|
| #1089 code (`189a887d27`) | M 25.605, 25.603 | 25.604 | |
| + conv + L2 in one pass, out-gate seeds wo's activation (`5ea9cc8aa9`) | 25.277, 25.451 | 25.364 | -0.94% |
| `5ea9cc8aa9` (next session) | M 25.272, 25.271 | 25.272 | |
| + residual in the GEMM epilogue (`cf05643597`) | 25.078, 25.050 | 25.064 | -0.81% |
| `cf05643597` (next session) | M 25.054, 25.072 | 25.063 | |
| + 16-byte conv loads, fused q path, K2 at 32 off (`c95627db13`, `MEMRA_GDN_K2_32=0`) | 24.769, 24.778 | 24.774 | -1.15% |
| + K2 at C = 32 (the default) | 24.568, 24.562 | 24.565 | -0.83% |

Every cell's last-row prime logits are `758739be...`. The residual epilogue saves less than its
bytes promise (about 3 s of norm traffic at 131k): the GEMM's epilogue now reads the residual
tile, and that read sits on the tile's critical path. The dp4a matvec probe (`probe/probe_dp4a.cu`)
found no exact layout faster than the shipped token-tiled kernel (0.75x to 1.01x).

### The 131k prime after the second pass (`profile-glue3-131k/`)

The second-pass code (`c95627db13`), the same cold 131,051-token served prefill under nsys: TTFT
127.365 s against 130.852 (-2.7%), kernel time 124.24 s against 127.66. Per kernel, against
`profile-glue2-131k/`:

| kernel(s) | first pass + conv/L2 | second pass | change |
|---|---:|---:|---:|
| residual add + norm (`add_rms_norm_e4m3v2_f32` -> `rms_norm_e4m3v2_f32`), 4,064 calls | 6.18 s (1.52 ms) | 2.94 s (0.72 ms) | -3.24 s |
| F8F4 v2 GEMM tile (the residual tile read in the epilogue), 10,752 calls | 58.08 s (5.40 ms) | 60.31 s (5.61 ms) | +2.23 s |
| GDN conv + L2 (16-byte loads), 1,584 calls | 3.26 s (2.06 ms) | 2.26 s (1.43 ms) | -1.00 s |
| K2 (`gdn_chunk_attn_f32` -> `gdn_chunk_attn32_f32`), 1,584 calls | 1.93 s (1.22 ms) | 1.28 s (0.81 ms) | -0.65 s |
| attention q path: split + q-norm + RoPE -> `q_split_norm_rope_f32` | 1.44 s | 0.87 s | -0.57 s |

The residual epilogue nets about 1 s at 131k: the norm kernels lose the projection's f32 round
trip, the GEMM pays most of it back reading the residual tile at the end of every output tile,
where no mainloop work hides the load.

### Exactness battery (`gates-glue2-unit2/`, `gates-glue3-unit2/`)

On unit 2, the second-pass binaries against the #1089 binary, with every lever of both passes
rolled back as the off arm (the first pass's seams plus `MEMRA_PRIME_RESID_EPI=0
MEMRA_GDN_K2_32=0`; `harness/gates_next3.sh`):

- after the conv + L2 fusion and the out-gate seed (`5ea9cc8aa9`, `gates-glue2-unit2/`): kernel-check
  ALL GREEN (123 cells) in both modes, argmax-margin flips=0 bad=0, run-spec K=1..8 PASS, prime
  logits identical at 16,219 and 33,890 tokens for the #1089 binary, the lane and the lane rolled
  back, forward logits identical, `prime-batch-gate --exact` ALL GREEN with its output identical
  rolled back;
- the second-pass head's code (`c95627db13`, `gates-glue3-unit2/`): the same battery, kernel-check
  ALL GREEN (124 cells) in both modes, with the new bit cells MMQ-F8F4-V2-RESID (the model's down and
  ssm_out weights, GGUF and split-plane layouts, T 130 / 1,024 / 4,096), the norm-only arm of
  ADD-RMS-NORM-E4M3V2, GDN-CONV-L2, SIG-MUL-E4M3V2, Q-SPLIT-NORM-ROPE and the K2 64-row arms of
  GDN-DB-GQA at 0 mismatches; the rest as above.

### Served (`served-glue3-unit2-pair/`)

A first served pair on unit 2, one fresh boot per arm, all three lengths per boot, B = the #1089
code (`189a887d27`), L = the second-pass code (`c95627db13`); `harness/chain12.sh`:

| prompt tokens | #1089 | second pass | change | answer sha256 |
|---:|---:|---:|---:|---|
| 32,784 | 25.406 s | 24.402 s | -3.95% | cb9da8ae3cec |
| 131,051 | 131.691 s | 127.396 s | -3.26% | 84980efddfee |
| 259,951 | 343.880 s | 332.218 s | -3.39% | 3759709dfbe4 |

MemAvailable floor (GiB): 01-B 65.72; 02-L 66.15. The same answers as every earlier boot.


## Next

What is left in the 131k prime after both passes (`profile-glue3-131k/`, 127.4 s): the F8F4 GEMM
tile (60.3 s) and its per-call weight images (5.9 s; resident images were rejected for memory in
`docs/decisions/GB10-F8F4-GEMM-V2.md`), prime attention (40.6 s), and in this lane's blocks about
14 s, most of it at its DRAM floor: the GDN chunk scan 6.0 s (fused K4+K5 1.59 ms and the solve
1.37 ms per call move their bytes at the stream rate; K2 0.81 ms), the norm 2.9 s, the conv + L2
2.3 s, the GDN out-norm 1.5 s, the beta/alpha dp4a 1.3 s and the attention q path and out-gate
1.4 s. Candidates not built: the f32 norm output before a GDN mixer has no prime reader when every
weight of the group takes the seeded activation (about 0.8 s at 131k; the GEMM group needs a
fail-closed check for an unwritten f32 input first); the GDN out-norm inside the fused K4+K5 (Y
never stored, about 0.8 s; the 128-wide row norm spans the kernel's two 64-column CTAs).
