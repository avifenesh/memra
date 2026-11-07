# GB10 prime, second pass: residual epilogue, K2 at C = 32, fused conv + L2, fused attention q path (2026-11-07)

**Status:** decided on one device class (unified-memory 120a family, DGX Spark): prime bench per
lever, exactness battery and served A/B. Follows `GB10-PRIME-GDN-GLUE.md` (same lane, same
receipts directory). The residual epilogue and the out-gate seed ride on the F8F4 v2 route, so they
reach only GB10. K2 at C = 32 (shared-memory footprint and occupancy), the fused conv + L2 (one
warp per 128 channels instead of a thread per channel) and the fused q path (one launch for three)
change their kernels' parallelism, so each is keyed on GB10 and waits for a discrete-card A/B.

## Question

After the first pass, a 131,051-token served prime on GB10 took 130.9 s (`profile-glue2-131k/`):
58.1 s in the F8F4 GEMMs, 40.7 s in prime attention, 6.8 s in the residual add + norm kernels (at
214 GB/s, the DRAM floor of their bytes), 6.7 s in the GDN chunk scan, 3.3 s in the fused conv and
L2 norms, 1.3 s in the beta/alpha dp4a matvecs and about 2 s in the attention q path. Which of the
remaining bytes and stalls can go with the output unchanged?

## Decision

Exact rewrites, default ON, each behind a seam (`docs/FLAGS.md` section 3):

- `MEMRA_PRIME_RESID_EPI`: the F8F4 v2 GEMM of each mixer's output projection (attention wo, GDN
  ssm_out) and of the dense FFN's down projection adds the residual in its epilogue
  (`res + acc * scale`, both roundings explicit); the norm that follows reads the sum. The f32
  projection output is never written and read back. The prime arms the residual right before the
  mixer and takes it right after; the seed guard clears it on every exit.
- `MEMRA_GDN_K2_32` (GB10): K2 at C = 32 runs on a 33,280-byte shared layout (the chunk's k rows
  serve as both tiles), so two blocks share an SM where the 67,072-byte layout fit one.
- `MEMRA_GDN_CONV_L2` (GB10, inside the `MEMRA_GDN_TRAFFIC` set): the state conv runs the q/k L2
  norms in the same pass, one warp per 128-channel row, with 16-byte loads at d_conv 4; the raw q/k
  rows are never stored.
- `MEMRA_PRIME_Q_FUSE` (GB10): the q split, q-norm and q RoPE run in one kernel.
- `MEMRA_PRIME_NORM_FUSE`: the attention out-gate writes wo's F8F4 v2 activation; the wo GEMM fails
  closed if it would not adopt it.

## Evidence

`research/gb10-prefill-next-20261107/` (README "Second pass", LEVER-LEDGER rows 14-23). Prime bench
at 33,890 tokens, unit 2, interleaved in both orders, last-row prime logits identical in every
cell (`758739be...`):

| arm | baseline | candidate | change |
|---|---:|---:|---:|
| conv + L2 fused, out-gate seed | 25.604 s | 25.364 s | -0.94% |
| + residual epilogue | 25.272 s | 25.064 s | -0.81% |
| + 16-byte conv loads, fused q path | 25.063 s | 24.774 s | -1.15% |
| + K2 at C = 32 | 24.774 s | 24.565 s | -0.83% |

At 131k (`profile-glue3-131k/` against `profile-glue2-131k/`, one cold served prefill each under
nsys): TTFT 130.85 -> 127.37 s. The residual epilogue takes 3.24 s out of the norm kernels and puts
2.23 s back into the GEMM tile, which now reads the residual tile at the end of each output tile;
the conv + L2 loses 1.00 s, K2 0.65 s and the q path 0.57 s.

Served cold TTFT, unit 2, one fresh boot per arm, the #1089 code against this pass: -3.95% /
-3.26% / -3.39% at 32,784 / 131,051 / 259,951 tokens, the same answers, MemAvailable floor
66.15 GiB.

The landing campaign, unit 2, B L L B, main `46172c1bac` (before both passes) against the landed
code: 31.94 -> 24.39 s at 32,784 tokens (-23.6%), 157.42 -> 127.35 s at 131,051 (-19.1%) and
392.88 -> 332.47 s at 259,951 (-15.4%); every lane boot faster than every main boot, the same
answers, MemAvailable floor 66.12 GiB or more in every boot (66.21 in the lane boots). Against the
first pass's three-boot campaign medians, the second pass is -3.9% / -3.3% / -2.5% (the pair above
used one first-pass boot as its reference).

## Rejected

- The beta/alpha dp4a matvec reshaped (probe, `probe/probe_dp4a.cu`): row loop unrolled 4 or 8,
  rows split 8 / 16 / 24 per block, 2 or 8 tokens per block. All bit-identical, 0.75x to 1.01x of
  the shipped token-tiled kernel; not built.
