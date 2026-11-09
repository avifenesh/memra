# The GDN out-norm inside the fused state/output pass (round 4, lever 4)

## What it costs now

Every GDN layer of a prime chunk runs two kernels back to back:

1. `gdn_chunk_state_output_mma` (fused K4+K5, grid (48 heads, 2 column halves), 8 warps,
   2 CTAs/SM) writes the chunk output `o`, f32 `[T, 48, 128]`: 100.7 MB at T = 4,096.
2. `gated_rmsnorm_e4m3v2_f32` (one 128-thread block per (token, head) row) reads `o` and `z`
   (100.7 MB each), and writes the ssm_out GEMM's F8F4 v2 activation (25.2 MB of e4m3 plus the
   scale plane). Nothing else reads `o`.

The 131k profile on the stack head (`profile-r4-stack-131k/`): `gated_rmsnorm_e4m3v2_f32`
1,536 calls, 1.002 ms each, 1.539 s; `gdn_chunk_state_output_mma` 1,551 calls, 1.632 ms each.
The `o` store is 100.7 MB of the state pass's traffic (about 0.47 ms at 215 GB/s).

## The change

The state pass computes the gated norm and the v2 activation itself and never stores `o`:
`gdn_chunk_state_output_norm_mma`, the shipped kernel's body with an epilogue after the output
chain. `z` is read once (as today, by a different kernel), `o` is neither written nor read back.
Per call this removes the norm kernel (1.0 ms) and the `o` store, and adds the epilogue's latency
to each chunk's critical path. Expected: 0.7-0.9 ms per call, 1.1-1.4 s at 131k, about twice that
at 260k (the call count doubles). The prime-table graph keeps the pair, as it does for the fused
K4+K5 today; the multi-sequence (`_vl8`) path and every non-seeded path are unchanged.

## Exactness: the same values in the same order

The norm's values come from `gated_rmsnorm_e4m3v2_f32`, which is `gated_rmsnorm_f32` verbatim:

- thread i of the 128-thread block squares element i (`sum = 0 + v * v`, one rounding);
- each warp reduces its 32 squares with `__shfl_down_sync` at offsets 16, 8, 4, 2, 1, so the warp
  sum s_w (warp w = columns 32w .. 32w + 31) is the fixed tree
  b_i = a_i + a_{i+16}, c_i = b_i + b_{i+8}, d_i = c_i + c_{i+4}, e_i = d_i + d_{i+2}, f = e_0 + e_1;
- warp 0 combines the four warp sums with the same shuffles over lanes 0-3 (the other lanes hold
  0.0f, and adding +0.0f to a non-negative sum is exact): total = (s0 + s2) + (s1 + s3);
- `scale = rsqrtf(total / ncols + eps)`, then per element
  `x = (o * scale * w[i]) * (z / (1 + expf(-z)))`;
- per 32-column group: amax of |x| (`fmaxf`, which is order-free), `d_inv = 448 / amax`,
  `cvt.rn.satfinite e4m3(x * d_inv)`, the scale `amax / 448` at the lane-permuted slot.

The fused kernel reproduces each step on the same f32 values (the output chain's `scale * CP`
is the value the norm kernel read back from `o`):

- **Group sums on the mma fragment layout.** In the output warps (nq = 1) warp mh holds columns
  mh*16 .. +15 of the CTA's 64; lane (fr, q) holds columns {2q, 2q+1, 8+2q, 9+2q} of rows fr,
  fr+8, fr+16, fr+24. A 32-column group g is warps 2g and 2g+1, so the tree's first level
  (a_i + a_{i+16}) pairs a value of warp 2g with the value at the same register position of warp
  2g+1: warp 2g+1 hands its squares over shared memory. The later levels fall inside one lane
  (i, i+8: hf 0 and 1), across lanes q and q+2 (i, i+4), q and q+1 (i, i+2), and inside lane q = 0
  (0, 1). Every add has the tree's operands; IEEE addition commutes, so operand order inside an
  add does not matter, only the grouping, which is the tree's.
- **The four warp sums across the two CTAs.** Columns 0-63 (s0, s1) live in the CTA with
  `blockIdx.y = 0`, columns 64-127 (s2, s3) in its partner. The pair launches as a cluster of 2
  (`__cluster_dims__(1, 2, 1)`); each CTA writes its two group sums per row into the partner's
  shared memory (`st.shared::cluster`) and arrives on the partner's mbarrier (one per chunk
  parity, so a CTA cannot run two exchanges ahead). Both CTAs then form (s0 + s2) + (s1 + s3).
- **Per element and per group.** `scale`, `x`, the cvt and the scale plane use the norm kernel's
  expressions in the same compilation unit (`cu/hybrid.cu`, the same nvcc flags; `ncols` stays a
  runtime int as in the norm kernel). The group amax is a quad `fmaxf` within each warp, then the
  two warps' partials over shared memory.

So the e4m3 bytes and the scale plane must equal the pair's bit for bit, and the state the pass
carries is untouched (the epilogue only reads the output chain's registers).

## Resources

Shipped: 128 registers with 16 B of spill (sm_120a, `--resource-usage`), 40,704 B dynamic shared
memory, 2 CTAs/SM. The epilogue's registers live after the step A / phase 1 chains, where `CA`
and `uval` are dead, so the peak should not rise; `z` loads issue after barrier S3 to hide their
DRAM latency behind the S4 hand-off and the output chain. Shared memory: the squares hand-off
reuses `hand` (free between S4's read and the next chunk's S2 write); new: the partner's sums
(2 parities x 2 groups x 32 rows), the local sums, the warp amax partials and two mbarriers,
about 1.3 KB, 42 KB per CTA (2 CTAs/SM fit in GB10's 100 KB with the 1 KB per-CTA reserve).
The shipped kernel is untouched: the fused form is a separate entry over a shared body template,
and the shipped entry's SASS must stay identical (`cuobjdump -sass` diff).

## Gates

1. Probe on a unit (`attn-r4/probe/probe_gdn_norm.cu`, includes the tree's `hybrid.cu`):
   the pair (shipped K4+K5 then `gated_rmsnorm_e4m3v2_f32`) against the fused form at the 27B
   shape, T 4,096 / 4,032 / 43 / 16: the e4m3 bytes, the scale plane and `state_out` bit for bit;
   time both, 5 launches x 2 passes. Stop here if the fused call does not beat the pair.
2. kernel-check cell `GDN-K45-NORM` (both forms through the engine entry points, the bytes and
   the state compared, plus a partial last chunk).
3. The SASS of the shipped `gdn_chunk_state_output_mma` unchanged.
4. Served B N N B B N on unit 1, 32k / 131k / 260k, one fresh boot per cell, 3 per arm; the
   answers identical; second unit; 16k argmax-margin, run-spec K=1..8, kernel-check ALL GREEN.

## Not chosen: folding the boundary tail into the last trunk chunk

The 43-row tail costs 0.46 s at 131k, but the stable boundary exists so that the cache entry's
state is the state a resumed request continues from. Folding needs the GDN state, conv state and
last hidden at a 32-row grid point inside a chunk, and the tail rows then run the 4,096-row
programs (h2p attention, the v3 tile) instead of the small-T ones a resumed request runs, so the
cold and resumed answers could differ. Less time, more risk, more code.
