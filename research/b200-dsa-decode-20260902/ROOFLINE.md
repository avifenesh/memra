# t=1 MLA/DSA decode roofline on 2x B200 SXM (sm_100a), GLM-5.3-Flash NVFP4

Lane: `lane/b200-dsa-decode-20260902`. Door: `MEMRA_B200_DSA_DECODE` (default OFF).
Prior lane this builds on: `research/b200-mla-decode-20260902/LANE.md` (PR #83, the t-keyed
output-range split arm `MEMRA_B200_MLA_DECODE_ARM`).

## 0. Geometry and machine constants used throughout

Read off the checkout, not assumed:

| symbol | value | source |
|---|---|---|
| MLA/DSA layers | 11 | task census |
| `n_head` | 64 | glm5_next geometry, `mla_decode_arm_gate.rs` |
| `kv_rank` | 512 | ditto |
| `d_nope` / `d_v` | 256 / 256 | ditto |
| `d_rope` | 0 (NoPE) | ditto |
| latent cache row | `(kv_rank + d_rope) * 4 B` = **2048 B** | `cu/mla_attn.cu` takes `const float*` |
| DSA `n_slots` (top-k) | 2048 | task census |
| indexer `heads` / `d` | 32 / 128 | `cu/mla_attn.cu` scoring header: 134 MB key plane at 262144 pools, 17 GB per-head score plane at 1M/512 |
| `pool` / `select_k` | 4 / 512 | ditto |
| `n_pools` | `t_kv / 4` (8192 at 32k, 32768 at 128k, 262144 at 1M) | `mla_kpool_pool_keys` |

Machine (per GPU): 148 SMs, HBM3e **8 TB/s**, fp32 FFMA peak
`148 x 128 x 2 x 1.86 GHz` = **70.5 TFLOP/s**. Machine balance = 8.8 FLOP/byte. There is no
tensor-core path for true f32 on Blackwell, so every kernel below is measured against the SIMT
FFMA peak, not against a TC number it cannot reach.

**First correction to the task's premise.** The task sized the gathered set as
`2048 x 512 x 2 B x 64 heads = 128 MB` and put a 0.18 ms floor under it. The cache in this
checkout is **f32, not bf16**, and the index list is shared across heads (one selection per
query, the indexer mixes heads before selecting), so the honest numbers are: **4.00 MiB unique
per layer per token**, which is L2-resident on this die. Re-reading it per head costs L2
bandwidth, not HBM bandwidth. `attn_gathered` is therefore not a bandwidth problem at all, and
head-batching to "read each row once for all 64 heads" would *cost* performance at t=1 by
removing the only CTAs the kernel has (see 1.3). The real gap is 190x, not 4x, and it is an
issue/occupancy gap.

## 1. `memra_mla_attn_gathered_kernel` (depth-FLAT, 726 us/layer, ~8.0 ms/token)

### 1.1 The floors

- **Bytes.** Unique: `2048 slots x 2048 B` = 4.00 MiB per layer per token, shared by all 64
  heads. HBM floor `4.19e6 / 8e12` = **0.52 us/layer**, 5.8 us/token.
- **Bytes actually moved.** The kernel reads every gathered row **twice per head**: once in the
  warp-per-slot score dot (`row[l]` from global), once again in the PV accumulate
  (`cache[tt*width + l]`, also from global). That is `64 heads x 2 passes x 4.00 MiB` =
  **512 MiB requested per layer per token**, ~128x the unique bytes, absorbed by L2.
- **FLOPs.** `2048 x 512 x 2` (QK) + `2048 x 512 x 2` (PV) = 4.19 MFLOP per (token, head);
  `x 64` = **268.4 MFLOP/layer**. FFMA floor **3.81 us/layer**, 41.9 us/token.
- **Measured** 722.8 us at 32k, 726.2 us at 128k (37-40 us at a 66-token context, where the
  gathered set is not yet full). **190x the FFMA floor, 1390x the HBM floor.**

### 1.2 Launch geometry and occupancy

Grid is `t_q * n_head` = **64 CTAs** of 256 threads. On 148 SMs that is 43% of the SMs holding
exactly one CTA and 8 of 64 warp slots each: **5.4% of the machine's warp slots are resident**,
and there is not a second CTA anywhere to hide a barrier or a load.

### 1.3 Why it is 190x off, in the order the cycles are actually spent

The kernel walks `n_slots / MLA_WARPS` = **256 tiles** of 8 slots and pays, per tile:

1. **Redundant transcendentals.** Every one of the 256 threads evaluates
   `expf(s_score[w] - mnew)` for all 8 `w` in the `tsum` loop, and evaluates the same 8 values
   **again inside the `l` loop**, once per owned output element (`kv_rank / blockDim.x` = 2 of
   them). That is **24 `expf` per thread per tile where 8 distinct values exist**: 1.57M `expf`
   per CTA per layer, **98.4% of them redundant**. `expf` is a ~8-instruction sequence whose
   `MUFU.EX2` runs at quarter rate, so this alone is the largest single term.
2. **Two `__syncthreads` per tile = 512 barriers**, with 8 warps and one CTA per SM. Nothing
   covers the barrier latency; the SM idles through each one.
3. **The PV accumulate reads global, not shared.** 16 dependent scalar `float` loads per thread
   per tile against `cache[tt * width + l]`, no `float4`, no unrolling, memory latency exposed
   at 8 warps of MLP.
4. **The score dot re-reads the same row from global** that step 3 will read again, so the row
   crosses the L2/SM boundary twice per head instead of being staged once in shared memory.

### 1.4 Why it is depth-FLAT (and why that is the good news)

`n_slots` is pinned at the DSA top-k budget, 2048. Past ~2048 tokens of context the gathered
set is always full, so the kernel's work is context-independent **by construction**: 722.8 us at
32k and 726.2 us at 128k is the same kernel doing the same 2048 slots. Every microsecond taken
out of it is taken out of the 1M token as well as the 2k token.

## 2. `memra_mla_kpool_score_*` (depth-LINEAR, the 1M slide)

### 2.1 The floors, per layer per token, at t_q=1

| context | `n_pools` | key bytes | HBM floor | FLOP | FFMA floor | measured | off by |
|---|---|---|---|---|---|---|---|
| 32k | 8192 | 4.19 MB | 0.52 us | 67.1 MFLOP | 0.95 us | 125.9 us (`_ref`) | **132x** |
| 128k | 32768 | 16.8 MB | 2.10 us | 268.4 MFLOP | 3.81 us | 167.2 us (`_tiled<64,1,1,1,16>`) | **44x** |
| 1M | 262144 | 134.2 MB | 16.8 us | 2.15 GFLOP | **30.5 us** | ~1.3 ms (extrapolated) | **43x** |

Arithmetic intensity is `heads * d * 2 / (d * 4)` = `2 * heads / 4` = **16 FLOP/byte**, against
a machine balance of 8.8, so a *perfect* f32 scorer is **FFMA-bound, not bandwidth-bound**: the
1M floor is 30.5 us/layer = **336 us/token**, not the 185 us/token the pool-row bytes alone
would suggest. State both; the FFMA number is the one that binds.

### 2.2 Why it is depth-LINEAR, and why no cache fixes that

`n_pools = t_kv / pool`, and the kernel scores **every causally visible pool**. A pool key is
immutable once its last row lands (`memra_mla_kpool_pool_keys_kernel` header says so and the
incremental `pool_begin` build relies on it) - but the *score* is `f(q_t, k_p)` and `q_t` is a
brand-new query every token, so **no score survives a decode step**. There is no scored-cache
formulation of exact DSA top-k. The scan is required; the only question is whether it runs at
the floor.

### 2.3 Why the shipped decode config is 44x off

Decode dispatches `memra_kpool_score_launch<TX=64, TY=1, RT=1, RP=1, KC=16>`, i.e.
`BT = TY*RT = 1` query and `BP = TX*RP = 64` pools per block, **one accumulator per thread**.
The inner step is `RT + RP = 2` shared loads for `RT * RP = 1` FFMA - the register-blocking
that makes the BT=128 prefill tile pay for itself is *absent* at BT=1, so the kernel is
shared-memory-issue bound at ~0.5 FFMA per LDS. Grid is `n_pools / 64` blocks of 64 threads:
512 blocks x 64 threads = 32768 threads at 128k, on a die that wants ~300k. The reference
kernel below the 16384-pool crossover is worse in the other direction: one 32-thread CTA per
(query, pool) - 32768 CTAs at 128k - with thread 0 doing the 32-add head mix while 31 threads
park.

The axis the decode shape actually has, and neither config uses, is **heads**: at t_q=1 there
is one query but 32 head-dots per pool, which is exactly the reuse a register tile wants.

## 3. `memra_mla_kpool_select_kernel` (170 us/layer, 1.87 ms/token)

Grid is `t_q` blocks: **ONE CTA at t=1**, 0.68% of the die, running 8 radix passes over
`n_pools`. Bytes at 128k are `32768 x 4 B x ~8 passes` = 1.05 MB -> 0.13 us floor. **~1300x
off, entirely because it is a single CTA.** Also depth-linear. It has no independent-output
axis to split on with the output-range technique, so a real fix is a hierarchical multi-CTA
radix select with its own numeric argument - out of this lane's door (which the task scoped to
gathered attention and pool scoring), named here so the next lane has the number.

## 4. `memra_mla_absorb_q_kernel` / `memra_mla_decompress_v_kernel` (70.9 / 71.0 us/layer)

Both are weight-streaming matvecs and both are **bandwidth-bound, not compute-bound**:

- `absorb_q` reads `wk_b` = `64 x 512 x 256 x 4 B` = **33.55 MB/layer**; FLOPs 16.8 MFLOP
  (0.24 us). Intensity 0.5 FLOP/byte. HBM floor **4.19 us/layer** -> 46 us/token.
- `decompress_v` reads `wv_b` = `64 x 256 x 512 x 4 B` = 33.55 MB; same 4.19 us floor.
- Measured 70.9 / 71.0 us at 32k: **17x off**. With PR #83's split=4 the box measured
  49.1 / 48.0 us: still **12x off**.

Geometry: 64 CTAs (5.4% of warp slots), each thread walking a serial 256- or 512-long scalar
dot with plain `float` loads. Saturating a 33.55 MB HBM stream needs thousands of CTAs issuing
`float4`, not 64 or 256 issuing `float`. These are depth-flat (weights, not context), so they
are a fixed ~90 us/token tax rather than a 1M problem - real, but a third of the size of the
two the door takes.

## 5. The token budget this adds up to (1M context, plain decode, 11 layers)

| stage | measured/token | floor/token | ratio |
|---|---|---|---|
| `attn_gathered` | 7.99 ms | 0.042 ms (FFMA) | 190x |
| `kpool_score` | ~14 ms | 0.336 ms (FFMA) | 43x |
| `kpool_select` | 1.87 ms | 0.001 ms | ~1300x |
| `absorb_q` + `decompress_v` | 1.56 ms | 0.092 ms (HBM) | 17x |
| **MLA/DSA total** | **~25.4 ms** | **~0.47 ms** | **54x** |

At 22.7 tok/s the 1M token is 44.1 ms, so MLA/DSA is **58% of it**. The owner target of 230
tok/s plain is a 4.35 ms token: the MLA/DSA stack alone must come down by ~10x before the
target is even arithmetically reachable, and the two kernels this lane's door rewrites are
22.0 ms of the 25.4 ms.

## 6. What the door does about it

`MEMRA_B200_DSA_DECODE` (default OFF; FLAGS.md row; sm_100a compile-gated like its sibling).

- **`=1`** engages only arms that are **bit-identical** to the shipped kernels:
  - `memra_mla_attn_gathered_dsa_kernel` - same 8-slot tile fold, same lane stride, same shuffle
    tree, same operation order, so bit identity is a construction; what changes is that the tile's
    8 KV rows are staged into shared memory **once** with `float4` loads and serve both the score
    dot and the PV accumulate (kills causes 3 and 4 of 1.3), and the 8 `expf` are computed once
    per thread per tile into registers and reused by both the `tsum` and the accumulate loop
    (kills cause 1: 24 -> 8, 3x fewer transcendentals).
  - `memra_mla_kpool_score_dsa_kernel<H>` - the decode-shaped scorer 2.3 asks for: one thread
    owns `RP` pools x **all H heads**, so the `c`-ascending dot and the `h`-ascending head mix
    both stay sequential in one thread and the six-step rounding sequence is preserved
    instruction for instruction. Pool keys are staged coalesced through shared memory in `KC`
    slabs; the `q` plane (`H x d x 4 B` = 16 KB at the glm5 shape) is resident for the whole
    block. LDS per FFMA drops from 2.0 to ~0.125 by holding `RP` dot accumulators per head in
    registers and reading `q` as `float4`.
- **`=2`** additionally engages `memra_mla_dsa_attn_partial_kernel` +
  `memra_mla_dsa_attn_combine_kernel`, the slot-split (flash-decoding) arm that is the only way
  to put more than 64 CTAs on the gathered attention at t=1. It combines per-chunk
  `(m, dsum, acc)` triples, which is a **different rounding program** from the single sequential
  2048-slot fold, so it ships under the named numeric class **`dsa-split-softmax-f32`** with an
  argmax gate on real-shaped inputs, never as a bit-identity claim.

Every arm's split/chunk factor is read from a `t_q`-keyed table whose unmeasured cells are 1
(= the shipped kernel), following the same discipline as PR #83: the box run either confirms a
cell or names the one to change.

## 7. What the door actually measured (RTX 5090, correctness rig, 2026-09-03)

`dsa-decode-gate 0 3 65536`, release, N=3 interleaved rounds, full log
`gate-5090-20260903.txt` in this directory. **Rig law applies**: the 5090 throttles, so these
microseconds are a correctness receipt and a direction, never a serving claim - and the door is
compile-gated to sm_100a, so this rig cannot even engage it. The gate calls every arm through
its raw FFI for exactly that reason.

**Verdict: PASS.** Every bit-identical arm matched bytewise at all five contexts x both query
widths; every `dsa-warp-online-f32` chunk count held argmax on every (token, head) latent row
(0 of 64 / 0 of 256 rows moved) with maxdiff <= 1.8e-6 and max-relative <= 3.5e-6; no
policy-selected arm regressed.

| stage | arm | t_q=1 | t_q=4 | vs shipped |
|---|---|---|---|---|
| `attn_gathered` | shipped | 442-501 us | 854-882 us | - |
| | single-pass (bit-identical) | 480-545 us | 1119-1154 us | **0.92x / 0.76x - a LOSS** |
| | warp-online, 16 chunks | **69.6-71.2 us** | **225.8-259.4 us** | **6.3x / 3.4x** |
| `kpool_score` @128k | head-blocked (bit-identical) | 38.3 us vs 517.9 | 126.7 vs 1597.6 | **13.5x / 12.6x** |
| `kpool_score` @1M | head-blocked (bit-identical) | **308.6 us vs 3093.7** | 1101.3 vs 12251.4 | **10.0x / 11.1x** |

### 7.1 The finding that killed the first design, kept on record

The single-pass bit-identical gathered kernel is a measured LOSS, and the reason is that both
savings it was built around were already gone:

- `expf(s_score[w] - mnew)` is **loop-invariant in the `l` loop**, so nvcc had already hoisted
  it. The "24 -> 8 exponentials per thread per tile" in section 1.3 was a source-level count,
  not a machine-level one.
- The second `cache[tt * width + l]` pass **hits L1/L2** (the gathered set is 4 MiB), so staging
  the row through shared memory ADDS a full smem write and read and buys back only L1 hits.

**Bit identity is the binding constraint on `attn_gathered`**: the shipped fold is already at a
local optimum inside it. That is what forced the named numeric class, and it is the single most
useful thing this lane learned. The arm stays in the tree (arm code 1) because the gate measures
it and this receipt is the reason the door does not select it.

### 7.2 What replaced it, and why it wins

`memra_mla_dsa_attn_warp_kernel<J, JP>`: one WARP owns one (token, head, slot-chunk) and holds
the whole `kv_rank`-wide accumulator in registers (`J = kv_rank/32` = 16 floats per lane).

- **Every KV element is read from memory exactly once** and consumed twice from registers - the
  QK dot and the PV accumulate use the same `kv[j]`. The shipped kernel reads it twice per head.
- **Zero `__syncthreads`.** The shipped kernel pays two barriers per 8-slot tile, 512 per CTA,
  with 8 warps and one CTA per SM to hide them. The fold here is warp-local and the reduction is
  a 5-step `__shfl_xor_sync` butterfly, so every lane ends with the sum and no broadcast is
  needed either.
- **Two `expf` per slot per warp**, against ~196k per warp per layer.
- **`chunks` is the occupancy knob the head axis cannot be**: at t_q=1 there are 64 independent
  (token, head) outputs for 148 SMs, and 16 chunks turns that into 1024 warps. 16 beats 32 at
  both widths, which is where the combine and the per-chunk softmax setup start costing more
  than the parallelism recovers.

It folds per SLOT where the shipped kernel folds in 8-slot tiles, so it is not bit-identical:
class **`dsa-warp-online-f32`**, argmax-gated.

### 7.3 Where that leaves the 1M token, if the B200 confirms the direction

| stage | before | after (5090 ratios applied) |
|---|---|---|
| `attn_gathered` | 7.99 ms | ~1.27 ms |
| `kpool_score` | ~14 ms | ~1.4 ms |
| `kpool_select` | 1.87 ms | 1.87 ms (untouched, next lane) |
| `absorb_q` + `decompress_v` | 1.56 ms | 1.56 ms (PR #83's door) |
| **MLA/DSA total** | **~25.4 ms** | **~6.1 ms** |

A 44.1 ms token at 1M would become ~24.8 ms, i.e. 22.7 -> ~40 tok/s, and `kpool_select`'s
single CTA (section 3) becomes the largest remaining depth item. That is an extrapolation from a
correctness rig and is written here as the thing the B200 run is supposed to confirm or refute,
not as a claim.

### 7.4 Open, for the session that owns the pair

1. Run `dsa-decode-gate <dev> 5` on the 2x B200 pair (full 262144-row cache, N=5). It either
   confirms the `MLA_DSA_ATTN_ARM` cells (16 at t_q=1 and t_q=4) or prints the cell to change.
2. Then the end-to-end serving A/B at `MEMRA_B200_DSA_DECODE` 0 / 1 / 2, interleaved x5, fresh
   boots, greedy exactness gate plus the vendor-default sampled twin with a spec-engagement
   receipt, TTFT/TPOT/ITL p50/p95/p99 - per the per-hardware-arm-selection and never-serve-greedy
   laws. Level 1 is the bit-identical scorer alone and should be judgeable on the exactness gate;
   level 2 adds the named class and needs the sampled arm.
3. `memra_mla_kpool_select_kernel` (section 3) is now the largest untouched depth item at 1.87
   ms/token and 1 CTA at t_q=1. It needs a hierarchical multi-CTA radix select with its own
   order-preservation argument.
