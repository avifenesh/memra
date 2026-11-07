# FA pipeline port plan — the last ~3.5% of 12B prefill (2026-07-22)

State: 12B pp1736 0.965x llama interleaved (laptop 175W), decode 0.984x. GEMM at parity
(kernel diff 2026-07-22); the残 excess is FA-structural. llama's `fattn-mma-f16.cuh` config
table (commit c818263f2) is the blueprint — measured configs, not guesses:

| shape | ncols (q-rows/CTA) | threads | occupancy | nbatch_fa (BK) | nstages | Q_in_reg |
|---|---|---|---|---|---|---|
| hd256, ncols 32-64 (Blackwell rows 69-72) | 32-64 | 128 | 2 | 32 | **2** | **true** |
| hd512 (rows 77-80) | 8-64 | 64-256 | 4→1 | 32 | 1 | false |

Our current kernels vs that:

- `fa_prefill_w_bf16_pp`/`_g4` (hd256 SWA): 64 rows, 4 warps, occ 2 — geometry matches, but
  **synchronous stage** (ld.global→cvt→st.shared between __syncthreads) and **sQ staged through
  smem** each Q load. llama overlaps the next K/V chunk load (cp.async, nstages=2) with the
  current chunk's mma, and holds Q in registers (no sQ traffic, no Q re-ldmatrix per step).
  Their chunking: nbatch_K2=128 — K/V staged in 128-element sub-tiles, so the double buffer
  costs 2 sub-tiles, not 2 full tiles (smem stays under the occ-2 budget). THE PORT:
  1. Q in registers (we already hold Qf fragments — drop the per-step sQ re-ldmatrix... done
     for pp body; the g4 variant re-uses load_q_frags_bf16 once — OK).
  2. cp.async 16B double-buffered K/V sub-tiles (128 elems = 16 int4 per row-chunk), ring of 2:
     preload chunk 0; loop { cp.async chunk i+1 → buf alt; cp.async.wait_group; compute chunk i }.
     bf16 source (pre-converted) keeps chunk bytes half of f32.
  3. Budget: sK/sV ring = 2 chunks × (32 rows × 128 elem × 2B) = 16KB + sP/sL ≈ 21KB/CTA →
     occ 2 holds. (Full-tile double-buffer = 64KB → occ 1 → loses; the CHUNKING is the trick.)
  Sized from kernel diff: fa_w ~40ms/prime vs llama hd256 class ~17ms → +2-3% pp1736.

- `fa_prefill_bf16_hd512_sp` (globals): 16 rows/CTA vs llama's 64 → they amortize each staged
  K/V tile over 4x more q-rows. Port: widen to 32 rows (2 warps × 16 rows each for GEMM0
  split-K stays; O per warp doubles to 64 CTiles = spill risk — so instead 4 warps × 16 rows,
  each warp owns a 128-dim V/O quarter, GEMM0 split-K 4 ways). smem: sQ 32KB + K/V chunked
  (nbatch style, not full 512) ≈ fits. Sized: fa512 ~26ms → ~15ms → +1-1.5% pp.

Order: hd256 first (bigger share: 40 SWA layers vs 8 globals). Both bit-identity-gated vs the
current kernels where op order is preserved (chunked cp.async staging does NOT change FP order —
the mma consumption order is unchanged; only the copy mechanism differs → bit-identical claim
holds for the hd256 port; the hd512 4-way split-K DOES reorder → own numeric config + battery).

Iteration rig: vast box (final numbers on the laptop only). Correctness: kernel-check windowed
gates + argmax/VERIFY battery per landing.

## §2 refined (2026-07-22 late): hd512 wide-tile geometry (from llama's 512-row config)

`fa_prefill_bf16_hd512_w64`: 64 q-rows/CTA, 8 warps (256 thr), grid (T/64, n_head).
- Q in SMEM (llama Q_in_reg=false at hd512): 64x512 bf16 = 64KB, staged once per CTA.
- K dim-CHUNKED (llama nbatch_K2=128): stage K[32 keys][128 dims] = 8KB per chunk; GEMM0
  accumulates over 4 chunks (kt order preserved within warp -> same FP order per row-dot).
- V dim-chunked likewise for GEMM1 (d0 walks chunk-local dims; O order preserved).
- Warps split ROWS (8 warps x 8 rows). Per-warp: Q re-ldmatrix from smem per kt; S = 8x32;
  softmax on own rows; O = 8 rows x 512 dims = 32 CTiles (128 regs, sp-proven).
- smem: sQ 64KB + K/V chunk ring 2x8KB + sP 8x32x2B x8 warps (4KB) + sL ~85KB -> occ 1 but
  K/V global traffic /4 vs sp (staged once per 64 rows).
- NUMERIC: per-row dot order = kt sequential as sp? NO — sp splits kt across warps (2-way
  partial sums); w64 keeps each row's FULL kt chain in ONE warp -> matches the ORIGINAL z2
  bf16 kernel's per-row order, NOT sp's. => own numeric config vs sp; gate oracle + battery.

## §2 VERDICT (2026-07-22): wide-tile CLOSED at the f32-exact class

Worked the geometry to exhaustion: (a) 64-row/8-warp with row-split warps needs 8-row warps —
breaks the 16-row CTile fragment mapping; (b) warp-pair dim-split brings back the z2 2x-GEMM0
duplication sp just removed; (c) sp-wide (32 rows, 2 row-blocks x 2 halves) needs
3x32KB(Q/K/V) + 4KB f32 partial-S + P/L = ~104KB > the 99KB sm_120 CTA cap; (d) 16-row warps
with full-512 register O = 64 f32 CTiles = 256 regs -> spills (the original z2 motivation).
llama's hd512 fits the same silicon because fa=1 accumulates S/P/O in F16 — half the register
and smem footprint for the same tile. OUR remaining hd512/hd256 FA gap vs llama is therefore a
NUMERIC-CLASS premium (we hold f32 online-softmax accumulation; they don't), not scheduling.

NEXT LEVER (legitimate, not tonight's): `BW24_FA_F16ACC` opt-in arm — f16-accum FA prefill
variants, the same speed/accuracy-tradeoff door class as BW24_MMQ W4A4 (precedented). It is
also the FAIRER A/B: llama's fa=1 default IS f16-accum, so exact-class bw24 vs f16 llama
undercounts us. Gate: full battery in-config (argmax/VERIFY/spec) + explicit flag doc.

## F16ACC vendor-port plan (2026-07-22, moved here from `cu/fattn_vendor.cu` on 2026-11-03)

This plan sat as comments in a skeleton translation unit, `crates/memra-engine/cu/fattn_vendor.cu`,
from 2026-07-22. The file never held code or joined `build.rs`, and the door it named,
`MEMRA_FA_F16ACC` (the `BW24_FA_F16ACC` lever above under the current prefix), was never read.
The file was deleted on 2026-11-03; the plan is kept here. No step below was started.

- Why: the §2 VERDICT. The remaining ~3% 12B prefill gap vs llama is a numeric-class premium.
  llama's fa=1 default accumulates S/P/O in f16, which halves the register and smem footprint
  of the same tile on the same silicon. The f32-exact FA cannot fit the wide-tile geometries
  that close the gap, and every scheduling axis was measured or closed: async ring flat,
  occupancy-2 +0.7%, tile width/rpb closed, wide tile blocked by the 99KB cap and registers
  at f32.
- Approach: the mmq vendor playbook (`mmq_q8_0.cu` precedent). Source: ggml-cuda @ `c818263f2`,
  `fattn-mma-f16.cuh` (kernel), `fattn-common.cuh` (combine, mask helpers) and `mma.cuh` (tile
  machinery; the bf16 subset is already proven in-tree in the mmq TUs, and the f16 tile ops get
  the same treatment). One ggml-decoupled, self-contained TU: everything static or internal,
  C-ABI launchers only.
- Instantiations: (DKQ, DV) = (256, 256) with ncols 32/64 (gemma SWA and hd256-global models)
  and (512, 512) with ncols 32 (gemma4 hd512 globals). type_K = type_V = F16.
- Inputs: memra prefill Q/K/V are f32, so pre-convert them with `f32_to_f16_flat` (a twin of
  the existing `f32_to_bf16_flat`). llama expresses causal and SWA masking through the KQ mask
  tensor, so build the [T, T_kv] f16 mask on device once per prime (`fattn_build_mask`).
- C-ABI: `memra_fattn_f16(Q, K, V, mask, O, dkq, dv, n_head, n_head_kv, T, T_kv, scale)` and
  `memra_fattn_f16_mask_build(mask, T, T_kv, pos0, window, causal, stream)`.
- Gating: `MEMRA_FA_F16ACC` opt-in door, on the `MEMRA_MMQ` W4A4 precedent (an explicit
  speed/accuracy tradeoff class). It is a new numeric config, so the full battery runs
  in-config before any default talk: the kernel-check oracle band at the f16 tolerance, run-gen
  argmax, VERIFY-GATE and spec self-consistency. A/B fairness note for an eventual default:
  llama's fa=1 is this numeric class, so exact-class memra against f16 llama undercounts memra.
  Under today's door rules the door also lands with its `docs/FLAGS.md` row and a decide-by
  date, and the TU goes back into `crates/memra-engine/THIRD-PARTY.md`, `tools/fast-gate/map.tsv`
  and `docs/KERNELS.md`.
- Port order. Every piece lands compilable or not at all, and the TU joins `build.rs` at step 5.
  1. `mma.cuh` f16 tile subset (`tile<I,J,half2>`, ldmatrix, mma m16n8k16 f16-accumulate).
  2. `fattn-mma-f16` config table (the CONFIG_CASE rows for 256/512, sm_120 branch).
  3. `flash_attn_ext_f16` kernel body (KQ f16 mma, online softmax in f16, VKQ f16 accumulate).
  4. `fattn-common` combine (the parallel-blocks fixup), only if nbatch_combine needs it.
  5. Mask builder, host launcher, FFI and dispatch behind `MEMRA_FA_F16ACC`.
  6. kernel-check gates (an f16-band oracle against CPU sdpa, not bit gates) and the battery.

## 31B glue lane (2026-07-23, from g31-glue-verdict): producers emit bf16

31B loses ~70ms/prime of GLUE to llama while winning GEMM and FA. Biggest safe cuts:
- `rope_neox2_bf16emit`: rope writes q/k f32 as today AND `qb/kb` bf16 (same __float2bfloat16
  the converter applied — FA operands bit-identical). Kills 2 of 3 converts + their re-reads.
- `rms_norm_qkv_w4` gains an optional `vb` bf16 emit (v is normed-but-never-roped).
- fa_prefill_* accept optional pre-converted operands (fall back to f32_to_bf16 when absent —
  non-gemma callers unchanged).
Both bit-identical on every stream -> bit-gates. Sized ~15-18ms/prime on 31B (+2%), ~+1% 12B.
NOT the block-per-row fused rms_norm_qkv_rope (would undo the warp-norm fix at depth).
