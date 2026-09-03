# lane/b200-dsa-decode-20260902: MEMRA_B200_DSA_DECODE, the t<=8 depth door

Owner target 2026-09-02: **230 tok/s plain, with the 1M window as the product.** Scope: the
t=1 (and small-t verify, t<=8) MLA/DSA decode path on 2x B200 SXM (sm_100a), GLM-5.3-Flash
NVFP4, resident PP2, plain decode. Builds on `lane/b200-mla-decode-20260902` (PR #83, the
t-keyed output-range split arm `MEMRA_B200_MLA_DECODE_ARM`), off
`origin/lane/glm5-b200-int2-20260902`.

No B200 in this worktree - the pair belongs to the spawning session. The local RTX 5090 was
used for CORRECTNESS and direction only, per the rig law.

Read the roofline first: `ROOFLINE.md` in this directory. The gate log is
`gate-5090-20260903.txt`.

## 1. What the roofline found (short form)

The task's premise sized the gathered set at `2048 x 512 x 2 B x 64 heads` = 128 MB and put a
0.18 ms floor under it. The cache is **f32** in this checkout and the DSA index list is shared
across heads, so the real number is **4.00 MiB unique per layer per token**, L2-resident.
`attn_gathered` is not a bandwidth problem: it is **190x off its f32-FFMA floor** (726.2 us vs
3.81 us/layer) and depth-FLAT because `n_slots` is pinned at the DSA top-k budget.
`kpool_score` is the depth-LINEAR half - **44x off** at 128k, ~14 ms/token extrapolated at 1M,
which is the 31.1 -> 22.7 tok/s slide from 256k to 1M - and its scan is required by the DSA
program (no score survives a decode step, because `q_t` is new every token), not by the
implementation. Full per-kernel roofline, including `kpool_select` (1.87 ms/token on ONE CTA)
and `absorb_q`/`decompress_v` (17x off their HBM floor), in ROOFLINE.md.

## 2. The door

`MEMRA_B200_DSA_DECODE`, **default OFF**, compile-gated to sm_100a, read per call (the rollback
seam is the next request). It is a LEVEL, and the level is the numeric-class boundary:

- `1` - bit-identical arms only.
- `2` - additionally admits `dsa-warp-online-f32`.

FLAGS.md row and KERNELS.md rows land in this same commit.

## 3. The three kernels, and which one survived

| kernel | class | 5090 verdict |
|---|---|---|
| `memra_mla_kpool_score_dsa_kernel<H,RP,KC>` | BIT-IDENTICAL | **10.0-13.5x**, selected at level 1 |
| `memra_mla_dsa_attn_warp_kernel<J,JP>` + combine | `dsa-warp-online-f32` | **6.3x / 3.4x**, argmax clean, selected at level 2 |
| `memra_mla_attn_gathered_dsa_kernel` | BIT-IDENTICAL | **a LOSS** (0.92x / 0.76x), NOT selected |

**The scorer** re-blocks decode scoring on (head, pool) instead of (query, pool), because at
t_q=1 the only reuse axis is heads: one thread owns `RP` pools and ALL `H` heads with
`dot[H][RP]` in registers, pool keys stream through smem in `KC` slabs stored transposed (row
stride `BP+1`), and the q slab is read `float4` over four heads at a time as a block-wide
broadcast. Shared loads per FFMA fall from 2.0 (the shipped `tiled<64,1,1,1,16>` decode config,
which has ONE accumulator per thread) to 0.156. Bit identity is a construction: the `c`-ascending
dot from `+0.0f`, the `h`-ascending head mix inside ONE thread, and all six rounding steps
spelled with explicit `__fmaf_rn`/`__fmul_rn`/`__fadd_rn`. That matters because the selection
downstream sorts on these scores with a score-DESC/index-ASC tie-break and ReLU makes exact 0.0
ties ordinary, so a last-ulp move is a different selection program.

**The warp-online arm** gives one WARP one (token, head, slot-chunk) and holds the whole
`kv_rank`-wide accumulator in registers (`J = kv_rank/32` = 16 floats per lane). Every KV
element is read from memory exactly once and consumed twice from registers; there is no
`__syncthreads` at all (warp-local fold, `__shfl_xor_sync` butterfly so every lane ends with the
sum); two `expf` per slot per warp replace ~196k per warp per layer; and `chunks` is the
occupancy knob the head axis cannot be (64 independent outputs for 148 SMs at t_q=1). It folds
per SLOT where the shipped kernel folds in 8-slot tiles, hence the named class and the argmax
gate.

## 4. The finding that killed the first design, kept on record

The single-pass bit-identical gathered kernel staged each tile's KV rows into shared memory once
(float4, serving both the score dot and the PV accumulate) and hoisted the 8 tile exponentials
into registers. Measured: **442-501 us -> 480-545 us at t_q=1, 854-882 -> 1119-1154 at t_q=4**.
Both savings were already gone:

- `expf(s_score[w] - mnew)` is loop-invariant in the `l` loop, so **nvcc had already hoisted
  it**. The "24 -> 8 exponentials per thread per tile" was a source-level count, not a
  machine-level one.
- The second `cache[tt * width + l]` pass **hits L1/L2** (4 MiB gathered set), so shared-memory
  staging adds a write and a read and buys back only L1 hits.

**Bit identity is the binding constraint on `attn_gathered`** - the shipped fold is already at a
local optimum inside it. That is what forced the named numeric class, and it is the most useful
thing this lane learned. The arm stays in the tree as arm code 1 because the gate measures it
and this receipt is the reason the door does not take it.

## 5. `dsa-decode-gate`

`crates/memra-engine/src/bin/dsa_decode_gate.rs`. Sweeps CONTEXT `{2k, 32k, 128k, 256k, 1M}` x
`t_q {1, 4}` - context, because the two stages fail in opposite ways with depth. Three hard
checks: bytewise bit-identity for the arms that claim it; an ARGMAX gate over every (token,
head) latent row plus reported maxdiff/max-relative for `dsa-warp-online-f32`; and a 5%
regression bar on the arm the SERVING POLICY selects (gate and policy read the same
`mla_ffi::MLA_DSA_ATTN_ARM`, so they cannot drift apart). Timing is interleaved: round `r` runs
every arm back to back.

```
MEMRA_CUDA_ARCH=<100a|120a> cargo build --release -p memra-engine --bin dsa-decode-gate
flock /tmp/memra-gpu.lock -c "NVIDIA_TF32_OVERRIDE=0 ./target/release/dsa-decode-gate <dev> 5"
```

(argv[3] caps the allocated latent-cache rows, default 262144 = 512 MB. On the B200 pair use the
default; the 5090 run used 65536 to stay clear of the desktop. Lock name: `/tmp/memra-gpu.lock`
per the lock-names table in CLAUDE.md; the 5090 run used `/tmp/memra-5090.lock`. If neither is
right for this box class, ask before a scored run rather than inventing a third name.)

## 6. Receipts (superseded in part by section 8; kept for the two-machine comparison)

- `dsa-decode-gate 0 3 65536` **PASS** on the local RTX 5090, 2026-09-03, release, N=3
  interleaved (`gate-5090-20260903.txt`): every bit-identical arm matched bytewise at all five
  contexts x both widths; every warp-online chunk count held argmax (0/64 and 0/256 rows moved)
  with maxdiff <= 1.8e-6, max-rel <= 3.5e-6; no policy cell regressed; zero `note` lines, i.e.
  the shipped table matches the measured winner at every cell.
- Build green on `MEMRA_CUDA_ARCH=100a` and `=120a`; `cargo fmt --all -- --check` clean;
  `cargo clippy -D warnings` clean on both arches; `tools/check-flags.sh` clean.
- **No B200 receipt.** Rig law: the 5090 microseconds are correctness plus direction, never a
  serving claim, and the door cannot even engage on a 120a build.

## 7. Open, for the session that owns the pair

1. `dsa-decode-gate <dev> 5` on the 2x B200 pair. It confirms `MLA_DSA_ATTN_ARM` (16 at t_q=1
   and t_q=4) or names the cell to change.
2. The end-to-end serving A/B at levels 0/1/2: interleaved x5, fresh boots, greedy exactness gate
   plus the vendor-default sampled twin with a spec-engagement receipt, TTFT/TPOT/ITL
   p50/p95/p99, per the per-hardware-arm-selection and never-serve-greedy laws. Level 1 is the
   bit-identical scorer alone and should be judgeable on the exactness gate; level 2 adds the
   named class and needs the sampled arm.
3. `memra_mla_kpool_select_kernel` is now the largest untouched depth item: 1.87 ms/token, ONE
   CTA at t_q=1 (0.68% of the die), ~1300x off its byte floor. It needs a hierarchical multi-CTA
   radix select with its own order-preservation argument - out of this door's scope, named here
   with the number so the next lane starts from it.
4. `docs/FLAGS.md` carries TWO rows for `MEMRA_B200_MLA_DECODE_ARM` (a merge artifact from the
   sibling lane's two cuts). Not touched here to keep this lane's diff clean; worth one dedup
   commit by whoever owns that row.


## 8. B200 receipts, the width rule, and the serving A/B (2026-09-03)

Full detail in `ROOFLINE.md` sections 8 and 9; logs `gate-b200-20260903.txt` and
`gate-5090-20260903.txt`.

### 8.1 Kernel gate on the pair (device 0, N=5, engine `f3a0091cd`)

- **Scorer BIT-IDENTICAL at every cell**, 3.97x (128k t_q=1: 167.8 -> 42.3 us), 6.86x (256k),
  5.79x (1M: 852.0 -> 147.2), and 7.31x / 7.64x / 6.98x at t_q=4.
- **Gathered, arm 32 at t_q=1**: 54.3-57.1 us against 552.1-556.3 shipped, **10.2x**, argmax
  MATCH at 128k, 256k and 1M.
- **The run FAILED as captured**: 4 ARGMAX lines at kv=131072 / t_q=4, every swept chunk count
  moving 1 of 256 latent rows.

### 8.2 What that failure bought: `MLA_DSA_NAMED_CLASS_T_MAX = 1`

The named class is now admissible at PLAIN DECODE ONLY. t_q=4..8 is the DFlash2 spec-verify
shape, where a moved argmax is a moved draft acceptance, so the spec-verify batch never sees it.
`mla_dsa_attn_arm_effective` enforces it in CODE, and demotes an offending cell to the **shipped**
kernel — the path that cannot be wrong, not the one that happens to be bit-identical at some
width nobody measured. The gate keeps MEASURING the class at t_q=4 and prints an `INFO` line when
it moves, so a future proposal to raise the rule arrives with evidence instead of an inference
from t_q=1.

### 8.3 The two machines disagreed in both directions

| | 5090 (82 SM) | B200 (148 SM) |
|---|---|---|
| single-pass at t_q=4 | 30% LOSS | 3.5-7.4% WIN |
| best chunks at t_q=1 | 16 | 32 (16 loses by 1.45x) |

Same code both times. That is the per-hardware-arm-selection law producing a real conflict, and
it is why `dsa-decode-gate`'s TIMING bar is now hard only on an sm_100a build (the only hardware
a 100a binary runs on) and prints as a `DIAGNOSTIC` elsewhere. Correctness bars stay hard on
every device. Shipped table: **t_q=1 -> 32**, **t_q=4 -> 1** (bit-identical, so the spec-verify
width carries no numeric risk at all), **0** at the unmeasured widths.

### 8.4 Serving A/B, vendor-default sampling, 256,756-token prompt

| door | 256k decode | 66-token | engagement |
|---|---|---|---|
| off | 30.07 tok/s | 48.5 | - |
| `=1` | **33.0** (+9.7%) | 48.2 | `kpool_score arm=head-blocked class=bit-identical` |
| `=2` | **43.04** (+43.1%) | 50.8 | `attn_gathered arm=warp-online chunks=16 class=dsa-warp-online-f32` |

TTFT 69.1 s in all three: prefill untouched, as designed. Flat short, +43% deep — the shape a
depth mechanism should have. And the gate PREDICTED both deltas to within 5% (scorer implied 315
us/layer vs 330.3 measured; gathered implied 750 vs the 726.2 nsys census), which is the receipt
that the synthetic shapes track the real path.

**The +43.1% was measured with chunks=16 and the shipped table now says 32.** That serving number
is a LOWER BOUND for the shipped configuration, not a receipt of it. The edit is worth ~0.27
ms/token at 256k (~43.0 -> ~43.5 tok/s), not a second +43%.

### 8.5 Default-ON candidacy

Candidate on sm_100a at both levels (`=1` everywhere, `=2` at t_q=1); still OFF here. Open:

1. ~~128-token greedy tape on the real artifact, off vs `=2`~~ — **DONE 2026-09-03, GREEN**:
   sha16 `9437b599f6b9d2a9` for door off, `=1` and `=2` alike, matching every other posture
   today; the `=2` run engaged `attn_gathered t=1 arm=warp-online chunks=16
   class=dsa-warp-online-f32` at 43.0 tok/s. The argmax gate is the kernel-level admission bar
   for a named class and the tape is the model-level one; both are now green, and neither
   substitutes for the other.
2. Serving A/B re-run on the shipped table (chunks=32).
3. kv=2048 and kv=32768 are absent from the relayed B200 capture.

`=1` is the easier call: bit-identical at every measured cell, +9.7% at 256k, no class to admit.

### 8.6 Next lane: `kpool_select`

Now the largest MLA/DSA item the door does not touch: ~170 us/layer, **1.87 ms/token**, ONE CTA
at t=1 (0.68% of the die), 8 radix passes over `n_pools`, depth-linear. It needs a hierarchical
multi-CTA radix select — and it can be made **EXACT rather than banded**, because the 64-bit
order key `(desc32(score) << 32) | pool_index` is already a total, tie-broken, distinct order, so
a hierarchical select over it returns the same set the single-CTA kernel does. That argument is
the lane's starting point.
