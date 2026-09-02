# lane/b200-mla-decode-20260902: B200 t<=8 head-parallel, launch-light MLA/DSA decode arm

Owner order (2026-09-02): "hardly improve the decode on these cards, before the full 1M."
Scope: the t=1 (and small-t verify, t<=8) MLA/DSA sparse-attention decode path on 2x B200
SXM (sm_100a), GLM-5.3-Flash NVFP4, resident PP2, plain decode. No GPU in this worktree —
the B200 box belongs to the spawning session, which owns the actual gate/bench run there.
This lane worked from `/home/avifenesh/projects/wt-b200-mla` (branch
`lane/b200-mla-decode-20260902`, off `lane/glm5-b200-20260902`), touching only
`crates/memra-engine/cu/mla_attn.cu` and `crates/memra-engine/src/mla_ffi.rs`, plus docs
and the gate bin. Did not touch `cu/qmatvec.cu` or `cu/dsv4_gpu.cu` (other lanes).

## 1. The census claim, and what tracing the code actually shows

The task's motivating nsys measurement (2x B200 SXM, plain decode t=1) named:

- `memra_mla_absorb_q_kernel` 16 x 71.0 us, `memra_mla_decompress_v_kernel` 16 x 70.9 us,
  `memra_mla_attn_gathered_kernel` 16 x 42.3 us, `memra_mla_kpool_select_kernel` 16 x 10.2
  us, `memra_mla_kpool_score_ref_kernel` 16 x 9.8 us — per-token GPU cost of the 11 MLA/DSA
  sparse layers.
- 321 cuBLAS gemv-class launches per token (`dot_kernel` 3.4 us + `reduce_1Block` 2.0 us
  each — the cublasLt m=1 algorithm) attributed to `memra_bf16_gemm_sb`
  (cu/f16_prefill.cu), batch=64 heads, n=512, k=256 "per the boot log" — a shape that
  matches the absorb_q geometry (n_head=64, kv_rank=512, d_nope=256) exactly.

**Tracing every call site of `memra_bf16_gemm_sb` in this checkout says decode cannot
reach it.** The only caller is `Engine::mla_bf16_gemm_sb_raw` (mla_ffi.rs), whose only
caller is `mla_tc_prefill_chain` (hybrid_forward.rs), which is gated:

```
if let Some((idx, slots)) = &gathered
    && dr == 0 && r == 512 && t >= 16 && !rows_exact
    && !crate::portable_mma_gated() && !mla.tp_shard
    && mla_tc_prefill_enabled()
    && let Some(attn) = self.mla_tc_prefill_chain(...)?
```

`t >= 16` is explicit and, per the comment at that site, deliberate: "Decode (t == 1) and
short resumes NEVER enter, which is what the decode byte-identity gate proves rather than
assumes." This matches the existing `MEMRA_MLA_TC_PREFILL` FLAGS.md row verbatim
("decode untouched", box A/B receipts) and the 2026-08-30 launch-diet census that row
cites, which attributes `memra_mla_attn_gathered_kernel` 139.1 ms + `absorb_q` 44.5 ms +
`decompress_v` 43.6 ms to a **cold 4626-token PRIME**, not decode. `git log -p -S"t >= 16"`
on hybrid_forward.rs shows this guard predates this lane (catch-up-sync commit
`49d1d6f65`); it was not added or weakened by any concurrent lane.

**What this lane did NOT do, and why:** build a decode-time custom kernel to replace
`memra_bf16_gemm_sb`, because that code path does not execute at t=1 in this checkout —
fabricating a replacement for a call that never happens would be invented work, not a fix.
Two ways to reconcile this before that work is built:

1. The nsys/boot-log evidence was captured on a different binary or a stale build (an
   older commit without the `t >= 16` guard, or a build with `MEMRA_MLA_TC_PREFILL`
   forced on in a way that bypassed the guard some other way this lane did not find).
2. The "boot log" printed the shape as a one-time cuBLASLt plan warm-up (the `F16Plan`
   cache in f16_prefill.cu prints/keys on first use per shape), and the nsys trace's 321
   launches per token are being attributed to the wrong symbol by a name-matching step
   rather than an NVTX range around the actual call.

Either way, the fix is to re-run nsys against a fresh `cargo build --bin <server>` of
THIS commit and re-check the attribution before building a decode-time GEMM replacement.
This is an explicit open item for the spawning session (which has both the box and nsys).

## 2. What the launch-geometry investigation found instead

Reading `cu/mla_attn.cu` (absorb_q ~L191, decompress_v ~L226, kpool_score_ref ~L626,
kpool_select ~L1110, attn_gathered ~L1291) shows `memra_mla_absorb_q_kernel`,
`memra_mla_decompress_v_kernel` and `memra_mla_attn_gathered_kernel` already grid
`blockIdx.x = i * n_head + h` — **already head-parallel by construction**, 64 CTAs at
t_q=1 on the glm5 geometry (n_head=64). This contradicts this lane's own opening
hypothesis ("grid = t rows"). The real problem is CTA count, not parallelism shape: 64
CTAs under-fills a B200 die (~132-148 SMs), and each block does 256-thread work with a
serial, `__syncthreads`-per-tile loop and no float4/uint4 vectorization — latency, not
throughput, bound.

`memra_mla_kpool_select_kernel` grids `t_q` blocks (ONE block at t=1) — the single
worst-parallelized kernel in the family — but it does a fused head-mixed radix-select
over the whole pool per query, so there is no independent-output axis to split it on the
same way; out of scope per the task's kernel list (absorb_q / decompress_v /
attn_gathered only) and left untouched.

## 3. What already existed: MEMRA_MLA_DECODE_SPLIT (lane/glm5-decode-diet, 2026-08-31)

`memra_mla_absorb_q_split_kernel` / `memra_mla_decompress_v_split_kernel` already exist
(commit `f6cd746ce`) and do exactly "grid over heads x row tiles, more CTAs": each
(token, head) block's OUTPUT RANGE is split across `split` blocks, and since every output
element is one thread's independent serial dot, the bytes are provably identical at any
split value. `MEMRA_MLA_DECODE_SPLIT` wires this at a rig-generic ~1024-block target,
default OFF, with a PRO6000-measured 1.0148x (-0.411ms, `diet-battery` commit) partial
result — real but small, and never promoted to default-ON.

## 4. This lane's mechanism: MEMRA_B200_MLA_DECODE_ARM

Per the per-hardware-arm-selection law (CLAUDE.md), a B200-specific door is the right
shape here, not a rename of the generic one: B200 SXM carries more SMs per device than
the PRO6000 pair the generic door was tuned on, and B200-vs-PRO6000 could easily disagree
on which split factor wins (the law's own `MEMRA_MOE_GROUPED` precedent).

- **absorb_q / decompress_v**: reuse the EXISTING split kernels (no new CUDA), wired
  through a NEW B200-tuned policy (`mla_b200_split_for`, mla_ffi.rs) targeting ~2048
  blocks instead of ~1024, checked FIRST in `mla_absorb_q`/`mla_decompress_v` (ahead of
  the generic door, so the two compose rather than race).
- **attn_gathered**: NEW kernel `memra_mla_attn_gathered_split_kernel` (cu/mla_attn.cu,
  appended after the existing `memra_mla_attn_gathered_f32` launcher). No split existed
  for this kernel before this lane.
- Compile-time gated to sm_100a builds only: `mla_b200_decode_arm_on()` is
  `cfg!(memra_sm100_tcgen05) && MEMRA_B200_MLA_DECODE_ARM == "1"`. On a 120a/90a/89 build
  the door is `false` unconditionally — dead code, correctly, off-target. Scoped to
  `t_q <= 8` per the owner's order.
- Door: `MEMRA_B200_MLA_DECODE_ARM` (default OFF, read per call). FLAGS.md row: default,
  both arms, rollback seam, receipt-pending status (below).

### 4.1 Why NOT a slot-range split for attn_gathered (the bit-identity argument)

absorb_q/decompress_v are independent-output matvecs: splitting WHICH block computes
output element `l` changes nothing about the sequence of floating-point operations that
produce it, so bit identity is free. `attn_gathered`'s output elements all share ONE
softmax normalizer (`m`, `dsum`) computed by walking every tile of the gathered slot
list. Splitting that WALK across CTAs (segment the slots, combine partial `(m, dsum,
acc)` triples) changes the NUMBER and ORDER of online-softmax rescale operations relative
to the single sequential fold — a provably-equal-in-real-arithmetic but NOT
bit-identical-in-floating-point transformation (the classic flash-attention split-K
non-associativity: an 8-tile sequential fold does 8 rescale combines; a 4+4 split-then-
merge does 4 + 4 + 1 = 9, at different operand pairings). That fails this lane's
bit-identity bar, so it was refused.

What IS bit-identical: splitting the OUTPUT WRITE RANGE the way absorb_q_split does.
Every split block reruns the shared score/softmax tile walk **in full** (same values,
same rounding, because the walk doesn't depend on which output range this block owns) and
only the final per-`l` accumulate-and-write loop is restricted to `[lo, hi)`. Each kept
output element's accumulate chain is therefore the exact same sequence of operations the
unsplit kernel computes for that element — only WHICH block runs it differs. Implemented
as `memra_mla_attn_gathered_split_kernel`.

**Trade-off, stated plainly and confirmed by the correctness-rig timing below:** every
split factor here also DUPLICATES the dominant score/softmax compute (the walk is redone
per split block), unlike the near-zero-marginal-cost absorb/decompress splits. The host
policy (`mla_b200_gathered_split_for`) caps this conservatively (factor <= 4, target 512
blocks) rather than chasing the same ~2048-block target as the independent-output
kernels.

## 5. Correctness receipt (run in this session, on the local 5090)

No GPU in this worktree by design (the B200 belongs to the spawning session), but a local
RTX 5090 dev rig was present in the sandbox. Per the standing rig law ("5090 laptop
throttles; correctness gates OK, timing numbers never" — docs/PERFORMANCE.md), this
session used it for CORRECTNESS ONLY: built `MEMRA_CUDA_ARCH=120a` (native to the 5090;
the `MEMRA_B200_MLA_DECODE_ARM` door itself cannot even compile-time-engage on this
build, so the gate bin calls the split kernels directly through their raw FFI, bypassing
the door, to prove the KERNELS are correct independent of which arch's door will use
them), ran under `flock /tmp/memra-5090.lock`, `NVIDIA_TF32_OVERRIDE=0`:

```
$ flock /tmp/memra-5090.lock -c "NVIDIA_TF32_OVERRIDE=0 ./target/debug/mla-decode-arm-gate 0"
mla-decode-arm-gate: device 0, geometry nh=64 kv_rank=512 d_nope=256 d_v=256 d_rope=0 n_slots=2048 pool_rows=32768
== t_q=1 ==
  absorb_q split=2: BIT-IDENTICAL
  absorb_q split=4: BIT-IDENTICAL
  absorb_q split=8: BIT-IDENTICAL
  decompress_v split=2: BIT-IDENTICAL
  decompress_v split=4: BIT-IDENTICAL
  decompress_v split=8: BIT-IDENTICAL
  attn_gathered split=2: BIT-IDENTICAL
  attn_gathered split=4: BIT-IDENTICAL
== t_q=4 ==
  absorb_q split={2,4,8}: BIT-IDENTICAL
  decompress_v split={2,4,8}: BIT-IDENTICAL
  attn_gathered split={2,4}: BIT-IDENTICAL
mla-decode-arm-gate PASS: every split arm BIT-IDENTICAL to its shipped kernel at t_q in {1,4}
```

Every tested split factor (2, 4, 8 for absorb_q/decompress_v; 2, 4 for attn_gathered) at
both t_q=1 and t_q=4, BIT-IDENTICAL, zero mismatches. This is the correctness gate the
task asked for, actually run, not merely argued.

### 5.1 5090 timing — diagnostic only, NOT a serving/perf claim (rig law)

Printed by the same run (N=5 per arm, mean us). Reported here because it is decisive
directional signal that changes what should be A/B'd on the B200, not because a 5090
laptop number is ever a serving claim:

| kernel | t_q | shipped mean | split=4 (absorb/decompress) or split=2 (gathered) mean | delta |
|---|---|---|---|---|
| absorb_q | 1 | 56.7 us | 57.6 us | -1.6% (flat/worse) |
| absorb_q | 4 | 243.1 us | 179.3 us | +26.3% faster |
| decompress_v | 1 | 55.7 us | 56.4 us | -1.3% (flat/worse) |
| decompress_v | 4 | 205.2 us | 168.9 us | +17.7% faster |
| attn_gathered | 1 | 467.2 us | 449.2 us (split=2) | +3.9% faster |
| attn_gathered | 4 | 865.5 us | 1453.9 us (split=2) | **-68.0% (much worse)** |

Reading this honestly: on THIS rig, the independent-output splits (absorb/decompress)
show a real win at t_q=4 and a wash at t_q=1 (launch overhead roughly cancels the
occupancy gain at only 64->256 blocks on a smaller die) — consistent with the PRO6000
diet-battery result being small-but-real. The gathered split shows a MODEST win at t_q=1
but a LARGE LOSS at t_q=4 — direct confirmation of the redundant-score-walk trade-off in
§4.1: at t_q=4 there are already 256 blocks (4*64) before splitting, so occupancy was
likely already adequate, and split=2 then pays 2x the dominant score walk for no
occupancy gain. This is a real, measured (if non-serving) caution against the gathered
split at anything but the most starved (t_q=1, 64-block) regime — worth weighing when the
B200 A/B is designed, since a 5090's ~170 SM-equivalent occupancy floor is a different
number from a B200 die's, and the crossover point where splitting stops paying for itself
needs its own B200 measurement, not an assumption that "more CTAs always wins."

## 6. Build / lint receipts

- `MEMRA_CUDA_ARCH=120a cargo build -p memra-engine` (whole crate): clean, `Finished` in
  4m21s, this session.
- `MEMRA_CUDA_ARCH=100a cargo build -p memra-engine --bin mla-decode-arm-gate`: clean
  (cross-compiled SASS for sm_100a on hardware that cannot run it — nvcc does not require
  the target device to be present, only the toolkit's arch support, which CUDA 13.1 has
  for `sm_100a`).
- `cargo fmt --all -- --check`: clean (two files needed one `cargo fmt --all` pass,
  applied).
- `tools/check-flags.sh`: `no uncovered runtime names` — `MEMRA_B200_MLA_DECODE_ARM`
  resolves against its new `docs/FLAGS.md` row.
- `cargo clippy -p memra-engine --bin mla-decode-arm-gate --no-deps`: see PR receipts for
  the final run (background at lane-close time).

## 7. Gate invocation

```
cd crates/memra-engine
MEMRA_CUDA_ARCH=<100a|120a> cargo build -p memra-engine --bin mla-decode-arm-gate
flock /tmp/memra-5090.lock -c \
  "NVIDIA_TF32_OVERRIDE=0 ./target/debug/mla-decode-arm-gate <device_ordinal>"
```

On the B200 box, replace the lock with `/tmp/memra-gpu.lock` per the lock-names table in
CLAUDE.md (this is a 2x B200 pair, not the 5090 dev rig or a RTX PRO 6000/RunPod pod — if
neither established lock name is correct for this box class, that is itself a question
for the spawning session before any scored run, per the "lock names are a correctness
surface" law: do not invent a third name).

## 8. Open items (for the spawning session, on the B200 box)

1. **Reconcile the cuBLASLt/321-launch attribution (§1).** Re-run nsys against a fresh
   build of this exact commit's server binary at t=1 and confirm whether
   `memra_bf16_gemm_sb` genuinely fires at decode on the B200 (it should not, per the
   `t >= 16` guard) or whether the original trace mis-attributed prime-time cost, a
   one-time plan-warmup print, or a different build. This determines whether a
   decode-time GEMM replacement kernel is even a real task.
2. **The B200 A/B itself — the actual perf receipt this door needs before any default
   flip.** Per the per-hardware-arm-selection and never-serve-greedy laws: interleaved
   x5, both arms (`MEMRA_B200_MLA_DECODE_ARM=0/1`), fresh boots, greedy exactness gate +
   vendor-default sampled twin with a spec-engagement receipt, real prompts, TTFT/TPOT/ITL
   p50/p95/p99, under `/tmp/memra-gpu.lock` (or the correct lock name for this box class,
   see §7). Sweep the two split-factor policies independently (absorb/decompress target
   block count; the gathered split's cap, or whether to disable it) — §5.1's 5090 signal
   suggests the gathered split may want a NARROWER engagement window than t_q<=8 (e.g.
   t_q==1 only), which only a B200 measurement can settle.
3. **`memra_mla_kpool_select_kernel`'s single-block-at-t=1 geometry** (§2) is a genuine
   remaining under-parallelized kernel (10.2 us x 16/token) but has no independent-output
   axis to split on with this lane's bit-identity technique — a real launch-geometry
   improvement here needs either a hierarchical/multi-block radix select (own reduction
   restructure, its own bit-identity argument) or accepting a banded, not bit-identical,
   numeric class. Out of this lane's scope (task named absorb_q/decompress_v/
   attn_gathered only); flagged for whoever picks up kpool_select next.
4. Default flip decision, FLAGS.md update to DEFAULT ON, and the two-box (B200 class + a
   second B200 or cross-check box per the two-rig evidence rule) sign-off all wait on
   item 2's receipt.
