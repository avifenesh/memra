//! FFI declarations + safe Engine wrappers for the MLA CUDA forward (`cu/mla_attn.cu`).
//!
//! House pattern (mmq_ffi / dsv4_ffi kind): C-ABI host launchers in the `libmemra_mmq.a`
//! static lib, returning 0 ok / 10000+cudaError / 40000+contract; the stream rides as
//! `*mut c_void` (`stream.cu_stream()`).
//!
//! The numeric truth for the dense core is `crate::mla` (the CPU f32 oracle), gated in
//! `tests/mla_gpu_forward.rs`. The truth for the DSA k-pool indexer wrappers at the bottom of
//! this file is `memra_reference::kpool_allowed_tokens`, gated in
//! `tests/glm5_kpool_indexer_gpu.rs`.

use crate::Engine;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use std::os::raw::c_void;

/// Engagement counter for the MLA decode-split door (`MEMRA_MLA_DECODE_SPLIT`): counted at
/// the arm's own call site, announced once per boot — the receipt a box A/B arm must show.
pub static MLA_DECODE_SPLIT_DISPATCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `MEMRA_MLA_DECODE_SPLIT=1` (default OFF, read per call — rollback seam): the absorb /
/// decompress launchers split each (token, head) block's output range across several blocks.
/// PURE LAUNCH GEOMETRY: every output element keeps the same one-thread serial dot, so the
/// bytes are identical for every split value (asserted in `tests/mla_decode_split_gpu.rs`);
/// only occupancy changes — 64 blocks at t=1 on the glm5 geometry is single-digit-percent
/// occupancy on the serving card class, the census's ~211 us/layer absorb+decompress pair.
/// Engagement counter for the coalesced warp-per-row door (`MEMRA_MLA_COALESCE`).
pub static MLA_COALESCE_DISPATCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `MEMRA_MLA_COALESCE=1` (default OFF, read per call — the rollback seam is its absence):
/// dispatch the warp-per-row absorb/decompress kernels instead of the shipped thread-per-row
/// ones. See `memra_mla_absorb_q_wp_f32` for the defect and the numeric class.
///
/// It composes with whichever split door is armed rather than replacing it: the split doors
/// choose the output-range partition (the GRID), this door chooses how a row is read (the
/// LOADS), and `split == 1` is simply the unsplit partition. So the dispatch below asks the
/// existing policy for a split first and passes whatever it returns.
fn mla_coalesce_on() -> bool {
    std::env::var("MEMRA_MLA_COALESCE").as_deref() == Ok("1")
}

/// Announce once per boot, naming the kernel and the split it composed with, because a door
/// that silently changes which kernel runs is the failure class this lane spent a day on.
fn mla_coalesce_announce(which: &str, t_q: usize, n_head: usize, split: i32) {
    use std::sync::atomic::Ordering;
    if MLA_COALESCE_DISPATCHES.fetch_add(1, Ordering::Relaxed) == 0 {
        eprintln!(
            "[mla-coalesce] engaged {which} t_q={t_q} n_head={n_head} split={split} \
             (warp-per-row coalesced loads + shuffle reduction; numeric class \
             mla_warp_row_reduce; MEMRA_MLA_COALESCE=1)"
        );
    }
}

fn mla_decode_split_on() -> bool {
    std::env::var("MEMRA_MLA_DECODE_SPLIT").as_deref() == Ok("1")
}

/// The split policy: engage only in the block-starved regime (fewer than 1024 (token, head)
/// blocks — decode and short verify widths; prefill widths already fill the card and the TC
/// prefill chain owns them anyway), aiming for ~1024 blocks while keeping at least 32 outputs
/// per block. The OUTPUT BYTES ARE SPLIT-INVARIANT by construction, so this arithmetic is a
/// throughput policy, never a numerics decision.
fn mla_decode_split_for(blocks: usize, out_dim: usize) -> Option<i32> {
    if !mla_decode_split_on() || blocks == 0 || blocks >= 1024 {
        return None;
    }
    let want = 1024usize.div_ceil(blocks);
    let cap = (out_dim / 32).max(1);
    let split = want.min(cap);
    if split <= 1 { None } else { Some(split as i32) }
}

fn mla_split_announce(kind: &str, t_q: usize, n_head: usize, split: i32) {
    use std::sync::atomic::Ordering;
    if MLA_DECODE_SPLIT_DISPATCHES.fetch_add(1, Ordering::Relaxed) == 0 {
        eprintln!(
            "[mla-decode-split] engaged {kind} t={t_q} heads={n_head} split={split} \
             (output-range split of the (token, head) blocks; MEMRA_MLA_DECODE_SPLIT=1)"
        );
    }
}

/// Engagement counter for the B200 decode arm (`MEMRA_B200_MLA_DECODE_ARM`), announced once
/// per boot: the receipt a B200 box A/B arm must show.
pub static MLA_B200_DECODE_ARM_DISPATCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `MEMRA_B200_MLA_DECODE_ARM=1` (default OFF, read per call: the rollback seam), compile-time
/// gated to sm_100a builds (`cfg!(memra_sm100_tcgen05)`, set by build.rs for
/// `MEMRA_CUDA_ARCH=100a`): on a 120a/90a/89 build this is `false` unconditionally, so naked
/// non-B200 commands and the flag census see no behavior change from a var they cannot even
/// engage. The arch guard is a compile-time fact here, not a per-call detection cost.
///
/// Owner order 2026-09-02: "hardly improve the decode on these cards, before the full 1M."
/// This is a genuinely separate door from `MEMRA_MLA_DECODE_SPLIT` (glm5-decode-diet lever 4,
/// rig-generic, target ~1024 blocks, PRO6000-tuned) rather than a rename of it, per the
/// per-hardware-arm-selection law in CLAUDE.md: B200 SXM carries more SMs per device than the
/// PRO6000 pair that door was tuned on, and this arm ALSO covers `attn_gathered`, which the
/// generic split door never touched (no independent-output split existed for it before this
/// lane; see `memra_mla_attn_gathered_split_kernel` in cu/mla_attn.cu).
fn mla_b200_decode_arm_on() -> bool {
    cfg!(memra_sm100_tcgen05) && std::env::var("MEMRA_B200_MLA_DECODE_ARM").as_deref() == Ok("1")
}

/// The three kernels the B200 arm covers. The gate bin (`mla_decode_arm_gate.rs`) walks this
/// same enum and the same table below, so its regression check and the serving policy cannot
/// disagree about which split a t_q gets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlaB200Kernel {
    AbsorbQ,
    DecompressV,
    AttnGathered,
}

impl MlaB200Kernel {
    pub const ALL: [MlaB200Kernel; 3] = [
        MlaB200Kernel::AbsorbQ,
        MlaB200Kernel::DecompressV,
        MlaB200Kernel::AttnGathered,
    ];

    pub fn name(self) -> &'static str {
        match self {
            MlaB200Kernel::AbsorbQ => "absorb_q",
            MlaB200Kernel::DecompressV => "decompress_v",
            MlaB200Kernel::AttnGathered => "attn_gathered",
        }
    }
}

/// Widest query width the B200 arm keys on. Wider widths fall through untouched: the generic
/// `MEMRA_MLA_DECODE_SPLIT` door if set, else the shipped kernels (t >= 16 reaches
/// `MEMRA_MLA_TC_PREFILL` before either).
pub const MLA_B200_ARM_T_MAX: usize = 8;

/// The B200 arm's split tables, keyed on t_q (index = t_q in 1..=MLA_B200_ARM_T_MAX; index 0
/// is unused and always 1). A cell of 1 means THE SHIPPED KERNEL: the wrapper falls through to
/// the unsplit launcher and the split twin is never launched with split=1, so "shipped" is the
/// shipped binary path, not a re-implementation of it. Any other cell is the output-range
/// split factor handed to the bit-identical split twin.
///
/// Why a table and not a block-count target: the first cut of this door aimed at ~2048 blocks
/// at every t_q <= 8 and the real box refuted that shape. Measured 2026-09-02 on the 2x B200
/// SXM pair (sm_100a), `mla-decode-arm-gate` device 0, geometry nh=64 kv_rank=512 d_nope=256
/// d_v=256 d_rope=0 n_slots=2048 pool_rows=32768, N=5, every arm BIT-IDENTICAL to shipped:
///
/// | kernel        | t_q | shipped  | arm           | verdict                   |
/// |---------------|-----|----------|---------------|---------------------------|
/// | absorb_q      | 1   | 81.8 us  | split=4 49.1  | win                       |
/// | decompress_v  | 1   | 82.2 us  | split=4 48.0  | win                       |
/// | attn_gathered | 1   | 564.6 us | split=2 516.4 | win                       |
/// | absorb_q      | 4   | 150.3 us | split=4 133.2 | win                       |
/// | decompress_v  | 4   | 150.4 us | split=4 246.6 | REGRESSION, shipped wins  |
/// | attn_gathered | 4   | 665.3 us | split=2 822.7 | REGRESSION, shipped wins  |
///
/// t_q=4..8 is the DFlash2 spec-verify shape the box serves, so a target-driven policy that
/// splits there costs the spec route. The tables ship exactly what that run showed and nothing
/// it did not: the measured winner at t_q=1 for all three kernels, absorb_q's measured split=4
/// win at t_q=4, and the shipped kernel everywhere else (t_q=2,3,5..8 are unmeasured, and
/// unmeasured behavior does not go on). The gate times every split in {1,2,4,8} at every t_q
/// in {1,2,4,8} for all three kernels, prints the per-t winner table, and FAILS (`REGRESSION`,
/// exit 1) when a cell of THESE tables is slower than shipped by more than
/// `MLA_B200_ARM_REGRESSION_MARGIN`, so a box run either confirms the tables or names the cell
/// to change. Cite the box run in this comment when editing a cell.
pub const MLA_B200_ABSORB_Q_SPLIT: [i32; MLA_B200_ARM_T_MAX + 1] = [1, 4, 1, 1, 4, 1, 1, 1, 1];
pub const MLA_B200_DECOMPRESS_V_SPLIT: [i32; MLA_B200_ARM_T_MAX + 1] = [1, 4, 1, 1, 1, 1, 1, 1, 1];
pub const MLA_B200_ATTN_GATHERED_SPLIT: [i32; MLA_B200_ARM_T_MAX + 1] = [1, 2, 1, 1, 1, 1, 1, 1, 1];

/// The gate's regression bar: at every measured t_q the table's arm may not be slower than
/// shipped by more than 5% (arm/shipped above this ratio fails `mla-decode-arm-gate`).
pub const MLA_B200_ARM_REGRESSION_MARGIN: f64 = 1.05;

/// Table lookup, independent of the door: 1 (shipped) outside 1..=MLA_B200_ARM_T_MAX. Pure,
/// so the gate bin can read the table on any build, including the 120a builds where the door
/// itself cannot engage.
pub fn mla_b200_arm_table_split(kernel: MlaB200Kernel, t_q: usize) -> i32 {
    if t_q == 0 || t_q > MLA_B200_ARM_T_MAX {
        return 1;
    }
    match kernel {
        MlaB200Kernel::AbsorbQ => MLA_B200_ABSORB_Q_SPLIT[t_q],
        MlaB200Kernel::DecompressV => MLA_B200_DECOMPRESS_V_SPLIT[t_q],
        MlaB200Kernel::AttnGathered => MLA_B200_ATTN_GATHERED_SPLIT[t_q],
    }
}

/// The serving policy: door on, table cell above 1, and the cell legal for this geometry (the
/// split twins need `split <= out_dim`; this keeps at least 32 outputs per block, the same
/// floor as the generic door). A cell the geometry cannot honour falls through to the shipped
/// kernel rather than clamping to a split the box never measured. The tables were measured on
/// the glm5 geometry (kv_rank=512, d_v=256); `None` here means "shipped path", and the caller
/// falls through in order to the generic split door, then the unsplit launcher.
fn mla_b200_split_for(kernel: MlaB200Kernel, t_q: usize, out_dim: usize) -> Option<i32> {
    if !mla_b200_decode_arm_on() {
        return None;
    }
    let split = mla_b200_arm_table_split(kernel, t_q);
    let cap = (out_dim / 32).max(1) as i32;
    if split <= 1 || split > cap {
        None
    } else {
        Some(split)
    }
}

fn mla_b200_split_announce(kind: &str, t_q: usize, n_head: usize, split: i32) {
    use std::sync::atomic::Ordering;
    if MLA_B200_DECODE_ARM_DISPATCHES.fetch_add(1, Ordering::Relaxed) == 0 {
        eprintln!(
            "[mla-b200-decode-arm] engaged {kind} t={t_q} heads={n_head} split={split} \
             (sm_100a output-range split; MEMRA_B200_MLA_DECODE_ARM=1)"
        );
    }
}

/// Engagement counter for the DSA decode door (`MEMRA_B200_DSA_DECODE`), announced once per
/// boot per arm: the receipt a B200 box A/B has to show.
pub static MLA_DSA_DECODE_DISPATCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `MEMRA_B200_DSA_DECODE` (default OFF = 0, read per call: the rollback seam), compile-time
/// gated to sm_100a builds exactly like its sibling `MEMRA_B200_MLA_DECODE_ARM`, so a
/// 120a/90a/89 build sees no behavior change from a var it cannot engage.
///
/// THE DOOR IS A LEVEL, not a boolean, and the level is the numeric-class boundary:
///
/// * `0` (default) - nothing engages. Every kernel is the shipped one.
/// * `1` - the BIT-IDENTICAL arms only: `memra_mla_attn_gathered_dsa_kernel` (same fold, same
///   lane stride, same shuffle tree; what changes is that each tile's KV rows are staged once
///   into shared memory with float4 loads and serve BOTH the score dot and the PV accumulate,
///   and that the 8 tile exponentials are hoisted into registers instead of being recomputed
///   3x per thread per tile) and `memra_mla_kpool_score_dsa_kernel` (head-blocked decode
///   scorer; c-ascending dot, h-ascending mix, all six rounding steps spelled with explicit
///   intrinsics). Both are asserted bytewise by `dsa-decode-gate`, not merely argued.
/// * `2` - additionally admits the WARP-ONLINE gathered arm
///   (`memra_mla_dsa_attn_warp_kernel` + `memra_mla_dsa_attn_combine_kernel`), numeric class
///   **`dsa-warp-online-f32`**: one warp owns one (token, head, slot-chunk) and holds the whole
///   kv_rank-wide accumulator in registers, so every KV element is read from memory ONCE and
///   consumed twice from registers, there is not a single `__syncthreads`, and the two `expf`
///   per slot replace ~196k per warp per layer. It folds PER SLOT and merges `chunks` partials,
///   where the shipped kernel folds in 8-slot tiles: same sum in real arithmetic, different
///   rounding, so `dsa-decode-gate` holds it to an ARGMAX gate plus a maxdiff/max-relative bound
///   on real-shaped inputs and never to bit identity. It exists because at t_q=1 the gathered
///   attention has exactly 64 independent (token, head) outputs for 148 SMs and the slot axis is
///   the only one that buys parallelism without duplicating the walk. ADMISSIBLE ONLY AT
///   `t_q <= MLA_DSA_NAMED_CLASS_T_MAX` = 1 (plain decode): the 2026-09-03 B200 run saw this
///   class move 1 of 256 latent-row argmaxes at kv=131072 / t_q=4, and t_q=4..8 is the DFlash2
///   spec-verify shape where a moved argmax is a moved draft acceptance. Enforced by
///   `mla_dsa_attn_arm_effective`, not by table convention.
///
/// Rollback seam: unset the var (or set 0). Both arms are read per call, so a rollback is the
/// next request, not a restart.
fn mla_dsa_decode_level() -> u32 {
    if !cfg!(memra_sm100_tcgen05) {
        return 0;
    }
    match std::env::var("MEMRA_B200_DSA_DECODE").as_deref() {
        Ok("1") => 1,
        Ok("2") => 2,
        _ => 0,
    }
}

/// Widest query width the DSA decode door keys on. Wider widths fall through untouched: the
/// `MEMRA_B200_MLA_DECODE_ARM` split door if set, then the shipped kernels (t >= 16 reaches
/// `MEMRA_MLA_TC_PREFILL` before either).
pub const MLA_DSA_ARM_T_MAX: usize = 8;

/// Which gathered-attention arm the door selects, keyed on t_q (index = t_q in
/// 1..=MLA_DSA_ARM_T_MAX; index 0 unused):
///
/// * `0` - the SHIPPED kernel (`memra_mla_attn_gathered_f32`). The door does nothing here.
/// * `1` - the single-pass BIT-IDENTICAL kernel (`memra_mla_attn_gathered_dsa_f32`).
/// * `n >= 2` - the WARP-ONLINE arm with `n` slot chunks, numeric class
///   `dsa-warp-online-f32`. Level 2 only.
///
/// B200-MEASURED, 2026-09-03, and the widths are not free: read the width rule below before
/// editing a cell. `dsa-decode-gate` on the 2x B200 SXM pair (sm_100a), device 0, N=5
/// interleaved, engine commit f3a0091cd; banked log
/// `darklanes:research/glm5-b200-20260902/box/gates/gate-dsa-decode.txt`. Means in us, and the
/// gathered stage is depth-flat so the three contexts agree to a few percent:
///
/// | t_q | context | shipped | single-pass (1) | c=4 | c=8 | c=16 | **c=32** |
/// |---|---|---|---|---|---|---|---|
/// | 1 | 128k | 556.3 | 573.1 | 272.9 | 136.9 | 80.5 | **57.1** |
/// | 1 | 256k | 553.3 | 572.4 | 273.3 | 136.7 | 79.7 | **54.8** |
/// | 1 | 1M | 552.1 | 571.5 | 273.1 | 139.0 | 79.9 | **54.3** |
/// | 4 | 128k | 641.3 | **618.8** | 276.9 | 156.0 | 159.6 | 131.3 |
/// | 4 | 256k | 663.1 | **616.6** | 277.1 | 154.7 | 158.9 | 131.8 |
/// | 4 | 1M | 667.6 | **618.5** | 277.5 | 156.5 | 160.1 | 132.3 |
///
/// THE WIDTH RULE, and it is a correctness rule, not a tuning one. The named class
/// (`dsa-warp-online-f32`, arm >= 2) is admissible ONLY at `t_q <= MLA_DSA_NAMED_CLASS_T_MAX`
/// = 1, i.e. plain decode. The same box run that produced the table above ALSO recorded the
/// class moving an argmax: at `kv=131072, t_q=4` every swept chunk count (4, 8, 16, 32) moved
/// **1 of 256** latent rows, maxdiff ~1.7e-6. It was argmax-clean at t_q=1 in every measured
/// cell (0 of 64, three contexts on the box plus five on the 5090) and clean at t_q=4 at 256k
/// and 1M, but "clean in the cells we measured" is not a proof, and t_q=4..8 is the DFlash2
/// SPEC-VERIFY shape: a moved argmax there is a moved draft acceptance. So the spec-verify
/// batch never sees the named class. `mla_dsa_attn_arm_effective` enforces this in code, not by
/// table convention: a cell >= 2 at any width above the rule is demoted to 0 (the shipped
/// kernel, the always-safe path), never silently run and never quietly promoted to the
/// single-pass arm at a width where nobody measured it.
///
/// The cells therefore ship exactly what the box measured, under that rule:
///
/// * `t_q=1` -> **32**. The fastest arm at every context (54.3-57.1 us, a 10.2x on the shipped
///   552.1 us at 1M) and argmax-clean at every context. 32 beats 16 by ~1.45x here where it lost
///   to 16 on the 5090 -- 148 SMs want `64 * 32` = 2048 warps, an 82-SM laptop part does not.
///   That disagreement is the per-hardware-arm-selection law working as intended.
/// * `t_q=4` -> **1**, the BIT-IDENTICAL single-pass kernel. On the B200 it is a 3.5-7.4% WIN
///   (618.5 vs 667.6 us at 1M), the opposite sign from the 5090, where it lost by 30% and this
///   table shipped 0. Same code, different machine: on 148 SMs the shared-memory staging pays
///   for itself where on 82 it did not. Bit-identical, so this cell carries no numeric risk at
///   the spec-verify width at all. Banked evidence covers 128k/256k/1M; the two shallow contexts
///   were not in the log this cell was set from, and the kernel is depth-flat.
/// * `t_q=2,3,5..8` -> **0** (shipped). Unmeasured, and unmeasured behavior does not go on.
///
/// The door is default OFF and arm >= 2 additionally needs level 2, so nothing here reaches a
/// request without two deliberate acts. `dsa-decode-gate` FAILS with a `REGRESSION` line if a
/// cell is slower than shipped by more than `MLA_DSA_REGRESSION_MARGIN` on a later run, so the
/// next box run either confirms these cells or names the one to change. Cite the run here when
/// a cell moves.
pub const MLA_DSA_ATTN_ARM: [i32; MLA_DSA_ARM_T_MAX + 1] = [0, 32, 0, 0, 1, 0, 0, 0, 0];

/// Widest query width at which the NAMED numeric class (`dsa-warp-online-f32`, arm >= 2) may be
/// selected. 1: plain decode only. Above it the door takes a bit-identical arm or the shipped
/// kernel, so the DFlash2 spec-verify batch (t_q=4..8) never runs a rounding program that could
/// move a draft acceptance. Set by the 2026-09-03 B200 run, which observed the class move 1 of
/// 256 latent-row argmaxes at kv=131072 / t_q=4 for every swept chunk count. Raising this needs
/// its own argmax evidence at the widths it opens, not an inference from t_q=1.
pub const MLA_DSA_NAMED_CLASS_T_MAX: usize = 1;

/// Chunk counts `dsa-decode-gate` sweeps for the warp-online arm. The warp arm puts
/// `t_q * n_head * chunks` WARPS on the die, so chunks is the whole occupancy knob at t_q=1
/// (64 pairs alone is 8 CTAs of 8 warps); 64 is the kernel's ceiling (`MLA_DSA_MAX_CHUNKS`).
pub const MLA_DSA_ATTN_CHUNK_SWEEP: [i32; 4] = [4, 8, 16, 32];

/// The decode scorer engages only from this pool count up. Below it the block count
/// (`n_pools / (128 * 2)`) cannot fill the die and the shipped dispatch's own measured
/// crossover already sends small-pool decode to the reference kernel, which wins there
/// (cu/mla_attn.cu, MLA_KPOOL_SMALL_TILE_MIN_POOLS note). 4096 pools = 16 blocks = 16k context
/// at the shipped pool size 4.
pub const MLA_DSA_SCORE_MIN_POOLS: usize = 4096;

/// The gate's regression bar, shared with the sibling arm's: an arm may not be slower than the
/// kernel it replaces by more than 5%.
pub const MLA_DSA_REGRESSION_MARGIN: f64 = 1.05;

/// The gathered-attention arm code at this width (see [`MLA_DSA_ATTN_ARM`]). Pure, so the gate
/// can read the policy on any build including the 120a ones where the door is dead.
pub fn mla_dsa_attn_arm(t_q: usize) -> i32 {
    if t_q == 0 || t_q > MLA_DSA_ARM_T_MAX {
        return 0;
    }
    MLA_DSA_ATTN_ARM[t_q]
}

/// The arm the door may actually run at this width: the table cell, with the named-class width
/// rule enforced in CODE rather than by table convention. A cell >= 2 above
/// `MLA_DSA_NAMED_CLASS_T_MAX` is demoted to 0 (the shipped kernel), not to the single-pass arm
/// — a width nobody measured gets the path that cannot be wrong, not the path that happens to
/// be bit-identical. The gate reads this same function, so an edit that violates the rule shows
/// up as the gate timing a shipped cell, never as a silently-served numeric class.
pub fn mla_dsa_attn_arm_effective(t_q: usize) -> i32 {
    let arm = mla_dsa_attn_arm(t_q);
    if arm >= 2 && t_q > MLA_DSA_NAMED_CLASS_T_MAX {
        return 0;
    }
    arm
}

/// Geometry refusals from the DSA launchers: the door has nothing for this shape, so the
/// caller falls through to the shipped kernel instead of failing the request. Every other
/// non-zero rc (a real cudaError included) still goes through `ck` and surfaces.
/// Engagement counter for the k-pool SELECT door (`MEMRA_B200_DSA_SELECT`), announced once per
/// boot: the receipt a B200 A/B has to show.
pub static MLA_DSA_SELECT_DISPATCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `MEMRA_B200_DSA_SELECT=1` (default OFF, read per call: the rollback seam), compile-time gated
/// to sm_100a builds exactly like its two siblings, so a 120a/90a/89 build sees no behaviour
/// change from a var it cannot engage.
///
/// WHAT IT REPLACES. `memra_mla_kpool_select_kernel` grids `t_q` blocks, so plain decode runs it
/// on ONE CTA -- 0.68% of a 148-SM die -- sweeping `n_pools` up to ten times (8 MSB-first radix
/// passes, an optional unique-resolution scan, then the membership count and the emit). It is
/// depth-LINEAR in `n_pools = t_kv / pool` and it is what the `MEMRA_B200_DSA_DECODE` lane's
/// scorer fix stopped hiding.
///
/// THE CLASS IS EXACT, not banded, and that is a construction rather than a hope. The emitted
/// plane is a pure function of ONE 64-bit number: the `select_k`-th smallest order key
/// `(desc32(score) << 32) | pool_index`. That key is a strictly decreasing injection composed
/// with a unique index, so keys are DISTINCT and "the k-th smallest" is unambiguous; reproducing
/// it bit-for-bit reproduces the selection bit-for-bit. The parallel pipeline computes the same
/// key and runs the same `key(p) <= thr` test, so this is a launch-geometry change with an exact
/// answer. `dsa-select-gate` asserts the `idx` plane byte-identical to the shipped kernel and
/// carries a RED ARM that must fail first.
fn mla_dsa_select_on() -> bool {
    cfg!(memra_sm100_tcgen05)
        && mla_dsa_select_on_from(std::env::var("MEMRA_B200_DSA_SELECT").ok().as_deref())
}

/// The pure parse behind [`mla_dsa_select_on`] (DEFAULT ON since 2026-09-04 on the builds that
/// carry the kernel at all, which is the `memra_sm100_tcgen05` cfg above): only an explicit `0`
/// disarms. RECEIPT (darklanes research/glm5-b200-20260902/LANE.md, onemsel 2026-09-04, 2x B200
/// pair, composed defaults, 1M context, four boots per arm against the banked 46.20-50.24 band):
/// the door reads outside the band on every armed boot. It stays INERT below its own floors
/// (`mla_dsa_select_floor`: 65,536 pools = 262,144 tokens at t_q == 1, 262,144 pools = 1,048,576
/// tokens for the spec widths), so short-context serving is untouched by construction and this
/// flip changes only the long-context shape it was written for.
pub fn mla_dsa_select_on_from(v: Option<&str>) -> bool {
    !matches!(v.map(str::trim), Some("0"))
}

/// Pool-count floor for PLAIN DECODE (`t_q == 1`). The floor for the spec-verify widths is
/// separate and much higher: see [`MLA_DSA_SELECT_MIN_POOLS_SPEC`].
///
/// **THE FLOOR IN TOKENS IS 262_144 EXACTLY (65536 pools x pool 4), AND A PROMPT CALLED "256k"
/// IS USUALLY BELOW IT.** Read that before sizing any cell against this door. A 256k serving
/// rung is ~256_756 tokens, which is 64_189 pools -- 1_347 pools and 5_388 tokens short, 2.06%
/// under the floor -- so it engages NOTHING and measures noise. This cost a real B200 cell on
/// 2026-09-03. Every context figure here is an exact token count, never a "k".
///
/// MEASURED ON THE TARGET, 2026-09-03 (`dsa-select-gate`, 2x B200 SXM sm_100a, dev 0, N=5
/// interleaved, binary built from main `3908a431`; receipts
/// `darklanes:research/glm5-b200-20260902/box/selgate/gate-b200.{txt,full}`, driver
/// `box/selgate.sh`):
///
/// | pools | tokens | t_q=1 | t_q=4 |
/// |---|---|---|---|
/// | 4_096 | 16_384 | 0.20x | 0.20x |
/// | 8_192 | 32_768 | 0.25x | 0.28x |
/// | 32_768 | 131_072 | 0.67x | 0.35x |
/// | **65_536** | **262_144** | **1.31x** | **0.94x** |
/// | 262_144 | 1_048_576 | **2.81x** | **2.06x** |
///
/// WHY THIS IS KEYED ON `t_q` AND NOT ONE NUMBER, and it is a policy-CORRECTNESS fix rather
/// than a tuning one. A single floor of 65536 was chosen from RTX 5090 data, where `t_q=4` at
/// that point measured 1.07x -- a small win. On the silicon this door is actually gated to it is
/// **0.94x, a 6.5% LOSS**, and `dsa-select-gate`'s own regression bar caught it
/// (`REGRESSION kpool_select n_pools=65536 t_q=4: 311.6 us vs shipped 292.7 us (1.064x)`). A
/// uniform floor therefore ADMITTED A SHAPE THAT REGRESSES ON THE TARGET: the door is
/// sm_100a-only, and its floor was set by evidence from a card it never runs on. Keying the
/// floor removes that shape without giving up either measured win.
///
/// Both values are measured cells with NO interpolation. `t_q == 1` keeps 65536 (1.31x on the
/// pair). `t_q >= 2` takes 262144, the ONLY pool count where the spec-verify width was measured
/// to win (2.06x); everything between 65536 and 262144 at those widths is unswept, and unswept
/// shapes do not engage.
pub const MLA_DSA_SELECT_MIN_POOLS: usize = 65_536;

/// Pool-count floor for the DFlash2 spec-verify widths (`t_q >= 2`), where the parallel selector
/// needs far more pools to pay for its six launches: the shipped kernel already has `t_q` CTAs
/// of parallelism at those widths, so there is much less to win. B200-measured 0.94x at 65536
/// pools and 2.06x at 262144, so this is 262144 -- the only measured win. See
/// [`MLA_DSA_SELECT_MIN_POOLS`] for the full ladder and why the floor is keyed at all.
pub const MLA_DSA_SELECT_MIN_POOLS_SPEC: usize = 262_144;

/// Widest query width the select door keys on: decode and the spec-verify batch. Wider widths
/// already have `t_q` CTAs of parallelism and fall through to the shipped kernel untouched.
pub const MLA_DSA_SELECT_T_MAX: usize = 8;

/// Whether the serving policy engages the parallel selector at this shape. Pure, so the gate
/// reads the same predicate the wrapper does and the two cannot drift apart.
pub fn mla_dsa_select_engages(t_q: usize, n_pools: usize) -> bool {
    if !(1..=MLA_DSA_SELECT_T_MAX).contains(&t_q) {
        return false;
    }
    n_pools >= mla_dsa_select_floor(t_q)
}

/// The pool-count floor at this width. Keyed because a single floor admitted a shape that
/// REGRESSES on sm_100a (t_q=4 at 65536 pools, 0.94x on the pair); see
/// [`MLA_DSA_SELECT_MIN_POOLS`]. Pure, so the gate reads the same floor the wrapper does.
pub fn mla_dsa_select_floor(t_q: usize) -> usize {
    if t_q == 1 {
        MLA_DSA_SELECT_MIN_POOLS
    } else {
        MLA_DSA_SELECT_MIN_POOLS_SPEC
    }
}

fn mla_dsa_select_announce(t_q: usize, n_pools: usize, n_ctas: i32) {
    use std::sync::atomic::Ordering;
    if MLA_DSA_SELECT_DISPATCHES.fetch_add(1, Ordering::Relaxed) == 0 {
        eprintln!(
            "[mla-b200-dsa-select] engaged kpool_select t={t_q} pools={n_pools} ctas={n_ctas} \
             class=exact (sm_100a; MEMRA_B200_DSA_SELECT=1)"
        );
    }
}

fn mla_dsa_geometry_refusal(rc: i32) -> bool {
    matches!(rc, 40020 | 40021 | 40023)
}

fn mla_dsa_announce(kind: &str, t_q: usize, detail: &str) {
    use std::sync::atomic::Ordering;
    if MLA_DSA_DECODE_DISPATCHES.fetch_add(1, Ordering::Relaxed) == 0 {
        eprintln!(
            "[mla-b200-dsa-decode] engaged {kind} t={t_q} {detail} \
             (sm_100a; MEMRA_B200_DSA_DECODE)"
        );
    }
}

unsafe extern "C" {
    pub fn memra_mla_rope_interleaved_f32(
        x: *mut f32,
        n_pos: i32,
        n_vec: i32,
        d_rope: i32,
        positions: *const i32,
        base: f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_mla_split_latent_f32(
        kv: *const f32,
        c_kv: *mut f32,
        k_pe: *mut f32,
        t: i32,
        kv_rank: i32,
        d_rope: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_mla_append_latent_f32(
        cache: *mut f32,
        c_kv: *const f32,
        k_pe: *const f32,
        slot: i32,
        t: i32,
        kv_rank: i32,
        d_rope: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_mla_append_latent_live_f32(
        cache: *mut f32,
        c_kv: *const f32,
        k_pe: *const f32,
        pos_d: *const i32,
        t: i32,
        kv_rank: i32,
        d_rope: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_mla_absorb_q_f32(
        q_nope: *const f32,
        wk_b: *const f32,
        q_lat: *mut f32,
        t_q: i32,
        n_head: i32,
        d_nope: i32,
        kv_rank: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_mla_decompress_v_f32(
        o_lat: *const f32,
        wv_b: *const f32,
        out: *mut f32,
        t_q: i32,
        n_head: i32,
        d_v: i32,
        kv_rank: i32,
        stream: *mut c_void,
    ) -> i32;
    /// COALESCED warp-per-row twin of `memra_mla_absorb_q_f32` (`MEMRA_MLA_COALESCE`,
    /// lane/mla-coalesce). The shipped kernel gives output row `l` to THREAD `l`, which then
    /// walks its row serially, so at step `p` a warp's 32 lanes read addresses `d_nope` floats
    /// (512 B) apart and pull 32 transactions where one would do. Here a WARP owns a row and
    /// lane `k` reads `row[k], row[k+32], ...`, which is one 128-byte transaction per step,
    /// finished by a shuffle reduction. Takes the same `split` output-range partition as the
    /// decode-split twins, so the two doors COMPOSE: the split fixes the GRID, this fixes the
    /// LOADS. NAMED NUMERIC CLASS `mla_warp_row_reduce`: the per-output sum becomes 32
    /// lane-partial sums combined by a shuffle tree instead of one serial ascending dot, so
    /// this is NOT bit-identical and NOT the split twins' contract.
    #[allow(clippy::too_many_arguments)]
    /// BF16-plane twins of the `_wp` decode kernels (lane/mla-absorb-bf16-20260905): the weight
    /// plane is `u16` BF16 bits, widened per element inside the kernel; same order, same math.
    pub fn memra_mla_absorb_q_wp_bf16(
        q_nope: *const f32,
        wk_b: *const u16,
        q_lat: *mut f32,
        t_q: i32,
        n_head: i32,
        d_nope: i32,
        kv_rank: i32,
        split: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_mla_decompress_v_wp_bf16(
        o_lat: *const f32,
        wv_b: *const u16,
        out: *mut f32,
        t_q: i32,
        n_head: i32,
        d_v: i32,
        kv_rank: i32,
        split: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_mla_absorb_q_wp_f32(
        q_nope: *const f32,
        wk_b: *const f32,
        q_lat: *mut f32,
        t_q: i32,
        n_head: i32,
        d_nope: i32,
        kv_rank: i32,
        split: i32,
        stream: *mut c_void,
    ) -> i32;
    /// COALESCED warp-per-row twin of `memra_mla_decompress_v_f32` (`MEMRA_MLA_COALESCE`).
    /// Same defect and same fix as `memra_mla_absorb_q_wp_f32`, and worse in the shipped form:
    /// the lane stride there is `kv_rank` floats (2 KB), and with `d_v` typically 128 against
    /// `MLA_THREADS` 256 half the block never enters the loop. Numeric class
    /// `mla_warp_row_reduce`.
    #[allow(clippy::too_many_arguments)]
    pub fn memra_mla_decompress_v_wp_f32(
        o_lat: *const f32,
        wv_b: *const f32,
        out: *mut f32,
        t_q: i32,
        n_head: i32,
        d_v: i32,
        kv_rank: i32,
        split: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_mla_decompress_v_wp_zq8_f32(
        o_lat: *const f32,
        wv_b: *const f32,
        out: *mut f32,
        out_q: *mut i8,
        out_d: *mut f32,
        t_q: i32,
        n_head: i32,
        d_v: i32,
        kv_rank: i32,
        split: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_mla_decompress_v_wp_bf16_zq8(
        o_lat: *const f32,
        wv_b: *const u16,
        out: *mut f32,
        out_q: *mut i8,
        out_d: *mut f32,
        t_q: i32,
        n_head: i32,
        d_v: i32,
        kv_rank: i32,
        split: i32,
        stream: *mut c_void,
    ) -> i32;
    /// Decode-split twin of `memra_mla_absorb_q_f32` (MEMRA_MLA_DECODE_SPLIT): the same
    /// per-output serial dot, its output range split across `split` blocks — bit-identical
    /// by construction, gated in `tests/mla_decode_split_gpu.rs`.
    #[allow(clippy::too_many_arguments)]
    pub fn memra_mla_absorb_q_split_f32(
        q_nope: *const f32,
        wk_b: *const f32,
        q_lat: *mut f32,
        t_q: i32,
        n_head: i32,
        d_nope: i32,
        kv_rank: i32,
        split: i32,
        stream: *mut c_void,
    ) -> i32;
    /// Decode-split twin of `memra_mla_decompress_v_f32` (see above).
    #[allow(clippy::too_many_arguments)]
    pub fn memra_mla_decompress_v_split_f32(
        o_lat: *const f32,
        wv_b: *const f32,
        out: *mut f32,
        t_q: i32,
        n_head: i32,
        d_v: i32,
        kv_rank: i32,
        split: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_mla_attn_absorbed_f32(
        q_lat: *const f32,
        q_pe: *const f32,
        cache: *const f32,
        o_lat: *mut f32,
        n_head: i32,
        kv_rank: i32,
        d_rope: i32,
        t_q: i32,
        t_kv: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_mla_attn_absorbed_live_f32(
        q_lat: *const f32,
        q_pe: *const f32,
        cache: *const f32,
        o_lat: *mut f32,
        n_head: i32,
        kv_rank: i32,
        d_rope: i32,
        t_q: i32,
        pos_d: *const i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_mla_index_append_ring_f32(
        plane: *mut f32,
        a: *const f32,
        b: *const f32,
        slot: i32,
        t: i32,
        wa: i32,
        wb: i32,
        rows: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_mla_kpool_pool_keys_f32(
        state: *const f32,
        ape: *const f32,
        pool_keys: *mut f32,
        pool_begin: i32,
        n_pools: i32,
        pool: i32,
        d: i32,
        state_rows: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_mla_kpool_score_f32(
        q: *const f32,
        pool_keys: *const f32,
        hw: *const f32,
        score: *mut f32,
        t_q: i32,
        heads: i32,
        d: i32,
        n_pools: i32,
        pool: i32,
        first_pos: i32,
        qk_scale: f32,
        head_scale: f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_mla_kpool_score_ref_f32(
        q: *const f32,
        pool_keys: *const f32,
        hw: *const f32,
        score: *mut f32,
        t_q: i32,
        heads: i32,
        d: i32,
        n_pools: i32,
        pool: i32,
        first_pos: i32,
        qk_scale: f32,
        head_scale: f32,
        stream: *mut c_void,
    ) -> i32;
    /// Ints of scratch one query needs for the parallel selector, given its CTA count.
    pub fn memra_mla_kpool_select_ws_ints(n_ctas: i32) -> i64;
    /// CTA count the parallel selector launches per query. The host sizes the workspace from
    /// this same entry point, so a mismatch is impossible by construction.
    pub fn memra_mla_kpool_select_ctas(n_pools: i32) -> i32;
    /// Exact multi-CTA k-pool selection (`MEMRA_B200_DSA_SELECT`): same threshold key, same
    /// membership test, same emit order, byte-identical `idx`.
    #[allow(clippy::too_many_arguments)]
    pub fn memra_mla_kpool_select_dsa_f32(
        score: *const f32,
        idx: *mut i32,
        ws: *mut i32,
        t_q: i32,
        n_pools: i32,
        pool: i32,
        select_k: i32,
        width: i32,
        first_pos: i32,
        always_tail: i32,
        stream: *mut c_void,
    ) -> i32;
    /// RED ARM for `dsa-select-gate`, never a serving path: the exact pipeline with the resolved
    /// threshold deliberately bumped, so the gate can prove its byte comparison actually fails
    /// on a wrong selection before it is allowed to pass the real kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn memra_mla_kpool_select_dsa_redarm_f32(
        score: *const f32,
        idx: *mut i32,
        ws: *mut i32,
        t_q: i32,
        n_pools: i32,
        pool: i32,
        select_k: i32,
        width: i32,
        first_pos: i32,
        always_tail: i32,
        bump: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_mla_kpool_select_f32(
        score: *const f32,
        idx: *mut i32,
        t_q: i32,
        n_pools: i32,
        pool: i32,
        select_k: i32,
        width: i32,
        first_pos: i32,
        always_tail: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_mla_kpool_select_ref_f32(
        score: *const f32,
        idx: *mut i32,
        t_q: i32,
        n_pools: i32,
        pool: i32,
        select_k: i32,
        width: i32,
        first_pos: i32,
        always_tail: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_mla_attn_gathered_f32(
        q_lat: *const f32,
        q_pe: *const f32,
        cache: *const f32,
        idx: *const i32,
        o_lat: *mut f32,
        n_head: i32,
        kv_rank: i32,
        d_rope: i32,
        t_q: i32,
        n_slots: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    /// B200 decode-arm twin of `memra_mla_attn_gathered_f32` (MEMRA_B200_MLA_DECODE_ARM): same
    /// per-l accumulate chain, its output range [0, kv_rank) split across `split` blocks; the
    /// shared score/softmax tile walk (m, dsum) is recomputed IN FULL, unchanged, by every
    /// split block — bit-identical by construction, gated in `mla_decode_arm_gate.rs`.
    #[allow(clippy::too_many_arguments)]
    /// Single-pass bit-identical rewrite of `memra_mla_attn_gathered_f32`
    /// (`MEMRA_B200_DSA_DECODE>=1`): each tile's KV rows staged once into shared memory with
    /// float4 loads and read back for BOTH the score dot and the PV accumulate, the 8 tile
    /// exponentials hoisted into registers. Same grid, same fold, same bits. Returns 40020
    /// (width not a multiple of 4) or 40021 (staging over the smem cap) for a geometry it
    /// refuses, and the caller falls through to the shipped kernel.
    pub fn memra_mla_attn_gathered_dsa_f32(
        q_lat: *const f32,
        q_pe: *const f32,
        cache: *const f32,
        idx: *const i32,
        o_lat: *mut f32,
        n_head: i32,
        kv_rank: i32,
        d_rope: i32,
        t_q: i32,
        n_slots: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    /// Slot-per-chunk span the partial kernel walks. The host MUST size the workspace and
    /// launch from this, never from its own division, so the two cannot disagree.
    pub fn memra_mla_dsa_attn_chunk_span(n_slots: i32, chunks: i32) -> i32;
    /// Warp-online slot-split gathered attention, numeric class `dsa-warp-online-f32`
    /// (`MEMRA_B200_DSA_DECODE=2`). `part_m` / `part_d` hold `t_q * n_head * chunks` floats
    /// each; `part_acc` holds `t_q * n_head * chunks * kv_rank`. Returns 40023 for a
    /// (kv_rank, d_rope) with no template instantiation, and the caller takes the shipped path.
    pub fn memra_mla_dsa_attn_split_f32(
        q_lat: *const f32,
        q_pe: *const f32,
        cache: *const f32,
        idx: *const i32,
        o_lat: *mut f32,
        part_m: *mut f32,
        part_d: *mut f32,
        part_acc: *mut f32,
        n_head: i32,
        kv_rank: i32,
        d_rope: i32,
        t_q: i32,
        n_slots: i32,
        chunks: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    /// Head-blocked decode pool scorer (`MEMRA_B200_DSA_DECODE>=1`), bit-identical to
    /// `memra_mla_kpool_score_ref_f32`. Returns 40023 when this (heads, d) has no
    /// instantiation, and the caller falls through to the shipped dispatch.
    pub fn memra_mla_kpool_score_dsa_f32(
        q: *const f32,
        pool_keys: *const f32,
        hw: *const f32,
        score: *mut f32,
        t_q: i32,
        heads: i32,
        d: i32,
        n_pools: i32,
        pool: i32,
        first_pos: i32,
        qk_scale: f32,
        head_scale: f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_mla_attn_gathered_split_f32(
        q_lat: *const f32,
        q_pe: *const f32,
        cache: *const f32,
        idx: *const i32,
        o_lat: *mut f32,
        n_head: i32,
        kv_rank: i32,
        d_rope: i32,
        t_q: i32,
        n_slots: i32,
        scale: f32,
        split: i32,
        stream: *mut c_void,
    ) -> i32;
    /// Strided-batched BF16 tensor-core GEMM (cu/f16_prefill.cu): per batch b,
    /// `y_b[m, n] = x_b[m, k] @ w_b[n, k]^T`, f32 accumulate, y f32 or bf16 by flag.
    /// The MEMRA_MLA_TC_PREFILL absorb/decompress engine (one launch replaces the
    /// per-position absorb_q / decompress_v kernels at prefill widths).
    fn memra_bf16_gemm_sb(
        w_bf16: *const c_void,
        x_bf16: *const c_void,
        y: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        x_rs: i64,
        x_bs: i64,
        y_rs: i64,
        y_bs: i64,
        batch: i32,
        y_is_bf16: i32,
        ws: *mut c_void,
        ws_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Turn a launcher's status band into a named error. Every MLA launch goes through this —
/// a silently-ignored non-zero status is how a contract violation becomes garbage activations.
fn ck(what: &str, rc: i32) -> Res<()> {
    if rc == 0 {
        return Ok(());
    }
    let detail = match rc {
        40001 => " (d_rope must be even — interleaved rope rotates (2j, 2j+1) pairs)",
        40002 => " (kv_rank exceeds the kernel's MLA_MAX_RANK shared-memory ceiling)",
        40003 => " (d_rope exceeds the kernel's MLA_MAX_ROPE ceiling)",
        40004 => " (t_q > t_kv — queries must be a suffix of the latent cache)",
        40010 => " (k-pool size out of range — 1..=MLA_MAX_POOL)",
        40011 => " (indexer head count out of range — 1..=1024, one thread per head)",
        40012 => " (t_q * n_pools exceeds the grid.x contract)",
        40017 => " (indexer head dim must be positive)",
        40013 => {
            " (always_select_tail=false: queries before the first complete pool would have an \
             empty candidate set, which the memra-reference oracle refuses outright)"
        }
        40014 => " (index-list width is narrower than select_k * pool + pool - 1)",
        40015 => " (empty gathered candidate list — a zero softmax denominator)",
        40020 => " (latent row width is not a multiple of 4 — the DSA float4 staging needs it)",
        40021 => " (DSA tile staging exceeds MLA_DSA_KV_SMEM_MAX)",
        40022 => " (DSA slot-chunk count out of range — 1..=64)",
        40023 => " (no DSA scorer instantiation for this (heads, d))",
        r if (10000..20000).contains(&r) => " (cudaError)",
        _ => "",
    };
    Err(format!("mla kernel `{what}` failed: rc {rc}{detail}").into())
}

impl Engine {
    /// Interleaved ("NORM") RoPE in place over `x` laid out [n_pos][n_vec][d_rope].
    /// `d_rope == 0` (NoPE, glm5_next) is a no-op — the caller must still not pass an empty
    /// slice through a path that dereferences it, which is why the rope plane is skipped
    /// entirely in the forward arm rather than launched with a zero extent.
    pub fn mla_rope_interleaved(
        &self,
        x: &mut CudaSlice<f32>,
        pos_d: &CudaSlice<i32>,
        n_pos: usize,
        n_vec: usize,
        d_rope: usize,
        base: f32,
    ) -> Res<()> {
        if d_rope == 0 {
            return Ok(());
        }
        let s = self.stream();
        unsafe {
            ck(
                "rope_interleaved",
                memra_mla_rope_interleaved_f32(
                    x.device_ptr_mut(&s).0 as *mut f32,
                    n_pos as i32,
                    n_vec as i32,
                    d_rope as i32,
                    pos_d.device_ptr(&s).0 as *const i32,
                    base,
                    s.cu_stream() as *mut c_void,
                ),
            )
        }
    }

    /// Split the `wkv_a` output rows [t][kv_rank + d_rope] into `c_kv` and `k_pe` planes.
    pub fn mla_split_latent(
        &self,
        kv: &CudaSlice<f32>,
        c_kv: &mut CudaSlice<f32>,
        k_pe: &mut CudaSlice<f32>,
        t: usize,
        kv_rank: usize,
        d_rope: usize,
    ) -> Res<()> {
        let s = self.stream();
        unsafe {
            ck(
                "split_latent",
                memra_mla_split_latent_f32(
                    kv.device_ptr(&s).0 as *const f32,
                    c_kv.device_ptr_mut(&s).0 as *mut f32,
                    k_pe.device_ptr_mut(&s).0 as *mut f32,
                    t as i32,
                    kv_rank as i32,
                    d_rope as i32,
                    s.cu_stream() as *mut c_void,
                ),
            )
        }
    }

    /// Append `t` latent rows `[c_kv | k_pe]` to the cache plane starting at row `slot`.
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    pub fn mla_append_latent(
        &self,
        cache: &mut CudaSlice<f32>,
        c_kv: &CudaSlice<f32>,
        k_pe: &CudaSlice<f32>,
        slot: usize,
        t: usize,
        kv_rank: usize,
        d_rope: usize,
    ) -> Res<()> {
        let s = self.stream();
        unsafe {
            ck(
                "append_latent",
                memra_mla_append_latent_f32(
                    cache.device_ptr_mut(&s).0 as *mut f32,
                    c_kv.device_ptr(&s).0 as *const f32,
                    k_pe.device_ptr(&s).0 as *const f32,
                    slot as i32,
                    t as i32,
                    kv_rank as i32,
                    d_rope as i32,
                    s.cu_stream() as *mut c_void,
                ),
            )
        }
    }

    /// Absorb: `q_lat[i][h][:] = w_uk[h]ᵀ · q_nope[i][h][:]` (rank space).
    #[allow(clippy::too_many_arguments)]
    // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    /// BENCH-ONLY raw arm dispatch for `mla-coalesce-bench` (structural ncu target): `arm` 0 =
    /// shipped thread-per-row, 1 = decode-split twin at `split`, 2 = warp-per-row coalesced twin
    /// at `split` (1 = unsplit). Bypasses every door on purpose so one process can be profiled
    /// across all three kernels. Not a serving entry point.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_absorb_q_raw_arm(
        &self,
        q_nope: &CudaSlice<f32>,
        wk_b: &CudaSlice<f32>,
        q_lat: &mut CudaSlice<f32>,
        t_q: usize,
        n_head: usize,
        d_nope: usize,
        kv_rank: usize,
        arm: u8,
        split: i32,
    ) -> Res<()> {
        let s = self.stream();
        unsafe {
            let (q, w, o) = (
                q_nope.device_ptr(&s).0 as *const f32,
                wk_b.device_ptr(&s).0 as *const f32,
                q_lat.device_ptr_mut(&s).0 as *mut f32,
            );
            let (t, h, dn, kr) = (t_q as i32, n_head as i32, d_nope as i32, kv_rank as i32);
            let st = s.cu_stream() as *mut c_void;
            match arm {
                0 => ck(
                    "absorb_q_raw",
                    memra_mla_absorb_q_f32(q, w, o, t, h, dn, kr, st),
                ),
                1 => ck(
                    "absorb_q_split_raw",
                    memra_mla_absorb_q_split_f32(q, w, o, t, h, dn, kr, split, st),
                ),
                _ => ck(
                    "absorb_q_wp_raw",
                    memra_mla_absorb_q_wp_f32(q, w, o, t, h, dn, kr, split, st),
                ),
            }
        }
    }

    /// BENCH-ONLY raw arm dispatch, decompress twin of `mla_absorb_q_raw_arm`.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_decompress_v_raw_arm(
        &self,
        o_lat: &CudaSlice<f32>,
        wv_b: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        t_q: usize,
        n_head: usize,
        d_v: usize,
        kv_rank: usize,
        arm: u8,
        split: i32,
    ) -> Res<()> {
        let s = self.stream();
        unsafe {
            let (a, w, o) = (
                o_lat.device_ptr(&s).0 as *const f32,
                wv_b.device_ptr(&s).0 as *const f32,
                out.device_ptr_mut(&s).0 as *mut f32,
            );
            let (t, h, dv, kr) = (t_q as i32, n_head as i32, d_v as i32, kv_rank as i32);
            let st = s.cu_stream() as *mut c_void;
            match arm {
                0 => ck(
                    "decompress_v_raw",
                    memra_mla_decompress_v_f32(a, w, o, t, h, dv, kr, st),
                ),
                1 => ck(
                    "decompress_v_split_raw",
                    memra_mla_decompress_v_split_f32(a, w, o, t, h, dv, kr, split, st),
                ),
                _ => ck(
                    "decompress_v_wp_raw",
                    memra_mla_decompress_v_wp_f32(a, w, o, t, h, dv, kr, split, st),
                ),
            }
        }
    }

    /// BENCH-ONLY raw dispatch of the fused hc pre-chain for `mla-coalesce-bench` (structural
    /// ncu target): `arm` 0 = `memra_dsv4_hc_pre_fused_v2` (block 128, the shipped `=2` arm),
    /// 1 = `_v3` at `block` with the shared-memory Sinkhorn, 2 = `_v3` at `block` with the
    /// register Sinkhorn (`MEMRA_HC_PRE_SINK_REG`). Bypasses the door readers on purpose. The
    /// `hc-fused-gate` calls the v1/v2 launchers directly and predates v3, so this is the only
    /// way to put the register Sinkhorn under a profiler.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_pre_raw_arm(
        &self,
        x: &CudaSlice<f32>,
        mixes: &CudaSlice<f32>,
        scale: &CudaSlice<f32>,
        base: &CudaSlice<f32>,
        pre: &mut CudaSlice<f32>,
        post: &mut CudaSlice<f32>,
        comb: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        s_rows: usize,
        hc: usize,
        d: usize,
        iters: usize,
        eps: f32,
        arm: u8,
        block: i32,
        niters: Option<&mut CudaSlice<i32>>,
    ) -> Res<()> {
        let st = self.stream();
        unsafe {
            let np: *mut i32 = match niters {
                Some(n) => n.device_ptr_mut(&st).0 as *mut i32,
                None => std::ptr::null_mut(),
            };
            let (xp, mp, sp, bp) = (
                x.device_ptr(&st).0 as *const f32,
                mixes.device_ptr(&st).0 as *const f32,
                scale.device_ptr(&st).0 as *const f32,
                base.device_ptr(&st).0 as *const f32,
            );
            let (pp, qp, cp, yp) = (
                pre.device_ptr_mut(&st).0 as *mut f32,
                post.device_ptr_mut(&st).0 as *mut f32,
                comb.device_ptr_mut(&st).0 as *mut f32,
                y.device_ptr_mut(&st).0 as *mut f32,
            );
            let (sr, h, dd, it) = (s_rows as i32, hc as i32, d as i32, iters as i32);
            let cs = st.cu_stream() as *mut c_void;
            let rc = match arm {
                0 => crate::dsv4_ffi::memra_dsv4_hc_pre_fused_v2(
                    xp, mp, sp, bp, pp, qp, cp, yp, sr, h, dd, it, eps, np, cs,
                ),
                1 => crate::dsv4_ffi::memra_dsv4_hc_pre_fused_v3(
                    xp, mp, sp, bp, pp, qp, cp, yp, sr, h, dd, it, eps, np, block, 0, cs,
                ),
                _ => crate::dsv4_ffi::memra_dsv4_hc_pre_fused_v3(
                    xp, mp, sp, bp, pp, qp, cp, yp, sr, h, dd, it, eps, np, block, 1, cs,
                ),
            };
            ck("hc_pre_raw", rc)
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// The BF16-plane arm of [`Engine::mla_absorb_q`] for the served `_wp` partition (door
    /// `MEMRA_MLA_ABSORB_BF16`): the same split the coalesce door chooses, on the BF16 copy of
    /// `wk_b`. `Ok(false)` when that partition is not the one in force (the caller runs the f32
    /// dispatch unchanged).
    #[allow(clippy::too_many_arguments)]
    pub fn mla_absorb_q_bf16(
        &self,
        q_nope: &CudaSlice<f32>,
        wk_b16: &CudaSlice<u16>,
        q_lat: &mut CudaSlice<f32>,
        t_q: usize,
        n_head: usize,
        d_nope: usize,
        kv_rank: usize,
    ) -> Res<bool> {
        if !mla_coalesce_on() {
            return Ok(false);
        }
        let split = mla_b200_split_for(MlaB200Kernel::AbsorbQ, t_q, kv_rank)
            .or_else(|| mla_decode_split_for(t_q * n_head, kv_rank))
            .unwrap_or(1);
        if split <= 1 {
            return Ok(false);
        }
        let s = self.stream();
        unsafe {
            ck(
                "absorb_q_wp_bf16",
                memra_mla_absorb_q_wp_bf16(
                    q_nope.device_ptr(&s).0 as *const f32,
                    wk_b16.device_ptr(&s).0 as *const u16,
                    q_lat.device_ptr_mut(&s).0 as *mut f32,
                    t_q as i32,
                    n_head as i32,
                    d_nope as i32,
                    kv_rank as i32,
                    split,
                    s.cu_stream() as *mut c_void,
                ),
            )?;
        }
        Ok(true)
    }

    /// The BF16-plane arm of [`Engine::mla_decompress_v`]; see [`Engine::mla_absorb_q_bf16`].
    #[allow(clippy::too_many_arguments)]
    pub fn mla_decompress_v_bf16(
        &self,
        o_lat: &CudaSlice<f32>,
        wv_b16: &CudaSlice<u16>,
        out: &mut CudaSlice<f32>,
        t_q: usize,
        n_head: usize,
        d_v: usize,
        kv_rank: usize,
    ) -> Res<bool> {
        if !mla_coalesce_on() {
            return Ok(false);
        }
        let split = mla_b200_split_for(MlaB200Kernel::DecompressV, t_q, d_v)
            .or_else(|| mla_decode_split_for(t_q * n_head, d_v))
            .unwrap_or(1);
        if split <= 1 {
            return Ok(false);
        }
        let s = self.stream();
        unsafe {
            ck(
                "decompress_v_wp_bf16",
                memra_mla_decompress_v_wp_bf16(
                    o_lat.device_ptr(&s).0 as *const f32,
                    wv_b16.device_ptr(&s).0 as *const u16,
                    out.device_ptr_mut(&s).0 as *mut f32,
                    t_q as i32,
                    n_head as i32,
                    d_v as i32,
                    kv_rank as i32,
                    split,
                    s.cu_stream() as *mut c_void,
                ),
            )?;
        }
        Ok(true)
    }

    /// The coalesce-arm split the `_wp` decompress launch takes at this shape, when the fused
    /// q8_1 epilogue can ride it: `MEMRA_MLA_COALESCE=1`, split > 1, and `d_v / split` a whole
    /// number of q8 blocks (<= 256 wide). `None` means the plain kernels run unchanged.
    fn mla_decompress_v_zq8_split(&self, t_q: usize, n_head: usize, d_v: usize) -> Option<i32> {
        if !mla_coalesce_on() {
            return None;
        }
        let split = mla_b200_split_for(MlaB200Kernel::DecompressV, t_q, d_v)
            .or_else(|| mla_decode_split_for(t_q * n_head, d_v))
            .unwrap_or(1);
        if split <= 1 || !d_v.is_multiple_of(split as usize) {
            return None;
        }
        let per = d_v / split as usize;
        (per.is_multiple_of(32) && per <= 256).then_some(split)
    }

    /// [`Engine::mla_decompress_v`]'s coalesce arm emitting `wo`'s q8_1 pair beside `out`
    /// (`memra_mla_decompress_v_wp_zq8_kernel`, MEMRA_MLA_WO_ZQ8): `out` bit-identical to the plain
    /// launch, the pair bit-identical to `quantize_q8_1(out, t_q, n_head * d_v)`. `Ok(None)` when
    /// the shape or the arm does not fit (the caller then runs the plain sequence).
    #[allow(clippy::too_many_arguments)]
    pub fn mla_decompress_v_zq8(
        &self,
        o_lat: &CudaSlice<f32>,
        wv_b: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        t_q: usize,
        n_head: usize,
        d_v: usize,
        kv_rank: usize,
    ) -> Res<Option<(CudaSlice<i8>, CudaSlice<f32>)>> {
        let Some(split) = self.mla_decompress_v_zq8_split(t_q, n_head, d_v) else {
            return Ok(None);
        };
        let n = t_q * n_head * d_v;
        let mut q = self.alloc_i8_uninit(n)?;
        let mut d = self.uninit(n / 32)?;
        let s = self.stream();
        mla_coalesce_announce("decompress_v_zq8", t_q, n_head, split);
        unsafe {
            ck(
                "decompress_v_wp_zq8",
                memra_mla_decompress_v_wp_zq8_f32(
                    o_lat.device_ptr(&s).0 as *const f32,
                    wv_b.device_ptr(&s).0 as *const f32,
                    out.device_ptr_mut(&s).0 as *mut f32,
                    q.device_ptr_mut(&s).0 as *mut i8,
                    d.device_ptr_mut(&s).0 as *mut f32,
                    t_q as i32,
                    n_head as i32,
                    d_v as i32,
                    kv_rank as i32,
                    split,
                    s.cu_stream() as *mut c_void,
                ),
            )?;
        }
        Ok(Some((q, d)))
    }

    /// BF16-plane twin of [`Engine::mla_decompress_v_zq8`] (the `MEMRA_MLA_ABSORB_BF16` arm).
    #[allow(clippy::too_many_arguments)]
    pub fn mla_decompress_v_bf16_zq8(
        &self,
        o_lat: &CudaSlice<f32>,
        wv_b16: &CudaSlice<u16>,
        out: &mut CudaSlice<f32>,
        t_q: usize,
        n_head: usize,
        d_v: usize,
        kv_rank: usize,
    ) -> Res<Option<(CudaSlice<i8>, CudaSlice<f32>)>> {
        let Some(split) = self.mla_decompress_v_zq8_split(t_q, n_head, d_v) else {
            return Ok(None);
        };
        let n = t_q * n_head * d_v;
        let mut q = self.alloc_i8_uninit(n)?;
        let mut d = self.uninit(n / 32)?;
        let s = self.stream();
        unsafe {
            ck(
                "decompress_v_wp_bf16_zq8",
                memra_mla_decompress_v_wp_bf16_zq8(
                    o_lat.device_ptr(&s).0 as *const f32,
                    wv_b16.device_ptr(&s).0 as *const u16,
                    out.device_ptr_mut(&s).0 as *mut f32,
                    q.device_ptr_mut(&s).0 as *mut i8,
                    d.device_ptr_mut(&s).0 as *mut f32,
                    t_q as i32,
                    n_head as i32,
                    d_v as i32,
                    kv_rank as i32,
                    split,
                    s.cu_stream() as *mut c_void,
                ),
            )?;
        }
        Ok(Some((q, d)))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mla_absorb_q(
        &self,
        q_nope: &CudaSlice<f32>,
        wk_b: &CudaSlice<f32>,
        q_lat: &mut CudaSlice<f32>,
        t_q: usize,
        n_head: usize,
        d_nope: usize,
        kv_rank: usize,
    ) -> Res<()> {
        let s = self.stream();
        // MEMRA_MLA_COALESCE door, checked FIRST because it changes how a row is READ, which
        // is orthogonal to every door below (they choose the output-range partition). It reuses
        // their policy rather than replacing it: ask the B200 arm, then the generic split, and
        // pass whatever split they choose (1 = unsplit). Consulting `mla_decode_split_for` also
        // ticks that door's own dispatch counter, which is correct — it WAS consulted and its
        // partition IS the one running.
        // MEASURED 2026-09-03, 2x B200: at split 1 (64 blocks) warp-per-row REGRESSES -11%
        // (51.01 vs 57.34): one row in flight per warp with a serial shuffle reduction after
        // each replaces 16,384 threads x 1 row with 512 warps x 1 row, a 32x loss of memory
        // parallelism that outweighs the coalescing. At split 16 (1,024 blocks) it is +1.9% on
        // top of the split. So the door only engages when a split door gave it a grid to spend
        // the coalescing on; at split 1 it falls through to the dispatch below, unchanged.
        let coalesce_split = if mla_coalesce_on() {
            mla_b200_split_for(MlaB200Kernel::AbsorbQ, t_q, kv_rank)
                .or_else(|| mla_decode_split_for(t_q * n_head, kv_rank))
                .unwrap_or(1)
        } else {
            1
        };
        if coalesce_split > 1 {
            let split = coalesce_split;
            mla_coalesce_announce("absorb_q", t_q, n_head, split);
            return unsafe {
                ck(
                    "absorb_q_wp",
                    memra_mla_absorb_q_wp_f32(
                        q_nope.device_ptr(&s).0 as *const f32,
                        wk_b.device_ptr(&s).0 as *const f32,
                        q_lat.device_ptr_mut(&s).0 as *mut f32,
                        t_q as i32,
                        n_head as i32,
                        d_nope as i32,
                        kv_rank as i32,
                        split,
                        s.cu_stream() as *mut c_void,
                    ),
                )
            };
        }
        // MEMRA_B200_MLA_DECODE_ARM door (checked first; split from the t_q-keyed table
        // MLA_B200_ABSORB_Q_SPLIT, a 1 cell falls through to the doors below; the split twin is
        // the same kernel the generic door launches, so this is only a policy pick).
        if let Some(split) = mla_b200_split_for(MlaB200Kernel::AbsorbQ, t_q, kv_rank) {
            mla_b200_split_announce("absorb_q", t_q, n_head, split);
            return unsafe {
                ck(
                    "absorb_q_split_b200",
                    memra_mla_absorb_q_split_f32(
                        q_nope.device_ptr(&s).0 as *const f32,
                        wk_b.device_ptr(&s).0 as *const f32,
                        q_lat.device_ptr_mut(&s).0 as *mut f32,
                        t_q as i32,
                        n_head as i32,
                        d_nope as i32,
                        kv_rank as i32,
                        split,
                        s.cu_stream() as *mut c_void,
                    ),
                )
            };
        }
        // MEMRA_MLA_DECODE_SPLIT door: same bytes at any split (see mla_decode_split_for).
        if let Some(split) = mla_decode_split_for(t_q * n_head, kv_rank) {
            mla_split_announce("absorb_q", t_q, n_head, split);
            return unsafe {
                ck(
                    "absorb_q_split",
                    memra_mla_absorb_q_split_f32(
                        q_nope.device_ptr(&s).0 as *const f32,
                        wk_b.device_ptr(&s).0 as *const f32,
                        q_lat.device_ptr_mut(&s).0 as *mut f32,
                        t_q as i32,
                        n_head as i32,
                        d_nope as i32,
                        kv_rank as i32,
                        split,
                        s.cu_stream() as *mut c_void,
                    ),
                )
            };
        }
        unsafe {
            ck(
                "absorb_q",
                memra_mla_absorb_q_f32(
                    q_nope.device_ptr(&s).0 as *const f32,
                    wk_b.device_ptr(&s).0 as *const f32,
                    q_lat.device_ptr_mut(&s).0 as *mut f32,
                    t_q as i32,
                    n_head as i32,
                    d_nope as i32,
                    kv_rank as i32,
                    s.cu_stream() as *mut c_void,
                ),
            )
        }
    }

    /// Decompress: `out[i][h][:] = w_uv[h] · o_lat[i][h][:]`.
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    pub fn mla_decompress_v(
        &self,
        o_lat: &CudaSlice<f32>,
        wv_b: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        t_q: usize,
        n_head: usize,
        d_v: usize,
        kv_rank: usize,
    ) -> Res<()> {
        let s = self.stream();
        // MEMRA_MLA_COALESCE door, checked FIRST because it changes how a row is READ, which
        // is orthogonal to every door below (they choose the output-range partition). It reuses
        // their policy rather than replacing it: ask the B200 arm, then the generic split, and
        // pass whatever split they choose (1 = unsplit). Consulting `mla_decode_split_for` also
        // ticks that door's own dispatch counter, which is correct — it WAS consulted and its
        // partition IS the one running.
        // MEASURED 2026-09-03, 2x B200: at split 1 (64 blocks) warp-per-row REGRESSES -11%
        // (51.01 vs 57.34): one row in flight per warp with a serial shuffle reduction after
        // each replaces 16,384 threads x 1 row with 512 warps x 1 row, a 32x loss of memory
        // parallelism that outweighs the coalescing. At split 16 (1,024 blocks) it is +1.9% on
        // top of the split. So the door only engages when a split door gave it a grid to spend
        // the coalescing on; at split 1 it falls through to the dispatch below, unchanged.
        let coalesce_split = if mla_coalesce_on() {
            mla_b200_split_for(MlaB200Kernel::DecompressV, t_q, d_v)
                .or_else(|| mla_decode_split_for(t_q * n_head, d_v))
                .unwrap_or(1)
        } else {
            1
        };
        if coalesce_split > 1 {
            let split = coalesce_split;
            mla_coalesce_announce("decompress_v", t_q, n_head, split);
            return unsafe {
                ck(
                    "decompress_v_wp",
                    memra_mla_decompress_v_wp_f32(
                        o_lat.device_ptr(&s).0 as *const f32,
                        wv_b.device_ptr(&s).0 as *const f32,
                        out.device_ptr_mut(&s).0 as *mut f32,
                        t_q as i32,
                        n_head as i32,
                        d_v as i32,
                        kv_rank as i32,
                        split,
                        s.cu_stream() as *mut c_void,
                    ),
                )
            };
        }
        // MEMRA_B200_MLA_DECODE_ARM door (checked first, table MLA_B200_DECOMPRESS_V_SPLIT; see
        // mla_absorb_q above).
        if let Some(split) = mla_b200_split_for(MlaB200Kernel::DecompressV, t_q, d_v) {
            mla_b200_split_announce("decompress_v", t_q, n_head, split);
            return unsafe {
                ck(
                    "decompress_v_split_b200",
                    memra_mla_decompress_v_split_f32(
                        o_lat.device_ptr(&s).0 as *const f32,
                        wv_b.device_ptr(&s).0 as *const f32,
                        out.device_ptr_mut(&s).0 as *mut f32,
                        t_q as i32,
                        n_head as i32,
                        d_v as i32,
                        kv_rank as i32,
                        split,
                        s.cu_stream() as *mut c_void,
                    ),
                )
            };
        }
        // MEMRA_MLA_DECODE_SPLIT door: same bytes at any split (see mla_decode_split_for).
        if let Some(split) = mla_decode_split_for(t_q * n_head, d_v) {
            mla_split_announce("decompress_v", t_q, n_head, split);
            return unsafe {
                ck(
                    "decompress_v_split",
                    memra_mla_decompress_v_split_f32(
                        o_lat.device_ptr(&s).0 as *const f32,
                        wv_b.device_ptr(&s).0 as *const f32,
                        out.device_ptr_mut(&s).0 as *mut f32,
                        t_q as i32,
                        n_head as i32,
                        d_v as i32,
                        kv_rank as i32,
                        split,
                        s.cu_stream() as *mut c_void,
                    ),
                )
            };
        }
        unsafe {
            ck(
                "decompress_v",
                memra_mla_decompress_v_f32(
                    o_lat.device_ptr(&s).0 as *const f32,
                    wv_b.device_ptr(&s).0 as *const f32,
                    out.device_ptr_mut(&s).0 as *mut f32,
                    t_q as i32,
                    n_head as i32,
                    d_v as i32,
                    kv_rank as i32,
                    s.cu_stream() as *mut c_void,
                ),
            )
        }
    }

    /// Absorbed-form MQA attention over the latent cache. `q_pe` is ignored when
    /// `d_rope == 0`; callers on the NoPE path may pass any allocated slice.
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    pub fn mla_attn_absorbed(
        &self,
        q_lat: &CudaSlice<f32>,
        q_pe: &CudaSlice<f32>,
        cache: &CudaSlice<f32>,
        o_lat: &mut CudaSlice<f32>,
        n_head: usize,
        kv_rank: usize,
        d_rope: usize,
        t_q: usize,
        t_kv: usize,
        scale: f32,
    ) -> Res<()> {
        let s = self.stream();
        unsafe {
            ck(
                "attn_absorbed",
                memra_mla_attn_absorbed_f32(
                    q_lat.device_ptr(&s).0 as *const f32,
                    q_pe.device_ptr(&s).0 as *const f32,
                    cache.device_ptr(&s).0 as *const f32,
                    o_lat.device_ptr_mut(&s).0 as *mut f32,
                    n_head as i32,
                    kv_rank as i32,
                    d_rope as i32,
                    t_q as i32,
                    t_kv as i32,
                    scale,
                    s.cu_stream() as *mut c_void,
                ),
            )
        }
    }

    /// Live-length twin of [`Engine::mla_attn_absorbed`]: `t_kv = pos_d[0] + t_q` read on the
    /// device (the decode-graph door's position word), fixed launch geometry, bit-identical to the
    /// scalar launch at the same length (`tests/mla_live_len_gpu.rs`). The caller owns the
    /// `t_q <= t_kv` invariant the scalar launcher checks on the host.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_attn_absorbed_live(
        &self,
        q_lat: &CudaSlice<f32>,
        q_pe: &CudaSlice<f32>,
        cache: &CudaSlice<f32>,
        o_lat: &mut CudaSlice<f32>,
        n_head: usize,
        kv_rank: usize,
        d_rope: usize,
        t_q: usize,
        pos_d: &CudaSlice<i32>,
        scale: f32,
    ) -> Res<()> {
        let s = self.stream();
        unsafe {
            ck(
                "attn_absorbed_live",
                memra_mla_attn_absorbed_live_f32(
                    q_lat.device_ptr(&s).0 as *const f32,
                    q_pe.device_ptr(&s).0 as *const f32,
                    cache.device_ptr(&s).0 as *const f32,
                    o_lat.device_ptr_mut(&s).0 as *mut f32,
                    n_head as i32,
                    kv_rank as i32,
                    d_rope as i32,
                    t_q as i32,
                    pos_d.device_ptr(&s).0 as *const i32,
                    scale,
                    s.cu_stream() as *mut c_void,
                ),
            )
        }
    }

    /// Live-slot twin of [`Engine::mla_append_latent`]: the row offset is `pos_d[0]` on the
    /// device. Bit-identical to the scalar launch at `slot == pos_d[0]`.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_append_latent_live(
        &self,
        cache: &mut CudaSlice<f32>,
        c_kv: &CudaSlice<f32>,
        k_pe: &CudaSlice<f32>,
        pos_d: &CudaSlice<i32>,
        t: usize,
        kv_rank: usize,
        d_rope: usize,
    ) -> Res<()> {
        let s = self.stream();
        unsafe {
            ck(
                "append_latent_live",
                memra_mla_append_latent_live_f32(
                    cache.device_ptr_mut(&s).0 as *mut f32,
                    c_kv.device_ptr(&s).0 as *const f32,
                    k_pe.device_ptr(&s).0 as *const f32,
                    pos_d.device_ptr(&s).0 as *const i32,
                    t as i32,
                    kv_rank as i32,
                    d_rope as i32,
                    s.cu_stream() as *mut c_void,
                ),
            )
        }
    }
}

/// Safe wrappers for the DSA k-pool indexer (`cu/mla_attn.cu`, "DSA k-pool indexer" section).
/// Numeric truth is `memra_reference::kpool_allowed_tokens`; the gate is
/// `tests/glm5_kpool_indexer_gpu.rs`.
impl Engine {
    /// Collapse pools `[pool_begin, n_pools)` of `pool` cached indexer rows each into one key by a
    /// learned per-channel softmax over (gate score + positional embedding).
    /// `state` rows are `[k | gate]`, `2 * d` wide; `ape` is `[pool][d]` row-major.
    ///
    /// `pool_begin` is the RESIDENCY seam: a pool's key depends only on its own `pool` state rows
    /// (append-only, never rewritten) and the constant `ape`, so it is final the instant the
    /// pool's last row lands. Pools below `pool_begin` are already resident and are left alone —
    /// bit-identically to what rebuilding them would produce. Pass 0 for a full rebuild.
    ///
    /// `state_rows` is the indexer plane's TAIL-RING size in rows (0 = flat, absolute
    /// addressing). It is always a multiple of `pool`, so a pool's members stay contiguous
    /// across the wrap and the collapse reads the same values in the same order either way.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_kpool_pool_keys(
        &self,
        state: &CudaSlice<f32>,
        ape: &CudaSlice<f32>,
        pool_keys: &mut CudaSlice<f32>,
        pool_begin: usize,
        n_pools: usize,
        pool: usize,
        d: usize,
        state_rows: usize,
    ) -> Res<()> {
        let s = self.stream();
        unsafe {
            ck(
                "kpool_pool_keys",
                memra_mla_kpool_pool_keys_f32(
                    state.device_ptr(&s).0 as *const f32,
                    ape.device_ptr(&s).0 as *const f32,
                    pool_keys.device_ptr_mut(&s).0 as *mut f32,
                    pool_begin as i32,
                    n_pools as i32,
                    pool as i32,
                    d as i32,
                    state_rows as i32,
                    s.cu_stream() as *mut c_void,
                ),
            )
        }
    }

    /// Append `t` packed indexer rows `[k_norm | gate]` at absolute row `slot`, wrapping mod
    /// `rows` when the plane is a TAIL RING (`rows == 0` is the flat plane).
    ///
    /// SEPARATE from [`Engine::mla_append_latent`] on purpose: the latent plane is re-read by
    /// every later query through the gathered attention walk and is NOT a ring, so the two planes
    /// must not share a row-addressing contract even though they share a row shape.
    #[allow(clippy::too_many_arguments)]
    ///
    /// `src_row` is the first SOURCE row of `a`/`b` to append: the call's `k_norm`/`gate` are
    /// computed once for the whole call, and the tail-ring drain (`mla_kpool_indices`) walks them
    /// in sub-ranges. `src_row` 0 is the whole-call append.
    pub fn mla_index_append(
        &self,
        plane: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        src_row: usize,
        slot: usize,
        t: usize,
        wa: usize,
        wb: usize,
        rows: usize,
    ) -> Res<()> {
        let s = self.stream();
        unsafe {
            ck(
                "index_append_ring",
                memra_mla_index_append_ring_f32(
                    plane.device_ptr_mut(&s).0 as *mut f32,
                    (a.device_ptr(&s).0 as *const f32).add(src_row * wa),
                    (b.device_ptr(&s).0 as *const f32).add(src_row * wb),
                    slot as i32,
                    t as i32,
                    wa as i32,
                    wb as i32,
                    rows as i32,
                    s.cu_stream() as *mut c_void,
                ),
            )
        }
    }

    /// Head-mixed pool scores, `-inf` on pools whose last token is not visible to the query.
    /// `first_pos` is the absolute cache row of query 0 (queries are the cache's last `t_q` rows).
    ///
    /// Register-tiled fused GEMM+head-reduce: the pool-key tile stays resident in shared memory
    /// across the head loop, so `pool_keys` is read once per query TILE instead of once per
    /// query, and the head mix lands in the accumulator instead of costing a second pass over a
    /// `[t_q * heads, n_pools]` plane (17 GB at the shipped 1M/512 shape). BIT-IDENTICAL to
    /// [`Engine::mla_kpool_score_ref`] by construction — same six-step rounding sequence, spelled
    /// with explicit intrinsics — and gated so
    /// (`gpu_kpool_scoring_is_byte_identical_to_the_reference_kernel`). See the scoring section
    /// of `cu/mla_attn.cu` for why that identity is the requirement and not a nicety.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_kpool_score(
        &self,
        q: &CudaSlice<f32>,
        pool_keys: &CudaSlice<f32>,
        head_weights: &CudaSlice<f32>,
        score: &mut CudaSlice<f32>,
        t_q: usize,
        heads: usize,
        d: usize,
        n_pools: usize,
        pool: usize,
        first_pos: usize,
        qk_scale: f32,
        head_scale: f32,
    ) -> Res<()> {
        let s = self.stream();
        // MEMRA_B200_DSA_DECODE door (level >= 1): the head-blocked decode scorer. Engages only
        // at decode widths and only from MLA_DSA_SCORE_MIN_POOLS up, where the block count can
        // fill the die; below that the shipped dispatch's own measured crossover already sends
        // decode to the reference kernel, which wins there. Bit-identical, so this is a speed
        // choice and nothing else. See research/b200-dsa-decode-20260902/ROOFLINE.md §2.
        if mla_dsa_decode_level() >= 1
            && (1..=MLA_DSA_ARM_T_MAX).contains(&t_q)
            && n_pools >= MLA_DSA_SCORE_MIN_POOLS
        {
            let rc = unsafe {
                memra_mla_kpool_score_dsa_f32(
                    q.device_ptr(&s).0 as *const f32,
                    pool_keys.device_ptr(&s).0 as *const f32,
                    head_weights.device_ptr(&s).0 as *const f32,
                    score.device_ptr_mut(&s).0 as *mut f32,
                    t_q as i32,
                    heads as i32,
                    d as i32,
                    n_pools as i32,
                    pool as i32,
                    first_pos as i32,
                    qk_scale,
                    head_scale,
                    s.cu_stream() as *mut c_void,
                )
            };
            if !mla_dsa_geometry_refusal(rc) {
                mla_dsa_announce(
                    "kpool_score",
                    t_q,
                    &format!("arm=head-blocked heads={heads} pools={n_pools} class=bit-identical"),
                );
                return ck("kpool_score_dsa", rc);
            }
        }
        unsafe {
            ck(
                "kpool_score",
                memra_mla_kpool_score_f32(
                    q.device_ptr(&s).0 as *const f32,
                    pool_keys.device_ptr(&s).0 as *const f32,
                    head_weights.device_ptr(&s).0 as *const f32,
                    score.device_ptr_mut(&s).0 as *mut f32,
                    t_q as i32,
                    heads as i32,
                    d as i32,
                    n_pools as i32,
                    pool as i32,
                    first_pos as i32,
                    qk_scale,
                    head_scale,
                    s.cu_stream() as *mut c_void,
                ),
            )
        }
    }

    /// The RETAINED reference scorer: block per (query, pool), one thread per head, head sum
    /// walked sequentially by thread 0. It defines the arithmetic [`Engine::mla_kpool_score`]
    /// reproduces, and it is the only consumer-visible reason this crate still builds the slow
    /// kernel. Not a serving path — `O(t_q * n_pools)` blocks of `heads` threads.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_kpool_score_ref(
        &self,
        q: &CudaSlice<f32>,
        pool_keys: &CudaSlice<f32>,
        head_weights: &CudaSlice<f32>,
        score: &mut CudaSlice<f32>,
        t_q: usize,
        heads: usize,
        d: usize,
        n_pools: usize,
        pool: usize,
        first_pos: usize,
        qk_scale: f32,
        head_scale: f32,
    ) -> Res<()> {
        let s = self.stream();
        unsafe {
            ck(
                "kpool_score_ref",
                memra_mla_kpool_score_ref_f32(
                    q.device_ptr(&s).0 as *const f32,
                    pool_keys.device_ptr(&s).0 as *const f32,
                    head_weights.device_ptr(&s).0 as *const f32,
                    score.device_ptr_mut(&s).0 as *mut f32,
                    t_q as i32,
                    heads as i32,
                    d as i32,
                    n_pools as i32,
                    pool as i32,
                    first_pos as i32,
                    qk_scale,
                    head_scale,
                    s.cu_stream() as *mut c_void,
                ),
            )
        }
    }

    /// Top-`select_k` pools per query expanded to ascending cache rows, tail appended, -1 padded.
    ///
    /// Radix select on the 64-bit order key `(desc32(score) << 32) | pool_index`, whose ascending
    /// order IS the oracle's "score descending, pool index ascending" — see the ORDER contract
    /// block in `cu/mla_attn.cu`. `O(8 * n_pools / threads)` per query, independent of `select_k`.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_kpool_select(
        &self,
        score: &CudaSlice<f32>,
        idx: &mut CudaSlice<i32>,
        t_q: usize,
        n_pools: usize,
        pool: usize,
        select_k: usize,
        width: usize,
        first_pos: usize,
        always_tail: bool,
    ) -> Res<()> {
        let s = self.stream();
        // MEMRA_B200_DSA_SELECT door: the exact multi-CTA selector. Byte-identical output, so
        // this is a speed choice and nothing else; it engages only where the single-CTA kernel
        // has parallelism to gain (see MLA_DSA_SELECT_MIN_POOLS).
        if mla_dsa_select_on() && mla_dsa_select_engages(t_q, n_pools) {
            let n_ctas = unsafe { memra_mla_kpool_select_ctas(n_pools as i32) };
            let stride = unsafe { memra_mla_kpool_select_ws_ints(n_ctas) };
            let mut ws = self.uninit_i32(t_q * stride as usize)?;
            mla_dsa_select_announce(t_q, n_pools, n_ctas);
            return unsafe {
                ck(
                    "kpool_select_dsa",
                    memra_mla_kpool_select_dsa_f32(
                        score.device_ptr(&s).0 as *const f32,
                        idx.device_ptr_mut(&s).0 as *mut i32,
                        ws.device_ptr_mut(&s).0 as *mut i32,
                        t_q as i32,
                        n_pools as i32,
                        pool as i32,
                        select_k as i32,
                        width as i32,
                        first_pos as i32,
                        i32::from(always_tail),
                        s.cu_stream() as *mut c_void,
                    ),
                )
            };
        }
        unsafe {
            ck(
                "kpool_select",
                memra_mla_kpool_select_f32(
                    score.device_ptr(&s).0 as *const f32,
                    idx.device_ptr_mut(&s).0 as *mut i32,
                    t_q as i32,
                    n_pools as i32,
                    pool as i32,
                    select_k as i32,
                    width as i32,
                    first_pos as i32,
                    i32::from(always_tail),
                    s.cu_stream() as *mut c_void,
                ),
            )
        }
    }

    /// The `select_k`-rounds reference selection — the DEFINITION of the order the radix kernel
    /// above must reproduce. NOT a serving path: it is `O(select_k * n_pools / threads)` and
    /// exists so `gpu_kpool_radix_selection_is_byte_identical_to_the_reference_kernel` can hold
    /// the fast kernel to it at shapes the micro fixture cannot reach.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_kpool_select_ref(
        &self,
        score: &CudaSlice<f32>,
        idx: &mut CudaSlice<i32>,
        t_q: usize,
        n_pools: usize,
        pool: usize,
        select_k: usize,
        width: usize,
        first_pos: usize,
        always_tail: bool,
    ) -> Res<()> {
        let s = self.stream();
        unsafe {
            ck(
                "kpool_select_ref",
                memra_mla_kpool_select_ref_f32(
                    score.device_ptr(&s).0 as *const f32,
                    idx.device_ptr_mut(&s).0 as *mut i32,
                    t_q as i32,
                    n_pools as i32,
                    pool as i32,
                    select_k as i32,
                    width as i32,
                    first_pos as i32,
                    i32::from(always_tail),
                    s.cu_stream() as *mut c_void,
                ),
            )
        }
    }

    /// Strided-batched BF16 tensor-core GEMM over per-head planes — the
    /// MEMRA_MLA_TC_PREFILL absorb/decompress engine. Per head `b` in `0..batch`:
    /// `y_b[m, n] = x_b[m, k] @ w_b[n, k]^T`, f32 accumulate.
    ///
    /// `w` is the bf16 conversion-split weight plane: per-head `[n, k]` row-major,
    /// batch stride `n * k` (baked into the C side). `x` is a bf16 VIEW of a
    /// `[m, batch, k]` activation plane: per-head row stride `x_rs`, per-head base
    /// offset `x_bs` — for the canonical `[t, n_head, d]` layout that is
    /// `x_rs = batch * k`, `x_bs = k`. `y` mirrors that with `y_rs`/`y_bs` over `n`.
    ///
    /// `y_bf16` selects the output dtype: `true` writes bf16 (feeds the TC attention
    /// kernel directly, one fewer convert), `false` writes f32 (re-enters the f32
    /// stream). The caller passes `y` as raw bytes either way; an f32 output slice
    /// is viewed through its byte layout by the caller (`mla_bf16_gemm_sb_f32out`).
    ///
    /// rc 2xxxx (no cuBLASLt heuristic for the shape) is a DECLINE class the caller
    /// may fall back on; everything else is a hard error.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_bf16_gemm_sb_raw(
        &self,
        w_bf16: &CudaSlice<u8>,
        x_bf16: &CudaSlice<u8>,
        y_ptr: u64,
        m: usize,
        n: usize,
        k: usize,
        x_rs: usize,
        x_bs: usize,
        y_rs: usize,
        y_bs: usize,
        batch: usize,
        y_bf16: bool,
    ) -> Res<i32> {
        // Workspace from the shared f16/bf16 Lt scratch (bf16_tc_gemm pattern).
        let mut guard = self.f16_scratch.lock().unwrap();
        if guard.is_none() {
            *guard = Some(crate::f16_ffi::F16Scratch::with_capacity(self, 2)?);
        }
        let s_scr = guard.as_mut().unwrap();
        let s = self.stream();
        let rc = unsafe {
            memra_bf16_gemm_sb(
                w_bf16.device_ptr(&s).0 as *const c_void,
                x_bf16.device_ptr(&s).0 as *const c_void,
                y_ptr as *mut c_void,
                m as i32,
                n as i32,
                k as i32,
                x_rs as i64,
                x_bs as i64,
                y_rs as i64,
                y_bs as i64,
                batch as i32,
                i32::from(y_bf16),
                s_scr.ws.device_ptr_mut(&s).0 as *mut c_void,
                crate::f16_ffi::F16_WS_BYTES,
                s.cu_stream() as *mut c_void,
            )
        };
        Ok(rc)
    }

    /// [`Engine::mla_bf16_gemm_sb_raw`] with a bf16 output plane (absorb: feeds the TC
    /// attention kernel). Non-decline errors are named; a 2xxxx decline is returned as
    /// `Ok(false)` so the door can fall back to the per-position kernels.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_bf16_gemm_sb_bf16out(
        &self,
        w_bf16: &CudaSlice<u8>,
        x_bf16: &CudaSlice<u8>,
        y_bf16: &mut CudaSlice<u8>,
        m: usize,
        n: usize,
        k: usize,
        x_rs: usize,
        x_bs: usize,
        y_rs: usize,
        y_bs: usize,
        batch: usize,
    ) -> Res<bool> {
        let s = self.stream();
        let (y_ptr, _gy) = y_bf16.device_ptr_mut(&s);
        let rc = self.mla_bf16_gemm_sb_raw(
            w_bf16, x_bf16, y_ptr, m, n, k, x_rs, x_bs, y_rs, y_bs, batch, true,
        )?;
        match rc {
            0 => Ok(true),
            r if (20000..30000).contains(&r) => Ok(false),
            r => Err(format!(
                "mla bf16 strided-batched GEMM (bf16 out) failed: rc {r} \
                 (m={m} n={n} k={k} batch={batch})"
            )
            .into()),
        }
    }

    /// [`Engine::mla_bf16_gemm_sb_raw`] with an f32 output plane (decompress: re-enters
    /// the f32 stream). Same decline contract as the bf16-out twin.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_bf16_gemm_sb_f32out(
        &self,
        w_bf16: &CudaSlice<u8>,
        x_bf16: &CudaSlice<u8>,
        y_f32: &mut CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
        x_rs: usize,
        x_bs: usize,
        y_rs: usize,
        y_bs: usize,
        batch: usize,
    ) -> Res<bool> {
        let s = self.stream();
        let (y_ptr, _gy) = y_f32.device_ptr_mut(&s);
        let rc = self.mla_bf16_gemm_sb_raw(
            w_bf16, x_bf16, y_ptr, m, n, k, x_rs, x_bs, y_rs, y_bs, batch, false,
        )?;
        match rc {
            0 => Ok(true),
            r if (20000..30000).contains(&r) => Ok(false),
            r => Err(format!(
                "mla bf16 strided-batched GEMM (f32 out) failed: rc {r} \
                 (m={m} n={n} k={k} batch={batch})"
            )
            .into()),
        }
    }

    /// Absorbed-form MQA attention over a GATHERED index list (one list per query, shared across
    /// heads). Same body as `mla_attn_absorbed`; only the cache walk differs.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_attn_gathered(
        &self,
        q_lat: &CudaSlice<f32>,
        q_pe: &CudaSlice<f32>,
        cache: &CudaSlice<f32>,
        idx: &CudaSlice<i32>,
        o_lat: &mut CudaSlice<f32>,
        n_head: usize,
        kv_rank: usize,
        d_rope: usize,
        t_q: usize,
        n_slots: usize,
        scale: f32,
    ) -> Res<()> {
        let s = self.stream();
        // MEMRA_B200_DSA_DECODE door, checked FIRST: its arms fight the same 64-CTA t_q=1
        // geometry the output-range split below does, without repeating the slot walk.
        // THE TWO LEVELS DIFFER TODAY, and the difference is this PR's headline: the shipped
        // table is [0, 32, 0, 0, 1, ...], so at t_q=1 level 1 takes arm 0 (falls through to
        // the sibling split door) while level 2 takes warp-online chunks=32 -- +9.7% vs
        // +43.1% in the 256k serving A/B. The `a >= 2 && dsa_level < 2` guard below exists
        // precisely because they differ, and the level boundary IS the numeric-class
        // admission boundary. See research/b200-dsa-decode-20260902/ROOFLINE.md.
        let dsa_level = mla_dsa_decode_level();
        let dsa_arm = if dsa_level >= 1 && t_q <= MLA_DSA_ARM_T_MAX {
            // `_effective` already enforces the named-class width rule (plain decode only, so
            // the spec-verify batch never sees `dsa-warp-online-f32`); level 2 is the second,
            // independent admission for the same class.
            let a = mla_dsa_attn_arm_effective(t_q);
            if a >= 2 && dsa_level < 2 { 0 } else { a }
        } else {
            0
        };
        if dsa_arm >= 2 {
            let cells = t_q * n_head * dsa_arm as usize;
            let mut part_m = self.uninit(cells)?;
            let mut part_d = self.uninit(cells)?;
            let mut part_acc = self.uninit(cells * kv_rank)?;
            let rc = unsafe {
                memra_mla_dsa_attn_split_f32(
                    q_lat.device_ptr(&s).0 as *const f32,
                    q_pe.device_ptr(&s).0 as *const f32,
                    cache.device_ptr(&s).0 as *const f32,
                    idx.device_ptr(&s).0 as *const i32,
                    o_lat.device_ptr_mut(&s).0 as *mut f32,
                    part_m.device_ptr_mut(&s).0 as *mut f32,
                    part_d.device_ptr_mut(&s).0 as *mut f32,
                    part_acc.device_ptr_mut(&s).0 as *mut f32,
                    n_head as i32,
                    kv_rank as i32,
                    d_rope as i32,
                    t_q as i32,
                    n_slots as i32,
                    dsa_arm,
                    scale,
                    s.cu_stream() as *mut c_void,
                )
            };
            if !mla_dsa_geometry_refusal(rc) {
                mla_dsa_announce(
                    "attn_gathered",
                    t_q,
                    &format!("arm=warp-online chunks={dsa_arm} class=dsa-warp-online-f32"),
                );
                return ck("attn_gathered_dsa_warp", rc);
            }
        } else if dsa_arm == 1 {
            let rc = unsafe {
                memra_mla_attn_gathered_dsa_f32(
                    q_lat.device_ptr(&s).0 as *const f32,
                    q_pe.device_ptr(&s).0 as *const f32,
                    cache.device_ptr(&s).0 as *const f32,
                    idx.device_ptr(&s).0 as *const i32,
                    o_lat.device_ptr_mut(&s).0 as *mut f32,
                    n_head as i32,
                    kv_rank as i32,
                    d_rope as i32,
                    t_q as i32,
                    n_slots as i32,
                    scale,
                    s.cu_stream() as *mut c_void,
                )
            };
            if !mla_dsa_geometry_refusal(rc) {
                mla_dsa_announce("attn_gathered", t_q, "arm=single-pass class=bit-identical");
                return ck("attn_gathered_dsa", rc);
            }
        }
        // MEMRA_B200_MLA_DECODE_ARM door: output-range split from the t_q-keyed table
        // MLA_B200_ATTN_GATHERED_SPLIT. This twin repeats the score/softmax walk per split
        // block, unlike the absorb/decompress splits, which is why the B200 run found it a win
        // at t_q=1 only (see the table comment); every other cell is the shipped kernel.
        if let Some(split) = mla_b200_split_for(MlaB200Kernel::AttnGathered, t_q, kv_rank) {
            mla_b200_split_announce("attn_gathered", t_q, n_head, split);
            return unsafe {
                ck(
                    "attn_gathered_split_b200",
                    memra_mla_attn_gathered_split_f32(
                        q_lat.device_ptr(&s).0 as *const f32,
                        q_pe.device_ptr(&s).0 as *const f32,
                        cache.device_ptr(&s).0 as *const f32,
                        idx.device_ptr(&s).0 as *const i32,
                        o_lat.device_ptr_mut(&s).0 as *mut f32,
                        n_head as i32,
                        kv_rank as i32,
                        d_rope as i32,
                        t_q as i32,
                        n_slots as i32,
                        scale,
                        split,
                        s.cu_stream() as *mut c_void,
                    ),
                )
            };
        }
        unsafe {
            ck(
                "attn_gathered",
                memra_mla_attn_gathered_f32(
                    q_lat.device_ptr(&s).0 as *const f32,
                    q_pe.device_ptr(&s).0 as *const f32,
                    cache.device_ptr(&s).0 as *const f32,
                    idx.device_ptr(&s).0 as *const i32,
                    o_lat.device_ptr_mut(&s).0 as *mut f32,
                    n_head as i32,
                    kv_rank as i32,
                    d_rope as i32,
                    t_q as i32,
                    n_slots as i32,
                    scale,
                    s.cu_stream() as *mut c_void,
                ),
            )
        }
    }
}

#[cfg(test)]
mod dsa_select_default_tests {
    use super::{mla_dsa_select_engages, mla_dsa_select_on_from};

    #[test]
    fn default_on_only_zero_disarms_and_the_floors_still_gate() {
        assert!(mla_dsa_select_on_from(None));
        assert!(mla_dsa_select_on_from(Some("1")));
        assert!(!mla_dsa_select_on_from(Some("0")));
        assert!(!mla_dsa_select_on_from(Some(" 0 ")));
        // The flip does not reach short context: the floors are unchanged.
        assert!(!mla_dsa_select_engages(1, 65_535));
        assert!(mla_dsa_select_engages(1, 65_536));
        assert!(!mla_dsa_select_engages(2, 65_536));
        assert!(mla_dsa_select_engages(2, 262_144));
        assert!(!mla_dsa_select_engages(9, 1_000_000));
    }
}
