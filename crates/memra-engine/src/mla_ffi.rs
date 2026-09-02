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
/// per boot — the receipt the B200 box A/B run must show (see LANE.md, receipt pending).
pub static MLA_B200_DECODE_ARM_DISPATCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `MEMRA_B200_MLA_DECODE_ARM=1` (default OFF, read per call — rollback seam), compile-time
/// gated to sm_100a builds (`cfg!(memra_sm100_tcgen05)`, set by build.rs for
/// `MEMRA_CUDA_ARCH=100a`): on a 120a/90a/89 build this is `false` unconditionally, so naked
/// non-B200 commands and the flag census see no behavior change from a var they cannot even
/// engage — the arch guard is a compile-time fact here, not a per-call detection cost.
///
/// Owner order 2026-09-02: "hardly improve the decode on these cards, before the full 1M."
/// This is a genuinely separate door from `MEMRA_MLA_DECODE_SPLIT` (glm5-decode-diet lever 4,
/// rig-generic, target ~1024 blocks, PRO6000-tuned) rather than a rename of it, per the
/// per-hardware-arm-selection law in CLAUDE.md: B200 SXM carries more SMs per device than the
/// PRO6000 pair that door was tuned on, and this arm ALSO covers `attn_gathered`, which the
/// generic split door never touched (no independent-output split existed for it before this
/// lane — see `memra_mla_attn_gathered_split_kernel` in cu/mla_attn.cu).
fn mla_b200_decode_arm_on() -> bool {
    cfg!(memra_sm100_tcgen05) && std::env::var("MEMRA_B200_MLA_DECODE_ARM").as_deref() == Ok("1")
}

/// Output-range split target for the B200 arm's absorb_q / decompress_v calls. Same
/// split-invariant arithmetic as `mla_decode_split_for` (bytes are identical at any split —
/// pure output-range partition of independent per-element matvecs), aimed at a higher block
/// count than the generic door's ~1024 target since a B200 SXM die carries more SMs than the
/// PRO6000 pair `MEMRA_MLA_DECODE_SPLIT` was tuned on. Scoped to t_q <= 8 per the owner's
/// "t=1 (and small-t verify, t<=8)" order — wider widths already reach the TC prefill chain
/// (`MEMRA_MLA_TC_PREFILL`, t >= 16) or fall through to the generic split door.
fn mla_b200_split_for(t_q: usize, blocks: usize, out_dim: usize) -> Option<i32> {
    const B200_TARGET_BLOCKS: usize = 2048;
    if !mla_b200_decode_arm_on()
        || t_q == 0
        || t_q > 8
        || blocks == 0
        || blocks >= B200_TARGET_BLOCKS
    {
        return None;
    }
    let want = B200_TARGET_BLOCKS.div_ceil(blocks);
    let cap = (out_dim / 32).max(1);
    let split = want.min(cap);
    if split <= 1 { None } else { Some(split as i32) }
}

/// Output-range split target for the B200 arm's `attn_gathered` call. Deliberately far more
/// conservative than `mla_b200_split_for`: every split factor here repeats the FULL
/// score/softmax tile walk (the kernel's dominant cost), so unlike the absorb/decompress
/// splits this is only a net win if the box is occupancy/latency-bound, which is a hardware
/// question this lane cannot answer without the B200 box (see cu/mla_attn.cu's
/// `memra_mla_attn_gathered_split_kernel` header and LANE.md "why not slot-split"). Caps at a
/// small fixed factor (fill roughly one extra wave, not many) rather than chasing the same
/// ~2048-block target as the independent-output kernels.
fn mla_b200_gathered_split_for(t_q: usize, blocks: usize, kv_rank: usize) -> Option<i32> {
    const B200_GATHERED_MAX_SPLIT: i32 = 4;
    if !mla_b200_decode_arm_on() || t_q == 0 || t_q > 8 || blocks == 0 || blocks >= 512 {
        return None;
    }
    let want = 512usize
        .div_ceil(blocks)
        .min(B200_GATHERED_MAX_SPLIT as usize);
    let cap = (kv_rank / 32).max(1);
    let split = want.min(cap);
    if split <= 1 { None } else { Some(split as i32) }
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
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
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
        // MEMRA_B200_MLA_DECODE_ARM door (checked first: sm_100a-tuned target, wider than the
        // generic door's; both split kernels are the same one, so this is only a policy pick).
        if let Some(split) = mla_b200_split_for(t_q, t_q * n_head, kv_rank) {
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
        // MEMRA_B200_MLA_DECODE_ARM door (checked first, see mla_absorb_q above).
        if let Some(split) = mla_b200_split_for(t_q, t_q * n_head, d_v) {
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
        // MEMRA_B200_MLA_DECODE_ARM door: conservative output-range split (see
        // mla_b200_gathered_split_for header — this repeats the score/softmax walk per split,
        // unlike the absorb/decompress splits, so the cap is deliberately small).
        if let Some(split) = mla_b200_gathered_split_for(t_q, t_q * n_head, kv_rank) {
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
