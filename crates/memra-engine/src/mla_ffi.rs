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
        r if r >= 10000 && r < 20000 => " (cudaError)",
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
    pub fn mla_index_append(
        &self,
        plane: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
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
                    a.device_ptr(&s).0 as *const f32,
                    b.device_ptr(&s).0 as *const f32,
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
