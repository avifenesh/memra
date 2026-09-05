//! FFI declarations for the DeepSeek-V4-Flash GPU kernels (cu/dsv4_gpu.cu, lane 4).
//!
//! House pattern (mmq_ffi kind): C-ABI host launchers in the libmemra_mmq.a static lib,
//! returning 0 ok / 10000+cudaError / 20000+heuristic / 30000+matmul / 4000x contract
//! bands; the stream rides as `*mut c_void` (`stream.cu_stream()`).

use std::os::raw::c_void;

unsafe extern "C" {
    // iteration-5 F-itemisation instrument (see dsv4_gpu.rs Dsv4Phase).
    pub fn memra_dsv4_nvtx_push(name: *const std::os::raw::c_char) -> i32;
    pub fn memra_dsv4_nvtx_pop() -> i32;
    pub fn memra_dsv4_nvfp4_deq_bf16(
        w: *const c_void,
        sc: *const c_void,
        scale2: f32,
        rows: i32,
        cols: i32,
        out: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_mxfp4_deq_bf16(
        w: *const c_void,
        sc: *const c_void,
        rows: i32,
        cols: i32,
        out: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_cvt_bf16(x: *const f32, o: *mut c_void, n: i64, stream: *mut c_void) -> i32;
    pub fn memra_dsv4_embed_rows(
        table_bf16: *const c_void,
        ids: *const i32,
        out: *mut f32,
        n_ids: i32,
        ncols: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_gather_bf16(
        x: *const c_void,
        idx: *const i32,
        out: *mut c_void,
        g: i32,
        d: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_scatter_add(
        y: *mut f32,
        contrib: *const f32,
        idx: *const i32,
        g: i32,
        d: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_add_inplace(y: *mut f32, x: *const f32, n: i64, stream: *mut c_void) -> i32;
    pub fn memra_dsv4_take_cols(
        src: *const f32,
        dst: *mut f32,
        s: i32,
        n: i32,
        stride: i64,
        col_off: i64,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_place_cols(
        src: *const f32,
        dst: *mut f32,
        s: i32,
        n: i32,
        stride: i64,
        col_off: i64,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_repeat_hc(
        e: *const f32,
        h: *mut f32,
        s: i32,
        hc: i32,
        d: i32,
        stream: *mut c_void,
    ) -> i32;
    /// iteration-5: row-blocked twin of `memra_dsv4_dots_f32`. Same arithmetic, same
    /// reduction tree, same order -- only the block geometry differs, so it is bit-identical.
    pub fn memra_dsv4_dots_f32_rowblk(
        x: *const f32,
        w: *const c_void,
        w_is_bf16: i32,
        y: *mut f32,
        s: i32,
        k: i32,
        n: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_dots_f32(
        x: *const f32,
        w: *const c_void,
        w_is_bf16: i32,
        y: *mut f32,
        s: i32,
        k: i32,
        n: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_dots_f32acc(
        x: *const f32,
        w: *const c_void,
        w_is_bf16: i32,
        y: *mut f32,
        s: i32,
        k: i32,
        n: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_gemm_bf16(
        w_bf16: *const c_void,
        x_bf16: *const c_void,
        y_f32: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        dev: i32,
        ws: *mut c_void,
        ws_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_rmsnorm(
        x: *const f32,
        w: *const f32,
        dst: *mut f32,
        rows: i32,
        ncols: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_rope(
        x: *mut f32,
        n_pos: i32,
        n_vec: i32,
        dim: i32,
        rd: i32,
        cs: *const f32,
        positions: *const i32,
        inverse: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_headrms(x: *mut f32, rows: i32, d: i32, eps: f32, stream: *mut c_void)
    -> i32;
    pub fn memra_dsv4_act_quant(
        x: *mut f32,
        rows: i32,
        stride: i64,
        prefix_len: i32,
        block: i32,
        clamp_only: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_fp4_act_quant(
        x: *mut f32,
        rows: i32,
        stride: i64,
        len: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_hadamard(
        x: *mut f32,
        rows: i32,
        d: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_compressor_pool(
        kv: *const f32,
        score: *const f32,
        ape: *const f32,
        out: *mut f32,
        nb: i32,
        ratio: i32,
        d: i32,
        latent: i32,
        overlap: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_indexer_score(
        q: *const f32,
        ckv: *const f32,
        w: *const f32,
        wscale: f32,
        score: *mut f32,
        s: i32,
        heads: i32,
        hd: i32,
        nb: i32,
        ratio: i32,
        lim0: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_sink_attn(
        q: *const f32,
        kv: *const f32,
        idxs: *const i32,
        sink: *const f32,
        o: *mut f32,
        s: i32,
        heads: i32,
        hd: i32,
        slots: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_rowsq_scale(
        x: *const f32,
        mixes: *mut f32,
        s: i32,
        w: i32,
        rows: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_hc_collapse(
        x: *const f32,
        pre: *const f32,
        y: *mut f32,
        s: i32,
        hc: i32,
        d: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_hc_post(
        f: *const f32,
        residual: *const f32,
        post: *const f32,
        comb: *const f32,
        out: *mut f32,
        s: i32,
        hc: i32,
        d: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_act_quant_fp8(
        x: *const f32,
        codes: *mut c_void,
        scales: *mut f32,
        rows: i32,
        kdim: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_fp4_gemm(
        a_codes: *const c_void,
        a_scales: *const f32,
        w: *const c_void,
        wsc: *const c_void,
        scale2: f32,
        kind: i32,
        out: *mut f32,
        g: i32,
        n: i32,
        kdim: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_gather_rows_u8(
        x: *const c_void,
        idx: *const i32,
        out: *mut c_void,
        g: i32,
        row_bytes: i64,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_swiglu(
        gate: *const f32,
        up: *const f32,
        dst: *mut f32,
        rows: i32,
        inter: i32,
        limit: f32,
        wrow: *const f32,
        stream: *mut c_void,
    ) -> i32;
    // ---- lane 8: device-resident decode step (RECEIPTS.md "Lane 8")
    pub fn memra_dsv4_rope_at(
        x: *mut f32,
        n_vec: i32,
        dim: i32,
        rd: i32,
        cs: *const f32,
        pos: i32,
        inverse: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_hc_sinkhorn(
        mixes: *const f32,
        scale: *const f32,
        base: *const f32,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        hc: i32,
        iters: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_hc_head_pre(
        mixes: *const f32,
        scale: *const f32,
        base: *const f32,
        pre: *mut f32,
        hc: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_route(
        raw: *const f32,
        bias: *const f32,
        tid2eid: *const i32,
        tok: *const i32,
        ne: i32,
        topk: i32,
        route_scale: f32,
        sel: *mut i32,
        selw: *mut f32,
        order: *mut i32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn memra_dsv4_fp4_gemm_sel(
        a_codes: *const c_void,
        a_scales: *const f32,
        w_base: *const c_void,
        sc_base: *const c_void,
        s2: *const f32,
        sel: *const i32,
        proj: i32,
        a_stride_rows: i32,
        kind: i32,
        out: *mut f32,
        slots: i32,
        n: i32,
        kdim: i32,
        wstride: i64,
        sstride: i64,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_combine_rows(
        contrib: *const f32,
        order: *const i32,
        topk: i32,
        y: *mut f32,
        d: i64,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_build_idx(
        idx: *mut i32,
        pos: i32,
        win: i32,
        nb: i32,
        cap: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_topk_idx(
        score: *const f32,
        nb: i32,
        kk: i32,
        win: i32,
        idx_out: *mut i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_argmax(v: *const f32, n: i64, out: *mut i32, stream: *mut c_void) -> i32;
    /// iteration-5: `dst[0..cols) = src[idx[slot] * cols ..]`, the index read on the
    /// DEVICE so the DSpark markov chain needs no host round trip between steps.
    pub fn memra_dsv4_gather_row_by_idx(
        src: *const f32,
        idx: *const i32,
        slot: i32,
        dst: *mut f32,
        cols: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_gemv_bf16(
        w_bf16: *const c_void,
        x_bf16: *const c_void,
        y: *mut f32,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> i32;
    /// iteration-5 FP8 dense arm: as-stored e4m3 codes + host-decoded f32 block scales
    /// (exact pow2). BIT-IDENTICAL to memra_dsv4_gemv_bf16 over the dequant slab by
    /// construction (same value, same accumulation order — see cu header note).
    #[allow(clippy::too_many_arguments)]
    pub fn memra_dsv4_gemv_fp8(
        w_codes: *const c_void,
        sc_f32: *const f32,
        sc_cols: i32,
        x_bf16: *const c_void,
        y: *mut f32,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn memra_dsv4_sink_attn_dec(
        q: *const f32,
        kv: *const f32,
        idxs: *const i32,
        sink: *const f32,
        scores: *mut f32,
        evals: *mut f32,
        den: *mut f64,
        o: *mut f32,
        heads: i32,
        hd: i32,
        slots: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;

    // ── 0731 re-gate extension rung (MEMRA_DSV4_DOTS_ARM=f32x): f32-accumulation twins
    // of the remaining device-path f64 chains. Same signatures as their f64 twins except
    // sink dec's `den`, which rides a FLOAT view of the caller's f64 workspace (written
    // and read within the one entry point). f64 kernels untouched.
    pub fn memra_dsv4_rmsnorm_f32acc(
        x: *const f32,
        w: *const f32,
        dst: *mut f32,
        rows: i32,
        ncols: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_headrms_f32acc(
        x: *mut f32,
        rows: i32,
        d: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_rowsq_scale_f32acc(
        x: *const f32,
        mixes: *mut f32,
        s: i32,
        w: i32,
        rows: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn memra_dsv4_indexer_score_f32acc(
        q: *const f32,
        ckv: *const f32,
        w: *const f32,
        wscale: f32,
        score: *mut f32,
        s: i32,
        heads: i32,
        hd: i32,
        nb: i32,
        ratio: i32,
        lim0: i32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn memra_dsv4_sink_attn_dec_f32acc(
        q: *const f32,
        kv: *const f32,
        idxs: *const i32,
        sink: *const f32,
        scores: *mut f32,
        evals: *mut f32,
        den: *mut f32,
        o: *mut f32,
        heads: i32,
        hd: i32,
        slots: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    // iteration 3 (GPU DSpark drafter path)
    pub fn memra_dsv4_hc_mean(
        h: *const f32,
        out: *mut f32,
        s: i32,
        hc: i32,
        hidden: i32,
        stream: *mut c_void,
    ) -> i32;
    /// Model-entry stream expand, the inverse of `memra_dsv4_hc_mean`. Added for the
    /// glm5_next trunk (crate::hyper), which drives the whole hc kernel family.
    pub fn memra_dsv4_hc_expand(
        e: *const f32,
        out: *mut f32,
        s: i32,
        hc: i32,
        hidden: i32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn memra_dsv4_build_idx_redirect(
        idx: *mut i32,
        pos: i32,
        win: i32,
        nb: i32,
        cap: i32,
        pos0: i32,
        trans_base: i32,
        stream: *mut c_void,
    ) -> i32;

    // ---- iteration 3, rung 4: batched T=k+1 verify twins. Every entry is BIT-EXACT
    // against `m`/`nq` sequential single-position calls of its pinned twin (the design
    // law in cu/dsv4_gpu.cu's batched section): the added dimension only hoists the
    // WEIGHT load, never reorders an accumulation.
    #[allow(clippy::too_many_arguments)]
    pub fn memra_dsv4_gemv_bf16_m(
        w_bf16: *const c_void,
        x_bf16: *const c_void,
        y: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        xstride: i32,
        ystride: i32,
        stream: *mut c_void,
    ) -> i32;
    /// FP8 dense arm, batched twin (see memra_dsv4_gemv_fp8).
    #[allow(clippy::too_many_arguments)]
    pub fn memra_dsv4_gemv_fp8_m(
        w_codes: *const c_void,
        sc_f32: *const f32,
        sc_cols: i32,
        x_bf16: *const c_void,
        y: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        xstride: i32,
        ystride: i32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn memra_dsv4_dots_f32_mrow(
        x: *const f32,
        w: *const c_void,
        w_is_bf16: i32,
        y: *mut f32,
        s: i32,
        k: i32,
        n: i32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn memra_dsv4_dots_f32acc_mrow(
        x: *const f32,
        w: *const c_void,
        w_is_bf16: i32,
        y: *mut f32,
        s: i32,
        k: i32,
        n: i32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn memra_dsv4_hc_sinkhorn_m(
        mixes: *const f32,
        scale: *const f32,
        base: *const f32,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        s: i32,
        hc: i32,
        iters: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    /// Fused hc pre-chain: rowsq_scale + Sinkhorn (bit-preserving stationarity exit) +
    /// collapse, one launch per (site, token). `niters` is nullable — per-token executed
    /// Sinkhorn iteration counts, the gate's convergence receipt. Bit-identical to the
    /// three-kernel chain (hc_fused_pre_gpu.rs); host seam MEMRA_HC_FUSED_PRE.
    #[allow(clippy::too_many_arguments)]
    pub fn memra_dsv4_hc_pre_fused(
        x: *const f32,
        mixes: *const f32,
        scale: *const f32,
        base: *const f32,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        y: *mut f32,
        s: i32,
        hc: i32,
        d: i32,
        iters: i32,
        eps: f32,
        niters: *mut i32,
        stream: *mut c_void,
    ) -> i32;
    /// `MEMRA_HC_FUSED_PRE=2` (lane/b200-sinkhorn-fusion-20260902 follow-up): same stages
    /// and same signature as `memra_dsv4_hc_pre_fused` above; the Sinkhorn stage runs
    /// warp-0-only with `__syncwarp()` instead of `__syncthreads()` when hc<=4 (rows<=32),
    /// falling back to `memra_dsv4_hc_pre_fused` internally otherwise. Bit-identical to
    /// both the three-kernel chain and to the `=1` fused kernel by construction (a
    /// synchronization-primitive substitution only, no operand/order change).
    #[allow(clippy::too_many_arguments)]
    pub fn memra_dsv4_hc_pre_fused_v2(
        x: *const f32,
        mixes: *const f32,
        scale: *const f32,
        base: *const f32,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        y: *mut f32,
        s: i32,
        hc: i32,
        d: i32,
        iters: i32,
        eps: f32,
        niters: *mut i32,
        stream: *mut c_void,
    ) -> i32;
    /// `dsv4_hc_pre_fused_v3` with `rms_norm_zq8_f32_v2` appended in the same block
    /// (lane/hcpre-zq8-fusion-20260905): stages 1-3 verbatim, then the norm's two passes over the
    /// `y` just written, with the norm's own block width `rms_bd` (must be whole warps and <=
    /// `block`) and its own epsilon `eps_norm`. Emits `z` (the normed f32 row) and the q8_1 pair
    /// `(out_q, out_d)`. Bit-identical to running the two kernels in sequence.
    #[allow(clippy::too_many_arguments)]
    pub fn memra_dsv4_hc_pre_zq8(
        x: *const f32,
        mixes: *const f32,
        scale: *const f32,
        base: *const f32,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        y: *mut f32,
        s: i32,
        hc: i32,
        d: i32,
        iters: i32,
        eps: f32,
        niters: *mut i32,
        block: i32,
        sink_reg: i32,
        norm_w: *const f32,
        z: *mut f32,
        out_q: *mut i8,
        out_d: *mut f32,
        rms_bd: i32,
        eps_norm: f32,
        stream: *mut c_void,
    ) -> i32;
    /// `memra_dsv4_hc_pre_fused_v2` with the block size as a parameter (door
    /// `MEMRA_HC_PRE_BLOCK`, lane/b200-hcpre-wide-20260903). v2 hardcodes `<<<s, 128>>>`,
    /// one block per row, so at t=1 decode the whole call is ONE block of 128 threads on a
    /// 148-SM B200 and nsys prices it as the LARGEST kernel in the decode profile (17.5%,
    /// 31.1 us avg, 90.7 launches per token = 2 per layer x 45 layers). It moves ~128 KB in
    /// those 31 us, which is 4.1 GB/s: four warps cannot cover HBM latency. `block` = 128
    /// reproduces v2's thread partition exactly and is therefore bit-identical to it; wider
    /// blocks change stage 1's `dsv4_block_sum` partition (stage 3 stays bit-identical at
    /// any width, and stage 2 is warp-0-only either way), which is the NAMED NUMERIC CLASS
    /// `hc_pre_rowsq_blockwide`. Refuses a `block` that is not a power of two in [32, 1024].
    #[allow(clippy::too_many_arguments)]
    /// BENCH-ONLY phase-stamped twin of `memra_dsv4_hc_pre_fused_v3` (no split_collapse, no
    /// niters): `stamps` is 12 u64 on the device, [0..6) %globaltimer ns and [6..12) clock64 at
    /// the six phase boundaries named in the kernel's header. Only the gate binary calls it.
    pub fn memra_dsv4_hc_pre_fused_v3_stamped(
        x: *const f32,
        mixes: *const f32,
        scale: *const f32,
        base: *const f32,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        y: *mut f32,
        s: i32,
        hc: i32,
        d: i32,
        iters: i32,
        eps: f32,
        block: i32,
        sink_reg: i32,
        stamps: *mut u64,
        stream: *mut c_void,
    ) -> i32;
    /// hc pre-chain v4 (lane/hc-pre-phases-20260905): v3's arithmetic in v3's order on a
    /// register schedule (one round of loads kept for the combine, two barriers, Sinkhorn
    /// overlapped with the combine). 40025 = shape does not fit (caller runs v3).
    /// BENCH-ONLY phase-stamped twin of `memra_dsv4_hc_pre_v4` (12 u64 stamps: [0..6)
    /// %globaltimer ns, [6..12) clock64). Only the gate binary calls it.
    pub fn memra_dsv4_hc_pre_v4_stamped(
        x: *const f32,
        mixes: *const f32,
        scale: *const f32,
        base: *const f32,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        y: *mut f32,
        s: i32,
        hc: i32,
        d: i32,
        iters: i32,
        eps: f32,
        stamps: *mut u64,
        stream: *mut c_void,
    ) -> i32;
    /// hc pre-chain v4 with the norm folded in (lane/hc-pre-v4z-20260905): v4's outputs plus the
    /// `rms_norm_zq8_f32_v2` replay of `y` (z, q8_1 pair) with every operation pinned to the
    /// served kernel's compiled form. 40025 = shape does not fit (caller runs the two launches).
    pub fn memra_dsv4_hc_pre_v4z(
        x: *const f32,
        mixes: *const f32,
        scale: *const f32,
        base: *const f32,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        y: *mut f32,
        s: i32,
        hc: i32,
        d: i32,
        iters: i32,
        eps: f32,
        niters: *mut i32,
        norm_w: *const f32,
        z: *mut f32,
        out_q: *mut i8,
        out_d: *mut f32,
        eps_norm: f32,
        nb: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_hc_pre_v4(
        x: *const f32,
        mixes: *const f32,
        scale: *const f32,
        base: *const f32,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        y: *mut f32,
        s: i32,
        hc: i32,
        d: i32,
        iters: i32,
        eps: f32,
        niters: *mut i32,
        block: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_hc_pre_fused_v3(
        x: *const f32,
        mixes: *const f32,
        scale: *const f32,
        base: *const f32,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        y: *mut f32,
        s: i32,
        hc: i32,
        d: i32,
        iters: i32,
        eps: f32,
        niters: *mut i32,
        block: i32,
        // `MEMRA_HC_PRE_SINK_REG=1`: run stage 2 (Sinkhorn) in REGISTERS with `__shfl_sync`
        // instead of shared memory. BIT-IDENTICAL by construction, not a numeric class: every
        // row/column sum is gathered in the SAME order the shared loop used
        // (`for k: sum += __shfl_sync(mask, cv, r*hc+k)` against `for k: sum += comb[t*hc+k]`),
        // so the same addends land in the same sequence in the same running float. A tree
        // reduction would be fewer instructions and a different association; it is deliberately
        // not used. Falls back to the shared path when `hc*hc > 32` (the matrix must fit one
        // warp), checked in the launcher rather than assumed.
        sink_reg: i32,
        split_collapse: i32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn memra_dsv4_hc_head_pre_m(
        mixes: *const f32,
        scale: *const f32,
        base: *const f32,
        pre: *mut f32,
        s: i32,
        hc: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn memra_dsv4_route_m(
        raw: *const f32,
        bias: *const f32,
        tid2eid: *const i32,
        tok: *const i32,
        s: i32,
        ne: i32,
        topk: i32,
        route_scale: f32,
        sel: *mut i32,
        selw: *mut f32,
        order: *mut i32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn memra_dsv4_fp4_gemm_sel_g(
        a_codes: *const c_void,
        a_scales: *const f32,
        w_base: *const c_void,
        sc_base: *const c_void,
        s2: *const f32,
        sel: *const i32,
        proj: i32,
        a_stride_rows: i32,
        kind: i32,
        out: *mut f32,
        slots: i32,
        n: i32,
        kdim: i32,
        wstride: i64,
        sstride: i64,
        a_group: i32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn memra_dsv4_combine_rows_m(
        contrib: *const f32,
        order: *const i32,
        topk: i32,
        y: *mut f32,
        d: i64,
        s: i32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn memra_dsv4_sink_attn_dec_mq(
        q: *const f32,
        kv: *const f32,
        idxs: *const i32,
        sink: *const f32,
        scores: *mut f32,
        evals: *mut f32,
        den: *mut f64,
        o: *mut f32,
        nq: i32,
        heads: i32,
        hd: i32,
        slots: i32,
        idx_stride: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn memra_dsv4_sink_attn_dec_mq_f32acc(
        q: *const f32,
        kv: *const f32,
        idxs: *const i32,
        sink: *const f32,
        scores: *mut f32,
        evals: *mut f32,
        den: *mut f32,
        o: *mut f32,
        nq: i32,
        heads: i32,
        hd: i32,
        slots: i32,
        idx_stride: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memra_dsv4_scatter_rows(
        src: *const f32,
        dst: *mut f32,
        dst_rows: *const i32,
        n: i32,
        d: i32,
        stream: *mut c_void,
    ) -> i32;
}

/// rc -> Err with the kernel name (refuse loudly, house style).
pub fn ck(name: &str, rc: i32) -> Result<(), String> {
    if rc == 0 {
        Ok(())
    } else {
        Err(format!("dsv4 kernel {name} failed rc={rc}"))
    }
}
