//! DeepSeek-V4.1-Flash on one GPU: the V4.1 program's kernels (`cu/dsv41_gpu.cu`) and a
//! single-card layer that runs them.
//!
//! A sibling of [`crate::dsv4_gpu`], not a parameterization of it: the DSv4 route is wired to two
//! cards, 43 layers, d 4096, compress ratios {4, 128} and 128-wide quant blocks, and keeps its
//! activations in f32. V4.1 keeps the reference program's bf16 activations, quantizes every
//! FP8/FP4 linear input per 32 values, shares one compressed KV and one indexer across a CSA2
//! layer group, narrows its decoder indexers through a candidate pool, carries the mHC pre-mix
//! from the previous FFN and adds Engram n-gram lookups at two layers. The semantic source is the
//! vendor `inference/model.py` at rev 1bae0e10; `crates/memra-gguf/src/model_plan/deepseek_v41.rs`
//! is the typed plan these kernels execute.
//!
//! Scope today: every V4.1 operation as a kernel plus the prefill/decode layer
//! ([`Ctx::block_forward`]) fed from the safetensors checkpoint, the same layer split across
//! N ranks ([`tp`]: tensor-parallel attention, expert-parallel MoE, a vocab-parallel head; gathers
//! only, so the N-rank program writes these bytes), and the whole trunk as a prefill followed by
//! one token per call with every sequence's state carried the vendor's way ([`decode`]). Serving
//! is not here yet.

use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, DevicePtr};
use memra_gguf::safetensors::StModel;
use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

pub mod decode;
pub mod draft;
pub mod engram;
pub mod experts;
pub mod graph;
pub mod tp;

pub mod ffi {
    use std::os::raw::c_void;
    /// Peers of one rank in a small gather inside kernels (`V41_PUSH_PEERS`).
    pub const PUSH_PEERS: usize = 7;
    /// The largest region a push kernel takes (`V41_PUSH_MAX`): the size of an inbox slot. What
    /// a gather actually pushes is bounded by `tp`'s measured limit, at most this.
    pub const PUSH_MAX: usize = 1 << 20;
    /// Device addresses of each peer's inbox slot for a producer: payload and word.
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    pub struct V41Push {
        pub d: [u64; PUSH_PEERS],
        pub w: [u64; PUSH_PEERS],
    }
    /// Each producer's slot in a rank's inbox (payload, word) and its region of the rank buffer.
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    pub struct V41Pull {
        pub d: [u64; PUSH_PEERS],
        pub w: [u64; PUSH_PEERS],
        pub off: [i64; PUSH_PEERS],
        pub pitch: [i64; PUSH_PEERS],
        pub width: [i32; PUSH_PEERS],
        pub rows: [i32; PUSH_PEERS],
    }
    // Return codes: 0 ok, 10000 + cudaError_t, 41xxx for a refused contract (cu/dsv41_gpu.cu).
    unsafe extern "C" {
        pub fn memra_v41_act_quant(
            x: *const c_void,
            ldx: i64,
            rowidx: *const i32,
            q: *mut c_void,
            ldq: i64,
            s: *mut c_void,
            lds: i64,
            y: *mut c_void,
            ldy: i64,
            rows: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_fp4_qdq(
            x: *mut c_void,
            ld: i64,
            rows: i32,
            n: i32,
            group: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_rmsnorm(
            x: *const c_void,
            ldx: i64,
            w: *const c_void,
            y: *mut c_void,
            ldy: i64,
            rows: i32,
            d: i32,
            eps: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_rope(
            x: *mut c_void,
            ld_row: i64,
            rows: i32,
            heads: i32,
            hd: i32,
            cosb: *const f32,
            sinb: *const f32,
            pos_base: i32,
            pos_step: i32,
            pos: *const i32,
            inverse: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_gemm_f8(
            a: *const c_void,
            lda: i64,
            a_s: *const c_void,
            ldas: i64,
            b: *const c_void,
            ldb: i64,
            b_s: *const c_void,
            ldbs: i64,
            bscale2: f32,
            wfmt: i32,
            c: *mut c_void,
            ldc: i64,
            out_f32: i32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_gemm_bf16(
            a: *const c_void,
            lda: i64,
            s_az: i64,
            b: *const c_void,
            ldb: i64,
            s_bz: i64,
            c: *mut c_void,
            ldc: i64,
            s_cz: i64,
            out_f32: i32,
            m: i32,
            n: i32,
            k: i32,
            batch: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_sparse_attn(
            q: *const c_void,
            ldq: i64,
            kv: *const c_void,
            idx: *const i32,
            ldi: i64,
            topk: i32,
            sink: *const f32,
            scale: f32,
            o: *mut c_void,
            ldo: i64,
            m: i32,
            heads: i32,
            hd: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_sparse_attn_keys(
            q: *const c_void,
            ldq: i64,
            keys: *const c_void,
            idx: *const i32,
            ldi: i64,
            topk: i32,
            sink: *const f32,
            scale: f32,
            o: *mut c_void,
            ldo: i64,
            m: i32,
            heads: i32,
            hd: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_comp_pool2(
            kv: *const f32,
            sc: *const f32,
            ld: i64,
            out: *mut c_void,
            ldo: i64,
            groups: i32,
            d: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_index_score(
            q: *const c_void,
            ldq: i64,
            k: *const c_void,
            t: i32,
            w: *const c_void,
            ldw: i64,
            wscale: f32,
            start_pos: i32,
            ratio: i32,
            cand: *const u8,
            ldc: i64,
            cb: i32,
            out: *mut c_void,
            ldo: i64,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_topk(
            score: *const c_void,
            lds: i64,
            t: i32,
            k: i32,
            out_w: i32,
            offset: i32,
            out: *mut i32,
            ldo: i64,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_candidates(
            score: *const c_void,
            lds: i64,
            t: i32,
            cb: i32,
            kblocks: i32,
            start_pos: i32,
            ratio: i32,
            keep: *mut u8,
            ldk: i64,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_topk_long(
            score: *const c_void,
            t: i32,
            k: i32,
            out_w: i32,
            offset: i32,
            out: *mut i32,
            ws: *mut i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_candidates_long(
            score: *const c_void,
            t: i32,
            cb: i32,
            kblocks: i32,
            start_pos: i32,
            ratio: i32,
            keep: *mut u8,
            ws: *mut i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_route(
            raw: *const f32,
            ldr: i64,
            bias: *const f32,
            ne: i32,
            topk: i32,
            norm: i32,
            route_scale: f32,
            sel: *mut i32,
            selw: *mut f32,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_swiglu(
            gate: *const c_void,
            up: *const c_void,
            ld: i64,
            wrow: *const f32,
            limit: f32,
            out: *mut c_void,
            ldo: i64,
            rows: i32,
            inter: i32,
            stream: *mut c_void,
        ) -> i32;
        /// h = SwiGLU(w1 a, w3 a) * wrow of a small-M tile list in one launch (FP4 experts).
        pub fn memra_v41_swiglu_tiles(
            a: *const c_void,
            lda: i64,
            a_s: *const c_void,
            ldas: i64,
            tiles: *const c_void,
            ntiles: *const i32,
            max_tiles: i32,
            ew: *const c_void,
            ldb: i64,
            ldbs: i64,
            wfmt: i32,
            wrow: *const f32,
            limit: f32,
            h: *mut c_void,
            ldh: i64,
            n: i32,
            k: i32,
            bm: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_scatter_add(
            src: *const c_void,
            lds: i64,
            rows_of: *const i32,
            y: *mut f32,
            ldy: i64,
            rows: i32,
            d: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_add_cast(
            y: *const f32,
            add: *const c_void,
            out: *mut c_void,
            n: i64,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_hc_mixes(
            x: *const c_void,
            fn_: *const f32,
            scale: *const f32,
            base: *const f32,
            rows: i32,
            hc: i32,
            d: i32,
            iters: i32,
            norm_eps: f32,
            hc_eps: f32,
            parts: *mut f32,
            mixes: *mut f32,
            pre: *mut f32,
            post: *mut f32,
            comb: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        /// The many-token partial pass for the virtual warps `[b0, b0 + nb)` (more than 8 rows).
        pub fn memra_v41_hc_mixes_part(
            x: *const c_void,
            fn_: *const f32,
            rows: i32,
            hc: i32,
            d: i32,
            b0: i32,
            nb: i32,
            parts: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        /// The finish of [`memra_v41_hc_mixes`] from complete partial sums.
        pub fn memra_v41_hc_mixes_finish(
            parts: *const f32,
            scale: *const f32,
            base: *const f32,
            rows: i32,
            hc: i32,
            d: i32,
            iters: i32,
            norm_eps: f32,
            hc_eps: f32,
            mixes: *mut f32,
            pre: *mut f32,
            post: *mut f32,
            comb: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_hc_pre(
            x: *const c_void,
            mix: *const f32,
            rows: i32,
            hc: i32,
            d: i32,
            y: *mut c_void,
            ldy: i64,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_hc_post(
            f: *const c_void,
            ldf: i64,
            res: *const c_void,
            post: *const f32,
            comb: *const f32,
            rows: i32,
            hc: i32,
            d: i32,
            out: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_engram_dequant(
            rows: *const c_void,
            sc: *const c_void,
            n_rows: i64,
            hd: i32,
            out: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_engram_gate(
            h: *const c_void,
            kv: *const c_void,
            qw: *const c_void,
            kw: *const c_void,
            rows: i32,
            hc: i32,
            d: i32,
            eps: f32,
            inv_sqrt_d: f32,
            out: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_gather_rows(
            src: *const c_void,
            lds: i64,
            rowidx: *const i32,
            dst: *mut c_void,
            ldd: i64,
            rows: i32,
            d: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_gemm_ffma(
            a: *const c_void,
            lda: i64,
            b: *const c_void,
            ldb: i64,
            c: *mut f32,
            ldc: i64,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_cast(
            x: *const c_void,
            y: *mut c_void,
            n: i64,
            to_f32: i32,
            stream: *mut c_void,
        ) -> i32;
        /// `grp`: device array of `ng` groups `{i64 row0, i32 rows, f32 scale2, u64 w, u64 s}`.
        /// The reference schedule of the block-scaled GEMM (gates only).
        pub fn memra_v41_gemm_f8_ref(
            a: *const c_void,
            lda: i64,
            a_s: *const c_void,
            ldas: i64,
            b: *const c_void,
            ldb: i64,
            bs: *const c_void,
            ldbs: i64,
            bscale2: f32,
            wfmt: i32,
            c: *mut c_void,
            ldc: i64,
            out_f32: i32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        /// The sparse attention's index rows (window part, then selection rows).
        pub fn memra_v41_attn_idx(
            sel: *const i32,
            tw: i32,
            r0: i32,
            rows: i32,
            sp: i32,
            win: i32,
            ww: i32,
            tot: i32,
            out: *mut i32,
            stream: *mut c_void,
        ) -> i32;
        /// The index rows of several decode tokens in one forward (window part into
        /// `[ring | the forward's rows]`, then the selection rows).
        pub fn memra_v41_attn_idx_rows(
            sel: *const i32,
            tw: i32,
            r0: i32,
            rows: i32,
            sp: i32,
            win: i32,
            tot: i32,
            out: *mut i32,
            stream: *mut c_void,
        ) -> i32;
        /// The index rows of a decode call over several sequences (each row its own position
        /// and its own region of the KV rows).
        pub fn memra_v41_attn_idx_batch(
            sel: *const i32,
            tw: i32,
            pos: *const i32,
            base: *const i32,
            nsel: *const i32,
            rows: i32,
            win: i32,
            tot: i32,
            out: *mut i32,
            stream: *mut c_void,
        ) -> i32;
        /// `*dst[r] = src[r]` (rows of `words` 16-bit words; `dst` a device array of row
        /// addresses, a null one skipped).
        pub fn memra_v41_put_rows(
            src: *const c_void,
            lds: i64,
            dst: *const c_void,
            rows: i32,
            words: i32,
            stream: *mut c_void,
        ) -> i32;
        /// `row += bias` in fp32 (bias may be null), then the maximum and its lowest index.
        pub fn memra_v41_add_argmax(
            row: *mut f32,
            bias: *const f32,
            n: i32,
            out_val: *mut f32,
            out_idx: *mut i32,
            stream: *mut c_void,
        ) -> i32;
        /// Ratio-2 pooling of `rows` pairs named by device addresses (`kv[j]`, `sc[j]`: two fp32
        /// rows of `d` each), latent row j into `out`.
        pub fn memra_v41_comp_pool2_keys(
            kv: *const c_void,
            sc: *const c_void,
            out: *mut c_void,
            ldo: i64,
            rows: i32,
            d: i32,
            stream: *mut c_void,
        ) -> i32;
        /// `row + bias` in fp32, its maximum and lowest index as one 64-bit key (low word the
        /// index), reduced over several blocks; `key` holds `0xffffffff00000000` before.
        pub fn memra_v41_add_argmax_key(
            row: *mut f32,
            bias: *const f32,
            n: i32,
            store: i32,
            key: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        /// Each fp32 row's maximum and its lowest index (NaN for a row with a non-finite value).
        pub fn memra_v41_argmax_rows(
            x: *const f32,
            ld: i64,
            rows: i32,
            n: i32,
            out_val: *mut f32,
            out_idx: *mut i32,
            stream: *mut c_void,
        ) -> i32;
        /// The MoE dispatch plan on the device (`MoePlanDev`).
        pub fn memra_v41_moe_plan(
            sel: *const i32,
            selw: *const f32,
            rows: i32,
            topk: i32,
            ne: i32,
            bm: i32,
            e0: i32,
            e1: i32,
            counts: *mut i32,
            off: *mut i32,
            ptok: *mut i32,
            pw: *mut f32,
            order: *mut i32,
            tiles: *mut c_void,
            ntiles: *mut i32,
            err: *mut i32,
            stream: *mut c_void,
        ) -> i32;
        /// The grouped expert GEMMs over a device tile list and the rank's expert weight table.
        pub fn memra_v41_gemm_f8_tiles(
            a: *const c_void,
            lda: i64,
            a_s: *const c_void,
            ldas: i64,
            tiles: *const c_void,
            ntiles: *const i32,
            max_tiles: i32,
            ew: *const c_void,
            proj: i32,
            ldb: i64,
            ldbs: i64,
            wfmt: i32,
            c: *mut c_void,
            ldc: i64,
            out_f32: i32,
            n: i32,
            k: i32,
            bm: i32,
            stream: *mut c_void,
        ) -> i32;
        /// [`memra_v41_gemm_f8_tiles`] over activations quantized once a token: the list's row
        /// r reads row `ridx[r]` of `a` and `a_s` (tiles of 64 or 128 rows).
        pub fn memra_v41_gemm_f8_tiles_rows(
            a: *const c_void,
            lda: i64,
            a_s: *const c_void,
            ldas: i64,
            ridx: *const i32,
            tiles: *const c_void,
            ntiles: *const i32,
            max_tiles: i32,
            ew: *const c_void,
            proj: i32,
            ldb: i64,
            ldbs: i64,
            wfmt: i32,
            c: *mut c_void,
            ldc: i64,
            out_f32: i32,
            n: i32,
            k: i32,
            bm: i32,
            stream: *mut c_void,
        ) -> i32;
        /// A bench's choice of the most rows of a call that keep a family's small-row body in
        /// groups of 8 (0 block-scaled dense, 1 bf16, 2 fp32 FMA, 3 the one-pass mHC kernel).
        pub fn memra_v41_wide_rows(family: i32, rows: i32);
        pub fn memra_v41_moe_combine(
            slots: *const c_void,
            lds: i64,
            order: *const i32,
            topk: i32,
            shared: *const c_void,
            ldsh: i64,
            out: *mut c_void,
            ldo: i64,
            rows: i32,
            d: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_moe_owner(
            order: *const i32,
            n: i32,
            off: *const i32,
            per: i32,
            nr: i32,
            np: i32,
            out: *mut i32,
            bad: *mut i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_index_score_part(
            q: *const c_void,
            ldq: i64,
            k: *const c_void,
            t: i32,
            w: *const c_void,
            ldw: i64,
            wscale: f32,
            heads: i32,
            part: *mut f32,
            ldo: i64,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_index_score_finish(
            parts: *const c_void,
            n: i32,
            ldp: i64,
            t: i32,
            start_pos: i32,
            ratio: i32,
            cand: *const u8,
            ldc: i64,
            cb: i32,
            out: *mut c_void,
            ldo: i64,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        /// The producer's half of a small gather inside a kernel (`dst`: a [`V41Push`] of the
        /// even gathers' slots, an odd gather's `alt` bytes on; `ctr`: the rank's count of
        /// gathers on its card, which numbers this one).
        pub fn memra_v41_push(
            src: *const c_void,
            pitch: i64,
            width: i32,
            rows: i32,
            dst: *const V41Push,
            npeer: i32,
            ctr: *const u32,
            alt: i64,
            word_first: i32,
            stream: *mut c_void,
        ) -> i32;
        /// The consumer's half (`src`: a [`V41Pull`]); it counts the gather in `ctr`. A peer that
        /// never arrives sets bit 16 of `bad`.
        pub fn memra_v41_pull(
            src: *const V41Pull,
            npeer: i32,
            ctr: *mut u32,
            alt: i64,
            dst: *mut c_void,
            bad: *mut i32,
            stream: *mut c_void,
        ) -> i32;
        /// A strided region to (`unpack` 0; 2: the block is on a peer card) or from (1) a
        /// contiguous block.
        pub fn memra_v41_blk_pack(
            reg: *mut c_void,
            pitch: i64,
            width: i64,
            rows: i64,
            packed: *mut c_void,
            unpack: i32,
            stream: *mut c_void,
        ) -> i32;
        /// Gate helper: a region of 32-bit words set to a pattern of `(seed, word)`, or with `diff`
        /// the number of its words that differ from the pattern added to `*diff`.
        pub fn memra_v41_pat(
            reg: *mut c_void,
            pitch_w: i64,
            width_w: i32,
            rows: i32,
            seed: u32,
            diff: *mut i32,
            stream: *mut c_void,
        ) -> i32;
        /// Instruction ceilings (measurement, `dsv41-bench ceil`).
        pub fn memra_v41_peak(
            kind: i32,
            blocks: i32,
            threads: i32,
            iters: i32,
            out: *mut f32,
            stream: *mut c_void,
        ) -> i32;
    }
}

/// `rc` from a `memra_v41_*` launcher as a `Result`.
pub fn ck(name: &str, rc: i32) -> Result<(), String> {
    tally::hit(name);
    match rc {
        0 => Ok(()),
        r if r >= 10000 => Err(format!("{name}: CUDA error {}", r - 10000)),
        r => Err(format!("{name}: contract refused ({r})")),
    }
}

/// A count of the launches a call issues, by layer and kernel, for a gate binary (off unless
/// started; one relaxed load a launch when off). Every kernel launch reports through [`ck`] and
/// every device copy through [`copy_d2d`]. Timed, each report first waits for its card, so a
/// kernel's entry also holds the time from the issuing thread's report before it to its own end
/// (the launches run one at a time: what each costs, not what the call takes). With several
/// ranks each rank's thread keeps its own clock and the entries sum the ranks; a rank's wait
/// includes its peers' part of a gather, so the group must issue in parallel then.
pub mod tally {
    use std::cell::Cell;
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::time::Instant;
    static ON: AtomicBool = AtomicBool::new(false);
    static TIMED: AtomicBool = AtomicBool::new(false);
    static LAYER: AtomicUsize = AtomicUsize::new(usize::MAX);
    /// which start the threads' clocks belong to
    static EPOCH: AtomicU64 = AtomicU64::new(0);
    thread_local! {
        /// this thread's last report: (the start it belongs to, when)
        static LAST: Cell<Option<(u64, Instant)>> = const { Cell::new(None) };
    }
    /// `(layer, kernel) -> (launches, nanoseconds)`
    pub type Table = BTreeMap<(usize, String), (u64, u64)>;
    static T: Mutex<Table> = Mutex::new(BTreeMap::new());
    /// Start counting from zero; `timed` also waits for the card at every report.
    pub fn start(timed: bool) {
        T.lock().unwrap().clear();
        LAYER.store(usize::MAX, Ordering::Relaxed);
        EPOCH.fetch_add(1, Ordering::Relaxed);
        TIMED.store(timed, Ordering::Relaxed);
        ON.store(true, Ordering::Relaxed);
    }
    /// The layer the next launches belong to (`usize::MAX`: outside the layers).
    pub fn layer(li: usize) {
        if ON.load(Ordering::Relaxed) {
            hit("(host between layers)");
            LAYER.store(li, Ordering::Relaxed);
        }
    }
    #[inline]
    pub fn hit(name: &str) {
        if ON.load(Ordering::Relaxed) {
            let ns = if TIMED.load(Ordering::Relaxed) {
                // the issuing thread holds its card's context
                unsafe { cudarc::driver::sys::cuCtxSynchronize() };
                let (now, epoch) = (Instant::now(), EPOCH.load(Ordering::Relaxed));
                let was = LAST.with(|l| l.replace(Some((epoch, now))));
                match was {
                    Some((e, t)) if e == epoch => (now - t).as_nanos() as u64,
                    _ => 0,
                }
            } else {
                0
            };
            let k = (LAYER.load(Ordering::Relaxed), name.to_string());
            let mut t = T.lock().unwrap();
            let e = t.entry(k).or_insert((0, 0));
            e.0 += 1;
            e.1 += ns;
        }
    }
    /// Stop counting and return the table.
    pub fn take() -> Table {
        ON.store(false, Ordering::Relaxed);
        std::mem::take(&mut *T.lock().unwrap())
    }
}

/// A device address inside a live [`Buf`] (or null). Only `Buf` makes one, so a kernel argument
/// always names device memory; the host never dereferences it. Sizes stay the caller's contract,
/// as with every CUDA launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dp(u64);

impl Dp {
    pub const NULL: Dp = Dp(0);
    pub fn c(self) -> *const c_void {
        self.0 as *const c_void
    }
    pub fn m(self) -> *mut c_void {
        self.0 as *mut c_void
    }
    pub fn i(self) -> *const i32 {
        self.0 as *const i32
    }
    pub fn im(self) -> *mut i32 {
        self.0 as *mut i32
    }
    pub fn f(self) -> *const f32 {
        self.0 as *const f32
    }
    pub fn fm(self) -> *mut f32 {
        self.0 as *mut f32
    }
    pub fn u8c(self) -> *const u8 {
        self.0 as *const u8
    }
    pub fn u8m(self) -> *mut u8 {
        self.0 as *mut u8
    }
    pub fn addr(self) -> u64 {
        self.0
    }
}

/// A device byte buffer. Every V4.1 kernel takes untyped pointers; the element type is the
/// caller's contract (bf16 as raw u16, fp8 and e8m0 as bytes, f32, i32).
pub struct Buf {
    pub(crate) s: ManuallyDrop<CudaSlice<u8>>,
    pub len: usize,
    home: Home,
}

/// Where a buffer's memory goes when the buffer drops.
enum Home {
    /// back to the device pool
    Free,
    /// to its stream's free list
    Pool(Arc<TmpPool>),
    /// nowhere: the memory is another buffer's ([`Buf::alias`])
    Alias,
}

impl Drop for Buf {
    fn drop(&mut self) {
        // SAFETY: `s` is taken once, here, and never touched again
        let s = unsafe { ManuallyDrop::take(&mut self.s) };
        match &self.home {
            Home::Free => drop(s),
            Home::Pool(h) => h.put(s),
            Home::Alias => {
                let _ = s.leak();
            }
        }
    }
}

/// The largest buffer a [`TmpPool`] keeps. A decode step's temporaries are a few rows of the
/// model width (the widest is 4 x 5120 fp32 rows of 4 streams, 320 KiB); a prefill's are
/// megabytes to gigabytes and go back to the device pool as before.
const TMP_POOL_MAX_BUF: usize = 1 << 20;
/// The most a [`TmpPool`] holds. A decode layer has about 45 temporaries alive or free at once,
/// under 4 MiB; 64 MiB leaves room for several widths and sequences and bounds what one stream
/// keeps out of the device pool.
const TMP_POOL_MAX_BYTES: usize = 64 << 20;

/// One stream's freed small buffers, by size, for its next allocation of that size.
///
/// A one-token forward allocates and frees about 45 temporaries a layer; through the driver that
/// is 90 calls a layer on the issuing thread, as many as its kernel launches. A buffer is handed
/// out again only on the stream that allocated it, and the kernels that write it are issued after
/// everything that read it before, so the order on the device is the order a stream-ordered free
/// and allocation give. (A peer stream that reads a buffer holds its producer's stream with an
/// event until the copy is done, exactly as before a free.)
#[derive(Default)]
pub struct TmpPool {
    free: Mutex<TmpFree>,
    /// the free list of a stream capture ([`graph`]): every buffer the captured launches use
    /// comes from it and stays in it, whatever its size, because the graph holds the addresses;
    /// what it lacks is allocated outside the stream's order (a stream-ordered allocation inside
    /// a capture would become a node of the graph)
    capture: bool,
}

/// The free buffers by size, and their bytes.
#[derive(Default)]
struct TmpFree {
    by_len: HashMap<usize, Vec<CudaSlice<u8>>>,
    bytes: usize,
}

impl TmpPool {
    fn take(&self, n: usize) -> Option<CudaSlice<u8>> {
        if n > TMP_POOL_MAX_BUF && !self.capture {
            return None;
        }
        let mut f = self.free.lock().unwrap_or_else(|e| e.into_inner());
        let s = f.by_len.get_mut(&n)?.pop()?;
        f.bytes -= n;
        Some(s)
    }
    fn put(&self, s: CudaSlice<u8>) {
        let n = s.len();
        if n > TMP_POOL_MAX_BUF && !self.capture {
            return;
        }
        let mut f = self.free.lock().unwrap_or_else(|e| e.into_inner());
        if f.bytes + n > TMP_POOL_MAX_BYTES && !self.capture {
            return;
        }
        f.bytes += n;
        f.by_len.entry(n).or_default().push(s);
    }
}

impl Buf {
    fn own(s: CudaSlice<u8>, len: usize, home: Home) -> Self {
        Self {
            s: ManuallyDrop::new(s),
            len,
            home,
        }
    }
    pub fn zeros(st: &Arc<CudaStream>, len: usize) -> Result<Self, String> {
        let s = st
            .alloc_zeros::<u8>(len.max(1))
            .map_err(|e| format!("alloc {len}: {e:?}"))?;
        Ok(Self::own(s, len, Home::Free))
    }
    /// Device memory without the zero fill, for a buffer its kernels write in full before
    /// anything reads it (stream-ordered like every allocation here).
    pub fn uninit(st: &Arc<CudaStream>, len: usize) -> Result<Self, String> {
        // SAFETY: the bytes are device memory the caller writes before reading; the host never
        // reads them uninitialized (every host copy follows the kernel that wrote them)
        let s = unsafe { st.alloc::<u8>(len.max(1)) }.map_err(|e| format!("alloc {len}: {e:?}"))?;
        Ok(Self::own(s, len, Home::Free))
    }
    /// [`Self::zeros`] or [`Self::uninit`] through `pool`: a freed buffer of this size when the
    /// pool has one (zero-filled on the stream when `zero`), and back to the pool on drop.
    fn pooled(
        st: &Arc<CudaStream>,
        pool: &Arc<TmpPool>,
        len: usize,
        zero: bool,
    ) -> Result<Self, String> {
        let n = len.max(1);
        let s = match pool.take(n) {
            Some(mut s) => {
                if zero {
                    st.memset_zeros(&mut s)
                        .map_err(|e| format!("memset {len}: {e:?}"))?;
                }
                s
            }
            None if pool.capture => {
                st.context()
                    .bind_to_thread()
                    .map_err(|e| format!("bind: {e:?}"))?;
                // SAFETY: `n` bytes of device memory owned by the slice made from them, which
                // frees them when it drops; written before read as in `uninit`, or zeroed here
                let mut s = unsafe {
                    let p = cudarc::driver::result::malloc_sync(n)
                        .map_err(|e| format!("alloc {len}: {e:?}"))?;
                    st.upgrade_device_ptr::<u8>(p, n)
                };
                if zero {
                    st.memset_zeros(&mut s)
                        .map_err(|e| format!("memset {len}: {e:?}"))?;
                }
                s
            }
            None if zero => st
                .alloc_zeros::<u8>(n)
                .map_err(|e| format!("alloc {len}: {e:?}"))?,
            // SAFETY: as in `uninit`
            None => unsafe { st.alloc::<u8>(n) }.map_err(|e| format!("alloc {len}: {e:?}"))?,
        };
        let home = if n <= TMP_POOL_MAX_BUF || pool.capture {
            Home::Pool(pool.clone())
        } else {
            Home::Free
        };
        Ok(Self::own(s, len, home))
    }
    pub fn from_bytes(st: &Arc<CudaStream>, b: &[u8]) -> Result<Self, String> {
        if b.is_empty() {
            return Self::zeros(st, 0);
        }
        let s = st
            .clone_htod(b)
            .map_err(|e| format!("htod {}: {e:?}", b.len()))?;
        Ok(Self::own(s, b.len(), Home::Free))
    }
    pub fn from_f32(st: &Arc<CudaStream>, v: &[f32]) -> Result<Self, String> {
        Self::from_bytes(st, &f32_bytes(v))
    }
    pub fn from_i32(st: &Arc<CudaStream>, v: &[i32]) -> Result<Self, String> {
        Self::from_bytes(st, &i32_bytes(v))
    }
    pub fn p(&self, st: &Arc<CudaStream>) -> Dp {
        let (p, _g) = (*self.s).device_ptr(st);
        Dp(p)
    }
    /// A second handle on this buffer's memory that frees nothing when it drops.
    ///
    /// # Safety
    /// The caller keeps the buffer alive, and its bytes meaningful, for as long as the alias is
    /// read: the alias does not hold it.
    pub unsafe fn alias(&self, st: &Arc<CudaStream>) -> Self {
        // SAFETY: the address and length of a live allocation; `Home::Alias` never frees it
        let s = unsafe { st.upgrade_device_ptr::<u8>(self.p(st).addr(), self.s.len()) };
        Self::own(s, self.len, Home::Alias)
    }
    /// Device address `off` bytes in.
    pub fn at(&self, st: &Arc<CudaStream>, off: usize) -> Dp {
        assert!(off <= self.len, "Buf::at past the end");
        Dp(self.p(st).0 + off as u64)
    }
    pub fn host(&self, st: &Arc<CudaStream>) -> Result<Vec<u8>, String> {
        let mut v = st
            .clone_dtoh(&*self.s)
            .map_err(|e| format!("dtoh: {e:?}"))?;
        v.truncate(self.len);
        Ok(v)
    }
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
pub fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
pub fn bytes_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
pub fn bytes_i32(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
pub fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}
/// fp32 -> bf16 bits, round to nearest even (finite inputs).
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return 0x7FC0;
    }
    let r = b.wrapping_add(0x7FFF + ((b >> 16) & 1));
    (r >> 16) as u16
}

/// float8_e4m3fn -> f32 (0x7F/0xFF are NaN).
pub fn e4m3_to_f32(c: u8) -> f32 {
    let s = if c & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((c >> 3) & 0xF) as i32;
    let m = (c & 7) as f32;
    if e == 0xF && (c & 7) == 7 {
        return f32::NAN;
    }
    if e == 0 {
        s * m / 8.0 * 2f32.powi(-6)
    } else {
        s * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}
/// float8_e8m0fnu -> f32 (255 is NaN).
pub fn e8m0_to_f32(b: u8) -> f32 {
    if b == 0xFF {
        f32::NAN
    } else {
        2f32.powi(b as i32 - 127)
    }
}

/// The kernels behind one stream, plus the RoPE tables the attention layers read.
pub struct Ops {
    pub st: Arc<CudaStream>,
    /// this stream's freed small temporaries ([`TmpPool`])
    pool: Arc<TmpPool>,
}

macro_rules! launch {
    ($name:ident ( $($a:expr),* $(,)? )) => {
        ck(stringify!($name), unsafe { ffi::$name($($a),*) })
    };
}

impl Ops {
    pub fn new(st: Arc<CudaStream>) -> Self {
        Self {
            st,
            pool: Arc::default(),
        }
    }
    /// The stream handle for a launch, with its context made current first: the launchers use the
    /// runtime API, which launches on the current device, and a rank's kernels must land on its card.
    pub fn sv(&self) -> *mut c_void {
        // a failed bind leaves the previous context current and the launch then fails on the
        // stream/device mismatch, so the error is not lost
        let _ = self.st.context().bind_to_thread();
        self.st.cu_stream() as *mut c_void
    }
    pub fn sync(&self) -> Result<(), String> {
        self.st.synchronize().map_err(|e| format!("sync: {e:?}"))
    }
    pub fn buf(&self, len: usize) -> Result<Buf, String> {
        Buf::pooled(&self.st, &self.pool, len, true)
    }
    /// A layer temporary its kernels write in full before reading (no zero-fill memset).
    pub fn tmp(&self, len: usize) -> Result<Buf, String> {
        Buf::pooled(&self.st, &self.pool, len, false)
    }

    /// act_quant(x, 32, "ue8m0") of `rows` bf16 rows of `k` (gathered through `rowidx` when it
    /// is not null) into e4m3 codes `[rows][k]` and ue8m0 scales `[rows][k/32]`.
    #[allow(clippy::too_many_arguments)]
    pub fn act_quant(
        &self,
        x: Dp,
        ldx: usize,
        rowidx: Dp,
        rows: usize,
        k: usize,
        q: Dp,
        s: Dp,
    ) -> Result<(), String> {
        launch!(memra_v41_act_quant(
            x.c(),
            ldx as i64,
            rowidx.i(),
            q.m(),
            k as i64,
            s.m(),
            (k / 32) as i64,
            std::ptr::null_mut(),
            0,
            rows as i32,
            k as i32,
            self.sv()
        ))
    }
    /// In-place fp8 quantize-dequantize (window KV, attention output before `wo_a`).
    pub fn fp8_qdq(&self, x: Dp, ld: usize, rows: usize, k: usize) -> Result<(), String> {
        launch!(memra_v41_act_quant(
            x.c(),
            ld as i64,
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            0,
            x.m(),
            ld as i64,
            rows as i32,
            k as i32,
            self.sv()
        ))
    }
    pub fn fp4_qdq(
        &self,
        x: Dp,
        ld: usize,
        rows: usize,
        n: usize,
        group: usize,
    ) -> Result<(), String> {
        launch!(memra_v41_fp4_qdq(
            x.m(),
            ld as i64,
            rows as i32,
            n as i32,
            group as i32,
            self.sv()
        ))
    }
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm(
        &self,
        x: Dp,
        ldx: usize,
        w: &Buf,
        y: Dp,
        ldy: usize,
        rows: usize,
        d: usize,
        eps: f32,
    ) -> Result<(), String> {
        launch!(memra_v41_rmsnorm(
            x.c(),
            ldx as i64,
            w.p(&self.st).c(),
            y.m(),
            ldy as i64,
            rows as i32,
            d as i32,
            eps,
            self.sv()
        ))
    }
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        x: Dp,
        ld_row: usize,
        rows: usize,
        heads: usize,
        hd: usize,
        t: &RopeTable,
        pos_base: usize,
        pos_step: usize,
        inverse: bool,
    ) -> Result<(), String> {
        if rows > 0 && pos_base + (rows - 1) * pos_step >= t.n {
            return Err(format!("rope: position beyond the table ({})", t.n));
        }
        launch!(memra_v41_rope(
            x.m(),
            ld_row as i64,
            rows as i32,
            heads as i32,
            hd as i32,
            t.cos.p(&self.st).f(),
            t.sin.p(&self.st).f(),
            pos_base as i32,
            pos_step as i32,
            std::ptr::null(),
            inverse as i32,
            self.sv()
        ))
    }
    /// [`Self::rope`] with each row's position read from a device array (`pos`: i32 per row);
    /// the caller has checked the positions against the table.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_pos(
        &self,
        x: Dp,
        ld_row: usize,
        rows: usize,
        heads: usize,
        hd: usize,
        t: &RopeTable,
        pos: Dp,
        inverse: bool,
    ) -> Result<(), String> {
        launch!(memra_v41_rope(
            x.m(),
            ld_row as i64,
            rows as i32,
            heads as i32,
            hd as i32,
            t.cos.p(&self.st).f(),
            t.sin.p(&self.st).f(),
            0,
            0,
            pos.i(),
            inverse as i32,
            self.sv()
        ))
    }
    /// `out[rows][w.n]` = linear(x, w) for an act-quantized input (`aq`, `as_`).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q(
        &self,
        aq: Dp,
        as_: Dp,
        rows: usize,
        w: &QW,
        out: Dp,
        ldc: usize,
        out_f32: bool,
    ) -> Result<(), String> {
        let (ldb, ldbs, fmt, s2) = match w.fmt {
            QFmt::Fp8Block32 => (w.k, w.k / 32, 0, 1.0),
            QFmt::Mxfp4 => (w.k / 2, w.k / 32, 1, 1.0),
            QFmt::Nvfp4Minted { scale2 } => (w.k / 2, w.k / 16, 2, scale2),
        };
        launch!(memra_v41_gemm_f8(
            aq.c(),
            w.k as i64,
            as_.c(),
            (w.k / 32) as i64,
            w.w.p(&self.st).c(),
            ldb as i64,
            w.s.p(&self.st).c(),
            ldbs as i64,
            s2,
            fmt,
            out.m(),
            ldc as i64,
            out_f32 as i32,
            rows as i32,
            w.n as i32,
            w.k as i32,
            self.sv()
        ))
    }
    /// linear(x, w) on bf16 rows: per-32 activation quant into `scr`, then the block GEMM.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_q(
        &self,
        x: Dp,
        ldx: usize,
        rows: usize,
        w: &QW,
        out: Dp,
        ldc: usize,
        out_f32: bool,
        scr: &mut Scratch,
    ) -> Result<(), String> {
        let (q, s) = scr.actq(self, rows, w.k)?;
        self.act_quant(x, ldx, Dp::NULL, rows, w.k, q, s)?;
        self.gemm_q(q, s, rows, w, out, ldc, out_f32)
    }
    /// F.linear on bf16 weights with fp32 accumulation (batched over `batch` with element strides).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_bf16(
        &self,
        a: Dp,
        lda: usize,
        s_az: usize,
        b: Dp,
        ldb: usize,
        s_bz: usize,
        c: Dp,
        ldc: usize,
        s_cz: usize,
        out_f32: bool,
        m: usize,
        n: usize,
        k: usize,
        batch: usize,
    ) -> Result<(), String> {
        launch!(memra_v41_gemm_bf16(
            a.c(),
            lda as i64,
            s_az as i64,
            b.c(),
            ldb as i64,
            s_bz as i64,
            c.m(),
            ldc as i64,
            s_cz as i64,
            out_f32 as i32,
            m as i32,
            n as i32,
            k as i32,
            batch as i32,
            self.sv()
        ))
    }
    /// F.linear on a bf16 weight. A bf16 output is the reference's bf16 GEMM (tensor cores, fp32
    /// accumulation); an fp32 output is its fp32 GEMM on the same values (CUDA-core FMA chain).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_bf16(
        &self,
        x: Dp,
        ldx: usize,
        rows: usize,
        w: &BW,
        out: Dp,
        ldc: usize,
        out_f32: bool,
    ) -> Result<(), String> {
        if out_f32 {
            return launch!(memra_v41_gemm_ffma(
                x.c(),
                ldx as i64,
                w.w.p(&self.st).c(),
                w.k as i64,
                out.fm(),
                ldc as i64,
                rows as i32,
                w.n as i32,
                w.k as i32,
                self.sv()
            ));
        }
        self.gemm_bf16(
            x,
            ldx,
            0,
            w.w.p(&self.st),
            w.k,
            0,
            out,
            ldc,
            0,
            false,
            rows,
            w.n,
            w.k,
            1,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attn(
        &self,
        q: Dp,
        ldq: usize,
        kv: Dp,
        idx: Dp,
        ldi: usize,
        topk: usize,
        sink: Dp,
        scale: f32,
        o: Dp,
        ldo: usize,
        m: usize,
        heads: usize,
    ) -> Result<(), String> {
        launch!(memra_v41_sparse_attn(
            q.c(),
            ldq as i64,
            kv.c(),
            idx.i(),
            ldi as i64,
            topk as i32,
            sink.f(),
            scale,
            o.m(),
            ldo as i64,
            m as i32,
            heads as i32,
            512,
            self.sv()
        ))
    }
    /// [`Self::sparse_attn`] with each query row's keys named by a [`key_refs`] table instead of
    /// one buffer: the window rows and the compressed rows are read where they live.
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attn_keys(
        &self,
        q: Dp,
        ldq: usize,
        keys: Dp,
        idx: Dp,
        ldi: usize,
        topk: usize,
        sink: Dp,
        scale: f32,
        o: Dp,
        ldo: usize,
        m: usize,
        heads: usize,
    ) -> Result<(), String> {
        launch!(memra_v41_sparse_attn_keys(
            q.c(),
            ldq as i64,
            keys.c(),
            idx.i(),
            ldi as i64,
            topk as i32,
            sink.f(),
            scale,
            o.m(),
            ldo as i64,
            m as i32,
            heads as i32,
            512,
            self.sv()
        ))
    }
}

/// Bytes of the zeroed scratch the one-row selects take (`V41_SN_CNT + 2 * V41_SN_BLOCKS` ints).
pub const SELECT_LONG_WS: usize = (544 + 2 * 256) * 4;

impl Ops {
    /// The top-k of one score row (`memra_v41_topk` at one row) as four launches over the row cut
    /// into blocks: a decode step's selection at long context (225 us for 131,072 keys in the
    /// one-block form). Same selection, same order.
    pub fn topk_long(
        &self,
        score: Dp,
        t: usize,
        k: usize,
        out_w: usize,
        offset: usize,
        out: Dp,
    ) -> Result<(), String> {
        let ws = self.buf(SELECT_LONG_WS)?;
        launch!(memra_v41_topk_long(
            score.c(),
            t as i32,
            k as i32,
            out_w as i32,
            offset as i32,
            out.im(),
            ws.p(&self.st).im(),
            self.sv()
        ))
    }
    /// The candidate blocks of one score row (`memra_v41_candidates` at one row, the query at
    /// `start_pos`), the same way.
    #[allow(clippy::too_many_arguments)]
    pub fn candidates_long(
        &self,
        score: Dp,
        t: usize,
        cb: usize,
        kblocks: usize,
        start_pos: usize,
        ratio: usize,
        keep: Dp,
    ) -> Result<(), String> {
        let ws = self.buf(SELECT_LONG_WS)?;
        launch!(memra_v41_candidates_long(
            score.c(),
            t as i32,
            cb as i32,
            kblocks as i32,
            start_pos as i32,
            ratio as i32,
            keep.u8m(),
            ws.p(&self.st).im(),
            self.sv()
        ))
    }
}

/// The key table of [`Ops::sparse_attn_keys`], one entry a query row: `(window rows, compressed
/// rows, window row count)`. An index below the count is a window row, any other index that many
/// rows into the compressed rows (which may be null when no index reaches them).
pub fn key_refs(rows: &[(Dp, Dp, usize)]) -> Vec<u8> {
    let mut t = Vec::with_capacity(rows.len() * 24);
    for &(win, comp, nwin) in rows {
        t.extend_from_slice(&win.addr().to_le_bytes());
        t.extend_from_slice(&comp.addr().to_le_bytes());
        t.extend_from_slice(&(nwin as i32).to_le_bytes());
        t.extend_from_slice(&0i32.to_le_bytes());
    }
    t
}

/// Weight formats of the block-quantized linears.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QFmt {
    /// e4m3 `[n][k]`, one ue8m0 scale per 32x32 block `[ceil(n/32)][k/32]`.
    Fp8Block32,
    /// e2m1 pairs `[n][k/2]` (low nibble first), one ue8m0 scale per row and 32 inputs.
    Mxfp4,
    /// The MXFP4 values re-expressed as NVFP4: e4m3 scales per 16 inputs `[n][k/16]`, each 32-block's
    /// pair equal, times `scale2`. Value-identical to the MXFP4 source by construction.
    Nvfp4Minted { scale2: f32 },
}

/// A block-quantized weight on the device.
pub struct QW {
    pub w: Buf,
    pub s: Buf,
    pub n: usize,
    pub k: usize,
    pub fmt: QFmt,
}

/// A bf16 weight `[n][k]` on the device.
pub struct BW {
    pub w: Buf,
    pub n: usize,
    pub k: usize,
}

/// Reusable activation-quant scratch (codes + scales), grown on demand.
#[derive(Default)]
pub struct Scratch {
    q: Option<Buf>,
    s: Option<Buf>,
}

impl Scratch {
    pub fn actq(&mut self, ops: &Ops, rows: usize, k: usize) -> Result<(Dp, Dp), String> {
        let nq = rows * k;
        let ns = rows * k / 32;
        if self.q.as_ref().is_none_or(|b| b.len < nq) {
            self.q = Some(ops.tmp(nq)?);
        }
        if self.s.as_ref().is_none_or(|b| b.len < ns) {
            self.s = Some(ops.tmp(ns)?);
        }
        Ok((
            self.q.as_ref().unwrap().p(&ops.st),
            self.s.as_ref().unwrap().p(&ops.st),
        ))
    }
}

/// fp32 RoPE cos/sin tables `[n][32]` for one (theta, YaRN) choice, built the way the reference
/// `precompute_freqs_cis` builds them (fp32 frequencies, YaRN ramp in fp32, fp32 angles).
pub struct RopeTable {
    pub cos: Buf,
    pub sin: Buf,
    pub n: usize,
}

/// Host side of [`RopeTable`]: `(cos, sin)` row-major `[n][dim/2]`.
pub fn rope_tables_host(
    dim: usize,
    n: usize,
    original_seq_len: usize,
    base: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let mut freqs: Vec<f32> = (0..half)
        .map(|i| 1.0f32 / base.powf((2 * i) as f32 / dim as f32))
        .collect();
    if original_seq_len > 0 {
        let corrected = |rot: f64| -> f64 {
            dim as f64 * (original_seq_len as f64 / (rot * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * (base as f64).ln())
        };
        let low = corrected(beta_fast as f64).floor().max(0.0);
        let high = corrected(beta_slow as f64).ceil().min((dim - 1) as f64);
        let span = (high - low).max(1e-3) as f32;
        for (i, f) in freqs.iter_mut().enumerate() {
            let ramp = ((i as f32 - low as f32) / span).clamp(0.0, 1.0);
            let smooth = 1.0f32 - ramp;
            *f = *f / factor * (1.0f32 - smooth) + *f * smooth;
        }
    }
    let mut c = vec![0f32; n * half];
    let mut s = vec![0f32; n * half];
    for p in 0..n {
        for i in 0..half {
            let a = p as f32 * freqs[i];
            c[p * half + i] = (a as f64).cos() as f32;
            s[p * half + i] = (a as f64).sin() as f32;
        }
    }
    (c, s)
}

impl RopeTable {
    pub fn new(
        ops: &Ops,
        n: usize,
        original_seq_len: usize,
        base: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> Result<Self, String> {
        let (c, s) = rope_tables_host(64, n, original_seq_len, base, factor, beta_fast, beta_slow);
        Ok(Self {
            cos: Buf::from_f32(&ops.st, &c)?,
            sin: Buf::from_f32(&ops.st, &s)?,
            n,
        })
    }
}

// ------------------------------------------------------------------------------------------------
// checkpoint loading

fn raw<'a>(
    stm: &'a StModel,
    name: &str,
) -> Result<(&'a memra_gguf::safetensors::StInfo, &'a [u8]), String> {
    stm.raw(name)
        .ok_or_else(|| format!("missing tensor {name}"))
}

fn want(
    info: &memra_gguf::safetensors::StInfo,
    name: &str,
    dtype: &str,
    shape: &[u64],
) -> Result<(), String> {
    if info.dtype != dtype || info.shape != shape {
        return Err(format!(
            "{name}: {} {:?}, want {dtype} {shape:?}",
            info.dtype, info.shape
        ));
    }
    Ok(())
}

/// FP8 32x32 dense weight `{p}.weight` / `{p}.scale`.
pub fn load_fp8(ops: &Ops, stm: &StModel, p: &str, n: usize, k: usize) -> Result<QW, String> {
    let (iw, bw) = raw(stm, &format!("{p}.weight"))?;
    want(iw, p, "F8_E4M3", &[n as u64, k as u64])?;
    let (is, bs) = raw(stm, &format!("{p}.scale"))?;
    want(
        is,
        p,
        "F8_E8M0",
        &[n.div_ceil(32) as u64, k.div_ceil(32) as u64],
    )?;
    Ok(QW {
        w: Buf::from_bytes(&ops.st, bw)?,
        s: Buf::from_bytes(&ops.st, bs)?,
        n,
        k,
        fmt: QFmt::Fp8Block32,
    })
}

/// Rows `rows` of the FP8 32x32 dense weight `{p}` (`[n][k]` in the checkpoint): the weight rows
/// and their scale-block rows, so `rows` starts on a 32-row block and ends on one or at `n`.
pub fn load_fp8_rows(
    ops: &Ops,
    stm: &StModel,
    p: &str,
    n: usize,
    k: usize,
    rows: std::ops::Range<usize>,
) -> Result<QW, String> {
    if rows == (0..n) {
        return load_fp8(ops, stm, p, n, k);
    }
    if !rows.start.is_multiple_of(32)
        || (!rows.end.is_multiple_of(32) && rows.end != n)
        || rows.end > n
    {
        return Err(format!("{p}: rows {rows:?} cut a 32-row scale block"));
    }
    let (iw, bw) = raw(stm, &format!("{p}.weight"))?;
    want(iw, p, "F8_E4M3", &[n as u64, k as u64])?;
    let (is, bs) = raw(stm, &format!("{p}.scale"))?;
    let kb = k.div_ceil(32);
    want(is, p, "F8_E8M0", &[n.div_ceil(32) as u64, kb as u64])?;
    Ok(QW {
        w: Buf::from_bytes(&ops.st, &bw[rows.start * k..rows.end * k])?,
        s: Buf::from_bytes(
            &ops.st,
            &bs[rows.start / 32 * kb..rows.end.div_ceil(32) * kb],
        )?,
        n: rows.len(),
        k,
        fmt: QFmt::Fp8Block32,
    })
}

/// Rows `rows` of the FP4 expert weight `{p}` ([`load_fp4`]): the weight rows and their scale rows.
/// A minted weight takes its `scale2` from the whole tensor, so a slice holds the bytes the whole
/// weight would.
pub fn load_fp4_rows(
    ops: &Ops,
    stm: &StModel,
    p: &str,
    n: usize,
    k: usize,
    rows: std::ops::Range<usize>,
    mint: bool,
) -> Result<QW, String> {
    if rows.end > n {
        return Err(format!("{p}: rows {rows:?} of {n}"));
    }
    let (iw, bw) = raw(stm, &format!("{p}.weight"))?;
    want(iw, p, "I8", &[n as u64, (k / 2) as u64])?;
    let (is, bs) = raw(stm, &format!("{p}.scale"))?;
    want(is, p, "F8_E8M0", &[n as u64, (k / 32) as u64])?;
    let w = Buf::from_bytes(&ops.st, &bw[rows.start * (k / 2)..rows.end * (k / 2)])?;
    if !mint {
        return Ok(QW {
            w,
            s: Buf::from_bytes(&ops.st, &bs[rows.start * (k / 32)..rows.end * (k / 32)])?,
            n: rows.len(),
            k,
            fmt: QFmt::Mxfp4,
        });
    }
    let (sc, scale2) = mint_mxfp4_scales(bs).map_err(|e| format!("{p}: {e}"))?;
    Ok(QW {
        w,
        s: Buf::from_bytes(&ops.st, &sc[rows.start * (k / 16)..rows.end * (k / 16)])?,
        n: rows.len(),
        k,
        fmt: QFmt::Nvfp4Minted { scale2 },
    })
}

/// FP4 expert weight `{p}.weight` (I8 pairs) / `{p}.scale` (ue8m0 per 32), optionally minted.
pub fn load_fp4(
    ops: &Ops,
    stm: &StModel,
    p: &str,
    n: usize,
    k: usize,
    mint: bool,
) -> Result<QW, String> {
    let (iw, bw) = raw(stm, &format!("{p}.weight"))?;
    want(iw, p, "I8", &[n as u64, (k / 2) as u64])?;
    let (is, bs) = raw(stm, &format!("{p}.scale"))?;
    want(is, p, "F8_E8M0", &[n as u64, (k / 32) as u64])?;
    if !mint {
        return Ok(QW {
            w: Buf::from_bytes(&ops.st, bw)?,
            s: Buf::from_bytes(&ops.st, bs)?,
            n,
            k,
            fmt: QFmt::Mxfp4,
        });
    }
    let (sc, scale2) = mint_mxfp4_scales(bs).map_err(|e| format!("{p}: {e}"))?;
    Ok(QW {
        w: Buf::from_bytes(&ops.st, bw)?,
        s: Buf::from_bytes(&ops.st, &sc)?,
        n,
        k,
        fmt: QFmt::Nvfp4Minted { scale2 },
    })
}

/// MXFP4 -> NVFP4 scale mint (the `dsv4_mint` cast, per tensor): each ue8m0 scale 2^(b-127)
/// becomes two equal e4m3 per-16 scales 2^(b-127)/scale2 with scale2 = 2^(max_b-127-8), exact when
/// the tensor's scales span at most 17 octaves (e4m3 holds 2^-9 .. 2^8 as powers of two).
pub fn mint_mxfp4_scales(sc: &[u8]) -> Result<(Vec<u8>, f32), String> {
    let mut hi = 0u8;
    for &b in sc {
        if b == 0xFF {
            return Err("NaN ue8m0 scale".into());
        }
        hi = hi.max(b);
    }
    let base = hi as i32 - 127 - 8;
    let mut out = Vec::with_capacity(sc.len() * 2);
    for &b in sc {
        let q = b as i32 - 127 - base; // exponent of the e4m3 scale
        let code = e4m3_pow2_code(q).ok_or_else(|| {
            format!(
                "scale 2^{} outside the e4m3 power-of-two range after the mint",
                b as i32 - 127
            )
        })?;
        out.push(code);
        out.push(code);
    }
    Ok((out, 2f32.powi(base)))
}

/// e4m3 code of 2^q, q in [-9, 8].
fn e4m3_pow2_code(q: i32) -> Option<u8> {
    match q {
        -9..=-7 => Some(1u8 << (q + 9)), // subnormals 2^-9, 2^-8, 2^-7: mantissa 1, 2, 4
        -6..=8 => Some(((q + 7) as u8) << 3),
        _ => None,
    }
}

pub fn load_bf16(ops: &Ops, stm: &StModel, name: &str, n: usize, k: usize) -> Result<BW, String> {
    let (i, b) = raw(stm, name)?;
    want(i, name, "BF16", &[n as u64, k as u64])?;
    Ok(BW {
        w: Buf::from_bytes(&ops.st, b)?,
        n,
        k,
    })
}
/// Rows `rows` of the bf16 weight `name` `[n][k]` (the vocab-parallel head).
pub fn load_bf16_rows(
    ops: &Ops,
    stm: &StModel,
    name: &str,
    n: usize,
    k: usize,
    rows: std::ops::Range<usize>,
) -> Result<BW, String> {
    let (i, b) = raw(stm, name)?;
    want(i, name, "BF16", &[n as u64, k as u64])?;
    if rows.end > n {
        return Err(format!("{name}: rows {rows:?} past {n}"));
    }
    Ok(BW {
        w: Buf::from_bytes(&ops.st, &b[rows.start * k * 2..rows.end * k * 2])?,
        n: rows.len(),
        k,
    })
}
pub fn load_vec_bf16(ops: &Ops, stm: &StModel, name: &str, n: usize) -> Result<Buf, String> {
    let (i, b) = raw(stm, name)?;
    if i.dtype != "BF16" || i.shape.iter().product::<u64>() != n as u64 {
        return Err(format!(
            "{name}: {} {:?}, want BF16 of {n}",
            i.dtype, i.shape
        ));
    }
    Buf::from_bytes(&ops.st, b)
}
/// Elements `r` of the F32 vector `name` of `n`.
pub fn load_vec_f32_range(
    ops: &Ops,
    stm: &StModel,
    name: &str,
    n: usize,
    r: std::ops::Range<usize>,
) -> Result<Buf, String> {
    let (i, b) = raw(stm, name)?;
    if i.dtype != "F32" || i.shape.iter().product::<u64>() != n as u64 || r.end > n {
        return Err(format!(
            "{name}: {} {:?}, want F32 of {n}",
            i.dtype, i.shape
        ));
    }
    Buf::from_bytes(&ops.st, &b[r.start * 4..r.end * 4])
}
pub fn load_vec_f32(ops: &Ops, stm: &StModel, name: &str, n: usize) -> Result<Buf, String> {
    let (i, b) = raw(stm, name)?;
    if i.dtype != "F32" || i.shape.iter().product::<u64>() != n as u64 {
        return Err(format!(
            "{name}: {} {:?}, want F32 of {n}",
            i.dtype, i.shape
        ));
    }
    Buf::from_bytes(&ops.st, b)
}

/// `wo_a`: FP8 32x32 in the checkpoint, dequantized to bf16 at load like the reference
/// (`[n_groups * o_lora_rank][heads * head_dim / n_groups]`; each product is exact in bf16).
pub fn load_wo_a(ops: &Ops, stm: &StModel, p: &str, n: usize, k: usize) -> Result<BW, String> {
    load_wo_a_rows(ops, stm, p, n, k, 0..n)
}

/// Rows `rows` of `wo_a`, dequantized like [`load_wo_a`].
pub fn load_wo_a_rows(
    ops: &Ops,
    stm: &StModel,
    p: &str,
    n: usize,
    k: usize,
    rows: std::ops::Range<usize>,
) -> Result<BW, String> {
    if rows.end > n {
        return Err(format!("{p}: rows {rows:?} past {n}"));
    }
    let (iw, bw) = raw(stm, &format!("{p}.weight"))?;
    want(iw, p, "F8_E4M3", &[n as u64, k as u64])?;
    let (is, bs) = raw(stm, &format!("{p}.scale"))?;
    want(
        is,
        p,
        "F8_E8M0",
        &[n.div_ceil(32) as u64, k.div_ceil(32) as u64],
    )?;
    let kb = k.div_ceil(32);
    let lut: Vec<f32> = (0..=255u8).map(e4m3_to_f32).collect();
    let mut out = Vec::with_capacity(rows.len() * k * 2);
    for r in rows.clone() {
        for c in 0..k {
            let v = lut[bw[r * k + c] as usize] * e8m0_to_f32(bs[(r / 32) * kb + c / 32]);
            let bits = f32_to_bf16_bits(v);
            debug_assert_eq!(
                f32::from_bits((bits as u32) << 16),
                v,
                "wo_a dequant must be exact in bf16"
            );
            out.extend_from_slice(&bits.to_le_bytes());
        }
    }
    Ok(BW {
        w: Buf::from_bytes(&ops.st, &out)?,
        n: rows.len(),
        k,
    })
}

// ------------------------------------------------------------------------------------------------
// one layer

/// V4.1 geometry the layer needs (from `DeepSeekV41Config` + the base config).
#[derive(Debug, Clone, Default)]
pub struct Geo {
    pub dim: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub q_lora: usize,
    pub o_groups: usize,
    pub o_lora: usize,
    pub window: usize,
    pub index_heads: usize,
    pub index_dim: usize,
    pub index_topk: usize,
    pub cand_blocks: usize,
    pub cand_block: usize,
    pub cand_source: usize,
    pub n_experts: usize,
    pub topk: usize,
    pub moe_inter: usize,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    pub hc: usize,
    pub sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub norm_eps: f32,
    pub compress_ratios: Vec<usize>,
    pub kv_sources: Vec<usize>,
    pub index_sources: Vec<usize>,
    pub engram_layers: Vec<usize>,
    pub engram_heads: usize,
    pub engram_head_dim: usize,
    pub engram_ngram: usize,
    pub engram_rows: Vec<usize>,
    pub engram_bucket_base: u64,
    pub engram_compressed_vocab: u64,
    pub engram_pad_id: usize,
    pub vocab: usize,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub yarn_factor: f32,
    pub yarn_orig: usize,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
}

impl Geo {
    pub fn from_config(mc: &memra_gguf::config::ModelConfig) -> Result<Self, String> {
        let c = mc.dsv41.as_ref().ok_or("not a deepseek_v41 config")?;
        let u = |v: &[u32]| v.iter().map(|&x| x as usize).collect::<Vec<_>>();
        Ok(Self {
            dim: mc.n_embd as usize,
            heads: mc.n_head as usize,
            head_dim: c.head_dim as usize,
            rope_dim: c.qk_rope_head_dim as usize,
            q_lora: c.q_lora_rank as usize,
            o_groups: c.o_groups as usize,
            o_lora: c.o_lora_rank as usize,
            window: c.sliding_window as usize,
            index_heads: c.index_n_heads as usize,
            index_dim: c.index_head_dim as usize,
            index_topk: c.index_topk as usize,
            cand_blocks: c.candidate_topk_blocks as usize,
            cand_block: c.candidate_block_size as usize,
            cand_source: c.candidate_source_layer_id as usize,
            n_experts: c.n_routed_experts as usize,
            topk: c.num_experts_per_tok as usize,
            moe_inter: c.moe_intermediate_size as usize,
            route_scale: c.routed_scaling_factor,
            swiglu_limit: c.swiglu_limit,
            hc: c.hc_mult as usize,
            sinkhorn_iters: c.hc_sinkhorn_iters as usize,
            hc_eps: c.hc_eps,
            norm_eps: mc.rms_eps,
            compress_ratios: u(&c.compress_ratios),
            kv_sources: u(&c.kv_source_layer_ids),
            index_sources: u(&c.index_source_layer_ids),
            engram_layers: u(&c.engram_layer_ids),
            engram_heads: c.engram_n_heads as usize,
            engram_head_dim: c.engram_head_dim as usize,
            engram_ngram: c.engram_max_ngram_size as usize,
            engram_rows: u(&c.engram_num_embeddings),
            engram_bucket_base: c.engram_vocab_size as u64,
            engram_compressed_vocab: c.engram_compressed_vocab_size as u64,
            engram_pad_id: c.engram_pad_token_id as usize,
            vocab: mc.n_vocab as usize,
            rope_theta: mc.rope_freq_base,
            compress_rope_theta: c.compress_rope_theta,
            yarn_factor: c.rope_yarn_factor,
            yarn_orig: c.rope_yarn_orig_ctx as usize,
            yarn_beta_fast: c.rope_yarn_beta_fast,
            yarn_beta_slow: c.rope_yarn_beta_slow,
        })
    }
    pub fn engram_cols(&self) -> usize {
        (self.engram_ngram - 1) * self.engram_heads
    }
}

pub struct CompW {
    pub wkv: BW,
    pub wgate: Option<BW>,
    pub norm: Buf,
}

pub struct IdxW {
    pub wq_b: QW,
    pub wproj: BW,
    pub wk: Option<BW>,
    pub k_norm: Option<Buf>,
}

pub struct EngramW {
    pub wkv: QW,
    pub qw: Buf,
    pub kw: Buf,
    /// table tensor names (`.weight` fp8 rows, `.scale` ue8m0) read from the host mmap
    pub table: String,
    pub rows: usize,
}

/// One rank's share of a layer under N-rank tensor and expert parallelism (`n` = 1: the whole layer
/// on one card). Attention heads, output groups, `wo_b` output rows, routed experts and head rows
/// split into `n` equal contiguous ranges; everything else is replicated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Part {
    pub rank: usize,
    pub n: usize,
}

impl Part {
    pub const WHOLE: Part = Part { rank: 0, n: 1 };
    /// This rank's range of `total` items (`total` divisible by `n`, checked by [`Part::check`]).
    pub fn range(&self, total: usize) -> std::ops::Range<usize> {
        let p = total / self.n;
        self.rank * p..(self.rank + 1) * p
    }
    /// The partition laws of the V4.1 layer at `n` ranks: whole heads per rank in sparse-attention
    /// tiles of 16, whole output groups, `wo_b` rows in 32-row scale blocks, whole experts and an
    /// even vocabulary. A degree that breaks one is refused, never forced.
    pub fn check(g: &Geo, n: usize) -> Result<(), String> {
        let fail = |what: String| Err(format!("{n} ranks: {what}"));
        if n == 0 {
            return fail("no ranks".into());
        }
        if !g.heads.is_multiple_of(n) || !(g.heads / n).is_multiple_of(16) {
            return fail(format!(
                "{} attention heads do not split into whole 16-head tiles",
                g.heads
            ));
        }
        if !g.o_groups.is_multiple_of(n) || !g.heads.is_multiple_of(g.o_groups) {
            return fail(format!("{} output groups do not split evenly", g.o_groups));
        }
        if !g.dim.is_multiple_of(n) || !(g.dim / n).is_multiple_of(32) {
            return fail(format!(
                "wo_b rows {} do not split into 32-row scale blocks",
                g.dim
            ));
        }
        if !g.moe_inter.is_multiple_of(n) || !(g.moe_inter / n).is_multiple_of(32) {
            return fail(format!(
                "the experts' {} hidden rows do not split into whole 32-row scale blocks",
                g.moe_inter
            ));
        }
        if !g.n_experts.is_multiple_of(n) {
            return fail(format!("{} experts do not split evenly", g.n_experts));
        }
        if !g.vocab.is_multiple_of(n) {
            return fail(format!("vocabulary {} does not split evenly", g.vocab));
        }
        Ok(())
    }
}

/// One V4.1 trunk layer's weights on the device (one rank's share under [`Part`]).
pub struct LayerW {
    pub id: usize,
    /// this load of the layer, unique in the process: what a captured span of the layer is
    /// keyed by ([`graph`]), since two loads can share a number (a draft stage and a trunk
    /// layer, a layer loaded again) and hold other weights at other addresses
    pub uid: u64,
    pub part: Part,
    /// this card's attention heads `[head0, head0 + heads_l)`
    pub head0: usize,
    pub heads_l: usize,
    /// this card's output groups `[group0, group0 + groups_l)`
    pub group0: usize,
    pub groups_l: usize,
    /// this card's `wo_b` output rows start (`wo_b.n` of them)
    pub dim0: usize,
    /// the routed and shared experts are split by rows across the ranks (N ranks, experts on the
    /// card): this rank holds hidden rows `[hid0, hid0 + hid_l)` of every w1 and w3 and output rows
    /// `[dim0, dim0 + wo_b.n)` of every w2
    pub split: bool,
    pub hid0: usize,
    pub hid_l: usize,
    /// `experts[i]` is routed expert `expert0 + i`
    pub expert0: usize,
    pub ratio: usize,
    pub kv_source: bool,
    pub index_source: bool,
    pub cand_source: bool,
    pub uses_cand: bool,
    pub hc_attn_fn: Buf,
    pub hc_attn_base: Buf,
    pub hc_attn_scale: Buf,
    pub hc_ffn_fn: Buf,
    pub hc_ffn_base: Buf,
    pub hc_ffn_scale: Buf,
    pub attn_norm: Buf,
    pub ffn_norm: Buf,
    pub wq_a: QW,
    pub q_norm: Buf,
    pub wq_b: QW,
    pub wkv: QW,
    pub kv_norm: Buf,
    pub wo_a: BW,
    pub wo_b: QW,
    pub sink: Buf,
    pub comp: Option<CompW>,
    pub idx: Option<IdxW>,
    pub gate_w: BW,
    pub gate_b: Buf,
    pub experts: Vec<[QW; 3]>,
    /// the routed experts' weight table for the device-planned dispatch (`V41ExpW` per expert:
    /// w1, w2, w3 weight and scale pointers and minted scale2)
    pub exp_tab: Buf,
    pub shared: [QW; 3],
    pub engram: Option<EngramW>,
}

impl LayerW {
    pub fn load(ops: &Ops, stm: &StModel, g: &Geo, id: usize, mint: bool) -> Result<Self, String> {
        Self::load_part(ops, stm, g, id, mint, Part::WHOLE)
    }

    /// This rank's share of layer `id` (see [`Part`]).
    pub fn load_part(
        ops: &Ops,
        stm: &StModel,
        g: &Geo,
        id: usize,
        mint: bool,
        part: Part,
    ) -> Result<Self, String> {
        Self::load_part_with(ops, stm, g, id, mint, part, true)
    }

    /// [`Self::load_part`], with this rank's routed experts on the card or (`routed` false) left
    /// to an [`experts::ExpertCache`]: `experts` is then empty and the MoE resolves each slice's
    /// experts through the rank's cache.
    #[allow(clippy::too_many_arguments)]
    pub fn load_part_with(
        ops: &Ops,
        stm: &StModel,
        g: &Geo,
        id: usize,
        mint: bool,
        part: Part,
        routed: bool,
    ) -> Result<Self, String> {
        Self::load_part_named(ops, stm, g, &format!("layers.{id}"), id, mint, part, routed)
    }

    /// [`Self::load_part_with`] for the block stored under `prefix`: a trunk layer is
    /// `layers.<id>`; a draft stage is `mtp.<stage>` at layer index `trunk layers + stage`
    /// (the config's compress ratios list the draft stages after the trunk), with `g` carrying
    /// the draft MoE's expert count and top-k.
    #[allow(clippy::too_many_arguments)]
    pub fn load_part_named(
        ops: &Ops,
        stm: &StModel,
        g: &Geo,
        prefix: &str,
        id: usize,
        mint: bool,
        part: Part,
        routed: bool,
    ) -> Result<Self, String> {
        Part::check(g, part.n)?;
        let heads = part.range(g.heads);
        let groups = part.range(g.o_groups);
        let dims = part.range(g.dim);
        let experts = part.range(g.n_experts);
        let p = prefix.to_string();
        let ratio = g.compress_ratios[id];
        let kv_source = g.kv_sources.contains(&id);
        let index_source = g.index_sources.contains(&id);
        let hcd = g.hc * g.dim;
        let nmix = (2 + g.hc) * g.hc;
        let comp = if kv_source {
            Some(CompW {
                wkv: load_bf16(
                    ops,
                    stm,
                    &format!("{p}.attn.compressor.wkv.weight"),
                    g.head_dim,
                    g.dim,
                )?,
                wgate: if ratio > 1 {
                    Some(load_bf16(
                        ops,
                        stm,
                        &format!("{p}.attn.compressor.wgate.weight"),
                        g.head_dim,
                        g.dim,
                    )?)
                } else {
                    None
                },
                norm: load_vec_bf16(
                    ops,
                    stm,
                    &format!("{p}.attn.compressor.norm.weight"),
                    g.head_dim,
                )?,
            })
        } else {
            None
        };
        let idx = if index_source {
            Some(IdxW {
                wq_b: load_fp8(
                    ops,
                    stm,
                    &format!("{p}.attn.indexer.wq_b"),
                    g.index_heads * g.index_dim,
                    g.q_lora,
                )?,
                wproj: load_bf16(
                    ops,
                    stm,
                    &format!("{p}.attn.indexer.weights_proj.weight"),
                    g.index_heads,
                    g.dim,
                )?,
                wk: if kv_source {
                    Some(load_bf16(
                        ops,
                        stm,
                        &format!("{p}.attn.indexer.wk.weight"),
                        g.index_dim,
                        g.head_dim,
                    )?)
                } else {
                    None
                },
                k_norm: if kv_source {
                    Some(load_vec_bf16(
                        ops,
                        stm,
                        &format!("{p}.attn.indexer.k_norm.weight"),
                        g.index_dim,
                    )?)
                } else {
                    None
                },
            })
        } else {
            None
        };
        // N ranks with the experts on the card: every expert is split by rows like a TP MLP (w1
        // and w3 by hidden rows, w2 by output rows), the shared expert with them; a rank then reads
        // 1/N of every routed expert whatever the routing, and combines its own output columns
        let split = part.n > 1 && routed;
        let hid = part.range(g.moe_inter);
        let (inter, dim) = (g.moe_inter, g.dim);
        let expert0 = if split { 0 } else { experts.start };
        let erange = if split { 0..g.n_experts } else { experts };
        let mut experts_w = Vec::with_capacity(erange.len());
        for e in erange.filter(|_| routed) {
            let ep = format!("{p}.ffn.experts.{e}");
            experts_w.push(if split {
                [
                    load_fp4_rows(ops, stm, &format!("{ep}.w1"), inter, dim, hid.clone(), mint)?,
                    load_fp4_rows(
                        ops,
                        stm,
                        &format!("{ep}.w2"),
                        dim,
                        inter,
                        dims.clone(),
                        mint,
                    )?,
                    load_fp4_rows(ops, stm, &format!("{ep}.w3"), inter, dim, hid.clone(), mint)?,
                ]
            } else {
                [
                    load_fp4(ops, stm, &format!("{ep}.w1"), inter, dim, mint)?,
                    load_fp4(ops, stm, &format!("{ep}.w2"), dim, inter, mint)?,
                    load_fp4(ops, stm, &format!("{ep}.w3"), inter, dim, mint)?,
                ]
            });
        }
        let exp_tab = Buf::from_bytes(&ops.st, &expert_table(ops, &experts_w))?;
        let sp = format!("{p}.ffn.shared_experts");
        let shared = if split {
            [
                load_fp8_rows(ops, stm, &format!("{sp}.w1"), inter, dim, hid.clone())?,
                load_fp8_rows(ops, stm, &format!("{sp}.w2"), dim, inter, dims.clone())?,
                load_fp8_rows(ops, stm, &format!("{sp}.w3"), inter, dim, hid.clone())?,
            ]
        } else {
            [
                load_fp8(ops, stm, &format!("{sp}.w1"), inter, dim)?,
                load_fp8(ops, stm, &format!("{sp}.w2"), dim, inter)?,
                load_fp8(ops, stm, &format!("{sp}.w3"), inter, dim)?,
            ]
        };
        let engram = if let Some(ei) = g.engram_layers.iter().position(|&l| l == id) {
            let cols = g.engram_cols();
            let table = format!("{p}.engram.embed");
            let rows = stm
                .info(&format!("{table}.weight"))
                .ok_or("engram table missing")?
                .shape[0] as usize;
            if rows != g.engram_rows[ei] {
                return Err(format!(
                    "{table}: {rows} rows, config says {}",
                    g.engram_rows[ei]
                ));
            }
            engram::populate(stm, &table);
            Some(EngramW {
                wkv: load_fp8(
                    ops,
                    stm,
                    &format!("{p}.engram.wkv"),
                    g.dim * (g.hc + 1),
                    cols * g.engram_head_dim,
                )?,
                qw: load_vec_bf16(ops, stm, &format!("{p}.engram.q_weight"), g.hc * g.dim)?,
                kw: load_vec_bf16(ops, stm, &format!("{p}.engram.k_weight"), g.hc * g.dim)?,
                table,
                rows,
            })
        } else {
            None
        };
        let hd = g.head_dim;
        let gin = g.heads * hd / g.o_groups;
        static LOADS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        Ok(Self {
            id,
            uid: LOADS.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            part,
            head0: heads.start,
            heads_l: heads.len(),
            group0: groups.start,
            groups_l: groups.len(),
            dim0: dims.start,
            split,
            hid0: hid.start,
            hid_l: hid.len(),
            expert0,
            ratio,
            kv_source,
            index_source,
            cand_source: id == g.cand_source,
            uses_cand: g.cand_source < id && ratio > 0,
            hc_attn_fn: load_vec_f32(ops, stm, &format!("{p}.hc_attn_fn"), nmix * hcd)?,
            hc_attn_base: load_vec_f32(ops, stm, &format!("{p}.hc_attn_base"), nmix)?,
            hc_attn_scale: load_vec_f32(ops, stm, &format!("{p}.hc_attn_scale"), 3)?,
            hc_ffn_fn: load_vec_f32(ops, stm, &format!("{p}.hc_ffn_fn"), nmix * hcd)?,
            hc_ffn_base: load_vec_f32(ops, stm, &format!("{p}.hc_ffn_base"), nmix)?,
            hc_ffn_scale: load_vec_f32(ops, stm, &format!("{p}.hc_ffn_scale"), 3)?,
            attn_norm: load_vec_bf16(ops, stm, &format!("{p}.attn_norm.weight"), g.dim)?,
            ffn_norm: load_vec_bf16(ops, stm, &format!("{p}.ffn_norm.weight"), g.dim)?,
            wq_a: load_fp8(ops, stm, &format!("{p}.attn.wq_a"), g.q_lora, g.dim)?,
            q_norm: load_vec_bf16(ops, stm, &format!("{p}.attn.q_norm.weight"), g.q_lora)?,
            wq_b: load_fp8_rows(
                ops,
                stm,
                &format!("{p}.attn.wq_b"),
                g.heads * hd,
                g.q_lora,
                heads.start * hd..heads.end * hd,
            )?,
            wkv: load_fp8(ops, stm, &format!("{p}.attn.wkv"), g.head_dim, g.dim)?,
            kv_norm: load_vec_bf16(ops, stm, &format!("{p}.attn.kv_norm.weight"), g.head_dim)?,
            wo_a: load_wo_a_rows(
                ops,
                stm,
                &format!("{p}.attn.wo_a"),
                g.o_groups * g.o_lora,
                gin,
                groups.start * g.o_lora..groups.end * g.o_lora,
            )?,
            wo_b: load_fp8_rows(
                ops,
                stm,
                &format!("{p}.attn.wo_b"),
                g.dim,
                g.o_groups * g.o_lora,
                dims,
            )?,
            sink: load_vec_f32_range(ops, stm, &format!("{p}.attn.attn_sink"), g.heads, heads)?,
            comp,
            idx,
            gate_w: load_bf16(
                ops,
                stm,
                &format!("{p}.ffn.gate.weight"),
                g.n_experts,
                g.dim,
            )?,
            gate_b: load_vec_f32(ops, stm, &format!("{p}.ffn.gate.bias"), g.n_experts)?,
            experts: experts_w,
            exp_tab,
            shared,
            engram,
        })
    }
}

/// The `V41ExpW` records (64 bytes each, `cu/dsv41_gpu.cu`) of `experts`.
fn expert_table(ops: &Ops, experts: &[[QW; 3]]) -> Vec<u8> {
    let mut t = Vec::with_capacity(experts.len() * 64);
    for ws in experts {
        for w in ws {
            t.extend_from_slice(&w.w.p(&ops.st).addr().to_le_bytes());
        }
        for w in ws {
            t.extend_from_slice(&w.s.p(&ops.st).addr().to_le_bytes());
        }
        for w in ws {
            let s2 = match w.fmt {
                QFmt::Nvfp4Minted { scale2 } => scale2,
                _ => 1.0f32,
            };
            t.extend_from_slice(&s2.to_le_bytes());
        }
        t.extend_from_slice(&0i32.to_le_bytes());
    }
    t
}

/// Rows of a GEMM that the small-M kernels take (`V41_SM_MAX` in `cu/dsv41_gpu.cu`).
pub const SMALL_M: usize = 8;
/// The most rows of a call that keep the small-row bodies, issued over groups of rows
/// (`v41_wide` in cu/dsv41_gpu.cu): the expert tiles stay 8-row tiles (an expert with more
/// rows gets several) and the mHC mixes stay on each rank whole. The two limits the walk
/// itself applies are held here; [`set_wide_rows`] moves all five.
pub static WIDE_TILES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(48);
pub static WIDE_MIX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(32);

/// The limits by family (block-scaled dense, bf16, fp32 FMA, the one-pass mHC kernel, the
/// expert tiles) for a group of `ranks`, from two sweeps of calls of 8 to 128 rows with one
/// family's limit raised at a time against all at 0. One rank on one card: the dense groups
/// pay to 12 rows and are level at 16, bf16 to 16, fp32 FMA to 48, mHC to 32, 8-row tiles to
/// 48. Two ranks on two cards, where a rank's dense launches are half as wide: dense to 24
/// (level at 32), bf16 to 16, fp32 FMA to 64, mHC to 48, tiles to 64. More ranks take the
/// two-rank limits until they are measured.
pub fn wide_rows_for(ranks: usize) -> [usize; 5] {
    if ranks <= 1 {
        [16, 16, 48, 32, 48]
    } else {
        [24, 16, 64, 48, 64]
    }
}

/// The limits of a group of `ranks` ([`wide_rows_for`]), set when the group is made.
pub fn set_wide_rows_default(ranks: usize) {
    let [a, b, c, d, e] = wide_rows_for(ranks);
    set_wide_rows(a, b, c, d, e);
}

/// A bench's or a gate's limits of the small-row bodies, by family: block-scaled dense, bf16,
/// fp32 FMA, the one-pass mHC kernel, the expert tiles. Bytes do not depend on them.
pub fn set_wide_rows(f8: usize, bf16: usize, ffma: usize, mix: usize, tiles: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    for (family, rows) in [f8, bf16, ffma, mix].into_iter().enumerate() {
        // SAFETY: a plain store of a launcher's dispatch limit
        unsafe { ffi::memra_v41_wide_rows(family as i32, rows as i32) };
    }
    WIDE_MIX.store(mix, Relaxed);
    WIDE_TILES.store(tiles, Relaxed);
}

/// The MoE dispatch plan of one slice on the device (`memra_v41_moe_plan`): the order of the host
/// [`tp::MoePlan`] (pairs sorted by expert, tokens ascending inside one; each token's pairs in
/// ascending expert id) and the tile list of one rank's experts for the grouped GEMMs, built from
/// the router's selection without a host round trip.
pub struct MoePlanDev {
    pub rows: usize,
    pub np: usize,
    /// tile rows of the grouped GEMMs (`SMALL_M`: the small-M kernel, one tile per expert)
    pub bm: usize,
    pub max_tiles: usize,
    /// i32 `[ne + 1]`: each expert's first pair
    pub off: Buf,
    /// i32 / f32 `[np]`: each pair's token (row in the slice) and routing weight
    pub ptok: Buf,
    pub pw: Buf,
    /// i32 `[rows][topk]`: each token's pair indices in ascending expert id
    pub order: Buf,
    pub tiles: Buf,
    pub ntiles: Buf,
    /// i32: expert ids outside the router's range (counted and skipped)
    pub err: Buf,
}

/// The CSA2 state one layer hands down the stack (vendor `SharedAttentionRuntime`): the group's
/// compressed KV and index keys from its kv source, the selection from the last index source,
/// the candidate blocks from the candidate source.
#[derive(Default)]
pub struct Shared {
    pub compress_kv: Option<Buf>, // bf16 [n_c][512], RoPE'd and fp4 g16 quant-dequantized
    pub n_c: usize,
    pub index_k: Option<Buf>, // bf16 [n_c][128], RoPE'd and fp4 g32 quant-dequantized
    pub topk: Option<Buf>,    // i32 [s][topk_w], positions already offset by the window length
    pub topk_w: usize,
    pub cand: Option<Buf>, // u8 [rows][n_blocks]: query rows from `cand_row0`
    pub cand_ld: usize,
    /// the first query row `cand` holds: 0 when it holds every row; under an N-rank prefill
    /// with the indexer's query rows split, each rank keeps only its own rows (its later index
    /// layers read no other), from the start of its range
    pub cand_row0: usize,
    /// row capacity of `compress_kv` / `index_k` (decode appends rows)
    pub compress_cap: usize,
    pub index_k_cap: usize,
}

impl Shared {
    /// Exchange the half a kv-source group owns (its compressed KV, index keys, row count and
    /// capacities) with `store`; the per-forward half (selection, candidate pool) stays. The vendor
    /// keeps a compressed KV cache per kv source and an index-key cache per key-owning indexer and
    /// points the shared slots at the current group's; a decode step reads every group's cache, so
    /// each group's half lives in its own store between calls.
    pub fn swap_group(&mut self, store: &mut Shared) {
        std::mem::swap(&mut self.compress_kv, &mut store.compress_kv);
        std::mem::swap(&mut self.n_c, &mut store.n_c);
        std::mem::swap(&mut self.index_k, &mut store.index_k);
        std::mem::swap(&mut self.compress_cap, &mut store.compress_cap);
        std::mem::swap(&mut self.index_k_cap, &mut store.index_k_cap);
    }
}

/// One layer's own decode state: the sliding-window ring (bf16 `[window][head_dim]`, position p at
/// slot p % window) and, on a ratio-2 kv source, the open pair's fp32 kv/gate rows.
///
/// A forward of several decode tokens (a drafted block's verification) writes neither: its
/// window KV rows and its compressor rows wait in `pend_kv` / `pend_c` until
/// [`Ctx::commit_rows`] copies the kept rows in, so a rejected row never reaches the state.
#[derive(Default)]
pub struct LayerState {
    pub ring: Option<Buf>,
    pub c_kv: Option<Buf>,
    pub c_sc: Option<Buf>,
    /// bf16 `[s][head_dim]`: the uncommitted forward's window KV rows
    pub pend_kv: Option<Buf>,
    /// fp32 `[s][head_dim]` kv and gate rows of the uncommitted forward (ratio-2 kv source)
    pub pend_c: Option<(Buf, Buf)>,
}

/// Grow `buf` (rows of `row_bytes`) to hold `rows`, keeping its first `cap` rows.
fn ensure_rows(
    ops: &Ops,
    buf: &mut Option<Buf>,
    cap: &mut usize,
    rows: usize,
    row_bytes: usize,
    hint: usize,
) -> Result<(), String> {
    if buf.is_some() && rows <= *cap {
        return Ok(());
    }
    let new_cap = rows.max(hint).max(*cap * 2);
    let nb = ops.buf(new_cap * row_bytes)?;
    if let Some(old) = buf.as_ref() {
        let keep = (*cap).min(old.len / row_bytes) * row_bytes;
        copy_d2d(ops, old, 0, &nb, 0, keep)?;
    }
    *buf = Some(nb);
    *cap = new_cap;
    Ok(())
}

/// What one prefill layer produced besides its output, for gates.
#[derive(Default)]
pub struct Trace {
    pub attn_mix: Option<Buf>, // f32 [s][24]
    pub attn_in: Option<Buf>,  // bf16 [s][dim], after attn_norm
    pub qr: Option<Buf>,
    pub latent: Option<Buf>, // pre-RoPE compressed latent
    pub idx_all: Option<Buf>,
    pub kv_all: Option<Buf>,
    pub q: Option<Buf>,
    pub attn_o: Option<Buf>,
    pub attn_out: Option<Buf>,
    pub mid: Option<Buf>,
    pub moe_in: Option<Buf>,
    pub route_sel: Vec<i32>,
    pub route_w: Vec<f32>,
    pub moe_out: Option<Buf>,
    pub engram_out: Option<Buf>,
}

/// What the keys side of one attention call hands its query side: `qr`, the KV rows the heads
/// read (`[window part | compressed part]`) and the selection width; the selection itself stays in
/// the [`Shared`] state the keys side wrote.
pub struct AttnKeys {
    pub qr: Buf,
    pub kv_all: Buf,
    pub ww: usize,
    pub tw: usize,
    pub tot: usize,
    pub win: usize,
    pub comp: bool,
    /// several decode tokens in one forward: the window part is `[ring | the forward's rows]`
    pub dec_rows: bool,
    /// the index rows `[rows][tot]` given outright (a draft block: every row reads the same
    /// ring slots and block rows; a batch of sequences: each row its own region); None builds
    /// them from the position
    pub idx: Option<Buf>,
    /// each row's position as a device i32 array (a batch of sequences); None: consecutive
    /// positions from the call's first
    pub pos: Option<Dp>,
    /// the [`key_refs`] table of a decode call, one entry a row: the heads then read the
    /// window rows and the compressed rows where they live ([`Ops::sparse_attn_keys`]) and
    /// `kv_all` holds no copy of them; None: `kv_all` is `[window rows | compressed rows]`
    pub keys: Option<Buf>,
    /// the same table when it lives in the rank's step block (a batched step) instead of a
    /// buffer of its own
    pub keys_at: Option<Dp>,
}

/// One row of a decode call over several sequences: a sequence's one-token step at its own
/// position, with that sequence's layer state and its CSA2 state (the kv-source group's
/// compressed KV and index keys swapped in, and the call's own candidate pool).
pub struct BatchRow<'a> {
    pub pos: usize,
    /// positions the sequence's compressed KV and index keys are first sized for
    pub cap: usize,
    pub ls: &'a mut LayerState,
    pub sh: &'a mut Shared,
}

/// What a batched decode call hands down the stack on one rank beside each row's [`Shared`]:
/// the selection rows of the last index source, each already offset to its row's region of the
/// KV rows (the regions are the same on every layer of one compress ratio).
#[derive(Default)]
pub struct BatchSel {
    /// i32 `[rows][tw]`
    pub topk: Option<Buf>,
    pub tw: usize,
    /// the rows that hold a selection (a sequence with no compressed row yet holds none)
    pub has: Vec<bool>,
}

/// What a layer is to a decode step, read off the geometry (no weights): its compress ratio and
/// whether it owns a compressed KV (and its index keys) or a selection.
#[derive(Clone, Copy, Debug)]
pub struct LayerKind {
    pub ratio: usize,
    pub kv_source: bool,
    pub index_source: bool,
}

impl LayerKind {
    pub fn of(g: &Geo, id: usize) -> Self {
        Self {
            ratio: g.compress_ratios[id],
            kv_source: g.kv_sources.contains(&id),
            index_source: g.index_sources.contains(&id),
        }
    }
}

/// Bytes of one layer's tables in a step block for `n` rows (the same for every layer kind, so
/// the block's layout is a function of the row count and the layer count alone). In order, a
/// row each: the heads' key reference (24 bytes, [`key_refs`]); then 8-byte device addresses:
/// the ring slot the row's window KV lands on, the open pair's slot for its fp32 kv row and for
/// its gate row, the pair's two buffers (the pooling reads both slots), the index key and the
/// compressed row a completed latent is appended at (null: the row completes none this step);
/// then i32: the position of the latent's group, the row's region base (0) and how many
/// selection entries it holds.
pub const fn step_layer_bytes(n: usize) -> usize {
    (n * 92).next_multiple_of(8)
}

/// The selection width a batched step runs for rows that hold at most `most` selected entries:
/// 0 for none, else the next of 64, 128, 256, ... not past `topk` (the model's top-k).
pub fn select_class(most: usize, topk: usize) -> usize {
    if most == 0 {
        0
    } else {
        most.next_power_of_two().max(64).min(topk)
    }
}

/// One layer of a batched step as the host decided it before any launch: where its tables sit
/// in the step block, and the selection width and holders the layer's attention reads.
#[derive(Clone, Debug, Default)]
pub struct StepLayer {
    pub off: usize,
    pub tw: usize,
    pub has: Vec<bool>,
}

/// A layer's view of the rank's step block on the device.
#[derive(Clone, Copy)]
pub struct StepRef<'a> {
    pub blk: Dp,
    pub sl: &'a StepLayer,
}

/// One batched step's host data on a rank: the rows' positions, pre-mix and embedding rows, the
/// Engram rows of every Engram layer and every layer's tables ([`step_layer_bytes`]), written
/// into page-locked memory and sent to the card in one copy before the step's first launch. The
/// device buffer stays where it is from step to step, so a launch names its inputs by a fixed
/// address inside it.
pub struct StepBlock {
    host: cudarc::driver::PinnedHostSlice<u8>,
    pub dev: Buf,
    cap: usize,
    /// recorded behind the upload; the next step waits for it before it rewrites `host`
    done: cudarc::driver::CudaEvent,
}

impl StepBlock {
    pub fn new(ops: &Ops, cap: usize) -> Result<Self, String> {
        let ctx = ops.st.context().clone();
        let cap = cap.next_multiple_of(4096);
        // SAFETY: plain bytes, every byte a step reads is written by that step's `upload`
        let host = unsafe { ctx.alloc_pinned::<u8>(cap) }.map_err(|e| format!("pinned: {e:?}"))?;
        Ok(Self {
            host,
            dev: ops.buf(cap)?,
            cap,
            done: ctx.new_event(None).map_err(|e| format!("event: {e:?}"))?,
        })
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Send `bytes` to the card (the start of the block), stream-ordered before what follows.
    pub fn upload(&mut self, ops: &Ops, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() > self.cap {
            return Err(format!(
                "step block: {} bytes, capacity {}",
                bytes.len(),
                self.cap
            ));
        }
        self.done
            .synchronize()
            .map_err(|e| format!("step block: {e:?}"))?;
        let host = self
            .host
            .as_mut_slice()
            .map_err(|e| format!("pinned: {e:?}"))?;
        host[..bytes.len()].copy_from_slice(bytes);
        // SAFETY: the source is pinned memory this block owns, rewritten only after `done`
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                self.dev.p(&ops.st).addr(),
                &host[..bytes.len()],
                ops.sv() as cudarc::driver::sys::CUstream,
            )
        }
        .map_err(|e| format!("step block h2d: {e:?}"))?;
        self.done
            .record(&ops.st)
            .map_err(|e| format!("step block: {e:?}"))?;
        Ok(())
    }
}

/// Runtime pieces shared by every layer on the card.
pub struct Ctx {
    pub ops: Ops,
    pub g: Geo,
    pub rope_base: RopeTable,
    pub rope_comp: RopeTable,
    pub scr: Scratch,
    pub stage: Option<engram::Stage>,
    pub trace: bool,
    /// positions the RoPE tables cover
    pub max_pos: usize,
    /// positions a sequence's compressed KV and index keys are first sized for (the running
    /// sequence's capacity, at most `max_pos`); a sequence that outgrows it doubles them
    pub rows_hint: usize,
    /// prefill tokens per query-side slice (attention, indexer selection, MoE, Engram); every
    /// sliced step acts per row, so the slice size never changes a result
    pub chunk: usize,
    /// the query rows this card's indexer selects (None: all of them). An N-rank prefill gives
    /// each rank a contiguous range and gathers the selection rows (`dsv41_gpu::tp`); the
    /// selection is per row, so the split never changes a result
    pub index_rows: Option<std::ops::Range<usize>>,
    /// NVTX ranges around each op (set by `dsv41-bench --nvtx` for an nsys profile; off, the
    /// range calls are skipped)
    pub nvtx: bool,
    /// this rank's routed experts when the layers hold none ([`LayerW::load_part_with`])
    pub experts: Option<experts::ExpertCache>,
    /// i32, sticky, read with the step's logits. Bit 0: an N-rank one-token MoE whose plan placed
    /// other than `rows * topk` pairs (`memra_v41_moe_owner`). Bit 16: a small gather whose
    /// peer never arrived (`memra_v41_pull`).
    pub route_bad: Buf,
    /// the card's second stream, made at the first one-token MoE ([`Side`])
    side: Option<Side>,
    /// a decode call's heads read the ring and the compressed KV where they live (a key table
    /// per row) instead of a copy of both made for the call; off: the copies (the reference arm
    /// of `dsv41-spec --copy-keys`; a traced call copies too, the gates read the copy)
    pub keyed: bool,
    /// A one-row selection over more keys than this takes the long-row forms
    /// ([`Ops::topk_long`], [`Ops::candidates_long`]: the row cut into blocks, the same
    /// selection): `(top-k, candidates)`. The top-k form is on from where it starts to win
    /// (32,768 keys). The candidate form wins past 131,072 keys but is off here (never) until
    /// its gate has run on a trunk that holds the candidate source and its readers (layers 20
    /// and 24 to 36). `dsv41-spec --select-long-min` sets both for the gates.
    pub select_long_min: (usize, usize),
    /// this rank's step block ([`StepBlock`]), made at the first batched step
    pub step: Option<StepBlock>,
}

/// A second stream on the card for a branch of a one-token layer that depends on nothing the main
/// stream is about to compute. At one row most of a layer's kernels are a few blocks waiting on a
/// dependency chain (the router's fp32 chain, the plan) and leave the card empty; a kernel that
/// fills it (the shared expert's three GEMMs) costs nothing run under them. The branch has its own
/// free list and scratch, starts behind an event the main stream records where the branch's inputs
/// are complete, and the main stream waits for the branch's last kernel before it reads the result.
pub struct Side {
    pub ops: Ops,
    pub scr: Scratch,
    fork: CudaEvent,
    join: CudaEvent,
}

impl Side {
    fn new(main: &Arc<CudaStream>) -> Result<Self, String> {
        let cx = main.context();
        let st = cx.new_stream().map_err(|e| format!("side stream: {e:?}"))?;
        Ok(Self {
            ops: Ops::new(st),
            scr: Scratch::default(),
            fork: cx.new_event(None).map_err(|e| format!("event: {e:?}"))?,
            join: cx.new_event(None).map_err(|e| format!("event: {e:?}"))?,
        })
    }
    /// The branch starts after everything issued on `main` so far.
    fn fork(&self, main: &Arc<CudaStream>) -> Result<(), String> {
        self.fork
            .record(main)
            .and_then(|()| self.ops.st.wait(&self.fork))
            .map_err(|e| format!("side fork: {e:?}"))
    }
    /// What `main` runs next starts after everything issued on the branch so far.
    fn join(&self, main: &Arc<CudaStream>) -> Result<(), String> {
        self.join
            .record(&self.ops.st)
            .and_then(|()| main.wait(&self.join))
            .map_err(|e| format!("side join: {e:?}"))
    }
}

/// The most pairs the plan builds in one kernel (`V41_PLAN_SMALL` in cu/dsv41_gpu.cu).
const PLAN_SMALL_PAIRS: usize = 256;

/// The most rows of a forward whose shared expert runs on the side stream: the small-M kernels'
/// rows. With more rows every GEMM fills the card and there is nothing to run under.
const SIDE_ROWS: usize = SMALL_M;

impl Ctx {
    pub fn new(st: Arc<CudaStream>, g: Geo, max_pos: usize) -> Result<Self, String> {
        let ops = Ops::new(st);
        let rope_base = RopeTable::new(
            &ops,
            max_pos,
            0,
            g.rope_theta,
            g.yarn_factor,
            g.yarn_beta_fast,
            g.yarn_beta_slow,
        )?;
        let rope_comp = RopeTable::new(
            &ops,
            max_pos,
            g.yarn_orig,
            g.compress_rope_theta,
            g.yarn_factor,
            g.yarn_beta_fast,
            g.yarn_beta_slow,
        )?;
        let route_bad = ops.buf(4)?;
        Ok(Self {
            ops,
            g,
            rope_base,
            rope_comp,
            scr: Scratch::default(),
            stage: None,
            trace: false,
            max_pos,
            rows_hint: max_pos,
            chunk: 8192,
            index_rows: None,
            nvtx: false,
            experts: None,
            route_bad,
            side: None,
            keyed: true,
            select_long_min: (32_768, usize::MAX),
            step: None,
        })
    }

    /// Open an NVTX range (profiling only; a no-op unless `nvtx` is set).
    pub fn nv(&self, name: &'static std::ffi::CStr) {
        if self.nvtx {
            unsafe { crate::dsv4_ffi::memra_dsv4_nvtx_push(name.as_ptr()) };
        }
    }
    /// Close the innermost NVTX range opened by [`Self::nv`].
    pub fn nv_end(&self) {
        if self.nvtx {
            unsafe { crate::dsv4_ffi::memra_dsv4_nvtx_pop() };
        }
    }

    /// One prefill row of the window selection (vendor `get_window_topk_idxs`, start_pos 0): the
    /// query at position p sees positions max(0, p-win+1) ..= p in order, -1 where that runs past p.
    pub fn window_row(p: usize, win: usize, row: &mut [i32]) {
        let start = (p + 1).saturating_sub(win);
        for (j, v) in row.iter_mut().enumerate() {
            let k = start + j;
            *v = if k > p { -1 } else { k as i32 };
        }
    }

    /// The window selection of one decode token at position `sp > 0` (vendor `get_window_topk_idxs`,
    /// start_pos > 0): the whole ring, oldest slot first, -1 for slots the ring has not reached.
    /// Below the window this is another key order than the prefill row of the same position.
    pub fn window_row_decode(sp: usize, win: usize, row: &mut [i32]) {
        let oldest = sp % win + 1;
        for (v, slot) in row.iter_mut().zip((oldest..win).chain(0..oldest)) {
            *v = if slot > sp { -1 } else { slot as i32 };
        }
    }

    /// The window selection of row `i` of a call of several decode tokens from `sp > 0`
    /// (`memra_v41_attn_idx_rows`, the host form): the decode row of position `sp + i`
    /// ([`Self::window_row_decode`]: the ring oldest slot first, -1 for slots it has not reached)
    /// as entries of `[the ring before the call (win rows) | the call's own rows]`. A slot's
    /// position is the newest one at or below the row's; a position the call feeds is entry
    /// `win + (position - sp)`, an older one is its ring slot.
    pub fn window_row_verify(sp: usize, i: usize, win: usize, row: &mut [i32]) {
        let p = sp + i;
        for (j, v) in row.iter_mut().enumerate() {
            let slot = (p % win + 1 + j) % win;
            *v = if slot > p {
                -1
            } else {
                let q = p - (p - slot) % win;
                if q >= sp {
                    (win + (q - sp)) as i32
                } else {
                    slot as i32
                }
            };
        }
    }

    /// The window selection of a whole prefill of `s` tokens: `([s][min(s, win)], width)`.
    pub fn window_idx(s: usize, win: usize) -> (Vec<i32>, usize) {
        let w = s.min(win);
        let mut v = vec![-1i32; s * w];
        for (i, row) in v.chunks_mut(w.max(1)).enumerate().take(s) {
            Self::window_row(i, win, row);
        }
        (v, w)
    }

    /// Engram (vendor `Engram.forward`) on the stream in place, `chunk` tokens at a time: gather
    /// the hashed table rows from the host mmap into pinned staging, H2D, dequant, `wkv`, the
    /// normalized gate. `hashes` is `[s][cols]`.
    pub fn engram(
        &mut self,
        stm: &StModel,
        lw: &LayerW,
        h: &Buf,
        hashes: &[i64],
        s: usize,
        tr: &mut Trace,
    ) -> Result<(), String> {
        let ew = lw.engram.as_ref().ok_or("layer has no engram")?;
        let g = self.g.clone();
        let cols = g.engram_cols();
        let hd = g.engram_head_dim;
        if hashes.len() != s * cols {
            return Err(format!(
                "engram: {} hash ids for {s} x {cols}",
                hashes.len()
            ));
        }
        let c = self.chunk.min(s).max(1);
        self.nv(c"engram");
        if self.stage.as_ref().is_none_or(|x| x.cap < c * cols) {
            self.stage = Some(engram::Stage::new(&self.ops, (c * cols).max(4096), hd)?);
        }
        let st = self.ops.st.clone();
        let nkv = g.dim * (g.hc + 1);
        let emb = self.ops.tmp(c * cols * hd * 2)?;
        let kv = self.ops.tmp(c * nkv * 2)?;
        let row_bytes = g.hc * g.dim * 2;
        for r0 in (0..s).step_by(c) {
            let n = c.min(s - r0);
            let stage = self.stage.as_mut().unwrap();
            stage.load(
                &self.ops,
                stm,
                &ew.table,
                ew.rows,
                &hashes[r0 * cols..(r0 + n) * cols],
            )?;
            ck("engram_dequant", unsafe {
                ffi::memra_v41_engram_dequant(
                    stage.d_rows.p(&st).m(),
                    stage.d_sc.p(&st).m(),
                    (n * cols) as i64,
                    hd as i32,
                    emb.p(&st).m(),
                    self.ops.sv(),
                )
            })?;
            self.ops.linear_q(
                emb.p(&st),
                cols * hd,
                n,
                &ew.wkv,
                kv.p(&st),
                nkv,
                false,
                &mut self.scr,
            )?;
            let hp = h.at(&st, r0 * row_bytes);
            ck("engram_gate", unsafe {
                ffi::memra_v41_engram_gate(
                    hp.m(),
                    kv.p(&st).m(),
                    ew.qw.p(&st).m(),
                    ew.kw.p(&st).m(),
                    n as i32,
                    g.hc as i32,
                    g.dim as i32,
                    g.norm_eps,
                    (g.dim as f32).powf(-0.5),
                    hp.m(),
                    self.ops.sv(),
                )
            })?;
            // the staging buffers are refilled next chunk: the copies and kernels must be done
            self.ops.sync()?;
        }
        self.nv_end();
        if self.trace {
            tr.engram_out = Some(clone_buf(&self.ops, h)?);
        }
        Ok(())
    }

    /// [`Self::engram`] for the rows of a batched step, whose hashed table rows and scales the
    /// host already gathered into the rank's step block (`rows` / `sc`: `[s * cols]` rows of
    /// `engram_head_dim` bytes and of a 32nd of that): dequant, `wkv`, the normalized gate. No
    /// staging and no wait: the block was uploaded before the step's first launch.
    pub fn engram_step(
        &mut self,
        lw: &LayerW,
        h: &Buf,
        rows: Dp,
        sc: Dp,
        s: usize,
    ) -> Result<(), String> {
        let ew = lw.engram.as_ref().ok_or("layer has no engram")?;
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let (cols, hd) = (g.engram_cols(), g.engram_head_dim);
        let nkv = g.dim * (g.hc + 1);
        let emb = self.ops.tmp(s * cols * hd * 2)?;
        let kv = self.ops.tmp(s * nkv * 2)?;
        ck("engram_dequant", unsafe {
            ffi::memra_v41_engram_dequant(
                rows.m(),
                sc.m(),
                (s * cols) as i64,
                hd as i32,
                emb.p(&st).m(),
                self.ops.sv(),
            )
        })?;
        self.ops.linear_q(
            emb.p(&st),
            cols * hd,
            s,
            &ew.wkv,
            kv.p(&st),
            nkv,
            false,
            &mut self.scr,
        )?;
        let hp = h.p(&st);
        ck("engram_gate", unsafe {
            ffi::memra_v41_engram_gate(
                hp.m(),
                kv.p(&st).m(),
                ew.qw.p(&st).m(),
                ew.kw.p(&st).m(),
                s as i32,
                g.hc as i32,
                g.dim as i32,
                g.norm_eps,
                (g.dim as f32).powf(-0.5),
                hp.m(),
                self.ops.sv(),
            )
        })
    }

    /// Index keys of the latent rows that just completed on a kv source (vendor `Indexer.forward`,
    /// its key half): `k_norm(wk(latent))`, RoPE at each group's first position, fp4 g32, written
    /// at rows `[row0, row0 + nrows)` of the shared index keys.
    fn index_keys(
        &mut self,
        lw: &LayerW,
        lat: &Buf,
        row0: usize,
        nrows: usize,
        sh: &mut Shared,
    ) -> Result<(), String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let iw = lw.idx.as_ref().unwrap();
        let (Some(wk), Some(kn)) = (iw.wk.as_ref(), iw.k_norm.as_ref()) else {
            return Ok(());
        };
        let ratio = lw.ratio;
        let rb = g.index_dim * 2;
        ensure_rows(
            &self.ops,
            &mut sh.index_k,
            &mut sh.index_k_cap,
            row0 + nrows,
            rb,
            self.rows_hint.min(self.max_pos) / ratio,
        )?;
        let kp = sh.index_k.as_ref().unwrap().at(&st, row0 * rb);
        self.ops
            .linear_bf16(lat.p(&st), g.head_dim, nrows, wk, kp, g.index_dim, false)?;
        self.ops.rmsnorm(
            kp,
            g.index_dim,
            kn,
            kp,
            g.index_dim,
            nrows,
            g.index_dim,
            g.norm_eps,
        )?;
        self.ops.rope(
            kp,
            g.index_dim,
            nrows,
            1,
            g.index_dim,
            &self.rope_comp,
            row0 * ratio,
            ratio,
            false,
        )?;
        self.ops.fp4_qdq(kp, g.index_dim, nrows, g.index_dim, 32)?;
        Ok(())
    }

    /// The indexer's selection for query rows `[r0, r1)` of a call of `s` tokens from position
    /// `sp` (vendor `Indexer.forward`, its query half, row-sliced like the oracle's long-context
    /// indexer: every step acts per query row). The fp4 query, the bf16 score over the `n_c`
    /// visible keys, the candidate pool on the candidate source, top-k shifted by the window
    /// length `offset`; written into rows `r0..r1` of `sh.topk` (and `sh.cand`).
    #[allow(clippy::too_many_arguments)]
    fn index_select(
        &mut self,
        lw: &LayerW,
        x: &Buf,
        qr: &Buf,
        r0: usize,
        r1: usize,
        sp: usize,
        s: usize,
        offset: usize,
        sh: &mut Shared,
    ) -> Result<(), String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let iw = lw.idx.as_ref().unwrap();
        let ratio = lw.ratio;
        let n_c = (sp + s) / ratio;
        let rows = r1 - r0;
        let ik = sh
            .index_k
            .as_ref()
            .ok_or("indexer: no index keys from a kv source")?;
        let qd = g.index_heads * g.index_dim;
        self.nv(c"index_q");
        let q = self.ops.tmp(rows * qd * 2)?;
        self.ops.linear_q(
            qr.at(&st, r0 * g.q_lora * 2),
            g.q_lora,
            rows,
            &iw.wq_b,
            q.p(&st),
            qd,
            false,
            &mut self.scr,
        )?;
        self.ops.rope(
            q.p(&st),
            qd,
            rows,
            g.index_heads,
            g.index_dim,
            &self.rope_comp,
            sp + r0,
            1,
            false,
        )?;
        self.ops.fp4_qdq(q.p(&st), qd, rows, qd, 32)?;
        let w = self.ops.tmp(rows * g.index_heads * 2)?;
        self.ops.linear_bf16(
            x.at(&st, r0 * g.dim * 2),
            g.dim,
            rows,
            &iw.wproj,
            w.p(&st),
            g.index_heads,
            false,
        )?;
        // the kernel applies weights * (index_head_dim^-0.5 * index_heads^-0.5) as a bf16 product
        let wscale = ((g.index_dim as f64).powf(-0.5) * (g.index_heads as f64).powf(-0.5)) as f32;
        self.nv_end();
        self.nv(c"index_score");
        let score = self.ops.tmp(rows * n_c * 2)?;
        let crow = r0
            .checked_sub(sh.cand_row0)
            .ok_or("indexer: query row before the candidate pool's first row")?;
        let (cand, ldc) = if lw.uses_cand {
            let c = sh.cand.as_ref().ok_or("indexer: candidate pool missing")?;
            (c.at(&st, crow * sh.cand_ld).u8c(), sh.cand_ld)
        } else {
            (std::ptr::null(), 0)
        };
        ck("index_score", unsafe {
            ffi::memra_v41_index_score(
                q.p(&st).m(),
                qd as i64,
                ik.p(&st).m(),
                n_c as i32,
                w.p(&st).m(),
                g.index_heads as i64,
                wscale,
                (sp + r0) as i32,
                ratio as i32,
                cand,
                ldc as i64,
                g.cand_block as i32,
                score.p(&st).m(),
                n_c as i64,
                rows as i32,
                self.ops.sv(),
            )
        })?;
        self.nv_end();
        if lw.cand_source {
            self.nv(c"candidates");
            let keep = sh
                .cand
                .as_ref()
                .ok_or("indexer: candidate pool not allocated")?;
            if rows == 1 && n_c > self.select_long_min.1 {
                // one decode row over a long row of keys: the row cut into blocks
                self.ops.candidates_long(
                    score.p(&st),
                    n_c,
                    g.cand_block,
                    g.cand_blocks,
                    sp + r0,
                    ratio,
                    keep.at(&st, crow * sh.cand_ld),
                )?;
            } else {
                ck("candidates", unsafe {
                    ffi::memra_v41_candidates(
                        score.p(&st).m(),
                        n_c as i64,
                        n_c as i32,
                        g.cand_block as i32,
                        g.cand_blocks as i32,
                        (sp + r0) as i32,
                        ratio as i32,
                        keep.at(&st, crow * sh.cand_ld).u8m(),
                        sh.cand_ld as i64,
                        rows as i32,
                        self.ops.sv(),
                    )
                })?;
            }
            self.nv_end();
        }
        self.nv(c"topk");
        let tw = sh.topk_w;
        let out = sh.topk.as_ref().ok_or("indexer: selection not allocated")?;
        if rows == 1 && n_c > self.select_long_min.0 {
            self.ops
                .topk_long(score.p(&st), n_c, tw, tw, offset, out.at(&st, r0 * tw * 4))?;
        } else {
            ck("topk", unsafe {
                ffi::memra_v41_topk(
                    score.p(&st).m(),
                    n_c as i64,
                    n_c as i32,
                    tw as i32,
                    tw as i32,
                    offset as i32,
                    out.at(&st, r0 * tw * 4).im(),
                    tw as i64,
                    rows as i32,
                    self.ops.sv(),
                )
            })?;
        }
        self.nv_end();
        Ok(())
    }

    /// CSA2 compressor (vendor `Compressor.forward`) for `s` tokens from `sp`: the pre-RoPE latent
    /// rows of every group that completes, with their first row index, or None while a ratio-2
    /// pair is still half full. Ratio 1 is a bf16 projection; ratio 2 pools pairs by a per-channel
    /// softmax gate in fp32, with the open pair's fp32 rows held in `ls`.
    fn compressor(
        &mut self,
        lw: &LayerW,
        x: &Buf,
        sp: usize,
        s: usize,
        ls: &mut LayerState,
    ) -> Result<Option<(Buf, usize, usize)>, String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let cw = lw.comp.as_ref().unwrap();
        let ratio = lw.ratio;
        let hd = g.head_dim;
        let norm = |c: &mut Self, lat: &Buf, n: usize| {
            c.ops
                .rmsnorm(lat.p(&st), hd, &cw.norm, lat.p(&st), hd, n, hd, g.norm_eps)
        };
        if ratio == 1 {
            let lat = self.ops.tmp(s * hd * 2)?;
            self.ops
                .linear_bf16(x.p(&st), g.dim, s, &cw.wkv, lat.p(&st), hd, false)?;
            norm(self, &lat, s)?;
            return Ok(Some((lat, sp, s)));
        }
        if ratio != 2 {
            return Err(format!("compressor ratio {ratio} has no pooling kernel"));
        }
        let kvf = self.ops.tmp(s * hd * 4)?;
        let scf = self.ops.tmp(s * hd * 4)?;
        self.ops
            .linear_bf16(x.p(&st), g.dim, s, &cw.wkv, kvf.p(&st), hd, true)?;
        self.ops.linear_bf16(
            x.p(&st),
            g.dim,
            s,
            cw.wgate.as_ref().unwrap(),
            scf.p(&st),
            hd,
            true,
        )?;
        if ls.c_kv.is_none() {
            // carried state: zeroed, so a slot no token wrote yet holds the same bytes on every rank
            ls.c_kv = Some(self.ops.buf(2 * hd * 4)?);
            ls.c_sc = Some(self.ops.buf(2 * hd * 4)?);
        }
        let (ckv, csc) = (ls.c_kv.as_ref().unwrap(), ls.c_sc.as_ref().unwrap());
        let pool = |c: &mut Self, kv: &Buf, sc: &Buf, n: usize, lat: &Buf| -> Result<(), String> {
            ck("comp_pool2", unsafe {
                ffi::memra_v41_comp_pool2(
                    kv.p(&st).f(),
                    sc.p(&st).f(),
                    hd as i64,
                    lat.p(&st).m(),
                    hd as i64,
                    n as i32,
                    hd as i32,
                    c.ops.sv(),
                )
            })
        };
        if sp == 0 {
            let n = s / 2;
            if s % 2 == 1 {
                // the trailing token waits in slot 0 for its pair
                copy_d2d(&self.ops, &kvf, (s - 1) * hd * 4, ckv, 0, hd * 4)?;
                copy_d2d(&self.ops, &scf, (s - 1) * hd * 4, csc, 0, hd * 4)?;
            }
            if n == 0 {
                return Ok(None);
            }
            let lat = self.ops.tmp(n * hd * 2)?;
            pool(self, &kvf, &scf, n, &lat)?;
            norm(self, &lat, n)?;
            return Ok(Some((lat, 0, n)));
        }
        if s != 1 {
            // Several decode tokens: the pairs that complete, in order. The first pair takes
            // the open slot's row when the call starts on an odd position; the rows wait in
            // `pend_c` for the commit, which leaves the open pair as the one-token program
            // would. Each pair is pooled by the kernel the one-token call uses, even row first.
            let odd = sp % 2;
            let n = (sp + s) / 2 - sp / 2;
            let lat = if n > 0 {
                let (pk, ps) = if odd == 1 {
                    let pk = self.ops.tmp(2 * n * hd * 4)?;
                    let ps = self.ops.tmp(2 * n * hd * 4)?;
                    copy_d2d(&self.ops, ckv, 0, &pk, 0, hd * 4)?;
                    copy_d2d(&self.ops, csc, 0, &ps, 0, hd * 4)?;
                    copy_d2d(&self.ops, &kvf, 0, &pk, hd * 4, (2 * n - 1) * hd * 4)?;
                    copy_d2d(&self.ops, &scf, 0, &ps, hd * 4, (2 * n - 1) * hd * 4)?;
                    (Some(pk), Some(ps))
                } else {
                    (None, None)
                };
                let lat = self.ops.tmp(n * hd * 2)?;
                pool(
                    self,
                    pk.as_ref().unwrap_or(&kvf),
                    ps.as_ref().unwrap_or(&scf),
                    n,
                    &lat,
                )?;
                norm(self, &lat, n)?;
                Some((lat, sp / 2, n))
            } else {
                None
            };
            ls.pend_c = Some((kvf, scf));
            return Ok(lat);
        }
        let slot = sp % 2;
        copy_d2d(&self.ops, &kvf, 0, ckv, slot * hd * 4, hd * 4)?;
        copy_d2d(&self.ops, &scf, 0, csc, slot * hd * 4, hd * 4)?;
        if !(sp + 1).is_multiple_of(2) {
            return Ok(None);
        }
        let lat = self.ops.tmp(hd * 2)?;
        pool(self, ckv, csc, 1, &lat)?;
        norm(self, &lat, 1)?;
        Ok(Some((lat, sp / 2, 1)))
    }

    /// Attention (vendor `Attention.forward`) on the normed input x `[s][dim]` from position `sp`:
    /// a prefill chunk at 0, or one decode token. Returns `[s][dim]` bf16. The keys side (window
    /// KV, compressor, index keys) runs over all `s` tokens; the query side (q, the indexer's
    /// selection, attention, the output projections) runs `chunk` query rows at a time, every
    /// step of it being per query row.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &mut self,
        lw: &LayerW,
        x: &Buf,
        sp: usize,
        s: usize,
        sh: &mut Shared,
        ls: &mut LayerState,
        tr: &mut Trace,
    ) -> Result<Buf, String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        self.nv(c"attn_keys");
        let k = self.attn_keys(lw, x, sp, s, sh, ls, tr)?;
        self.nv_end();
        let qdim = lw.heads_l * g.head_dim;
        let c = self.chunk.min(s).max(1);
        let out = self.ops.tmp(s * g.dim * 2)?;
        let q = self.ops.tmp(c * qdim * 2)?;
        let o = self.ops.tmp(c * qdim * 2)?;
        let oa_ld = g.o_groups * g.o_lora;
        let oa = self.ops.tmp(c * oa_ld * 2)?;
        let single = c == s;
        for r0 in (0..s).step_by(c) {
            let rows = c.min(s - r0);
            self.attn_heads(
                lw,
                &k,
                sh,
                sp,
                r0,
                rows,
                &q,
                &o,
                oa.p(&st),
                oa_ld,
                single,
                tr,
            )?;
            self.attn_wo_b(
                lw,
                oa.p(&st),
                oa_ld,
                rows,
                out.at(&st, r0 * g.dim * 2),
                g.dim,
            )?;
        }
        Ok(out)
    }

    /// The keys side of [`Self::attention`], identical on every rank: `qr`, this call's window KV
    /// into the ring, the compressor and index keys on a source, the indexer's selection, the
    /// compressed KV append, and the KV rows `[window part | compressed part]` the heads read.
    #[allow(clippy::too_many_arguments)]
    fn attn_keys(
        &mut self,
        lw: &LayerW,
        x: &Buf,
        sp: usize,
        s: usize,
        sh: &mut Shared,
        ls: &mut LayerState,
        tr: &mut Trace,
    ) -> Result<AttnKeys, String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        self.nv(c"qr");
        let qr = self.ops.tmp(s * g.q_lora * 2)?;
        self.ops.linear_q(
            x.p(&st),
            g.dim,
            s,
            &lw.wq_a,
            qr.p(&st),
            g.q_lora,
            false,
            &mut self.scr,
        )?;
        self.ops.rmsnorm(
            qr.p(&st),
            g.q_lora,
            &lw.q_norm,
            qr.p(&st),
            g.q_lora,
            s,
            g.q_lora,
            g.norm_eps,
        )?;
        self.nv_end();
        // Everything else of the keys side reads `x` and `qr` and nothing the heads' query
        // projection writes. On a few rows it runs on the second stream, under wq_b and its
        // RoPE (the layer's longest dense launch), and the heads join it before their first
        // read of the keys ([`Self::attn_heads`]). Not when the ranks split the indexer's rows
        // (their selections are gathered on the main stream right after) or under a trace.
        let fork = s <= SIDE_ROWS && !self.trace && self.index_rows.is_none();
        if fork {
            self.side_begin()?;
        }
        let k = self.attn_keys_rest(lw, x, qr, sp, s, sh, ls, tr);
        if fork {
            self.side_end();
        }
        k
    }

    /// [`Self::attn_keys`] after `qr`: on the stream the rank issues on.
    #[allow(clippy::too_many_arguments)]
    fn attn_keys_rest(
        &mut self,
        lw: &LayerW,
        x: &Buf,
        qr: Buf,
        sp: usize,
        s: usize,
        sh: &mut Shared,
        ls: &mut LayerState,
        tr: &mut Trace,
    ) -> Result<AttnKeys, String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let hd = g.head_dim;
        let ratio = lw.ratio;
        let win = g.window;
        let comp = ratio > 0;
        // several decode tokens in one forward (a drafted block's verification): each row runs
        // the one-token program against the ring as it stood plus the rows before it
        let dec_rows = sp > 0 && s > 1;
        // this call's window KV: kv_norm(wkv(x)), RoPE, fp8 per-32 quant-dequant
        self.nv(c"kv_window");
        let kvw = self.ops.tmp(s * hd * 2)?;
        self.ops.linear_q(
            x.p(&st),
            g.dim,
            s,
            &lw.wkv,
            kvw.p(&st),
            hd,
            false,
            &mut self.scr,
        )?;
        self.ops.rmsnorm(
            kvw.p(&st),
            hd,
            &lw.kv_norm,
            kvw.p(&st),
            hd,
            s,
            hd,
            g.norm_eps,
        )?;
        self.ops.rope(
            kvw.p(&st),
            hd,
            s,
            1,
            hd,
            if comp {
                &self.rope_comp
            } else {
                &self.rope_base
            },
            sp,
            1,
            false,
        )?;
        self.ops.fp8_qdq(kvw.p(&st), hd, s, hd)?;
        if ls.ring.is_none() {
            ls.ring = Some(self.ops.buf(win * hd * 2)?);
        }
        let ring = ls.ring.as_ref().unwrap();
        let ring_dp = ring.p(&st);
        // a decode call's heads read the ring and the compressed KV in place (a key table)
        let keyed = self.keyed && !self.trace && sp > 0;
        if !dec_rows {
            // the ring holds position p at slot p % win; a prefill seeds it with its last win
            // rows, as at most two contiguous runs (split where the slots wrap)
            Self::ring_write(&self.ops, &kvw, ring, sp, s, win, hd)?;
        }
        let (win_rows, ww) = if sp == 0 {
            (s, s.min(win))
        } else if dec_rows {
            (win + s, win)
        } else {
            (win, win)
        };
        let n_c = (sp + s).checked_div(ratio).unwrap_or(0);
        // keyed: only a several-token call needs window rows of its own, the ring as it stood
        // and then the call's rows; the compressed rows are never copied
        let kv_rows = match (keyed, dec_rows) {
            (true, true) => win + s,
            (true, false) => 1,
            (false, _) => (win_rows + n_c).max(1),
        };
        let kv_all = self.ops.tmp(kv_rows * hd * 2)?;
        if sp == 0 {
            copy_d2d(&self.ops, &kvw, 0, &kv_all, 0, s * hd * 2)?;
        } else if !keyed || dec_rows {
            copy_d2d(&self.ops, ring, 0, &kv_all, 0, win * hd * 2)?;
            if dec_rows {
                copy_d2d(&self.ops, &kvw, 0, &kv_all, win * hd * 2, s * hd * 2)?;
            }
        }
        if dec_rows {
            ls.pend_kv = Some(kvw);
        } else {
            drop(kvw);
        }
        self.nv_end();
        let c = self.chunk.min(s).max(1);
        if ratio > 0 {
            let latent = if lw.kv_source {
                self.nv(c"compressor");
                let l = self.compressor(lw, x, sp, s, ls)?;
                self.nv_end();
                l
            } else {
                None
            };
            if self.trace {
                tr.latent = match &latent {
                    Some((l, _, _)) => Some(clone_buf(&self.ops, l)?),
                    None => None,
                };
            }
            if lw.index_source {
                if n_c == 0 {
                    sh.topk = None;
                    sh.topk_w = 0;
                } else {
                    if let Some((lat, row0, nrows)) = latent.as_ref() {
                        self.nv(c"index_keys");
                        self.index_keys(lw, lat, *row0, *nrows, sh)?;
                        self.nv_end();
                    }
                    let tw = g.index_topk.min(n_c);
                    sh.topk = Some(self.ops.tmp(s * tw * 4)?);
                    sh.topk_w = tw;
                    let span = self.index_rows.clone().unwrap_or(0..s);
                    if lw.cand_source {
                        // only the rows this card selects: its later index layers read no other
                        let nb = n_c.div_ceil(g.cand_block);
                        let (a, b) = (span.start.min(s), span.end.min(s));
                        sh.cand = Some(self.ops.tmp((b - a).max(1) * nb)?);
                        sh.cand_ld = nb;
                        sh.cand_row0 = a;
                    }
                    for r0 in (span.start..span.end.min(s)).step_by(c) {
                        let r1 = (r0 + c).min(span.end.min(s));
                        self.index_select(lw, x, &qr, r0, r1, sp, s, win_rows, sh)?;
                    }
                }
            }
            self.nv(c"kv_append");
            if let Some((lat, row0, nrows)) = latent {
                self.ops.rope(
                    lat.p(&st),
                    hd,
                    nrows,
                    1,
                    hd,
                    &self.rope_comp,
                    row0 * ratio,
                    ratio,
                    false,
                )?;
                self.ops.fp4_qdq(lat.p(&st), hd, nrows, hd, 16)?;
                ensure_rows(
                    &self.ops,
                    &mut sh.compress_kv,
                    &mut sh.compress_cap,
                    row0 + nrows,
                    hd * 2,
                    self.rows_hint.min(self.max_pos) / ratio,
                )?;
                copy_d2d(
                    &self.ops,
                    &lat,
                    0,
                    sh.compress_kv.as_ref().unwrap(),
                    row0 * hd * 2,
                    nrows * hd * 2,
                )?;
                sh.n_c = row0 + nrows;
            }
            if n_c > 0 {
                let ckv = sh
                    .compress_kv
                    .as_ref()
                    .ok_or("attention: no compressed KV from the group source")?;
                if sh.n_c < n_c {
                    return Err(format!(
                        "attention: compressed KV holds {} rows, need {n_c}",
                        sh.n_c
                    ));
                }
                if !keyed {
                    copy_d2d(&self.ops, ckv, 0, &kv_all, win_rows * hd * 2, n_c * hd * 2)?;
                }
            }
            self.nv_end();
        }
        // the key table, after the ring write and the compressed append: every row of the call
        // reads the same window rows (the ring, or the call's `[ring | rows]`) and the group's
        // compressed rows
        let keys = if keyed {
            let comp_dp = match (n_c > 0, sh.compress_kv.as_ref()) {
                (true, Some(c)) => c.p(&st),
                _ => Dp::NULL,
            };
            let window = if dec_rows {
                (kv_all.p(&st), win + s)
            } else {
                (ring_dp, win)
            };
            Some(Buf::from_bytes(
                &st,
                &key_refs(&vec![(window.0, comp_dp, window.1); s]),
            )?)
        } else {
            None
        };
        let tw = if ratio > 0 && n_c > 0 { sh.topk_w } else { 0 };
        let tot = ww + tw;
        if tw > 0 && sh.topk.is_none() {
            return Err("attention: no selection from the index source".into());
        }
        Ok(AttnKeys {
            qr,
            kv_all,
            ww,
            tw,
            tot,
            win,
            comp,
            dec_rows,
            idx: None,
            pos: None,
            keys,
            keys_at: None,
        })
    }

    /// The keys side of attention for a decode call whose rows belong to different sequences
    /// (`rows[i]` is row i: one token of its sequence at its own position), in the form the
    /// gates read: every row runs the one-token program of [`Self::attn_keys`] and the heads'
    /// key rows are copies side by side, `[ring | compressed rows]` a sequence, written and
    /// read by one copy a sequence (`memra_v41_attn_idx_batch` offsets each row to its region).
    /// A traced call and `dsv41-spec --copy-keys` run this; a served step runs
    /// [`Self::attn_keys_step`], which is gated against it.
    pub fn attn_keys_batch(
        &mut self,
        lw: &LayerW,
        x: &Buf,
        pos: Dp,
        rows: &mut [BatchRow<'_>],
        sel: &mut BatchSel,
    ) -> Result<AttnKeys, String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let (hd, win, ratio) = (g.head_dim, g.window, lw.ratio);
        let comp = ratio > 0;
        let n = rows.len();
        if n == 0 || rows.iter().any(|r| r.pos == 0 || r.pos >= self.max_pos) {
            return Err("a batched decode row is one token past its sequence's prefill".into());
        }
        let qr = self.ops.tmp(n * g.q_lora * 2)?;
        self.ops.linear_q(
            x.p(&st),
            g.dim,
            n,
            &lw.wq_a,
            qr.p(&st),
            g.q_lora,
            false,
            &mut self.scr,
        )?;
        self.ops.rmsnorm(
            qr.p(&st),
            g.q_lora,
            &lw.q_norm,
            qr.p(&st),
            g.q_lora,
            n,
            g.q_lora,
            g.norm_eps,
        )?;
        // every row's window KV: kv_norm(wkv(x)), RoPE at its position, fp8 per-32 quant-dequant
        let kvw = self.ops.tmp(n * hd * 2)?;
        self.ops.linear_q(
            x.p(&st),
            g.dim,
            n,
            &lw.wkv,
            kvw.p(&st),
            hd,
            false,
            &mut self.scr,
        )?;
        self.ops.rmsnorm(
            kvw.p(&st),
            hd,
            &lw.kv_norm,
            kvw.p(&st),
            hd,
            n,
            hd,
            g.norm_eps,
        )?;
        self.ops.rope_pos(
            kvw.p(&st),
            hd,
            n,
            1,
            hd,
            if comp {
                &self.rope_comp
            } else {
                &self.rope_base
            },
            pos,
            false,
        )?;
        self.ops.fp8_qdq(kvw.p(&st), hd, n, hd)?;
        // each sequence's region of the KV rows: its ring, then its visible compressed rows
        let ncs: Vec<usize> = rows
            .iter()
            .map(|r| (r.pos + 1).checked_div(ratio).unwrap_or(0))
            .collect();
        // the regions are copies side by side
        let mut bases = Vec::with_capacity(n);
        let mut total = 0usize;
        for &c in &ncs {
            bases.push(total);
            total += win + c;
        }
        let kv_all = self.ops.tmp(total.max(1) * hd * 2)?;
        if rows.iter().any(|r| r.ls.ring.is_none()) {
            return Err("a batched decode row's sequence has no window ring (no prefill)".into());
        }
        // the latents that complete on a kv source: (row, its compressed row)
        let emit: Vec<(usize, usize)> = match (comp && lw.kv_source, ratio) {
            (false, _) => Vec::new(),
            (true, 1) => rows.iter().enumerate().map(|(i, r)| (i, r.pos)).collect(),
            (true, 2) => rows
                .iter()
                .enumerate()
                .filter(|(_, r)| (r.pos + 1).is_multiple_of(2))
                .map(|(i, r)| (i, r.pos / 2))
                .collect(),
            (true, _) => return Err(format!("compressor ratio {ratio} has no pooling kernel")),
        };
        for (i, r) in rows.iter().enumerate() {
            let ring = r.ls.ring.as_ref().unwrap();
            copy_d2d(
                &self.ops,
                &kvw,
                i * hd * 2,
                ring,
                (r.pos % win) * hd * 2,
                hd * 2,
            )?;
            copy_d2d(&self.ops, ring, 0, &kv_all, bases[i] * hd * 2, win * hd * 2)?;
        }

        drop(kvw);
        if comp {
            let mut lat: Option<Buf> = None;
            if lw.kv_source {
                let cw = lw.comp.as_ref().unwrap();
                if ratio == 1 {
                    let l = self.ops.tmp(n * hd * 2)?;
                    self.ops
                        .linear_bf16(x.p(&st), g.dim, n, &cw.wkv, l.p(&st), hd, false)?;
                    lat = Some(l);
                } else {
                    let kvf = self.ops.tmp(n * hd * 4)?;
                    let scf = self.ops.tmp(n * hd * 4)?;
                    self.ops
                        .linear_bf16(x.p(&st), g.dim, n, &cw.wkv, kvf.p(&st), hd, true)?;
                    self.ops.linear_bf16(
                        x.p(&st),
                        g.dim,
                        n,
                        cw.wgate.as_ref().unwrap(),
                        scf.p(&st),
                        hd,
                        true,
                    )?;
                    let l = self.ops.tmp(n * hd * 2)?;
                    let mut done = 0usize;
                    for (i, r) in rows.iter_mut().enumerate() {
                        if r.ls.c_kv.is_none() {
                            r.ls.c_kv = Some(self.ops.buf(2 * hd * 4)?);
                            r.ls.c_sc = Some(self.ops.buf(2 * hd * 4)?);
                        }
                        let (ckv, csc) = (r.ls.c_kv.as_ref().unwrap(), r.ls.c_sc.as_ref().unwrap());
                        let slot = r.pos % 2;
                        copy_d2d(&self.ops, &kvf, i * hd * 4, ckv, slot * hd * 4, hd * 4)?;
                        copy_d2d(&self.ops, &scf, i * hd * 4, csc, slot * hd * 4, hd * 4)?;

                        let (ckv, csc) = (r.ls.c_kv.as_ref().unwrap(), r.ls.c_sc.as_ref().unwrap());
                        if (r.pos + 1).is_multiple_of(2) {
                            ck("comp_pool2", unsafe {
                                ffi::memra_v41_comp_pool2(
                                    ckv.p(&st).f(),
                                    csc.p(&st).f(),
                                    hd as i64,
                                    l.at(&st, done * hd * 2).m(),
                                    hd as i64,
                                    1,
                                    hd as i32,
                                    self.ops.sv(),
                                )
                            })?;
                            done += 1;
                        }
                    }
                    lat = Some(l);
                }
                if let (Some(l), false) = (lat.as_ref(), emit.is_empty()) {
                    self.ops.rmsnorm(
                        l.p(&st),
                        hd,
                        &cw.norm,
                        l.p(&st),
                        hd,
                        emit.len(),
                        hd,
                        g.norm_eps,
                    )?;
                }
            }
            // the position of each latent's group (its first token), for the keys' and the KV's RoPE
            let epos = if emit.is_empty() {
                None
            } else {
                let p: Vec<i32> = emit.iter().map(|&(_, r0)| (r0 * ratio) as i32).collect();
                Some(Buf::from_i32(&st, &p)?)
            };
            if lw.index_source {
                let iw = lw.idx.as_ref().unwrap();
                let idim = g.index_dim;
                if let (Some(l), Some(ep), Some(wk), Some(kn)) = (
                    lat.as_ref(),
                    epos.as_ref(),
                    iw.wk.as_ref(),
                    iw.k_norm.as_ref(),
                ) {
                    // the completed latents' index keys, each into its sequence's keys
                    let e = emit.len();
                    let kp = self.ops.tmp(e * idim * 2)?;
                    self.ops
                        .linear_bf16(l.p(&st), hd, e, wk, kp.p(&st), idim, false)?;
                    self.ops
                        .rmsnorm(kp.p(&st), idim, kn, kp.p(&st), idim, e, idim, g.norm_eps)?;
                    self.ops.rope_pos(
                        kp.p(&st),
                        idim,
                        e,
                        1,
                        idim,
                        &self.rope_comp,
                        ep.p(&st),
                        false,
                    )?;
                    self.ops.fp4_qdq(kp.p(&st), idim, e, idim, 32)?;
                    for (j, &(i, row0)) in emit.iter().enumerate() {
                        let hint = rows[i].cap.min(self.max_pos) / ratio;
                        let sh = &mut *rows[i].sh;
                        ensure_rows(
                            &self.ops,
                            &mut sh.index_k,
                            &mut sh.index_k_cap,
                            row0 + 1,
                            idim * 2,
                            hint,
                        )?;
                        copy_d2d(
                            &self.ops,
                            &kp,
                            j * idim * 2,
                            sh.index_k.as_ref().unwrap(),
                            row0 * idim * 2,
                            idim * 2,
                        )?;
                    }
                }
                // the selection: the query and the head weights of every row at once, then each
                // sequence's score over its own keys, its candidate pool and its top-k
                let tw = ncs.iter().map(|&c| g.index_topk.min(c)).max().unwrap_or(0);
                if tw == 0 {
                    *sel = BatchSel {
                        topk: None,
                        tw: 0,
                        has: vec![false; n],
                    };
                } else {
                    let qd = g.index_heads * idim;
                    let q = self.ops.tmp(n * qd * 2)?;
                    self.ops.linear_q(
                        qr.p(&st),
                        g.q_lora,
                        n,
                        &iw.wq_b,
                        q.p(&st),
                        qd,
                        false,
                        &mut self.scr,
                    )?;
                    self.ops.rope_pos(
                        q.p(&st),
                        qd,
                        n,
                        g.index_heads,
                        idim,
                        &self.rope_comp,
                        pos,
                        false,
                    )?;
                    self.ops.fp4_qdq(q.p(&st), qd, n, qd, 32)?;
                    let w = self.ops.tmp(n * g.index_heads * 2)?;
                    self.ops.linear_bf16(
                        x.p(&st),
                        g.dim,
                        n,
                        &iw.wproj,
                        w.p(&st),
                        g.index_heads,
                        false,
                    )?;
                    let wscale =
                        ((idim as f64).powf(-0.5) * (g.index_heads as f64).powf(-0.5)) as f32;
                    let out = self.ops.tmp(n * tw * 4)?;
                    let mut has = vec![false; n];
                    for (i, r) in rows.iter_mut().enumerate() {
                        let n_c = ncs[i];
                        if n_c == 0 {
                            continue;
                        }
                        let sh = &mut *r.sh;
                        let ik = sh
                            .index_k
                            .as_ref()
                            .ok_or("indexer: no index keys from a kv source")?;
                        let score = self.ops.tmp(n_c * 2)?;
                        let (cand, ldc) = if lw.uses_cand {
                            let c = sh.cand.as_ref().ok_or("indexer: candidate pool missing")?;
                            (c.p(&st).u8c(), sh.cand_ld)
                        } else {
                            (std::ptr::null(), 0)
                        };
                        ck("index_score", unsafe {
                            ffi::memra_v41_index_score(
                                q.at(&st, i * qd * 2).m(),
                                qd as i64,
                                ik.p(&st).m(),
                                n_c as i32,
                                w.at(&st, i * g.index_heads * 2).m(),
                                g.index_heads as i64,
                                wscale,
                                r.pos as i32,
                                ratio as i32,
                                cand,
                                ldc as i64,
                                g.cand_block as i32,
                                score.p(&st).m(),
                                n_c as i64,
                                1,
                                self.ops.sv(),
                            )
                        })?;
                        if lw.cand_source {
                            let nb = n_c.div_ceil(g.cand_block);
                            let keep = self.ops.tmp(nb.max(1))?;
                            if n_c > self.select_long_min.1 {
                                self.ops.candidates_long(
                                    score.p(&st),
                                    n_c,
                                    g.cand_block,
                                    g.cand_blocks,
                                    r.pos,
                                    ratio,
                                    keep.p(&st),
                                )?;
                            } else {
                                ck("candidates", unsafe {
                                    ffi::memra_v41_candidates(
                                        score.p(&st).m(),
                                        n_c as i64,
                                        n_c as i32,
                                        g.cand_block as i32,
                                        g.cand_blocks as i32,
                                        r.pos as i32,
                                        ratio as i32,
                                        keep.p(&st).u8m(),
                                        nb as i64,
                                        1,
                                        self.ops.sv(),
                                    )
                                })?;
                            }
                            sh.cand = Some(keep);
                            sh.cand_ld = nb;
                            sh.cand_row0 = 0;
                        }
                        if n_c > self.select_long_min.0 {
                            self.ops.topk_long(
                                score.p(&st),
                                n_c,
                                g.index_topk.min(n_c),
                                tw,
                                bases[i] + win,
                                out.at(&st, i * tw * 4),
                            )?;
                        } else {
                            ck("topk", unsafe {
                                ffi::memra_v41_topk(
                                    score.p(&st).m(),
                                    n_c as i64,
                                    n_c as i32,
                                    g.index_topk.min(n_c) as i32,
                                    tw as i32,
                                    (bases[i] + win) as i32,
                                    out.at(&st, i * tw * 4).im(),
                                    tw as i64,
                                    1,
                                    self.ops.sv(),
                                )
                            })?;
                        }
                        has[i] = true;
                    }
                    *sel = BatchSel {
                        topk: Some(out),
                        tw,
                        has,
                    };
                }
            }
            // the completed latents into each sequence's compressed KV
            if let (Some(l), Some(ep)) = (lat.as_ref(), epos.as_ref()) {
                let e = emit.len();
                self.ops
                    .rope_pos(l.p(&st), hd, e, 1, hd, &self.rope_comp, ep.p(&st), false)?;
                self.ops.fp4_qdq(l.p(&st), hd, e, hd, 16)?;
                for (j, &(i, row0)) in emit.iter().enumerate() {
                    let hint = rows[i].cap.min(self.max_pos) / ratio;
                    let sh = &mut *rows[i].sh;
                    ensure_rows(
                        &self.ops,
                        &mut sh.compress_kv,
                        &mut sh.compress_cap,
                        row0 + 1,
                        hd * 2,
                        hint,
                    )?;
                    copy_d2d(
                        &self.ops,
                        l,
                        j * hd * 2,
                        sh.compress_kv.as_ref().unwrap(),
                        row0 * hd * 2,
                        hd * 2,
                    )?;

                    sh.n_c = row0 + 1;
                }
            }
            // each sequence's visible compressed rows after its ring
            for (i, r) in rows.iter().enumerate() {
                let n_c = ncs[i];
                if n_c == 0 {
                    continue;
                }
                let ckv =
                    r.sh.compress_kv
                        .as_ref()
                        .ok_or("attention: no compressed KV from the group source")?;
                if r.sh.n_c < n_c {
                    return Err(format!(
                        "attention: compressed KV holds {} rows, need {n_c}",
                        r.sh.n_c
                    ));
                }
                copy_d2d(
                    &self.ops,
                    ckv,
                    0,
                    &kv_all,
                    (bases[i] + win) * hd * 2,
                    n_c * hd * 2,
                )?;
            }
        }
        let tw = if comp && ncs.iter().any(|&c| c > 0) {
            if sel.topk.is_none() || sel.has.len() != n {
                return Err("attention: no selection from the index source".into());
            }
            sel.tw
        } else {
            0
        };
        let tot = win + tw;
        // each row's region base and how many selection entries it holds
        let mut meta: Vec<i32> = bases.iter().map(|&b| b as i32).collect();
        meta.extend((0..n).map(|i| if tw > 0 && sel.has[i] { tw as i32 } else { 0 }));
        let meta = Buf::from_i32(&st, &meta)?;
        let idx = self.ops.tmp(n * tot * 4)?;
        let selp = match (tw, sel.topk.as_ref()) {
            (0, _) | (_, None) => Dp::NULL,
            (_, Some(t)) => t.p(&st),
        };
        ck("attn_idx_batch", unsafe {
            ffi::memra_v41_attn_idx_batch(
                selp.i(),
                tw as i32,
                pos.i(),
                meta.p(&st).i(),
                meta.at(&st, n * 4).i(),
                n as i32,
                win as i32,
                tot as i32,
                idx.p(&st).im(),
                self.ops.sv(),
            )
        })?;
        Ok(AttnKeys {
            qr,
            kv_all,
            ww: win,
            tw,
            tot,
            win,
            comp,
            dec_rows: false,
            idx: Some(idx),
            pos: Some(pos),
            keys: None,
            keys_at: None,
        })
    }

    /// Rows of `kv` (bf16 `[s][hd]`, positions `sp ..`) into the ring, position p at slot
    /// p % win: the last `win` of them, as at most two contiguous runs (split where the slots wrap).
    /// The host half of [`Self::attn_keys_step`] for one layer, before any launch of the step:
    /// size what a completing row appends to (a first use allocates, a row past the sequence's
    /// first sizing grows it), count the rows a completed latent adds, decide the selection the
    /// layer's attention reads, and write the layer's tables ([`step_layer_bytes`]) into `out`.
    /// `carried` is the selection width and holders of the last index source, as the layers
    /// after it read them.
    pub fn plan_keys_step(
        &mut self,
        kind: LayerKind,
        rows: &mut [BatchRow<'_>],
        carried: &mut (usize, Vec<bool>),
        out: &mut [u8],
    ) -> Result<(usize, Vec<bool>), String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let (hd, win, idim, ratio) = (g.head_dim, g.window, g.index_dim, kind.ratio);
        let comp = ratio > 0;
        let n = rows.len();
        if n == 0 || rows.iter().any(|r| r.pos == 0 || r.pos >= self.max_pos) {
            return Err("a batched decode row is one token past its sequence's prefill".into());
        }
        if rows.iter().any(|r| r.ls.ring.is_none()) {
            return Err("a batched decode row's sequence has no window ring (no prefill)".into());
        }
        if out.len() < n * 92 {
            return Err("step block: a layer's tables do not fit".into());
        }
        let ncs: Vec<usize> = rows
            .iter()
            .map(|r| (r.pos + 1).checked_div(ratio).unwrap_or(0))
            .collect();
        // the compressed row a row's latent completes on a kv source (None: it completes none)
        let done: Vec<Option<usize>> = match (comp && kind.kv_source, ratio) {
            (false, _) => vec![None; n],
            (true, 1) => rows.iter().map(|r| Some(r.pos)).collect(),
            (true, 2) => rows
                .iter()
                .map(|r| (r.pos + 1).is_multiple_of(2).then_some(r.pos / 2))
                .collect(),
            (true, _) => return Err(format!("compressor ratio {ratio} has no pooling kernel")),
        };
        let pair = comp && kind.kv_source && ratio == 2;
        let own_keys = kind.kv_source && kind.index_source;
        for (r, d) in rows.iter_mut().zip(&done) {
            if pair && r.ls.c_kv.is_none() {
                r.ls.c_kv = Some(self.ops.buf(2 * hd * 4)?);
                r.ls.c_sc = Some(self.ops.buf(2 * hd * 4)?);
            }
            if let Some(row0) = *d {
                let hint = r.cap.min(self.max_pos) / ratio;
                let sh = &mut *r.sh;
                if own_keys {
                    ensure_rows(
                        &self.ops,
                        &mut sh.index_k,
                        &mut sh.index_k_cap,
                        row0 + 1,
                        idim * 2,
                        hint,
                    )?;
                }
                ensure_rows(
                    &self.ops,
                    &mut sh.compress_kv,
                    &mut sh.compress_cap,
                    row0 + 1,
                    hd * 2,
                    hint,
                )?;
                sh.n_c = row0 + 1;
            }
        }
        for (r, &c) in rows.iter().zip(&ncs) {
            if c == 0 {
                continue;
            }
            if r.sh.compress_kv.is_none() {
                return Err("attention: no compressed KV from the group source".into());
            }
            if r.sh.n_c < c {
                return Err(format!(
                    "attention: compressed KV holds {} rows, need {c}",
                    r.sh.n_c
                ));
            }
            if kind.index_source && r.sh.index_k.is_none() {
                return Err("indexer: no index keys from a kv source".into());
            }
        }
        if comp && kind.index_source {
            // the selection width in classes (64, 128, 256, ... up to the model's top-k): the
            // rows are padded with -1, which the heads skip, so the width is byte-neutral and a
            // step's launch parameters take a handful of values instead of one a position
            let most = ncs.iter().map(|&c| g.index_topk.min(c)).max().unwrap_or(0);
            *carried = (
                select_class(most, g.index_topk),
                ncs.iter().map(|&c| c > 0).collect(),
            );
        }
        let (tw, has) = if comp && ncs.iter().any(|&c| c > 0) {
            if carried.1.len() != n {
                return Err("attention: no selection from the index source".into());
            }
            carried.clone()
        } else {
            (0, vec![false; n])
        };
        let refs: Vec<(Dp, Dp, usize)> = rows
            .iter()
            .zip(&ncs)
            .map(|(r, &c)| {
                let comp_dp = match (c > 0, r.sh.compress_kv.as_ref()) {
                    (true, Some(b)) => b.p(&st),
                    _ => Dp::NULL,
                };
                (r.ls.ring.as_ref().unwrap().p(&st), comp_dp, win)
            })
            .collect();
        let mut t = key_refs(&refs);
        let mut put = |a: u64| t.extend_from_slice(&a.to_le_bytes());
        for r in rows.iter() {
            let ring = r.ls.ring.as_ref().unwrap();
            put(ring.at(&st, (r.pos % win) * hd * 2).addr());
        }
        // the open pair: this row's slot of the kv rows and of the gate rows, then both buffers
        for (slot, which) in [(true, 0), (true, 1), (false, 0), (false, 1)] {
            for r in rows.iter() {
                let b = if which == 0 { &r.ls.c_kv } else { &r.ls.c_sc };
                put(match (pair, b.as_ref()) {
                    (true, Some(b)) if slot => b.at(&st, (r.pos % 2) * hd * 4).addr(),
                    (true, Some(b)) => b.p(&st).addr(),
                    _ => 0,
                });
            }
        }
        for (r, d) in rows.iter().zip(&done) {
            put(match (own_keys, *d, r.sh.index_k.as_ref()) {
                (true, Some(row0), Some(b)) => b.at(&st, row0 * idim * 2).addr(),
                _ => 0,
            });
        }
        for (r, d) in rows.iter().zip(&done) {
            put(match (*d, r.sh.compress_kv.as_ref()) {
                (Some(row0), Some(b)) => b.at(&st, row0 * hd * 2).addr(),
                _ => 0,
            });
        }
        // the position of each latent's group (its first token); a row that completes none
        // still runs the launches, on a latent nobody reads
        for (r, d) in rows.iter().zip(&done) {
            let p = d.map_or(r.pos, |row0| row0 * ratio);
            t.extend_from_slice(&(p as i32).to_le_bytes());
        }
        t.extend_from_slice(&vec![0u8; n * 4]);
        for h in &has {
            let k = if tw > 0 && *h { tw as i32 } else { 0 };
            t.extend_from_slice(&k.to_le_bytes());
        }
        debug_assert_eq!(t.len(), n * 92);
        out[..t.len()].copy_from_slice(&t);
        Ok((tw, has))
    }

    /// The keys side of attention for one layer of a batched step ([`Runner::forward_batch`]):
    /// row i is one token of its sequence at its own position and runs the one-token program
    /// of [`Self::attn_keys`]. Everything the host decides sits in the rank's step block
    /// before the first launch (`step`: [`Self::plan_keys_step`]'s tables; `pos`: the rows'
    /// positions in it), so the launches of a layer depend on the row count and the layer's
    /// kind alone: the projections, norms, RoPE and quantize-dequantize steps over all rows,
    /// one `memra_v41_put_rows` a kind of write (ring slot, open pair slots, appended index key
    /// and compressed row), and on a ratio-2 kv source the pooling of every row's pair
    /// (`memra_v41_comp_pool2_keys`): a row whose pair is not complete is pooled too and its
    /// latent goes nowhere (a null address in the table), so nothing depends on which
    /// sequences complete a pair this step. The heads read each row's ring and its group's
    /// compressed rows where they live (`memra_v41_sparse_attn_keys`). What stays one launch a
    /// sequence is what grows with it: the index score over its keys, its candidate pool and
    /// its top-k. Gated against [`Self::attn_keys_batch`], row for row.
    pub fn attn_keys_step(
        &mut self,
        lw: &LayerW,
        x: &Buf,
        pos: Dp,
        step: StepRef<'_>,
        rows: &mut [BatchRow<'_>],
        sel: &mut BatchSel,
    ) -> Result<AttnKeys, String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let n = rows.len();
        let qr = self.ops.tmp(n * g.q_lora * 2)?;
        self.ops.linear_q(
            x.p(&st),
            g.dim,
            n,
            &lw.wq_a,
            qr.p(&st),
            g.q_lora,
            false,
            &mut self.scr,
        )?;
        self.ops.rmsnorm(
            qr.p(&st),
            g.q_lora,
            &lw.q_norm,
            qr.p(&st),
            g.q_lora,
            n,
            g.q_lora,
            g.norm_eps,
        )?;
        // the rest of the keys side on the second stream, under the heads' wq_b
        // (see [`Self::attn_keys`])
        let fork = n <= SIDE_ROWS;
        if fork {
            self.side_begin()?;
        }
        let k = self.attn_keys_step_rest(lw, x, qr, pos, step, rows, sel);
        if fork {
            self.side_end();
        }
        k
    }

    /// [`Self::attn_keys_step`] after `qr`: on the stream the rank issues on.
    #[allow(clippy::too_many_arguments)]
    fn attn_keys_step_rest(
        &mut self,
        lw: &LayerW,
        x: &Buf,
        qr: Buf,
        pos: Dp,
        step: StepRef<'_>,
        rows: &mut [BatchRow<'_>],
        sel: &mut BatchSel,
    ) -> Result<AttnKeys, String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let (hd, win, ratio) = (g.head_dim, g.window, lw.ratio);
        let comp = ratio > 0;
        let n = rows.len();
        let tab = |section: usize| Dp(step.blk.addr() + (step.sl.off + section * n) as u64);
        // the layer's tables: byte offsets a row (see `step_layer_bytes`)
        let (t_ring, t_ckv, t_csc, t_pkv, t_psc, t_ik, t_ca, t_epos, t_base, t_nsel) =
            (24, 32, 40, 48, 56, 64, 72, 80, 84, 88);
        // every row's window KV into its ring slot
        let kvw = self.ops.tmp(n * hd * 2)?;
        self.ops.linear_q(
            x.p(&st),
            g.dim,
            n,
            &lw.wkv,
            kvw.p(&st),
            hd,
            false,
            &mut self.scr,
        )?;
        self.ops.rmsnorm(
            kvw.p(&st),
            hd,
            &lw.kv_norm,
            kvw.p(&st),
            hd,
            n,
            hd,
            g.norm_eps,
        )?;
        let table = if comp {
            &self.rope_comp
        } else {
            &self.rope_base
        };
        self.ops
            .rope_pos(kvw.p(&st), hd, n, 1, hd, table, pos, false)?;
        self.ops.fp8_qdq(kvw.p(&st), hd, n, hd)?;
        put_rows(&self.ops, kvw.p(&st), hd, tab(t_ring), n, hd)?;
        drop(kvw);
        let ncs: Vec<usize> = rows
            .iter()
            .map(|r| (r.pos + 1).checked_div(ratio).unwrap_or(0))
            .collect();
        let tw = step.sl.tw;
        if comp {
            // a kv source: every row's latent (a ratio-2 row whose pair is open gives one too)
            let mut lat: Option<Buf> = None;
            if lw.kv_source {
                let cw = lw.comp.as_ref().unwrap();
                let l = self.ops.tmp(n * hd * 2)?;
                if ratio == 1 {
                    self.ops
                        .linear_bf16(x.p(&st), g.dim, n, &cw.wkv, l.p(&st), hd, false)?;
                } else {
                    let kvf = self.ops.tmp(n * hd * 4)?;
                    let scf = self.ops.tmp(n * hd * 4)?;
                    self.ops
                        .linear_bf16(x.p(&st), g.dim, n, &cw.wkv, kvf.p(&st), hd, true)?;
                    self.ops.linear_bf16(
                        x.p(&st),
                        g.dim,
                        n,
                        cw.wgate.as_ref().unwrap(),
                        scf.p(&st),
                        hd,
                        true,
                    )?;
                    // the fp32 rows as 16-bit words, each into its open pair's slot; then the
                    // pairs pooled where they are
                    put_rows(&self.ops, kvf.p(&st), hd * 2, tab(t_ckv), n, hd * 2)?;
                    put_rows(&self.ops, scf.p(&st), hd * 2, tab(t_csc), n, hd * 2)?;
                    ck("comp_pool2_keys", unsafe {
                        ffi::memra_v41_comp_pool2_keys(
                            tab(t_pkv).c(),
                            tab(t_psc).c(),
                            l.p(&st).m(),
                            hd as i64,
                            n as i32,
                            hd as i32,
                            self.ops.sv(),
                        )
                    })?;
                }
                self.ops
                    .rmsnorm(l.p(&st), hd, &cw.norm, l.p(&st), hd, n, hd, g.norm_eps)?;
                lat = Some(l);
            }
            if lw.index_source {
                let iw = lw.idx.as_ref().unwrap();
                let idim = g.index_dim;
                if let (Some(l), Some(wk), Some(kn)) =
                    (lat.as_ref(), iw.wk.as_ref(), iw.k_norm.as_ref())
                {
                    // the latents' index keys, each completed one into its sequence's keys
                    let kp = self.ops.tmp(n * idim * 2)?;
                    self.ops
                        .linear_bf16(l.p(&st), hd, n, wk, kp.p(&st), idim, false)?;
                    self.ops
                        .rmsnorm(kp.p(&st), idim, kn, kp.p(&st), idim, n, idim, g.norm_eps)?;
                    self.ops.rope_pos(
                        kp.p(&st),
                        idim,
                        n,
                        1,
                        idim,
                        &self.rope_comp,
                        tab(t_epos),
                        false,
                    )?;
                    self.ops.fp4_qdq(kp.p(&st), idim, n, idim, 32)?;
                    put_rows(&self.ops, kp.p(&st), idim, tab(t_ik), n, idim)?;
                }
                // the selection: the query and the head weights of every row at once, then each
                // sequence's score over its own keys, its candidate pool and its top-k
                if tw == 0 {
                    *sel = BatchSel {
                        topk: None,
                        tw: 0,
                        has: vec![false; n],
                    };
                } else {
                    let qd = g.index_heads * idim;
                    let q = self.ops.tmp(n * qd * 2)?;
                    self.ops.linear_q(
                        qr.p(&st),
                        g.q_lora,
                        n,
                        &iw.wq_b,
                        q.p(&st),
                        qd,
                        false,
                        &mut self.scr,
                    )?;
                    self.ops.rope_pos(
                        q.p(&st),
                        qd,
                        n,
                        g.index_heads,
                        idim,
                        &self.rope_comp,
                        pos,
                        false,
                    )?;
                    self.ops.fp4_qdq(q.p(&st), qd, n, qd, 32)?;
                    let w = self.ops.tmp(n * g.index_heads * 2)?;
                    self.ops.linear_bf16(
                        x.p(&st),
                        g.dim,
                        n,
                        &iw.wproj,
                        w.p(&st),
                        g.index_heads,
                        false,
                    )?;
                    let wscale =
                        ((idim as f64).powf(-0.5) * (g.index_heads as f64).powf(-0.5)) as f32;
                    let out = self.ops.tmp(n * tw * 4)?;
                    let mut has = vec![false; n];
                    for (i, r) in rows.iter_mut().enumerate() {
                        let n_c = ncs[i];
                        if n_c == 0 {
                            continue;
                        }
                        let sh = &mut *r.sh;
                        let ik = sh
                            .index_k
                            .as_ref()
                            .ok_or("indexer: no index keys from a kv source")?;
                        let score = self.ops.tmp(n_c * 2)?;
                        let (cand, ldc) = if lw.uses_cand {
                            let c = sh.cand.as_ref().ok_or("indexer: candidate pool missing")?;
                            (c.p(&st).u8c(), sh.cand_ld)
                        } else {
                            (std::ptr::null(), 0)
                        };
                        ck("index_score", unsafe {
                            ffi::memra_v41_index_score(
                                q.at(&st, i * qd * 2).m(),
                                qd as i64,
                                ik.p(&st).m(),
                                n_c as i32,
                                w.at(&st, i * g.index_heads * 2).m(),
                                g.index_heads as i64,
                                wscale,
                                r.pos as i32,
                                ratio as i32,
                                cand,
                                ldc as i64,
                                g.cand_block as i32,
                                score.p(&st).m(),
                                n_c as i64,
                                1,
                                self.ops.sv(),
                            )
                        })?;
                        if lw.cand_source {
                            let nb = n_c.div_ceil(g.cand_block);
                            let keep = self.ops.tmp(nb.max(1))?;
                            if n_c > self.select_long_min.1 {
                                self.ops.candidates_long(
                                    score.p(&st),
                                    n_c,
                                    g.cand_block,
                                    g.cand_blocks,
                                    r.pos,
                                    ratio,
                                    keep.p(&st),
                                )?;
                            } else {
                                ck("candidates", unsafe {
                                    ffi::memra_v41_candidates(
                                        score.p(&st).m(),
                                        n_c as i64,
                                        n_c as i32,
                                        g.cand_block as i32,
                                        g.cand_blocks as i32,
                                        r.pos as i32,
                                        ratio as i32,
                                        keep.p(&st).u8m(),
                                        nb as i64,
                                        1,
                                        self.ops.sv(),
                                    )
                                })?;
                            }
                            sh.cand = Some(keep);
                            sh.cand_ld = nb;
                            sh.cand_row0 = 0;
                        }
                        if n_c > self.select_long_min.0 {
                            self.ops.topk_long(
                                score.p(&st),
                                n_c,
                                g.index_topk.min(n_c),
                                tw,
                                win,
                                out.at(&st, i * tw * 4),
                            )?;
                        } else {
                            ck("topk", unsafe {
                                ffi::memra_v41_topk(
                                    score.p(&st).m(),
                                    n_c as i64,
                                    n_c as i32,
                                    g.index_topk.min(n_c) as i32,
                                    tw as i32,
                                    win as i32,
                                    out.at(&st, i * tw * 4).im(),
                                    tw as i64,
                                    1,
                                    self.ops.sv(),
                                )
                            })?;
                        }
                        has[i] = true;
                    }
                    debug_assert_eq!(&has, step.sl.has.as_slice());
                    *sel = BatchSel {
                        topk: Some(out),
                        tw,
                        has,
                    };
                }
            }
            // the completed latents into each sequence's compressed KV
            if let Some(l) = lat.as_ref() {
                self.ops
                    .rope_pos(l.p(&st), hd, n, 1, hd, &self.rope_comp, tab(t_epos), false)?;
                self.ops.fp4_qdq(l.p(&st), hd, n, hd, 16)?;
                put_rows(&self.ops, l.p(&st), hd, tab(t_ca), n, hd)?;
            }
        }
        let tw = if comp && ncs.iter().any(|&c| c > 0) {
            if sel.topk.is_none() || sel.has.len() != n || sel.tw != tw {
                return Err("attention: no selection from the index source".into());
            }
            tw
        } else {
            0
        };
        let tot = win + tw;
        let idx = self.ops.tmp(n * tot * 4)?;
        let selp = match (tw, sel.topk.as_ref()) {
            (0, _) | (_, None) => Dp::NULL,
            (_, Some(t)) => t.p(&st),
        };
        ck("attn_idx_batch", unsafe {
            ffi::memra_v41_attn_idx_batch(
                selp.i(),
                tw as i32,
                pos.i(),
                tab(t_base).i(),
                tab(t_nsel).i(),
                n as i32,
                win as i32,
                tot as i32,
                idx.p(&st).im(),
                self.ops.sv(),
            )
        })?;
        Ok(AttnKeys {
            qr,
            kv_all: self.ops.tmp(1)?,
            ww: win,
            tw,
            tot,
            win,
            comp,
            dec_rows: false,
            idx: Some(idx),
            pos: Some(pos),
            keys: None,
            keys_at: Some(tab(0)),
        })
    }

    fn ring_write(
        ops: &Ops,
        kv: &Buf,
        ring: &Buf,
        sp: usize,
        s: usize,
        win: usize,
        hd: usize,
    ) -> Result<(), String> {
        let mut p = (sp + s).saturating_sub(win).max(sp);
        while p < sp + s {
            let run = (win - p % win).min(sp + s - p);
            copy_d2d(
                ops,
                kv,
                (p - sp) * hd * 2,
                ring,
                (p % win) * hd * 2,
                run * hd * 2,
            )?;
            p += run;
        }
        Ok(())
    }

    /// Close a forward of several decode tokens from `sp` on this layer: its first `keep` rows
    /// become state (the ring slots and, on a ratio-2 kv source, the open pair's slots, as the
    /// one-token program leaves them after position `sp + keep - 1`); the rows past them are
    /// dropped. A layer the forward did not reach has nothing pending.
    pub fn commit_rows(
        &mut self,
        ls: &mut LayerState,
        sp: usize,
        keep: usize,
    ) -> Result<(), String> {
        let (win, hd) = (self.g.window, self.g.head_dim);
        if let Some(kv) = ls.pend_kv.take() {
            if keep * hd * 2 > kv.len {
                return Err("commit: more rows than the forward ran".into());
            }
            let ring = ls.ring.as_ref().ok_or("commit: the layer has no ring")?;
            Self::ring_write(&self.ops, &kv, ring, sp, keep, win, hd)?;
        }
        if let Some((kvf, scf)) = ls.pend_c.take() {
            let (ckv, csc) = (
                ls.c_kv.as_ref().ok_or("commit: no open pair")?,
                ls.c_sc.as_ref().ok_or("commit: no open pair")?,
            );
            for i in keep.saturating_sub(2)..keep {
                let slot = (sp + i) % 2;
                copy_d2d(&self.ops, &kvf, i * hd * 4, ckv, slot * hd * 4, hd * 4)?;
                copy_d2d(&self.ops, &scf, i * hd * 4, csc, slot * hd * 4, hd * 4)?;
            }
        }
        Ok(())
    }

    /// The query side of [`Self::attention`] for rows `[r0, r0 + rows)` over this card's heads
    /// (`lw.heads_l` from `lw.head0`; all of them on one card): q, RoPE, the selection rows, sparse
    /// attention, inverse RoPE, the fp8 quantize-dequantize of o and the grouped `wo_a` of this
    /// card's output groups, written at element column `lw.group0 * o_lora` of `oa` (row pitch
    /// `oa_ld` elements). Every step is per row and per head.
    #[allow(clippy::too_many_arguments)]
    fn attn_heads(
        &mut self,
        lw: &LayerW,
        k: &AttnKeys,
        sh: &Shared,
        sp: usize,
        r0: usize,
        rows: usize,
        q: &Buf,
        o: &Buf,
        oa: Dp,
        oa_ld: usize,
        single: bool,
        tr: &mut Trace,
    ) -> Result<(), String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let hd = g.head_dim;
        let heads = lw.heads_l;
        let qdim = heads * hd;
        let gk = g.heads * hd / g.o_groups; // 4096 inputs per group
        let scale = (hd as f64).powf(-0.5) as f32;
        let (ww, tw, tot, win) = (k.ww, k.tw, k.tot, k.win);
        self.nv(c"q_b");
        self.ops.linear_q(
            k.qr.at(&st, r0 * g.q_lora * 2),
            g.q_lora,
            rows,
            &lw.wq_b,
            q.p(&st),
            qdim,
            false,
            &mut self.scr,
        )?;
        let table = if k.comp {
            &self.rope_comp
        } else {
            &self.rope_base
        };
        match k.pos {
            // a batch of sequences: each row at its own position
            Some(p) => self.ops.rope_pos(
                q.p(&st),
                qdim,
                rows,
                heads,
                hd,
                table,
                Dp(p.addr() + (r0 * 4) as u64),
                false,
            )?,
            None => self
                .ops
                .rope(q.p(&st), qdim, rows, heads, hd, table, sp + r0, 1, false)?,
        }
        self.nv_end();
        // index rows on the device: the window part (`window_row`, or the decode ring oldest slot
        // first), then the selection rows
        // the keys side ran on the second stream under the projection above (a few rows): what
        // follows reads the ring, the compressed rows and the selection it wrote
        self.side_join()?;
        self.nv(c"attn_idx_rows");
        let mut built: Option<Buf> = None;
        let idx = if let Some(given) = k.idx.as_ref() {
            given.at(&st, r0 * tot * 4)
        } else {
            let built = built.insert(self.ops.tmp(rows * tot * 4)?);
            let sel = if tw > 0 {
                sh.topk
                    .as_ref()
                    .ok_or("attention: no selection from the index source")?
                    .p(&st)
            } else {
                Dp::NULL
            };
            if k.dec_rows {
                ck("attn_idx_rows", unsafe {
                    ffi::memra_v41_attn_idx_rows(
                        sel.i(),
                        tw as i32,
                        r0 as i32,
                        rows as i32,
                        sp as i32,
                        win as i32,
                        tot as i32,
                        built.p(&st).im(),
                        self.ops.sv(),
                    )
                })?;
            } else {
                ck("attn_idx", unsafe {
                    ffi::memra_v41_attn_idx(
                        sel.i(),
                        tw as i32,
                        r0 as i32,
                        rows as i32,
                        sp as i32,
                        win as i32,
                        ww as i32,
                        tot as i32,
                        built.p(&st).im(),
                        self.ops.sv(),
                    )
                })?;
            }
            built.p(&st)
        };
        self.nv_end();
        self.nv(c"sparse_attn");
        match k.keys_at.or(k.keys.as_ref().map(|t| t.p(&st))) {
            // the rows' keys where they live: each row's table entry names its window rows and
            // its compressed rows
            Some(t) => self.ops.sparse_attn_keys(
                q.p(&st),
                qdim,
                Dp(t.addr() + (r0 * 24) as u64),
                idx,
                tot,
                tot,
                lw.sink.p(&st),
                scale,
                o.p(&st),
                qdim,
                rows,
                heads,
            )?,
            None => self.ops.sparse_attn(
                q.p(&st),
                qdim,
                k.kv_all.p(&st),
                idx,
                tot,
                tot,
                lw.sink.p(&st),
                scale,
                o.p(&st),
                qdim,
                rows,
                heads,
            )?,
        }
        self.nv_end();
        if self.trace && single {
            tr.qr = Some(clone_buf(&self.ops, &k.qr)?);
            tr.q = Some(clone_buf(&self.ops, q)?);
            tr.kv_all = Some(clone_buf(&self.ops, &k.kv_all)?);
            let ib = self.ops.tmp(rows * tot * 4)?;
            match (k.idx.as_ref(), built.as_ref()) {
                (Some(given), _) => {
                    copy_d2d(&self.ops, given, r0 * tot * 4, &ib, 0, rows * tot * 4)?
                }
                (None, Some(b)) => copy_d2d(&self.ops, b, 0, &ib, 0, rows * tot * 4)?,
                (None, None) => unreachable!("the index rows are given or built"),
            }
            tr.idx_all = Some(ib);
            tr.attn_o = Some(clone_buf(&self.ops, o)?);
        }
        self.nv(c"wo_a");
        match k.pos {
            Some(p) => self.ops.rope_pos(
                o.p(&st),
                qdim,
                rows,
                heads,
                hd,
                table,
                Dp(p.addr() + (r0 * 4) as u64),
                true,
            )?,
            None => self
                .ops
                .rope(o.p(&st), qdim, rows, heads, hd, table, sp + r0, 1, true)?,
        }
        self.ops.fp8_qdq(o.p(&st), qdim, rows, qdim)?;
        self.ops.gemm_bf16(
            o.p(&st),
            qdim,
            gk,
            lw.wo_a.w.p(&st),
            gk,
            g.o_lora * gk,
            Dp(oa.addr() + (lw.group0 * g.o_lora * 2) as u64),
            oa_ld,
            g.o_lora,
            false,
            rows,
            g.o_lora,
            gk,
            lw.groups_l,
        )?;
        self.nv_end();
        Ok(())
    }

    /// `wo_b` over the gathered `oa` rows (pitch `oa_ld` elements): this card's output rows
    /// `[lw.dim0, lw.dim0 + lw.wo_b.n)` written at that column of `out` (row pitch `ldc` elements).
    fn attn_wo_b(
        &mut self,
        lw: &LayerW,
        oa: Dp,
        oa_ld: usize,
        rows: usize,
        out: Dp,
        ldc: usize,
    ) -> Result<(), String> {
        self.nv(c"wo_b");
        let r = self.ops.linear_q(
            oa,
            oa_ld,
            rows,
            &lw.wo_b,
            Dp(out.addr() + (lw.dim0 * 2) as u64),
            ldc,
            false,
            &mut self.scr,
        );
        self.nv_end();
        r
    }

    /// MoE (vendor `MoE.forward`): the sqrtsoftplus gate with its selection bias, routed MXFP4
    /// (or minted NVFP4) experts, the shared FP8 expert, bf16 out. The dispatch is planned on the
    /// device and runs as one grouped launch per projection; each token's expert rows are added in
    /// ascending expert id into an fp32 zero, then the shared expert, then rounded, which is the
    /// per-expert loop's order (the N-rank gates compare the two byte for byte). Nothing here waits
    /// on the host: every buffer is stream-ordered.
    pub fn moe(
        &mut self,
        lw: &LayerW,
        x: Dp,
        s: usize,
        out: Dp,
        tr: &mut Trace,
    ) -> Result<(), String> {
        if lw.experts.is_empty() {
            return Err(
                "the layer holds no routed experts on the card; cached experts run in dsv41_gpu::tp"
                    .into(),
            );
        }
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let d = g.dim;
        // one-token forward: the shared expert (it reads only `x`) on the side stream, under the
        // router's chain and the plan; the join is before the combine, the first reader of it
        let side_shared = if s <= SIDE_ROWS {
            if self.side.is_none() {
                self.side = Some(Side::new(&st)?);
            }
            self.nv(c"shared_expert");
            let o = {
                let side = self.side.as_mut().unwrap();
                side.fork(&st)
                    .and_then(|()| tp::shared_expert_on(&side.ops, &mut side.scr, &g, lw, x, s))
            };
            self.nv_end();
            Some(o?)
        } else {
            None
        };
        self.nv(c"router");
        let (sel, selw) = self.moe_route(lw, x, s)?;
        self.nv_end();
        if self.trace {
            tr.route_sel.extend_from_slice(&bytes_i32(&sel.host(&st)?));
            tr.route_w.extend_from_slice(&bytes_f32(&selw.host(&st)?));
        }
        self.nv(c"moe_plan");
        let ex = lw.expert0..lw.expert0 + lw.experts.len();
        let plan = self.moe_plan(&sel, &selw, s, ex)?;
        self.nv_end();
        self.nv(c"experts");
        let slots = self.ops.tmp(plan.np * d * 2)?;
        self.experts_tiles(lw, x, &plan, 0, plan.np, slots.p(&st), None)?;
        self.nv_end();
        let shared = match side_shared {
            Some(b) => {
                self.side.as_ref().unwrap().join(&st)?;
                b
            }
            None => {
                self.nv(c"shared_expert");
                let b = self.shared_expert(lw, x, s);
                self.nv_end();
                b?
            }
        };
        self.nv(c"combine");
        ck("moe_combine", unsafe {
            ffi::memra_v41_moe_combine(
                slots.p(&st).c(),
                d as i64,
                plan.order.p(&st).i(),
                g.topk as i32,
                shared.p(&st).c(),
                d as i64,
                out.m(),
                d as i64,
                s as i32,
                d as i32,
                self.ops.sv(),
            )
        })?;
        self.nv_end();
        if self.trace {
            let err = bytes_i32(&plan.err.host(&st)?)[0];
            if err != 0 {
                return Err(format!("route: {err} expert ids out of range"));
            }
        }
        Ok(())
    }

    /// The router of `rows` rows of `x` (the fp32 gate linear, then sqrtsoftplus with the selection
    /// bias, normalized top-k weights): `(sel i32 [rows][topk], selw f32 [rows][topk])` on the device.
    pub fn moe_route(&mut self, lw: &LayerW, x: Dp, rows: usize) -> Result<(Buf, Buf), String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let ne = g.n_experts;
        let raw = self.ops.tmp(rows * ne * 4)?;
        self.ops
            .linear_bf16(x, g.dim, rows, &lw.gate_w, raw.p(&st), ne, true)?;
        let sel = self.ops.tmp(rows * g.topk * 4)?;
        let selw = self.ops.tmp(rows * g.topk * 4)?;
        ck("route", unsafe {
            ffi::memra_v41_route(
                raw.p(&st).f(),
                ne as i64,
                lw.gate_b.p(&st).f(),
                ne as i32,
                g.topk as i32,
                1,
                g.route_scale,
                sel.p(&st).im(),
                selw.p(&st).fm(),
                rows as i32,
                self.ops.sv(),
            )
        })?;
        Ok((sel, selw))
    }

    /// Tile rows of the grouped expert GEMMs for a slice of `rows` tokens: the small-M kernel
    /// when no expert can hold more than `SMALL_M` rows (a token picks an expert once, so a tile
    /// has at most `rows` rows and the kernel's shared memory is sized for that many), else the
    /// pipelined tile, 128 rows once the mean expert holds that many. Measured both ways on one
    /// card with the padding skipped: 128-row tiles at 4,096 rows (mean 64) are 0.8% slower than
    /// 64-row ones, 64-row tiles at 32,768 rows 3.5% slower than 128-row ones.
    pub fn moe_bm(&self, rows: usize) -> usize {
        if rows <= SMALL_M {
            rows.max(1)
        } else if rows <= WIDE_TILES.load(std::sync::atomic::Ordering::Relaxed) {
            // a batch of 9 to 32 rows: an expert still holds a row or two, so 8-row tiles
            SMALL_M
        } else if rows * self.g.topk >= 128 * self.g.n_experts {
            128
        } else {
            64
        }
    }

    /// The device dispatch plan of a slice from its routing, with the tile list of experts `ex`.
    pub fn moe_plan(
        &mut self,
        sel: &Buf,
        selw: &Buf,
        rows: usize,
        ex: std::ops::Range<usize>,
    ) -> Result<MoePlanDev, String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let (ne, topk) = (g.n_experts, g.topk);
        let np = rows * topk;
        let bm = self.moe_bm(rows);
        // an expert's pairs split into tiles of `bm` rows; with `bm >= rows` an expert is one tile
        let max_tiles = if bm >= rows {
            ex.len().min(np)
        } else {
            np.div_ceil(bm) + ex.len().min(np)
        };
        // a few rows: one kernel writes every output, nothing needs a zero-fill (`V41_PLAN_SMALL`)
        let small = np <= PLAN_SMALL_PAIRS;
        let zeroed = |ops: &Ops, len: usize| if small { ops.tmp(len) } else { ops.buf(len) };
        let counts = self.ops.tmp(if small {
            4
        } else {
            rows.div_ceil(256) * ne * 4
        })?;
        let plan = MoePlanDev {
            rows,
            np,
            bm,
            max_tiles,
            off: self.ops.tmp((ne + 1) * 4)?,
            // zeroed: a pair the placement skips (an out-of-range id) still names token 0
            ptok: zeroed(&self.ops, np * 4)?,
            pw: zeroed(&self.ops, np * 4)?,
            order: self.ops.tmp(np * 4)?,
            tiles: self.ops.tmp(max_tiles * 16)?,
            ntiles: self.ops.tmp(4)?,
            // counted with atomics: starts at zero
            err: zeroed(&self.ops, 4)?,
        };
        ck("moe_plan", unsafe {
            ffi::memra_v41_moe_plan(
                sel.p(&st).i(),
                selw.p(&st).f(),
                rows as i32,
                topk as i32,
                ne as i32,
                bm as i32,
                ex.start as i32,
                ex.end as i32,
                counts.p(&st).im(),
                plan.off.p(&st).im(),
                plan.ptok.p(&st).im(),
                plan.pw.p(&st).fm(),
                plan.order.p(&st).im(),
                plan.tiles.p(&st).m(),
                plan.ntiles.p(&st).im(),
                plan.err.p(&st).im(),
                self.ops.sv(),
            )
        })?;
        Ok(plan)
    }

    /// The `V41ExpW` table of this rank's expert range `er` for a layer whose routed experts live
    /// in the rank's [`experts::ExpertCache`]: the record of each expert of `need` (the experts
    /// with rows, ascending) points at its cache slot `bases[i]` (the checkpoint's MXFP4 bytes in
    /// [`experts::Layout`] order); an expert without rows has no tile and its record stays zero.
    pub fn cached_expert_table(
        &mut self,
        er: std::ops::Range<usize>,
        need: &[usize],
        bases: &[u64],
    ) -> Result<Buf, String> {
        if need.len() != bases.len() {
            return Err(format!(
                "{} cached expert slots for {} experts with rows",
                bases.len(),
                need.len()
            ));
        }
        let lay = experts::Layout::of(&self.g);
        let mut t = vec![0u8; er.len() * 64];
        for (&e, &b) in need.iter().zip(bases) {
            if !er.contains(&e) {
                return Err(format!("expert {e} is not in this rank's range {er:?}"));
            }
            let rec = &mut t[(e - er.start) * 64..(e - er.start + 1) * 64];
            // weights w1, w2, w3, then their scales, then the three second-level scales (1.0)
            for (i, k) in [0usize, 4, 2, 1, 5, 3].into_iter().enumerate() {
                rec[i * 8..i * 8 + 8].copy_from_slice(&(b + lay.off[k] as u64).to_le_bytes());
            }
            for i in 0..3 {
                rec[48 + i * 4..52 + i * 4].copy_from_slice(&1.0f32.to_le_bytes());
            }
        }
        Buf::from_bytes(&self.ops.st, &t)
    }

    /// The routed experts of `lw` over pairs `[p0, p1)` of a device plan whose tile list holds
    /// exactly this rank's experts (all pairs at N = 1): the pairs' rows of `x` quantized, `w1` and
    /// `w3` as one tile-list launch each, the weighted clamped SwiGLU, `w2` into `slots` (bf16
    /// `[np][dim]`, each pair's row at its plan index). Per row, the per-expert loop's arithmetic.
    /// A caller that does not know its pair range passes all of `[0, np)`: the tile lists still
    /// name only this rank's pairs, so the other pairs' rows of the temporaries hold whatever the
    /// buffers held and their `slots` rows are not written.
    #[allow(clippy::too_many_arguments)]
    pub fn experts_tiles(
        &mut self,
        lw: &LayerW,
        x: Dp,
        plan: &MoePlanDev,
        p0: usize,
        p1: usize,
        slots: Dp,
        cached: Option<&Buf>,
    ) -> Result<(), String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let (d, inter, np) = (g.dim, g.moe_inter, plan.np);
        let n = p1 - p0;
        if n == 0 {
            return Ok(());
        }
        let wfmt = match (cached, lw.experts.first()) {
            // the cache holds the checkpoint's MXFP4 bytes as stored
            (Some(_), _) => 1,
            (None, Some(ws)) => match ws[0].fmt {
                QFmt::Mxfp4 => 1,
                QFmt::Nvfp4Minted { .. } => 2,
                QFmt::Fp8Block32 => return Err("routed experts are FP4".into()),
            },
            (None, None) => {
                return Err(
                    "the layer holds no routed experts and no cached table was given".into(),
                );
            }
        };
        let exp_tab = cached.unwrap_or(&lw.exp_tab);
        let kdiv = if wfmt == 1 { 32 } else { 16 };
        // many rows, all pairs: the input is quantized once a token and each tile reads its
        // pairs' tokens through the plan's pair-to-token list (a sixth of the quantization of
        // the pairs' gathered copies, and the same codes); else the pairs' rows are gathered
        let by_token = plan.bm > SMALL_M && p0 == 0 && p1 == np;
        let rows = np / g.topk;
        let (aq, as_) = if by_token {
            (self.ops.tmp(rows * d)?, self.ops.tmp(rows * d / 32)?)
        } else {
            (self.ops.tmp(np * d)?, self.ops.tmp(np * d / 32)?)
        };
        if by_token {
            self.ops
                .act_quant(x, d, Dp::NULL, rows, d, aq.p(&st), as_.p(&st))?;
        } else {
            self.ops.act_quant(
                x,
                d,
                plan.ptok.at(&st, p0 * 4),
                n,
                d,
                aq.at(&st, p0 * d),
                as_.at(&st, p0 * d / 32),
            )?;
        }
        let gb = self.ops.tmp(np * inter * 2)?;
        let ub = self.ops.tmp(np * inter * 2)?;
        let hb = self.ops.tmp(np * inter * 2)?;
        let tiles = |ops: &Ops, a: &Buf, a_s: &Buf, k: usize, proj: i32, c: Dp, nout: usize| {
            ck("gemm_f8_tiles", unsafe {
                ffi::memra_v41_gemm_f8_tiles(
                    a.p(&st).c(),
                    k as i64,
                    a_s.p(&st).c(),
                    (k / 32) as i64,
                    plan.tiles.p(&st).c(),
                    plan.ntiles.p(&st).i(),
                    plan.max_tiles as i32,
                    exp_tab.p(&st).c(),
                    proj,
                    (k / 2) as i64,
                    (k / kdiv) as i64,
                    wfmt,
                    c.m(),
                    nout as i64,
                    0,
                    nout as i32,
                    k as i32,
                    plan.bm as i32,
                    ops.sv(),
                )
            })
        };
        let tiles_rows = |ops: &Ops, proj: i32, c: Dp| {
            ck("gemm_f8_tiles_rows", unsafe {
                ffi::memra_v41_gemm_f8_tiles_rows(
                    aq.p(&st).c(),
                    d as i64,
                    as_.p(&st).c(),
                    (d / 32) as i64,
                    plan.ptok.p(&st).i(),
                    plan.tiles.p(&st).c(),
                    plan.ntiles.p(&st).i(),
                    plan.max_tiles as i32,
                    exp_tab.p(&st).c(),
                    proj,
                    (d / 2) as i64,
                    (d / kdiv) as i64,
                    wfmt,
                    c.m(),
                    inter as i64,
                    0,
                    inter as i32,
                    d as i32,
                    plan.bm as i32,
                    ops.sv(),
                )
            })
        };
        if by_token {
            tiles_rows(&self.ops, 0, gb.p(&st))?;
            tiles_rows(&self.ops, 2, ub.p(&st))?;
        } else {
            tiles(&self.ops, &aq, &as_, d, 0, gb.p(&st), inter)?;
            tiles(&self.ops, &aq, &as_, d, 2, ub.p(&st), inter)?;
        }
        ck("swiglu", unsafe {
            ffi::memra_v41_swiglu(
                gb.at(&st, p0 * inter * 2).m(),
                ub.at(&st, p0 * inter * 2).m(),
                inter as i64,
                plan.pw.at(&st, p0 * 4).f(),
                g.swiglu_limit,
                hb.at(&st, p0 * inter * 2).m(),
                inter as i64,
                n as i32,
                inter as i32,
                self.ops.sv(),
            )
        })?;
        let aq2 = self.ops.tmp(np * inter)?;
        let as2 = self.ops.tmp(np * inter / 32)?;
        self.ops.act_quant(
            hb.at(&st, p0 * inter * 2),
            inter,
            Dp::NULL,
            n,
            inter,
            aq2.at(&st, p0 * inter),
            as2.at(&st, p0 * inter / 32),
        )?;
        tiles(&self.ops, &aq2, &as2, inter, 1, slots, d)
    }

    /// Single-pass mHC mixes of the stream `x` `[s][hc][dim]`: `(mixes, pre, post, comb)`.
    pub fn hc_mix(
        &mut self,
        fn_: &Buf,
        sc: &Buf,
        base: &Buf,
        x: &Buf,
        s: usize,
    ) -> Result<(Buf, Buf, Buf, Buf), String> {
        hc_mix_on(&self.ops, &self.g, fn_, sc, base, x, s)
    }

    /// [`Self::hc_mix`] of a few rows on the card's second stream, behind everything issued on
    /// the main stream so far. Nothing between here and the mixes' first reader needs them (the
    /// collapse before a sublayer uses the mix carried from the one before), so they run under
    /// the sublayer; [`Self::side_join`] before that reader. More rows: the main stream.
    pub fn hc_mix_under(
        &mut self,
        fn_: &Buf,
        sc: &Buf,
        base: &Buf,
        x: &Buf,
        s: usize,
    ) -> Result<(Buf, Buf, Buf, Buf), String> {
        if s > SIDE_ROWS {
            return self.hc_mix(fn_, sc, base, x, s);
        }
        let st = self.ops.st.clone();
        if self.side.is_none() {
            self.side = Some(Side::new(&st)?);
        }
        let side = self.side.as_ref().unwrap();
        side.fork(&st)?;
        hc_mix_on(&side.ops, &self.g, fn_, sc, base, x, s)
    }

    /// From here to [`Self::side_end`] this rank issues on its second stream: the branch starts
    /// after everything issued on the main stream so far, and the rank's kernels and scratch are
    /// exchanged with the branch's, so every launch and temporary in between is the branch's.
    /// What the branch writes is read after [`Self::side_join`].
    pub fn side_begin(&mut self) -> Result<(), String> {
        let st = self.ops.st.clone();
        if self.side.is_none() {
            self.side = Some(Side::new(&st)?);
        }
        let side = self.side.as_mut().unwrap();
        side.fork(&st)?;
        std::mem::swap(&mut self.ops, &mut side.ops);
        std::mem::swap(&mut self.scr, &mut side.scr);
        Ok(())
    }

    /// Back on the main stream after [`Self::side_begin`] (the branch is not joined here).
    pub fn side_end(&mut self) {
        let side = self
            .side
            .as_mut()
            .expect("side_begin made the second stream");
        std::mem::swap(&mut self.ops, &mut side.ops);
        std::mem::swap(&mut self.scr, &mut side.scr);
    }

    /// The main stream waits for everything issued on the second stream (a no-op without one).
    pub fn side_join(&self) -> Result<(), String> {
        match &self.side {
            Some(s) => s.join(&self.ops.st),
            None => Ok(()),
        }
    }

    /// The partial pass of [`Self::hc_mix`] for the virtual warps `[b0, b0 + nb)` of 8: the sums
    /// land in their slots of `parts` (`[s][8][mixes + 1]` f32), the other slots are a peer's.
    pub fn hc_mix_part(
        &mut self,
        fn_: &Buf,
        x: &Buf,
        s: usize,
        b0: usize,
        nb: usize,
    ) -> Result<Buf, String> {
        let g = &self.g;
        let st = self.ops.st.clone();
        let (hc, d) = (g.hc, g.dim);
        let parts = self.ops.tmp(s * 8 * ((2 + hc) * hc + 1) * 4)?;
        ck("hc_mixes_part", unsafe {
            ffi::memra_v41_hc_mixes_part(
                x.p(&st).c(),
                fn_.p(&st).f(),
                s as i32,
                hc as i32,
                d as i32,
                b0 as i32,
                nb as i32,
                parts.p(&st).fm(),
                self.ops.sv(),
            )
        })?;
        Ok(parts)
    }

    /// The finish of [`Self::hc_mix`] from all 8 virtual warps' sums: `(mixes, pre, post, comb)`.
    pub fn hc_mix_finish(
        &mut self,
        parts: &Buf,
        sc: &Buf,
        base: &Buf,
        s: usize,
    ) -> Result<(Buf, Buf, Buf, Buf), String> {
        let g = &self.g;
        let st = self.ops.st.clone();
        let (hc, d) = (g.hc, g.dim);
        let nmix = (2 + hc) * hc;
        let mixes = self.ops.tmp(s * nmix * 4)?;
        let pre = self.ops.tmp(s * hc * 4)?;
        let post = self.ops.tmp(s * hc * 4)?;
        let comb = self.ops.tmp(s * hc * hc * 4)?;
        ck("hc_mixes_finish", unsafe {
            ffi::memra_v41_hc_mixes_finish(
                parts.p(&st).f(),
                sc.p(&st).f(),
                base.p(&st).f(),
                s as i32,
                hc as i32,
                d as i32,
                g.sinkhorn_iters as i32,
                g.norm_eps,
                g.hc_eps,
                mixes.p(&st).fm(),
                pre.p(&st).fm(),
                post.p(&st).fm(),
                comb.p(&st).fm(),
                self.ops.sv(),
            )
        })?;
        Ok((mixes, pre, post, comb))
    }

    /// The collapse of the stream `h` with `mix` `[s][hc]`, then RMSNorm with `norm`: `[s][dim]`.
    pub fn hc_pre_norm(&mut self, h: &Buf, mix: &Buf, norm: &Buf, s: usize) -> Result<Buf, String> {
        let g = &self.g;
        let st = self.ops.st.clone();
        let (hc, d) = (g.hc, g.dim);
        let x = self.ops.tmp(s * d * 2)?;
        ck("hc_pre", unsafe {
            ffi::memra_v41_hc_pre(
                h.p(&st).m(),
                mix.p(&st).f(),
                s as i32,
                hc as i32,
                d as i32,
                x.p(&st).m(),
                d as i64,
                self.ops.sv(),
            )
        })?;
        self.ops
            .rmsnorm(x.p(&st), d, norm, x.p(&st), d, s, d, g.norm_eps)?;
        Ok(x)
    }

    /// The expansion of a sublayer output `f` `[s][dim]` into the residual stream `res`.
    pub fn hc_post(
        &mut self,
        f: &Buf,
        res: &Buf,
        post: &Buf,
        comb: &Buf,
        s: usize,
    ) -> Result<Buf, String> {
        let g = &self.g;
        let st = self.ops.st.clone();
        let (hc, d) = (g.hc, g.dim);
        let out = self.ops.tmp(s * hc * d * 2)?;
        ck("hc_post", unsafe {
            ffi::memra_v41_hc_post(
                f.p(&st).m(),
                d as i64,
                res.p(&st).m(),
                post.p(&st).f(),
                comb.p(&st).f(),
                s as i32,
                hc as i32,
                d as i32,
                out.p(&st).m(),
                self.ops.sv(),
            )
        })?;
        Ok(out)
    }

    /// [`Self::hc_post`] written over `res` itself (`h` `[s][hc][dim]`). The kernel reads a
    /// column's `hc` residual values into registers before it writes that column's `hc` outputs,
    /// and no other thread touches the column, so the bytes are those of a separate output
    /// buffer; the stream then needs one `[s][hc][dim]` buffer per layer instead of three.
    pub fn hc_post_in_place(
        &mut self,
        f: &Buf,
        res: &Buf,
        post: &Buf,
        comb: &Buf,
        s: usize,
    ) -> Result<(), String> {
        let g = &self.g;
        let st = self.ops.st.clone();
        let (hc, d) = (g.hc, g.dim);
        if res.len < s * hc * d * 2 {
            return Err("hc_post_in_place: stream buffer too small".into());
        }
        ck("hc_post", unsafe {
            ffi::memra_v41_hc_post(
                f.p(&st).m(),
                d as i64,
                res.p(&st).m(),
                post.p(&st).f(),
                comb.p(&st).f(),
                s as i32,
                hc as i32,
                d as i32,
                res.p(&st).m(),
                self.ops.sv(),
            )
        })
    }

    /// One trunk block (vendor `Block.forward`) for `s` tokens from position `sp`: a prefill chunk
    /// at 0 or one decode token. `h` is the bf16 stream `[s][hc][dim]` after this layer's Engram
    /// (if any), `pre_mix` the carried f32 `[s][hc]` from the previous FFN; `ls` is this layer's
    /// own window ring and compressor state, `sh` the CSA2 state handed down the stack. Returns the
    /// new stream and the FFN's pre-mix for the next layer.
    #[allow(clippy::too_many_arguments)]
    pub fn block_forward(
        &mut self,
        lw: &LayerW,
        h: &Buf,
        pre_mix: &Buf,
        sp: usize,
        s: usize,
        sh: &mut Shared,
        ls: &mut LayerState,
        tr: &mut Trace,
    ) -> Result<(Buf, Buf), String> {
        if lw.part.n != 1 {
            return Err(
                "block_forward runs a whole layer; a rank's share runs in dsv41_gpu::tp".into(),
            );
        }
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let d = g.dim;
        self.nv(c"hc_attn_pre");
        let (amix, a_pre, a_post, a_comb) =
            self.hc_mix(&lw.hc_attn_fn, &lw.hc_attn_scale, &lw.hc_attn_base, h, s)?;
        let x = self.hc_pre_norm(h, pre_mix, &lw.attn_norm, s)?;
        self.nv_end();
        if self.trace {
            tr.attn_mix = Some(amix);
            tr.attn_in = Some(clone_buf(&self.ops, &x)?);
        }
        let a = self.attention(lw, &x, sp, s, sh, ls, tr)?;
        self.nv(c"hc_ffn_pre");
        let h1 = self.hc_post(&a, h, &a_post, &a_comb, s)?;
        let (_fmix, f_pre, f_post, f_comb) =
            self.hc_mix(&lw.hc_ffn_fn, &lw.hc_ffn_scale, &lw.hc_ffn_base, &h1, s)?;
        let x2 = self.hc_pre_norm(&h1, &a_pre, &lw.ffn_norm, s)?;
        self.nv_end();
        // the MoE runs `chunk` tokens at a time: each token's experts still add in ascending id order
        let m = self.ops.tmp(s * d * 2)?;
        // larger slices here: the per-expert loop runs once per slice
        let cm = decode::moe_slice(self.chunk).min(s).max(1);
        for r0 in (0..s).step_by(cm) {
            let rows = cm.min(s - r0);
            self.moe(lw, x2.at(&st, r0 * d * 2), rows, m.at(&st, r0 * d * 2), tr)?;
        }
        self.nv(c"hc_post");
        let h2 = self.hc_post(&m, &h1, &f_post, &f_comb, s)?;
        self.nv_end();
        if self.trace {
            tr.attn_out = Some(a);
            tr.mid = Some(h1);
            tr.moe_in = Some(x2);
            tr.moe_out = Some(m);
        }
        // no drain: every buffer this layer made is freed stream-ordered and nothing waits on the host
        Ok((h2, f_pre))
    }
}

fn clone_buf(ops: &Ops, b: &Buf) -> Result<Buf, String> {
    let o = ops.tmp(b.len)?;
    copy_d2d(ops, b, 0, &o, 0, b.len)?;
    Ok(o)
}

/// `*tab[r] = src[r]` for `rows` rows of `words` 16-bit words (`src` row pitch `lds` words; `tab`
/// a device array of row addresses, a null one skipped): one launch for the writes of every
/// row of a batched decode call (`memra_v41_put_rows`).
pub fn put_rows(
    ops: &Ops,
    src: Dp,
    lds: usize,
    tab: Dp,
    rows: usize,
    words: usize,
) -> Result<(), String> {
    ck("put_rows", unsafe {
        ffi::memra_v41_put_rows(
            src.c(),
            lds as i64,
            tab.c(),
            rows as i32,
            words as i32,
            ops.sv(),
        )
    })
}

/// Single-pass mHC mixes of the stream `x` `[s][hc][dim]` on `ops`' stream: `(mixes, pre, post, comb)`.
pub fn hc_mix_on(
    ops: &Ops,
    g: &Geo,
    fn_: &Buf,
    sc: &Buf,
    base: &Buf,
    x: &Buf,
    s: usize,
) -> Result<(Buf, Buf, Buf, Buf), String> {
    let st = ops.st.clone();
    let (hc, d) = (g.hc, g.dim);
    let nmix = (2 + hc) * hc;
    let mixes = ops.tmp(s * nmix * 4)?;
    let pre = ops.tmp(s * hc * 4)?;
    let post = ops.tmp(s * hc * 4)?;
    let comb = ops.tmp(s * hc * hc * 4)?;
    // the partial pass's 8 virtual-warp sums of each mix and of the square sum
    let parts = ops.tmp(s * 8 * (nmix + 1) * 4)?;
    ck("hc_mixes", unsafe {
        ffi::memra_v41_hc_mixes(
            x.p(&st).m(),
            fn_.p(&st).f(),
            sc.p(&st).f(),
            base.p(&st).f(),
            s as i32,
            hc as i32,
            d as i32,
            g.sinkhorn_iters as i32,
            g.norm_eps,
            g.hc_eps,
            parts.p(&st).fm(),
            mixes.p(&st).fm(),
            pre.p(&st).fm(),
            post.p(&st).fm(),
            comb.p(&st).fm(),
            ops.sv(),
        )
    })?;
    Ok((mixes, pre, post, comb))
}

/// `dst[doff..doff+n] = src[soff..soff+n]` on the stream.
pub fn copy_d2d(
    ops: &Ops,
    src: &Buf,
    soff: usize,
    dst: &Buf,
    doff: usize,
    n: usize,
) -> Result<(), String> {
    if soff + n > src.len || doff + n > dst.len {
        return Err(format!("copy_d2d: {n} bytes out of range"));
    }
    if n == 0 {
        return Ok(());
    }
    tally::hit("copy_d2d");
    let r = unsafe {
        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
            dst.p(&ops.st).addr() + doff as u64,
            src.p(&ops.st).addr() + soff as u64,
            n,
            ops.sv() as cudarc::driver::sys::CUstream,
        )
    };
    if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("memcpy d2d: {r:?}"));
    }
    Ok(())
}

/// Copy `rows` rows of `width` bytes from `src` (pitch `spitch`, starting at row `src_row`) into `dst` at byte column
/// `dcol` (pitch `dpitch`).
#[allow(clippy::too_many_arguments)]
pub fn copy_2d(
    ops: &Ops,
    src: &Buf,
    src_row: usize,
    spitch: usize,
    dst: &Buf,
    dcol: usize,
    dpitch: usize,
    width: usize,
    rows: usize,
) -> Result<(), String> {
    if rows == 0 || width == 0 {
        return Ok(());
    }
    use cudarc::driver::sys::*;
    let p = CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: src_row,
        srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(),
        srcDevice: src.p(&ops.st).addr(),
        srcArray: std::ptr::null_mut(),
        srcPitch: spitch,
        dstXInBytes: dcol,
        dstY: 0,
        dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(),
        dstDevice: dst.p(&ops.st).addr(),
        dstArray: std::ptr::null_mut(),
        dstPitch: dpitch,
        WidthInBytes: width,
        Height: rows,
    };
    let r = unsafe { cuMemcpy2DAsync_v2(&p, ops.sv() as CUstream) };
    if r != CUresult::CUDA_SUCCESS {
        return Err(format!("memcpy2d: {r:?}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mint_is_exact_and_refuses_what_e4m3_cannot_hold() {
        // 2^-3, 2^4 and 2^-9 relative to a tensor maximum of 2^4: 13 octaves, inside e4m3's 17
        let sc = [124u8, 131, 118];
        let (codes, s2) = mint_mxfp4_scales(&sc).unwrap();
        assert_eq!(codes.len(), 6);
        for (i, &b) in sc.iter().enumerate() {
            assert_eq!(
                codes[2 * i],
                codes[2 * i + 1],
                "both per-16 halves carry the per-32 scale"
            );
            assert_eq!(
                e4m3_to_f32(codes[2 * i]) * s2,
                e8m0_to_f32(b),
                "scale {b} survives the mint"
            );
        }
        // 18 octaves of spread cannot be one e4m3 power of two each
        assert!(mint_mxfp4_scales(&[100u8, 118]).is_err());
        assert!(mint_mxfp4_scales(&[0xFF]).is_err());
    }

    #[test]
    fn verify_rows_read_the_positions_the_one_token_calls_read() {
        // the ring as positions: slot j holds the newest position below `upto` with p % win == j
        let ring_at = |upto: usize, win: usize| -> Vec<Option<usize>> {
            (0..win)
                .map(|j| (0..upto).rev().find(|p| p % win == j))
                .collect()
        };
        for win in [4usize, 8, 128] {
            for sp in [
                1usize,
                2,
                3,
                win - 1,
                win,
                win + 1,
                2 * win - 2,
                2 * win + 3,
                5 * win,
            ] {
                for s in 2..=6usize {
                    let before = ring_at(sp, win);
                    for i in 0..s {
                        let p = sp + i;
                        // the one-token call at p: the ring after its own write, in its row order
                        let after = ring_at(p + 1, win);
                        let mut one = vec![0i32; win];
                        Ctx::window_row_decode(p, win, &mut one);
                        let want: Vec<Option<usize>> = one
                            .iter()
                            .map(|&x| (x >= 0).then(|| after[x as usize].unwrap()))
                            .collect();
                        let mut row = vec![0i32; win];
                        Ctx::window_row_verify(sp, i, win, &mut row);
                        let got: Vec<Option<usize>> = row
                            .iter()
                            .map(|&x| {
                                (x >= 0).then(|| {
                                    let x = x as usize;
                                    if x >= win {
                                        sp + (x - win)
                                    } else {
                                        before[x].unwrap()
                                    }
                                })
                            })
                            .collect();
                        assert_eq!(got, want, "win {win} sp {sp} rows {s} row {i}");
                        // a row never reads a row after it
                        assert!(row.iter().all(|&x| x < (win + i + 1) as i32));
                    }
                }
            }
        }
    }

    #[test]
    fn decode_rows_read_the_ring_oldest_slot_first() {
        // vendor: oldest = start_pos % win + 1; cat(arange(oldest, win), arange(oldest)); -1 where > start_pos
        let row = |sp: usize| {
            let mut r = vec![0i32; 8];
            Ctx::window_row_decode(sp, 8, &mut r);
            r
        };
        // the ring still filling: empty slots first, then positions 0..=sp in order
        assert_eq!(row(5), vec![-1, -1, 0, 1, 2, 3, 4, 5]);
        assert_eq!(row(7), vec![0, 1, 2, 3, 4, 5, 6, 7]);
        // wrapped: slots 3..7 then 0..2 hold positions 3..10, oldest first
        assert_eq!(row(10), vec![3, 4, 5, 6, 7, 0, 1, 2]);
        assert_eq!(row(15), vec![0, 1, 2, 3, 4, 5, 6, 7]);
        // the prefill row of the same position reads the same keys in another order
        let mut p = vec![0i32; 6];
        Ctx::window_row(5, 8, &mut p);
        assert_eq!(p, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn the_selection_width_of_a_step_is_one_of_a_few_classes() {
        assert_eq!(select_class(0, 512), 0);
        assert_eq!(select_class(1, 512), 64);
        assert_eq!(select_class(64, 512), 64);
        assert_eq!(select_class(65, 512), 128);
        assert_eq!(select_class(200, 512), 256);
        assert_eq!(select_class(257, 512), 512);
        assert_eq!(select_class(512, 512), 512);
        // a class never passes the model's top-k, and never falls under what the rows hold
        assert_eq!(select_class(300, 384), 384);
        for most in 1..=512usize {
            let c = select_class(most, 512);
            assert!(c >= most && c <= 512);
        }
    }

    #[test]
    fn window_rows_match_the_reference_prefill_selection() {
        let (v, w) = Ctx::window_idx(5, 3);
        assert_eq!(w, 3);
        assert_eq!(v, vec![0, -1, -1, 0, 1, -1, 0, 1, 2, 1, 2, 3, 2, 3, 4]);
    }
}
