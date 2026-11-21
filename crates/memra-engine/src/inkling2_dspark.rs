//! Inkling 2 Mini's DSpark drafter on one GPU (`cu/inkling2_dspark.cu`): the plan's
//! `Inkling2DrafterPlan` with BF16 weights as stored and f32 activations, beside the trunk's
//! route (`crate::inkling2_gpu`), with which it shares the token embedding, its norm and the
//! output projection. The reference is memra-reference `inkling2_drafter` (f64).
//!
//! A proposal never changes a committed token: [`crate::inkling2_spec`] verifies every
//! proposal with the trunk's own forward, so this program's numeric class decides acceptance
//! only. Its context K/V is written once per committed position from the trunk's taps; a
//! rejected row never reaches it.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use memra_reference::inkling2_drafter::{DraftGeometry, DraftSource};

use crate::inkling2_gpu::Dev;

mod ffi {
    use core::ffi::c_void;
    unsafe extern "C" {
        pub fn memra_ink2d_gemv(
            x: *const f32,
            ldx: i64,
            w: *const c_void,
            ldw: i64,
            bias: *const f32,
            y: *mut f32,
            ldy: i64,
            m: i32,
            n: i32,
            k: i32,
            acc: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2d_gemv_f32w(
            x: *const f32,
            w: *const f32,
            ldw: i64,
            bias: *const f32,
            y: *mut f32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2d_mix_norm(
            taps: *const f32,
            tap_stride: i64,
            row_stride: i64,
            ntaps: i32,
            alpha: *const f32,
            w: *const f32,
            eps: f64,
            out: *mut f32,
            rows: i32,
            h: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2d_norm(
            x: *mut f32,
            add: *const f32,
            w: *const f32,
            eps: f64,
            y: *mut f32,
            rows: i32,
            h: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2d_embed(
            table: *const c_void,
            id: *const u32,
            w: *const f32,
            eps: f64,
            out: *mut f32,
            h: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2d_head_rope(
            x: *mut f32,
            ld: i64,
            w: *const f32,
            norm: i32,
            eps: f64,
            pos0: i64,
            theta: f64,
            rows: i32,
            heads: i32,
            d: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2d_attn(
            q: *const f32,
            kb: *const f32,
            vb: *const f32,
            kc: *const f32,
            vc: *const f32,
            cap: i64,
            ring: i32,
            first: i64,
            p0: i64,
            rows: i32,
            heads: i32,
            kv_heads: i32,
            d: i32,
            scale: f32,
            out: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2d_silu_mul(
            g: *const f32,
            u: *const f32,
            y: *mut f32,
            n: i64,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2d_scatter(
            src: *const f32,
            dst: *mut f32,
            cap: i64,
            ring: i32,
            pos0: i64,
            rows: i32,
            width: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2d_markov_in(
            state: *const f32,
            w1: *const c_void,
            prev: *const u32,
            out: *mut f32,
            rank: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2d_markov_gate(
            ctx: *const f32,
            add: *const f32,
            add2: *const f32,
            state: *mut f32,
            latent: *mut f32,
            rank: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2d_cap_pick(
            base: *const f32,
            bias: *const f32,
            cap: f32,
            logits: *mut f32,
            n: i32,
            part_v: *mut f32,
            part_i: *mut u32,
            token: *mut u32,
            prev: *mut u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2d_fold(
            head: *const c_void,
            how: *const c_void,
            hob: *const c_void,
            fold: *mut f32,
            fold_b: *mut f32,
            vocab: i32,
            h: i32,
            rank: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

/// Picks a proposal from its softcapped logits: `(step, logits) -> token`.
pub type Chooser<'a> = dyn FnMut(usize, &[f32]) -> u32 + 'a;

fn p<T>(s: &CudaSlice<T>, st: &CudaStream) -> u64 {
    let (ptr, _g) = s.device_ptr(st);
    ptr
}

fn pm<T>(s: &mut CudaSlice<T>, st: &CudaStream) -> u64 {
    let (ptr, _g) = s.device_ptr_mut(st);
    ptr
}

/// Device addresses of the trunk tensors the drafter reads: the BF16 token embedding
/// `[padded vocab, hidden]`, its f32 norm weight, and the BF16 output projection whose first
/// `vocab` rows get logits.
#[derive(Clone, Copy, Debug)]
pub struct Shared {
    pub embed: u64,
    pub embed_norm: u64,
    pub head: u64,
}

impl Shared {
    pub fn of_route(route: &crate::inkling2_gpu::Route) -> Self {
        let st = route.dev.stream();
        let (embed, norm, head) = route.shared_weights();
        Self {
            embed: p(embed, st),
            embed_norm: p(norm, st),
            head: p(head, st),
        }
    }
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

struct LayerW {
    in_norm: CudaSlice<f32>,
    post_norm: CudaSlice<f32>,
    fuse_norm: CudaSlice<f32>,
    q: CudaSlice<u8>,
    k: CudaSlice<u8>,
    v: CudaSlice<u8>,
    o: CudaSlice<u8>,
    k_ctx: CudaSlice<u8>,
    v_ctx: CudaSlice<u8>,
    q_norm: CudaSlice<f32>,
    k_norm: CudaSlice<f32>,
    k_ctx_norm: CudaSlice<f32>,
    gate: CudaSlice<u8>,
    up: CudaSlice<u8>,
    down: CudaSlice<u8>,
    kc: CudaSlice<f32>,
    vc: CudaSlice<f32>,
    cap: usize,
    ring: bool,
    local: bool,
}

/// The drafter's weights, context K/V and work buffers on the device.
pub struct DraftGpu {
    pub g: DraftGeometry,
    st: Arc<CudaStream>,
    layers: Vec<LayerW>,
    norm: CudaSlice<f32>,
    masks: CudaSlice<f32>,
    alpha: CudaSlice<f32>,
    head_norm: CudaSlice<f32>,
    w1: CudaSlice<u8>,
    rnn_w: CudaSlice<u8>,
    rnn_b: CudaSlice<f32>,
    fold: CudaSlice<f32>,
    fold_b: CudaSlice<f32>,
    /// Committed positions the context holds.
    pub len: usize,
    pub context: usize,
    pub chunk: usize,
    x: CudaSlice<f32>,
    xn: CudaSlice<f32>,
    q: CudaSlice<f32>,
    kb: CudaSlice<f32>,
    vb: CudaSlice<f32>,
    att: CudaSlice<f32>,
    o: CudaSlice<f32>,
    gt: CudaSlice<f32>,
    upb: CudaSlice<f32>,
    act: CudaSlice<f32>,
    dn: CudaSlice<f32>,
    hidden: CudaSlice<f32>,
    feature: CudaSlice<f32>,
    ctxb: CudaSlice<f32>,
    kt: CudaSlice<f32>,
    vt: CudaSlice<f32>,
    ctxr: CudaSlice<f32>,
    fctx: CudaSlice<f32>,
    minp: CudaSlice<f32>,
    raw: CudaSlice<f32>,
    state: CudaSlice<f32>,
    latent: CudaSlice<f32>,
    base: CudaSlice<f32>,
    bias: CudaSlice<f32>,
    /// `[rows - 1, vocab]`: the softcapped logits of every proposal of the last block.
    logits: CudaSlice<f32>,
    part_v: CudaSlice<f32>,
    part_i: CudaSlice<u32>,
    tokens: CudaSlice<u32>,
    prev: CudaSlice<u32>,
    /// Bytes allocated for the drafter (weights, fold, caches, work).
    pub bytes: usize,
    /// Of them, the context K/V (f32; the full layers grow with the context).
    pub kv: usize,
}

fn ck(name: &str, rc: i32) -> Result<(), String> {
    if rc == 0 {
        Ok(())
    } else {
        Err(format!("{name}: rc {rc}"))
    }
}

impl DraftGpu {
    /// Upload the drafter. `context`: positions a request may reach; `chunk`: rows one commit
    /// may hold (the trunk's call size). The fold `head @ hidden_out` is computed here in f64
    /// from the trunk's output projection and kept in f32.
    pub fn load(
        dev: &Dev,
        g: DraftGeometry,
        src: &dyn DraftSource,
        shared: Shared,
        context: usize,
        chunk: usize,
    ) -> Result<Self, String> {
        let before = dev.allocated.get();
        let st = dev.stream().clone();
        let (h, d, kvw, qw, f, r, rows) = (
            g.hidden,
            g.head_dim,
            g.kv_heads * g.head_dim,
            g.heads * g.head_dim,
            g.intermediate,
            g.rank,
            g.rows,
        );
        if g.taps == 0 || rows < 2 || chunk == 0 || context < rows {
            return Err("a drafter with no taps, a block under two rows or no room".into());
        }
        // admission: everything the drafter allocates, against the driver's free memory, before
        // the first upload (a margin of 256 MiB for the allocator and the driver)
        let kv = Self::kv_bytes(&g, context, chunk);
        let per_layer = (qw * h + 4 * kvw * h + h * qw + 3 * f * h) * 2;
        let need = g.layers * per_layer
            + g.padded_vocab * r * 2
            + 3 * r * (2 * r + 2 * h) * 2
            + g.vocab * (r + 1) * 4
            + 2 * (rows - 1) * g.vocab * 4
            + kv
            + chunk.max(rows) * (h + 2 * kvw) * 4
            + rows * (4 * h + 2 * qw + 2 * kvw + 3 * f) * 4;
        let (free, _) = dev.driver_free()?;
        if need + (256 << 20) > free {
            return Err(format!(
                "the drafter needs {need} bytes ({kv} of them context K/V for {context} positions) and the driver has {free} free"
            ));
        }
        let b16 = |name: String, shape: &[usize]| -> Result<CudaSlice<u8>, String> {
            dev.up(&bf16_bytes(&src.tensor(&name, shape)?))
        };
        let f32d = |name: String, shape: &[usize]| -> Result<CudaSlice<f32>, String> {
            dev.up(&src.tensor(&name, shape)?)
        };
        let mut layers = Vec::new();
        for i in 0..g.layers {
            let n = |s: &str| format!("layers.{i}.{s}");
            let local = g.local_layers.contains(&i);
            // a local layer reads the last local_context positions: a ring that also holds one
            // commit of chunk rows
            let (cap, ring) = if local {
                ((g.local_context + chunk).next_power_of_two(), true)
            } else {
                (context, false)
            };
            layers.push(LayerW {
                in_norm: f32d(n("input_layernorm.weight"), &[h])?,
                post_norm: f32d(n("post_attention_layernorm.weight"), &[h])?,
                fuse_norm: f32d(n("dflare_norm.weight"), &[h])?,
                q: b16(n("self_attn.q_proj.weight"), &[qw, h])?,
                k: b16(n("self_attn.k_proj.weight"), &[kvw, h])?,
                v: b16(n("self_attn.v_proj.weight"), &[kvw, h])?,
                o: b16(n("self_attn.o_proj.weight"), &[h, qw])?,
                k_ctx: b16(n("self_attn.k_proj_ctx.weight"), &[kvw, h])?,
                v_ctx: b16(n("self_attn.v_proj_ctx.weight"), &[kvw, h])?,
                q_norm: f32d(n("self_attn.q_norm.weight"), &[d])?,
                k_norm: f32d(n("self_attn.k_norm.weight"), &[d])?,
                k_ctx_norm: f32d(n("self_attn.k_norm_ctx.weight"), &[d])?,
                gate: b16(n("mlp.gate_proj.weight"), &[f, h])?,
                up: b16(n("mlp.up_proj.weight"), &[f, h])?,
                down: b16(n("mlp.down_proj.weight"), &[h, f])?,
                kc: dev.zeros(cap * kvw)?,
                vc: dev.zeros(cap * kvw)?,
                cap,
                ring,
                local,
            });
        }
        // softmax(fuse_weight) per row, in f64
        let fuse = src.tensor("fuse_weight", &[g.layers + 1, g.taps])?;
        let mut alpha = Vec::new();
        for row in fuse.chunks_exact(g.taps) {
            let max = row
                .iter()
                .fold(f64::NEG_INFINITY, |m, v| m.max(f64::from(*v)));
            let e: Vec<f64> = row.iter().map(|v| (f64::from(*v) - max).exp()).collect();
            let s: f64 = e.iter().sum();
            alpha.extend(e.iter().map(|v| (v / s) as f32));
        }
        let mask = src.tensor("mask_embed", &[h])?;
        let masks: Vec<f32> = (0..rows - 1).flat_map(|_| mask.iter().copied()).collect();
        let m = |s: &str| format!("markov_head.{s}");
        let how = b16(m("hidden_out.weight"), &[h, r])?;
        let hob = b16(m("hidden_out.bias"), &[h])?;
        let mut fold = dev.zeros::<f32>(g.vocab * r)?;
        let mut fold_b = dev.zeros::<f32>(g.vocab)?;
        dev.bind()?;
        ck("memra_ink2d_fold", unsafe {
            ffi::memra_ink2d_fold(
                shared.head as *const _,
                p(&how, &st) as *const _,
                p(&hob, &st) as *const _,
                pm(&mut fold, &st) as *mut f32,
                pm(&mut fold_b, &st) as *mut f32,
                g.vocab as i32,
                h as i32,
                r as i32,
                dev.sp(),
            )
        })?;
        dev.sync()?;
        drop((how, hob));
        let wide = chunk.max(rows);
        let me = Self {
            layers,
            norm: f32d("norm.weight".into(), &[h])?,
            masks: dev.up(&masks)?,
            alpha: dev.up(&alpha)?,
            head_norm: f32d(m("dflare_norm.weight"), &[h])?,
            w1: b16(m("markov_w1.weight"), &[g.padded_vocab, r])?,
            rnn_w: b16(m("rnn_proj.weight"), &[3 * r, 2 * r + 2 * h])?,
            rnn_b: f32d(m("rnn_proj.bias"), &[3 * r])?,
            fold,
            fold_b,
            len: 0,
            context,
            chunk,
            x: dev.zeros(rows * h)?,
            xn: dev.zeros(rows * h)?,
            q: dev.zeros(rows * qw)?,
            kb: dev.zeros(rows * kvw)?,
            vb: dev.zeros(rows * kvw)?,
            att: dev.zeros(rows * qw)?,
            o: dev.zeros(rows * h)?,
            gt: dev.zeros(rows * f)?,
            upb: dev.zeros(rows * f)?,
            act: dev.zeros(rows * f)?,
            dn: dev.zeros(rows * h)?,
            hidden: dev.zeros(rows * h)?,
            feature: dev.zeros(h)?,
            ctxb: dev.zeros(wide * h)?,
            kt: dev.zeros(wide * kvw)?,
            vt: dev.zeros(wide * kvw)?,
            ctxr: dev.zeros((rows - 1) * 3 * r)?,
            fctx: dev.zeros(3 * r)?,
            minp: dev.zeros(2 * r)?,
            raw: dev.zeros(3 * r)?,
            state: dev.zeros(r)?,
            latent: dev.zeros(r)?,
            base: dev.zeros((rows - 1) * g.vocab)?,
            bias: dev.zeros(g.vocab)?,
            logits: dev.zeros((rows - 1) * g.vocab)?,
            part_v: dev.zeros(g.vocab.div_ceil(256))?,
            part_i: dev.zeros(g.vocab.div_ceil(256))?,
            tokens: dev.zeros(rows)?,
            prev: dev.zeros(1)?,
            bytes: 0,
            kv: 0,
            st,
            g,
        };
        let mut me = me;
        me.bytes = dev.allocated.get() - before;
        me.kv = Self::kv_bytes(&me.g, context, chunk);
        Ok(me)
    }

    /// Context K/V bytes of a drafter at `context` positions and `chunk` rows per commit.
    pub fn kv_bytes(g: &DraftGeometry, context: usize, chunk: usize) -> usize {
        let kvw = g.kv_heads * g.head_dim;
        (0..g.layers)
            .map(|i| {
                let cap = if g.local_layers.contains(&i) {
                    (g.local_context + chunk).next_power_of_two()
                } else {
                    context
                };
                2 * cap * kvw * 4
            })
            .sum()
    }

    /// Enter the drafter's memory in a route's account.
    pub fn charge(&self, account: &mut crate::inkling2_gpu::MemoryAccount) {
        account.drafter = self.bytes;
        account.drafter_kv = self.kv;
    }

    /// Forget the request.
    pub fn reset(&mut self) {
        self.len = 0;
    }

    #[allow(clippy::too_many_arguments)] // allow: one GEMV over one stored weight
    fn gemv(
        dev: &Dev,
        x: u64,
        ldx: usize,
        w: u64,
        ldw: usize,
        bias: u64,
        y: u64,
        ldy: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        dev.launched("memra_ink2d_gemv", unsafe {
            ffi::memra_ink2d_gemv(
                x as *const f32,
                ldx as i64,
                w as *const _,
                ldw as i64,
                bias as *const f32,
                y as *mut f32,
                ldy as i64,
                m as i32,
                n as i32,
                k as i32,
                0,
                dev.sp(),
            )
        })
    }

    /// Commit `rows` positions (`len..len + rows`) from rows `row0..` of a tap buffer laid out
    /// `[taps, tap_rows, hidden]` (the route's capture), and set the feature of the head from
    /// the last of them.
    pub fn commit(
        &mut self,
        dev: &Dev,
        taps: &CudaSlice<f32>,
        tap_rows: usize,
        row0: usize,
        rows: usize,
    ) -> Result<(), String> {
        if rows == 0 {
            return Ok(());
        }
        if rows > self.chunk || self.len + rows > self.context {
            return Err(format!(
                "a commit of {rows} rows at {}; the drafter holds {} per commit and {} in all",
                self.len, self.chunk, self.context
            ));
        }
        dev.bind()?;
        let st = self.st.clone();
        let g = &self.g;
        let (h, kvw, d) = (g.hidden, g.kv_heads * g.head_dim, g.head_dim);
        let base = p(taps, &st) + (row0 * h * 4) as u64;
        for l in 0..g.layers {
            let lw = &mut self.layers[l];
            dev.launched("memra_ink2d_mix_norm", unsafe {
                ffi::memra_ink2d_mix_norm(
                    base as *const f32,
                    (tap_rows * h) as i64,
                    h as i64,
                    g.taps as i32,
                    (p(&self.alpha, &st) + (l * g.taps * 4) as u64) as *const f32,
                    p(&lw.fuse_norm, &st) as *const f32,
                    g.epsilon,
                    pm(&mut self.ctxb, &st) as *mut f32,
                    rows as i32,
                    h as i32,
                    dev.sp(),
                )
            })?;
            let ctx = p(&self.ctxb, &st);
            Self::gemv(
                dev,
                ctx,
                h,
                p(&lw.k_ctx, &st),
                h,
                0,
                pm(&mut self.kt, &st),
                kvw,
                rows,
                kvw,
                h,
            )?;
            Self::gemv(
                dev,
                ctx,
                h,
                p(&lw.v_ctx, &st),
                h,
                0,
                pm(&mut self.vt, &st),
                kvw,
                rows,
                kvw,
                h,
            )?;
            dev.launched("memra_ink2d_head_rope", unsafe {
                ffi::memra_ink2d_head_rope(
                    pm(&mut self.kt, &st) as *mut f32,
                    kvw as i64,
                    p(&lw.k_ctx_norm, &st) as *const f32,
                    1,
                    g.epsilon,
                    self.len as i64,
                    g.theta,
                    rows as i32,
                    g.kv_heads as i32,
                    d as i32,
                    dev.sp(),
                )
            })?;
            for (src, dst) in [
                (p(&self.kt, &st), pm(&mut lw.kc, &st)),
                (p(&self.vt, &st), pm(&mut lw.vc, &st)),
            ] {
                dev.launched("memra_ink2d_scatter", unsafe {
                    ffi::memra_ink2d_scatter(
                        src as *const f32,
                        dst as *mut f32,
                        lw.cap as i64,
                        i32::from(lw.ring),
                        self.len as i64,
                        rows as i32,
                        kvw as i32,
                        dev.sp(),
                    )
                })?;
            }
        }
        // the head's feature: fuse row `layers` of the last committed row
        let last = base + ((rows - 1) * h * 4) as u64;
        dev.launched("memra_ink2d_mix_norm", unsafe {
            ffi::memra_ink2d_mix_norm(
                last as *const f32,
                (tap_rows * h) as i64,
                h as i64,
                g.taps as i32,
                (p(&self.alpha, &st) + (g.layers * g.taps * 4) as u64) as *const f32,
                p(&self.head_norm, &st) as *const f32,
                g.epsilon,
                pm(&mut self.feature, &st) as *mut f32,
                1,
                h as i32,
                dev.sp(),
            )
        })?;
        self.len += rows;
        Ok(())
    }

    /// The block for `anchor` at position `len`: `hidden` holds its final-norm rows.
    fn block(&mut self, dev: &Dev, shared: Shared, anchor: u32) -> Result<(), String> {
        let st = self.st.clone();
        let g = self.g.clone();
        let (h, kvw, qw, d, f, rows) = (
            g.hidden,
            g.kv_heads * g.head_dim,
            g.heads * g.head_dim,
            g.head_dim,
            g.intermediate,
            g.rows,
        );
        let p0 = self.len;
        dev.put(&[anchor], &mut self.prev)?;
        dev.launched("memra_ink2d_embed", unsafe {
            ffi::memra_ink2d_embed(
                shared.embed as *const _,
                p(&self.prev, &st) as *const u32,
                shared.embed_norm as *const f32,
                g.embed_epsilon,
                pm(&mut self.x, &st) as *mut f32,
                h as i32,
                dev.sp(),
            )
        })?;
        let xp = pm(&mut self.x, &st);
        dev.copy_raw(xp + (h * 4) as u64, p(&self.masks, &st), (rows - 1) * h * 4)?;
        for l in 0..g.layers {
            let lw = &self.layers[l];
            let add = if l == 0 { 0 } else { p(&self.dn, &st) };
            dev.launched("memra_ink2d_norm", unsafe {
                ffi::memra_ink2d_norm(
                    xp as *mut f32,
                    add as *const f32,
                    p(&lw.in_norm, &st) as *const f32,
                    g.epsilon,
                    pm(&mut self.xn, &st) as *mut f32,
                    rows as i32,
                    h as i32,
                    dev.sp(),
                )
            })?;
            let xn = p(&self.xn, &st);
            Self::gemv(
                dev,
                xn,
                h,
                p(&lw.q, &st),
                h,
                0,
                pm(&mut self.q, &st),
                qw,
                rows,
                qw,
                h,
            )?;
            Self::gemv(
                dev,
                xn,
                h,
                p(&lw.k, &st),
                h,
                0,
                pm(&mut self.kb, &st),
                kvw,
                rows,
                kvw,
                h,
            )?;
            Self::gemv(
                dev,
                xn,
                h,
                p(&lw.v, &st),
                h,
                0,
                pm(&mut self.vb, &st),
                kvw,
                rows,
                kvw,
                h,
            )?;
            for (buf, ld, w, heads) in [
                (pm(&mut self.q, &st), qw, p(&lw.q_norm, &st), g.heads),
                (pm(&mut self.kb, &st), kvw, p(&lw.k_norm, &st), g.kv_heads),
            ] {
                dev.launched("memra_ink2d_head_rope", unsafe {
                    ffi::memra_ink2d_head_rope(
                        buf as *mut f32,
                        ld as i64,
                        w as *const f32,
                        1,
                        g.epsilon,
                        p0 as i64,
                        g.theta,
                        rows as i32,
                        heads as i32,
                        d as i32,
                        dev.sp(),
                    )
                })?;
            }
            let first = if lw.local {
                p0.saturating_sub(g.local_context)
            } else {
                0
            };
            dev.launched("memra_ink2d_attn", unsafe {
                ffi::memra_ink2d_attn(
                    p(&self.q, &st) as *const f32,
                    p(&self.kb, &st) as *const f32,
                    p(&self.vb, &st) as *const f32,
                    p(&lw.kc, &st) as *const f32,
                    p(&lw.vc, &st) as *const f32,
                    lw.cap as i64,
                    i32::from(lw.ring),
                    first as i64,
                    p0 as i64,
                    rows as i32,
                    g.heads as i32,
                    g.kv_heads as i32,
                    d as i32,
                    g.scale as f32,
                    pm(&mut self.att, &st) as *mut f32,
                    dev.sp(),
                )
            })?;
            Self::gemv(
                dev,
                p(&self.att, &st),
                qw,
                p(&lw.o, &st),
                qw,
                0,
                pm(&mut self.o, &st),
                h,
                rows,
                h,
                qw,
            )?;
            dev.launched("memra_ink2d_norm", unsafe {
                ffi::memra_ink2d_norm(
                    xp as *mut f32,
                    p(&self.o, &st) as *const f32,
                    p(&lw.post_norm, &st) as *const f32,
                    g.epsilon,
                    pm(&mut self.xn, &st) as *mut f32,
                    rows as i32,
                    h as i32,
                    dev.sp(),
                )
            })?;
            let xn = p(&self.xn, &st);
            Self::gemv(
                dev,
                xn,
                h,
                p(&lw.gate, &st),
                h,
                0,
                pm(&mut self.gt, &st),
                f,
                rows,
                f,
                h,
            )?;
            Self::gemv(
                dev,
                xn,
                h,
                p(&lw.up, &st),
                h,
                0,
                pm(&mut self.upb, &st),
                f,
                rows,
                f,
                h,
            )?;
            dev.launched("memra_ink2d_silu_mul", unsafe {
                ffi::memra_ink2d_silu_mul(
                    p(&self.gt, &st) as *const f32,
                    p(&self.upb, &st) as *const f32,
                    pm(&mut self.act, &st) as *mut f32,
                    (rows * f) as i64,
                    dev.sp(),
                )
            })?;
            Self::gemv(
                dev,
                p(&self.act, &st),
                f,
                p(&lw.down, &st),
                f,
                0,
                pm(&mut self.dn, &st),
                h,
                rows,
                h,
                f,
            )?;
        }
        dev.launched("memra_ink2d_norm", unsafe {
            ffi::memra_ink2d_norm(
                xp as *mut f32,
                p(&self.dn, &st) as *const f32,
                p(&self.norm, &st) as *const f32,
                g.epsilon,
                pm(&mut self.hidden, &st) as *mut f32,
                rows as i32,
                h as i32,
                dev.sp(),
            )
        })
    }

    /// Propose `k` (1..=rows - 1) tokens after `anchor`, the token at position `len` (not yet
    /// run by the trunk). `choose`, when given, picks each proposal from its softcapped logits
    /// (a sampler); otherwise the proposal is the first maximum, chosen on the device.
    pub fn propose(
        &mut self,
        dev: &Dev,
        shared: Shared,
        anchor: u32,
        k: usize,
        mut choose: Option<&mut Chooser<'_>>,
    ) -> Result<Vec<u32>, String> {
        let g = self.g.clone();
        if k == 0 || k > g.rows - 1 {
            return Err(format!("{k} proposals; a block holds 1..={}", g.rows - 1));
        }
        if self.len == 0 {
            return Err("a proposal needs a committed position (the head's feature)".into());
        }
        // the block's rows are scratch (no cache row is written for them), so a block may run
        // past the context; only the committed positions must fit
        if self.len > self.context {
            return Err("the drafter's context is full".into());
        }
        dev.bind()?;
        self.block(dev, shared, anchor)?;
        let st = self.st.clone();
        let (h, r, vocab) = (g.hidden, g.rank, g.vocab);
        let width = 2 * r + 2 * h;
        let rows1 = p(&self.hidden, &st) + (h * 4) as u64;
        // base logits of the k proposal rows: one pass over the output projection
        Self::gemv(
            dev,
            rows1,
            h,
            shared.head,
            h,
            0,
            pm(&mut self.base, &st),
            vocab,
            k,
            vocab,
            h,
        )?;
        // the projection of [row hidden] per row, and of [feature] plus the bias once
        let rw = p(&self.rnn_w, &st);
        Self::gemv(
            dev,
            rows1,
            h,
            rw + (2 * r * 2) as u64,
            width,
            0,
            pm(&mut self.ctxr, &st),
            3 * r,
            k,
            3 * r,
            h,
        )?;
        Self::gemv(
            dev,
            p(&self.feature, &st),
            h,
            rw + ((2 * r + h) * 2) as u64,
            width,
            p(&self.rnn_b, &st),
            pm(&mut self.fctx, &st),
            3 * r,
            1,
            3 * r,
            h,
        )?;
        let zeros = vec![0f32; r];
        dev.put(&zeros, &mut self.state)?;
        let mut out = Vec::with_capacity(k);
        for i in 0..k {
            dev.launched("memra_ink2d_markov_in", unsafe {
                ffi::memra_ink2d_markov_in(
                    p(&self.state, &st) as *const f32,
                    p(&self.w1, &st) as *const _,
                    p(&self.prev, &st) as *const u32,
                    pm(&mut self.minp, &st) as *mut f32,
                    r as i32,
                    dev.sp(),
                )
            })?;
            Self::gemv(
                dev,
                p(&self.minp, &st),
                2 * r,
                rw,
                width,
                0,
                pm(&mut self.raw, &st),
                3 * r,
                1,
                3 * r,
                2 * r,
            )?;
            dev.launched("memra_ink2d_markov_gate", unsafe {
                ffi::memra_ink2d_markov_gate(
                    (p(&self.ctxr, &st) + (i * 3 * r * 4) as u64) as *const f32,
                    p(&self.raw, &st) as *const f32,
                    p(&self.fctx, &st) as *const f32,
                    pm(&mut self.state, &st) as *mut f32,
                    pm(&mut self.latent, &st) as *mut f32,
                    r as i32,
                    dev.sp(),
                )
            })?;
            dev.launched("memra_ink2d_gemv_f32w", unsafe {
                ffi::memra_ink2d_gemv_f32w(
                    p(&self.latent, &st) as *const f32,
                    p(&self.fold, &st) as *const f32,
                    r as i64,
                    p(&self.fold_b, &st) as *const f32,
                    pm(&mut self.bias, &st) as *mut f32,
                    vocab as i32,
                    r as i32,
                    dev.sp(),
                )
            })?;
            let logits_i = pm(&mut self.logits, &st) + (i * vocab * 4) as u64;
            let token_i = pm(&mut self.tokens, &st) + (i * 4) as u64;
            dev.launched("memra_ink2d_cap_pick", unsafe {
                ffi::memra_ink2d_cap_pick(
                    (p(&self.base, &st) + (i * vocab * 4) as u64) as *const f32,
                    p(&self.bias, &st) as *const f32,
                    g.softcap as f32,
                    logits_i as *mut f32,
                    vocab as i32,
                    pm(&mut self.part_v, &st) as *mut f32,
                    pm(&mut self.part_i, &st) as *mut u32,
                    token_i as *mut u32,
                    if choose.is_some() {
                        std::ptr::null_mut()
                    } else {
                        pm(&mut self.prev, &st) as *mut u32
                    },
                    dev.sp(),
                )
            })?;
            if let Some(c) = choose.as_deref_mut() {
                let row = dev.down(&self.logits, i * vocab, vocab)?;
                let t = c(i, &row);
                dev.put(&[t], &mut self.prev)?;
                out.push(t);
            }
        }
        if choose.is_none() {
            out = dev.down(&self.tokens, 0, k)?;
        }
        Ok(out)
    }

    /// The last block's proposal logits, `[k, vocab]`.
    pub fn logits(&self, dev: &Dev, k: usize) -> Result<Vec<f32>, String> {
        dev.down(&self.logits, 0, k * self.g.vocab)
    }

    /// The last block's final-norm rows `[rows, hidden]` and the head's feature.
    pub fn hidden(&self, dev: &Dev) -> Result<(Vec<f32>, Vec<f32>), String> {
        Ok((
            dev.down(&self.hidden, 0, self.g.rows * self.g.hidden)?,
            dev.down(&self.feature, 0, self.g.hidden)?,
        ))
    }
}

/// The drafter tensors of a checkpoint's auxiliary file (`mtp.safetensors`), as f32 values.
pub struct CkptDraft<'a>(pub &'a memra_gguf::safetensors::StModel);

impl DraftSource for CkptDraft<'_> {
    fn tensor(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let (info, bytes) = self
            .0
            .raw(name)
            .ok_or_else(|| format!("the drafter file has no tensor {name}"))?;
        let got: Vec<usize> = info.shape.iter().map(|v| *v as usize).collect();
        if info.dtype != "BF16" || got != shape {
            return Err(format!(
                "{name}: {} {got:?}, the drafter needs BF16 {shape:?}",
                info.dtype
            ));
        }
        Ok(bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect())
    }
}
