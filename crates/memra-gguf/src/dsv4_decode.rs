//! DeepSeek-V4-Flash CPU stateful decode path (lane 10, research/dsv4-dspark-20260818).
//!
//! Semantic source (THE LAW): the 0731 checkpoint's own `inference/model.py` decode
//! branches — `Compressor.forward` start_pos>0 (M:349-383), `Indexer.forward`
//! (M:408-439), `Attention.forward` (M:490-548), `get_window_topk_idxs` (M:260-271),
//! `get_compress_topk_idxs` (M:274-282) — under the lane-2 f32 fixture contract, the
//! same program shape as the Gate C torch decode driver (`dsv4_cpu_greedy_0731.py`,
//! banked in darklanes fixtures-0731/). The lane-3 prefill oracle
//! ([`crate::dsv4_forward`]) supplies every numeric primitive (f64-accumulated dots,
//! QAT sims, rope, Sinkhorn hc, sparse-attention math, the gated compressor/indexer
//! prefill); this module adds only the decode STATE mechanics: per-layer window ring
//! @ p % 128, compressor pending kv/score state with the fine cur→prev overlap shift,
//! growing compressed-block and indexer stores.
//!
//! State structs own BUFFERS only; weights stay in the lane-3 loader structs and are
//! borrowed per call — one weight decode, two drivers (prefill + decode), no duplicate
//! numeric program.
//!
//! Gates for this module (formulas banked in the lane receipts BEFORE runs): the
//! free-running greedy over this path must reproduce the banked 0731 REF greedy
//! trajectory token-for-token, and the teacher-forced component gate compares against
//! torch-decode fixtures at derived thresholds.

use crate::dsv4_forward::{
    ActQuantVariant, AttnW, BlockW, CompressorW, Dsv4Model, FreqsCis, HcSet, IndexerW, act_quant,
    apply_rope, dot, fp4_act_quant, hadamard, hc_head, hc_post, hc_pre, matmul, rmsnorm,
};

/// Max sequence the decode-state buffers are sized for (prompt 32 + 160 new + drafter
/// lookahead + margin, mirroring the Gate C torch driver's MAX_LEN).
pub const MAX_LEN: usize = 256;

// ============================ decode-side index builders (model.py:260-282) ============================

/// get_window_topk_idxs decode branches (M:261-267): ring-slot order for a query at
/// `start_pos` (width == win; -1 padding while the ring is not yet full).
pub fn window_idxs_decode(win: usize, start_pos: usize) -> Vec<i64> {
    if start_pos >= win - 1 {
        let sp = start_pos % win;
        let mut m: Vec<i64> = ((sp + 1)..win).map(|x| x as i64).collect();
        m.extend((0..=sp).map(|x| x as i64));
        m
    } else {
        let mut m: Vec<i64> = (0..=start_pos).map(|x| x as i64).collect();
        m.resize(win, -1);
        m
    }
}

/// get_compress_topk_idxs decode branch (M:276-277): all completed blocks, + offset.
pub fn compress_idxs_decode(ratio: usize, start_pos: usize, offset: usize) -> Vec<i64> {
    (0..(start_pos + 1) / ratio)
        .map(|j| (j + offset) as i64)
        .collect()
}

// ============================ sparse attention over gathered rows ============================

/// One-query sparse attention (kernel.py sparse_attn semantics, K:277-352): scores over
/// the selected kv rows (-1 = masked slot), softmax with attn_sink denominator-only
/// mass, f64 sums — identical arithmetic to the lane-3 prefill implementation,
/// factored for row-gather callers. `row(i)` returns the kv row for logical index i.
#[allow(clippy::too_many_arguments)]
pub fn sparse_attn_query<'a>(
    q: &[f32],
    heads: usize,
    hd: usize,
    idxs: &[i64],
    row: impl Fn(usize) -> &'a [f32],
    sink: &[f32],
    scale: f32,
    out: &mut [f32],
) {
    assert_eq!(q.len(), heads * hd);
    assert_eq!(out.len(), heads * hd);
    for h in 0..heads {
        let qv = &q[h * hd..(h + 1) * hd];
        let mut scores = vec![f32::NEG_INFINITY; idxs.len()];
        for (sl, &ix) in idxs.iter().enumerate() {
            if ix >= 0 {
                scores[sl] = dot(qv, row(ix as usize)) * scale;
            }
        }
        let mut m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        m = m.max(-1e30);
        let mut denom = 0f64;
        let mut acc = vec![0f64; hd];
        for (sl, &ix) in idxs.iter().enumerate() {
            if ix < 0 {
                continue;
            }
            let e = (scores[sl] - m).exp();
            denom += e as f64;
            let krow = row(ix as usize);
            for i in 0..hd {
                acc[i] += e as f64 * krow[i] as f64;
            }
        }
        denom += (sink[h] - m).exp() as f64;
        let dst = &mut out[h * hd..(h + 1) * hd];
        for i in 0..hd {
            dst[i] = (acc[i] / denom) as f32;
        }
    }
}

// ============================ batched-verify checkpoint (§3.1 ring hazard, iteration 3) ============================

/// Snapshot + replay payload for ONE compressor's pending state across a batched
/// verify round (DSPARK-SEMANTICS §3.1 item 2 — the compressor pending + emissions
/// are the non-idempotent trunk-side hazard). Decision banked in the iteration-3
/// record: SNAPSHOT-ROLLBACK with bounded replay, not SGLang-style ring doubling —
/// the pending state is tiny (fine [8,1024]×2, coarse [128,512]×2 per layer), the
/// append-only stores roll back by high-water mark, and the emitted store rows for
/// the committed prefix are already bit-identical (the batch advanced the pending
/// in place sequentially, so every emission's pooling inputs equal the sequential
/// twin's), so replay is pure row writes + shift bookkeeping — no re-pooling.
pub struct CompCkpt {
    kv_snap: Vec<f32>,
    score_snap: Vec<f32>,
    n_blocks0: usize,
    /// per verified position, in order: (dst row, kv row, POST-ape score row, emitted)
    rows: Vec<(usize, Vec<f32>, Vec<f32>, bool)>,
}

// ============================ compressor state (model.py:285-383, both branches) ============================

/// Decode-state buffers for one compressor: the pending kv/score window
/// (model.py:309-310 — overlap: rows [0:ratio] = previous window FULL width, rows
/// [ratio:2ratio] = current; pool reads prev[:, :d] and cur[:, d:]) plus the owned
/// store of emitted compressed rows (post norm/rope/QAT — what attention reads).
pub struct CompressorState {
    pub store: Vec<f32>, // [max_blocks, d]
    pub n_blocks: usize,
    kv_state: Vec<f32>,    // [coff*ratio, coff*d]
    score_state: Vec<f32>, // same shape, init -inf
    d: usize,
}

impl CompressorState {
    pub fn new(w: &CompressorW, max_len: usize) -> Self {
        let coff = if w.overlap { 2 } else { 1 };
        assert_eq!(coff * w.d, w.latent, "compressor latent width");
        CompressorState {
            store: vec![0f32; (max_len / w.ratio) * w.d],
            n_blocks: 0,
            kv_state: vec![0f32; (coff * w.ratio) * w.latent],
            score_state: vec![f32::NEG_INFINITY; (coff * w.ratio) * w.latent],
            d: w.d,
        }
    }

    pub fn block_row(&self, j: usize) -> &[f32] {
        assert!(j < self.n_blocks, "compressed block {j} not yet written");
        &self.store[j * self.d..(j + 1) * self.d]
    }

    /// Store prefill-pooled rows (produced by the GATED lane-3 CompressorW/IndexerW
    /// forward — identical program) and seed the pending state from the tail
    /// positions (M:331-341).
    pub fn seed_prefill(
        &mut self,
        w: &CompressorW,
        pooled: Option<&(Vec<f32>, usize)>,
        x: &[f32],
        s: usize,
        hidden: usize,
    ) {
        if let Some((rows, nb)) = pooled {
            assert!(nb * self.d <= self.store.len(), "compressor store overflow");
            self.store[..nb * self.d].copy_from_slice(rows);
            self.n_blocks = *nb;
        }
        let (ratio, latent, overlap) = (w.ratio, w.latent, w.overlap);
        let cutoff = s - s % ratio;
        let remainder = s % ratio;
        let seed_from = if overlap {
            cutoff.saturating_sub(ratio)
        } else {
            cutoff
        };
        if seed_from >= s {
            return;
        }
        let tail = &x[seed_from * hidden..s * hidden];
        let n_tail = s - seed_from;
        let kv_t = matmul(tail, n_tail, hidden, &w.wkv, latent);
        let score_t = matmul(tail, n_tail, hidden, &w.wgate, latent);
        if overlap && cutoff >= ratio {
            // kv_state[:ratio] = kv[cutoff-ratio:cutoff]; score + ape (M:337-338)
            for p in 0..ratio {
                let src = (cutoff - ratio + p) - seed_from;
                self.kv_state[p * latent..(p + 1) * latent]
                    .copy_from_slice(&kv_t[src * latent..(src + 1) * latent]);
                for c in 0..latent {
                    self.score_state[p * latent + c] =
                        score_t[src * latent + c] + w.ape[p * latent + c];
                }
            }
        }
        if remainder > 0 {
            let offset = if overlap { ratio } else { 0 }; // M:335
            for p in 0..remainder {
                let src = (cutoff + p) - seed_from;
                let dst = offset + p;
                self.kv_state[dst * latent..(dst + 1) * latent]
                    .copy_from_slice(&kv_t[src * latent..(src + 1) * latent]);
                for c in 0..latent {
                    self.score_state[dst * latent + c] =
                        score_t[src * latent + c] + w.ape[p * latent + c];
                }
            }
        }
    }

    /// Decode (start_pos > 0, seqlen 1): state update (M:349-365) + pooled emission on
    /// every ratio-th position, then norm/rope(first-pos)/QAT identical to prefill
    /// (M:366-383). `x_row` is the post-attn-norm activation row.
    #[allow(clippy::too_many_arguments)]
    pub fn decode(
        &mut self,
        w: &CompressorW,
        x_row: &[f32],
        hidden: usize,
        start_pos: usize,
        fc: &FreqsCis,
        rd: usize,
        eps: f32,
        variant: ActQuantVariant,
    ) {
        self.decode_ck(w, x_row, hidden, start_pos, fc, rd, eps, variant, None);
    }

    /// [`Self::decode`] with optional checkpoint recording (batched verify): identical
    /// arithmetic — the recording is pure copies and never changes what is computed.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
    pub fn decode_ck(
        &mut self,
        w: &CompressorW,
        x_row: &[f32],
        hidden: usize,
        start_pos: usize,
        fc: &FreqsCis,
        rd: usize,
        eps: f32,
        variant: ActQuantVariant,
        ck: Option<&mut CompCkpt>,
    ) {
        let (ratio, d, latent, overlap) = (w.ratio, w.d, w.latent, w.overlap);
        let kv = matmul(x_row, 1, hidden, &w.wkv, latent);
        let mut score = matmul(x_row, 1, hidden, &w.wgate, latent);
        let p = start_pos % ratio;
        for (v, a) in score.iter_mut().zip(&w.ape[p * latent..(p + 1) * latent]) {
            *v += *a; // M:351
        }
        let should = (start_pos + 1) % ratio == 0;
        let dst = if overlap { ratio + p } else { p };
        if let Some(ck) = ck {
            ck.rows.push((dst, kv.clone(), score.clone(), should));
        }
        self.kv_state[dst * latent..(dst + 1) * latent].copy_from_slice(&kv);
        self.score_state[dst * latent..(dst + 1) * latent].copy_from_slice(&score);
        if !should {
            return;
        }
        // pool: overlap reads prev rows' dims [0:d] + cur rows' dims [d:2d] (M:356-358)
        let positions = if overlap { 2 * ratio } else { ratio };
        let mut pooled = vec![0f32; d];
        #[allow(clippy::needless_range_loop)]
        for c in 0..d {
            let mut sc = Vec::with_capacity(positions);
            let mut kvv = Vec::with_capacity(positions);
            for r in 0..ratio {
                sc.push(self.score_state[r * latent + c]);
                kvv.push(self.kv_state[r * latent + c]);
            }
            if overlap {
                for r in ratio..2 * ratio {
                    sc.push(self.score_state[r * latent + d + c]);
                    kvv.push(self.kv_state[r * latent + d + c]);
                }
            }
            let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut den = 0f64;
            let mut num = 0f64;
            for i in 0..positions {
                let e = (sc[i] - mx).exp();
                den += e as f64;
                num += e as f64 * kvv[i] as f64;
            }
            pooled[c] = (num / den) as f32;
        }
        if overlap {
            // cur -> prev shift (M:359-360), full-width rows
            for r in 0..ratio {
                self.kv_state
                    .copy_within((ratio + r) * latent..(ratio + r + 1) * latent, r * latent);
                self.score_state
                    .copy_within((ratio + r) * latent..(ratio + r + 1) * latent, r * latent);
            }
        }
        let mut row = rmsnorm(&pooled, &w.norm_w, eps);
        // the block carries the rope of its FIRST position (M:372)
        apply_rope(&mut row, 1, 1, d, rd, fc, &[start_pos + 1 - ratio], false);
        if w.rotate {
            hadamard(&mut row, d);
            fp4_act_quant(&mut row, 32);
        } else {
            act_quant(&mut row[..d - rd], 64, variant);
        }
        let j = start_pos / ratio; // == (start_pos+1)/ratio - 1 when `should`
        assert_eq!(j, self.n_blocks, "compressed blocks must append in order");
        assert!((j + 1) * d <= self.store.len(), "compressor store overflow");
        self.store[j * d..(j + 1) * d].copy_from_slice(&row);
        self.n_blocks = j + 1;
    }

    /// Open a verify-round checkpoint: pending-state snapshot + store high-water mark.
    pub fn begin_ckpt(&self) -> CompCkpt {
        CompCkpt {
            kv_snap: self.kv_state.clone(),
            score_snap: self.score_state.clone(),
            n_blocks0: self.n_blocks,
            rows: Vec::new(),
        }
    }

    /// §3.1 rollback: restore the pending snapshot + truncate the store to the high-
    /// water mark, then REPLAY the first `n_commit` recorded positions (row writes +
    /// the cur→prev shift + block accounting; the emitted store rows are kept as
    /// written by the batch — bit-identical, see [`CompCkpt`] doc). After this, state
    /// == plain sequential decode of exactly the committed positions.
    pub fn rollback_replay(&mut self, w: &CompressorW, ck: &CompCkpt, n_commit: usize) {
        assert!(
            n_commit <= ck.rows.len(),
            "commit beyond recorded positions"
        );
        if n_commit == ck.rows.len() {
            return; // fully committed: in-place batch state is already exact
        }
        let (ratio, latent, overlap) = (w.ratio, w.latent, w.overlap);
        self.kv_state.copy_from_slice(&ck.kv_snap);
        self.score_state.copy_from_slice(&ck.score_snap);
        self.n_blocks = ck.n_blocks0;
        for (dst, kv, score, emitted) in &ck.rows[..n_commit] {
            self.kv_state[dst * latent..(dst + 1) * latent].copy_from_slice(kv);
            self.score_state[dst * latent..(dst + 1) * latent].copy_from_slice(score);
            if *emitted {
                if overlap {
                    for r in 0..ratio {
                        self.kv_state.copy_within(
                            (ratio + r) * latent..(ratio + r + 1) * latent,
                            r * latent,
                        );
                        self.score_state.copy_within(
                            (ratio + r) * latent..(ratio + r + 1) * latent,
                            r * latent,
                        );
                    }
                }
                self.n_blocks += 1;
            }
        }
    }

    /// The live state views for bit-level gate comparison: (pending kv, pending score,
    /// live store rows). Store bytes beyond `n_blocks` are dead, never state.
    pub fn state_views(&self) -> (&[f32], &[f32], &[f32]) {
        (
            &self.kv_state,
            &self.score_state,
            &self.store[..self.n_blocks * self.d],
        )
    }

    /// Reset to exact construction state (fresh-instance twin runs).
    pub fn reset(&mut self) {
        self.kv_state.fill(0f32);
        self.score_state.fill(f32::NEG_INFINITY);
        self.n_blocks = 0;
    }
}

// ============================ indexer state (model.py:386-439) ============================

pub struct IndexerState {
    pub compressor: CompressorState,
}

impl IndexerState {
    pub fn new(ix: &IndexerW, max_len: usize) -> Self {
        IndexerState {
            compressor: CompressorState::new(&ix.compressor, max_len),
        }
    }

    /// Decode branch (M:408-439, seqlen 1): compressor FIRST (M:423 — the block this
    /// position completes is scoreable), fp4-QAT'd rotated q against the stored
    /// compressed rows, relu·weights head sum, top-k + `offset` (= win), no re-mask.
    /// `ck` (batched verify) records the indexer-compressor state advance for §3.1
    /// rollback; None == the plain sequential path, bit-identical either way.
    #[allow(clippy::too_many_arguments)]
    pub fn decode(
        &mut self,
        ix: &IndexerW,
        x_row: &[f32],
        qr: &[f32],
        hidden: usize,
        q_lora: usize,
        start_pos: usize,
        offset: usize,
        fc: &FreqsCis,
        rd: usize,
        eps: f32,
        variant: ActQuantVariant,
        ck: Option<&mut CompCkpt>,
    ) -> Vec<i64> {
        let (heads, hd) = (ix.heads, ix.hd);
        let ratio = ix.compressor.ratio;
        let mut q = matmul(qr, 1, q_lora, &ix.wq_b, heads * hd);
        apply_rope(&mut q, 1, heads, hd, rd, fc, &[start_pos], false);
        hadamard(&mut q, hd);
        fp4_act_quant(&mut q, 32);
        self.compressor.decode_ck(
            &ix.compressor,
            x_row,
            hidden,
            start_pos,
            fc,
            rd,
            eps,
            variant,
            ck,
        );
        let nb = (start_pos + 1) / ratio;
        assert_eq!(nb, self.compressor.n_blocks, "indexer block count drift");
        if nb == 0 {
            return Vec::new();
        }
        let scale = ((hd as f64).powf(-0.5) * (heads as f64).powf(-0.5)) as f32;
        let mut weights = matmul(x_row, 1, hidden, &ix.weights_proj, heads);
        for v in &mut weights {
            *v *= scale;
        }
        let mut score = vec![0f32; nb];
        for (j, o) in score.iter_mut().enumerate() {
            let ck = self.compressor.block_row(j);
            let mut acc = 0f64;
            for h in 0..heads {
                let sc = dot(&q[h * hd..(h + 1) * hd], ck);
                acc += (sc.max(0.0) * weights[h]) as f64;
            }
            *o = acc as f32;
        }
        // top-k: value desc, index asc on ties (lane-3 convention == torch.topk)
        let k = ix.topk.min(nb);
        let mut order: Vec<usize> = (0..nb).collect();
        order.sort_by(|&a, &b| {
            score[b]
                .partial_cmp(&score[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        order
            .into_iter()
            .take(k)
            .map(|j| (j + offset) as i64)
            .collect()
    }
}

// ============================ attention state (model.py:442-548, both branches) ============================

/// One layer's verify-round checkpoint (§3.1): the transient window kv rows (never
/// in the ring until commit) + the two compressor replay payloads.
pub struct AttnBatchCkpt {
    kv_rows: Vec<f32>, // [t, hd]
    comp: Option<CompCkpt>,
    idx: Option<CompCkpt>,
    t: usize,
    pos0: usize,
}

/// Per-layer decode state: 128-slot window ring + compressor store + indexer store,
/// exactly the reference cache geometry (ring slot = p % win; compressed block j at
/// logical index win + j at decode — M:491, M:509, M:530, M:535).
pub struct AttnState {
    pub w: AttnW,
    pub ring: Vec<f32>, // [win, hd]
    pub comp: Option<CompressorState>,
    pub idx: Option<IndexerState>,
    win: usize,
    hd: usize,
}

impl AttnState {
    pub fn new(aw: AttnW, win: usize, hd: usize, max_len: usize) -> Self {
        let comp = aw
            .compressor
            .as_ref()
            .map(|c| CompressorState::new(c, max_len));
        let idx = aw.indexer.as_ref().map(|ix| IndexerState::new(ix, max_len));
        AttnState {
            w: aw,
            ring: vec![0f32; win * hd],
            comp,
            idx,
            win,
            hd,
        }
    }

    /// Prefill (start_pos == 0): the lane-3 program with state capture. x [s, hidden]
    /// post-attn-norm. Returns the attention output [s, hidden].
    pub fn prefill(
        &mut self,
        model: &Dsv4Model,
        x: &[f32],
        s: usize,
        variant: ActQuantVariant,
    ) -> Vec<f32> {
        let d = model.cfg();
        let hidden = model.mc.n_embd as usize;
        let heads = model.mc.n_head as usize;
        let hd = self.hd;
        let rd = d.qk_rope_head_dim as usize;
        let q_lora = d.q_lora_rank as usize;
        let win = self.win;
        let eps = model.mc.rms_eps;
        let positions: Vec<usize> = (0..s).collect();

        // q path (M:496-499)
        let qr = rmsnorm(
            &matmul(x, s, hidden, &self.w.wq_a, q_lora),
            &self.w.q_norm,
            eps,
        );
        let mut q = matmul(&qr, s, q_lora, &self.w.wq_b, heads * hd);
        for head in q.chunks_exact_mut(hd) {
            let mut acc = 0f64;
            for v in head.iter() {
                acc += (*v as f64) * (*v as f64);
            }
            let rsq = 1.0f32 / ((acc / hd as f64) as f32 + eps).sqrt();
            for v in head.iter_mut() {
                *v *= rsq;
            }
        }
        apply_rope(&mut q, s, heads, hd, rd, &self.w.fc, &positions, false);

        // shared kv latent (M:502-506) + window QAT
        let mut kv = rmsnorm(&matmul(x, s, hidden, &self.w.wkv, hd), &self.w.kv_norm, eps);
        apply_rope(&mut kv, s, 1, hd, rd, &self.w.fc, &positions, false);
        for row in kv.chunks_exact_mut(hd) {
            act_quant(&mut row[..hd - rd], 64, variant);
        }
        // window ring init (M:523-528): last min(s, win) rows land at slot p % win
        for p in s.saturating_sub(win)..s {
            self.ring[(p % win) * hd..(p % win + 1) * hd]
                .copy_from_slice(&kv[p * hd..(p + 1) * hd]);
        }

        let (mut idxs, mut slots) = crate::dsv4_forward::window_topk_idxs(win, s);
        if self.w.ratio != 0 {
            let offset = s; // prefill indexing: [kv rows | compressed rows] (M:515)
            let (cidx, cslots) = if let Some(iw) = &self.w.indexer {
                let out = iw.forward(
                    x, &qr, s, hidden, q_lora, offset, &self.w.fc, rd, eps, variant, true,
                );
                let ist = self.idx.as_mut().expect("indexer state");
                ist.compressor
                    .seed_prefill(&iw.compressor, out.indexer_kv.as_ref(), x, s, hidden);
                (out.idxs, out.slots)
            } else {
                crate::dsv4_forward::compress_topk_idxs(self.w.ratio, s, offset)
            };
            if cslots > 0 {
                let mut merged = vec![-1i64; s * (slots + cslots)];
                for t in 0..s {
                    merged[t * (slots + cslots)..t * (slots + cslots) + slots]
                        .copy_from_slice(&idxs[t * slots..(t + 1) * slots]);
                    merged[t * (slots + cslots) + slots..(t + 1) * (slots + cslots)]
                        .copy_from_slice(&cidx[t * cslots..(t + 1) * cslots]);
                }
                idxs = merged;
                slots += cslots;
            }
            let cw = self.w.compressor.as_ref().expect("ratio != 0");
            let pooled = cw.forward(x, s, hidden, &self.w.fc, rd, eps, variant);
            self.comp.as_mut().expect("compressor state").seed_prefill(
                cw,
                pooled.as_ref(),
                x,
                s,
                hidden,
            );
        }

        // attention with PREFILL indexing: i < s -> kv row i; i >= s -> block i - s.
        let comp_ref = self.comp.as_ref();
        let scale = (hd as f64).powf(-0.5) as f32;
        let mut o = vec![0f32; s * heads * hd];
        for t in 0..s {
            let ti = &idxs[t * slots..(t + 1) * slots];
            let row = |ix: usize| -> &[f32] {
                if ix < s {
                    &kv[ix * hd..(ix + 1) * hd]
                } else {
                    comp_ref.expect("compressed index").block_row(ix - s)
                }
            };
            sparse_attn_query(
                &q[t * heads * hd..(t + 1) * heads * hd],
                heads,
                hd,
                ti,
                row,
                &self.w.sink,
                scale,
                &mut o[t * heads * hd..(t + 1) * heads * hd],
            );
        }
        apply_rope(&mut o, s, heads, hd, rd, &self.w.fc, &positions, true);
        self.output_proj(&o, s, heads, hd, hidden, model)
    }

    /// Decode (start_pos > 0, one position). x_row [hidden] post-attn-norm.
    pub fn decode(
        &mut self,
        model: &Dsv4Model,
        x_row: &[f32],
        start_pos: usize,
        variant: ActQuantVariant,
    ) -> Vec<f32> {
        let d = model.cfg();
        let hidden = model.mc.n_embd as usize;
        let heads = model.mc.n_head as usize;
        let hd = self.hd;
        let rd = d.qk_rope_head_dim as usize;
        let q_lora = d.q_lora_rank as usize;
        let win = self.win;
        let eps = model.mc.rms_eps;

        let qr = rmsnorm(
            &matmul(x_row, 1, hidden, &self.w.wq_a, q_lora),
            &self.w.q_norm,
            eps,
        );
        let mut q = matmul(&qr, 1, q_lora, &self.w.wq_b, heads * hd);
        for head in q.chunks_exact_mut(hd) {
            let mut acc = 0f64;
            for v in head.iter() {
                acc += (*v as f64) * (*v as f64);
            }
            let rsq = 1.0f32 / ((acc / hd as f64) as f32 + eps).sqrt();
            for v in head.iter_mut() {
                *v *= rsq;
            }
        }
        apply_rope(&mut q, 1, heads, hd, rd, &self.w.fc, &[start_pos], false);

        let mut kv = rmsnorm(
            &matmul(x_row, 1, hidden, &self.w.wkv, hd),
            &self.w.kv_norm,
            eps,
        );
        apply_rope(&mut kv, 1, 1, hd, rd, &self.w.fc, &[start_pos], false);
        act_quant(&mut kv[..hd - rd], 64, variant);

        // decode indexing: [ring slots | win + block j] (M:509)
        let mut idxs = window_idxs_decode(win, start_pos);
        if self.w.ratio != 0 {
            let cidx = if let Some(iw) = &self.w.indexer {
                self.idx.as_mut().expect("indexer state").decode(
                    iw, x_row, &qr, hidden, q_lora, start_pos, win, &self.w.fc, rd, eps, variant,
                    None,
                )
            } else {
                compress_idxs_decode(self.w.ratio, start_pos, win)
            };
            idxs.extend(cidx);
        }
        // ring write BEFORE attention (M:535); attention compressor AFTER it (M:537)
        self.ring[(start_pos % win) * hd..(start_pos % win + 1) * hd].copy_from_slice(&kv);
        if let Some(comp) = &mut self.comp {
            let cw = self.w.compressor.as_ref().expect("ratio != 0");
            comp.decode(cw, x_row, hidden, start_pos, &self.w.fc, rd, eps, variant);
        }

        let scale = (hd as f64).powf(-0.5) as f32;
        let mut o = vec![0f32; heads * hd];
        {
            let ring = &self.ring;
            let comp = self.comp.as_ref();
            let row = |ix: usize| -> &[f32] {
                if ix < win {
                    &ring[ix * hd..(ix + 1) * hd]
                } else {
                    comp.expect("compressed index").block_row(ix - win)
                }
            };
            sparse_attn_query(&q, heads, hd, &idxs, row, &self.w.sink, scale, &mut o);
        }
        apply_rope(&mut o, 1, heads, hd, rd, &self.w.fc, &[start_pos], true);
        self.output_proj(&o, 1, heads, hd, hidden, model)
    }

    /// Batched T-position verify forward (§3.1): the sequential [`Self::decode`]
    /// program per position, with TWO controlled differences that never change a bit
    /// of arithmetic:
    ///   1. window-ring writes go to a TRANSIENT buffer (the reference's own
    ///      [ring | draft] gather shape, M:784) — the ring is read-only during the
    ///      round, so a rejected suffix never touches it and the batched attention
    ///      of the first query still sees the pre-round slot contents it needs;
    ///      reads of in-round positions are redirected to the transient rows,
    ///      resolving to the SAME floats sequential decode would read.
    ///   2. compressor/indexer pending advances in place (sequentially, position
    ///      order — later in-round queries must see in-round block emissions) while
    ///      recording the [`CompCkpt`] replay payload for partial-accept rollback.
    ///
    /// Returns per-position attention outputs [t, hidden] + the round checkpoint.
    pub fn decode_batch(
        &mut self,
        model: &Dsv4Model,
        xs: &[f32],
        t: usize,
        pos0: usize,
        variant: ActQuantVariant,
    ) -> (Vec<f32>, AttnBatchCkpt) {
        let d = model.cfg();
        let hidden = model.mc.n_embd as usize;
        let heads = model.mc.n_head as usize;
        let hd = self.hd;
        let rd = d.qk_rope_head_dim as usize;
        let q_lora = d.q_lora_rank as usize;
        let win = self.win;
        let eps = model.mc.rms_eps;
        assert!(pos0 > 0, "batched verify is a decode-path round");
        assert!(t <= win, "round depth must stay below the window");
        assert_eq!(xs.len(), t * hidden);

        let mut ck = AttnBatchCkpt {
            kv_rows: vec![0f32; t * hd],
            comp: self.comp.as_ref().map(|c| c.begin_ckpt()),
            idx: self.idx.as_ref().map(|ix| ix.compressor.begin_ckpt()),
            t,
            pos0,
        };
        let mut out = vec![0f32; t * heads * hd];
        for i in 0..t {
            let start_pos = pos0 + i;
            let x_row = &xs[i * hidden..(i + 1) * hidden];

            let qr = rmsnorm(
                &matmul(x_row, 1, hidden, &self.w.wq_a, q_lora),
                &self.w.q_norm,
                eps,
            );
            let mut q = matmul(&qr, 1, q_lora, &self.w.wq_b, heads * hd);
            for head in q.chunks_exact_mut(hd) {
                let mut acc = 0f64;
                for v in head.iter() {
                    acc += (*v as f64) * (*v as f64);
                }
                let rsq = 1.0f32 / ((acc / hd as f64) as f32 + eps).sqrt();
                for v in head.iter_mut() {
                    *v *= rsq;
                }
            }
            apply_rope(&mut q, 1, heads, hd, rd, &self.w.fc, &[start_pos], false);

            let mut kv = rmsnorm(
                &matmul(x_row, 1, hidden, &self.w.wkv, hd),
                &self.w.kv_norm,
                eps,
            );
            apply_rope(&mut kv, 1, 1, hd, rd, &self.w.fc, &[start_pos], false);
            act_quant(&mut kv[..hd - rd], 64, variant);

            let mut idxs = window_idxs_decode(win, start_pos);
            if self.w.ratio != 0 {
                let cidx = if let Some(iw) = &self.w.indexer {
                    self.idx.as_mut().expect("indexer state").decode(
                        iw,
                        x_row,
                        &qr,
                        hidden,
                        q_lora,
                        start_pos,
                        win,
                        &self.w.fc,
                        rd,
                        eps,
                        variant,
                        ck.idx.as_mut(),
                    )
                } else {
                    compress_idxs_decode(self.w.ratio, start_pos, win)
                };
                idxs.extend(cidx);
            }
            // ring write REDIRECTED to the transient row (sequential writes the ring
            // here, M:535); attention compressor advances AFTER it (M:537), recorded.
            ck.kv_rows[i * hd..(i + 1) * hd].copy_from_slice(&kv);
            if let Some(comp) = &mut self.comp {
                let cw = self.w.compressor.as_ref().expect("ratio != 0");
                comp.decode_ck(
                    cw,
                    x_row,
                    hidden,
                    start_pos,
                    &self.w.fc,
                    rd,
                    eps,
                    variant,
                    ck.comp.as_mut(),
                );
            }

            let scale = (hd as f64).powf(-0.5) as f32;
            {
                let ring = &self.ring;
                let kv_rows = &ck.kv_rows;
                let comp = self.comp.as_ref();
                // slot -> in-round transient row: positions pos0..=start_pos occupy
                // distinct slots (t <= win); everything else reads the untouched ring.
                let row = |ix: usize| -> &[f32] {
                    if ix < win {
                        for j in (0..=i).rev() {
                            if (pos0 + j) % win == ix {
                                return &kv_rows[j * hd..(j + 1) * hd];
                            }
                        }
                        &ring[ix * hd..(ix + 1) * hd]
                    } else {
                        comp.expect("compressed index").block_row(ix - win)
                    }
                };
                sparse_attn_query(
                    &q,
                    heads,
                    hd,
                    &idxs,
                    row,
                    &self.w.sink,
                    scale,
                    &mut out[i * heads * hd..(i + 1) * heads * hd],
                );
            }
            apply_rope(
                &mut out[i * heads * hd..(i + 1) * heads * hd],
                1,
                heads,
                hd,
                rd,
                &self.w.fc,
                &[start_pos],
                true,
            );
        }
        let o = self.output_proj(&out, t, heads, hd, hidden, model);
        (o, ck)
    }

    /// Commit the first `n_commit` positions of a verify round and roll back the rest
    /// (§3.1 invariant: state == plain sequential decode of exactly the committed
    /// positions, bit-level, every cache class).
    pub fn commit_batch(&mut self, ck: &AttnBatchCkpt, n_commit: usize) {
        assert!(n_commit <= ck.t, "commit beyond round width");
        let hd = self.hd;
        let win = self.win;
        for j in 0..n_commit {
            let slot = (ck.pos0 + j) % win;
            self.ring[slot * hd..(slot + 1) * hd]
                .copy_from_slice(&ck.kv_rows[j * hd..(j + 1) * hd]);
        }
        if let Some(cc) = &ck.comp {
            let cw = self
                .w
                .compressor
                .as_ref()
                .expect("comp ckpt without weights");
            self.comp
                .as_mut()
                .expect("compressor state")
                .rollback_replay(cw, cc, n_commit);
        }
        if let Some(ic) = &ck.idx {
            let iw = self.w.indexer.as_ref().expect("idx ckpt without weights");
            self.idx
                .as_mut()
                .expect("indexer state")
                .compressor
                .rollback_replay(&iw.compressor, ic, n_commit);
        }
    }

    /// Reset all decode state to exact construction state (fresh-twin gate runs).
    pub fn reset(&mut self) {
        self.ring.fill(0f32);
        if let Some(c) = &mut self.comp {
            c.reset();
        }
        if let Some(ix) = &mut self.idx {
            ix.compressor.reset();
        }
    }

    /// Ring view for bit-level gate comparison.
    pub fn ring_view(&self) -> &[f32] {
        &self.ring
    }

    /// Grouped low-rank output projection (M:537-542), shared by both branches.
    fn output_proj(
        &self,
        o: &[f32],
        s: usize,
        heads: usize,
        hd: usize,
        hidden: usize,
        model: &Dsv4Model,
    ) -> Vec<f32> {
        let d = model.cfg();
        let o_groups = d.o_groups as usize;
        let o_lora = d.o_lora_rank as usize;
        let gw = heads / o_groups * hd;
        let mut og = vec![0f32; s * o_groups * o_lora];
        for t in 0..s {
            for g in 0..o_groups {
                let src = &o[t * heads * hd + g * gw..t * heads * hd + (g + 1) * gw];
                let wg = &self.w.wo_a[g * o_lora * gw..(g + 1) * o_lora * gw];
                let dst = &mut og[(t * o_groups + g) * o_lora..(t * o_groups + g + 1) * o_lora];
                for (r, out_v) in dst.iter_mut().enumerate() {
                    *out_v = dot(src, &wg[r * gw..(r + 1) * gw]);
                }
            }
        }
        matmul(&og, s, o_groups * o_lora, &self.w.wo_b, hidden)
    }
}

// ============================ block + trunk driver ============================

/// One trunk block with decode state: BlockW's hc/norm/MoE weights + [`AttnState`].
pub struct BlockState {
    pub attn: AttnState,
    pub moe: crate::dsv4_forward::MoeW,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub hc_attn: HcSet,
    pub hc_ffn: HcSet,
}

impl BlockState {
    pub fn load(model: &Dsv4Model, prefix: &str, layer_id: u32, max_len: usize) -> Self {
        let d = model.cfg();
        let BlockW {
            attn,
            moe,
            attn_norm,
            ffn_norm,
            hc_attn,
            hc_ffn,
        } = BlockW::load(model, prefix, layer_id, max_len);
        BlockState {
            attn: AttnState::new(
                attn,
                d.sliding_window as usize,
                d.head_dim as usize,
                max_len,
            ),
            moe,
            attn_norm,
            ffn_norm,
            hc_attn,
            hc_ffn,
        }
    }

    /// x [s, hc, hidden] -> same shape; prefill iff start_pos == 0 (s may be > 1),
    /// decode otherwise (s == 1).
    pub fn forward(
        &mut self,
        model: &Dsv4Model,
        x: &[f32],
        s: usize,
        ids: &[u32],
        start_pos: usize,
        variant: ActQuantVariant,
    ) -> Vec<f32> {
        let d = model.cfg();
        let hc = d.hc_mult as usize;
        let hidden = model.mc.n_embd as usize;
        let eps = model.mc.rms_eps;
        let iters = d.hc_sinkhorn_iters;
        let hc_eps = d.hc_eps;
        assert!(start_pos == 0 || s == 1, "decode processes one position");

        let (h, post, comb) = hc_pre(x, s, hc, hidden, &self.hc_attn, iters, hc_eps);
        let h = rmsnorm(&h, &self.attn_norm, eps);
        let h = if start_pos == 0 {
            self.attn.prefill(model, &h, s, variant)
        } else {
            self.attn.decode(model, &h, start_pos, variant)
        };
        let x = hc_post(&h, x, s, hc, hidden, &post, &comb);

        let (h, post, comb) = hc_pre(&x, s, hc, hidden, &self.hc_ffn, iters, hc_eps);
        let h = rmsnorm(&h, &self.ffn_norm, eps);
        let h = self.moe.forward(model, &h, s, ids);
        hc_post(&h, &x, s, hc, hidden, &post, &comb)
    }

    /// Batched verify twin of [`Self::forward`] (decode, t positions pos0..pos0+t-1):
    /// hc/norm/MoE are per-position programs (identical arithmetic at any s); only
    /// the attention rides the §3.1 transient-ring/recorded-compressor path.
    pub fn forward_batch(
        &mut self,
        model: &Dsv4Model,
        x: &[f32],
        t: usize,
        ids: &[u32],
        pos0: usize,
        variant: ActQuantVariant,
    ) -> (Vec<f32>, AttnBatchCkpt) {
        let d = model.cfg();
        let hc = d.hc_mult as usize;
        let hidden = model.mc.n_embd as usize;
        let eps = model.mc.rms_eps;
        let iters = d.hc_sinkhorn_iters;
        let hc_eps = d.hc_eps;

        let (h, post, comb) = hc_pre(x, t, hc, hidden, &self.hc_attn, iters, hc_eps);
        let h = rmsnorm(&h, &self.attn_norm, eps);
        let (h, ck) = self.attn.decode_batch(model, &h, t, pos0, variant);
        let x = hc_post(&h, x, t, hc, hidden, &post, &comb);

        let (h, post, comb) = hc_pre(&x, t, hc, hidden, &self.hc_ffn, iters, hc_eps);
        let h = rmsnorm(&h, &self.ffn_norm, eps);
        let h = self.moe.forward(model, &h, t, ids);
        (hc_post(&h, &x, t, hc, hidden, &post, &comb), ck)
    }
}

/// Trunk with decode state + the DSpark tap (M:917-925) + shared head.
pub struct TrunkState {
    pub blocks: Vec<BlockState>,
    pub target_layer_ids: Vec<usize>,
    pub hc_head_set: HcSet,
    pub norm_w: Vec<f32>,
    pub n_trunk: usize,
}

pub struct TrunkStepOut {
    /// last-position logits [vocab]
    pub logits: Vec<f32>,
    /// concat of hc-mean hiddens at the target layers, per position [s, n_targets*hidden]
    pub main_hidden: Vec<f32>,
}

impl TrunkState {
    pub fn load(model: &Dsv4Model, target_layer_ids: &[usize], max_len: usize) -> Self {
        let n_trunk = (model.mc.n_layer - model.mc.nextn_predict_layers) as usize;
        let blocks = (0..n_trunk)
            .map(|lid| BlockState::load(model, &format!("layers.{lid}"), lid as u32, max_len))
            .collect();
        TrunkState {
            blocks,
            target_layer_ids: target_layer_ids.to_vec(),
            hc_head_set: HcSet {
                rows: model.tensor_f32("hc_head_fn").0[0],
                fn_w: model.tensor_f32("hc_head_fn").1,
                base: model.tensor_f32("hc_head_base").1,
                scale: model.tensor_f32("hc_head_scale").1,
            },
            norm_w: model.tensor_f32("norm.weight").1,
            n_trunk,
        }
    }

    /// hc-state mean over the hc dim (the M:917-921 tap): [s, hc, hidden] -> [s, hidden].
    fn hc_mean(h: &[f32], s: usize, hc: usize, hidden: usize) -> Vec<f32> {
        let mut out = vec![0f32; s * hidden];
        for t in 0..s {
            for i in 0..hidden {
                let mut acc = 0f32;
                for c in 0..hc {
                    acc += h[(t * hc + c) * hidden + i];
                }
                out[t * hidden + i] = acc / hc as f32;
            }
        }
        out
    }

    /// One trunk forward (prefill when start_pos == 0, else one-position decode),
    /// returning last-position logits + the per-position DSpark tap.
    pub fn forward(
        &mut self,
        model: &Dsv4Model,
        ids: &[u32],
        start_pos: usize,
        variant: ActQuantVariant,
    ) -> TrunkStepOut {
        let d = model.cfg();
        let hc = d.hc_mult as usize;
        let hidden = model.mc.n_embd as usize;
        let s = ids.len();
        let e = model.embed_rows(ids);
        let mut h = crate::dsv4_forward::hc_expand(&e, s, hc, hidden);
        let mut taps: Vec<Vec<f32>> = vec![Vec::new(); self.target_layer_ids.len()];
        for lid in 0..self.n_trunk {
            h = self.blocks[lid].forward(model, &h, s, ids, start_pos, variant);
            if let Some(k) = self.target_layer_ids.iter().position(|&t| t == lid) {
                taps[k] = Self::hc_mean(&h, s, hc, hidden);
            }
        }
        // concat taps along the channel dim, per position (M:925)
        let n_t = self.target_layer_ids.len();
        let mut main_hidden = vec![0f32; s * n_t * hidden];
        for t in 0..s {
            for (k, tap) in taps.iter().enumerate() {
                main_hidden[(t * n_t + k) * hidden..(t * n_t + k + 1) * hidden]
                    .copy_from_slice(&tap[t * hidden..(t + 1) * hidden]);
            }
        }
        let eps = model.mc.rms_eps;
        let collapsed = hc_head(&h, s, hc, hidden, &self.hc_head_set, eps, d.hc_eps);
        let final_h = rmsnorm(&collapsed, &self.norm_w, eps);
        let logits = model.head_logits(&final_h[(s - 1) * hidden..s * hidden]);
        TrunkStepOut {
            logits,
            main_hidden,
        }
    }
}

/// A whole-trunk verify-round checkpoint: one [`AttnBatchCkpt`] per layer.
pub struct TrunkCkpt {
    layers: Vec<AttnBatchCkpt>,
    pub t: usize,
    pub pos0: usize,
}

/// Batched verify output: logits for ALL t positions + the per-position DSpark tap.
pub struct TrunkVerifyOut {
    /// [t, vocab]
    pub logits: Vec<f32>,
    /// [t, n_targets*hidden]
    pub main_hidden: Vec<f32>,
}

impl TrunkState {
    /// Batched T=k+1 verify forward (§3.1): every layer rides
    /// [`BlockState::forward_batch`]; logits are computed for EVERY position (the
    /// accept walk needs all rows). State for all t positions advances provisionally;
    /// [`Self::commit_batch`] makes the first n positions permanent and rolls back
    /// the rest.
    pub fn verify_batch(
        &mut self,
        model: &Dsv4Model,
        ids: &[u32],
        pos0: usize,
        variant: ActQuantVariant,
    ) -> (TrunkVerifyOut, TrunkCkpt) {
        let d = model.cfg();
        let hc = d.hc_mult as usize;
        let hidden = model.mc.n_embd as usize;
        let t = ids.len();
        let e = model.embed_rows(ids);
        let mut h = crate::dsv4_forward::hc_expand(&e, t, hc, hidden);
        let mut taps: Vec<Vec<f32>> = vec![Vec::new(); self.target_layer_ids.len()];
        let mut layers = Vec::with_capacity(self.n_trunk);
        for lid in 0..self.n_trunk {
            let (nh, ck) = self.blocks[lid].forward_batch(model, &h, t, ids, pos0, variant);
            h = nh;
            layers.push(ck);
            if let Some(k) = self.target_layer_ids.iter().position(|&tl| tl == lid) {
                taps[k] = Self::hc_mean(&h, t, hc, hidden);
            }
        }
        let n_t = self.target_layer_ids.len();
        let mut main_hidden = vec![0f32; t * n_t * hidden];
        for p in 0..t {
            for (k, tap) in taps.iter().enumerate() {
                main_hidden[(p * n_t + k) * hidden..(p * n_t + k + 1) * hidden]
                    .copy_from_slice(&tap[p * hidden..(p + 1) * hidden]);
            }
        }
        let eps = model.mc.rms_eps;
        let collapsed = hc_head(&h, t, hc, hidden, &self.hc_head_set, eps, d.hc_eps);
        let final_h = rmsnorm(&collapsed, &self.norm_w, eps);
        let row0 = model.head_logits(&final_h[..hidden]);
        let vocab = row0.len();
        let mut logits = vec![0f32; t * vocab];
        logits[..vocab].copy_from_slice(&row0);
        for p in 1..t {
            logits[p * vocab..(p + 1) * vocab]
                .copy_from_slice(&model.head_logits(&final_h[p * hidden..(p + 1) * hidden]));
        }
        (
            TrunkVerifyOut {
                logits,
                main_hidden,
            },
            TrunkCkpt { layers, t, pos0 },
        )
    }

    /// Commit the first `n_commit` positions of the round on every layer, rolling
    /// back the rejected suffix (§3.1 invariant).
    pub fn commit_batch(&mut self, ck: &TrunkCkpt, n_commit: usize) {
        for (lid, lck) in ck.layers.iter().enumerate() {
            self.blocks[lid].attn.commit_batch(lck, n_commit);
        }
    }

    /// Reset all decode state to construction state (fresh-twin gate runs).
    pub fn reset_state(&mut self) {
        for b in &mut self.blocks {
            b.attn.reset();
        }
    }
}

/// dsv4 trunk behind the family-generic [`crate::spec_oracle::TrunkOracle`] seam:
/// the tap is the concat of hc-mean hiddens at `target_layer_ids` (M:917-925).
pub struct TrunkOracleAdapter<'m> {
    pub trunk: &'m mut TrunkState,
    pub model: &'m Dsv4Model,
    pub variant: ActQuantVariant,
}

impl crate::spec_oracle::TrunkOracle for TrunkOracleAdapter<'_> {
    fn tap_width(&self) -> usize {
        self.trunk.target_layer_ids.len() * self.model.mc.n_embd as usize
    }
    fn forward(&mut self, ids: &[u32], start_pos: usize) -> (Vec<f32>, Vec<f32>) {
        let o = self.trunk.forward(self.model, ids, start_pos, self.variant);
        (o.logits, o.main_hidden)
    }
}

/// dsv4 trunk behind the BATCHED verify seam
/// ([`crate::spec_oracle::TrunkOracleBatched`]): one open [`TrunkCkpt`] between
/// `verify_batch` and `commit`, §3.1 rollback classes wired per layer.
pub struct TrunkBatchAdapter<'m> {
    pub trunk: &'m mut TrunkState,
    pub model: &'m Dsv4Model,
    pub variant: ActQuantVariant,
    open: Option<TrunkCkpt>,
}

impl<'m> TrunkBatchAdapter<'m> {
    pub fn new(trunk: &'m mut TrunkState, model: &'m Dsv4Model, variant: ActQuantVariant) -> Self {
        TrunkBatchAdapter {
            trunk,
            model,
            variant,
            open: None,
        }
    }
}

impl crate::spec_oracle::TrunkOracle for TrunkBatchAdapter<'_> {
    fn tap_width(&self) -> usize {
        self.trunk.target_layer_ids.len() * self.model.mc.n_embd as usize
    }
    fn forward(&mut self, ids: &[u32], start_pos: usize) -> (Vec<f32>, Vec<f32>) {
        assert!(self.open.is_none(), "forward with an open verify round");
        let o = self.trunk.forward(self.model, ids, start_pos, self.variant);
        (o.logits, o.main_hidden)
    }
}

impl crate::spec_oracle::TrunkOracleBatched for TrunkBatchAdapter<'_> {
    fn verify_batch(&mut self, ids: &[u32], pos0: usize) -> (Vec<f32>, Vec<f32>) {
        assert!(self.open.is_none(), "verify_batch with an open round");
        let (o, ck) = self.trunk.verify_batch(self.model, ids, pos0, self.variant);
        self.open = Some(ck);
        (o.logits, o.main_hidden)
    }
    fn commit(&mut self, n_commit: usize) {
        let ck = self.open.take().expect("commit without an open round");
        self.trunk.commit_batch(&ck, n_commit);
    }
}

/// argmax with the torch convention (first index on exact ties).
pub fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as u32
}

// ============================ tests (pure math) ============================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_idxs_decode_matches_reference() {
        // start_pos < win-1: [0..=start_pos] then -1 padding to width win
        assert_eq!(window_idxs_decode(4, 1), vec![0, 1, -1, -1]);
        // start_pos >= win-1: ring order [sp+1..win) ++ [0..=sp]
        assert_eq!(window_idxs_decode(4, 3), vec![0, 1, 2, 3]);
        assert_eq!(window_idxs_decode(4, 5), vec![2, 3, 0, 1]); // sp = 1
        assert_eq!(window_idxs_decode(4, 8), vec![1, 2, 3, 0]); // sp = 0
    }

    #[test]
    fn compress_idxs_decode_matches_reference() {
        assert_eq!(compress_idxs_decode(4, 3, 100), vec![100]); // (3+1)/4 = 1 block
        assert_eq!(compress_idxs_decode(4, 6, 100), vec![100]);
        assert_eq!(compress_idxs_decode(4, 7, 100), vec![100, 101]);
        assert!(compress_idxs_decode(128, 100, 0).is_empty());
    }

    #[test]
    fn sparse_attn_query_single_row_passthrough() {
        // one valid row + a hugely negative sink: output == that row
        let q = vec![1.0f32; 8];
        let kv = [0.5f32; 8];
        let idxs = vec![0i64, -1];
        let sink = vec![-1e30f32];
        let mut out = vec![0f32; 8];
        sparse_attn_query(&q, 1, 8, &idxs, |_| &kv[..], &sink, 1.0, &mut out);
        for v in &out {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn argmax_first_on_ties() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[-1.0]), 0);
    }

    /// §3.1 rollback machinery on a synthetic compressor: state after
    /// (batched advance of t positions, rollback_replay to n_commit) must be
    /// BIT-identical to a twin that plain-decoded exactly the committed positions —
    /// across emission boundaries, the overlap cur→prev shift, and the full-commit
    /// fast path. (The real-model twin gate runs on the box against the artifact;
    /// this pins the pure state mechanics.)
    #[test]
    fn comp_ckpt_rollback_replay_matches_plain_decode() {
        use crate::dsv4_forward::{CompressorW, precompute_freqs_cis};
        let (hidden, d, rd, ratio) = (8usize, 68usize, 4usize, 2usize);
        let latent = 2 * d; // overlap coff = 2
        let mut lcg = 0x2545F4914F6CDD1Du64;
        let mut rnd = move || {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((lcg >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let w = CompressorW {
            ratio,
            d,
            latent,
            overlap: true,
            rotate: false,
            wkv: (0..latent * hidden).map(|_| rnd()).collect(),
            wgate: (0..latent * hidden).map(|_| rnd()).collect(),
            norm_w: (0..d).map(|_| rnd() + 1.0).collect(),
            ape: (0..ratio * latent).map(|_| rnd()).collect(),
        };
        let fc = precompute_freqs_cis(rd, 64, 0, 10000.0, 1.0, 32.0, 1.0);
        let eps = 1e-6f32;
        let variant = ActQuantVariant::RefFp8Round;
        let max_len = 64usize;
        let n_pre = 5usize; // shared plain-decoded prefix (positions 1..=5)
        let t = 6usize; // round width (positions 6..=11 — crosses 3 emissions)
        let rows: Vec<Vec<f32>> = (0..n_pre + t)
            .map(|_| (0..hidden).map(|_| rnd()).collect())
            .collect();
        for n_commit in 0..=t {
            // twin A: plain decode of prefix + exactly the committed positions
            let mut a = CompressorState::new(&w, max_len);
            for (i, row) in rows[..n_pre + n_commit].iter().enumerate() {
                a.decode(&w, row, hidden, 1 + i, &fc, rd, eps, variant);
            }
            // twin B: plain prefix, then a recorded batch of t + rollback_replay
            let mut b = CompressorState::new(&w, max_len);
            for (i, row) in rows[..n_pre].iter().enumerate() {
                b.decode(&w, row, hidden, 1 + i, &fc, rd, eps, variant);
            }
            let mut ck = b.begin_ckpt();
            for (j, row) in rows[n_pre..].iter().enumerate() {
                b.decode_ck(
                    &w,
                    row,
                    hidden,
                    1 + n_pre + j,
                    &fc,
                    rd,
                    eps,
                    variant,
                    Some(&mut ck),
                );
            }
            b.rollback_replay(&w, &ck, n_commit);
            let (akv, asc, ast) = a.state_views();
            let (bkv, bsc, bst) = b.state_views();
            assert_eq!(a.n_blocks, b.n_blocks, "n_commit {n_commit}: block count");
            assert_eq!(
                akv.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                bkv.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "n_commit {n_commit}: pending kv"
            );
            assert_eq!(
                asc.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                bsc.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "n_commit {n_commit}: pending score"
            );
            assert_eq!(
                ast.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                bst.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "n_commit {n_commit}: live store"
            );
        }
    }

    /// reset() must restore exact construction state.
    #[test]
    fn comp_reset_is_construction_state() {
        use crate::dsv4_forward::{CompressorW, precompute_freqs_cis};
        let (hidden, d, rd, ratio) = (8usize, 68usize, 4usize, 2usize);
        let w = CompressorW {
            ratio,
            d,
            latent: 2 * d,
            overlap: true,
            rotate: false,
            wkv: vec![0.01; 2 * d * hidden],
            wgate: vec![0.02; 2 * d * hidden],
            norm_w: vec![1.0; d],
            ape: vec![0.1; ratio * 2 * d],
        };
        let fc = precompute_freqs_cis(rd, 64, 0, 10000.0, 1.0, 32.0, 1.0);
        let mut s = CompressorState::new(&w, 64);
        let fresh = CompressorState::new(&w, 64);
        let row = vec![0.3f32; hidden];
        for p in 1..6 {
            s.decode(
                &w,
                &row,
                hidden,
                p,
                &fc,
                rd,
                1e-6,
                ActQuantVariant::RefFp8Round,
            );
        }
        s.reset();
        let (skv, ssc, sst) = s.state_views();
        let (fkv, fsc, fst) = fresh.state_views();
        assert_eq!(skv, fkv);
        assert_eq!(
            ssc.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            fsc.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(sst, fst);
        assert_eq!(s.n_blocks, 0);
    }
}
