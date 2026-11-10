//! The V4.1 DSpark draft head on the rank group (vendor `inference/model.py`: `DSparkAttention`,
//! `DSparkMarkovHead`, `DSparkConfidenceHead`, `DSparkBlock`, `Transformer.forward_spec`), and
//! the pieces a draft-and-verify decode loop is made of.
//!
//! What the head reads. For every trunk position, `main_hidden`: the mean over the `hc` stream
//! copies at the INPUT of each target layer (config `dspark_target_layer_ids`, 37..39), the three
//! means side by side ([`Runner::set_taps`], [`decode::Out::taps`]). `main_x =
//! main_norm(main_proj(main_hidden))` is one row per position; every stage turns it into a KV
//! row (`kv_norm(wkv(main_x))`, RoPE at the position, fp8 per-32 quantize-dequantize) kept in
//! the stage's own ring, position p at slot p % window ([`Runner::draft_commit`]).
//!
//! One draft ([`Runner::draft`]) for the token `t` at position p, with the rings holding the
//! positions below p: the block rows are `[t, noise, .., noise]` (`dspark_block_size` rows) at
//! RoPE positions p .. p + block; each stage is a trunk block whose attention reads the first
//! `min(window, p)` ring slots in slot order and then every block row (no causal mask inside
//! the block); the trunk's head on the last stage's collapsed stream (through the stage's own
//! norm) gives each row's logits; row i's token is the argmax of those logits plus the Markov
//! head's bias of the token before it (`t` for row 0), so the picks run in order; row i's token
//! sits at position p + 1 + i. The confidence head scores each row from its collapsed stream and
//! its Markov embedding.
//!
//! Deviations from the vendor's code shape, both stated in memra's DSv4 drafter as well: the ring
//! write is its own step, run for every position the trunk commits (the vendor writes one row
//! inside `forward_spec` because its example calls it at every position), and the draft is
//! always the argmax (a one-hot proposal; the vendor samples at the request temperature). A
//! draft right after the prefill reads the rings the prefill seeded, the same rows the vendor's
//! `forward_spec` at position 0 writes.
//!
//! The stages load as stored: MXFP4 experts (128 per stage, the stage's own top-k), FP8 dense
//! weights, on the ranks' cards (three stages are about 7 GB over the group), partitioned like a
//! trunk layer. The Markov head's vocabulary rows split like the trunk head's.

use super::decode::{Out, Runner};
use super::tp::AttnKind;
use super::{
    AttnKeys, BW, Buf, Ctx, Geo, LayerState, LayerW, Part, QW, Shared, Trace, bf16_to_f32,
    bytes_f32, bytes_i32, ck, copy_2d, copy_d2d, ffi, load_bf16_rows, load_fp8, load_vec_bf16,
};
use memra_gguf::config::ModelConfig;

/// The draft head's shape (config `dspark_*`).
#[derive(Debug, Clone)]
pub struct DraftCfg {
    pub stages: usize,
    pub block: usize,
    pub noise: u32,
    pub targets: Vec<usize>,
    pub markov_rank: usize,
    pub n_experts: usize,
    pub topk: usize,
}

impl DraftCfg {
    pub fn from_config(mc: &ModelConfig) -> Result<Self, String> {
        let c = mc.dsv41.as_ref().ok_or("not a deepseek_v41 config")?;
        let cfg = Self {
            stages: c.num_nextn_predict_layers as usize,
            block: c.dspark_block_size as usize,
            noise: c.dspark_noise_token_id,
            targets: c
                .dspark_target_layer_ids
                .iter()
                .map(|&x| x as usize)
                .collect(),
            markov_rank: c.dspark_markov_rank as usize,
            n_experts: c.dspark_n_routed_experts as usize,
            topk: c.dspark_num_experts_per_tok as usize,
        };
        if cfg.stages == 0 || cfg.block == 0 || cfg.targets.is_empty() {
            return Err("the checkpoint declares no DSpark draft head".into());
        }
        Ok(cfg)
    }
}

/// The draft head's weights on every rank.
pub struct DraftW {
    pub cfg: DraftCfg,
    /// the stages' geometry: the trunk's with the draft MoE's expert count and top-k
    g: Geo,
    /// `[stage][rank]`
    pub stages: Vec<Vec<LayerW>>,
    main_proj: Vec<QW>,
    main_norm: Vec<Buf>,
    norm: Vec<Buf>,
    /// each rank's vocabulary rows of the Markov head `[vocab / n][markov_rank]`
    m_head: Vec<BW>,
    /// the confidence projection `[dim + markov_rank]`
    conf_w: Vec<f32>,
}

/// One sequence's draft state: every stage's ring of `main_x` KV rows on every rank.
pub struct DraftSeq {
    ls: Vec<Vec<LayerState>>,
}

/// What a draft kept for the gates (`dsv41-gate draft`).
#[derive(Default)]
pub struct DraftTrace {
    /// `[stage]`: the stage's output stream bf16 `[block][hc][dim]` and pre-mix f32 `[block][hc]`
    pub h: Vec<Vec<u8>>,
    pub pre: Vec<Vec<f32>>,
    /// `[stage]`: attention input and output, MoE input and output rows, bf16 `[block][dim]`
    pub attn_in: Vec<Vec<u8>>,
    pub attn_out: Vec<Vec<u8>>,
    pub ffn_out: Vec<Vec<u8>>,
    /// `[stage]`: the router's selection `[block][topk]`
    pub route_sel: Vec<Vec<i32>>,
    /// the collapsed last-stage stream bf16 `[block][dim]` (before the norm)
    pub head_x: Vec<u8>,
    /// fp32 `[block][vocab]`: the logits with the Markov bias added, and the bias alone
    pub logits: Vec<f32>,
    pub bias: Vec<f32>,
}

/// One drafted block.
pub struct Draft {
    /// the tokens of positions p + 1 ..= p + block
    pub ids: Vec<u32>,
    /// the confidence head's score of each row (a logit)
    pub conf: Vec<f32>,
    pub trace: Option<DraftTrace>,
}

impl DraftW {
    /// Load the draft head beside `run`'s trunk and turn the trunk's taps on (when the trunk
    /// runs the head's target layers).
    pub fn load(run: &mut Runner, mc: &ModelConfig) -> Result<Self, String> {
        let cfg = DraftCfg::from_config(mc)?;
        let stm = run.stm;
        let n = run.grp.n();
        let mut g = run.g.clone();
        g.n_experts = cfg.n_experts;
        g.topk = cfg.topk;
        Part::check(&g, n)?;
        let trunk = mc
            .dsv41
            .as_ref()
            .ok_or("not a deepseek_v41 config")?
            .trunk_layers() as usize;
        if g.compress_ratios.len() < trunk + cfg.stages
            || g.compress_ratios[trunk..trunk + cfg.stages]
                .iter()
                .any(|&r| r != 0)
        {
            return Err("the draft stages are window-only blocks after the trunk".into());
        }
        let mut stages = Vec::with_capacity(cfg.stages);
        for si in 0..cfg.stages {
            stages.push(run.grp.load_layer_named(
                stm,
                &g,
                &format!("mtp.{si}"),
                trunk + si,
                false,
                true,
            )?);
        }
        let last = cfg.stages - 1;
        let (d, v, mr) = (g.dim, g.vocab, cfg.markov_rank);
        let nt = cfg.targets.len();
        let (mut main_proj, mut main_norm, mut norm, mut m_head) = (vec![], vec![], vec![], vec![]);
        for (r, cx) in run.grp.cx.iter().enumerate() {
            let ops = &cx.ops;
            main_proj.push(load_fp8(ops, stm, "mtp.0.main_proj", d, nt * d)?);
            main_norm.push(load_vec_bf16(ops, stm, "mtp.0.main_norm.weight", d)?);
            norm.push(load_vec_bf16(
                ops,
                stm,
                &format!("mtp.{last}.norm.weight"),
                d,
            )?);
            m_head.push(load_bf16_rows(
                ops,
                stm,
                &format!("mtp.{last}.markov_head.head.weight"),
                v,
                mr,
                Part { rank: r, n }.range(v),
            )?);
        }
        let cname = format!("mtp.{last}.confidence_head.proj.weight");
        let (ci, cb) = stm.raw(&cname).ok_or_else(|| format!("{cname} missing"))?;
        if ci.dtype != "BF16" || ci.shape != [1, (d + mr) as u64] {
            return Err(format!("{cname}: {} {:?}", ci.dtype, ci.shape));
        }
        let ename = format!("mtp.{last}.markov_head.embed.weight");
        let (ei, _) = stm.raw(&ename).ok_or_else(|| format!("{ename} missing"))?;
        if ei.dtype != "BF16" || ei.shape != [v as u64, mr as u64] {
            return Err(format!("{ename}: {} {:?}", ei.dtype, ei.shape));
        }
        // a shortened trunk does not run every target layer: it keeps no taps and the head then
        // drafts only from rows given to it (a tape's)
        if cfg.targets.iter().all(|&l| l < run.n_layers) {
            run.set_taps(&cfg.targets)?;
        }
        if run.head.is_none() {
            run.head = Some(run.grp.load_head(stm, &run.g)?);
        }
        run.grp.sync_all()?;
        Ok(Self {
            cfg,
            g,
            stages,
            main_proj,
            main_norm,
            norm,
            m_head,
            conf_w: bf16_to_f32(cb),
        })
    }

    /// An empty draft state for one sequence.
    pub fn seq(&self, run: &Runner) -> DraftSeq {
        let n = run.grp.n();
        DraftSeq {
            ls: (0..self.cfg.stages)
                .map(|_| (0..n).map(|_| LayerState::default()).collect())
                .collect(),
        }
    }

    /// Device bytes one sequence's draft state takes on each rank.
    pub fn seq_state_bytes(&self) -> usize {
        self.cfg.stages * self.g.window * self.g.head_dim * 2
    }
}

impl DraftSeq {
    /// A stage's ring as rank 0 holds it, bf16 `[window][head_dim]` (gates).
    pub fn ring(&self, run: &Runner, stage: usize) -> Result<Vec<u8>, String> {
        let st = run.grp.cx[0].ops.st.clone();
        self.ls[stage][0]
            .ring
            .as_ref()
            .ok_or("the stage has no ring yet")?
            .host(&st)
    }
}

impl Ctx {
    /// The keys side of a draft stage's attention over its block `x` (normed, `[b][dim]`) at
    /// positions `pos ..`: `qr`, the block's own KV rows after the stage's ring, and the index
    /// rows every block row shares (the first `filled` ring slots, then the block).
    pub fn draft_keys(
        &mut self,
        lw: &LayerW,
        x: &Buf,
        pos: usize,
        b: usize,
        filled: usize,
        ls: &LayerState,
    ) -> Result<AttnKeys, String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let (hd, win) = (g.head_dim, g.window);
        if filled > win {
            return Err(format!("draft: {filled} ring slots of a {win}-slot window"));
        }
        let qr = self.ops.tmp(b * g.q_lora * 2)?;
        self.ops.linear_q(
            x.p(&st),
            g.dim,
            b,
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
            b,
            g.q_lora,
            g.norm_eps,
        )?;
        let ring = ls.ring.as_ref().ok_or("draft: the stage's ring is empty")?;
        let kv_all = self.ops.tmp((win + b) * hd * 2)?;
        copy_d2d(&self.ops, ring, 0, &kv_all, 0, win * hd * 2)?;
        let kv = kv_all.at(&st, win * hd * 2);
        self.ops
            .linear_q(x.p(&st), g.dim, b, &lw.wkv, kv, hd, false, &mut self.scr)?;
        self.ops
            .rmsnorm(kv, hd, &lw.kv_norm, kv, hd, b, hd, g.norm_eps)?;
        self.ops
            .rope(kv, hd, b, 1, hd, &self.rope_base, pos, 1, false)?;
        self.ops.fp8_qdq(kv, hd, b, hd)?;
        let tot = filled + b;
        let mut idx = Vec::with_capacity(b * tot);
        for _ in 0..b {
            idx.extend((0..filled as i32).chain(win as i32..(win + b) as i32));
        }
        Ok(AttnKeys {
            qr,
            kv_all,
            ww: tot,
            tw: 0,
            tot,
            win,
            comp: false,
            dec_rows: false,
            idx: Some(Buf::from_i32(&st, &idx)?),
        })
    }
}

impl<'a> Runner<'a> {
    /// Set every rank's MoE geometry (expert count, top-k) to `g`'s and return what it was.
    fn swap_moe_geo(&mut self, ne: usize, topk: usize) -> (usize, usize) {
        let was = (self.grp.cx[0].g.n_experts, self.grp.cx[0].g.topk);
        for c in self.grp.cx.iter_mut() {
            c.g.n_experts = ne;
            c.g.topk = topk;
        }
        was
    }

    /// Advance the draft rings over trunk positions `pos0 .. pos0 + rows`: rows `row0 ..` of a
    /// forward's taps ([`Out::taps`], rows counted from the call's first token) through
    /// `main_norm(main_proj(.))`, then each stage's `kv_norm(wkv(.))`, RoPE at the position and
    /// the fp8 quantize-dequantize, into the stage's ring. The rows of a prefill seed the rings
    /// (its last `window` positions); the rows of a verified block are written for the kept
    /// tokens only. Returns `main_x` as rank 0 holds it when `keep_main_x` (gates).
    pub fn draft_commit(
        &mut self,
        dw: &DraftW,
        ds: &mut DraftSeq,
        out: &Out,
        row0: usize,
        rows: usize,
        keep_main_x: bool,
    ) -> Result<Option<Vec<u8>>, String> {
        if rows == 0 {
            return Ok(None);
        }
        if out.taps.is_empty() || row0 < out.tap_row0 || row0 + rows > out.tap_row0 + out.tap_rows {
            return Err(format!(
                "draft commit: rows {row0}..{} of a call whose taps hold {}..{}",
                row0 + rows,
                out.tap_row0,
                out.tap_row0 + out.tap_rows
            ));
        }
        let g = self.g.clone();
        let (d, hd, win) = (g.dim, g.head_dim, g.window);
        let nt = dw.cfg.targets.len();
        let pos0 = out.sp + row0;
        let t0 = row0 - out.tap_row0;
        let taps = &out.taps;
        let stages = &dw.stages;
        // `[rank][stage]` views of the rings
        let mut by_rank: Vec<Vec<&mut LayerState>> =
            (0..self.grp.n()).map(|_| Vec::new()).collect();
        for st in ds.ls.iter_mut() {
            for (r, l) in st.iter_mut().enumerate() {
                by_rank[r].push(l);
            }
        }
        let res = self.grp.each_mut(&mut by_rank, |r, cx, rings| {
            let st = cx.ops.st.clone();
            let x = cx.ops.tmp(rows * d * 2)?;
            cx.ops.linear_q(
                taps[r].at(&st, t0 * nt * d * 2),
                nt * d,
                rows,
                &dw.main_proj[r],
                x.p(&st),
                d,
                false,
                &mut cx.scr,
            )?;
            cx.ops.rmsnorm(
                x.p(&st),
                d,
                &dw.main_norm[r],
                x.p(&st),
                d,
                rows,
                d,
                g.norm_eps,
            )?;
            for (si, ring) in rings.iter_mut().enumerate() {
                let lw = &stages[si][r];
                let kv = cx.ops.tmp(rows * hd * 2)?;
                cx.ops.linear_q(
                    x.p(&st),
                    d,
                    rows,
                    &lw.wkv,
                    kv.p(&st),
                    hd,
                    false,
                    &mut cx.scr,
                )?;
                cx.ops.rmsnorm(
                    kv.p(&st),
                    hd,
                    &lw.kv_norm,
                    kv.p(&st),
                    hd,
                    rows,
                    hd,
                    g.norm_eps,
                )?;
                cx.ops
                    .rope(kv.p(&st), hd, rows, 1, hd, &cx.rope_base, pos0, 1, false)?;
                cx.ops.fp8_qdq(kv.p(&st), hd, rows, hd)?;
                if ring.ring.is_none() {
                    ring.ring = Some(cx.ops.buf(win * hd * 2)?);
                }
                Ctx::ring_write(
                    &cx.ops,
                    &kv,
                    ring.ring.as_ref().unwrap(),
                    pos0,
                    rows,
                    win,
                    hd,
                )?;
            }
            if keep_main_x && r == 0 {
                Ok(Some(x.host(&st)?))
            } else {
                Ok(None)
            }
        })?;
        Ok(res.into_iter().next().flatten())
    }

    /// Draft one block for the token `tok` at position `pos` (see the module doc). The rings must
    /// hold every position below `pos` ([`Self::draft_commit`]). `trace` keeps the per-stage
    /// tensors and the full logits for the gates.
    pub fn draft(
        &mut self,
        dw: &DraftW,
        ds: &mut DraftSeq,
        tok: u32,
        pos: usize,
        trace: bool,
    ) -> Result<Draft, String> {
        let was = self.swap_moe_geo(dw.cfg.n_experts, dw.cfg.topk);
        let was_trace = self.grp.trace;
        self.grp.trace = trace;
        for c in self.grp.cx.iter_mut() {
            c.trace = trace;
        }
        let r = self.draft_inner(dw, ds, tok, pos, trace);
        self.swap_moe_geo(was.0, was.1);
        self.grp.trace = was_trace;
        for c in self.grp.cx.iter_mut() {
            c.trace = was_trace;
        }
        r
    }

    fn draft_inner(
        &mut self,
        dw: &DraftW,
        ds: &mut DraftSeq,
        tok: u32,
        pos: usize,
        trace: bool,
    ) -> Result<Draft, String> {
        let g = self.g.clone();
        let n = self.grp.n();
        let (hc, d, win, v) = (g.hc, g.dim, g.window, g.vocab);
        let b = dw.cfg.block;
        let mr = dw.cfg.markov_rank;
        if pos == 0 || pos + b > self.max_pos {
            return Err(format!(
                "draft at position {pos}: a block of {b} past the prompt and inside {}",
                self.max_pos
            ));
        }
        let filled = pos.min(win);
        let mut ids = vec![dw.cfg.noise; b];
        ids[0] = tok;
        let (eb, pm) = self.embed(&ids)?;
        let up = self.grp.each(|_, cx| {
            let e = Buf::from_bytes(&cx.ops.st, &eb)?;
            let hh = cx.ops.buf(b * hc * d * 2)?;
            for k in 0..hc {
                copy_2d(&cx.ops, &e, 0, d * 2, &hh, k * d * 2, hc * d * 2, d * 2, b)?;
            }
            Ok((hh, Buf::from_f32(&cx.ops.st, &pm)?))
        })?;
        let (mut h, mut pre): (Vec<Buf>, Vec<Buf>) = up.into_iter().unzip();
        let mut t = trace.then(DraftTrace::default);
        let st0 = self.grp.cx[0].ops.st.clone();
        for si in 0..dw.cfg.stages {
            let mut sh: Vec<Shared> = (0..n).map(|_| Shared::default()).collect();
            let mut tr: Vec<Trace> = (0..n).map(|_| Trace::default()).collect();
            if let Some(t) = t.as_mut() {
                // the stage's attention input: the collapse and norm the block forward repeats
                let lw = &dw.stages[si][0];
                let x = self.grp.cx[0].hc_pre_norm(&h[0], &pre[0], &lw.attn_norm, b)?;
                t.attn_in.push(x.host(&st0)?);
            }
            let (h2, p2) = self.grp.block_forward_kind(
                &dw.stages[si],
                h,
                &pre,
                pos,
                b,
                &mut sh,
                &mut ds.ls[si],
                &mut tr,
                AttnKind::Draft { filled },
            )?;
            h = h2;
            pre = p2;
            if let Some(t) = t.as_mut() {
                t.h.push(h[0].host(&st0)?);
                t.pre.push(bytes_f32(&pre[0].host(&st0)?));
                let tr0 = &mut tr[0];
                t.attn_out.push(match tr0.attn_out.take() {
                    Some(x) => x.host(&st0)?,
                    None => Vec::new(),
                });
                t.ffn_out.push(match tr0.moe_out.take() {
                    Some(x) => x.host(&st0)?,
                    None => Vec::new(),
                });
                t.route_sel.push(std::mem::take(&mut tr0.route_sel));
            }
        }
        // the head: collapse with the carried pre-mix, the last stage's norm, the trunk's fp32 head
        let (_, hw) = self
            .head
            .as_ref()
            .ok_or("draft: the trunk head is not loaded")?;
        let (hr, pr) = (&h, &pre);
        let heads = self.grp.each(|r, cx| {
            let st = cx.ops.st.clone();
            let x = cx.ops.tmp(b * d * 2)?;
            ck("hc_pre", unsafe {
                ffi::memra_v41_hc_pre(
                    hr[r].p(&st).c(),
                    pr[r].p(&st).f(),
                    b as i32,
                    hc as i32,
                    d as i32,
                    x.p(&st).m(),
                    d as i64,
                    cx.ops.sv(),
                )
            })?;
            let xn = cx.ops.tmp(b * d * 2)?;
            cx.ops
                .rmsnorm(x.p(&st), d, &dw.norm[r], xn.p(&st), d, b, d, g.norm_eps)?;
            let lg = cx.ops.tmp(b * hw[r].n * 4)?;
            cx.ops
                .linear_bf16(xn.p(&st), d, b, &hw[r], lg.p(&st), hw[r].n, true)?;
            Ok((x, lg))
        })?;
        let (xs, lgs): (Vec<Buf>, Vec<Buf>) = heads.into_iter().unzip();
        // the Markov chain of picks: each row's bias comes from the token before it
        let ename = format!("mtp.{}.markov_head.embed.weight", dw.cfg.stages - 1);
        let (_, etab) = self.stm.raw(&ename).ok_or("markov embed missing")?;
        let mut prev = tok as usize;
        let mut out_ids = Vec::with_capacity(b);
        let mut embeds: Vec<f32> = Vec::with_capacity(b * mr);
        let mut bias_all: Vec<f32> = Vec::new();
        for i in 0..b {
            if prev >= v {
                return Err(format!("draft: token {prev} outside the vocabulary"));
            }
            let erow = &etab[prev * mr * 2..(prev + 1) * mr * 2];
            embeds.extend(bf16_to_f32(erow));
            let lg_ref = &lgs;
            let picks = self.grp.each(|r, cx| {
                let st = cx.ops.st.clone();
                let w = &dw.m_head[r];
                let e = Buf::from_bytes(&st, erow)?;
                let bias = cx.ops.tmp(w.n * 4)?;
                cx.ops
                    .linear_bf16(e.p(&st), mr, 1, w, bias.p(&st), w.n, true)?;
                let res = cx.ops.tmp(8)?;
                ck("add_argmax", unsafe {
                    ffi::memra_v41_add_argmax(
                        lg_ref[r].at(&st, i * w.n * 4).fm(),
                        bias.p(&st).f(),
                        w.n as i32,
                        res.p(&st).fm(),
                        res.at(&st, 4).im(),
                        cx.ops.sv(),
                    )
                })?;
                let rb = res.host(&st)?;
                let val = bytes_f32(&rb[..4])[0];
                let idx = bytes_i32(&rb[4..])[0];
                let bias_h = if trace {
                    bytes_f32(&bias.host(&st)?)
                } else {
                    Vec::new()
                };
                Ok((val, idx, bias_h))
            })?;
            // the largest value; among equal ones the lowest id (the ranks hold ascending rows)
            let mut best = (f32::NEG_INFINITY, 0usize);
            let mut col = 0usize;
            for (r, (val, idx, bias_h)) in picks.into_iter().enumerate() {
                if !val.is_finite() {
                    return Err("draft: non-finite logits".into());
                }
                if val > best.0 {
                    best = (val, col + idx as usize);
                }
                col += dw.m_head[r].n;
                bias_all.extend(bias_h);
            }
            prev = best.1;
            out_ids.push(prev as u32);
        }
        // the confidence head: fp32 dot of [collapsed stream | Markov embedding] with its row
        let xh = bf16_to_f32(&xs[0].host(&st0)?);
        let mut conf = Vec::with_capacity(b);
        for i in 0..b {
            let mut acc = 0f32;
            for (a, w) in xh[i * d..(i + 1) * d].iter().zip(&dw.conf_w[..d]) {
                acc += a * w;
            }
            for (a, w) in embeds[i * mr..(i + 1) * mr].iter().zip(&dw.conf_w[d..]) {
                acc += a * w;
            }
            conf.push(acc);
        }
        if let Some(t) = t.as_mut() {
            t.head_x = xs[0].host(&st0)?;
            let mut lg = vec![0f32; b * v];
            let mut col = 0;
            for (r, buf) in lgs.iter().enumerate() {
                let st = self.grp.cx[r].ops.st.clone();
                let part = bytes_f32(&buf.host(&st)?);
                let w = dw.m_head[r].n;
                for i in 0..b {
                    lg[i * v + col..i * v + col + w].copy_from_slice(&part[i * w..(i + 1) * w]);
                }
                col += w;
            }
            t.logits = lg;
            t.bias = bias_all;
        }
        Ok(Draft {
            ids: out_ids,
            conf,
            trace: t,
        })
    }
}

/// What a decode call costs by its rows on the residency that runs it, in milliseconds: a table
/// the caller fills from its own steps (an exponential average per row count), never a constant.
/// The cost of a row depends on where the routed experts live (every expert a block's rows
/// route to is read once: on one card that is most of a row's cost, split across ranks it
/// divides by them, behind an expert cache it includes the fetches), so the table belongs to
/// the route, and a row count nobody has run yet is read off the line through its measured
/// neighbours.
#[derive(Debug, Clone)]
pub struct RowCost {
    ms: Vec<Option<f64>>,
}

impl RowCost {
    /// Weight of a new observation in the running average.
    const ALPHA: f64 = 0.25;

    pub fn new(max_rows: usize) -> Self {
        Self {
            ms: vec![None; max_rows.max(1)],
        }
    }

    /// One call of `rows` rows took `ms`.
    pub fn observe(&mut self, rows: usize, ms: f64) {
        if rows == 0 || rows > self.ms.len() || !ms.is_finite() || ms <= 0.0 {
            return;
        }
        let slot = &mut self.ms[rows - 1];
        *slot = Some(match *slot {
            Some(old) => old + Self::ALPHA * (ms - old),
            None => ms,
        });
    }

    /// The cost of a call of `rows` rows: its own average, or the line through the nearest
    /// measured row counts on either side (the two nearest on one side when it lies outside
    /// them); None until two row counts are measured.
    pub fn ms(&self, rows: usize) -> Option<f64> {
        if rows == 0 || rows > self.ms.len() {
            return None;
        }
        if let Some(v) = self.ms[rows - 1] {
            return Some(v);
        }
        let known: Vec<(f64, f64)> = self
            .ms
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|v| ((i + 1) as f64, v)))
            .collect();
        if known.len() < 2 {
            return None;
        }
        let x = rows as f64;
        let below = known.iter().rev().find(|k| k.0 < x);
        let above = known.iter().find(|k| k.0 > x);
        let (a, b) = match (below, above) {
            (Some(a), Some(b)) => (*a, *b),
            (None, _) => (known[0], known[1]),
            (_, None) => (known[known.len() - 2], known[known.len() - 1]),
        };
        let v = a.1 + (b.1 - a.1) * (x - a.0) / (b.0 - a.0);
        // a call never costs less than the cheapest one measured
        Some(v.max(known.iter().map(|k| k.1).fold(f64::INFINITY, f64::min)))
    }
}

/// Tokens a verification of the first `k` drafts yields in expectation: the token after the
/// block's first row always, and draft i when it and every draft before it hold. The
/// confidence head's sigmoid is the chance a draft holds given that the ones before it do (on
/// the vendor's tapes, 959 drafts: 0.99 held at a sigmoid of 0.9 and up, 0.93 at 0.8..0.9,
/// 0.50 at 0.4..0.5, 0.17 at 0.1..0.2), so the survival of draft i is the product of the
/// sigmoids up to it.
pub fn expected_tokens(conf: &[f32], k: usize) -> f64 {
    let mut alive = 1.0f64;
    let mut tokens = 1.0f64;
    for &c in conf.iter().take(k) {
        alive *= 1.0 / (1.0 + (-(c as f64)).exp());
        tokens += alive;
    }
    tokens
}

/// The drafts to verify this step: the depth, 0 to `max`, with the most expected tokens per
/// millisecond of the step (the verification call of depth + 1 rows plus the draft that
/// follows it, `draft_ms`). A deeper block is taken only when it is strictly better. While the
/// table cannot price a call yet, every draft is verified (those calls fill it).
pub fn price_depth(conf: &[f32], cost: &RowCost, draft_ms: f64, max: usize) -> usize {
    let max = max.min(conf.len());
    let mut best = (0usize, f64::NEG_INFINITY);
    for k in 0..=max {
        let Some(ms) = cost.ms(k + 1) else {
            return max;
        };
        let rate = expected_tokens(conf, k) / (ms + draft_ms.max(0.0));
        if rate > best.1 {
            best = (k, rate);
        }
    }
    best.0
}

/// Whether a sequence drafts at all, from what its drafted steps have yielded: a drafted step
/// costs a draft on top of its verification, so on text whose drafts rarely hold (prose) the
/// undrafted step is the cheaper way to a token. Each drafted step reports its tokens per
/// millisecond against the one-token step's; while their running average stays under one the
/// sequence stops drafting for a stretch (longer each time it happens again), then tries again.
/// It only skips drafts, never changes a token.
#[derive(Debug, Clone)]
pub struct DraftGate {
    gain: f64,
    pause: usize,
    stretch: usize,
}

impl Default for DraftGate {
    fn default() -> Self {
        Self::new()
    }
}

impl DraftGate {
    /// Weight of a new drafted step in the running gain.
    const ALPHA: f64 = 0.25;
    /// Undrafted steps of the first stretch and of the longest (it doubles in between).
    const STRETCH: (usize, usize) = (8, 64);

    pub fn new() -> Self {
        Self {
            gain: 1.0,
            pause: 0,
            stretch: Self::STRETCH.0,
        }
    }

    /// Whether the next step drafts; an undrafted step counts the stretch down.
    pub fn drafts(&mut self) -> bool {
        if self.pause > 0 {
            self.pause -= 1;
            return false;
        }
        true
    }

    /// A drafted step yielded `tokens` in `step_ms` (its verification call and its draft) where
    /// a one-token step costs `plain_ms`.
    pub fn observe(&mut self, tokens: usize, step_ms: f64, plain_ms: f64) {
        if !(step_ms > 0.0 && plain_ms > 0.0) {
            return;
        }
        let gain = tokens as f64 * plain_ms / step_ms;
        self.gain += Self::ALPHA * (gain - self.gain);
        if self.gain < 1.0 {
            self.pause = self.stretch;
            self.stretch = (self.stretch * 2).min(Self::STRETCH.1);
            // the next try starts even
            self.gain = 1.0;
        } else if self.gain > 1.0 + Self::ALPHA {
            self.stretch = Self::STRETCH.0;
        }
    }
}

/// The accept walk of a verified block: `targets[i]` is the target's own token after the block's
/// row i (row 0 is the token the block started from), `drafts[i]` the draft's token for that
/// same position. Returns how many drafts the target agrees with from the front and the token
/// that follows them (the target's own).
pub fn accept_walk(targets: &[u32], drafts: &[u32]) -> (usize, u32) {
    let mut n = 0;
    while n < drafts.len() && n + 1 < targets.len() && targets[n] == drafts[n] {
        n += 1;
    }
    (n, targets[n])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_accept_walk_keeps_the_matching_prefix_and_the_targets_next_token() {
        // target after rows 0..3 says 7, 8, 9, 4; the drafts for those positions are 7, 8, 5
        assert_eq!(accept_walk(&[7, 8, 9, 4], &[7, 8, 5]), (2, 9));
        assert_eq!(accept_walk(&[7, 8, 9, 4], &[7, 8, 9]), (3, 4));
        assert_eq!(accept_walk(&[1, 2], &[3]), (0, 1));
        // a one-row verification (no draft forwarded) yields the target's token
        assert_eq!(accept_walk(&[6], &[]), (0, 6));
    }

    fn logit(p: f64) -> f32 {
        (p / (1.0 - p)).ln() as f32
    }

    /// The row costs of a 9-layer trunk resident on one card, ms (`dspark/README.md`).
    fn resident() -> RowCost {
        let mut c = RowCost::new(6);
        for (rows, ms) in [
            (1, 5.68),
            (2, 6.99),
            (3, 8.20),
            (4, 9.13),
            (5, 11.20),
            (6, 12.06),
        ] {
            c.observe(rows, ms);
        }
        c
    }

    #[test]
    fn a_row_count_nobody_ran_is_read_off_its_measured_neighbours() {
        let mut c = RowCost::new(6);
        assert_eq!(c.ms(1), None);
        c.observe(1, 10.0);
        assert_eq!(c.ms(1), Some(10.0));
        assert_eq!(c.ms(3), None, "one point prices nothing else");
        c.observe(6, 20.0);
        assert_eq!(c.ms(3), Some(14.0));
        c.observe(3, 12.0);
        assert_eq!(c.ms(2), Some(11.0));
        assert_eq!(c.ms(5), Some(12.0 + (20.0 - 12.0) * 2.0 / 3.0));
        // an average, not the last value
        c.observe(1, 14.0);
        assert_eq!(c.ms(1), Some(11.0));
        assert_eq!(c.ms(0), None);
        assert_eq!(c.ms(7), None);
        // outside the measured range: the line through the two nearest, never under the cheapest
        let mut d = RowCost::new(6);
        d.observe(3, 9.0);
        d.observe(4, 10.0);
        assert_eq!(d.ms(6), Some(12.0));
        assert_eq!(d.ms(1), Some(9.0));
    }

    #[test]
    fn expected_tokens_follow_the_survival_of_the_drafts() {
        let conf = [logit(0.5), logit(0.5), logit(0.5)];
        assert!((expected_tokens(&conf, 0) - 1.0).abs() < 1e-9);
        assert!((expected_tokens(&conf, 1) - 1.5).abs() < 1e-6);
        assert!((expected_tokens(&conf, 3) - 1.875).abs() < 1e-6);
        assert!((expected_tokens(&conf, 9) - 1.875).abs() < 1e-6);
    }

    #[test]
    fn confident_drafts_are_verified_whole_and_doubtful_ones_cut_short() {
        let cost = resident();
        // code-like: every draft near certain
        let sure = [
            logit(0.99),
            logit(0.99),
            logit(0.97),
            logit(0.97),
            logit(0.95),
        ];
        assert_eq!(price_depth(&sure, &cost, 2.0, 5), 5);
        // prose-like: the first draft a coin flip, the rest unlikely
        let prose = [
            logit(0.47),
            logit(0.29),
            logit(0.19),
            logit(0.17),
            logit(0.15),
        ];
        let k = price_depth(&prose, &cost, 2.0, 5);
        assert!((1..=2).contains(&k), "prose depth {k}");
        // hopeless drafts: the one-token call
        let none = [logit(0.02); 5];
        assert_eq!(price_depth(&none, &cost, 2.0, 5), 0);
        // the cap holds
        assert_eq!(price_depth(&sure, &cost, 2.0, 3), 3);
        // rows that cost almost nothing more: doubtful drafts are verified as far as they add
        // anything (the fourth's survival is 0.004)
        let mut flat = RowCost::new(6);
        flat.observe(1, 10.0);
        flat.observe(6, 10.5);
        assert_eq!(price_depth(&prose, &flat, 2.0, 5), 3);
        // rows that cost a whole step each: doubtful drafts are not worth one, and even
        // near-certain ones stop paying before the block ends
        let mut steep = RowCost::new(6);
        steep.observe(1, 10.0);
        steep.observe(6, 60.0);
        assert_eq!(price_depth(&prose, &steep, 2.0, 5), 0);
        assert_eq!(price_depth(&sure, &steep, 2.0, 5), 3);
        // an unpriced table verifies everything
        assert_eq!(price_depth(&prose, &RowCost::new(6), 2.0, 5), 5);
    }

    #[test]
    fn a_sequence_whose_drafts_do_not_pay_stops_drafting_for_a_stretch() {
        let mut g = DraftGate::new();
        assert!(g.drafts());
        // one token for a step that costs 1.5 one-token steps: a loss
        g.observe(1, 15.0, 10.0);
        let mut skipped = 0;
        while !g.drafts() {
            skipped += 1;
        }
        assert_eq!(skipped, 8);
        // the same again: a longer stretch
        g.observe(1, 15.0, 10.0);
        let mut skipped = 0;
        while !g.drafts() {
            skipped += 1;
        }
        assert_eq!(skipped, 16);
        // drafts that pay keep it drafting and shorten the next stretch
        for _ in 0..8 {
            g.observe(5, 20.0, 10.0);
            assert!(g.drafts());
        }
        // a run of losses after a good spell: the average has to fall under one first
        for _ in 0..3 {
            g.observe(1, 60.0, 10.0);
            assert!(g.drafts());
        }
        g.observe(1, 60.0, 10.0);
        let mut skipped = 0;
        while !g.drafts() {
            skipped += 1;
        }
        assert_eq!(skipped, 8);
    }
}
