//! DSpark drafter CPU oracle (lane 10, research/dsv4-dspark-20260818).
//!
//! Semantic source (THE LAW): the 0731 checkpoint's own `inference/model.py` DSpark
//! classes — `get_dspark_topk_idxs` (M:743-747), `DSparkAttention` (M:750-792),
//! `DSparkMarkovHead` (M:795-804), `DSparkConfidenceHead` (M:807-815), `DSparkBlock`
//! (M:818-874), `Transformer.forward_spec` (M:928-936) — mechanism census with
//! file:line cites in darklanes research/deepseek-flash-20260818/DSPARK-SEMANTICS.md.
//! Numeric contract: the lane-2 f32 fixture program (dequantized weights, f64 dots,
//! QAT value sims), governing variant RefFp8Round (0731 ships the reference kernel).
//!
//! Deviation from the reference code shape, stated: the reference writes one main_kv
//! ring row inside DSparkAttention's decode branch (M:783) because its smoke calls
//! forward_spec at every position. Here the ring write is factored into
//! [`DsparkModule::write_rings`], called by the driver after EVERY trunk step — ring
//! content at draft time is identical to the reference's every-step pattern, and it is
//! the shape a verifying engine needs (rings must advance for accepted positions that
//! never get a forward_spec call). `forward_spec` itself never mutates trunk or ring
//! state.

use std::path::Path;

use crate::config::JsonObj;
use crate::dsv4_decode::{argmax, sparse_attn_query};
use crate::dsv4_forward::{
    ActQuantVariant, AttnW, BlockW, Dsv4Model, HcSet, act_quant, apply_rope, dot, hc_expand,
    hc_head, hc_post, hc_pre, matmul, par_rows, rmsnorm,
};

/// The four `dspark_*` config keys (+ derived block count), parsed refuse-by-name from
/// the artifact's config.json and pinned against the DSPARK-SEMANTICS census — a
/// different value is a different semantic program.
#[derive(Debug, Clone)]
pub struct DsparkConfig {
    pub block_size: usize,
    pub noise_token_id: u32,
    pub target_layer_ids: Vec<usize>,
    pub markov_rank: usize,
    pub n_blocks: usize,
}

impl DsparkConfig {
    pub fn load(model_dir: &Path, model: &Dsv4Model) -> Self {
        let txt = std::fs::read_to_string(model_dir.join("config.json")).expect("config.json");
        let j = JsonObj::parse(&txt);
        let req = |k: &str| -> u64 {
            j.u64(k)
                .unwrap_or_else(|| panic!("config.json missing required dspark key {k}"))
        };
        let targets: Vec<usize> = j
            .u64_array("dspark_target_layer_ids")
            .unwrap_or_else(|| panic!("config.json missing dspark_target_layer_ids"))
            .into_iter()
            .map(|x| x as usize)
            .collect();
        let cfg = DsparkConfig {
            block_size: req("dspark_block_size") as usize,
            noise_token_id: req("dspark_noise_token_id") as u32,
            target_layer_ids: targets,
            markov_rank: req("dspark_markov_rank") as usize,
            n_blocks: model.mc.nextn_predict_layers as usize,
        };
        // census pins (DSPARK-SEMANTICS §0/§1); refuse on drift
        assert_eq!(cfg.block_size, 5, "dspark_block_size != census pin");
        assert_eq!(
            cfg.noise_token_id, 128799,
            "dspark_noise_token_id != census pin"
        );
        assert_eq!(
            cfg.target_layer_ids,
            vec![40, 41, 42],
            "target_layer_ids != census pin"
        );
        assert_eq!(cfg.markov_rank, 256, "dspark_markov_rank != census pin");
        assert_eq!(cfg.n_blocks, 3, "drafter block count != derived 3");
        // NextN-head refusal (prep §1.2): this module is NOT the preview MTP oracle
        assert!(
            model.st.raw("mtp.0.e_proj.weight").is_none(),
            "mtp.0.e_proj present — NextN checkpoint, not a DSpark drafter; refuse"
        );
        cfg
    }
}

/// get_dspark_topk_idxs (M:743-747): ring slots 0..min(win, start_pos+1)-1 then the
/// transient draft kv at win+0..win+block-1. ONE shared row for every draft query —
/// intra-block attention is bidirectional (semantics §1.3).
pub fn dspark_topk_idxs(win: usize, block: usize, start_pos: usize) -> Vec<i64> {
    assert!(start_pos > 0, "drafting requires start_pos > 0 (M:745)");
    let mut m: Vec<i64> = (0..win.min(start_pos + 1)).map(|x| x as i64).collect();
    m.extend((0..block).map(|j| (win + j) as i64));
    m
}

/// One DSpark block: a full trunk Block (hc + attn + score-routed MoE) under the
/// `mtp.k` prefix (ratio 0: window-only, theta 10000, no YaRN) + the main_kv ring.
pub struct DsparkBlockState {
    pub w: BlockW,
    pub ring: Vec<f32>, // [win, hd]
}

impl DsparkBlockState {
    pub fn load(model: &Dsv4Model, stage: usize, max_len: usize) -> Self {
        let n_trunk = (model.mc.n_layer - model.mc.nextn_predict_layers) as usize;
        let d = model.cfg();
        let layer_id = (n_trunk + stage) as u32;
        assert_eq!(
            d.compress_ratio(layer_id),
            0,
            "dspark block mtp.{stage} must be ratio 0"
        );
        let w = BlockW::load(model, &format!("mtp.{stage}"), layer_id, max_len);
        assert!(w.attn.compressor.is_none() && w.attn.indexer.is_none());
        assert!(!w.moe.hash, "dspark blocks are score-routed (gate.bias)");
        DsparkBlockState {
            w,
            ring: vec![0f32; (d.sliding_window as usize) * (d.head_dim as usize)],
        }
    }

    /// main_kv rows (M:759-761): kv_norm(wkv(main_x)) + rope(real positions) +
    /// group-64 FP8 QAT on the nope dims. main_x [s, hidden].
    fn main_kv_rows(
        &self,
        model: &Dsv4Model,
        main_x: &[f32],
        s: usize,
        positions: &[usize],
        variant: ActQuantVariant,
    ) -> Vec<f32> {
        let d = model.cfg();
        let hidden = model.mc.n_embd as usize;
        let hd = d.head_dim as usize;
        let rd = d.qk_rope_head_dim as usize;
        let eps = model.mc.rms_eps;
        let mut kv = rmsnorm(
            &matmul(main_x, s, hidden, &self.w.attn.wkv, hd),
            &self.w.attn.kv_norm,
            eps,
        );
        apply_rope(&mut kv, s, 1, hd, rd, &self.w.attn.fc, positions, false);
        for row in kv.chunks_exact_mut(hd) {
            act_quant(&mut row[..hd - rd], 64, variant);
        }
        kv
    }

    /// Prefill priming (M:763-769): last min(s, win) positions land at slot p % win;
    /// block body does not run at start_pos == 0.
    pub fn prime_prefill(
        &mut self,
        model: &Dsv4Model,
        main_x: &[f32],
        s: usize,
        variant: ActQuantVariant,
    ) {
        let d = model.cfg();
        let hd = d.head_dim as usize;
        let win = d.sliding_window as usize;
        let positions: Vec<usize> = (0..s).collect();
        let kv = self.main_kv_rows(model, main_x, s, &positions, variant);
        for p in s.saturating_sub(win)..s {
            self.ring[(p % win) * hd..(p % win + 1) * hd]
                .copy_from_slice(&kv[p * hd..(p + 1) * hd]);
        }
    }

    /// The per-accepted-position ring write (module-header deviation note): main_kv of
    /// real position `pos` into slot pos % win (reference site M:783).
    pub fn write_ring(
        &mut self,
        model: &Dsv4Model,
        main_x_row: &[f32],
        pos: usize,
        variant: ActQuantVariant,
    ) {
        let d = model.cfg();
        let hd = d.head_dim as usize;
        let win = d.sliding_window as usize;
        let kv = self.main_kv_rows(model, main_x_row, 1, &[pos], variant);
        self.ring[(pos % win) * hd..(pos % win + 1) * hd].copy_from_slice(&kv);
    }

    /// One draft-block forward (M:695-707 Block body with DSparkAttention M:771-792):
    /// x [block, hc, hidden] -> same shape. Reads the ring; never writes it.
    pub fn forward_draft(
        &self,
        model: &Dsv4Model,
        x: &[f32],
        block: usize,
        start_pos: usize,
        variant: ActQuantVariant,
    ) -> Vec<f32> {
        let d = model.cfg();
        let hc = d.hc_mult as usize;
        let hidden = model.mc.n_embd as usize;
        let eps = model.mc.rms_eps;
        let iters = d.hc_sinkhorn_iters;
        let hc_eps = d.hc_eps;

        let (h, post, comb) = hc_pre(x, block, hc, hidden, &self.w.hc_attn, iters, hc_eps);
        let h = rmsnorm(&h, &self.w.attn_norm, eps);
        let h = self.attn_draft(model, &h, block, start_pos, variant);
        let x = hc_post(&h, x, block, hc, hidden, &post, &comb);

        let (h, post, comb) = hc_pre(&x, block, hc, hidden, &self.w.hc_ffn, iters, hc_eps);
        let h = rmsnorm(&h, &self.w.ffn_norm, eps);
        // score-routed MoE: ids are unused by a non-hash gate (M:581-584)
        let ids = vec![0u32; block];
        let h = self.w.moe.forward(model, &h, block, &ids);
        hc_post(&h, &x, block, hc, hidden, &post, &comb)
    }

    /// DSparkAttention decode branch (M:771-792) minus the ring write.
    fn attn_draft(
        &self,
        model: &Dsv4Model,
        x: &[f32],
        block: usize,
        start_pos: usize,
        variant: ActQuantVariant,
    ) -> Vec<f32> {
        let d = model.cfg();
        let aw: &AttnW = &self.w.attn;
        let hidden = model.mc.n_embd as usize;
        let heads = model.mc.n_head as usize;
        let hd = d.head_dim as usize;
        let rd = d.qk_rope_head_dim as usize;
        let q_lora = d.q_lora_rank as usize;
        let win = d.sliding_window as usize;
        let eps = model.mc.rms_eps;
        // draft positions start_pos+1 .. start_pos+block (M:772)
        let positions: Vec<usize> = (1..=block).map(|j| start_pos + j).collect();

        let qr = rmsnorm(&matmul(x, block, hidden, &aw.wq_a, q_lora), &aw.q_norm, eps);
        let mut q = matmul(&qr, block, q_lora, &aw.wq_b, heads * hd);
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
        apply_rope(&mut q, block, heads, hd, rd, &aw.fc, &positions, false);

        let mut kv = rmsnorm(&matmul(x, block, hidden, &aw.wkv, hd), &aw.kv_norm, eps);
        apply_rope(&mut kv, block, 1, hd, rd, &aw.fc, &positions, false);
        for row in kv.chunks_exact_mut(hd) {
            act_quant(&mut row[..hd - rd], 64, variant);
        }

        let idxs = dspark_topk_idxs(win, block, start_pos);
        let ring = &self.ring;
        let row = |ix: usize| -> &[f32] {
            if ix < win {
                &ring[ix * hd..(ix + 1) * hd]
            } else {
                &kv[(ix - win) * hd..(ix - win + 1) * hd]
            }
        };
        let scale = (hd as f64).powf(-0.5) as f32;
        let mut o = vec![0f32; block * heads * hd];
        for t in 0..block {
            sparse_attn_query(
                &q[t * heads * hd..(t + 1) * heads * hd],
                heads,
                hd,
                &idxs,
                row,
                &aw.sink,
                scale,
                &mut o[t * heads * hd..(t + 1) * heads * hd],
            );
        }
        apply_rope(&mut o, block, heads, hd, rd, &aw.fc, &positions, true);
        // grouped low-rank output projection (M:788-791)
        let o_groups = d.o_groups as usize;
        let o_lora = d.o_lora_rank as usize;
        let gw = heads / o_groups * hd;
        let mut og = vec![0f32; block * o_groups * o_lora];
        for t in 0..block {
            for g in 0..o_groups {
                let src = &o[t * heads * hd + g * gw..t * heads * hd + (g + 1) * gw];
                let wg = &aw.wo_a[g * o_lora * gw..(g + 1) * o_lora * gw];
                let dst = &mut og[(t * o_groups + g) * o_lora..(t * o_groups + g + 1) * o_lora];
                for (r, out_v) in dst.iter_mut().enumerate() {
                    *out_v = dot(src, &wg[r * gw..(r + 1) * gw]);
                }
            }
        }
        matmul(&og, block, o_groups * o_lora, &aw.wo_b, hidden)
    }
}

/// Captured component arrays from one forward_spec call (fixture comparisons).
pub struct SpecCapture {
    pub main_x: Vec<f32>,          // [hidden]
    pub block_outs: Vec<Vec<f32>>, // per block [block_size, hc, hidden]
    pub x_collapsed: Vec<f32>,     // [block_size, hidden] (pre-norm, feeds confidence)
    pub logits_pre: Vec<f32>,      // [block_size, vocab]
    pub logits_post: Vec<f32>,     // [block_size, vocab] (in-place markov bias added)
    pub markov_embed: Vec<f32>,    // [block_size, rank]
}

pub struct SpecOut {
    /// [block_size + 1]: input token then the greedy chained drafts (M:864-871).
    pub out_ids: Vec<u32>,
    /// fp32 confidence per draft position (M:873).
    pub confidence: Vec<f32>,
    /// top1 - top2 margin of each biased logits row (adjudication instrument).
    pub margins: Vec<f32>,
    /// top1 logit value of each biased row (band derivation).
    pub top1_logits: Vec<f32>,
    pub capture: Option<SpecCapture>,
}

pub struct DsparkModule {
    pub cfg: DsparkConfig,
    pub blocks: Vec<DsparkBlockState>,
    pub main_proj: Vec<f32>, // [hidden, n_targets*hidden]
    pub main_norm_w: Vec<f32>,
    pub norm_w: Vec<f32>,
    pub markov_w1: Vec<f32>, // [vocab, rank]
    pub markov_w2: Vec<f32>, // [vocab, rank]
    pub conf_w: Vec<f32>,    // [hidden + rank]
    pub hc_head_set: HcSet,
    pub vocab: usize,
}

impl DsparkModule {
    pub fn load(model: &Dsv4Model, model_dir: &Path, max_len: usize) -> Self {
        let cfg = DsparkConfig::load(model_dir, model);
        let hidden = model.mc.n_embd as usize;
        let blocks = (0..cfg.n_blocks)
            .map(|k| DsparkBlockState::load(model, k, max_len))
            .collect();
        let (mp_shape, main_proj) = model.tensor_f32("mtp.0.main_proj");
        assert_eq!(
            mp_shape,
            vec![hidden, cfg.target_layer_ids.len() * hidden],
            "main_proj shape"
        );
        let last = format!("mtp.{}", cfg.n_blocks - 1);
        let (w1_shape, markov_w1) =
            model.tensor_f32(&format!("{last}.markov_head.markov_w1.weight"));
        let (w2_shape, markov_w2) =
            model.tensor_f32(&format!("{last}.markov_head.markov_w2.weight"));
        let vocab = w1_shape[0];
        assert_eq!(w1_shape[1], cfg.markov_rank, "markov_w1 rank");
        assert_eq!(w2_shape, vec![vocab, cfg.markov_rank], "markov_w2 shape");
        let (cf_shape, conf_w) = model.tensor_f32(&format!("{last}.confidence_head.proj.weight"));
        assert_eq!(
            cf_shape,
            vec![1, hidden + cfg.markov_rank],
            "confidence proj shape"
        );
        DsparkModule {
            blocks,
            main_proj,
            main_norm_w: model.tensor_f32("mtp.0.main_norm.weight").1,
            norm_w: model.tensor_f32(&format!("{last}.norm.weight")).1,
            markov_w1,
            markov_w2,
            conf_w,
            hc_head_set: HcSet {
                rows: model.tensor_f32(&format!("{last}.hc_head_fn")).0[0],
                fn_w: model.tensor_f32(&format!("{last}.hc_head_fn")).1,
                base: model.tensor_f32(&format!("{last}.hc_head_base")).1,
                scale: model.tensor_f32(&format!("{last}.hc_head_scale")).1,
            },
            vocab,
            cfg,
        }
    }

    /// main_x = main_norm(main_proj(main_hidden)) (M:853). main_hidden [s, n_t*hidden].
    pub fn main_x(&self, model: &Dsv4Model, main_hidden: &[f32], s: usize) -> Vec<f32> {
        let hidden = model.mc.n_embd as usize;
        let k = self.cfg.target_layer_ids.len() * hidden;
        rmsnorm(
            &matmul(main_hidden, s, k, &self.main_proj, hidden),
            &self.main_norm_w,
            model.mc.rms_eps,
        )
    }

    pub fn prime_prefill(
        &mut self,
        model: &Dsv4Model,
        main_hidden: &[f32],
        s: usize,
        variant: ActQuantVariant,
    ) {
        let mx = self.main_x(model, main_hidden, s);
        for blk in &mut self.blocks {
            blk.prime_prefill(model, &mx, s, variant);
        }
    }

    /// Per-block ring views for bit-level gate comparison (§3.1 gate: rings written
    /// per ACCEPTED position must equal the every-position sequential twin's).
    pub fn ring_views(&self) -> Vec<&[f32]> {
        self.blocks.iter().map(|b| &b.ring[..]).collect()
    }

    /// Reset drafter state to construction state (fresh-twin gate runs).
    pub fn reset_state(&mut self) {
        for b in &mut self.blocks {
            b.ring.fill(0f32);
        }
    }

    /// Ring advance for one real position (see module header deviation note).
    pub fn write_rings(
        &mut self,
        model: &Dsv4Model,
        main_hidden_row: &[f32],
        pos: usize,
        variant: ActQuantVariant,
    ) {
        let mx = self.main_x(model, main_hidden_row, 1);
        for blk in &mut self.blocks {
            blk.write_ring(model, &mx, pos, variant);
        }
    }

    /// Batched trunk-head logits for n hidden rows (BF16 head rows decoded once).
    fn head_logits_multi(&self, model: &Dsv4Model, xs: &[f32], n: usize) -> Vec<f32> {
        let (info, raw) = model.st.raw("head.weight").expect("head.weight");
        assert_eq!(info.dtype, "BF16");
        let v = info.shape[0] as usize;
        let h = info.shape[1] as usize;
        assert_eq!(xs.len(), n * h);
        // transposed [v, n] so par threads own contiguous chunks, then transpose
        let mut out_t = vec![0f32; v * n];
        par_rows(&mut out_t, n, |j, orow| {
            let row: Vec<f32> = raw[j * h * 2..(j + 1) * h * 2]
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect();
            for (t, o) in orow.iter_mut().enumerate() {
                *o = dot(&xs[t * h..(t + 1) * h], &row);
            }
        });
        let mut out = vec![0f32; n * v];
        for j in 0..v {
            for t in 0..n {
                out[t * v + j] = out_t[j * n + t];
            }
        }
        out
    }

    /// Rank-256 bigram bias (M:801-804): bias[v] = markov_w1[prev] · markov_w2[v].
    fn markov_bias(&self, prev: u32) -> Vec<f32> {
        let rank = self.cfg.markov_rank;
        let emb = &self.markov_w1[prev as usize * rank..(prev as usize + 1) * rank];
        let mut bias = vec![0f32; self.vocab];
        par_rows(&mut bias, 1, |vv, o| {
            o[0] = dot(emb, &self.markov_w2[vv * rank..(vv + 1) * rank]);
        });
        bias
    }

    /// forward_spec (M:928-936) + forward_head (M:860-874): parallel noise-block draft,
    /// sequential markov chaining (greedy, temperature 0), fp32 confidence. Reads the
    /// rings; mutates NOTHING (drafting is side-effect-free — semantics §3).
    pub fn forward_spec(
        &self,
        model: &Dsv4Model,
        input_token: u32,
        main_hidden_row: &[f32],
        start_pos: usize,
        variant: ActQuantVariant,
        capture: bool,
    ) -> SpecOut {
        let d = model.cfg();
        let hc = d.hc_mult as usize;
        let hidden = model.mc.n_embd as usize;
        let eps = model.mc.rms_eps;
        let block = self.cfg.block_size;
        let rank = self.cfg.markov_rank;
        let mx = self.main_x(model, main_hidden_row, 1);
        // draft block: [input_token, noise x (block-1)] via the SHARED trunk embed
        let mut draft_ids = vec![self.cfg.noise_token_id; block];
        draft_ids[0] = input_token;
        let e = model.embed_rows(&draft_ids);
        let mut x = hc_expand(&e, block, hc, hidden);
        let mut block_outs = Vec::new();
        for blk in &self.blocks {
            // side-effect-free: each block reads its own ring; main_x reaches the
            // rings only through prime_prefill/write_rings (main_kv rows).
            x = blk.forward_draft(model, &x, block, start_pos, variant);
            if capture {
                block_outs.push(x.clone());
            }
        }
        let xc = hc_head(&x, block, hc, hidden, &self.hc_head_set, eps, d.hc_eps);
        let normed = rmsnorm(&xc, &self.norm_w, eps);
        let mut logits = self.head_logits_multi(model, &normed, block); // [block, vocab]
        let logits_pre = if capture { Some(logits.clone()) } else { None };
        let mut out_ids = vec![input_token];
        let mut membeds = Vec::with_capacity(block * rank);
        let mut margins = Vec::with_capacity(block);
        let mut top1_logits = Vec::with_capacity(block);
        for i in 0..block {
            let prev = out_ids[i];
            let bias = self.markov_bias(prev);
            let row = &mut logits[i * self.vocab..(i + 1) * self.vocab];
            for (r, b) in row.iter_mut().zip(&bias) {
                *r += *b; // in-place (M:869)
            }
            let top = argmax(row);
            // top-2 margin for near-tie adjudication
            let mut second = f32::NEG_INFINITY;
            for (vv, &val) in row.iter().enumerate() {
                if vv as u32 != top && val > second {
                    second = val;
                }
            }
            margins.push(row[top as usize] - second);
            top1_logits.push(row[top as usize]);
            out_ids.push(top);
            membeds.extend_from_slice(
                &self.markov_w1[prev as usize * rank..(prev as usize + 1) * rank],
            );
        }
        // confidence (M:807-815, M:873): fp32 proj of concat(PRE-norm xc, markov_embed)
        let mut confidence = Vec::with_capacity(block);
        let mut buf = vec![0f32; hidden + rank];
        for i in 0..block {
            buf[..hidden].copy_from_slice(&xc[i * hidden..(i + 1) * hidden]);
            buf[hidden..].copy_from_slice(&membeds[i * rank..(i + 1) * rank]);
            confidence.push(dot(&buf, &self.conf_w));
        }
        SpecOut {
            out_ids,
            confidence,
            margins,
            top1_logits,
            capture: capture.then(|| SpecCapture {
                main_x: mx,
                block_outs,
                x_collapsed: xc,
                logits_pre: logits_pre.unwrap(),
                logits_post: logits,
                markov_embed: membeds,
            }),
        }
    }
}

// ============================ fixture spec (lane-10 JSON contract) ============================

/// Parsed dsv4_dspark_fixtures_ref.json (produced by the torch generator).
pub struct DsparkFixtureSpec {
    pub variant_tag: String,
    pub npz_path: std::path::PathBuf,
    pub prompt: Vec<u32>,
    pub tokens: Vec<u32>,
    pub capture_positions: Vec<usize>,
    pub first_pos: usize,
    pub last_pos: usize,
    /// name -> (shape, sha256)
    pub arrays: std::collections::BTreeMap<String, (Vec<usize>, String)>,
    /// per-array ref-vs-clamp contract fork (max-abs), from the twin pass
    pub fork: std::collections::BTreeMap<String, f64>,
    pub accept_mean: f64,
}

impl DsparkFixtureSpec {
    pub fn load(json_path: &Path) -> Self {
        let txt = std::fs::read_to_string(json_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", json_path.display()));
        let j = JsonObj::parse(&txt);
        let arrays_obj = j.object("arrays").expect("fixture json: arrays");
        let mut arrays = std::collections::BTreeMap::new();
        for (name, _) in arrays_obj.fields() {
            let a = arrays_obj.object(name).expect("array entry");
            let shape: Vec<usize> = a
                .u64_array("shape")
                .expect("array shape")
                .into_iter()
                .map(|x| x as usize)
                .collect();
            arrays.insert(
                name.to_string(),
                (shape, a.string("sha256_le_f32").expect("array sha")),
            );
        }
        let mut fork = std::collections::BTreeMap::new();
        if let Some(fj) = j.object("contract_fork_maxabs") {
            for (name, _) in fj.fields() {
                if let Some(v) = fj.f64(name) {
                    fork.insert(name.to_string(), v);
                }
            }
        }
        DsparkFixtureSpec {
            variant_tag: j.string("variant").expect("variant"),
            npz_path: json_path
                .parent()
                .unwrap_or(Path::new("."))
                .join(j.string("npz").expect("npz")),
            prompt: j.u32_array("prompt").expect("prompt"),
            tokens: j.u32_array("tokens").expect("tokens"),
            capture_positions: j
                .u64_array("capture_positions")
                .expect("capture_positions")
                .into_iter()
                .map(|x| x as usize)
                .collect(),
            first_pos: j.u64("first_pos").expect("first_pos") as usize,
            last_pos: j.u64("last_pos").expect("last_pos") as usize,
            accept_mean: j.f64("accept_mean").unwrap_or(f64::NAN),
            arrays,
            fork,
        }
    }
}

// ============================ spec-oracle family adapters ============================

/// dsv4 DSpark behind the family-generic [`crate::spec_oracle::OracleDrafter`] seam.
/// Family specifics live entirely below this line: the tap is the layer-40/41/42
/// hc-mean concat (12288 wide), `on_commit` is the per-block main_kv ring write, and
/// `propose` is ONE parallel noise-block forward + the sequential markov chaining.
pub struct DsparkOracleAdapter<'m> {
    pub module: &'m mut DsparkModule,
    pub model: &'m Dsv4Model,
    pub variant: ActQuantVariant,
}

impl crate::spec_oracle::OracleDrafter for DsparkOracleAdapter<'_> {
    fn prime_prefill(&mut self, taps: &[f32], s: usize) {
        self.module.prime_prefill(self.model, taps, s, self.variant);
    }
    fn on_commit(&mut self, tap_row: &[f32], pos: usize) {
        self.module
            .write_rings(self.model, tap_row, pos, self.variant);
    }
    fn propose(
        &mut self,
        input_token: u32,
        tap_row: &[f32],
        start_pos: usize,
    ) -> crate::spec_oracle::Proposal {
        let o = self.module.forward_spec(
            self.model,
            input_token,
            tap_row,
            start_pos,
            self.variant,
            false,
        );
        crate::spec_oracle::Proposal {
            out_ids: o.out_ids,
            confidence: o.confidence,
            margins: o.margins,
            top1_logits: o.top1_logits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dspark_topk_idxs_matches_reference() {
        // start_pos 32, win 128: slots 0..=32 then 128..133
        let m = dspark_topk_idxs(128, 5, 32);
        assert_eq!(m.len(), 33 + 5);
        assert_eq!(m[0], 0);
        assert_eq!(m[32], 32);
        assert_eq!(&m[33..], &[128, 129, 130, 131, 132]);
        // saturated ring: all 128 slots
        let m = dspark_topk_idxs(128, 5, 500);
        assert_eq!(m.len(), 128 + 5);
        assert_eq!(m[127], 127);
        assert_eq!(m[128], 128);
    }

    #[test]
    #[should_panic(expected = "start_pos > 0")]
    fn dspark_topk_idxs_refuses_prefill() {
        dspark_topk_idxs(128, 5, 0);
    }
}
