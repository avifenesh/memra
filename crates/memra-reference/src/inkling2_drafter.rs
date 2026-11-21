//! Inkling 2 Mini DSpark drafter (`mtp.safetensors`): the unfused f64 reference of
//! `Inkling2DrafterPlan` (memra-gguf `model_plan::inkling2`).
//!
//! The program, per committed position `p` (the trunk ran the token at `p`):
//! `ctx_l = rms(sum_t alpha[l][t] * tap_t(p), dflare_norm_l)`, `k = rope(rms_head(k_ctx(ctx_l)))`,
//! `v = v_ctx(ctx_l)`, written at `p` for each draft layer `l`. Per proposal block at `P` (the
//! number of committed positions): row 0 is the anchor's normed trunk embedding, rows 1.. the
//! learned mask; three pre-norm layers with bidirectional block attention over the context keys
//! (local layers: the `local_context` positions before `P`); the final norm; then the recurrent
//! head over rows `1..rows`, one proposal per row, each the token at its own position.
//!
//! The oracle is a float64 torch port of the vendor serving image's drafter
//! (`research/modelplan-onboarding-inkling2mini-20261121/drafter/dspark_oracle.py` says which
//! parts are ported line by line); this executor matches it on the tiny fixture to f64 rounding.
//! The numeric class is exact f64 over the stored BF16 weights. The image's serving class rounds
//! to bf16 at many points; that class is not implemented here. A proposal never changes a
//! committed token: verification runs the trunk's own program.

use memra_gguf::model_plan::{DrafterHead, Inkling2DrafterPlan, ModelPlan};

/// Geometry and constants of one drafter, read from the plan.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftGeometry {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub rank: usize,
    pub rows: usize,
    pub taps: usize,
    pub local_layers: Vec<usize>,
    pub local_context: usize,
    pub theta: f64,
    pub epsilon: f64,
    pub softcap: f64,
    /// Trunk embedding norm epsilon (the anchor row's norm).
    pub embed_epsilon: f64,
    /// Rows of the trunk's output projection that get logits (the unpadded vocabulary).
    pub vocab: usize,
    /// Rows of `markov_w1` (the padded vocabulary).
    pub padded_vocab: usize,
    /// Attention softmax scale (the plan's `head_dim^-0.5`, taken exactly).
    pub scale: f64,
}

fn decimal(value: f32) -> f64 {
    crate::inkling2::ops::decimal(value)
}

impl DraftGeometry {
    /// The drafter of an inkling2 plan; refuses a plan without one or a head it does not run.
    pub fn of_plan(plan: &ModelPlan) -> Result<Self, String> {
        let program = plan.inkling2.as_ref().ok_or("not an inkling2 plan")?;
        let d: &Inkling2DrafterPlan = &program.drafter;
        let DrafterHead::RnnHidden { rank } = d.head;
        if rank != d.rnn.rank || !d.rnn.folded_through_output || !d.mlp_swiglu {
            return Err("a drafter head or MLP this executor does not run".into());
        }
        use memra_gguf::model_plan::{DraftRowInput, RnnChunk, RnnInput};
        if d.rnn.inputs
            != [
                RnnInput::State,
                RnnInput::PreviousToken,
                RnnInput::RowHidden,
                RnnInput::Feature,
            ]
        {
            return Err(format!(
                "recurrent head inputs {:?}; this executor runs [State, PreviousToken, RowHidden, Feature]",
                d.rnn.inputs
            ));
        }
        if d.rnn.chunks != [RnnChunk::Gate, RnnChunk::Candidate, RnnChunk::Out] {
            return Err(format!(
                "recurrent head chunks {:?}; this executor runs [Gate, Candidate, Out]",
                d.rnn.chunks
            ));
        }
        if d.block.anchor != DraftRowInput::TrunkEmbeddingNormed
            || d.block.mask != DraftRowInput::LearnedMask
        {
            return Err(
                "a draft block whose anchor or mask rows this executor does not build".into(),
            );
        }
        // the plan states the scale in f32; it must be head_dim^-0.5 rounded to f32
        if d.attention.scale != (1.0 / f64::from(d.head_dim).sqrt()) as f32 {
            return Err(format!(
                "attention scale {} is not head_dim^-0.5 for head_dim {}",
                d.attention.scale, d.head_dim
            ));
        }
        if !(d.attention.qk_norm && d.attention.context_key_norm && d.attention.bidirectional_block)
        {
            return Err("a drafter attention this executor does not run".into());
        }
        if d.block.proposals + 1 != d.block.rows || d.fuse.taps as usize != d.taps.len() {
            return Err("a drafter block or fuse shape this executor does not run".into());
        }
        if d.fuse.rows != d.blocks + 1 {
            return Err("a fuse matrix without one row per layer plus the head".into());
        }
        let mut vocab = plan.vocab_size as usize;
        let mut softcap = None;
        for t in &plan.logits {
            match t {
                memra_gguf::model_plan::LogitsTransform::Truncate(n) => vocab = *n as usize,
                memra_gguf::model_plan::LogitsTransform::Softcap(c) => softcap = Some(*c),
                _ => {}
            }
        }
        if softcap != Some(d.rnn.softcap) {
            return Err("the drafter's softcap differs from the trunk's final softcap".into());
        }
        Ok(Self {
            hidden: d.hidden_size as usize,
            layers: d.blocks as usize,
            heads: d.query_heads as usize,
            kv_heads: d.kv_heads as usize,
            head_dim: d.head_dim as usize,
            intermediate: d.intermediate_size as usize,
            rank: rank as usize,
            rows: d.block.rows as usize,
            taps: d.taps.len(),
            local_layers: d
                .attention
                .local_layers
                .iter()
                .map(|l| *l as usize)
                .collect(),
            local_context: d.attention.local_context as usize,
            theta: decimal(d.attention.rope.theta),
            epsilon: decimal(d.epsilon),
            softcap: decimal(d.rnn.softcap),
            embed_epsilon: decimal(program.embedding_norm.epsilon),
            vocab,
            padded_vocab: plan.vocab_size as usize,
            scale: 1.0 / f64::from(d.head_dim).sqrt(),
        })
    }

    fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

/// The 54 drafter tensors as their exact f32 values (BF16 in the checkpoint), named as in
/// `mtp.safetensors`.
pub trait DraftSource {
    fn tensor(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String>;
}

/// One draft layer.
pub struct DraftLayer {
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub fuse_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub o: Vec<f32>,
    pub k_ctx: Vec<f32>,
    pub v_ctx: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub k_ctx_norm: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

/// The drafter's own weights.
pub struct DraftWeights {
    pub layers: Vec<DraftLayer>,
    pub norm: Vec<f32>,
    pub mask: Vec<f32>,
    pub fuse: Vec<f32>,
    pub head_norm: Vec<f32>,
    pub w1: Vec<f32>,
    pub rnn_w: Vec<f32>,
    pub rnn_b: Vec<f32>,
    pub out_w: Vec<f32>,
    pub out_b: Vec<f32>,
}

impl DraftWeights {
    pub fn load(g: &DraftGeometry, src: &dyn DraftSource) -> Result<Self, String> {
        let (h, d, kvw, qw, f, r) = (
            g.hidden,
            g.head_dim,
            g.kv_width(),
            g.heads * g.head_dim,
            g.intermediate,
            g.rank,
        );
        let mut layers = Vec::new();
        for i in 0..g.layers {
            let t = |n: &str, s: &[usize]| src.tensor(&format!("layers.{i}.{n}"), s);
            layers.push(DraftLayer {
                input_norm: t("input_layernorm.weight", &[h])?,
                post_norm: t("post_attention_layernorm.weight", &[h])?,
                fuse_norm: t("dflare_norm.weight", &[h])?,
                q: t("self_attn.q_proj.weight", &[qw, h])?,
                k: t("self_attn.k_proj.weight", &[kvw, h])?,
                v: t("self_attn.v_proj.weight", &[kvw, h])?,
                o: t("self_attn.o_proj.weight", &[h, qw])?,
                k_ctx: t("self_attn.k_proj_ctx.weight", &[kvw, h])?,
                v_ctx: t("self_attn.v_proj_ctx.weight", &[kvw, h])?,
                q_norm: t("self_attn.q_norm.weight", &[d])?,
                k_norm: t("self_attn.k_norm.weight", &[d])?,
                k_ctx_norm: t("self_attn.k_norm_ctx.weight", &[d])?,
                gate: t("mlp.gate_proj.weight", &[f, h])?,
                up: t("mlp.up_proj.weight", &[f, h])?,
                down: t("mlp.down_proj.weight", &[h, f])?,
            });
        }
        let m = |n: &str, s: &[usize]| src.tensor(&format!("markov_head.{n}"), s);
        Ok(Self {
            layers,
            norm: src.tensor("norm.weight", &[h])?,
            mask: src.tensor("mask_embed", &[h])?,
            fuse: src.tensor("fuse_weight", &[g.layers + 1, g.taps])?,
            head_norm: m("dflare_norm.weight", &[h])?,
            w1: m("markov_w1.weight", &[g.padded_vocab, r])?,
            rnn_w: m("rnn_proj.weight", &[3 * r, 2 * r + 2 * h])?,
            rnn_b: m("rnn_proj.bias", &[3 * r])?,
            out_w: m("hidden_out.weight", &[h, r])?,
            out_b: m("hidden_out.bias", &[h])?,
        })
    }
}

/// What the drafter reads from the trunk's weights: the token embedding, its norm and the
/// output projection's kept rows (`[vocab, hidden]`).
pub struct TrunkShared<'a> {
    pub embed: &'a [f32],
    pub embed_norm: &'a [f32],
    pub head: &'a [f32],
}

/// One proposal block.
#[derive(Debug, Clone)]
pub struct Proposal {
    /// `[rows, hidden]`: the final norm output of every block row.
    pub hidden: Vec<f64>,
    pub feature: Vec<f64>,
    /// `[rows - 1, vocab]` softcapped logits, one row per proposal.
    pub logits: Vec<f64>,
    pub tokens: Vec<u32>,
}

/// The drafter with its context K/V.
pub struct Drafter {
    pub g: DraftGeometry,
    pub w: DraftWeights,
    /// `softmax(fuse_weight)` per row, `[layers + 1, taps]`.
    pub alpha: Vec<f64>,
    /// Per layer, `[positions, kv_width]`.
    pub ctx_k: Vec<Vec<f64>>,
    pub ctx_v: Vec<Vec<f64>>,
    /// Wrong programs the oracle must tell apart (test arms; all off by default).
    pub red: DraftRed,
}

/// Red arms of the drafter's own gates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DraftRed {
    /// `latent = tanh(new state)` instead of `tanh(out)`.
    pub latent_from_state: bool,
    /// `softcap(base) + bias` instead of `softcap(base + bias)`.
    pub softcap_before_bias: bool,
    /// Context keys without RoPE.
    pub no_rope_context_keys: bool,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn linear(w: &[f32], x: &[f64]) -> Vec<f64> {
    crate::inkling2::ops::linear(w, x)
}

fn rms(x: &[f64], w: &[f32], eps: f64) -> Vec<f64> {
    crate::inkling2::ops::rms_norm(x, Some(w), eps)
}

/// NeoX RoPE of one head vector at `position`, angles in f64.
pub fn rope(x: &mut [f64], position: usize, theta: f64) {
    let d = x.len();
    let half = d / 2;
    for i in 0..half {
        let freq = theta.powf(-((2 * i) as f64) / d as f64);
        let (sin, cos) = (position as f64 * freq).sin_cos();
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * cos - b * sin;
        x[i + half] = b * cos + a * sin;
    }
}

fn heads_norm_rope(x: &mut [f64], width: usize, w: &[f32], eps: f64, pos: usize, theta: f64) {
    for head in x.chunks_exact_mut(width) {
        let n = rms(head, w, eps);
        head.copy_from_slice(&n);
        rope(head, pos, theta);
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// First index of the largest value.
pub fn argmax(row: &[f64]) -> u32 {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = i;
        }
    }
    best as u32
}

impl Drafter {
    pub fn new(g: DraftGeometry, w: DraftWeights) -> Self {
        let mut alpha = vec![0.0; (g.layers + 1) * g.taps];
        for (row, out) in w
            .fuse
            .chunks_exact(g.taps)
            .zip(alpha.chunks_exact_mut(g.taps))
        {
            let max = row
                .iter()
                .fold(f64::NEG_INFINITY, |m, v| m.max(f64::from(*v)));
            let e: Vec<f64> = row.iter().map(|v| (f64::from(*v) - max).exp()).collect();
            let s: f64 = e.iter().sum();
            for (o, v) in out.iter_mut().zip(&e) {
                *o = v / s;
            }
        }
        let layers = g.layers;
        Self {
            g,
            w,
            alpha,
            ctx_k: vec![Vec::new(); layers],
            ctx_v: vec![Vec::new(); layers],
            red: DraftRed::default(),
        }
    }

    /// Committed positions the context holds.
    pub fn len(&self) -> usize {
        self.ctx_k
            .first()
            .map_or(0, |k| k.len() / self.g.kv_width())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Forget positions `len..` (rejected rows never reach the context; this is for reuse).
    pub fn truncate(&mut self, len: usize) {
        let kvw = self.g.kv_width();
        for (k, v) in self.ctx_k.iter_mut().zip(&mut self.ctx_v) {
            k.truncate(len * kvw);
            v.truncate(len * kvw);
        }
    }

    /// The fuse mix of one position's taps (`[taps, hidden]`) for fuse row `row`, normed.
    pub fn mix(&self, taps: &[f64], row: usize, norm: &[f32]) -> Vec<f64> {
        let h = self.g.hidden;
        let mut m = vec![0.0; h];
        for t in 0..self.g.taps {
            let a = self.alpha[row * self.g.taps + t];
            for (o, x) in m.iter_mut().zip(&taps[t * h..(t + 1) * h]) {
                *o += a * x;
            }
        }
        rms(&m, norm, self.g.epsilon)
    }

    /// Append the context K/V of the next committed position from its taps `[taps, hidden]`.
    pub fn commit(&mut self, taps: &[f64]) {
        let pos = self.len();
        for l in 0..self.g.layers {
            let lw = &self.w.layers[l];
            let ctx = self.mix(taps, l, &lw.fuse_norm);
            let mut k = linear(&lw.k_ctx, &ctx);
            let v = linear(&lw.v_ctx, &ctx);
            if self.red.no_rope_context_keys {
                for head in k.chunks_exact_mut(self.g.head_dim) {
                    let n = rms(head, &lw.k_ctx_norm, self.g.epsilon);
                    head.copy_from_slice(&n);
                }
            } else {
                heads_norm_rope(
                    &mut k,
                    self.g.head_dim,
                    &lw.k_ctx_norm,
                    self.g.epsilon,
                    pos,
                    self.g.theta,
                );
            }
            self.ctx_k[l].extend(k);
            self.ctx_v[l].extend(v);
        }
    }

    fn attention(&self, l: usize, x: &[f64], p0: usize) -> Vec<f64> {
        let g = &self.g;
        let lw = &self.w.layers[l];
        let (rows, d, kvw, qw) = (g.rows, g.head_dim, g.kv_width(), g.heads * g.head_dim);
        let mut q = Vec::with_capacity(rows * qw);
        let mut k = Vec::with_capacity(rows * kvw);
        let mut v = Vec::with_capacity(rows * kvw);
        for r in 0..rows {
            let xr = &x[r * g.hidden..(r + 1) * g.hidden];
            let mut qr = linear(&lw.q, xr);
            let mut kr = linear(&lw.k, xr);
            heads_norm_rope(&mut qr, d, &lw.q_norm, g.epsilon, p0 + r, g.theta);
            heads_norm_rope(&mut kr, d, &lw.k_norm, g.epsilon, p0 + r, g.theta);
            q.extend(qr);
            k.extend(kr);
            v.extend(linear(&lw.v, xr));
        }
        let first = if g.local_layers.contains(&l) {
            p0.saturating_sub(g.local_context)
        } else {
            0
        };
        let key = |pos: usize, kh: usize| -> (&[f64], &[f64]) {
            let (keys, values, row) = if pos < p0 {
                (&self.ctx_k[l], &self.ctx_v[l], pos)
            } else {
                (&k, &v, pos - p0)
            };
            let at = row * kvw + kh * d..row * kvw + (kh + 1) * d;
            (&keys[at.clone()], &values[at])
        };
        let scale = g.scale;
        let group = g.heads / g.kv_heads;
        let mut out = vec![0.0; rows * qw];
        for r in 0..rows {
            for h in 0..g.heads {
                let qh = &q[r * qw + h * d..r * qw + (h + 1) * d];
                let kh = h / group;
                let positions: Vec<usize> = (first..p0 + rows).collect();
                let scores: Vec<f64> = positions
                    .iter()
                    .map(|&p| dot(qh, key(p, kh).0) * scale)
                    .collect();
                let max = scores.iter().fold(f64::NEG_INFINITY, |m, s| m.max(*s));
                let e: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = e.iter().sum();
                let o = &mut out[r * qw + h * d..r * qw + (h + 1) * d];
                for (p, w) in positions.iter().zip(&e) {
                    for (oi, vi) in o.iter_mut().zip(key(*p, kh).1) {
                        *oi += w / sum * vi;
                    }
                }
            }
        }
        out.chunks_exact(qw)
            .flat_map(|row| linear(&lw.o, row))
            .collect()
    }

    /// The block's final-norm rows for an anchor at position `len()`.
    pub fn block(&self, anchor: u32, shared: &TrunkShared<'_>) -> Vec<f64> {
        let g = &self.g;
        let (h, rows, p0) = (g.hidden, g.rows, self.len());
        let a = anchor as usize;
        let embed: Vec<f64> = shared.embed[a * h..(a + 1) * h]
            .iter()
            .map(|v| f64::from(*v))
            .collect();
        let mut x = rms(&embed, shared.embed_norm, g.embed_epsilon);
        for _ in 1..rows {
            x.extend(self.w.mask.iter().map(|v| f64::from(*v)));
        }
        for l in 0..g.layers {
            let lw = &self.w.layers[l];
            let normed: Vec<f64> = x
                .chunks_exact(h)
                .flat_map(|r| rms(r, &lw.input_norm, g.epsilon))
                .collect();
            let o = self.attention(l, &normed, p0);
            for (xi, oi) in x.iter_mut().zip(&o) {
                *xi += oi;
            }
            let mut add = Vec::with_capacity(rows * h);
            for r in x.chunks_exact(h) {
                let n = rms(r, &lw.post_norm, g.epsilon);
                let gate = linear(&lw.gate, &n);
                let up = linear(&lw.up, &n);
                let act: Vec<f64> = gate
                    .iter()
                    .zip(&up)
                    .map(|(gv, uv)| gv / (1.0 + (-gv).exp()) * uv)
                    .collect();
                add.extend(linear(&lw.down, &act));
            }
            for (xi, ai) in x.iter_mut().zip(&add) {
                *xi += ai;
            }
        }
        x.chunks_exact(h)
            .flat_map(|r| rms(r, &self.w.norm, g.epsilon))
            .collect()
    }

    /// Propose `rows - 1` tokens after `anchor` (at position `len()`). `feature_taps` are the
    /// last committed position's taps. `choose(step, logits)` picks each proposal (greedy:
    /// [`argmax`]); `forced`, when given, replaces the token the head conditions on next.
    pub fn propose(
        &self,
        anchor: u32,
        feature_taps: &[f64],
        shared: &TrunkShared<'_>,
        choose: &mut dyn FnMut(usize, &[f64]) -> u32,
        forced: Option<&[u32]>,
    ) -> Proposal {
        let g = &self.g;
        let (h, r, vocab) = (g.hidden, g.rank, g.vocab);
        let hidden = self.block(anchor, shared);
        let feature = self.mix(feature_taps, g.layers, &self.w.head_norm);
        let width = 2 * r + 2 * h;
        let mut state = vec![0.0; r];
        let mut prev = anchor;
        let mut logits = Vec::with_capacity((g.rows - 1) * vocab);
        let mut tokens = Vec::new();
        for i in 0..g.rows - 1 {
            let row = &hidden[(i + 1) * h..(i + 2) * h];
            let p = prev as usize;
            let mut input = state.clone();
            input.extend(self.w.w1[p * r..(p + 1) * r].iter().map(|v| f64::from(*v)));
            input.extend_from_slice(row);
            input.extend_from_slice(&feature);
            let raw: Vec<f64> = self
                .w
                .rnn_w
                .chunks_exact(width)
                .zip(&self.w.rnn_b)
                .map(|(wr, b)| {
                    f64::from(*b)
                        + wr.iter()
                            .zip(&input)
                            .map(|(w, x)| f64::from(*w) * x)
                            .sum::<f64>()
                })
                .collect();
            let mut latent = vec![0.0; r];
            for j in 0..r {
                let gate = sigmoid(raw[j]);
                let cand = raw[r + j].tanh();
                state[j] = gate * state[j] + (1.0 - gate) * cand;
                latent[j] = if self.red.latent_from_state {
                    state[j].tanh()
                } else {
                    raw[2 * r + j].tanh()
                };
            }
            // hidden_out(latent), then the shared head: the fold's value in exact arithmetic
            let u: Vec<f64> = self
                .w
                .out_w
                .chunks_exact(r)
                .zip(&self.w.out_b)
                .map(|(wr, b)| {
                    f64::from(*b)
                        + wr.iter()
                            .zip(&latent)
                            .map(|(w, x)| f64::from(*w) * x)
                            .sum::<f64>()
                })
                .collect();
            let start = logits.len();
            for v in 0..vocab {
                let head = &shared.head[v * h..(v + 1) * h];
                let mut base = 0.0;
                let mut bias = 0.0;
                for ((w, x), uu) in head.iter().zip(row).zip(&u) {
                    base += f64::from(*w) * x;
                    bias += f64::from(*w) * uu;
                }
                logits.push(if self.red.softcap_before_bias {
                    g.softcap * (base / g.softcap).tanh() + bias
                } else {
                    g.softcap * ((base + bias) / g.softcap).tanh()
                });
            }
            let tok = choose(i, &logits[start..]);
            tokens.push(tok);
            prev = forced.map_or(tok, |f| f[i]);
        }
        Proposal {
            hidden,
            feature,
            logits,
            tokens,
        }
    }
}

/// A safetensors image in memory (the tiny fixture and its oracle): BF16, F32, F64 and I64.
pub struct StImage<'a> {
    data: &'a [u8],
    entries: std::collections::BTreeMap<String, (String, Vec<usize>, [usize; 2])>,
}

impl<'a> StImage<'a> {
    pub fn parse(image: &'a [u8]) -> Result<Self, String> {
        let len = image
            .get(..8)
            .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")) as usize)
            .ok_or("a safetensors image starts with its header length")?;
        let header = image
            .get(8..8 + len)
            .and_then(|b| std::str::from_utf8(b).ok())
            .ok_or("a safetensors header is UTF-8 inside the image")?;
        let parsed = memra_gguf::safetensors::parse_header_json_checked(header)
            .map_err(|e| format!("safetensors header: {e:?}"))?;
        let entries = parsed
            .into_iter()
            .map(|(name, info)| {
                let shape = info.shape.iter().map(|d| *d as usize).collect();
                (
                    name,
                    (
                        info.dtype.clone(),
                        shape,
                        [info.data_offsets[0], info.data_offsets[1]],
                    ),
                )
            })
            .collect();
        Ok(Self {
            data: &image[8 + len..],
            entries,
        })
    }

    fn raw(&self, name: &str, dtype: &str) -> Result<(&[u8], &[usize]), String> {
        let (dt, shape, [a, b]) = self
            .entries
            .get(name)
            .ok_or_else(|| format!("no tensor {name}"))?;
        if dt != dtype {
            return Err(format!("{name} is {dt}, not {dtype}"));
        }
        let bytes = self
            .data
            .get(*a..*b)
            .ok_or_else(|| format!("{name} outside the image"))?;
        Ok((bytes, shape))
    }

    pub fn f64s(&self, name: &str) -> Result<Vec<f64>, String> {
        let (b, _) = self.raw(name, "F64")?;
        Ok(b.chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes")))
            .collect())
    }

    pub fn i64s(&self, name: &str) -> Result<Vec<i64>, String> {
        let (b, _) = self.raw(name, "I64")?;
        Ok(b.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")))
            .collect())
    }

    /// A BF16 tensor as exact f32 values, with its shape checked.
    pub fn bf16(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let (b, got) = self.raw(name, "BF16")?;
        if got != shape {
            return Err(format!("{name}: shape {got:?}, expected {shape:?}"));
        }
        Ok(b.chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect())
    }
}

impl DraftSource for StImage<'_> {
    fn tensor(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        self.bf16(name, shape)
    }
}

/// The tiny fixture's geometry (`drafter/dspark_oracle.py` TINY).
pub fn tiny_geometry() -> DraftGeometry {
    DraftGeometry {
        hidden: 64,
        layers: 3,
        heads: 4,
        kv_heads: 2,
        head_dim: 16,
        intermediate: 96,
        rank: 8,
        rows: 8,
        taps: 4,
        local_layers: vec![0, 2],
        local_context: 5,
        theta: 1e6,
        epsilon: 1e-6,
        softcap: 30.0,
        embed_epsilon: 1e-6,
        vocab: 40,
        padded_vocab: 48,
        scale: 0.25,
    }
}

/// The tiny fixture (weights, trunk tensors, taps, anchors, forced chains) and its pinned
/// float64 oracle.
pub const TINY_FIXTURE: &[u8] = include_bytes!("fixtures/inkling2-drafter-tiny.bin");
pub const TINY_ORACLE: &[u8] = include_bytes!("fixtures/inkling2-drafter-tiny-oracle.bin");
/// Committed positions of each oracle scenario.
pub const TINY_SCENARIOS: [usize; 3] = [1, 4, 13];

#[cfg(test)]
mod tests;
