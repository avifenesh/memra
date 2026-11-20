//! The Hy4-flash DFly2 drafter, in f64: one draft round from the target's state.
//!
//! The semantic source is vLLM PR #68296 at 8cbb80efa397 (`vllm/models/hy_v4/nvidia/dfly.py`,
//! `dfly_speculator.py`, `dfly_inputs.py`, `dfly_target_reuse.py`); transformers PR #50903 does not
//! run the drafter. research/modelplan-onboarding-hy4flash-20261119/dfly/make_dfly_oracle.py
//! transcribes that program onto torch on CPU and is this module's oracle.
//!
//! A round starts at the target's last accepted position `p`: the block `[anchor, mask x k]` at
//! positions `p + 1 ..= p + 1 + k` (`anchor` the token the target chose at `p`); slot 0's
//! embedding is fused with the target's final normed hidden row at `p`; the drafter's decoder
//! layers run single-stream (gated MLA with its sink and `post_proj_norm`, a dense SwiGLU), and
//! draft layer `i` attends, per block row `j`, the target KV layer's cached latent and rope key at
//! the slots the target index layer's attention read for `p` (its top-k row; every slot up to `p`
//! while the target attends densely), then block rows `0..=j`. The drafts follow the final norm:
//! step `s` corrects `hidden[s + 1]` with the embedding of the token before it
//! (`HYV4DFlyHiddenCorrection`) and takes the argmax of the target's head.

use crate::hy_v4_flash::{linear, rope_in_place};

/// The drafter program's geometry and switches (vLLM `_derive_dfly_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct DflyGeometry {
    pub hidden: usize,
    pub heads: usize,
    pub nope: usize,
    pub rope: usize,
    pub value: usize,
    pub q_lora: usize,
    pub kv_lora: usize,
    /// the MLP width, from the checkpoint's shapes
    pub ffn: usize,
    pub correction: usize,
    pub vocab: usize,
    /// `rms_norm_eps` as the program computes with it (its f32 value)
    pub epsilon: f64,
    /// the target index layers' RoPE base (every target layer is an indexed layer)
    pub rope_base: f64,
    pub mask_token: u32,
    pub target_index_layers: Vec<u32>,
    pub target_kv_layers: Vec<u32>,
    pub target_index_topk: usize,
    pub local_window: usize,
    /// every indexed trunk layer, ascending (the wrong-layer red arm reads the previous one)
    pub indexed_layers: Vec<u32>,
}

impl DflyGeometry {
    /// The target layer a draft layer reads under `red`: its own, or for [`DflyRed::Layers`] the
    /// indexed layer before it.
    pub fn read_layer(&self, layer: u32, red: DflyRed) -> u32 {
        if red != DflyRed::Layers {
            return layer;
        }
        let at = self
            .indexed_layers
            .iter()
            .position(|l| *l == layer)
            .unwrap_or(0);
        self.indexed_layers[at.saturating_sub(1)]
    }
}

impl DflyGeometry {
    /// From the pack's config and the checkpoint's MLP width.
    pub fn of(hy: &memra_gguf::config::HyV4FlashConfig, ffn: usize) -> Self {
        // every target layer is an indexed layer (the config refuses another), whose RoPE base the
        // drafter's attention uses (`HYV4FlashDFlySpeculatorConfig.target_rope_parameters`)
        let g = hy.indexed;
        Self {
            hidden: hy.hidden_size as usize,
            heads: g.heads as usize,
            nope: g.qk_nope_head_dim as usize,
            rope: g.qk_rope_head_dim as usize,
            value: g.v_head_dim as usize,
            q_lora: hy.q_lora_rank as usize,
            kv_lora: hy.kv_lora_rank as usize,
            ffn,
            correction: hy.dfly.hidden_correction_intermediate_size as usize,
            vocab: hy.vocab_size as usize,
            epsilon: f64::from(hy.rms_norm_eps),
            rope_base: f64::from(g.rope_theta),
            mask_token: hy.dfly.mask_token_id,
            target_index_layers: hy.dfly.target_index_layers.clone(),
            target_kv_layers: hy.dfly.target_kv_layers.clone(),
            target_index_topk: hy.dfly.target_index_topk as usize,
            local_window: hy.dfly.context_local_window as usize,
            indexed_layers: (0..hy.num_hidden_layers)
                .filter(|l| {
                    hy.layer_kinds[*l as usize] == memra_gguf::config::HyV4LayerKind::Indexed
                })
                .collect(),
        }
    }
}

/// One drafter layer's weights, `[out, in]`, dequantized.
#[derive(Debug, Clone, Default)]
pub struct DflyLayerWeights {
    pub input_norm: Vec<f64>,
    pub post_attention_norm: Vec<f64>,
    pub q_a: Vec<f64>,
    pub q_a_norm: Vec<f64>,
    pub q_b: Vec<f64>,
    pub kv_a: Vec<f64>,
    pub kv_a_norm: Vec<f64>,
    pub kv_b: Vec<f64>,
    pub gate: Vec<f64>,
    pub o: Vec<f64>,
    pub post_norm: Vec<f64>,
    pub sinks: Vec<f64>,
    pub mlp_gate: Vec<f64>,
    pub mlp_up: Vec<f64>,
    pub mlp_down: Vec<f64>,
}

/// The drafter's weights (`model.dfly_layers.*`); the embedding and the head are the target's.
#[derive(Debug, Clone, Default)]
pub struct DflyWeights {
    pub layers: Vec<DflyLayerWeights>,
    pub final_norm: Vec<f64>,
    pub eh_proj: Vec<f64>,
    pub enorm: Vec<f64>,
    pub hnorm: Vec<f64>,
    pub fusion_gate: Vec<f64>,
    pub hc_gate: Vec<f64>,
    pub hc_up: Vec<f64>,
    pub hc_down: Vec<f64>,
    pub hc_embed_norm: Vec<f64>,
    pub hc_hidden_norm: Vec<f64>,
}

/// The checkpoint names [`DflyWeights`] reads, in its field order: per layer, then the globals.
pub fn layer_tensor_names(layer: usize) -> [String; 15] {
    let p = format!("model.dfly_layers.{layer}.");
    [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_a_proj.weight",
        "self_attn.q_a_layernorm.weight",
        "self_attn.q_b_proj.weight",
        "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.kv_a_layernorm.weight",
        "self_attn.kv_b_proj.weight",
        "self_attn.linear_gate.weight",
        "self_attn.o_proj.weight",
        "self_attn.post_proj_norm.weight",
        "self_attn.learnable_sink_param",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    .map(|n| format!("{p}{n}"))
}

/// The drafter's global tensors, in [`DflyWeights`]' field order after `layers`.
pub const GLOBAL_TENSOR_NAMES: [&str; 10] = [
    "model.dfly_layers.final_norm.weight",
    "model.dfly_layers.mtp_fusion.eh_proj.weight",
    "model.dfly_layers.mtp_fusion.enorm.weight",
    "model.dfly_layers.mtp_fusion.hnorm.weight",
    "model.dfly_layers.mtp_fusion.gate",
    "model.dfly_layers.hidden_correction.gate_proj.weight",
    "model.dfly_layers.hidden_correction.up_proj.weight",
    "model.dfly_layers.hidden_correction.down_proj.weight",
    "model.dfly_layers.hidden_correction.embed_norm.weight",
    "model.dfly_layers.hidden_correction.hidden_norm.weight",
];

impl DflyWeights {
    /// Refuse weights whose shapes are not the geometry's: an MLP of another width (the
    /// checkpoint's width is the program's; a config default that disagrees reads no weight).
    pub fn check(&self, g: &DflyGeometry) -> Result<(), String> {
        let (h, heads, qk) = (g.hidden, g.heads, g.nope + g.rope);
        let want = |name: &str, got: usize, n: usize| -> Result<(), String> {
            if got != n {
                return Err(format!(
                    "drafter {name}: {got} values, the geometry needs {n}"
                ));
            }
            Ok(())
        };
        for (i, l) in self.layers.iter().enumerate() {
            let at = |n: &str| format!("layer {i} {n}");
            want(&at("q_a"), l.q_a.len(), g.q_lora * h)?;
            want(&at("q_b"), l.q_b.len(), heads * qk * g.q_lora)?;
            want(&at("kv_a"), l.kv_a.len(), (g.kv_lora + g.rope) * h)?;
            want(
                &at("kv_b"),
                l.kv_b.len(),
                heads * (g.nope + g.value) * g.kv_lora,
            )?;
            want(&at("linear_gate"), l.gate.len(), heads * g.value * h)?;
            want(&at("o_proj"), l.o.len(), h * heads * g.value)?;
            want(&at("sinks"), l.sinks.len(), heads)?;
            want(&at("mlp.gate_proj"), l.mlp_gate.len(), g.ffn * h)?;
            want(&at("mlp.up_proj"), l.mlp_up.len(), g.ffn * h)?;
            want(&at("mlp.down_proj"), l.mlp_down.len(), h * g.ffn)?;
        }
        want("eh_proj", self.eh_proj.len(), h * 2 * h)?;
        want(
            "hidden_correction.gate_proj",
            self.hc_gate.len(),
            g.correction * 2 * h,
        )?;
        want(
            "hidden_correction.down_proj",
            self.hc_down.len(),
            h * g.correction,
        )?;
        Ok(())
    }

    /// Read every tensor through `get(name)` (dequantized values, row-major).
    pub fn read(
        layers: usize,
        mut get: impl FnMut(&str) -> Result<Vec<f64>, String>,
    ) -> Result<Self, String> {
        let mut out = Self::default();
        for i in 0..layers {
            let n = layer_tensor_names(i);
            let mut v = n.iter().map(|name| get(name));
            let mut next = || v.next().unwrap();
            out.layers.push(DflyLayerWeights {
                input_norm: next()?,
                post_attention_norm: next()?,
                q_a: next()?,
                q_a_norm: next()?,
                q_b: next()?,
                kv_a: next()?,
                kv_a_norm: next()?,
                kv_b: next()?,
                gate: next()?,
                o: next()?,
                post_norm: next()?,
                sinks: next()?,
                mlp_gate: next()?,
                mlp_up: next()?,
                mlp_down: next()?,
            });
        }
        let mut g = GLOBAL_TENSOR_NAMES.iter().map(|name| get(name));
        let mut next = || g.next().unwrap();
        out.final_norm = next()?;
        out.eh_proj = next()?;
        out.enorm = next()?;
        out.hnorm = next()?;
        out.fusion_gate = next()?;
        out.hc_gate = next()?;
        out.hc_up = next()?;
        out.hc_down = next()?;
        out.hc_embed_norm = next()?;
        out.hc_hidden_norm = next()?;
        Ok(out)
    }
}

/// A wrong drafter program, for the gate's red arms. Each must fail the gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DflyRed {
    #[default]
    None,
    /// every draft layer reads the indexed target layer before its own (21, 25, 29, 33, 37 on the
    /// artifact)
    Layers,
    /// the MLP at half its width (the first half of gate, up and down's channels)
    Width,
    /// hidden correction off: the head reads `hidden[s + 1]` itself
    NoCorrection,
    /// `dfly_target_index_topk` at half the target's: the history keeps the first half of the row
    TopK,
}

impl DflyRed {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "" | "none" => Self::None,
            "layers" => Self::Layers,
            "width" => Self::Width,
            "no-correction" => Self::NoCorrection,
            "topk" => Self::TopK,
            _ => return None,
        })
    }
}

/// What the drafter reads of the target at the source position `p`.
pub struct DflyTarget<'a> {
    /// the target's final normed hidden row at `p` (the head's input), `[hidden]`
    pub hidden: &'a [f64],
    /// per target layer id: its cache rows `[slots, kv_lora + rope]` (normalized latent, rotated
    /// rope key), at least `p + 1` of them
    pub kv: &'a dyn Fn(u32) -> Option<&'a [f64]>,
    /// per target index layer id: the slots its attention read for `p`, ascending
    pub history: &'a dyn Fn(u32) -> Option<Vec<usize>>,
}

/// One draft round's output: the logits of each step (`[k, vocab]`, f32) and the drafter's own
/// argmax of each.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftRound {
    pub logits: Vec<f32>,
    pub drafts: Vec<u32>,
}

fn rms(x: &[f64], weight: &[f64], epsilon: f64) -> Vec<f64> {
    let width = weight.len();
    let mut y = Vec::with_capacity(x.len());
    for row in x.chunks(width) {
        let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f64>() / width as f64 + epsilon).sqrt();
        y.extend(row.iter().zip(weight).map(|(v, w)| v * inv * w));
    }
    y
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The first of equal maxima (`greedy_argmax`, the tapes' rule).
pub fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if v.total_cmp(&row[best]).is_gt() {
            best = i;
        }
    }
    best as u32
}

/// One draft round from source position `p` with `k` drafts. `anchor` is the token the target
/// chose at `p`; `embed(id)` reads the target's embedding row and `head(row)` its head (logits in
/// f64). `forced` replaces each step's previous token after step 0 (the gate's forced runs).
#[allow(clippy::too_many_arguments)]
pub fn draft_round(
    g: &DflyGeometry,
    w: &DflyWeights,
    embed: &dyn Fn(u32) -> Vec<f64>,
    head: &dyn Fn(&[f64]) -> Vec<f64>,
    target: &DflyTarget,
    p: usize,
    anchor: u32,
    k: usize,
    forced: Option<&[u32]>,
    red: DflyRed,
) -> Result<DraftRound, String> {
    let (h, heads, nope, rope, vdim) = (g.hidden, g.heads, g.nope, g.rope, g.value);
    let (ql, kvl) = (g.q_lora, g.kv_lora);
    let qk = nope + rope;
    let rows = k + 1;
    let eps = g.epsilon;
    if w.layers.len() != g.target_kv_layers.len() || g.target_index_layers.len() != w.layers.len() {
        return Err("one target layer per draft layer".into());
    }
    w.check(g)?;
    // the query block and slot 0's fusion with the target's hidden row
    let mut x = Vec::with_capacity(rows * h);
    x.extend(embed(anchor));
    for _ in 0..k {
        x.extend(embed(g.mask_token));
    }
    let cat: Vec<f64> = rms(&x[..h], &w.enorm, eps)
        .into_iter()
        .chain(rms(target.hidden, &w.hnorm, eps))
        .collect();
    let fused = linear(&cat, &w.eh_proj, 2 * h, h);
    for d in 0..h {
        x[d] += w.fusion_gate[d] * fused[d];
    }
    let positions: Vec<u32> = (p as u32 + 1..=p as u32 + 1 + k as u32).collect();
    let mut hidden = x;
    let mut residual: Option<Vec<f64>> = None;
    for (i, lw) in w.layers.iter().enumerate() {
        let index_layer = g.read_layer(g.target_index_layers[i], red);
        let kv_layer = g.read_layer(g.target_kv_layers[i], red);
        if let Some(r) = residual.take() {
            for (a, b) in hidden.iter_mut().zip(&r) {
                *a += b;
            }
        }
        let res = hidden.clone();
        let xn = rms(&hidden, &lw.input_norm, eps);
        let q = linear(
            &rms(&linear(&xn, &lw.q_a, h, ql), &lw.q_a_norm, eps),
            &lw.q_b,
            ql,
            heads * qk,
        );
        let mut q = q;
        let kv = linear(&xn, &lw.kv_a, h, kvl + rope);
        let mut latent = Vec::with_capacity(rows * kvl);
        let mut k_rope = Vec::with_capacity(rows * rope);
        for t in 0..rows {
            latent.extend_from_slice(&kv[t * (kvl + rope)..t * (kvl + rope) + kvl]);
            let mut r = kv[t * (kvl + rope) + kvl..(t + 1) * (kvl + rope)].to_vec();
            rope_in_place(&mut r, positions[t], g.rope_base);
            k_rope.extend(r);
        }
        let latent = rms(&latent, &lw.kv_a_norm, eps);
        for t in 0..rows {
            for hd in 0..heads {
                let at = t * heads * qk + hd * qk + nope;
                rope_in_place(&mut q[at..at + rope], positions[t], g.rope_base);
            }
        }
        // the history: the target index layer's top-k row of `p`, the first `target_index_topk`
        // slots, those before `anchor - local_window`; then the local window, then the block
        let topk = if red == DflyRed::TopK {
            g.target_index_topk / 2
        } else {
            g.target_index_topk
        };
        let row = (target.history)(index_layer)
            .ok_or_else(|| format!("no target history for layer {index_layer}"))?;
        let start = (p + 1).saturating_sub(g.local_window);
        let mut hist: Vec<usize> = row.into_iter().take(topk).filter(|s| *s < start).collect();
        hist.extend(start..p + 1);
        let cache =
            (target.kv)(kv_layer).ok_or_else(|| format!("no target KV for layer {kv_layer}"))?;
        let width = kvl + rope;
        if cache.len() < (p + 1) * width {
            return Err(format!(
                "the target KV of layer {kv_layer} holds fewer than {} slots",
                p + 1
            ));
        }
        // absorbed MLA: q_lat[h] = W_UK[h]^T q_nope[h]; score (q_lat . latent + q_rope . k_rope)
        // * qk^-0.5, the sink in the max and the denominator; out = W_UV[h] (sum p latent)
        let scale = (qk as f64).powf(-0.5);
        let mut attended = vec![0.0; rows * heads * vdim];
        for hd in 0..heads {
            let kvb = &lw.kv_b[hd * (nope + vdim) * kvl..(hd + 1) * (nope + vdim) * kvl];
            for j in 0..rows {
                let qrow = &q[j * heads * qk + hd * qk..j * heads * qk + (hd + 1) * qk];
                let mut q_lat = vec![0.0; kvl];
                for (n, qn) in qrow[..nope].iter().enumerate() {
                    let wr = &kvb[n * kvl..(n + 1) * kvl];
                    for (l, wv) in wr.iter().enumerate() {
                        q_lat[l] += qn * wv;
                    }
                }
                let keys: Vec<(&[f64], &[f64])> = hist
                    .iter()
                    .map(|s| {
                        let r = &cache[s * width..(s + 1) * width];
                        (&r[..kvl], &r[kvl..])
                    })
                    .chain((0..=j).map(|b| {
                        (
                            &latent[b * kvl..(b + 1) * kvl],
                            &k_rope[b * rope..(b + 1) * rope],
                        )
                    }))
                    .collect();
                let logits: Vec<f64> = keys
                    .iter()
                    .map(|(lat, kr)| {
                        let a: f64 = q_lat.iter().zip(*lat).map(|(x, y)| x * y).sum();
                        let b: f64 = qrow[nope..].iter().zip(*kr).map(|(x, y)| x * y).sum();
                        (a + b) * scale
                    })
                    .collect();
                let sink = lw.sinks[hd];
                let max = logits.iter().copied().fold(sink, f64::max);
                let denom: f64 =
                    logits.iter().map(|l| (l - max).exp()).sum::<f64>() + (sink - max).exp();
                let mut out_lat = vec![0.0; kvl];
                for ((lat, _), l) in keys.iter().zip(&logits) {
                    let pr = (l - max).exp() / denom;
                    for (o, v) in out_lat.iter_mut().zip(*lat) {
                        *o += pr * v;
                    }
                }
                for d in 0..vdim {
                    let wr = &kvb[(nope + d) * kvl..(nope + d + 1) * kvl];
                    attended[j * heads * vdim + hd * vdim + d] =
                        wr.iter().zip(&out_lat).map(|(a, b)| a * b).sum();
                }
            }
        }
        let gate = linear(&xn, &lw.gate, h, heads * vdim);
        for (a, gv) in attended.iter_mut().zip(&gate) {
            *a *= sigmoid(*gv);
        }
        let mut hs = rms(
            &linear(&attended, &lw.o, heads * vdim, h),
            &lw.post_norm,
            eps,
        );
        for (a, b) in hs.iter_mut().zip(&res) {
            *a += b;
        }
        let res2 = hs.clone();
        let xn = rms(&hs, &lw.post_attention_norm, eps);
        let ffn = if red == DflyRed::Width {
            g.ffn / 2
        } else {
            g.ffn
        };
        let gw = &lw.mlp_gate[..ffn * h];
        let uw = &lw.mlp_up[..ffn * h];
        let dw: Vec<f64> = (0..h)
            .flat_map(|o| lw.mlp_down[o * g.ffn..o * g.ffn + ffn].iter().copied())
            .collect();
        let gv = linear(&xn, gw, h, ffn);
        let uv = linear(&xn, uw, h, ffn);
        let act: Vec<f64> = gv
            .iter()
            .zip(&uv)
            .map(|(a, b)| a * sigmoid(*a) * b)
            .collect();
        hidden = linear(&act, &dw, ffn, h);
        residual = Some(res2);
    }
    if let Some(r) = residual {
        for (a, b) in hidden.iter_mut().zip(&r) {
            *a += b;
        }
    }
    let out = rms(&hidden, &w.final_norm, eps);
    let mut prev = anchor;
    let mut logits = Vec::with_capacity(k * g.vocab);
    let mut drafts = Vec::with_capacity(k);
    for s in 0..k {
        let hrow = &out[(s + 1) * h..(s + 2) * h];
        let corrected: Vec<f64> = if red == DflyRed::NoCorrection {
            hrow.to_vec()
        } else {
            let comb: Vec<f64> = rms(hrow, &w.hc_hidden_norm, eps)
                .into_iter()
                .chain(rms(&embed(prev), &w.hc_embed_norm, eps))
                .collect();
            let gv = linear(&comb, &w.hc_gate, 2 * h, g.correction);
            let uv = linear(&comb, &w.hc_up, 2 * h, g.correction);
            let act: Vec<f64> = gv
                .iter()
                .zip(&uv)
                .map(|(a, b)| a * sigmoid(*a) * b)
                .collect();
            let delta = linear(&act, &w.hc_down, g.correction, h);
            hrow.iter().zip(&delta).map(|(a, b)| a + b).collect()
        };
        let row: Vec<f32> = head(&corrected).into_iter().map(|v| v as f32).collect();
        let top = argmax(&row);
        drafts.push(top);
        logits.extend(row);
        prev = match forced {
            Some(f) => f[s],
            None => top,
        };
    }
    Ok(DraftRound { logits, drafts })
}

/// The slots a query at `slot` read on an indexed layer, from its call's recalled groups (`None`:
/// the layer attended densely, every slot up to `slot`).
pub fn attended_slots(recalled: Option<&[usize]>, pool: usize, slot: usize) -> Vec<usize> {
    match recalled {
        None => (0..=slot).collect(),
        Some(groups) => {
            let mut v: Vec<usize> = groups
                .iter()
                .flat_map(|g| g * pool..(g + 1) * pool)
                .filter(|s| *s <= slot)
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        }
    }
}

/// The tiny drafter's rounds: the reference trunk on the tiny fixture's main request
/// (`hy_v4_flash_trunk::fixture`), then a draft round from each anchor with the tiny drafter
/// (`hy_v4_flash_trunk::dfly_fixture`, `TINY_DFLY_CONFIG`). Returns the request's tokens, `k`
/// and the rounds.
pub fn tiny_rounds(
    anchors: &[usize],
    red: DflyRed,
) -> Result<(Vec<u32>, usize, DflyGeometry, Vec<DraftRound>), String> {
    use crate::hy_v4_flash_trunk as trunk;
    use memra_gguf::model_packs::hy_v4_flash as pack;
    let plan = memra_gguf::model_packs::by_alias("hy_v4_flash")
        .ok_or("no hy_v4_flash pack")?
        .compile_tiny_plan()
        .map_err(|e| format!("{e:?}"))?;
    let dfly = trunk::dfly_fixture().map_err(|e| format!("{e:?}"))?;
    let fixture = trunk::fixture(&plan).map_err(|e| format!("{e:?}"))?;
    let hy = &dfly.hy;
    let g = DflyGeometry::of(hy, pack::dfly_ffn_width(hy) as usize);
    let want: Vec<u32> = hy
        .dfly
        .target_kv_layers
        .iter()
        .chain(&hy.dfly.target_index_layers)
        .map(|l| g.read_layer(*l, red))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let tokens = fixture.token_ids.clone();
    let (_, _, inputs) =
        trunk::execute_for_draft(&plan, &fixture.weights, &tokens, fixture.decode_from, &want)
            .map_err(|e| format!("{e:?}"))?;
    let w = DflyWeights::read(hy.dfly.num_hidden_layers as usize, |name| {
        dfly.weights
            .get(&pack::semantic_tensor_id(name))
            .map(|t| t.data.iter().map(|v| f64::from(*v)).collect())
            .ok_or_else(|| format!("the tiny drafter has no {name}"))
    })?;
    let f = |name: &str| -> Result<Vec<f64>, String> {
        Ok(fixture
            .weights
            .get(&pack::semantic_tensor_id(name))
            .ok_or_else(|| format!("the tiny trunk has no {name}"))?
            .data
            .iter()
            .map(|v| f64::from(*v))
            .collect())
    };
    let (embed_t, head_t) = (f("model.embed_tokens.weight")?, f("lm_head.weight")?);
    let h = g.hidden;
    let embed = |id: u32| embed_t[id as usize * h..(id as usize + 1) * h].to_vec();
    let head = |row: &[f64]| linear(row, &head_t, h, g.vocab);
    let k = hy
        .dfly
        .num_speculative_tokens
        .ok_or("the tiny drafter config declares its drafts")? as usize;
    let mut rounds = Vec::with_capacity(anchors.len());
    for &p in anchors {
        if p + 2 > tokens.len() {
            return Err(format!("anchor {p}: no token after it"));
        }
        let kvf = |l: u32| {
            inputs
                .kv
                .iter()
                .find(|(x, _)| *x == l)
                .map(|(_, v)| v.as_slice())
        };
        let hf = |l: u32| {
            inputs
                .history
                .iter()
                .find(|(x, _)| *x == l)
                .map(|(_, v)| v[p].clone())
        };
        let target = DflyTarget {
            hidden: &inputs.hidden[p * h..(p + 1) * h],
            kv: &kvf,
            history: &hf,
        };
        rounds.push(draft_round(
            &g,
            &w,
            &embed,
            &head,
            &target,
            p,
            tokens[p + 1],
            k,
            None,
            red,
        )?);
    }
    Ok((tokens, k, g, rounds))
}
