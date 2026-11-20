//! Hy4-flash (`hy_v4_flash`) reference operations, in f64.
//!
//! The semantic source is transformers `modeling_hy_v4_flash.py` (PR #50903 at b865850d0532).
//! Each function is one module of that file and takes its program from the typed plan
//! (`memra_gguf::model_plan::hy_v4_flash`): gated MLA attention ([`gated_mla_attention`],
//! `HYV4FlashAttention` below the indexer budget), the independent hyper-connection
//! ([`hyper_connection`], `HYV4FlashHyperConnection`) and its write-back, the dense SwiGLU MLP
//! (`HYV4FlashMLP`), the latent MoE ([`latent_moe`], `HYV4FlashMoE` with its router and experts),
//! the exit collapse ([`hyper_head`], `HYV4FlashHyperHead`) and the decoder layer that composes
//! them ([`decoder_layer`]). Indexed layers past the indexer budget attend to the groups the
//! pooled indexer recalls ([`index_pool_recall`], `HYV4FlashCompressedIndexer`). Engram is a
//! later step.
//!
//! Activations are row-major `[tokens, width]` (`[tokens, streams, width]` for the streams).
//! Weights are `[out, in]`. Gated by research/modelplan-onboarding-hy4flash-20261119/modules/
//! (a float64 run of the pinned transformers code, compared to 1e-12).

use memra_gguf::model_plan::{
    ActivationPlan, GroupTieOrder, HyV4AttentionKind, HyV4AttentionPlan, HyV4FlashNumerics,
    HyV4ResidualPlan, IndexPoolPlan, LatentMoePlan, NormPlan, RouterPlan,
};

/// `y[t, o] = sum_i x[t, i] * w[o, i]`.
pub fn linear(x: &[f64], w: &[f64], input: usize, out: usize) -> Vec<f64> {
    assert_eq!(w.len(), input * out, "linear weight [{out}, {input}]");
    assert_eq!(x.len() % input, 0, "linear input width {input}");
    let tokens = x.len() / input;
    let mut y = vec![0.0; tokens * out];
    for t in 0..tokens {
        let row = &x[t * input..(t + 1) * input];
        for o in 0..out {
            let wr = &w[o * input..(o + 1) * input];
            y[t * out + o] = row.iter().zip(wr).map(|(a, b)| a * b).sum();
        }
    }
    y
}

/// RMSNorm over each `width` row, then the elementwise weight (`HYV4FlashRMSNorm`).
pub fn rms_norm(x: &[f64], weight: &[f64], norm: NormPlan) -> Vec<f64> {
    let width = weight.len();
    let mut y = Vec::with_capacity(x.len());
    for row in x.chunks(width) {
        let inv = 1.0
            / (row.iter().map(|v| v * v).sum::<f64>() / width as f64 + norm.epsilon as f64).sqrt();
        y.extend(row.iter().zip(weight).map(|(v, w)| w * (v * inv)));
    }
    y
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// Gated MLA attention weights, in the checkpoint's roles.
pub struct AttentionWeights<'a> {
    pub q_a: &'a [f64],
    pub q_a_norm: &'a [f64],
    pub q_b: &'a [f64],
    pub kv_a: &'a [f64],
    pub kv_a_norm: &'a [f64],
    pub kv_b: &'a [f64],
    /// `linear_gate` (transformers `gate_proj`): `[heads * value_head_dim, hidden]`.
    pub gate: &'a [f64],
    pub o: &'a [f64],
    pub post_norm: &'a [f64],
    /// `learnable_sink_param` (transformers `sinks`): one logit per head.
    pub sinks: &'a [f64],
    /// `compressed_indexer.*`: present on indexed layers, absent on sliding ones.
    pub indexer: Option<IndexerWeights<'a>>,
}

/// The pooled indexer's weights (`self_attn.compressed_indexer.*`), `[out, in]`.
pub struct IndexerWeights<'a> {
    /// `query_projection`: `[heads * head_dim, q_lora_rank]`, read from `q_resid`.
    pub query: &'a [f64],
    /// `head_weight_projection`: `[heads, hidden]`.
    pub head_weight: &'a [f64],
    /// `token_key_projection`: `[head_dim, hidden]`.
    pub token_key: &'a [f64],
    pub token_key_norm: &'a [f64],
    /// `key_value_projection` and `key_gate_projection`: `[head_dim, head_dim]`, F32 in the
    /// checkpoint.
    pub key_value: &'a [f64],
    pub key_gate: &'a [f64],
    /// `intra_group_ape`: `[pool, head_dim]`, added to the gate logits of each slot.
    pub ape: &'a [f64],
    pub key_norm: &'a [f64],
}

/// Rotate the trailing `rope` dims of a row in place: adjacent pairs `(2i, 2i+1)` turn by
/// `position * base^(-2i/rope)` (transformers `apply_rotary_pos_emb`; text positions on all three
/// M-RoPE axes make the interleaved recomposition a no-op).
fn rope_in_place(row: &mut [f64], position: u32, base: f64) {
    let dims = row.len();
    for i in 0..dims / 2 {
        let freq = (position as f64) * base.powf(-((2 * i) as f64) / dims as f64);
        let (s, c) = freq.sin_cos();
        let (a, b) = (row[2 * i], row[2 * i + 1]);
        row[2 * i] = a * c - b * s;
        row[2 * i + 1] = b * c + a * s;
    }
}

/// `HYV4FlashCompressedIndexer` token keys: `token_key_norm(token_key_projection(x))`,
/// `[tokens, head_dim]`. These are what the cache keeps per token.
pub fn index_token_keys(
    plan: &IndexPoolPlan,
    norm: NormPlan,
    w: &IndexerWeights,
    x: &[f64],
    hidden: usize,
) -> Vec<f64> {
    let dim = plan.head_dim as usize;
    rms_norm(&linear(x, w.token_key, hidden, dim), w.token_key_norm, norm)
}

/// `HYV4FlashCompressedIndexer.pool_groups`: every complete group of `pool` token keys becomes
/// `key_norm(sum_slot value(k_slot) * softmax_slot(gate(k_slot) + ape_slot))`, elementwise over
/// the head dims (the softmax runs over the slots). A trailing partial group is not pooled.
/// Returns `[groups, head_dim]`.
pub fn index_pool_groups(
    plan: &IndexPoolPlan,
    norm: NormPlan,
    w: &IndexerWeights,
    token_keys: &[f64],
) -> Vec<f64> {
    let dim = plan.head_dim as usize;
    let pool = plan.pool as usize;
    assert_eq!(w.ape.len(), pool * dim, "intra_group_ape [{pool}, {dim}]");
    let groups = token_keys.len() / dim / pool;
    let grouped = &token_keys[..groups * pool * dim];
    let values = linear(grouped, w.key_value, dim, dim);
    let mut gates = linear(grouped, w.key_gate, dim, dim);
    for (i, g) in gates.iter_mut().enumerate() {
        *g += w.ape[i % (pool * dim)];
    }
    let mut pooled = vec![0.0; groups * dim];
    for g in 0..groups {
        for d in 0..dim {
            let at = |slot: usize| (g * pool + slot) * dim + d;
            let max = (0..pool)
                .map(|s| gates[at(s)])
                .fold(f64::NEG_INFINITY, f64::max);
            let denom: f64 = (0..pool).map(|s| (gates[at(s)] - max).exp()).sum();
            pooled[g * dim + d] = (0..pool)
                .map(|s| values[at(s)] * (gates[at(s)] - max).exp() / denom)
                .sum();
        }
    }
    rms_norm(&pooled, w.key_norm, norm)
}

/// The indexer's group logits, `[tokens, groups]`: `sum_h hw[t, h] * relu(q[t, h] . k[g])` with
/// `q = query_projection(q_resid)` and the pooled keys `k`, both rotated on their trailing
/// `rope_head_dim` dims (queries at their position, a group at its first token's) at `rope_base`,
/// and `hw = head_weight_projection(x) * heads^-0.5 * head_dim^-0.5`. A group that has not closed
/// before the query (`g >= t / pool`, its own group included) reads `-inf`. Queries and keys are
/// the same `tokens` (a prefill); `x` is the attention input, `q_resid` the normalized
/// `q_a_proj` output.
#[allow(clippy::too_many_arguments)]
pub fn index_pool_logits(
    plan: &IndexPoolPlan,
    rope_base: f64,
    norm: NormPlan,
    w: &IndexerWeights,
    x: &[f64],
    q_resid: &[f64],
    positions: &[u32],
    hidden: usize,
    q_lora: usize,
) -> Vec<f64> {
    let tokens = positions.len();
    let heads = plan.heads as usize;
    let dim = plan.head_dim as usize;
    let rope = plan.rope_head_dim as usize;
    let pool = plan.pool as usize;
    assert!(rope <= dim, "indexer rope dims {rope} of {dim}");
    let token_keys = index_token_keys(plan, norm, w, x, hidden);
    let mut keys = index_pool_groups(plan, norm, w, &token_keys);
    let groups = keys.len() / dim;
    for g in 0..groups {
        rope_in_place(
            &mut keys[g * dim + dim - rope..(g + 1) * dim],
            positions[g * pool],
            rope_base,
        );
    }
    let mut query = linear(q_resid, w.query, q_lora, heads * dim);
    for (t, &position) in positions.iter().enumerate() {
        for h in 0..heads {
            let row = (t * heads + h) * dim;
            rope_in_place(&mut query[row + dim - rope..row + dim], position, rope_base);
        }
    }
    let scale = (heads as f64).powf(-0.5) * (dim as f64).powf(-0.5);
    let head_weights = linear(x, w.head_weight, hidden, heads);
    let mut logits = vec![f64::NEG_INFINITY; tokens * groups];
    for t in 0..tokens {
        for g in 0..groups.min(t / pool) {
            let key = &keys[g * dim..(g + 1) * dim];
            logits[t * groups + g] = (0..heads)
                .map(|h| {
                    let q = &query[(t * heads + h) * dim..(t * heads + h + 1) * dim];
                    let dot: f64 = q.iter().zip(key).map(|(a, b)| a * b).sum();
                    head_weights[t * heads + h] * scale * dot.max(0.0)
                })
                .reduce(|a, b| a + b)
                .expect("at least one head");
        }
    }
    logits
}

/// The `k` largest of `row` under the plan's tie order, as indices (unordered set semantics).
pub fn group_top_k(row: &[f64], k: usize, ties: GroupTieOrder) -> Vec<usize> {
    let mut order: Vec<usize> = (0..row.len()).collect();
    match ties {
        // A stable sort on the value keeps ascending index among equal values; `total_cmp`
        // orders `-0.0` below `0.0`, as the CUDA radix key does.
        GroupTieOrder::LowerIndexFirst => order.sort_by(|a, b| row[*b].total_cmp(&row[*a])),
    }
    order.truncate(k);
    order
}

/// The keys each query of a prefill may attend on an indexed layer, `[tokens, tokens]` row-major:
/// the `top_k / pool - 1` best groups by [`index_pool_logits`] (taken among all groups, so a
/// query with too few closed groups also takes `-inf` ones, which are never causal-visible) plus
/// its own group, expanded to their token slots (clamped to the last token), intersected with
/// the causal mask. `None` at or below the budget (`tokens <= top_k`), where the layer attends
/// densely and the indexer is skipped.
pub fn index_pool_recall(plan: &IndexPoolPlan, logits: &[f64], tokens: usize) -> Option<Vec<bool>> {
    if tokens <= plan.top_k as usize {
        return None;
    }
    let pool = plan.pool as usize;
    assert_eq!(
        plan.top_k % plan.pool,
        0,
        "top_k {} is whole groups",
        plan.top_k
    );
    let groups = logits.len() / tokens;
    let keep = (plan.top_k / plan.pool) as usize - 1;
    let mut visible = vec![false; tokens * tokens];
    for t in 0..tokens {
        let mut recalled = group_top_k(
            &logits[t * groups..(t + 1) * groups],
            keep.min(groups),
            plan.ties,
        );
        recalled.push(t / pool);
        for g in recalled {
            for slot in g * pool..(g + 1) * pool {
                let s = slot.min(tokens - 1);
                if s <= t {
                    visible[t * tokens + s] = true;
                }
            }
        }
    }
    Some(visible)
}

/// `HYV4FlashAttention.forward` on `x` (the layer input after `input_layernorm`): causal for an
/// indexed layer, narrowed past the indexer budget to the keys [`index_pool_recall`] returns;
/// the last `window` keys for a sliding one. Returns
/// `post_proj_norm(o_proj(attention * sigmoid(gate(x))))`.
pub fn gated_mla_attention(
    plan: &HyV4AttentionPlan,
    norm: NormPlan,
    w: &AttentionWeights,
    x: &[f64],
    positions: &[u32],
    hidden: usize,
) -> Vec<f64> {
    gated_mla_attention_recalled(plan, norm, w, x, positions, hidden, None)
}

/// [`gated_mla_attention`] with the indexed layer's visible keys given (`[tokens, tokens]`)
/// instead of computed; `None` computes them from the plan.
pub fn gated_mla_attention_recalled(
    plan: &HyV4AttentionPlan,
    norm: NormPlan,
    w: &AttentionWeights,
    x: &[f64],
    positions: &[u32],
    hidden: usize,
    recalled: Option<&[bool]>,
) -> Vec<f64> {
    let tokens = positions.len();
    assert_eq!(x.len(), tokens * hidden);
    let heads = plan.query_heads as usize;
    let nope = plan.qk_nope_head_dim as usize;
    let rope = plan.qk_rope_head_dim as usize;
    let vdim = plan.value_head_dim as usize;
    let qk = nope + rope;
    let q_lora = plan.q_lora_rank as usize;
    let kv_lora = plan.kv_lora_rank as usize;
    assert_eq!(
        plan.sink,
        memra_gguf::model_plan::TensorPresence::Required,
        "every layer has sinks"
    );
    assert_eq!(
        plan.rope.factors,
        memra_gguf::model_plan::RopeFactors::None,
        "plain RoPE only"
    );
    assert_eq!(
        plan.rope.dimensions, plan.qk_rope_head_dim,
        "RoPE spans the rope head dims"
    );
    let base = plan.rope.base as f64;

    let q_resid = rms_norm(&linear(x, w.q_a, hidden, q_lora), w.q_a_norm, norm);
    let computed;
    let recalled: Option<&[bool]> = match (plan.kind, &w.indexer) {
        (HyV4AttentionKind::Indexed(index), Some(indexer)) => {
            assert_eq!(
                index.rope_head_dim, plan.qk_rope_head_dim,
                "indexer rope dims"
            );
            match recalled {
                Some(given) => {
                    assert!(
                        tokens > index.top_k as usize,
                        "recalled keys below the budget"
                    );
                    Some(given)
                }
                None => {
                    computed = if tokens > index.top_k as usize {
                        let logits = index_pool_logits(
                            &index, base, norm, indexer, x, &q_resid, positions, hidden, q_lora,
                        );
                        index_pool_recall(&index, &logits, tokens)
                    } else {
                        None
                    };
                    computed.as_deref()
                }
            }
        }
        (HyV4AttentionKind::Sliding { .. }, None) => {
            assert!(recalled.is_none(), "a sliding layer recalls no keys");
            None
        }
        (kind, indexer) => panic!(
            "attention kind {kind:?} with indexer weights present: {}",
            indexer.is_some()
        ),
    };
    if let Some(visible) = recalled {
        assert_eq!(
            visible.len(),
            tokens * tokens,
            "recalled keys [tokens, tokens]"
        );
    }
    let mut q = linear(&q_resid, w.q_b, q_lora, heads * qk);
    let kv_a = linear(x, w.kv_a, hidden, kv_lora + rope);
    let mut latent = Vec::with_capacity(tokens * kv_lora);
    let mut k_rope = Vec::with_capacity(tokens * rope);
    for t in 0..tokens {
        let row = &kv_a[t * (kv_lora + rope)..(t + 1) * (kv_lora + rope)];
        latent.extend_from_slice(&row[..kv_lora]);
        k_rope.extend_from_slice(&row[kv_lora..]);
    }
    let latent = rms_norm(&latent, w.kv_a_norm, norm);
    for t in 0..tokens {
        rope_in_place(&mut k_rope[t * rope..(t + 1) * rope], positions[t], base);
        for h in 0..heads {
            let at = t * heads * qk + h * qk + nope;
            rope_in_place(&mut q[at..at + rope], positions[t], base);
        }
    }
    let kv = linear(&latent, w.kv_b, kv_lora, heads * (nope + vdim));
    let gate = linear(x, w.gate, hidden, heads * vdim);
    let scale = (qk as f64).powf(-0.5);
    let mut attended = vec![0.0; tokens * heads * vdim];
    for h in 0..heads {
        for t in 0..tokens {
            let visible = |s: usize| {
                s <= t
                    && match plan.kind {
                        HyV4AttentionKind::Sliding { window } => t - s < window as usize,
                        HyV4AttentionKind::Indexed(_) => recalled.is_none_or(|v| v[t * tokens + s]),
                    }
            };
            let qrow = &q[t * heads * qk + h * qk..t * heads * qk + (h + 1) * qk];
            let mut logits: Vec<(usize, f64)> = (0..tokens)
                .filter(|s| visible(*s))
                .map(|s| {
                    let kvrow = &kv[s * heads * (nope + vdim) + h * (nope + vdim)..];
                    let dot_nope: f64 = (0..nope).map(|d| qrow[d] * kvrow[d]).sum();
                    let dot_rope: f64 = (0..rope)
                        .map(|d| qrow[nope + d] * k_rope[s * rope + d])
                        .sum();
                    (s, (dot_nope + dot_rope) * scale)
                })
                .collect();
            let sink = w.sinks[h];
            let max = logits.iter().map(|(_, l)| *l).fold(sink, f64::max);
            let denom: f64 =
                logits.iter().map(|(_, l)| (l - max).exp()).sum::<f64>() + (sink - max).exp();
            for (s, l) in logits.iter_mut() {
                let p = (*l - max).exp() / denom;
                let vrow = &kv[*s * heads * (nope + vdim) + h * (nope + vdim) + nope..];
                for d in 0..vdim {
                    attended[t * heads * vdim + h * vdim + d] += p * vrow[d];
                }
            }
        }
    }
    for (a, g) in attended.iter_mut().zip(&gate) {
        *a *= sigmoid(*g);
    }
    let out = linear(&attended, w.o, heads * vdim, hidden);
    rms_norm(&out, w.post_norm, plan.post_projection_norm)
}

/// One independent hyper-connection's parameters (`hc_pre.hc_fn` / `hc_base` / `hc_scale`).
pub struct HyperConnectionWeights<'a> {
    pub function: &'a [f64],
    pub base: &'a [f64],
    pub scale: &'a [f64],
}

/// `HYV4FlashHyperConnection.forward`: returns `(post [tokens, streams], collapsed [tokens,
/// hidden])`. From the flat streams, `function` gives `streams` pre and `streams` post logits,
/// scaled by the inverse RMS of the flat row; `pre = sigmoid(l * scale0 + base) + eps` collapses
/// the streams, `post = magnitude * sigmoid(l * scale1 + base) + eps` is the write-back weight.
pub fn hyper_connection(
    plan: &HyV4ResidualPlan,
    w: &HyperConnectionWeights,
    streams: &[f64],
    hidden: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = plan.streams as usize;
    let flat = n * hidden;
    let tokens = streams.len() / flat;
    let mixes = linear(streams, w.function, flat, 2 * n);
    let mut post = Vec::with_capacity(tokens * n);
    let mut collapsed = vec![0.0; tokens * hidden];
    let eps = plan.epsilon as f64;
    for t in 0..tokens {
        let row = &streams[t * flat..(t + 1) * flat];
        let inv = 1.0
            / (row.iter().map(|v| v * v).sum::<f64>() / flat as f64 + plan.mix_norm_epsilon as f64)
                .sqrt();
        let m = &mixes[t * 2 * n..(t + 1) * 2 * n];
        for s in 0..n {
            let pre = sigmoid(m[s] * inv * w.scale[0] + w.base[s]) + eps;
            for d in 0..hidden {
                collapsed[t * hidden + d] += pre * row[s * hidden + d];
            }
        }
        for s in 0..n {
            post.push(
                plan.post_magnitude as f64 * sigmoid(m[n + s] * inv * w.scale[1] + w.base[n + s])
                    + eps,
            );
        }
    }
    (post, collapsed)
}

/// The decoder layer's write-back: `stream_s += post_s * output` for every stream.
pub fn write_back(
    residual: &[f64],
    post: &[f64],
    output: &[f64],
    streams: usize,
    hidden: usize,
) -> Vec<f64> {
    let tokens = output.len() / hidden;
    let mut out = residual.to_vec();
    for t in 0..tokens {
        for s in 0..streams {
            for d in 0..hidden {
                out[(t * streams + s) * hidden + d] +=
                    post[t * streams + s] * output[t * hidden + d];
            }
        }
    }
    out
}

/// `HYV4FlashMLP`: `down(silu(gate(x)) * up(x))`.
pub fn dense_mlp(
    x: &[f64],
    gate: &[f64],
    up: &[f64],
    down: &[f64],
    hidden: usize,
    inter: usize,
) -> Vec<f64> {
    let g = linear(x, gate, hidden, inter);
    let u = linear(x, up, hidden, inter);
    let act: Vec<f64> = g.iter().zip(&u).map(|(g, u)| silu(*g) * u).collect();
    linear(&act, down, inter, hidden)
}

/// Latent MoE weights. The routed banks are `[experts, out, in]`, gate and up separate as the
/// checkpoint stores them.
pub struct LatentMoeWeights<'a> {
    pub router: &'a [f64],
    pub router_bias: &'a [f64],
    pub latent_down: &'a [f64],
    pub latent_up: &'a [f64],
    pub expert_gate: &'a [f64],
    pub expert_up: &'a [f64],
    pub expert_down: &'a [f64],
    pub shared_gate: &'a [f64],
    pub shared_up: &'a [f64],
    pub shared_down: &'a [f64],
    pub output_norm: &'a [f64],
}

/// `HYV4FlashMoE.forward`: sigmoid router (the bias steers the top-k selection only; the
/// selected raw scores are renormalized with `+1e-20` and scaled; an exact tie in the biased
/// scores picks the lower expert index, where torch's `topk` order is unspecified), routed experts on
/// `latent_down(x)` with the pre-clamped SwiGLU, `latent_up` back, plus the shared expert on
/// `x`, then the output norm.
pub fn latent_moe(
    plan: &LatentMoePlan,
    norm: NormPlan,
    w: &LatentMoeWeights,
    x: &[f64],
    hidden: usize,
) -> Vec<f64> {
    let experts = plan.moe.expert_count as usize;
    let top_k = plan.moe.experts_per_token as usize;
    let latent = plan.latent_size as usize;
    let inter = plan.moe.expert_intermediate_size as usize;
    let RouterPlan::Sigmoid {
        normalize_selected,
        scaling_factor,
        selection_bias,
    } = plan.moe.router
    else {
        panic!("hy_v4_flash routes with sigmoid scores");
    };
    let ActivationPlan::SwiGluPreClamped { limit } = plan.moe.activation else {
        panic!("hy_v4_flash routed experts run the pre-clamped SwiGLU");
    };
    assert_eq!(
        plan.shared_activation,
        ActivationPlan::Silu,
        "plain shared SwiGLU"
    );
    let limit = limit as f64;
    let tokens = x.len() / hidden;
    let logits = linear(x, w.router, hidden, experts);
    let routed_in = linear(x, w.latent_down, hidden, latent);
    let mut routed = vec![0.0; tokens * latent];
    for t in 0..tokens {
        let scores: Vec<f64> = logits[t * experts..(t + 1) * experts]
            .iter()
            .map(|l| sigmoid(*l))
            .collect();
        let mut order: Vec<usize> = (0..experts).collect();
        let choice = |e: usize| {
            scores[e]
                + if selection_bias {
                    w.router_bias[e]
                } else {
                    0.0
                }
        };
        order.sort_by(|a, b| choice(*b).total_cmp(&choice(*a)));
        let chosen = &order[..top_k];
        let mut weights: Vec<f64> = chosen.iter().map(|e| scores[*e]).collect();
        if normalize_selected {
            let denom = weights.iter().sum::<f64>() + 1e-20;
            weights.iter_mut().for_each(|v| *v /= denom);
        }
        weights.iter_mut().for_each(|v| *v *= scaling_factor as f64);
        let xin = &routed_in[t * latent..(t + 1) * latent];
        for (e, weight) in chosen.iter().zip(&weights) {
            fn bank(b: &[f64], e: usize, out: usize, input: usize) -> &[f64] {
                &b[e * out * input..(e + 1) * out * input]
            }
            let g = linear(xin, bank(w.expert_gate, *e, inter, latent), latent, inter);
            let u = linear(xin, bank(w.expert_up, *e, inter, latent), latent, inter);
            let act: Vec<f64> = g
                .iter()
                .zip(&u)
                .map(|(g, u)| silu(g.min(limit)) * u.clamp(-limit, limit))
                .collect();
            let down = linear(&act, bank(w.expert_down, *e, latent, inter), inter, latent);
            for d in 0..latent {
                routed[t * latent + d] += down[d] * weight;
            }
        }
    }
    let routed_out = linear(&routed, w.latent_up, latent, hidden);
    let shared_plan = plan
        .moe
        .shared
        .as_ref()
        .expect("hy_v4_flash has a shared expert");
    assert!(
        !shared_plan.gated,
        "the Hy4-flash shared expert has no sigmoid gate"
    );
    let shared = shared_plan.intermediate_size as usize;
    let shared_out = dense_mlp(x, w.shared_gate, w.shared_up, w.shared_down, hidden, shared);
    let sum: Vec<f64> = routed_out
        .iter()
        .zip(&shared_out)
        .map(|(a, b)| a + b)
        .collect();
    rms_norm(&sum, w.output_norm, norm)
}

/// `HYV4FlashHyperHead.forward`: the exit collapse of the streams into one row per token,
/// `sum_s (sigmoid(mix_s * inv_rms * scale + base_s) + eps) * stream_s` with the plan's
/// `exit_epsilon`; its mix norm takes the model's RMS epsilon, which the plan carries as
/// `output_norm` (`HYV4FlashHyperHead.input_norm` and `model.norm` share `rms_norm_eps`).
/// `w.scale` has one entry.
pub fn hyper_head(
    numerics: &HyV4FlashNumerics,
    output_norm: NormPlan,
    w: &HyperConnectionWeights,
    streams: &[f64],
    hidden: usize,
) -> Vec<f64> {
    let n = numerics.streams as usize;
    let flat = n * hidden;
    let tokens = streams.len() / flat;
    let mixes = linear(streams, w.function, flat, n);
    let mut out = vec![0.0; tokens * hidden];
    for t in 0..tokens {
        let row = &streams[t * flat..(t + 1) * flat];
        let inv = 1.0
            / (row.iter().map(|v| v * v).sum::<f64>() / flat as f64 + output_norm.epsilon as f64)
                .sqrt();
        for s in 0..n {
            let pre = sigmoid(mixes[t * n + s] * inv * w.scale[0] + w.base[s])
                + numerics.exit_epsilon as f64;
            for d in 0..hidden {
                out[t * hidden + d] += pre * row[s * hidden + d];
            }
        }
    }
    out
}

/// The MLP of one decoder layer.
pub enum MlpWeights<'a> {
    /// Layer 0: the plain SwiGLU MLP and its output norm (`post_fc2_norm`).
    Dense {
        gate: &'a [f64],
        up: &'a [f64],
        down: &'a [f64],
        output_norm: &'a [f64],
        intermediate: usize,
    },
    Latent(LatentMoeWeights<'a>),
}

/// Everything one decoder layer reads (no Engram on this layer).
pub struct LayerWeights<'a> {
    pub input_norm: &'a [f64],
    pub mlp_input_norm: &'a [f64],
    pub attention: AttentionWeights<'a>,
    pub attention_hc: HyperConnectionWeights<'a>,
    pub mlp_hc: HyperConnectionWeights<'a>,
    pub mlp: MlpWeights<'a>,
}

/// `HYV4FlashDecoderLayer.forward` on `[tokens, streams, hidden]` streams, from the typed layer
/// plan: attention sublayer then MLP sublayer, each read through its hyper-connection and
/// written back onto every stream beside the identity residual.
pub fn decoder_layer(
    layer: &memra_gguf::model_plan::LayerPlan,
    w: &LayerWeights,
    streams: &[f64],
    positions: &[u32],
    hidden: usize,
) -> Vec<f64> {
    let memra_gguf::model_plan::AttentionPlan::HyV4(attention) = &layer.attention else {
        panic!("layer {} is not a Hy4-flash layer", layer.index);
    };
    let memra_gguf::model_plan::ResidualTopology::HyV4(residual) = &layer.residual else {
        panic!("layer {} has no Hy4-flash residual", layer.index);
    };
    assert!(residual.engram.is_none(), "Engram is a later step");
    let n = residual.streams as usize;
    let (post, read) = hyper_connection(residual, &w.attention_hc, streams, hidden);
    let x = rms_norm(&read, w.input_norm, layer.pre_attention_norm);
    let attn = gated_mla_attention(
        attention,
        layer.pre_attention_norm,
        &w.attention,
        &x,
        positions,
        hidden,
    );
    let streams = write_back(streams, &post, &attn, n, hidden);
    let (post, read) = hyper_connection(residual, &w.mlp_hc, &streams, hidden);
    let x = rms_norm(&read, w.mlp_input_norm, layer.pre_mlp_norm);
    let mlp = match (&layer.mlp, &w.mlp) {
        (
            memra_gguf::model_plan::MlpPlan::Dense(dense),
            MlpWeights::Dense {
                gate,
                up,
                down,
                output_norm,
                intermediate,
            },
        ) => {
            assert_eq!(
                dense.activation,
                ActivationPlan::Silu,
                "plain SwiGLU dense MLP"
            );
            assert_eq!(
                dense.intermediate_size as usize, *intermediate,
                "dense MLP width"
            );
            rms_norm(
                &dense_mlp(&x, gate, up, down, hidden, *intermediate),
                output_norm,
                residual.mlp_output_norm,
            )
        }
        (memra_gguf::model_plan::MlpPlan::LatentMoe(plan), MlpWeights::Latent(weights)) => {
            latent_moe(plan, residual.mlp_output_norm, weights, &x, hidden)
        }
        _ => panic!("layer {}: MLP plan and weights disagree", layer.index),
    };
    write_back(&streams, &post, &mlp, n, hidden)
}
