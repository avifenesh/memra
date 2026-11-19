//! Hy4-flash program types, compiled only by the `hy_v4_flash` pack.
//!
//! The semantic source is transformers `modeling_hy_v4_flash.py` (PR #50903 at b865850d0532),
//! read with vLLM PR #68296 at 8cbb80efa397 (research/modelplan-onboarding-hy4flash-20261119/).
//! None of these types parameterizes an existing plan: the attention is MLA with a learned sink,
//! an elementwise sigmoid output gate and a norm after `o_proj` on every layer; the indexer pools
//! cached token keys four at a time through an F32 softmax gate; the residual is a 4-stream
//! hyper-connection without Sinkhorn whose write-back magnitude is a config value; the routed
//! experts run in a latent width; the Engram memory hashes a normalized vocabulary through a
//! checkpoint lookup table. No reference executor or tuned surface implements them yet; every
//! operation they emit is a registry row with no surface.

use super::{MoeMlpPlan, NormPlan, RopePlan, TensorPresence};

/// Storage of a weight the program multiplies by, as the checkpoint ships it. Weight-only:
/// activations stay BF16 (the artifact has no activation or KV-cache scales).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyV4WeightFormat {
    Bf16,
    /// ModelOpt `FP8_PB_WO`: e4m3 with one F32 multiplier per `block` x `block` tile
    /// (`weight_scale_inv`).
    Fp8Block {
        block: u32,
    },
    /// ModelOpt `W4A16_NVFP4`: e2m1 codes, an e4m3 scale per `group` inputs and an F32
    /// second-level scale per tensor (per expert when stacked). Gate and up are separate
    /// tensors with separate second-level scales.
    Nvfp4W4A16 {
        group: u32,
    },
}

/// Which keys a layer attends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyV4AttentionKind {
    /// Full causal attention, narrowed by the pooled indexer once the context exceeds its
    /// budget.
    Indexed(IndexPoolPlan),
    /// The last `window` keys, the query included.
    Sliding { window: u32 },
}

/// transformers `HYV4FlashCompressedIndexer`. Token keys `token_key_norm(token_key_projection(x))`
/// are cached per token; complete groups of `pool` tokens are pooled in F32 as
/// `key_norm(sum_slot(value_projection(k) * softmax_slot(gate_projection(k) + ape)))`; queries
/// (`query_projection(q_resid)`, `heads` x `head_dim`) and pooled keys rotate their trailing
/// `rope_head_dim` dims (queries at their position, groups at their first token's); the score is
/// `sum_h head_weight_h * relu(q_h . k)` with head weights `head_weight_projection(x)` scaled by
/// `heads^-0.5 * head_dim^-0.5`. A query keeps the `top_k / pool - 1` best groups that closed
/// before it plus its own group, expanded back to tokens. With at most `top_k` keys the layer
/// attends densely and no pooling happens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexPoolPlan {
    pub heads: u32,
    pub head_dim: u32,
    pub rope_head_dim: u32,
    pub pool: u32,
    pub top_k: u32,
}

/// transformers `HYV4FlashAttention`: gated MLA. `q = q_b(q_a_norm(q_a(x)))`, `kv_a(x)` splits
/// into a `kv_lora_rank` latent (normalized, cached) and one shared `qk_rope_head_dim` rope key
/// (cached); `kv_b` expands the latent to per-head nope keys and values. RoPE rotates the
/// trailing rope dims as adjacent pairs. Softmax scale `(nope + rope)^-0.5` with a learned
/// per-head sink in the denominator. The output is gated elementwise by
/// `sigmoid(linear_gate(x))` (`heads` x `value_head_dim`), projected by `o_proj` and normalized
/// by `post_projection_norm`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyV4AttentionPlan {
    pub kind: HyV4AttentionKind,
    pub query_heads: u32,
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub value_head_dim: u32,
    /// Text positions on all three M-RoPE axes reduce interleaved M-RoPE to this plain RoPE.
    pub rope: RopePlan,
    pub sink: TensorPresence,
    pub post_projection_norm: NormPlan,
    /// `q_b_proj`, `o_proj` and `linear_gate`.
    pub projection_weights: HyV4WeightFormat,
}

/// transformers `HYV4FlashEngramLayer`, added to the hidden streams before the layer's
/// attention hyper-connection. `layer_id` is the config's 1-based id; the module runs in 0-based
/// decoder layer `layer_id - 1`. Token ids map through the checkpoint `lookup_table` (ids past it
/// read as the pad id's entry), each n-gram order mixes the mapped ids with odd multipliers drawn
/// from `numpy.random.default_rng(hash_seed)` and XOR, and `heads` hash heads take the mixed
/// value modulo their own prime to gather one `head_dim` row each from the table. Rows are read
/// through a per-row FP8 quantization, normalized per head, keyed against every stream and
/// written back through a signed-square-root sigmoid gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyV4EngramPlan {
    pub layer_id: u32,
    pub hash_seed: u64,
    pub max_ngram: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub table_rows: u64,
    pub lookup_table_size: u32,
    pub pad_token_id: u32,
    /// FP8 e4m3 rows with one F32 scale per row (`weight_scale`).
    pub table_fp8_per_row: bool,
}

/// transformers `HYV4FlashHyperConnection` around each sublayer: from the RMS-normalized flat
/// streams, `fn` gives `streams` pre and `streams` post logits;
/// `pre = sigmoid(l * scale0 + base) + epsilon` collapses the streams into the sublayer input,
/// `post = magnitude * sigmoid(l * scale1 + base) + epsilon` writes the sublayer output onto each
/// stream beside the identity residual, all in F32. The MLP output passes `mlp_output_norm` (dense `post_fc2_norm`, MoE
/// `combine_output_norm`) before the write-back.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyV4ResidualPlan {
    pub streams: u32,
    pub epsilon: f32,
    pub post_magnitude: f32,
    pub mix_norm_epsilon: f32,
    pub mlp_output_norm: NormPlan,
    pub engram: Option<HyV4EngramPlan>,
}

/// transformers `HYV4FlashMoE`: the router and the shared expert read the full-width input; the
/// routed experts read `routed_expert_down_proj(x)` (`latent_size` wide) and their weighted sum
/// returns through `routed_expert_up_proj`.
#[derive(Debug, Clone, PartialEq)]
pub struct LatentMoePlan {
    pub latent_size: u32,
    pub moe: MoeMlpPlan,
    pub expert_weights: HyV4WeightFormat,
}

/// Per-layer cache: the normalized `latent_width` latent plus the `rope_width` rope key per
/// token, over the window on sliding layers; indexed layers also cache `index_key_width` token
/// keys for the indexer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyV4StatePlan {
    pub latent_width: u32,
    pub rope_width: u32,
    pub window: Option<u32>,
    pub index_key_width: Option<u32>,
}

/// Family-level numerics, carried on `ModelPlan::hy_v4_flash`: the embedding is copied onto
/// `streams` streams, the exit collapses them with `hc_head` (`sigmoid(mix * scale + base) +
/// epsilon` per stream) before the final norm, and the head is computed in F32.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyV4FlashNumerics {
    pub streams: u32,
    pub exit_epsilon: f32,
    pub output_projection_f32: bool,
}
