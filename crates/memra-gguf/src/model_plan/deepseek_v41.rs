//! DeepSeek-V4.1 program types, compiled only by the `deepseek_v41` pack.
//!
//! The semantic source is the vendor reference `inference/model.py` of
//! `deepseek-ai/DeepSeek-V4.1-Flash-1017` and its tech report (§2.3 CSA2, §2.4 single-pass mHC
//! and Engram). None of these types is a parameterization of a DeepSeek-V4 plan: CSA2 shares one
//! compressed KV across a layer group, compresses without overlap or positional embedding, and
//! narrows its decoder indexers through a candidate pool, which the DSv4 `CompressedKv` plan does
//! not compute. No reference executor or tuned surface implements them yet; every operation they
//! emit is a registry row with no surface.

use super::{RopePlan, TensorPresence};

/// Element format of a quantize-dequantize step in the source program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockElement {
    Fp8E4m3,
    Fp4E2m1,
}

/// Encoding of a block's scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockScale {
    /// Power-of-two scale (`float8_e8m0fnu`, ue8m0 rounding).
    Ue8m0,
    /// `float8_e4m3fn` scale.
    E4m3,
}

/// A quantize-dequantize the source applies to an activation, a cache row or a table row:
/// absmax scale per `block` contiguous elements along the last axis, rounded to `scale`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockQuant {
    pub element: BlockElement,
    pub block: u32,
    pub scale: BlockScale,
}

impl BlockQuant {
    /// `act_quant(x, 32, "ue8m0")`: every FP8 and FP4 linear input, the window KV and the
    /// attention output before `wo_a`.
    pub const FP8_32_UE8M0: Self = Self {
        element: BlockElement::Fp8E4m3,
        block: 32,
        scale: BlockScale::Ue8m0,
    };
    /// `fp4_act_quant(x, 32)`: index query and index key.
    pub const FP4_32_UE8M0: Self = Self {
        element: BlockElement::Fp4E2m1,
        block: 32,
        scale: BlockScale::Ue8m0,
    };
    /// `fp4_act_quant(x, 16, scale_dtype=e4m3)`: the compressed main KV.
    pub const FP4_16_E4M3: Self = Self {
        element: BlockElement::Fp4E2m1,
        block: 16,
        scale: BlockScale::E4m3,
    };
}

/// Family-level numerics of the checkpoint program, carried on `ModelPlan::deepseek_v41`.
/// Emits `OperationKind::Fp8Block32Linear`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeekV41Numerics {
    /// Dense FP8 weights: e4m3 with one ue8m0 scale per `dense_weight_block` x
    /// `dense_weight_block` block (`quantization_config.weight_block_size` [32, 32]).
    pub dense_weight_block: u32,
    /// Routed experts: e2m1 pairs packed in I8 with one ue8m0 scale per 32 inputs (MXFP4 layout).
    pub expert_weight: BlockQuant,
    /// Dynamic quantize-dequantize on the input of every FP8 and FP4 linear.
    pub linear_activation: BlockQuant,
    /// The routing weight multiplies the SwiGLU output before `w2`, so before `w2`'s activation
    /// quantization (vendor `Expert.forward`).
    pub routed_weight_before_down: bool,
    /// `wo_a` ships FP8 32x32 and the reference program dequantizes it to BF16 at load.
    pub output_lora_a_dequantized_bf16: bool,
}

/// One CSA2 attention layer (vendor `Attention`): low-rank query, one latent KV head of
/// `head_dim` with K = V, RoPE on the trailing `rope_head_dim` dims as adjacent pairs, a
/// layer-local sliding window, optionally `top_k` selected compressed entries, a learned per-head
/// sink in the softmax denominator, inverse RoPE on the output tail, then the block-diagonal
/// grouped `wo_a` and `wo_b`. Softmax scale `head_dim^-0.5`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Csa2AttentionPlan {
    pub query_heads: u32,
    pub q_lora_rank: u32,
    pub head_dim: u32,
    pub rope_head_dim: u32,
    pub output_lora_rank: u32,
    pub output_groups: u32,
    pub window: u32,
    /// Compressed layers rotate at `compress_rope_theta` with YaRN; window-only layers at
    /// `rope_theta` with none. Applies to query, window KV, compressed latent and index rows.
    pub rope: RopePlan,
    pub sink: TensorPresence,
    pub mode: Csa2Mode,
    /// Window KV rows after `kv_norm` and RoPE.
    pub window_kv_quant: BlockQuant,
    /// Attention output per group before `wo_a`.
    pub output_quant: BlockQuant,
}

/// CSA2 layer mode (tech report §2.3.1). Every mode computes its own query and window KV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Csa2Mode {
    /// Compress ratio 0: the sliding window only.
    Window,
    /// Computes the group's compressed KV and index keys and runs its own indexer.
    Full {
        ratio: u32,
        compressor: Csa2CompressorPlan,
        indexer: Csa2IndexerPlan,
    },
    /// Reads `kv_source`'s compressed KV and index keys; rescores with its own index query.
    Reindex {
        ratio: u32,
        kv_source: u32,
        indexer: Csa2IndexerPlan,
    },
    /// Reads `kv_source`'s compressed KV and `index_source`'s top-k.
    Reuse {
        ratio: u32,
        kv_source: u32,
        index_source: u32,
    },
}

impl Csa2Mode {
    pub fn ratio(&self) -> u32 {
        match *self {
            Self::Window => 0,
            Self::Full { ratio, .. } | Self::Reindex { ratio, .. } | Self::Reuse { ratio, .. } => {
                ratio
            }
        }
    }
}

/// CSA2 compressor (vendor `Compressor`): `ratio` consecutive tokens become one latent of
/// `head_dim`, then RMSNorm. No overlap between groups and no positional embedding (the DSv4 CSA
/// compressor has both). The latent stands for its group's first token position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Csa2CompressorPlan {
    pub ratio: u32,
    pub pooling: Csa2Pooling,
    /// The compressed main KV row after RoPE, as stored.
    pub latent_quant: BlockQuant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Csa2Pooling {
    /// Ratio 1: `norm(wkv(x))` in the checkpoint's BF16, no gate.
    Projection,
    /// Ratio > 1: `sum(wkv(x) * softmax(wgate(x)))` over the group in FP32, the partial group
    /// carried across decode steps.
    SoftmaxGated,
}

/// CSA2 indexer (vendor `Indexer`): `heads` FP4 query heads from `wq_b(qr)` against one index key
/// per compressed entry, `relu` scores weighted by `weights_proj(x) * head_dim^-0.5 * heads^-0.5`,
/// causal over complete groups, then `top_k`. A Full-mode layer also projects the index keys from
/// its RoPE-free compressed latent (`k_norm(wk(latent))`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Csa2IndexerPlan {
    pub heads: u32,
    pub head_dim: u32,
    pub top_k: u32,
    pub query_quant: BlockQuant,
    pub key_quant: BlockQuant,
    pub candidates: Csa2Candidates,
}

/// Hierarchical sparse indexer (tech report §2.3.2, vendor `select_candidate_blocks`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Csa2Candidates {
    /// Scores every reachable compressed entry.
    None,
    /// Level one: scores every entry, keeps the `top_blocks` blocks of `block_size` entries by
    /// their best score, the block holding the newest entry always kept, and publishes the mask.
    Produce { top_blocks: u32, block_size: u32 },
    /// Level two: scores only inside `source_layer`'s published candidate blocks.
    Restrict { source_layer: u32 },
}

/// Per-sequence state of a CSA2 layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Csa2StatePlan {
    /// Ring of the last `window` window-KV rows, `head_dim` wide.
    pub window: u32,
    pub head_dim: u32,
    pub compressed: Csa2CompressedState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Csa2CompressedState {
    None,
    /// A Full-mode layer owns the group's compressed KV (`context / ratio` rows of `head_dim`),
    /// its index keys (`index_head_dim` per row) and, above ratio 1, the partial-group carry.
    Own {
        ratio: u32,
        index_head_dim: u32,
        partial_group: bool,
    },
    /// Reads the compressed KV and index keys `source_layer` owns.
    Shared {
        ratio: u32,
        source_layer: u32,
    },
}

/// Single-pass mHC (tech report §2.4.1, vendor `Block`): the residual stream is `streams` copies
/// of hidden. Each sublayer derives pre / post / comb mixes from the stream (RMS over the
/// flattened stream with `mix_norm_epsilon`, Sinkhorn on comb), collapses its input with the pre
/// mix the PREVIOUS sublayer produced (the identity one-hot at the first block), and hands its own
/// pre mix to the next sublayer. The model and the drafter exit collapse with the last FFN's pre
/// mix; there is no learned head collapse and no `hc_head_*` tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SinglePassHcPlan {
    pub streams: u32,
    pub epsilon: f32,
    pub sinkhorn_iterations: u32,
    pub mix_norm_epsilon: f32,
    /// Engram memory written into this block's stream before its attention mix.
    pub engram: Option<EngramPlan>,
}

/// Engram (tech report §2.4.2, vendor `Engram` + `engram.py`). Each position hashes the
/// `max_ngram - 1` n-grams ending at it over a tokenizer-derived compressed vocabulary (pad id
/// fills missing history, image spans block it), `heads` hashes per n-gram into prime-sized
/// bucket ranges of this layer's table. The `hash_columns` gathered rows are dequantized and
/// projected by `wkv` into one key per stream plus a shared value; a signed-sqrt sigmoid gate of
/// the normalized stream-key dot product (`q_weight * k_weight`, scale `hidden^-0.5`, clamp
/// `gate_clamp`) adds the value into each stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EngramPlan {
    /// Position of this layer in `engram_layer_ids` (its hash multiplier row and prime set).
    pub table_index: u32,
    pub table_rows: u64,
    pub head_dim: u32,
    pub heads: u32,
    pub max_ngram: u32,
    pub hash_columns: u32,
    /// Bucket primes are drawn above this value, in order, never reused across layers.
    pub bucket_base: u32,
    /// Size of the normalized token map the hash multipliers derive from.
    pub compressed_vocab_size: u32,
    pub pad_token_id: u32,
    /// Table rows: e4m3 with a ue8m0 scale per 32 elements, dequantized on lookup.
    pub table_quant: BlockQuant,
    pub gate_clamp: f32,
}
