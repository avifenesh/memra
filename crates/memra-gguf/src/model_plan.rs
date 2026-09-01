//! Source-agnostic semantic model plan.
//!
//! `ModelConfig` is the normalized output of both the GGUF and HF readers. This module compiles
//! that metadata into typed operations without changing the existing loaders or execution paths.
//! Runtime migration can therefore compare a plan against today's behavior before selecting it.

use crate::config::{Arch, AttentionGateKind, LayerKind, ModelConfig};

#[derive(Debug, Clone, PartialEq)]
pub struct ModelPlan {
    pub arch: Arch,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub context_length: u32,
    pub embedding_scale: f32,
    pub vision: Option<VisionPlan>,
    pub multimodal: Option<VisionTokenInjectionPlan>,
    pub layers: Vec<LayerPlan>,
    pub output_norm: NormPlan,
    pub logits: Vec<LogitsTransform>,
    pub mtp_blocks: Vec<MtpBlockPlan>,
    pub drafter: Option<DrafterPlan>,
    pub draft_source: DraftSourcePlan,
    pub sampling_defaults: Option<SamplingDefaultsPlan>,
    /// Valid cuts are between complete trunk blocks. Whether a backend can transport every state
    /// crossing one of these cuts is derived from the operations, not from the architecture name.
    pub partition_boundaries: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingDefaultsPlan {
    pub temperature: f32,
    pub top_p: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftSourcePlan {
    None,
    Embedded,
    ExternalArtifact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionTokenInjectionPlan {
    pub placeholder_token_id: u32,
    /// `Some(n)`: the config declares a fixed per-image placeholder count (gemma-4).
    /// `None`: grid-derived — each image contributes `(gh/merge) * (gw/merge)` placeholders
    /// and the chat template emits ONE placeholder that the serving layer expands
    /// (glm5_next; the upstream `Glm5NextProcessor.replace_image_token` convention).
    pub tokens_per_image: Option<u32>,
    /// Image-span delimiters when the vocabulary carries them (glm5_next:
    /// `<|begin_of_image|>` / `<|end_of_image|>`). The delimiters keep their ordinary token
    /// embeddings; only the placeholder rows are replaced by tower output.
    pub start_token_id: Option<u32>,
    pub end_token_id: Option<u32>,
}

/// One vision tower program. Each variant is a distinct semantic program with its own
/// tensor census and executor arm — variants are never approximated by one another
/// (no-generic-support law).
#[derive(Debug, Clone, PartialEq)]
pub enum VisionPlan {
    /// Factored additive x/y position tables, sandwich RMS norms, unscaled attention,
    /// avg-pool head + single output projection (gemma-4 family).
    Factored(VisionEncoderPlan),
    /// Fused-qkv ViT with per-head q/k RMS norms, 2D rope (rope-only positions), biased
    /// clamped-SwiGLU block MLPs, post-encoder RMS norm, conv `merge x merge` downsample,
    /// gated clamped merger (glm5_next family; upstream transformers
    /// `Glm5NextVisionModel`, vision classes inherited from `GlmOcrVisionModel`).
    Glm5Fused(Glm5VisionPlan),
}

/// glm5_next vision tower plan. Geometry from `Glm5VisionConfig` (config.json truth);
/// `rope_theta` is 10000.0 by upstream hardcode (`Glm5NextVisionRotaryEmbedding.__init__`
/// default — the checkpoint config carries NO vision rope key).
#[derive(Debug, Clone, PartialEq)]
pub struct Glm5VisionPlan {
    pub depth: u32,
    pub hidden_size: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub patch_size: u32,
    pub temporal_patch_size: u32,
    pub spatial_merge_size: u32,
    /// Merger output width == trunk `n_embd` (validated at compile).
    pub out_hidden_size: u32,
    pub projection_intermediate_size: u32,
    pub swiglu_limit: f32,
    pub rope_theta: f32,
    pub norm: NormPlan,
    pub in_channels: u32,
    /// Patch-row width the tower consumes: `in_channels * temporal_patch_size * patch^2`.
    pub patch_input_width: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VisionEncoderPlan {
    pub hidden_size: u32,
    pub context_length: u32,
    pub patch: VisionPatchPlan,
    pub layers: Vec<VisionLayerPlan>,
    pub projection_output_size: u32,
    pub pooling_kernel_size: u32,
    pub standardize: bool,
    pub clipped_linears: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionPatchPlan {
    pub channels: u32,
    pub patch_size: u32,
    pub position_axes: u32,
    pub position_embedding_size: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VisionLayerPlan {
    pub index: u32,
    pub input_norm: NormPlan,
    pub attention: VisionAttentionPlan,
    pub post_attention_norm: NormPlan,
    pub pre_mlp_norm: NormPlan,
    pub mlp: DenseMlpPlan,
    pub post_mlp_norm: NormPlan,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisionAttentionPlan {
    pub query_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub rope: RopePlan,
    pub bidirectional: bool,
    pub qk_norm: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerPlan {
    pub index: u32,
    pub pre_attention_norm: NormPlan,
    pub attention: AttentionPlan,
    pub pre_mlp_norm: NormPlan,
    pub mlp: MlpPlan,
    pub residual: ResidualTopology,
    pub state: StatePlan,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormPlan {
    pub kind: NormKind,
    pub epsilon: f32,
    pub weight_transform: WeightTransform,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    Rms,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightTransform {
    Identity,
    AddOne,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AttentionPlan {
    Full(FullAttentionPlan),
    SlidingWindow {
        attention: FullAttentionPlan,
        window: u32,
    },
    Mla(MlaAttentionPlan),
    GatedDeltaNet(GatedDeltaNetPlan),
    KimiDeltaNet(KimiDeltaNetPlan),
}

#[derive(Debug, Clone, PartialEq)]
pub struct FullAttentionPlan {
    pub query_heads: u32,
    pub kv_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub rope: RopePlan,
    pub qk_norm: TensorPresence,
    pub output_gate: AttentionGateKind,
    pub scale: AttentionScale,
    pub value_projection: ValueProjection,
    pub value_norm: ValueNorm,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttentionScale {
    InverseSqrtKeyDim,
    Fixed(f32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueProjection {
    Separate,
    ReuseKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueNorm {
    None,
    WeightlessRms,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopePlan {
    pub dimensions: u32,
    pub base: f32,
    pub factors: RopeFactors,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeFactors {
    None,
    Checkpoint,
    PartialRotary {
        factor: f32,
    },
    Yarn {
        factor: f32,
        original_context: u32,
        beta_fast: f32,
        beta_slow: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorPresence {
    Absent,
    Optional,
    Required,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MlaAttentionPlan {
    LatentKv {
        query_heads: u32,
        q_lora_rank: u32,
        kv_lora_rank: u32,
        qk_head_dim: u32,
        rope_head_dim: u32,
        value_head_dim: u32,
        rope: RopePlan,
        sparse_index: SparseIndexPlan,
    },
    CompressedKv {
        query_heads: u32,
        q_lora_rank: u32,
        latent_head_dim: u32,
        rope_head_dim: u32,
        output_lora_rank: u32,
        output_groups: u32,
        window: u32,
        rope: RopePlan,
        compressor: Option<KvCompressorPlan>,
        sparse_index: SparseIndexPlan,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCompressorPlan {
    pub ratio: u32,
    pub latent_dim: u32,
}

/// K-pool compression on a sparse indexer (glm5_next). Keys are grouped into pools of
/// `pool` tokens; each pool is collapsed by a learned softmax (per-token gate scores +
/// a per-slot additive positional embedding), scored as one candidate, and a selected
/// pool expands back into its raw token indices. `always_select_tail` appends the
/// current incomplete pool as raw indices on top of the `top_k` budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KpoolPlan {
    pub pool: u32,
    pub always_select_tail: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseIndexPlan {
    None,
    Own {
        heads: u32,
        head_dim: u32,
        top_k: u32,
        /// `None` = per-token scoring (GLM-5.2 / dsv4 lightning indexer); `Some` =
        /// pool-compressed scoring (glm5_next). Distinct arithmetic, same budget field.
        kpool: Option<KpoolPlan>,
    },
    SharedFromPrevious {
        top_k: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatedDeltaNetPlan {
    pub key_heads: u32,
    pub value_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub conv_kernel: u32,
}

/// Kimi Delta Attention (glm5_next linear-attn layers). Deliberately NOT
/// `GatedDeltaNetPlan`: heads are symmetric (q=k=v), decay is PER-CHANNEL through a
/// low-rank forget gate (`f_a`/`f_b` + `dt_bias`, clamped `lower_bound * sigmoid(exp(A_log)*g)`)
/// with a per-head `beta = sigmoid(b_proj)` write gate, q/k are l2-normalized in f32
/// (FLA semantics: `x / sqrt(sum(x^2) + eps)`), and the output passes a SIGMOID-gated
/// RMSNorm fed by a second low-rank gate (`g_a`/`g_b`) before `o_proj`. The checkpoint
/// stores q/k/v short-convs as three tensors; the executor fuses them into one grouped
/// conv over `3 * num_heads * head_dim` channels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KimiDeltaNetPlan {
    pub num_heads: u32,
    pub head_dim: u32,
    pub conv_kernel: u32,
    /// `linear_attn_config.gate_lower_bound` — the forget-gate log-decay floor (−5.0 on
    /// GLM-5.3-Flash). Part of the decay arithmetic, not a tuning knob.
    pub gate_lower_bound: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MlpPlan {
    Dense(DenseMlpPlan),
    Moe(MoeMlpPlan),
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseMlpPlan {
    pub intermediate_size: u32,
    pub activation: ActivationPlan,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoeMlpPlan {
    pub expert_count: u32,
    pub experts_per_token: u32,
    pub expert_intermediate_size: u32,
    pub router: RouterPlan,
    pub shared: Option<SharedMlpPlan>,
    pub activation: ActivationPlan,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SharedMlpPlan {
    pub intermediate_size: u32,
    pub gated: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RouterPlan {
    Softmax,
    Sigmoid {
        normalize_selected: bool,
        scaling_factor: f32,
        selection_bias: bool,
    },
    SqrtSoftplus {
        normalize_selected: bool,
        scaling_factor: f32,
        selection_bias: bool,
    },
    TokenIdHash {
        score: RouterScorePlan,
        normalize_selected: bool,
        scaling_factor: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterScorePlan {
    Softmax,
    Sigmoid,
    SqrtSoftplus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ActivationPlan {
    Silu,
    GeluTanh,
    SwiGluOai {
        alpha: f32,
        limit: f32,
    },
    SwiGluClamped {
        limit: f32,
    },
    /// glm5_next: the clamp is PRE-activation and asymmetric — `silu(gate.min(limit)) *
    /// up.clamp(-limit, limit)` (gate has NO lower bound). `SwiGluClamped` clamps the silu
    /// OUTPUT instead (`silu(gate).min(limit)`); the two differ numerically above the limit.
    SwiGluPreClamped {
        limit: f32,
    },
    Named(String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResidualTopology {
    Serial,
    Gemma {
        post_attention_norm: NormPlan,
        post_mlp_norm: NormPlan,
        layer_scale: GemmaLayerScale,
        parallel_moe: Option<GemmaParallelMoePlan>,
    },
    HyperConnections {
        streams: u32,
        epsilon: f32,
        sinkhorn_iterations: u32,
        /// How the streams collapse back to one at model exit. Per-layer mixing is the
        /// same Sinkhorn program either way.
        collapse: HcCollapse,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HcCollapse {
    /// dsv4: sigmoid-gated head collapse (`hc_head`, dsv4_forward.rs).
    GatedHead,
    /// glm5_next: unweighted mean over the streams (`Glm5NextTextHyperHead`).
    Mean,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GemmaParallelMoePlan {
    pub shared_post_norm: NormPlan,
    pub routed_pre_norm: NormPlan,
    pub routed_post_norm: NormPlan,
    pub router_input_scale: bool,
    pub per_expert_output_scale: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmaLayerScale {
    Learned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatePlan {
    KvCache {
        key_width: u32,
        value_width: u32,
    },
    SlidingKvCache {
        key_width: u32,
        value_width: u32,
        window: u32,
    },
    Recurrent {
        conv_width: u32,
        conv_kernel: u32,
        state_width: u32,
    },
    LatentKvCache {
        width: u32,
        /// Width of the per-token DSA indexer state row (`[k_norm(wk(x)) | gate_scores]`,
        /// `2 * index_head_dim`), or 0 when the layer runs no k-pool indexer. The indexer
        /// must re-derive pool keys from EVERY cached position, so its packed state is
        /// context-linear exactly like the latent plane and is allocated beside it.
        index_width: u32,
    },
    CompressedAttention {
        window: u32,
        head_dim: u32,
        compressor_ratio: Option<u32>,
        sparse_top_k: Option<u32>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum LogitsTransform {
    Softcap(f32),
    SuppressTokens(Vec<u32>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct MtpBlockPlan {
    pub depth: u32,
    pub layer: LayerPlan,
    pub input: MtpInputPlan,
    pub output: MtpOutputPlan,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MtpInputPlan {
    pub embedding_norm: NormPlan,
    pub hidden_norm: NormPlan,
    pub fusion: MtpFusionPlan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpFusionPlan {
    ConcatenateProjection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtpOutputPlan {
    pub norm: MtpTensorPolicy,
    pub projection: MtpTensorPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpTensorPolicy {
    PreferPrivateThenModel,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DrafterPlan {
    Dspark(DsparkPlan),
}

#[derive(Debug, Clone, PartialEq)]
pub struct DsparkPlan {
    pub block_size: u32,
    pub noise_token_id: u32,
    pub target_layer_ids: Vec<u32>,
    pub markov_rank: u32,
    pub blocks: Vec<LayerPlan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanCompileError {
    EmptyModel,
    ModelPackMismatch { pack: &'static str, arch: String },
    MissingTinyFixture { pack: &'static str },
    MissingGatedDeltaNetConfig { layer: u32 },
    InvalidGatedDeltaNetConfig { layer: u32, field: &'static str },
    InvalidAttentionGeometry { layer: u32, field: &'static str },
    UndeclaredAttentionGate { layer: u32 },
    InvalidMoeConfig { layer: u32, field: &'static str },
    InvalidVisionConfig { field: &'static str },
    InvalidMultimodalConfig { field: &'static str },
    MissingExternalDraftPlan,
    ExternalDraftMismatch { field: &'static str },
}

impl std::fmt::Display for PlanCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyModel => write!(f, "model plan requires at least one trunk layer"),
            Self::ModelPackMismatch { pack, arch } => {
                write!(f, "model pack {pack} does not accept architecture {arch}")
            }
            Self::MissingTinyFixture { pack } => {
                write!(f, "model pack {pack} has no native tiny reference fixture")
            }
            Self::MissingGatedDeltaNetConfig { layer } => {
                write!(f, "layer {layer} is gated-deltanet but has no SSM metadata")
            }
            Self::InvalidGatedDeltaNetConfig { layer, field } => {
                write!(f, "layer {layer} has invalid gated-deltanet field {field}")
            }
            Self::InvalidAttentionGeometry { layer, field } => {
                write!(f, "layer {layer} has invalid attention field {field}")
            }
            Self::UndeclaredAttentionGate { layer } => {
                write!(f, "layer {layer} has no declared attention gate layout")
            }
            Self::InvalidMoeConfig { layer, field } => {
                write!(f, "layer {layer} has invalid MoE field {field}")
            }
            Self::InvalidVisionConfig { field } => {
                write!(f, "vision encoder has invalid field {field}")
            }
            Self::InvalidMultimodalConfig { field } => {
                write!(f, "multimodal injection has invalid field {field}")
            }
            Self::MissingExternalDraftPlan => {
                write!(f, "external draft artifact has no typed MTP blocks")
            }
            Self::ExternalDraftMismatch { field } => {
                write!(f, "external draft plan mismatches the trunk field {field}")
            }
        }
    }
}

impl std::error::Error for PlanCompileError {}

impl ModelPlan {
    pub fn compile(cfg: &ModelConfig) -> Result<Self, PlanCompileError> {
        let trunk_layers = cfg.n_layer.saturating_sub(cfg.nextn_predict_layers);
        if trunk_layers == 0 {
            return Err(PlanCompileError::EmptyModel);
        }

        let norm = NormPlan {
            kind: NormKind::Rms,
            epsilon: cfg.rms_eps,
            weight_transform: norm_weight_transform(cfg),
        };
        let mut layers = Vec::with_capacity(trunk_layers as usize);
        for index in 0..trunk_layers {
            layers.push(compile_layer(cfg, index, false, norm)?);
        }

        let mut mtp_blocks = Vec::with_capacity(cfg.nextn_predict_layers as usize);
        for depth in 0..cfg.nextn_predict_layers {
            let index = trunk_layers + depth;
            mtp_blocks.push(MtpBlockPlan {
                depth,
                layer: compile_layer(cfg, index, true, norm)?,
                input: MtpInputPlan {
                    embedding_norm: norm,
                    hidden_norm: norm,
                    fusion: MtpFusionPlan::ConcatenateProjection,
                },
                output: MtpOutputPlan {
                    norm: MtpTensorPolicy::PreferPrivateThenModel,
                    projection: MtpTensorPolicy::PreferPrivateThenModel,
                },
            });
        }

        let mut logits = Vec::new();
        if let Some(gemma) = cfg.gemma4.as_ref() {
            if gemma.final_logit_softcapping > 0.0 {
                logits.push(LogitsTransform::Softcap(gemma.final_logit_softcapping));
            }
            if !gemma.suppress_tokens.is_empty() {
                logits.push(LogitsTransform::SuppressTokens(
                    gemma.suppress_tokens.clone(),
                ));
            }
        }

        if cfg.vision.is_some() && cfg.vision_glm5.is_some() {
            return Err(PlanCompileError::InvalidVisionConfig {
                field: "two vision programs in one config",
            });
        }
        let vision = cfg
            .vision
            .as_ref()
            .map(|vision| compile_vision(cfg, vision).map(VisionPlan::Factored))
            .or_else(|| {
                cfg.vision_glm5
                    .as_ref()
                    .map(|vision| compile_vision_glm5(cfg, vision).map(VisionPlan::Glm5Fused))
            })
            .transpose()?;
        let multimodal = match (cfg.multimodal, vision.as_ref()) {
            // glm5_next carries its splice ids inside vision_config truth, not the generic
            // image_token_id/vision_soft_tokens_per_image pair.
            (None, Some(VisionPlan::Glm5Fused(_))) => {
                let v = cfg.vision_glm5.as_ref().expect("Glm5Fused implies config");
                for (field, id) in [
                    ("image_token_id", v.image_token_id),
                    ("image_start_token_id", v.image_start_token_id),
                    ("image_end_token_id", v.image_end_token_id),
                ] {
                    if id >= cfg.n_vocab {
                        return Err(PlanCompileError::InvalidMultimodalConfig { field });
                    }
                }
                Some(VisionTokenInjectionPlan {
                    placeholder_token_id: v.image_token_id,
                    tokens_per_image: None, // grid-derived (see the field's doc)
                    start_token_id: Some(v.image_start_token_id),
                    end_token_id: Some(v.image_end_token_id),
                })
            }
            (None, _) => None,
            (Some(_), None) => {
                return Err(PlanCompileError::InvalidMultimodalConfig {
                    field: "vision_encoder",
                });
            }
            (Some(_), Some(VisionPlan::Glm5Fused(_))) => {
                return Err(PlanCompileError::InvalidMultimodalConfig {
                    field: "generic multimodal config on a glm5_next tower",
                });
            }
            (Some(multimodal), Some(VisionPlan::Factored(vision))) => {
                if multimodal.image_token_id >= cfg.n_vocab {
                    return Err(PlanCompileError::InvalidMultimodalConfig {
                        field: "image_token_id",
                    });
                }
                if multimodal.vision_soft_tokens_per_image == 0 {
                    return Err(PlanCompileError::InvalidMultimodalConfig {
                        field: "vision_soft_tokens_per_image",
                    });
                }
                if vision.projection_output_size != cfg.n_embd {
                    return Err(PlanCompileError::InvalidMultimodalConfig {
                        field: "projection_output_size",
                    });
                }
                Some(VisionTokenInjectionPlan {
                    placeholder_token_id: multimodal.image_token_id,
                    tokens_per_image: Some(multimodal.vision_soft_tokens_per_image),
                    start_token_id: None,
                    end_token_id: None,
                })
            }
        };

        Ok(Self {
            arch: cfg.arch.clone(),
            hidden_size: cfg.n_embd,
            vocab_size: cfg.n_vocab,
            context_length: cfg.context_length,
            embedding_scale: if cfg.gemma4.is_some() {
                (cfg.n_embd as f32).sqrt()
            } else {
                1.0
            },
            vision,
            multimodal,
            partition_boundaries: (1..trunk_layers as usize).collect(),
            layers,
            output_norm: norm,
            logits,
            mtp_blocks,
            drafter: None,
            draft_source: if cfg.nextn_predict_layers > 0 {
                DraftSourcePlan::Embedded
            } else {
                DraftSourcePlan::None
            },
            sampling_defaults: None,
        })
    }

    pub fn attach_external_draft(&mut self, draft: &ModelPlan) -> Result<(), PlanCompileError> {
        if self.hidden_size != draft.hidden_size {
            return Err(PlanCompileError::ExternalDraftMismatch {
                field: "hidden_size",
            });
        }
        if draft.vocab_size != 0 && self.vocab_size != draft.vocab_size {
            return Err(PlanCompileError::ExternalDraftMismatch {
                field: "vocab_size",
            });
        }
        if draft.mtp_blocks.is_empty() {
            return Err(PlanCompileError::MissingExternalDraftPlan);
        }
        self.mtp_blocks = draft.mtp_blocks.clone();
        self.drafter = draft.drafter.clone();
        Ok(())
    }

    pub fn operations(&self) -> Vec<OperationKind> {
        self.collect_operations(true, true)
    }

    /// Operations selected by trunk priming. MTP blocks are a separate executable subplan and do
    /// not get to disable a trunk-only execution mode merely because they share the checkpoint.
    pub fn trunk_operations(&self) -> Vec<OperationKind> {
        self.collect_operations(false, false)
    }

    pub fn multimodal_prefill_operations(&self) -> Vec<OperationKind> {
        self.collect_operations(false, true)
    }

    pub fn draft_operations(&self) -> Option<Vec<OperationKind>> {
        let mut operations = vec![OperationKind::Embedding];
        for block in &self.mtp_blocks {
            operations.push(OperationKind::Mtp);
            operations.push(OperationKind::MtpFusion);
            block.layer.operations(&mut operations);
            operations.push(OperationKind::MtpHead);
        }
        if let Some(DrafterPlan::Dspark(dspark)) = self.drafter.as_ref() {
            operations.push(OperationKind::DsparkFusion);
            for block in &dspark.blocks {
                block.operations(&mut operations);
            }
            operations.push(OperationKind::DsparkMarkovHead);
            operations.push(OperationKind::DsparkConfidenceHead);
        }
        if self.mtp_blocks.is_empty() && self.drafter.is_none() {
            return None;
        }
        operations.push(OperationKind::RmsNorm);
        for transform in &self.logits {
            operations.push(match transform {
                LogitsTransform::Softcap(_) => OperationKind::LogitsSoftcap,
                LogitsTransform::SuppressTokens(_) => OperationKind::LogitsMask,
            });
        }
        operations.push(OperationKind::OutputProjection);
        Some(operations)
    }

    fn collect_operations(&self, include_mtp: bool, include_frontend: bool) -> Vec<OperationKind> {
        let mut operations = Vec::new();
        if include_frontend && let Some(vision) = self.vision.as_ref() {
            match vision {
                VisionPlan::Factored(vision) => {
                    operations.push(OperationKind::VisionPatchEmbedding);
                    for _ in &vision.layers {
                        operations.push(OperationKind::VisionBidirectionalAttention);
                        operations.push(OperationKind::VisionMlp);
                    }
                    if vision.standardize {
                        operations.push(OperationKind::VisionStandardize);
                    }
                    operations.push(OperationKind::VisionProjection);
                }
                VisionPlan::Glm5Fused(vision) => {
                    operations.push(OperationKind::VisionPatchEmbedding);
                    for _ in 0..vision.depth {
                        operations.push(OperationKind::VisionBidirectionalAttention);
                        operations.push(OperationKind::VisionMlp);
                    }
                    operations.push(OperationKind::VisionDownsample);
                    operations.push(OperationKind::VisionProjection);
                }
            }
        }
        if include_frontend && self.multimodal.is_some() {
            operations.push(OperationKind::VisionTokenInjection);
        }
        operations.push(OperationKind::Embedding);
        for layer in &self.layers {
            layer.operations(&mut operations);
        }
        if include_mtp {
            for block in &self.mtp_blocks {
                operations.push(OperationKind::Mtp);
                operations.push(OperationKind::MtpFusion);
                block.layer.operations(&mut operations);
                operations.push(OperationKind::MtpHead);
            }
            if let Some(DrafterPlan::Dspark(dspark)) = self.drafter.as_ref() {
                operations.push(OperationKind::DsparkFusion);
                for block in &dspark.blocks {
                    block.operations(&mut operations);
                }
                operations.push(OperationKind::DsparkMarkovHead);
                operations.push(OperationKind::DsparkConfidenceHead);
            }
        }
        operations.push(OperationKind::RmsNorm);
        for transform in &self.logits {
            operations.push(match transform {
                LogitsTransform::Softcap(_) => OperationKind::LogitsSoftcap,
                LogitsTransform::SuppressTokens(_) => OperationKind::LogitsMask,
            });
        }
        operations.push(OperationKind::OutputProjection);
        operations
    }

    /// Reduce per-operation implementation facts into model-level execution capabilities. The
    /// callback is monomorphized at plan-selection time; it is not a token-path plugin interface.
    pub fn derive_capabilities(
        &self,
        mut support: impl FnMut(OperationKind) -> OperationSupport,
    ) -> PlanCapabilities {
        derive_capabilities(
            self.operations(),
            self.draft_operations(),
            self.trunk_operations(),
            self.valid_partition_boundaries(),
            &mut support,
        )
    }

    pub fn derive_trunk_capabilities(
        &self,
        mut support: impl FnMut(OperationKind) -> OperationSupport,
    ) -> PlanCapabilities {
        derive_capabilities(
            self.trunk_operations(),
            None,
            self.trunk_operations(),
            self.valid_partition_boundaries(),
            &mut support,
        )
    }

    pub fn derive_multimodal_prefill_capabilities(
        &self,
        mut support: impl FnMut(OperationKind) -> OperationSupport,
    ) -> PlanCapabilities {
        derive_capabilities(
            self.multimodal_prefill_operations(),
            self.draft_operations(),
            self.trunk_operations(),
            self.valid_partition_boundaries(),
            &mut support,
        )
    }

    fn valid_partition_boundaries(&self) -> bool {
        !self.partition_boundaries.is_empty()
            && self
                .partition_boundaries
                .iter()
                .all(|&boundary| boundary > 0 && boundary < self.layers.len())
    }

    pub fn requires_hybrid_executor(&self) -> bool {
        self.operations().into_iter().any(|operation| {
            !matches!(
                operation,
                OperationKind::Embedding
                    | OperationKind::RmsNorm
                    | OperationKind::FullAttention
                    | OperationKind::DenseMlp
                    | OperationKind::SiluActivation
                    | OperationKind::SerialResidual
                    | OperationKind::KvState
                    | OperationKind::LogitsSoftcap
                    | OperationKind::LogitsMask
                    | OperationKind::OutputProjection
            )
        })
    }
}

fn compile_vision(
    cfg: &ModelConfig,
    vision: &crate::config::VisionConfig,
) -> Result<VisionEncoderPlan, PlanCompileError> {
    for (field, valid) in [
        ("hidden_size", vision.hidden_size > 0),
        ("intermediate_size", vision.intermediate_size > 0),
        ("layer_count", vision.layer_count > 0),
        ("attention_heads", vision.attention_heads > 0),
        ("kv_heads", vision.kv_heads > 0),
        ("head_dim", vision.head_dim > 0),
        ("patch_size", vision.patch_size > 0),
        (
            "position_embedding_size",
            vision.position_embedding_size > 0,
        ),
        ("position_axes", vision.position_axes > 0),
    ] {
        if !valid {
            return Err(PlanCompileError::InvalidVisionConfig { field });
        }
    }
    let norm = NormPlan {
        kind: NormKind::Rms,
        epsilon: vision.rms_eps,
        weight_transform: WeightTransform::Identity,
    };
    let activation = match vision.activation.as_str() {
        "gelu_pytorch_tanh" | "gelu_tanh" => ActivationPlan::GeluTanh,
        other => ActivationPlan::Named(other.to_string()),
    };
    let layers = (0..vision.layer_count)
        .map(|index| VisionLayerPlan {
            index,
            input_norm: norm,
            attention: VisionAttentionPlan {
                query_heads: vision.attention_heads,
                kv_heads: vision.kv_heads,
                head_dim: vision.head_dim,
                rope: RopePlan {
                    dimensions: vision.head_dim,
                    base: vision.rope_theta,
                    factors: RopeFactors::None,
                },
                bidirectional: true,
                qk_norm: true,
            },
            post_attention_norm: norm,
            pre_mlp_norm: norm,
            mlp: DenseMlpPlan {
                intermediate_size: vision.intermediate_size,
                activation: activation.clone(),
            },
            post_mlp_norm: norm,
        })
        .collect();
    Ok(VisionEncoderPlan {
        hidden_size: vision.hidden_size,
        context_length: vision.context_length,
        patch: VisionPatchPlan {
            channels: 3,
            patch_size: vision.patch_size,
            position_axes: vision.position_axes,
            position_embedding_size: vision.position_embedding_size,
        },
        layers,
        projection_output_size: cfg.n_embd,
        pooling_kernel_size: vision.pooling_kernel_size,
        standardize: vision.standardize,
        clipped_linears: vision.clipped_linears,
    })
}

/// Compile the glm5_next tower program. Semantics pinned against transformers 5.16.1
/// `Glm5NextVisionModel` (lane/glm5-vision, research/glm5-vision-20260830): rope-only
/// positions (theta 10000 upstream hardcode), silu clamped-SwiGLU, RMS eps from
/// vision_config, merger output == trunk embedding width.
fn compile_vision_glm5(
    cfg: &ModelConfig,
    vision: &crate::config::Glm5VisionConfig,
) -> Result<Glm5VisionPlan, PlanCompileError> {
    for (field, valid) in [
        ("depth", vision.depth > 0),
        ("hidden_size", vision.hidden_size > 0),
        (
            "num_heads",
            vision.num_heads > 0 && vision.hidden_size.is_multiple_of(vision.num_heads),
        ),
        ("intermediate_size", vision.intermediate_size > 0),
        ("patch_size", vision.patch_size > 0),
        ("temporal_patch_size", vision.temporal_patch_size > 0),
        ("spatial_merge_size", vision.spatial_merge_size > 0),
        (
            "projection_intermediate_size",
            vision.projection_intermediate_size > 0,
        ),
        ("swiglu_limit", vision.swiglu_limit > 0.0),
        ("rms_norm_eps", vision.rms_norm_eps > 0.0),
        ("in_channels", vision.in_channels == 3),
        // silu is the pinned activation; a different name is a different program.
        ("hidden_act", vision.hidden_act == "silu"),
        (
            "attention_bias",
            vision.attention_bias, // census: qkv/proj/mlp biases are all present
        ),
        // The merger feeds trunk embedding rows directly; a width mismatch means the
        // artifact belongs to a different trunk.
        ("out_hidden_size", vision.out_hidden_size == cfg.n_embd),
    ] {
        if !valid {
            return Err(PlanCompileError::InvalidVisionConfig { field });
        }
    }
    let head_dim = vision.hidden_size / vision.num_heads;
    // The 2D rope splits head_dim into an h half and a w half, each pair-rotated: four-way
    // divisibility is structural.
    if !head_dim.is_multiple_of(4) {
        return Err(PlanCompileError::InvalidVisionConfig { field: "head_dim" });
    }
    Ok(Glm5VisionPlan {
        depth: vision.depth,
        hidden_size: vision.hidden_size,
        heads: vision.num_heads,
        head_dim,
        intermediate_size: vision.intermediate_size,
        patch_size: vision.patch_size,
        temporal_patch_size: vision.temporal_patch_size,
        spatial_merge_size: vision.spatial_merge_size,
        out_hidden_size: vision.out_hidden_size,
        projection_intermediate_size: vision.projection_intermediate_size,
        swiglu_limit: vision.swiglu_limit,
        rope_theta: 10_000.0,
        norm: NormPlan {
            kind: NormKind::Rms,
            epsilon: vision.rms_norm_eps,
            weight_transform: WeightTransform::Identity,
        },
        in_channels: vision.in_channels,
        patch_input_width: vision.in_channels
            * vision.temporal_patch_size
            * vision.patch_size
            * vision.patch_size,
    })
}

impl ModelConfig {
    /// Select the existing handwritten executor from canonical operations. This is a migration
    /// bridge: the generic reference executor consumes the plan directly, while tuned runtime
    /// paths still have two loaders. New families do not enter either path by architecture name.
    pub fn uses_hybrid_executor(&self) -> bool {
        let Ok(plan) = ModelPlan::compile(self) else {
            return true;
        };
        plan.requires_hybrid_executor()
    }
}

fn derive_capabilities(
    operations: Vec<OperationKind>,
    draft_operations: Option<Vec<OperationKind>>,
    verify_operations: Vec<OperationKind>,
    valid_partition_boundaries: bool,
    support: &mut impl FnMut(OperationKind) -> OperationSupport,
) -> PlanCapabilities {
    let mut batch = CapabilityStatus::supported();
    let mut draft = CapabilityStatus::supported();
    let mut verify = CapabilityStatus::supported();
    let mut pipeline = CapabilityStatus::supported();
    let mut graphs = CapabilityStatus::supported();

    for operation in operations {
        let implemented = support(operation);
        batch.require(operation, implemented.batch);
        pipeline.require(operation, implemented.pipeline);
        graphs.require(operation, implemented.cuda_graph);
    }
    if let Some(operations) = draft_operations {
        for operation in operations {
            draft.require(operation, support(operation).spec_draft);
        }
    } else {
        draft.require(OperationKind::DraftPlan, false);
    }
    for operation in verify_operations {
        verify.require(operation, support(operation).spec_verify);
    }
    let boundary_support = support(OperationKind::PipelineBoundary);
    pipeline.require(
        OperationKind::PipelineBoundary,
        valid_partition_boundaries && boundary_support.pipeline,
    );

    PlanCapabilities {
        batch,
        speculative: draft.and(verify),
        pipeline,
        cuda_graph: graphs,
    }
}

impl LayerPlan {
    fn operations(&self, operations: &mut Vec<OperationKind>) {
        operations.push(OperationKind::RmsNorm);
        match &self.attention {
            AttentionPlan::Full(attention) => {
                operations.push(OperationKind::FullAttention);
                push_gate(attention.output_gate, operations);
            }
            AttentionPlan::SlidingWindow { attention, .. } => {
                operations.push(OperationKind::SlidingWindowAttention);
                push_gate(attention.output_gate, operations);
            }
            AttentionPlan::Mla(MlaAttentionPlan::LatentKv { sparse_index, .. }) => {
                operations.push(OperationKind::LatentMlaAttention);
                push_sparse_index(*sparse_index, operations);
            }
            AttentionPlan::Mla(MlaAttentionPlan::CompressedKv {
                compressor,
                sparse_index,
                ..
            }) => {
                operations.push(OperationKind::CompressedMlaAttention);
                if compressor.is_some() {
                    operations.push(OperationKind::KvCompressor);
                }
                push_sparse_index(*sparse_index, operations);
            }
            AttentionPlan::GatedDeltaNet(_) => operations.push(OperationKind::GatedDeltaNet),
            AttentionPlan::KimiDeltaNet(_) => operations.push(OperationKind::KimiDeltaNet),
        }
        operations.push(match self.state {
            StatePlan::KvCache { .. } => OperationKind::KvState,
            StatePlan::SlidingKvCache { .. } => OperationKind::SlidingKvState,
            StatePlan::Recurrent { .. } => OperationKind::RecurrentState,
            StatePlan::LatentKvCache { .. } => OperationKind::LatentKvState,
            StatePlan::CompressedAttention { .. } => OperationKind::CompressedAttentionState,
        });
        operations.push(OperationKind::RmsNorm);
        match &self.mlp {
            MlpPlan::Dense(mlp) => {
                operations.push(OperationKind::DenseMlp);
                push_activation(&mlp.activation, operations);
            }
            MlpPlan::Moe(moe) => {
                operations.push(OperationKind::MoeMlp);
                operations.push(match moe.router {
                    RouterPlan::Softmax => OperationKind::SoftmaxRouter,
                    RouterPlan::Sigmoid { .. } => OperationKind::SigmoidRouter,
                    RouterPlan::SqrtSoftplus { .. } => OperationKind::SqrtSoftplusRouter,
                    RouterPlan::TokenIdHash { .. } => OperationKind::TokenHashRouter,
                });
                if moe.shared.is_some() {
                    operations.push(OperationKind::SharedMlp);
                }
                push_activation(&moe.activation, operations);
            }
        }
        operations.push(match self.residual {
            ResidualTopology::Serial => OperationKind::SerialResidual,
            ResidualTopology::Gemma {
                parallel_moe: Some(_),
                ..
            } => OperationKind::GemmaParallelMoeResidual,
            ResidualTopology::Gemma {
                parallel_moe: None, ..
            } => OperationKind::GemmaResidual,
            ResidualTopology::HyperConnections { .. } => OperationKind::HyperConnections,
        });
    }
}

fn push_activation(activation: &ActivationPlan, operations: &mut Vec<OperationKind>) {
    operations.push(match activation {
        ActivationPlan::Silu => OperationKind::SiluActivation,
        ActivationPlan::GeluTanh => OperationKind::GeluTanhActivation,
        ActivationPlan::SwiGluOai { .. } => OperationKind::SwiGluOaiActivation,
        ActivationPlan::SwiGluClamped { .. } => OperationKind::SwiGluClampedActivation,
        ActivationPlan::SwiGluPreClamped { .. } => OperationKind::SwiGluPreClampedActivation,
        ActivationPlan::Named(_) => OperationKind::NamedActivation,
    });
}

fn push_sparse_index(index: SparseIndexPlan, operations: &mut Vec<OperationKind>) {
    match index {
        SparseIndexPlan::None => {}
        SparseIndexPlan::Own { .. } => operations.push(OperationKind::SparseIndex),
        SparseIndexPlan::SharedFromPrevious { .. } => {
            operations.push(OperationKind::SharedSparseIndex)
        }
    }
}

fn push_gate(gate: AttentionGateKind, operations: &mut Vec<OperationKind>) {
    match gate {
        AttentionGateKind::None => {}
        AttentionGateKind::FusedQ => operations.push(OperationKind::FusedAttentionGate),
        AttentionGateKind::SeparateHead => operations.push(OperationKind::SeparateAttentionGate),
    }
}

fn compile_layer(
    cfg: &ModelConfig,
    index: u32,
    mtp: bool,
    norm: NormPlan,
) -> Result<LayerPlan, PlanCompileError> {
    let (attention, state) = compile_attention(cfg, index, mtp)?;
    Ok(LayerPlan {
        index,
        pre_attention_norm: norm,
        attention,
        pre_mlp_norm: norm,
        mlp: compile_mlp(cfg, index, mtp)?,
        // glm5_next's NextN layer carries NO hc_* tensors (census: 45 trunk rows only) —
        // the MTP block runs a plain serial residual outside the stream stack.
        residual: if mtp && cfg.glm5.is_some() {
            ResidualTopology::Serial
        } else {
            residual_topology(cfg)
        },
        state,
    })
}

/// Per-token indexer state width for a `LatentKvCache` layer: `[k_norm(wk(x)) | gate_scores]`,
/// both `index_head_dim` wide. Only the k-pool indexer keeps state — the per-token variant
/// (GLM-5.2 / dsv4) rescores raw cache rows and owns no packed plane here.
fn latent_index_width(sparse_index: &SparseIndexPlan) -> u32 {
    match sparse_index {
        SparseIndexPlan::Own {
            head_dim,
            kpool: Some(_),
            ..
        } => 2 * head_dim,
        _ => 0,
    }
}

fn compile_attention(
    cfg: &ModelConfig,
    index: u32,
    mtp: bool,
) -> Result<(AttentionPlan, StatePlan), PlanCompileError> {
    if let Some(g5) = cfg.glm5.as_ref() {
        // Trunk KDA layers only — the NextN/MTP layer is MLA+indexer (census: the 12th
        // indexer tensor set lives on the MTP layer).
        if !mtp && g5.is_kda_layer(index) {
            let qkv = g5.linear_num_heads * g5.linear_head_dim;
            return Ok((
                AttentionPlan::KimiDeltaNet(KimiDeltaNetPlan {
                    num_heads: g5.linear_num_heads,
                    head_dim: g5.linear_head_dim,
                    conv_kernel: g5.linear_conv_kernel,
                    gate_lower_bound: g5.gate_lower_bound,
                }),
                StatePlan::Recurrent {
                    // One grouped conv over the fused [q|k|v] planes.
                    conv_width: 3 * qkv,
                    conv_kernel: g5.linear_conv_kernel,
                    state_width: g5.linear_num_heads * g5.linear_head_dim * g5.linear_head_dim,
                },
            ));
        }
        // Every glm5_next attention layer owns its indexer (indexer_types is all "full";
        // MTP carries its own tensors and `index_share_for_mtp_iteration` refers to reuse
        // across MTP iterations, not base-vs-MTP sharing).
        let sparse_index = SparseIndexPlan::Own {
            heads: g5.index_n_heads,
            head_dim: g5.index_head_dim,
            top_k: g5.index_topk,
            kpool: g5.index_kpool_compress.then_some(KpoolPlan {
                pool: g5.index_kpool,
                always_select_tail: g5.index_kpool_always_select_tail,
            }),
        };
        return Ok((
            AttentionPlan::Mla(MlaAttentionPlan::LatentKv {
                query_heads: cfg.n_head,
                q_lora_rank: g5.q_lora_rank,
                kv_lora_rank: g5.kv_lora_rank,
                qk_head_dim: g5.qk_head_dim,
                rope_head_dim: g5.qk_rope_head_dim,
                value_head_dim: g5.v_head_dim,
                // NoPE: qk_rope_head_dim is 0 and no rotary applies anywhere in the text
                // stack (mla_use_nope, cross-checked at config parse).
                rope: RopePlan {
                    dimensions: g5.qk_rope_head_dim,
                    base: cfg.rope_freq_base,
                    factors: RopeFactors::None,
                },
                sparse_index,
            }),
            StatePlan::LatentKvCache {
                width: g5.kv_lora_rank + g5.qk_rope_head_dim,
                index_width: latent_index_width(&sparse_index),
            },
        ));
    }

    if let Some(mla) = cfg.mla.as_ref() {
        let sparse_index = match mla.dsa.as_ref() {
            None => SparseIndexPlan::None,
            Some(_) if mtp => SparseIndexPlan::None,
            Some(dsa)
                if dsa
                    .indexer_full
                    .get(index as usize)
                    .copied()
                    .unwrap_or(true) =>
            {
                SparseIndexPlan::Own {
                    heads: dsa.index_n_heads,
                    head_dim: dsa.index_head_dim,
                    top_k: dsa.index_top_k,
                    kpool: None,
                }
            }
            Some(dsa) => SparseIndexPlan::SharedFromPrevious {
                top_k: dsa.index_top_k,
            },
        };
        return Ok((
            AttentionPlan::Mla(MlaAttentionPlan::LatentKv {
                query_heads: cfg.n_head,
                q_lora_rank: mla.q_lora_rank,
                kv_lora_rank: mla.kv_lora_rank,
                qk_head_dim: mla.qk_head_dim,
                rope_head_dim: mla.qk_rope_head_dim,
                value_head_dim: mla.v_head_dim,
                rope: RopePlan {
                    dimensions: mla.qk_rope_head_dim,
                    base: cfg.rope_freq_base,
                    factors: RopeFactors::None,
                },
                sparse_index,
            }),
            StatePlan::LatentKvCache {
                width: mla.latent_dim(),
                index_width: latent_index_width(&sparse_index),
            },
        ));
    }

    if let Some(dsv4) = cfg.dsv4.as_ref() {
        let ratio = dsv4.compress_ratio(index);
        let sparse_index = if dsv4.has_indexer(index) && !mtp {
            SparseIndexPlan::Own {
                heads: dsv4.index_n_heads,
                head_dim: dsv4.index_head_dim,
                top_k: dsv4.index_topk,
                kpool: None,
            }
        } else {
            SparseIndexPlan::None
        };
        return Ok((
            AttentionPlan::Mla(MlaAttentionPlan::CompressedKv {
                query_heads: cfg.n_head,
                q_lora_rank: dsv4.q_lora_rank,
                latent_head_dim: dsv4.head_dim,
                rope_head_dim: dsv4.qk_rope_head_dim,
                output_lora_rank: dsv4.o_lora_rank,
                output_groups: dsv4.o_groups,
                window: dsv4.sliding_window,
                rope: RopePlan {
                    dimensions: dsv4.qk_rope_head_dim,
                    base: if ratio > 0 {
                        dsv4.compress_rope_theta
                    } else {
                        cfg.rope_freq_base
                    },
                    factors: if ratio > 0 {
                        RopeFactors::Yarn {
                            factor: dsv4.rope_yarn_factor,
                            original_context: dsv4.rope_yarn_orig_ctx,
                            beta_fast: dsv4.rope_yarn_beta_fast,
                            beta_slow: dsv4.rope_yarn_beta_slow,
                        }
                    } else {
                        RopeFactors::None
                    },
                },
                compressor: (ratio > 0).then_some(KvCompressorPlan {
                    ratio,
                    latent_dim: if dsv4.has_indexer(index) {
                        2 * dsv4.head_dim
                    } else {
                        dsv4.head_dim
                    },
                }),
                sparse_index,
            }),
            StatePlan::CompressedAttention {
                window: dsv4.sliding_window,
                head_dim: dsv4.head_dim,
                compressor_ratio: (ratio > 0).then_some(ratio),
                sparse_top_k: dsv4.has_indexer(index).then_some(dsv4.index_topk),
            },
        ));
    }

    if !mtp && cfg.layer_kind(index) == LayerKind::LinearAttention {
        let ssm = cfg
            .ssm
            .as_ref()
            .ok_or(PlanCompileError::MissingGatedDeltaNetConfig { layer: index })?;
        for (field, value) in [
            ("group_count", ssm.group_count),
            ("time_step_rank", ssm.time_step_rank),
            ("state_size", ssm.state_size),
            ("inner_size", ssm.inner_size),
            ("conv_kernel", ssm.conv_kernel),
        ] {
            if value == 0 {
                return Err(PlanCompileError::InvalidGatedDeltaNetConfig {
                    layer: index,
                    field,
                });
            }
        }
        let value_head_dim = ssm.inner_size / ssm.time_step_rank;
        let attention = GatedDeltaNetPlan {
            key_heads: ssm.group_count,
            value_heads: ssm.time_step_rank,
            key_head_dim: ssm.state_size,
            value_head_dim,
            conv_kernel: ssm.conv_kernel,
        };
        return Ok((
            AttentionPlan::GatedDeltaNet(attention),
            StatePlan::Recurrent {
                conv_width: 2 * ssm.group_count * ssm.state_size
                    + ssm.time_step_rank * value_head_dim,
                conv_kernel: ssm.conv_kernel,
                state_width: ssm.time_step_rank * ssm.state_size * value_head_dim,
            },
        ));
    }

    let geometry = attention_geometry(cfg, index)?;
    for (field, value) in [
        ("query_heads", geometry.query_heads),
        ("kv_heads", geometry.kv_heads),
        ("key_head_dim", geometry.key_head_dim),
        ("value_head_dim", geometry.value_head_dim),
        ("rope_dimensions", geometry.rope.dimensions),
    ] {
        if value == 0 {
            return Err(PlanCompileError::InvalidAttentionGeometry {
                layer: index,
                field,
            });
        }
    }
    let key_width = geometry.kv_heads * geometry.key_head_dim;
    let value_width = geometry.kv_heads * geometry.value_head_dim;
    if let Some(window) = attention_window(cfg, index, mtp) {
        Ok((
            AttentionPlan::SlidingWindow {
                attention: geometry,
                window,
            },
            StatePlan::SlidingKvCache {
                key_width,
                value_width,
                window,
            },
        ))
    } else {
        Ok((
            AttentionPlan::Full(geometry),
            StatePlan::KvCache {
                key_width,
                value_width,
            },
        ))
    }
}

fn attention_geometry(
    cfg: &ModelConfig,
    index: u32,
) -> Result<FullAttentionPlan, PlanCompileError> {
    if let Some(gemma) = cfg.gemma4.as_ref() {
        let swa = gemma
            .swa_pattern
            .get(index as usize)
            .copied()
            .unwrap_or(false);
        let key_head_dim = if swa {
            gemma.key_length_swa
        } else {
            gemma.key_length_global
        };
        return Ok(FullAttentionPlan {
            query_heads: cfg.n_head,
            kv_heads: gemma
                .head_count_kv
                .get(index as usize)
                .copied()
                .unwrap_or(cfg.n_head_kv),
            key_head_dim,
            value_head_dim: key_head_dim,
            rope: RopePlan {
                dimensions: if swa {
                    gemma.rope_dims_swa
                } else {
                    gemma.rope_dims_global
                },
                base: if swa {
                    gemma.rope_base_swa
                } else {
                    gemma.rope_base_global
                },
                factors: if swa {
                    RopeFactors::None
                } else {
                    RopeFactors::PartialRotary {
                        factor: gemma.partial_rotary_global,
                    }
                },
            },
            qk_norm: TensorPresence::Required,
            output_gate: AttentionGateKind::None,
            scale: AttentionScale::Fixed(1.0),
            value_projection: if swa {
                ValueProjection::Separate
            } else {
                ValueProjection::ReuseKey
            },
            value_norm: ValueNorm::WeightlessRms,
        });
    }

    if let Some(step) = cfg.step35.as_ref() {
        let swa = step.is_swa(index);
        return Ok(FullAttentionPlan {
            query_heads: step.n_head(index),
            kv_heads: step.n_head_kv(index),
            key_head_dim: cfg.head_dim_k,
            value_head_dim: cfg.head_dim_v,
            rope: RopePlan {
                dimensions: step.n_rot(index),
                base: step.rope_base(index),
                factors: if swa {
                    RopeFactors::None
                } else {
                    RopeFactors::Checkpoint
                },
            },
            qk_norm: TensorPresence::Required,
            output_gate: AttentionGateKind::SeparateHead,
            scale: AttentionScale::InverseSqrtKeyDim,
            value_projection: ValueProjection::Separate,
            value_norm: ValueNorm::None,
        });
    }

    let geometry = cfg.full_attention_geometry_at(index);
    let output_gate = cfg
        .layer_geometry(index)
        .map(|layer| layer.attention_gate)
        .or_else(|| cfg.arch.attention_gate_kind())
        .ok_or(PlanCompileError::UndeclaredAttentionGate { layer: index })?;
    Ok(FullAttentionPlan {
        query_heads: geometry.n_head,
        kv_heads: geometry.n_head_kv,
        key_head_dim: geometry.head_dim_k,
        value_head_dim: geometry.head_dim_v,
        rope: RopePlan {
            dimensions: geometry.n_rot,
            base: geometry.rope_base,
            factors: if geometry.rope_factors {
                RopeFactors::Checkpoint
            } else {
                RopeFactors::None
            },
        },
        qk_norm: qk_norm_presence(cfg),
        output_gate,
        scale: AttentionScale::InverseSqrtKeyDim,
        value_projection: ValueProjection::Separate,
        value_norm: ValueNorm::None,
    })
}

fn attention_window(cfg: &ModelConfig, index: u32, mtp: bool) -> Option<u32> {
    if let Some(step) = cfg.step35.as_ref() {
        return step.is_swa(index).then_some(step.sliding_window);
    }
    if mtp {
        return None;
    }
    if let Some(gemma) = cfg.gemma4.as_ref() {
        return gemma
            .swa_pattern
            .get(index as usize)
            .copied()
            .unwrap_or(false)
            .then_some(gemma.sliding_window);
    }
    cfg.layer_geometry(index)
        .and_then(|geometry| geometry.window)
}

fn compile_mlp(cfg: &ModelConfig, index: u32, mtp: bool) -> Result<MlpPlan, PlanCompileError> {
    // step35-family MTP blocks are DENSE canonical blocks by family semantics: the released
    // Step3.7-flash artifacts ship blk.45/46/47 with ffn_gate/up/down.weight and NO expert
    // tensors while the flat config carries the TRUNK's expert hparams. Family-scoped on
    // purpose — dsv4-class embedded MTP mirrors a MoE trunk layer and stays plan-Moe.
    let mtp_dense = mtp && cfg.step35.is_some();
    if mtp_dense || !layer_uses_moe(cfg, index) {
        let intermediate_size = cfg
            .m3
            .as_ref()
            .map(|m3| m3.dense_intermediate_size)
            .unwrap_or(cfg.n_ff);
        return Ok(MlpPlan::Dense(DenseMlpPlan {
            intermediate_size,
            activation: activation(cfg, index, false),
        }));
    }

    let moe = cfg.moe.as_ref().ok_or(PlanCompileError::InvalidMoeConfig {
        layer: index,
        field: "moe",
    })?;
    for (field, value) in [
        ("expert_count", moe.expert_count),
        ("expert_used_count", moe.expert_used_count),
        ("expert_ff_length", moe.expert_ff_length),
    ] {
        if value == 0 {
            return Err(PlanCompileError::InvalidMoeConfig {
                layer: index,
                field,
            });
        }
    }
    let shared_intermediate_size = if moe.expert_shared_ff_length > 0 {
        moe.expert_shared_ff_length
    } else if let Some(mla) = cfg.mla.as_ref() {
        mla.n_shared_experts * moe.expert_ff_length
    } else if let Some(dsv4) = cfg.dsv4.as_ref() {
        dsv4.n_shared_experts * moe.expert_ff_length
    } else if let Some(g5) = cfg.glm5.as_ref() {
        g5.n_shared_experts * moe.expert_ff_length
    } else {
        0
    };
    Ok(MlpPlan::Moe(MoeMlpPlan {
        expert_count: moe.expert_count,
        experts_per_token: moe.expert_used_count,
        expert_intermediate_size: moe.expert_ff_length,
        router: router(cfg, index),
        shared: (shared_intermediate_size > 0).then_some(SharedMlpPlan {
            intermediate_size: shared_intermediate_size,
            gated: matches!(cfg.arch, Arch::Qwen35Moe),
        }),
        activation: activation(cfg, index, true),
    }))
}

fn layer_uses_moe(cfg: &ModelConfig, index: u32) -> bool {
    let Some(moe) = cfg.moe.as_ref() else {
        return false;
    };
    if moe.expert_count == 0 {
        return false;
    }
    if let Some(m3) = cfg.m3.as_ref() {
        return m3.moe_layer_freq.get(index as usize).copied().unwrap_or(1) != 0;
    }
    if let Some(hy3) = cfg.hy3.as_ref() {
        return index >= hy3.first_k_dense_replace;
    }
    if let Some(mla) = cfg.mla.as_ref() {
        return index >= mla.first_k_dense_replace;
    }
    if let Some(step) = cfg.step35.as_ref() {
        return index >= step.first_k_dense_replace;
    }
    if let Some(g5) = cfg.glm5.as_ref() {
        // mlp_layer_types is cross-checked against first_k_dense_replace at parse; the
        // MTP layer mirrors a MoE trunk layer (dsv4-class embedded MTP) and its index
        // is past the trunk vec, where is_dense_layer answers false.
        return !g5.is_dense_layer(index);
    }
    true
}

fn router(cfg: &ModelConfig, index: u32) -> RouterPlan {
    if let Some(g5) = cfg.glm5.as_ref() {
        // noaux_tc == DeepSeek-V3 recipe: sigmoid scores, selection-only bias
        // (e_score_correction_bias), sum-normalize the selected, then scale.
        return RouterPlan::Sigmoid {
            normalize_selected: g5.norm_topk_prob,
            scaling_factor: g5.routed_scaling_factor,
            selection_bias: true,
        };
    }
    if let Some(dsv4) = cfg.dsv4.as_ref() {
        if dsv4.is_hash_layer(index) {
            return RouterPlan::TokenIdHash {
                score: RouterScorePlan::SqrtSoftplus,
                normalize_selected: dsv4.norm_topk_prob,
                scaling_factor: dsv4.routed_scaling_factor,
            };
        }
        return RouterPlan::SqrtSoftplus {
            normalize_selected: dsv4.norm_topk_prob,
            scaling_factor: dsv4.routed_scaling_factor,
            selection_bias: true,
        };
    }
    if let Some((scaling_factor, normalize_selected)) = cfg.sigmoid_router() {
        let selection_bias = cfg.m3.as_ref().is_some_and(|m3| m3.use_routing_bias)
            || cfg.hy3.as_ref().is_some_and(|hy3| hy3.use_routing_bias)
            || cfg.mla.is_some()
            || cfg.step35.is_some();
        return RouterPlan::Sigmoid {
            normalize_selected,
            scaling_factor,
            selection_bias,
        };
    }
    RouterPlan::Softmax
}

fn activation(cfg: &ModelConfig, index: u32, routed: bool) -> ActivationPlan {
    if let Some(g5) = cfg.glm5.as_ref() {
        // Same pre-activation clamp on dense, routed, and shared MLPs (reference:
        // Glm5NextTextMLP and Glm5NextTextExperts share the clamp shape).
        return ActivationPlan::SwiGluPreClamped {
            limit: g5.swiglu_limit,
        };
    }
    if let Some(m3) = cfg.m3.as_ref() {
        return ActivationPlan::SwiGluOai {
            alpha: m3.swiglu_alpha,
            limit: m3.swiglu_limit,
        };
    }
    if let Some(hy3) = cfg.hy3.as_ref() {
        return match hy3.hidden_act.as_str() {
            "silu" | "swiglu" => ActivationPlan::Silu,
            other => ActivationPlan::Named(other.to_string()),
        };
    }
    if let Some(step) = cfg.step35.as_ref() {
        let limit = if routed {
            step.clamp_exp(index)
        } else {
            step.clamp_shexp(index)
        };
        if let Some(limit) = limit {
            return ActivationPlan::SwiGluClamped { limit };
        }
    }
    if cfg.gemma4.is_some() {
        return ActivationPlan::GeluTanh;
    }
    ActivationPlan::Silu
}

fn residual_topology(cfg: &ModelConfig) -> ResidualTopology {
    if let Some(dsv4) = cfg.dsv4.as_ref() {
        return ResidualTopology::HyperConnections {
            streams: dsv4.hc_mult,
            epsilon: dsv4.hc_eps,
            sinkhorn_iterations: dsv4.hc_sinkhorn_iters,
            collapse: HcCollapse::GatedHead,
        };
    }
    if let Some(g5) = cfg.glm5.as_ref() {
        return ResidualTopology::HyperConnections {
            streams: g5.hc_mult,
            epsilon: g5.hc_eps,
            sinkhorn_iterations: g5.hc_sinkhorn_iters,
            collapse: HcCollapse::Mean,
        };
    }
    if cfg.gemma4.is_some() {
        let norm = NormPlan {
            kind: NormKind::Rms,
            epsilon: cfg.rms_eps,
            weight_transform: WeightTransform::Identity,
        };
        return ResidualTopology::Gemma {
            post_attention_norm: norm,
            post_mlp_norm: norm,
            layer_scale: GemmaLayerScale::Learned,
            parallel_moe: cfg
                .moe
                .as_ref()
                .is_some_and(|moe| moe.expert_count > 0)
                .then_some(GemmaParallelMoePlan {
                    shared_post_norm: norm,
                    routed_pre_norm: norm,
                    routed_post_norm: norm,
                    router_input_scale: true,
                    per_expert_output_scale: true,
                }),
        };
    }
    ResidualTopology::Serial
}

fn norm_weight_transform(cfg: &ModelConfig) -> WeightTransform {
    if matches!(cfg.arch, Arch::Qwen35 | Arch::Qwen35Moe)
        || cfg.m3.as_ref().is_some_and(|m3| m3.use_gemma_norm)
    {
        WeightTransform::AddOne
    } else {
        WeightTransform::Identity
    }
}

fn qk_norm_presence(cfg: &ModelConfig) -> TensorPresence {
    if cfg.geometry.is_some()
        || cfg.gemma4.is_some()
        || cfg.mla.is_some()
        || cfg.dsv4.is_some()
        || cfg.m3.is_some()
        || cfg.hy3.is_some()
        || cfg.step35.is_some()
    {
        TensorPresence::Required
    } else if matches!(cfg.arch, Arch::Llama | Arch::Other(_)) {
        TensorPresence::Absent
    } else {
        TensorPresence::Optional
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationKind {
    Embedding,
    VisionPatchEmbedding,
    VisionBidirectionalAttention,
    VisionMlp,
    VisionStandardize,
    /// glm5_next: post-encoder norm + conv `merge x merge` spatial downsample into the
    /// merger width (the step between the ViT blocks and the gated merger).
    VisionDownsample,
    VisionProjection,
    VisionTokenInjection,
    RmsNorm,
    FullAttention,
    SlidingWindowAttention,
    LatentMlaAttention,
    CompressedMlaAttention,
    KvCompressor,
    SparseIndex,
    SharedSparseIndex,
    GatedDeltaNet,
    KimiDeltaNet,
    FusedAttentionGate,
    SeparateAttentionGate,
    DenseMlp,
    MoeMlp,
    SoftmaxRouter,
    SigmoidRouter,
    SqrtSoftplusRouter,
    TokenHashRouter,
    SharedMlp,
    SiluActivation,
    GeluTanhActivation,
    SwiGluOaiActivation,
    SwiGluClampedActivation,
    SwiGluPreClampedActivation,
    NamedActivation,
    SerialResidual,
    GemmaResidual,
    GemmaParallelMoeResidual,
    HyperConnections,
    KvState,
    SlidingKvState,
    RecurrentState,
    LatentKvState,
    CompressedAttentionState,
    Mtp,
    DraftPlan,
    MtpFusion,
    MtpHead,
    DsparkFusion,
    DsparkMarkovHead,
    DsparkConfidenceHead,
    PipelineBoundary,
    LogitsSoftcap,
    LogitsMask,
    OutputProjection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationSupport {
    pub batch: bool,
    pub spec_draft: bool,
    pub spec_verify: bool,
    pub pipeline: bool,
    pub cuda_graph: bool,
}

impl OperationSupport {
    pub const fn none() -> Self {
        Self {
            batch: false,
            spec_draft: false,
            spec_verify: false,
            pipeline: false,
            cuda_graph: false,
        }
    }

    pub const fn all() -> Self {
        Self {
            batch: true,
            spec_draft: true,
            spec_verify: true,
            pipeline: true,
            cuda_graph: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanCapabilities {
    pub batch: CapabilityStatus,
    pub speculative: CapabilityStatus,
    pub pipeline: CapabilityStatus,
    pub cuda_graph: CapabilityStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityStatus {
    pub supported: bool,
    pub blockers: Vec<OperationKind>,
}

impl CapabilityStatus {
    fn supported() -> Self {
        Self {
            supported: true,
            blockers: Vec::new(),
        }
    }

    fn require(&mut self, operation: OperationKind, implemented: bool) {
        if !implemented {
            self.supported = false;
            if !self.blockers.contains(&operation) {
                self.blockers.push(operation);
            }
        }
    }

    fn and(mut self, other: Self) -> Self {
        self.supported &= other.supported;
        for blocker in other.blockers {
            if !self.blockers.contains(&blocker) {
                self.blockers.push(blocker);
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HfConfig;

    fn config(json: &str) -> ModelConfig {
        ModelConfig::from_hf(&HfConfig::parse(json))
    }

    #[test]
    fn compiles_dense_hf_config_into_typed_layers() {
        let cfg = config(
            r#"{"model_type":"qwen3","num_hidden_layers":2,"hidden_size":256,
            "num_attention_heads":8,"num_key_value_heads":2,"head_dim":32,
            "intermediate_size":512,"vocab_size":1024,"max_position_embeddings":2048}"#,
        );
        let plan = ModelPlan::compile(&cfg).unwrap();

        assert_eq!(plan.layers.len(), 2);
        assert!(plan.mtp_blocks.is_empty());
        assert!(matches!(plan.layers[0].attention, AttentionPlan::Full(_)));
        assert!(matches!(plan.layers[0].mlp, MlpPlan::Dense(_)));
        assert_eq!(plan.layers[0].residual, ResidualTopology::Serial);
        assert_eq!(plan.partition_boundaries, vec![1]);
    }

    #[test]
    fn compiles_gdn_full_attention_and_mtp_from_one_config() {
        let cfg = config(
            r#"{"model_type":"qwen3_5","num_hidden_layers":4,
            "num_nextn_predict_layers":1,"hidden_size":4096,"num_attention_heads":32,
            "num_key_value_heads":8,"head_dim":256,"intermediate_size":12288,
            "vocab_size":151936,"max_position_embeddings":262144,"rms_norm_eps":0.000001,
            "rope_theta":5000000,"partial_rotary_factor":0.25,"full_attention_interval":4,
            "linear_conv_kernel_dim":4,"linear_key_head_dim":128,
            "linear_value_head_dim":128,"linear_num_key_heads":16,
            "linear_num_value_heads":32}"#,
        );
        let plan = ModelPlan::compile(&cfg).unwrap();

        assert_eq!(plan.layers.len(), 4);
        assert!(matches!(
            plan.layers[0].attention,
            AttentionPlan::GatedDeltaNet(_)
        ));
        assert_eq!(
            plan.layers[0].state,
            StatePlan::Recurrent {
                conv_width: 8192,
                conv_kernel: 4,
                state_width: 524_288,
            }
        );
        let AttentionPlan::Full(full) = &plan.layers[3].attention else {
            panic!("periodic layer must be full attention");
        };
        assert_eq!(full.rope.dimensions, 64);
        assert_eq!(full.output_gate, AttentionGateKind::FusedQ);
        assert_eq!(plan.mtp_blocks.len(), 1);
        assert!(matches!(
            plan.mtp_blocks[0].layer.attention,
            AttentionPlan::Full(_)
        ));
        assert_eq!(plan.output_norm.weight_transform, WeightTransform::AddOne);
    }

    #[test]
    fn external_draft_attachment_updates_the_canonical_spec_subplan() {
        let mut trunk = ModelPlan::compile(&config(
            r#"{"model_type":"qwen3","num_hidden_layers":2,"hidden_size":8,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32}"#,
        ))
        .unwrap();
        assert!(trunk.draft_operations().is_none());
        let draft = ModelPlan::compile(&config(
            r#"{"model_type":"qwen3_5","num_hidden_layers":2,
            "num_nextn_predict_layers":2,"hidden_size":8,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32,
            "full_attention_interval":2,"linear_conv_kernel_dim":3,
            "linear_key_head_dim":4,"linear_value_head_dim":4,
            "linear_num_key_heads":1,"linear_num_value_heads":2}"#,
        ))
        .unwrap();
        trunk.attach_external_draft(&draft).unwrap();
        assert_eq!(trunk.mtp_blocks.len(), 2);
        assert!(trunk.draft_operations().is_some());

        let mut wrong = draft.clone();
        wrong.hidden_size += 1;
        assert!(matches!(
            trunk.attach_external_draft(&wrong),
            Err(PlanCompileError::ExternalDraftMismatch {
                field: "hidden_size"
            })
        ));
    }

    #[test]
    fn standalone_step_draft_metadata_attaches_three_typed_swa_blocks() {
        let root = std::env::temp_dir().join(format!(
            "memra-model-plan-step-draft-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let trunk_path = root.join("trunk.gguf");
        let draft_path = root.join("draft.gguf");
        crate::micro_gguf::write_step35_meta_only(&trunk_path).unwrap();
        crate::micro_gguf::write_step35_mtp_meta_only(&draft_path).unwrap();
        let trunk_cfg = ModelConfig::from_gguf(&crate::GgufFile::open(&trunk_path).unwrap());
        let draft_cfg = ModelConfig::from_gguf(&crate::GgufFile::open(&draft_path).unwrap());
        let mut trunk = ModelPlan::compile(&trunk_cfg).unwrap();
        let draft = ModelPlan::compile(&draft_cfg).unwrap();
        assert!(trunk.mtp_blocks.is_empty());
        assert_eq!(draft.mtp_blocks.len(), 3);
        trunk.attach_external_draft(&draft).unwrap();
        assert_eq!(trunk.mtp_blocks.len(), 3);
        assert!(trunk.mtp_blocks.iter().all(|block| matches!(
            block.layer.attention,
            AttentionPlan::SlidingWindow { window: 512, .. }
        )));
        assert_eq!(
            trunk.mtp_blocks[0].layer.attention,
            draft.mtp_blocks[0].layer.attention
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn compiles_moe_and_shared_mlp_without_architecture_capability_checks() {
        let cfg = config(
            r#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":2048,
            "num_attention_heads":16,"num_key_value_heads":4,"intermediate_size":6144,
            "vocab_size":151936,"max_position_embeddings":40960,"num_experts":128,
            "num_experts_per_tok":8,"moe_intermediate_size":768,
            "shared_expert_intermediate_size":512}"#,
        );
        let plan = ModelPlan::compile(&cfg).unwrap();
        let MlpPlan::Moe(moe) = &plan.layers[0].mlp else {
            panic!("expected MoE layer");
        };
        assert_eq!(moe.expert_count, 128);
        assert_eq!(moe.experts_per_token, 8);
        assert_eq!(moe.shared.as_ref().unwrap().intermediate_size, 512);
    }

    #[test]
    fn compiles_gemma_swa_and_logits_transforms() {
        let cfg = config(
            r#"{"model_type":"gemma4","num_hidden_layers":2,"hidden_size":2816,
            "num_attention_heads":8,"num_key_value_heads":4,"head_dim":256,
            "global_head_dim":512,"intermediate_size":11264,"vocab_size":262144,
            "max_position_embeddings":131072,"rms_norm_eps":0.000001,
            "sliding_window":1024,"final_logit_softcapping":30,
            "layer_types":["sliding_attention","full_attention"],
            "rope_parameters":{"full_attention":{"rope_theta":1000000,
            "partial_rotary_factor":0.25},"sliding_attention":{"rope_theta":10000}}}"#,
        );
        assert!(
            cfg.moe.is_none(),
            "dense Gemma must not manufacture a zero-expert config"
        );
        let plan = ModelPlan::compile(&cfg).unwrap();

        assert!(matches!(
            plan.layers[0].attention,
            AttentionPlan::SlidingWindow { window: 1024, .. }
        ));
        assert!(matches!(
            plan.layers[1].attention,
            AttentionPlan::Full(FullAttentionPlan {
                value_projection: ValueProjection::ReuseKey,
                value_norm: ValueNorm::WeightlessRms,
                scale: AttentionScale::Fixed(1.0),
                ..
            })
        ));
        assert_eq!(plan.logits, vec![LogitsTransform::Softcap(30.0)]);
        assert!(matches!(
            plan.layers[0].residual,
            ResidualTopology::Gemma { .. }
        ));
    }

    #[test]
    fn capabilities_are_intersection_of_required_operations() {
        let cfg = config(
            r#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":256,
            "num_attention_heads":8,"num_key_value_heads":2,"head_dim":32,
            "intermediate_size":512,"vocab_size":1024,"max_position_embeddings":2048,
            "num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":128}"#,
        );
        let plan = ModelPlan::compile(&cfg).unwrap();
        let capabilities = plan.derive_capabilities(|operation| {
            let mut support = OperationSupport::all();
            if operation == OperationKind::MoeMlp {
                support.batch = false;
                support.cuda_graph = false;
            }
            support
        });

        assert!(!capabilities.batch.supported);
        assert_eq!(capabilities.batch.blockers, vec![OperationKind::MoeMlp]);
        assert!(!capabilities.speculative.supported);
        assert_eq!(
            capabilities.speculative.blockers,
            vec![OperationKind::DraftPlan]
        );
        assert!(capabilities.pipeline.supported);
        assert!(!capabilities.cuda_graph.supported);

        let no_boundary = config(
            r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":64,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
            "intermediate_size":128,"vocab_size":16,"max_position_embeddings":128}"#,
        );
        let no_boundary = ModelPlan::compile(&no_boundary)
            .unwrap()
            .derive_capabilities(|_| OperationSupport::all());
        assert_eq!(
            no_boundary.pipeline.blockers,
            vec![OperationKind::PipelineBoundary]
        );

        let boundary_disabled = plan.derive_capabilities(|operation| {
            let mut support = OperationSupport::all();
            if operation == OperationKind::PipelineBoundary {
                support.pipeline = false;
            }
            support
        });
        assert_eq!(
            boundary_disabled.pipeline.blockers,
            vec![OperationKind::PipelineBoundary]
        );
    }

    #[test]
    fn invalid_gdn_metadata_fails_during_compilation() {
        let cfg = config(
            r#"{"model_type":"qwen3_5","num_hidden_layers":2,"hidden_size":256,
            "num_attention_heads":8,"num_key_value_heads":2,"head_dim":32,
            "intermediate_size":512,"vocab_size":1024,"max_position_embeddings":2048,
            "full_attention_interval":2}"#,
        );
        assert_eq!(
            ModelPlan::compile(&cfg),
            Err(PlanCompileError::MissingGatedDeltaNetConfig { layer: 0 })
        );
    }

    #[test]
    fn executor_selection_is_derived_from_plan_operations() {
        let dense = config(
            r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":64,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
            "intermediate_size":128,"vocab_size":16,"max_position_embeddings":128}"#,
        );
        assert!(!dense.uses_hybrid_executor());

        let moe = config(
            r#"{"model_type":"qwen3_moe","num_hidden_layers":1,"hidden_size":64,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
            "intermediate_size":128,"vocab_size":16,"max_position_embeddings":128,
            "num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":32}"#,
        );
        assert!(moe.uses_hybrid_executor());

        let gated = config(
            r#"{"model_type":"qwen3_5","num_hidden_layers":2,"hidden_size":64,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
            "intermediate_size":128,"vocab_size":16,"max_position_embeddings":128,
            "full_attention_interval":2,"linear_conv_kernel_dim":3,
            "linear_key_head_dim":32,"linear_value_head_dim":32,
            "linear_num_key_heads":1,"linear_num_value_heads":2}"#,
        );
        assert!(gated.uses_hybrid_executor());
    }
}
