use super::*;
use crate::model_plan::{
    ActivationPlan, AttentionPlan, Glm5VisionPlan, HcCollapse, KimiDeltaNetPlan, KpoolPlan,
    LayerPlan, MlaAttentionPlan, MlpPlan, MoeMlpPlan, NormKind, NormPlan, ResidualTopology,
    RopeFactors, RopePlan, RouterPlan, SharedMlpPlan, SparseIndexPlan, StatePlan, VisionPlan,
    VisionTokenInjectionPlan, WeightTransform,
};

/// GLM-5.3-Flash (glm5_next): 34 KDA + 11 MLA/DSA hybrid, NoPE, k-pool indexer,
/// Sinkhorn hyper-connections with mean collapse, sigmoid noaux_tc MoE, 1 NextN layer.
/// Bring-up lane research/glm53-flash-bringup-20260827/ (census, plan design, oracle bank).
pub static PACK: ModelPack = ModelPack {
    family: "glm5_next",
    aliases: &["glm5_next_text", "glm5-next"],
    config_layout: ConfigLayout::FlatOrTextConfig,
    tokenizer_sources: &[TokenizerSource::TokenizerJson],
    template: TemplateContract::ArtifactRequired,
    support: Some(NativeSupport::NativeReference),
    gates: &[
        Gate::Config,
        Gate::TokenizerTemplate,
        Gate::TensorCensus,
        Gate::TinyParity,
        Gate::CheckpointParity,
        Gate::RewriteParity,
        Gate::Serve,
    ],
    checkpoint_parity: Some(CheckpointParityGate {
        max_abs: 0.005,
        max_rel: 0.005,
        require_argmax: true,
    }),
    matches_config: |config| matches!(config.arch, Arch::Glm5Next),
    plan_builder: canonical_plan,
    tensor_schema: glm5_tensor_schema,
    tiny_plan: Some(tiny_plan),
};

/// Two trunk layers exercising every glm5_next-specific program element: one KDA layer
/// (dense MLP, pre-clamped swiglu) and one MLA layer with a k-pool indexer (sigmoid
/// noaux_tc MoE + shared expert), both under hyper-connection residual streams with the
/// glm5_next MEAN collapse. NoPE throughout (rope dimensions 0).
fn tiny_plan() -> Result<ModelPlan, PlanCompileError> {
    let norm = NormPlan {
        kind: NormKind::Rms,
        epsilon: 1e-5,
        weight_transform: WeightTransform::Identity,
    };
    let residual = ResidualTopology::HyperConnections {
        streams: 2,
        epsilon: 1e-6,
        sinkhorn_iterations: 4,
        collapse: HcCollapse::Mean,
    };
    Ok(ModelPlan {
        // glm5_next's mHC exit is its own program; the qwen4_exp exit-mixer field stays unset.
        exit_mixer: None,
        arch: Arch::Glm5Next,
        hidden_size: 8,
        vocab_size: 32,
        context_length: 32,
        embedding_scale: 1.0,
        // Tiny twin of the glm5_next tower program (lane/glm5-vision): every element the
        // real tower exercises — fused qkv + biases, per-head q/k RMS, 2D rope, clamped
        // SwiGLU, downsample conv, gated merger, grid-derived splice with delimiters.
        vision: Some(VisionPlan::Glm5Fused(Glm5VisionPlan {
            depth: 2,
            hidden_size: 8,
            heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            patch_size: 2,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            out_hidden_size: 8,
            projection_intermediate_size: 16,
            swiglu_limit: 10.0,
            rope_theta: 10_000.0,
            norm,
            in_channels: 3,
            patch_input_width: 3 * 2 * 2 * 2,
        })),
        multimodal: Some(VisionTokenInjectionPlan {
            placeholder_token_id: 3,
            tokens_per_image: None,
            start_token_id: Some(4),
            end_token_id: Some(5),
        }),
        layers: vec![
            LayerPlan {
                ple: None,
                sparse_overlay: None,
                index: 0,
                pre_attention_norm: norm,
                attention: AttentionPlan::KimiDeltaNet(KimiDeltaNetPlan {
                    num_heads: 2,
                    head_dim: 4,
                    conv_kernel: 4,
                    gate_lower_bound: -5.0,
                }),
                pre_mlp_norm: norm,
                mlp: MlpPlan::Dense(crate::model_plan::DenseMlpPlan {
                    intermediate_size: 16,
                    activation: ActivationPlan::SwiGluPreClamped { limit: 10.0 },
                }),
                residual,
                state: StatePlan::Recurrent {
                    conv_width: 24,
                    conv_kernel: 4,
                    state_width: 32,
                },
            },
            LayerPlan {
                ple: None,
                sparse_overlay: None,
                index: 1,
                pre_attention_norm: norm,
                attention: AttentionPlan::Mla(MlaAttentionPlan::LatentKv {
                    query_heads: 2,
                    q_lora_rank: 4,
                    kv_lora_rank: 4,
                    qk_head_dim: 4,
                    rope_head_dim: 0,
                    value_head_dim: 4,
                    rope: RopePlan {
                        dimensions: 0,
                        base: 10_000.0,
                        factors: RopeFactors::None,
                    },
                    sparse_index: SparseIndexPlan::Own {
                        heads: 1,
                        head_dim: 2,
                        top_k: 8,
                        kpool: Some(KpoolPlan {
                            pool: 4,
                            always_select_tail: true,
                        }),
                    },
                }),
                pre_mlp_norm: norm,
                mlp: MlpPlan::Moe(MoeMlpPlan {
                    expert_count: 4,
                    experts_per_token: 2,
                    expert_intermediate_size: 8,
                    router: RouterPlan::Sigmoid {
                        normalize_selected: true,
                        scaling_factor: 2.5,
                        selection_bias: true,
                    },
                    shared: Some(SharedMlpPlan {
                        intermediate_size: 8,
                        gated: false,
                    }),
                    activation: ActivationPlan::SwiGluPreClamped { limit: 10.0 },
                }),
                residual,
                state: StatePlan::LatentKvCache {
                    width: 4,
                    // index head_dim 2 -> [k | gate] = 4.
                    index_width: 4,
                },
            },
        ],
        output_norm: norm,
        logits: Vec::new(),
        mtp_blocks: Vec::new(),
        drafter: None,
        draft_source: crate::model_plan::DraftSourcePlan::Embedded,
        sampling_defaults: None,
        partition_boundaries: vec![1],
    })
}

/// canonical schema + the vision tower. The generic contract has no vision section (the
/// other VL families sidecar their towers in separate artifacts); GLM ships the tower in
/// the same shards, and the census binds every physical tensor exactly once — so the
/// tower is censused here by hand from artifact truth (shard headers, banked in the lane:
/// 24 blocks, hidden 1024, fused qkv 3072 with bias, gated MLP 4096, conv patch embed
/// [1024,3,2,14,14], conv downsample [4096,1024,2,2], gated merger 4096->10240->4096).
#[allow(clippy::result_large_err)] // allow: the fat error type is the diagnostic contract here; boxing it would change the error surface
fn glm5_tensor_schema(
    config: &crate::config::ModelConfig,
    plan: &ModelPlan,
    dialect: crate::tensor_contract::CheckpointDialect,
    options: crate::tensor_contract::ContractOptions,
) -> Result<crate::tensor_contract::TensorContract, crate::tensor_contract::TensorContractError> {
    use crate::tensor_contract::{
        QuantConstraint, TensorId, TensorMatch, TensorOwner, TensorRequirement, TensorTransform,
        VisionTensor,
    };
    let mut contract = canonical_tensor_schema(config, plan, dialect, options)?;
    if dialect != crate::tensor_contract::CheckpointDialect::HfSafetensors {
        return Ok(contract);
    }
    // The tower census applies only when the plan carries the tower (a text-only
    // glm5_next config censusing visual tensors would fail every shard). The hand-census
    // constants below are ARTIFACT truth; a plan disagreeing with them is a different
    // artifact, refused rather than re-derived.
    let Some(VisionPlan::Glm5Fused(vision)) = plan.vision.as_ref() else {
        return Ok(contract);
    };
    if vision.depth != 24
        || vision.hidden_size != 1024
        || vision.intermediate_size != 4096
        || vision.head_dim != 64
        || vision.out_hidden_size != 4096
        || vision.projection_intermediate_size != 10_240
        || vision.patch_size != 14
        || vision.temporal_patch_size != 2
        || vision.spatial_merge_size != 2
    {
        return Err(
            crate::tensor_contract::TensorContractError::UnsupportedPlanOperation {
                operation: "glm5_next vision census only covers the GLM-5.3-Flash tower geometry",
            },
        );
    }
    let mut push = |layer: Option<u32>, tensor: VisionTensor, name: String, shape: Vec<u64>| {
        contract.requirements.push(TensorRequirement {
            id: TensorId::Vision { layer, tensor },
            names: vec![name],
            match_mode: TensorMatch::OneOf,
            shape,
            owner: TensorOwner::Vision(layer),
            transform: TensorTransform::Identity,
            quant: QuantConstraint::FloatOnly,
            auxiliaries: None,
            required: true,
        });
    };
    const BLOCKS: u32 = 24;
    const H: u64 = 1024;
    const FF: u64 = 4096;
    for i in 0..BLOCKS {
        let n = |s: &str| format!("model.visual.blocks.{i}.{s}");
        let l = Some(i);
        push(
            l,
            VisionTensor::FusedQkv,
            n("attn.qkv.weight"),
            vec![3 * H, H],
        );
        push(
            l,
            VisionTensor::FusedQkvBias,
            n("attn.qkv.bias"),
            vec![3 * H],
        );
        push(
            l,
            VisionTensor::AttentionOutput,
            n("attn.proj.weight"),
            vec![H, H],
        );
        push(
            l,
            VisionTensor::AttentionOutputBias,
            n("attn.proj.bias"),
            vec![H],
        );
        push(
            l,
            VisionTensor::QueryNorm,
            n("attn.q_norm.weight"),
            vec![64],
        );
        push(l, VisionTensor::KeyNorm, n("attn.k_norm.weight"), vec![64]);
        push(
            l,
            VisionTensor::MlpGate,
            n("mlp.gate_proj.weight"),
            vec![FF, H],
        );
        push(
            l,
            VisionTensor::MlpGateBias,
            n("mlp.gate_proj.bias"),
            vec![FF],
        );
        push(l, VisionTensor::MlpUp, n("mlp.up_proj.weight"), vec![FF, H]);
        push(l, VisionTensor::MlpUpBias, n("mlp.up_proj.bias"), vec![FF]);
        push(
            l,
            VisionTensor::MlpDown,
            n("mlp.down_proj.weight"),
            vec![H, FF],
        );
        push(
            l,
            VisionTensor::MlpDownBias,
            n("mlp.down_proj.bias"),
            vec![H],
        );
        push(l, VisionTensor::InputNorm, n("norm1.weight"), vec![H]);
        push(l, VisionTensor::PreMlpNorm, n("norm2.weight"), vec![H]);
    }
    let g = |s: &str| format!("model.visual.{s}");
    push(
        None,
        VisionTensor::PatchProjection,
        g("patch_embed.proj.weight"),
        vec![H, 3, 2, 14, 14],
    );
    push(
        None,
        VisionTensor::PatchProjectionBias,
        g("patch_embed.proj.bias"),
        vec![H],
    );
    push(
        None,
        VisionTensor::Downsample,
        g("downsample.weight"),
        vec![4096, H, 2, 2],
    );
    push(
        None,
        VisionTensor::DownsampleBias,
        g("downsample.bias"),
        vec![4096],
    );
    push(
        None,
        VisionTensor::MergerGate,
        g("merger.gate_proj.weight"),
        vec![10240, 4096],
    );
    push(
        None,
        VisionTensor::MergerUp,
        g("merger.up_proj.weight"),
        vec![10240, 4096],
    );
    push(
        None,
        VisionTensor::MergerDown,
        g("merger.down_proj.weight"),
        vec![4096, 10240],
    );
    push(
        None,
        VisionTensor::MergerProjection,
        g("merger.proj.weight"),
        vec![4096, 4096],
    );
    push(
        None,
        VisionTensor::MergerPostProjectionNorm,
        g("merger.post_projection_norm.weight"),
        vec![4096],
    );
    push(
        None,
        VisionTensor::MergerPostProjectionNormBias,
        g("merger.post_projection_norm.bias"),
        vec![4096],
    );
    push(
        None,
        VisionTensor::PostEncoderNorm,
        g("post_layernorm.weight"),
        vec![H],
    );
    Ok(contract)
}
