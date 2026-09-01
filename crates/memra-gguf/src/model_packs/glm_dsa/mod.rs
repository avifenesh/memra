use super::*;
use crate::model_plan::{
    ActivationPlan, AttentionPlan, DenseMlpPlan, LayerPlan, MlaAttentionPlan, MlpPlan, MoeMlpPlan,
    NormKind, NormPlan, ResidualTopology, RopeFactors, RopePlan, RouterPlan, SharedMlpPlan,
    SparseIndexPlan, StatePlan, WeightTransform,
};

pub static PACK: ModelPack = ModelPack {
    family: "glm_dsa",
    aliases: &["glm_moe_dsa", "glm-dsa"],
    config_layout: ConfigLayout::Flat,
    tokenizer_sources: &[
        TokenizerSource::TokenizerJson,
        TokenizerSource::GgufMetadata,
    ],
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
    matches_config: |config| matches!(config.arch, Arch::GlmDsa),
    plan_builder: canonical_plan,
    tensor_schema: canonical_tensor_schema,
    tiny_plan: Some(tiny_plan),
};

fn tiny_plan() -> Result<ModelPlan, PlanCompileError> {
    let norm = NormPlan {
        kind: NormKind::Rms,
        epsilon: 1e-6,
        weight_transform: WeightTransform::Identity,
    };
    let attention = |sparse_index| {
        AttentionPlan::Mla(MlaAttentionPlan::LatentKv {
            query_heads: 2,
            q_lora_rank: 4,
            kv_lora_rank: 4,
            qk_head_dim: 4,
            rope_head_dim: 2,
            value_head_dim: 4,
            rope: RopePlan {
                dimensions: 2,
                base: 10_000.0,
                factors: RopeFactors::None,
            },
            sparse_index,
        })
    };
    let layer = |index, attention, mlp| LayerPlan {
        index,
        pre_attention_norm: norm,
        attention,
        pre_mlp_norm: norm,
        mlp,
        residual: ResidualTopology::Serial,
        state: StatePlan::LatentKvCache {
            width: 6,
            index_width: 0,
        },
    };
    Ok(ModelPlan {
        arch: Arch::GlmDsa,
        hidden_size: 8,
        vocab_size: 32,
        context_length: 32,
        embedding_scale: 1.0,
        vision: None,
        multimodal: None,
        layers: vec![
            layer(
                0,
                attention(SparseIndexPlan::Own {
                    heads: 1,
                    head_dim: 2,
                    top_k: 8,
                    kpool: None,
                }),
                MlpPlan::Dense(DenseMlpPlan {
                    intermediate_size: 16,
                    activation: ActivationPlan::Silu,
                }),
            ),
            layer(
                1,
                attention(SparseIndexPlan::SharedFromPrevious { top_k: 8 }),
                MlpPlan::Moe(MoeMlpPlan {
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
                    activation: ActivationPlan::Silu,
                }),
            ),
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
