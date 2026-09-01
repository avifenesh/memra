use super::*;
use crate::config::HfConfig;

pub static PACK: ModelPack = ModelPack {
    family: "qwen3_moe",
    aliases: &["qwen3moe"],
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
    matches_config: |config| {
        matches!(config.arch, Arch::Qwen3Moe)
            || (matches!(config.arch, Arch::Qwen3)
                && config.moe.as_ref().is_some_and(|moe| moe.expert_count > 0))
    },
    plan_builder: canonical_plan,
    tensor_schema: canonical_tensor_schema,
    tiny_plan: Some(tiny_plan),
};

fn tiny_plan() -> Result<ModelPlan, PlanCompileError> {
    canonical_plan(&ModelConfig::from_hf(&HfConfig::parse(
        r#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":8,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
        "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32,
        "num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":8,
        "shared_expert_intermediate_size":8}"#,
    )))
}
