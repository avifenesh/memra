use super::*;
use crate::config::HfConfig;

pub static PACK: ModelPack = ModelPack {
    family: "qwen35",
    aliases: &[
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_next",
        "qwen35",
        "qwen3next",
    ],
    config_layout: ConfigLayout::FlatOrTextConfig,
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
        matches!(config.arch, Arch::Qwen35)
            && !config.moe.as_ref().is_some_and(|moe| moe.expert_count > 0)
    },
    plan_builder: canonical_plan,
    tensor_schema: canonical_tensor_schema,
    tiny_plan: Some(tiny_plan),
};

fn tiny_plan() -> Result<ModelPlan, PlanCompileError> {
    canonical_plan(&ModelConfig::from_hf(&HfConfig::parse(
        r#"{"model_type":"qwen3_5","num_hidden_layers":4,
        "num_nextn_predict_layers":1,"hidden_size":8,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
        "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32,
        "rms_norm_eps":0.000001,"full_attention_interval":2,
        "linear_conv_kernel_dim":3,"linear_key_head_dim":4,
        "linear_value_head_dim":4,"linear_num_key_heads":1,
        "linear_num_value_heads":2}"#,
    )))
}
