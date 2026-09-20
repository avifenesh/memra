use super::*;

/// Explicit ownership of the existing MiniMax-M3 program, formerly reached through the
/// loader's generic fallback. Native checkpoint/serving qualification remains separate.
pub static PACK: ModelPack = ModelPack {
    family: "minimax_m3",
    aliases: &["minimax-m3", "minimax_m3_vl", "minimax_m3_text"],
    config_layout: ConfigLayout::FlatOrTextConfig,
    tokenizer_sources: &[
        TokenizerSource::TokenizerJson,
        TokenizerSource::GgufMetadata,
    ],
    template: TemplateContract::ArtifactRequired,
    support: None,
    gates: &[
        Gate::Config,
        Gate::TokenizerTemplate,
        Gate::TensorCensus,
        Gate::TinyParity,
        Gate::CheckpointParity,
        Gate::RewriteParity,
        Gate::Serve,
    ],
    checkpoint_parity: None,
    matches_config: |config| matches!(config.arch, Arch::MinimaxM3) && config.m3.is_some(),
    plan_builder: canonical_plan,
    tensor_schema: canonical_tensor_schema,
    tiny_plan: None,
};
