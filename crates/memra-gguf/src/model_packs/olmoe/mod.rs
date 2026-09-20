use super::*;

/// Explicit ownership of the existing OLMoE program, formerly reached through the loader's
/// generic fallback. Registration preserves that program; it does not promote qualification.
pub static PACK: ModelPack = ModelPack {
    family: "olmoe",
    aliases: &["olmoe"],
    config_layout: ConfigLayout::Flat,
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
    matches_config: |config| {
        matches!(config.arch, Arch::Olmoe)
            && config.moe.as_ref().is_some_and(|moe| moe.expert_count > 0)
    },
    plan_builder: canonical_plan,
    tensor_schema: canonical_tensor_schema,
    tiny_plan: None,
};
