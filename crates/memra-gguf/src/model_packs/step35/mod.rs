use super::*;
use crate::model_plan::{DraftSourcePlan, SamplingDefaultsPlan};

pub static PACK: ModelPack = ModelPack {
    family: "step35",
    aliases: &["step35", "step37", "step-3.7-flash"],
    config_layout: ConfigLayout::Flat,
    tokenizer_sources: &[TokenizerSource::GgufMetadata],
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
    matches_config: |config| config.step35.is_some(),
    plan_builder,
    tensor_schema: canonical_tensor_schema,
    tiny_plan: None,
};

fn plan_builder(config: &ModelConfig) -> Result<ModelPlan, PlanCompileError> {
    let mut plan = canonical_plan(config)?;
    if plan.mtp_blocks.is_empty() {
        plan.draft_source = DraftSourcePlan::ExternalArtifact;
    }
    plan.sampling_defaults = Some(SamplingDefaultsPlan {
        temperature: 0.5,
        top_p: 0.9,
    });
    Ok(plan)
}
