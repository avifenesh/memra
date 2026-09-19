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

/// Validate the small RoPE auxiliary before any model weights are allocated. Keep the
/// validated values in the normalized config so the loader uploads exactly this buffer.
pub(super) fn prepare_rope_factors(
    config: &mut ModelConfig,
    plan: &ModelPlan,
    source: &dyn crate::source::TensorSource,
) -> Result<(), PlanCompileError> {
    let Some(step) = config.step35.as_mut() else {
        return Ok(());
    };
    let invalid = |value: String| PlanCompileError::UnsupportedSemantics {
        field: "rope_freqs.weight",
        value,
    };
    let width = crate::tensor_contract::rope_factor_width(plan).unwrap_or(0) as usize;
    let factors = match source.find("rope_freqs.weight") {
        Some(tensor) => {
            let element_bytes = match tensor.ggml_type {
                crate::GgmlType::F32 => 4,
                crate::GgmlType::F16 | crate::GgmlType::BF16 => 2,
                _ => return Err(invalid("expected floating-point factors".into())),
            };
            if tensor.ne != [width as u64] || tensor.bytes.len() != width * element_bytes {
                return Err(invalid(format!(
                    "expected [{width}] factors; got {:?} ({} bytes)",
                    tensor.ne,
                    tensor.bytes.len()
                )));
            }
            Some(crate::dequant::dequantize(
                tensor.ggml_type,
                &tensor.bytes,
                width,
            ))
        }
        None => step.rope_freq_factors.clone(),
    };
    if let Some(factors) = &factors {
        if factors.len() != width
            || factors
                .iter()
                .any(|factor| !factor.is_finite() || *factor <= 0.0)
        {
            return Err(invalid(format!("expected {width} finite positive factors")));
        }
    } else if config.rope_scaling_hint.as_deref() == Some("llama3") {
        return Err(invalid(
            "llama3 requires rope_freqs.weight or normalized frequency factors".into(),
        ));
    }
    step.rope_freq_factors = factors;
    Ok(())
}
