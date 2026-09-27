use super::*;
use crate::model_plan::{AttentionPlan, DraftSourcePlan, RopeFactors, SamplingDefaultsPlan};
mod inventory;
pub(crate) mod tensors;

pub static PACK: ModelPack = ModelPack {
    inventory_schema: Some(inventory::compile),
    default_output_head: crate::tensor_contract::OutputHead::Separate,
    family: "step35",
    output_head: OutputHeadContract::SeparateHead,
    tensor_consumption: TensorConsumption::Report,
    aliases: &["step35", "step37", "step-3.7-flash"],
    config_layout: ConfigLayout::Flat,
    tokenizer_sources: &[
        TokenizerSource::GgufMetadata,
        TokenizerSource::TokenizerJson,
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
    matches_config: |config| config.step35.is_some(),
    plan_builder,
    tensor_schema,
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

fn checkpoint_factor_width(plan: &ModelPlan) -> Option<u32> {
    plan.layers
        .iter()
        .chain(plan.mtp_blocks.iter().map(|block| &block.layer))
        .filter_map(|layer| match &layer.attention {
            AttentionPlan::Full(attention) | AttentionPlan::SlidingWindow { attention, .. }
                if matches!(attention.rope.factors, RopeFactors::Checkpoint) =>
            {
                Some(attention.key_head_dim / 2)
            }
            _ => None,
        })
        .max()
}

#[allow(clippy::result_large_err)] // allow: preserve the shared tensor-contract diagnostic type
fn tensor_schema(
    config: &ModelConfig,
    plan: &ModelPlan,
    dialect: CheckpointDialect,
    options: ContractOptions,
) -> Result<TensorContract, TensorContractError> {
    let mut contract = canonical_tensor_schema(config, plan, dialect, options)?;
    if dialect == CheckpointDialect::Gguf
        && let Some(width) = checkpoint_factor_width(plan)
        && width >= crate::tensor_contract::rope_factor_width(plan).unwrap_or(0)
        && config
            .step35
            .as_ref()
            .and_then(|step| step.rope_freq_shape.as_deref())
            .is_some_and(|shape| shape == [width as u64])
        && let Some(factors) = contract
            .requirements
            .iter_mut()
            .find(|tensor| tensor.id == crate::tensor_contract::TensorId::RopeFactors)
    {
        // Preserve full-head storage only when the source actually declares it.
        // Compact sources keep the canonical n_rot/2 contract. Never copy arbitrary
        // header shapes into the requirement: malformed extents/ranks must fail bind.
        factors.shape = vec![width as u64];
    }
    Ok(contract)
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
    let stored_width = checkpoint_factor_width(plan).unwrap_or(0) as usize;
    let valid_width = |len: usize| len >= width && (len == width || len == stored_width);
    let factors = match source.find("rope_freqs.weight") {
        Some(tensor) => {
            let element_bytes = match tensor.ggml_type {
                crate::GgmlType::F32 => 4,
                crate::GgmlType::F16 | crate::GgmlType::BF16 => 2,
                _ => return Err(invalid("expected floating-point factors".into())),
            };
            let tensor_width = tensor.ne.first().copied().unwrap_or(0) as usize;
            if tensor.ne.len() != 1
                || !valid_width(tensor_width)
                || tensor.bytes.len() != tensor_width * element_bytes
            {
                return Err(invalid(format!(
                    "expected [{width}] or full-head [{stored_width}] factors; got {:?} ({} bytes)",
                    tensor.ne,
                    tensor.bytes.len()
                )));
            }
            step.rope_freq_shape = Some(tensor.ne.clone());
            Some(crate::dequant::dequantize(
                tensor.ggml_type,
                &tensor.bytes,
                tensor_width,
            ))
        }
        None => step.rope_freq_factors.clone(),
    };
    if let Some(factors) = &factors {
        if !valid_width(factors.len())
            || factors
                .iter()
                .any(|factor| !factor.is_finite() || *factor <= 0.0)
        {
            return Err(invalid(format!(
                "expected {width} or {stored_width} finite positive factors"
            )));
        }
    } else if width > 0 {
        return Err(invalid(
            "compiled plan requires rope_freqs.weight or normalized frequency factors".into(),
        ));
    }
    step.rope_freq_factors = factors;
    Ok(())
}
