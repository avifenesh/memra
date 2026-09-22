//! Separate capture class for a complete, post-barrier target-head trim proof.
use super::*;
use crate::hybrid::root_trim::CompleteTargetTrimProof;

pub(crate) fn capture_target_trim_rewrite_identity(
    model: &HybridModel,
    source: &dyn memra_gguf::source::TensorSource,
    target_artifact_sha256: String,
    proof: CompleteTargetTrimProof,
) -> Result<(RewriteIdentity, RewriteLoadState), Box<dyn std::error::Error>> {
    proof.validate(model, source, &target_artifact_sha256)?;
    let artifact_sha256 = proof.artifact_sha256().to_owned();
    let state = RewriteLoadState {
        mutation_generation: model.rewrite_mutations(),
        pipeline: model.is_multi_device(),
        model_sha256: loaded_model_sha256(model),
        environment: numeric_environment(std::env::vars_os()),
        libraries: Some(LoadedLibraries::capture()?),
    };
    let metadata = source.runtime_metadata()?;
    let interpretation = format!(
        "gguf={} safetensors={} activation={:?} preserve_experts={} nvfp4={} loaded_libraries={} complete_target_trim={} selected_programs={}",
        metadata.is_gguf,
        metadata.is_safetensors,
        metadata.expert_activation_precision,
        metadata.preserve_expert_encodings,
        metadata.nvfp4_cache_tag,
        state.libraries.as_ref().unwrap().sha256,
        artifact_sha256,
        proof.program_descriptor(),
    );
    let identity = RewriteIdentity {
        artifact_sha256,
        implementation_sha256: running_implementation_sha256()?,
        numeric_program_sha256: numeric_program_sha256(
            true,
            &interpretation,
            &hardware_snapshot(&model.devices())?,
            &state.model_sha256,
            &state.environment,
        ),
    };
    identity.validate()?;
    proof.validate(model, source, &target_artifact_sha256)?;
    state.validate(model)?;
    Ok((identity, state))
}
