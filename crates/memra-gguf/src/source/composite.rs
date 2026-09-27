//! Compiler-private access to selected opened components. No whole-file handles are exported.
use super::*;
use crate::bound_source::BoundTensorRequest;

pub(crate) enum OpenedComponent<'a> {
    Repack(&'a Hy3RepackSource),
    Safetensors(&'a SafetensorsSource),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ComponentInterpretation {
    Repack {
        expert_activation_precision: ExpertActivationPrecision,
        active_experts: BTreeMap<u32, Vec<bool>>,
    },
    Safetensors(BoundSourceInterpretation),
}

impl Hy3RepackSource {
    pub(crate) fn binding_components(&self) -> Vec<OpenedComponent<'_>> {
        let mut components = Vec::new();
        let mut current = self;
        loop {
            components.push(OpenedComponent::Repack(current));
            match &current.fallback {
                Some(RepackFallback::Repack(next)) => current = next,
                Some(RepackFallback::Safetensors(source)) => {
                    components.push(OpenedComponent::Safetensors(source));
                    return components;
                }
                None => return components,
            }
        }
    }
}

impl OpenedComponent<'_> {
    pub(crate) fn is_overlay(&self) -> bool {
        matches!(self,Self::Repack(source) if source.fallback.is_some())
    }
    pub(crate) fn st_dir(&self) -> Option<&Path> {
        match self {
            Self::Repack(source) => source.st_dir(),
            Self::Safetensors(source) => source.st_dir(),
        }
    }
    pub(crate) fn output_root(
        &self,
        request: &BoundTensorRequest<'_>,
    ) -> Result<Option<&crate::bound_output::RetainedOutputRoot>, String> {
        self.validate(request)?;
        match self {
            Self::Repack(_) => Ok(None),
            Self::Safetensors(source) => source.bound_output_root(request),
        }
    }
    pub(crate) fn artifact_sha256(&self) -> Result<String, String> {
        match self {
            Self::Repack(source) => source.repack_artifact_sha256(true),
            Self::Safetensors(source) => source.artifact_sha256(),
        }
    }

    pub(crate) fn read(&self, r: &BoundTensorRequest<'_>) -> Result<TensorView<'_>, String> {
        match self {
            Self::Repack(source) => {
                source.validate_own_bound_repack(r)?;
                // Validation proves this physical name belongs to this component; find cannot
                // reach its fallback. Preserve the existing BF16-vector widening program.
                source
                    .find(&r.record.physical_name)
                    .ok_or_else(|| r.error("own repack payload missing"))
            }
            Self::Safetensors(source) => source.read_bound(r),
        }
    }

    pub(crate) fn validate_values(&self, r: &BoundTensorRequest<'_>) -> Result<(), String> {
        self.validate(r)?;
        match self {
            Self::Safetensors(source) => source.validate_bound_values(r),
            Self::Repack(_)
                if matches!(r.id, crate::tensor_contract::TensorId::QuantAux { .. }) =>
            {
                let tensor = self.read(r)?;
                let values = crate::dequant::dequantize(
                    tensor.ggml_type,
                    &tensor.bytes,
                    tensor.ne.iter().product::<u64>() as usize,
                );
                if values.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                    return Err(r.error("independent scale must be finite and positive"));
                }
                Ok(())
            }
            Self::Repack(_) => Ok(()),
        }
    }

    pub(crate) fn auxiliary(
        &self,
        r: &BoundTensorRequest<'_>,
        kind: crate::tensor_contract::QuantAuxTensor,
    ) -> Result<Option<TensorView<'_>>, String> {
        self.validate_values(r)?;
        match self {
            Self::Repack(_) => Ok(None),
            Self::Safetensors(source) => source.auxiliary_bound(r, kind),
        }
    }
    pub(crate) fn fp8(&self, r: &BoundTensorRequest<'_>) -> Result<Option<Fp8Native<'_>>, String> {
        self.validate_values(r)?;
        match self {
            Self::Repack(_) => Ok(None),
            Self::Safetensors(source) => source.fp8_bound(r),
        }
    }
    pub(crate) fn nvfp4(
        &self,
        r: &BoundTensorRequest<'_>,
    ) -> Result<Option<Nvfp4Native<'_>>, String> {
        self.validate_values(r)?;
        match self {
            Self::Repack(_) => Ok(None),
            Self::Safetensors(source) => source.nvfp4_bound(r),
        }
    }
    pub(crate) fn fp8_stacked(
        &self,
        r: &BoundTensorRequest<'_>,
    ) -> Result<Option<Fp8StackedNative<'_>>, String> {
        self.validate_values(r)?;
        match self {
            Self::Repack(_) => Ok(None),
            Self::Safetensors(source) => source.fp8_stacked_bound(r),
        }
    }
    pub(crate) fn nvfp4_stacked(
        &self,
        r: &BoundTensorRequest<'_>,
    ) -> Result<Option<Nvfp4StackedNative<'_>>, String> {
        self.validate_values(r)?;
        match self {
            Self::Repack(_) => Ok(None),
            Self::Safetensors(source) => source.nvfp4_stacked_bound(r),
        }
    }
    pub(crate) fn disk(&self, r: &BoundTensorRequest<'_>) -> Result<Option<DiskExtent>, String> {
        self.validate_values(r)?;
        match self {
            Self::Repack(source) => Ok(source.find_expert_disk(&r.record.physical_name)),
            Self::Safetensors(source) => source.disk_bound(r),
        }
    }

    pub(crate) fn config(&self) -> ModelConfig {
        match self {
            Self::Repack(source) => source.cfg.clone(),
            Self::Safetensors(source) => source.cfg.clone(),
        }
    }

    pub(crate) fn raw_config(&self) -> Option<&str> {
        match self {
            Self::Repack(source) => source.raw_config.as_deref(),
            Self::Safetensors(source) => source.raw_config_json(),
        }
    }

    pub(crate) fn census(&self) -> Result<TensorCensus, String> {
        match self {
            Self::Repack(source) => source.own_tensor_census(),
            Self::Safetensors(source) => source.tensor_census(),
        }
    }

    pub(crate) fn interpretation(&self) -> Result<ComponentInterpretation, String> {
        match self {
            Self::Repack(source) => Ok(ComponentInterpretation::Repack {
                expert_activation_precision: source.expert_activation_precision,
                active_experts: source.active_experts.clone(),
            }),
            Self::Safetensors(source) => source
                .bound_interpretation()
                .map(ComponentInterpretation::Safetensors),
        }
    }

    pub(crate) fn validate(&self, request: &BoundTensorRequest<'_>) -> Result<(), String> {
        match self {
            Self::Repack(source) => source.validate_own_bound_repack(request),
            Self::Safetensors(source) => source.validate_bound_metadata(request),
        }
    }
}
