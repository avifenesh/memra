//! CPU-visible contracts shared by source preparation and the existing native consumers.
use crate::config::{ModelConfig, SwigluClamp};
use crate::model_plan::ActivationPlan;
use crate::source::TensorView;
use crate::tensor_contract::{FloatType, StorageLayout, TensorCensusEntry};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FfnActivation {
    Silu,
    SwiGluOai { alpha: f32, limit: f32 },
    PostClamped { limit: f32 },
    PreClamped { limit: f32 },
}

impl FfnActivation {
    /// The order is the executor's: M3 owns alpha/limit; otherwise an explicit per-layer
    /// clamp owns its form and limit; otherwise the kernel executes ordinary SiLU*up.
    pub fn for_target(config: &ModelConfig, clamp: Option<SwigluClamp>) -> Self {
        if let Some(m3) = config.m3.as_ref() {
            return Self::SwiGluOai {
                alpha: m3.swiglu_alpha,
                limit: m3.swiglu_limit,
            };
        }
        match clamp {
            Some(SwigluClamp::Post(limit)) => Self::PostClamped { limit },
            Some(SwigluClamp::Pre(limit)) => Self::PreClamped { limit },
            None => Self::Silu,
        }
    }

    pub fn validate_declared(self, declared: &ActivationPlan) -> Result<(), String> {
        let executed = match self {
            Self::Silu => ActivationPlan::Silu,
            Self::SwiGluOai { alpha, limit } => ActivationPlan::SwiGluOai { alpha, limit },
            Self::PostClamped { limit } => ActivationPlan::SwiGluClamped { limit },
            Self::PreClamped { limit } => ActivationPlan::SwiGluPreClamped { limit },
        };
        if &executed == declared {
            Ok(())
        } else {
            Err(format!(
                "external draft FFN declares {declared:?}, but target-driven execution selects {executed:?}"
            ))
        }
    }
}

/// Step's dense MTP consumer carries the source block's resolved SHEXP limit, not the
/// target's last-layer clamp. Shared with Step35MtpGeom::from_plan.
pub fn step_mtp_clamp(activation: &ActivationPlan) -> Option<f32> {
    match activation {
        ActivationPlan::SwiGluClamped { limit } if *limit > 0.0 => Some(*limit),
        _ => None,
    }
}

pub(crate) fn validate_nvfp4_macro_metadata(entry: &TensorCensusEntry) -> Result<(), String> {
    if entry.storage != StorageLayout::Float(FloatType::F32)
        || entry.shape != [1]
        || entry.physical_bytes != 4
    {
        return Err(format!(
            "{}: selected NVFP4 macro scale requires an F32 scalar [1] with exactly four bytes",
            entry.name
        ));
    }
    Ok(())
}

/// The resident consumer accepts one F32 value, including the existing scalar HF shape.
/// It never reinterprets F16/BF16 bytes or silently normalizes a different encoding.
pub fn read_nvfp4_macro_scale(view: &TensorView<'_>) -> Result<f32, String> {
    if view.ggml_type != crate::GgmlType::F32
        || view.ne.iter().try_fold(1u64, |n, &d| n.checked_mul(d)) != Some(1)
    {
        return Err("NVFP4 macro scale consumer requires one F32 value".into());
    }
    let bytes: [u8; 4] = view
        .bytes
        .as_ref()
        .try_into()
        .map_err(|_| "NVFP4 macro scale consumer requires exactly four bytes")?;
    Ok(f32::from_le_bytes(bytes))
}
