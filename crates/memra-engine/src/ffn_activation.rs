//! Native activation dispatch, kept CPU-testable without executing a device kernel.
use crate::{CudaSlice, Engine};
use memra_gguf::bound_source::consumer::FfnActivation;
use memra_gguf::config::{ModelConfig, SwigluClamp};

#[allow(clippy::too_many_arguments)] // allow: preserve the existing native FFN call signature
pub(crate) fn apply(
    e: &Engine,
    cfg: &ModelConfig,
    gate: &CudaSlice<f32>,
    up: &CudaSlice<f32>,
    gs: f32,
    us: f32,
    limit: Option<SwigluClamp>,
    act: &mut CudaSlice<f32>,
    n: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    debug_assert!(
        cfg.m3.is_none() || limit.is_none(),
        "m3 swigluoai and the step35/glm5_next clamps are different archs"
    );
    match FfnActivation::for_target(cfg, limit) {
        FfnActivation::SwiGluOai { alpha, limit } => {
            e.swigluoai_mul_scaled(gate, up, gs, us, alpha, limit, act, n)
        }
        FfnActivation::PostClamped { limit } => {
            e.swiglu_clamped_mul_scaled(gate, up, gs, us, limit, act, n)
        }
        FfnActivation::PreClamped { limit } => {
            e.swiglu_preclamped_mul_scaled(gate, up, gs, us, limit, act, n)
        }
        FfnActivation::Silu if gs == 1.0 && us == 1.0 => e.silu_mul(gate, up, act, n),
        FfnActivation::Silu => e.silu_mul_scaled(gate, up, gs, us, act, n),
    }
}
