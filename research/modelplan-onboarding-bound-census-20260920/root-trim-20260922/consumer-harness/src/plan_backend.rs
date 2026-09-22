#![allow(dead_code)] // focused host harness imports the real identity/snapshot helpers without running their broad suites
use crate::hybrid::HybridModel;
pub use memra_gguf::execution_manifest::*;
use std::{collections::BTreeMap, ffi::OsString};
#[path = "../../../../../crates/memra-engine/src/plan_backend/execution_snapshot.rs"]
mod execution_snapshot;
pub(crate) use execution_snapshot::{ProgramGeneration, RewriteExecutionSnapshot, TrackedProgram};
#[path = "../../../../../crates/memra-engine/src/plan_backend/runtime_identity.rs"]
mod runtime_identity;
use runtime_identity::{
    hash_parts, numeric_environment, numeric_program_sha256, running_implementation_sha256,
};
#[path = "../../../../../crates/memra-engine/src/plan_backend/target_trim.rs"]
mod target_trim;
pub(crate) use runtime_identity::install_rewrite_admission;
pub(crate) use target_trim::capture_target_trim_rewrite_identity;

// Only unavailable process-library/hardware inspection and the full native model metadata
// provider are stand-ins. The new capture function, byte framing, executable hash, env rules,
// qualifications and mutation/snapshot machinery are production code.
pub(super) struct LoadedLibraries {
    pub(super) sha256: String,
}
impl LoadedLibraries {
    fn capture() -> Result<Self, String> {
        Ok(Self {
            sha256: "a".repeat(64),
        })
    }
    fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}
pub(crate) struct RewriteLoadState {
    mutation_generation: u64,
    pipeline: bool,
    model_sha256: String,
    environment: BTreeMap<OsString, OsString>,
    libraries: Option<LoadedLibraries>,
}
impl RewriteLoadState {
    pub(crate) fn validate(&self, model: &HybridModel) -> Result<(), String> {
        if self.mutation_generation != model.rewrite_mutations() {
            return Err("loaded program mutated".into());
        }
        self.libraries
            .as_ref()
            .ok_or("library capture missing")?
            .validate()?;
        if self.environment != numeric_environment(std::env::vars_os()) {
            return Err("numeric environment changed".into());
        }
        Ok(())
    }
}
fn loaded_model_sha256(model: &HybridModel) -> String {
    let slots = model
        .mtp
        .iter()
        .chain(&model.mtp_extra)
        .map(|head| {
            let tensor = head.shared_head_head.as_ref().map(|t| match t {
                crate::GpuTensor::Quant {
                    qtype,
                    row_bytes,
                    ne,
                    scale,
                    rp,
                    ..
                } => format!("{qtype} {row_bytes} {ne:?} {} {rp}", scale.to_bits()),
                crate::GpuTensor::Float { ne, .. } => format!("f32 {ne:?}"),
                crate::GpuTensor::FloatBf16 { ne, .. } => format!("bf16 {ne:?}"),
            });
            format!("{tensor:?} {:?} {}", head.d2t, head.d2t_from_target_head)
        })
        .collect::<Vec<_>>();
    hash_parts([format!("{:?} {:?} {slots:?}", model.cfg, model.plan)])
}
fn hardware_snapshot(devices: &[usize]) -> Result<String, Box<dyn std::error::Error>> {
    Ok(format!("host-recorder-devices={devices:?}"))
}
