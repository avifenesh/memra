use crate::{Engine, GgmlType, GpuTensor};
use memra_gguf::{config::ModelConfig, model_plan::ModelPlan};
use std::sync::Arc;
#[path = "../../../../../crates/memra-engine/src/hybrid/root_trim.rs"]
pub(crate) mod root_trim;

pub struct MtpHead {
    external_source_identity: Option<memra_gguf::bound_source::BoundArtifactIdentity>,
    embedded_block_index: Option<u32>,
    pending_trim_slot: Option<root_trim::SlotReservation>,
    pub geom: Option<()>,
    pub shared_head_head: Option<GpuTensor>,
    pub d2t: Option<Vec<u32>>,
    pub d2t_from_target_head: bool,
}
pub struct HybridProgram {
    pub cfg: ModelConfig,
    pub plan: ModelPlan,
    pub mtp: Option<MtpHead>,
    pub mtp_extra: Vec<MtpHead>,
    pub frspec_src_sha16: Option<String>,
    pub glm5_dflash: Option<()>,
    pub dflash_trim: Option<()>,
}
pub struct HybridModel {
    program: crate::plan_backend::TrackedProgram<HybridProgram>,
    rewrite_generation: Arc<crate::plan_backend::ProgramGeneration>,
}
impl std::ops::Deref for HybridModel {
    type Target = HybridProgram;
    fn deref(&self) -> &HybridProgram {
        &self.program
    }
}
impl std::ops::DerefMut for HybridModel {
    fn deref_mut(&mut self) -> &mut HybridProgram {
        &mut self.program
    }
}
impl HybridModel {
    pub(crate) fn rewrite_mutations(&self) -> u64 {
        self.rewrite_generation.mutations()
    }
    pub fn devices(&self) -> Vec<usize> {
        vec![2]
    }
    pub fn is_multi_device(&self) -> bool {
        false
    }
}

#[path = "tests.rs"]
mod tests;
