//! Physical inventory is distinct from executable program support.
use crate::model_plan::ModelPlan;
use crate::tensor_contract::{TensorContract, TensorId, TensorOwner, TensorRequirement};
use std::collections::BTreeSet;

/// Component selection is separate from eager/batch/graph/speculative execution rewrites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadScope {
    Full,
    Text,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ArtifactComponent {
    Text,
    Vision,
}

/// Metadata-only catalog. No reference executor or native capability is implied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InventorySurface {
    pub component: ArtifactComponent,
    pub execution_unavailable: &'static str,
    pub requirements: Vec<TensorRequirement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogScope {
    pub requested: LoadScope,
    pub selected: BTreeSet<TensorId>,
    pub inventory_only: Vec<InventorySurface>,
}

impl CatalogScope {
    pub(crate) fn compile(
        requested: LoadScope,
        plan: &ModelPlan,
        contract: &mut TensorContract,
        inventory_only: Vec<InventorySurface>,
    ) -> Result<Self, String> {
        let mut surfaces = BTreeSet::new();
        for surface in &inventory_only {
            if !surfaces.insert(surface.component) {
                return Err("duplicate inventory-only component".into());
            }
            if surface.component == ArtifactComponent::Vision && plan.vision.is_some() {
                return Err("vision cannot be both executable and inventory-only".into());
            }
            if surface.requirements.iter().any(|requirement| {
                matches!(requirement.owner, TensorOwner::Vision(_))
                    != (surface.component == ArtifactComponent::Vision)
            }) {
                return Err("inventory component and tensor owner disagree".into());
            }
            contract
                .requirements
                .extend(surface.requirements.iter().cloned());
        }
        if let Some(surface) = inventory_only.iter().find(|surface| {
            requested == LoadScope::Full || surface.component == ArtifactComponent::Text
        }) {
            return Err(format!(
                "requested {:?} execution is unsupported by this bound plan: {}",
                surface.component, surface.execution_unavailable
            ));
        }
        let selected = contract
            .requirements
            .iter()
            .filter(|r| match requested {
                LoadScope::Full => true,
                LoadScope::Text => !matches!(r.owner, TensorOwner::Vision(_)),
            })
            .map(|r| r.id.clone())
            .collect();
        Ok(Self {
            requested,
            selected,
            inventory_only,
        })
    }
    pub fn permits(&self, id: &TensorId) -> bool {
        self.selected.contains(id)
    }
    pub fn authorize(&self, id: &TensorId) -> Result<(), String> {
        if self.permits(id) {
            Ok(())
        } else {
            Err(format!(
                "tensor {id:?} is outside selected {:?} execution",
                self.requested
            ))
        }
    }
}
