//! Metadata binding for sparse overlays, before runtime activation.
//!
//! Every component is validated in its own dialect before precedence is applied. The resulting
//! catalog carries no source, file, or materialization authority. Runtime fallback readers and
//! artifact identity remain refused until component-aware consumers are implemented.
use super::*;
use crate::source::Hy3RepackSource;
use crate::source::composite::ComponentInterpretation;
use crate::tensor_contract::{TensorMatch, TensorOwner, TensorRequirement};
use std::collections::BTreeSet;

mod access;
mod runtime;
pub use access::BoundCompositeSource;
pub use runtime::CompositeRuntimeSource;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeTensorMember {
    pub component_path: Vec<u32>,
    pub dialect: CheckpointDialect,
    pub record: TensorCensusRecord,
    pub transform: TensorTransform,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeTensorSelection {
    pub owner: TensorOwner,
    /// Original router IDs for expert groups; None for a whole physical tensor.
    pub member_ids: Option<Vec<u32>>,
    /// Canonical contract order for a group; exactly one member for a whole tensor.
    pub members: Vec<CompositeTensorMember>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeComponentCatalog {
    component_path: Vec<u32>,
    census: TensorCensus,
    contract: TensorContract,
    binding: BoundTensorContract,
    by_name: BTreeMap<String, usize>,
    member_ids: BTreeMap<TensorId, Vec<u32>>,
    interpretation: ComponentInterpretation,
    scope: CatalogScope,
}

impl CompositeComponentCatalog {
    pub fn component_path(&self) -> &[u32] {
        &self.component_path
    }
    pub fn census(&self) -> &TensorCensus {
        &self.census
    }
    pub fn binding(&self) -> &BoundTensorContract {
        &self.binding
    }
}

/// An immutable metadata result, constructible only from the actual opened repack chain.
/// It does not implement TensorSource and cannot be substituted for a bound runtime source.
pub struct CompositeTensorCatalog {
    config: ModelConfig,
    plan: ModelPlan,
    active_experts: BTreeMap<u32, Vec<bool>>,
    components: Vec<CompositeComponentCatalog>,
    selected: BTreeMap<TensorId, CompositeTensorSelection>,
    charges: Vec<crate::source::BoundTensorCharge>,
    digest: String,
}

impl CompositeTensorCatalog {
    pub fn compile(source: &Hy3RepackSource, requested: LoadScope) -> Result<Self, String> {
        let opened = source.binding_components();
        if opened.len() < 2 {
            return Err("composite catalog requires a fallback component".into());
        }
        let mut components = Vec::new();
        let mut plan = None;
        let interpretations = opened
            .iter()
            .enumerate()
            .map(|(index, component)| {
                component
                    .interpretation()
                    .map_err(|e| format!("component {:?}: {e}", vec![0u32; index]))
            })
            .collect::<Result<Vec<_>, String>>()?;
        // Each component inherits the nearest lower declaration unless it explicitly replaces
        // that layer's mask. A higher sparse overlay cannot resurrect an inherited pruned ID.
        let mut masks_by_component = vec![BTreeMap::new(); opened.len()];
        let mut active_experts = BTreeMap::new();
        for index in (0..opened.len()).rev() {
            if let ComponentInterpretation::Repack {
                active_experts: masks,
                ..
            } = &interpretations[index]
            {
                active_experts.extend(masks.iter().map(|(&layer, mask)| (layer, mask.clone())));
            }
            masks_by_component[index] = active_experts.clone();
        }
        let config = source.config();
        for (index, component) in opened.iter().enumerate() {
            let path = vec![0; index];
            let context = |error: String| format!("component {path:?}: {error}");
            let config = component.config();
            let mut component_plan =
                model_packs::compile_for_load(&config).map_err(|e| context(e.to_string()))?;
            if let Some(expected) = &plan {
                if expected != &component_plan {
                    return Err(context(
                        "component numerical plan differs from the overlay plan".into(),
                    ));
                }
            } else {
                plan = Some(component_plan.clone());
            }
            let interpretation = interpretations[index].clone();
            if let ComponentInterpretation::Repack {
                expert_activation_precision,
                active_experts: _,
            } = &interpretation
            {
                bind_retained_experts(
                    &mut component_plan,
                    &BoundSourceInterpretation::RetainedRepack {
                        expert_activation_precision: *expert_activation_precision,
                        active_experts: masks_by_component[index].clone(),
                    },
                )
                .map_err(context)?;
            }
            let mut census = component.census().map_err(context)?;
            census
                .tensors
                .sort_by(|a, b| a.entry.name.cmp(&b.entry.name));
            validate_physical_records(&census).map_err(context)?;
            let pack =
                model_packs::for_config(&config).ok_or("compiled model has no model pack")?;
            // A GGUF-dialect text overlay is a sparse storage fragment, not a replacement
            // vision program. The HF component still validates every vision row. Any vision
            // tensor declared by this unsupported overlay dialect remains an extra and refuses.
            let mut schema_plan = component_plan.clone();
            if census.dialect == CheckpointDialect::Gguf && component.is_overlay() {
                schema_plan.vision = None;
                schema_plan.multimodal = None;
            }
            let mut contract = pack
                .compile_tensor_contract(
                    &config,
                    &schema_plan,
                    census.dialect,
                    ContractOptions {
                        output_head: crate::checkpoint_binding::declared_output_head_for(
                            pack,
                            &config,
                            census.dialect,
                        )
                        .map_err(|error| context(error.to_string()))?,
                    },
                )
                .map_err(|e| context(e.to_string()))?;
            let additional = pack
                .additional_inventory(&config, census.dialect, component.raw_config())
                .map_err(context)?;
            let entries: Vec<_> = census.tensors.iter().map(|r| r.entry.clone()).collect();
            contract
                .declared_expert_members(&component_plan, &entries)
                .map_err(context)?;
            let scope =
                CatalogScope::compile(requested, &component_plan, &mut contract, additional)
                    .map_err(context)?;
            let member_ids = contract
                .requirements
                .iter()
                .filter(|r| r.match_mode == TensorMatch::All)
                .map(|r| {
                    let ids = crate::tensor_contract::expert_member_ids(&component_plan, &r.id)
                        .unwrap_or_else(|| (0..r.names.len() as u32).collect());
                    if ids.len() != r.names.len() {
                        return Err(context(format!(
                            "{:?}: group slots disagree with original IDs",
                            r.id
                        )));
                    }
                    Ok((r.id.clone(), ids))
                })
                .collect::<Result<BTreeMap<_, _>, String>>()?;
            let binding = contract
                .bind_fragment(&entries)
                .map_err(|e| context(e.to_string()))?;
            let by_name: BTreeMap<_, _> = census
                .tensors
                .iter()
                .enumerate()
                .map(|(i, row)| (row.entry.name.clone(), i))
                .collect();
            // Validate every own header, including tensors subsequently shadowed by an overlay.
            for (id, tensor) in &binding.tensors {
                for name in &tensor.checkpoint_names {
                    let record = &census.tensors[by_name[name]];
                    component
                        .validate(&BoundTensorRequest {
                            id,
                            record,
                            dialect: census.dialect,
                            transform: tensor.transform,
                            view: BoundTensorView::Whole,
                        })
                        .map_err(context)?;
                }
                if let TensorId::QuantAux { tensor, .. } = id
                    && !binding.tensors.contains_key(tensor.as_ref())
                {
                    return Err(context(format!(
                        "auxiliary {id:?} has no owning weight in this component"
                    )));
                }
            }
            components.push(CompositeComponentCatalog {
                component_path: path,
                census,
                contract,
                binding,
                by_name,
                member_ids,
                interpretation,
                scope,
            });
        }

        let mut plan = plan.unwrap();
        bind_retained_experts(
            &mut plan,
            &BoundSourceInterpretation::RetainedRepack {
                expert_activation_precision: source.expert_activation_precision(),
                active_experts: active_experts.clone(),
            },
        )?;

        // The leaf contract declares completeness for the base artifact. Overlays are sparse
        // physical replacements: merely using a GGUF norm does not make an HF base require the
        // GGUF-only rope_freqs tensor. Every declared overlay row has already been validated.
        // A group is filled only from the same declared group representation, never by slicing
        // a lower stacked bank as an implicit substitute for a missing split member.
        let required_ids: BTreeSet<_> = components
            .last()
            .unwrap()
            .contract
            .requirements
            .iter()
            .filter(|r| r.required)
            .map(|r| r.id.clone())
            .collect();
        let ids: BTreeSet<_> = components
            .iter()
            .flat_map(|c| c.contract.requirements.iter().map(|r| r.id.clone()))
            .collect();
        let mut selected = BTreeMap::new();
        let mut charges = Vec::new();
        for id in ids {
            let candidates: Vec<_> = components
                .iter()
                .filter_map(|component| {
                    let requirement = component
                        .contract
                        .requirements
                        .iter()
                        .find(|r| r.id == id)?;
                    Some((component, requirement))
                })
                .collect();
            let required = required_ids.contains(&id);
            let first = candidates
                .iter()
                .find(|(c, _)| c.binding.tensors.contains_key(&id));
            let Some(&(first_component, first_requirement)) = first else {
                if required {
                    return Err(format!("composite is missing required tensor {id:?}"));
                }
                continue;
            };
            let owner = first_requirement.owner;
            if candidates.iter().any(|(_, r)| r.owner != owner) {
                return Err(format!("composite ownership disagrees for {id:?}"));
            }
            let canonical_ids = crate::tensor_contract::expert_member_ids(&plan, &id);
            let retained = if let TensorId::Layer { index, .. } = &id {
                active_experts.contains_key(index) && canonical_ids.is_some()
            } else {
                false
            };
            if retained && first_requirement.match_mode == TensorMatch::OneOf {
                return Err(format!(
                    "{id:?}: retained program cannot select a stacked bank; explicit retained members are required"
                ));
            }
            let selected_ids = if first_requirement.match_mode == TensorMatch::All {
                Some(canonical_ids.unwrap_or_else(|| first_component.member_ids[&id].clone()))
            } else {
                None
            };
            let members = if first_requirement.match_mode == TensorMatch::OneOf {
                vec![selected_member(first_component, first_requirement, None)?]
            } else {
                let mut members = Vec::new();
                for &original in selected_ids.as_ref().unwrap() {
                    let candidate = candidates.iter().find_map(|(c, r)| {
                        let member = c.member_ids.get(&id)?.iter().position(|&i| i == original)?;
                        c.binding
                            .tensors
                            .get(&id)?
                            .checkpoint_names
                            .contains(&r.names[member])
                            .then_some((*c, *r, member))
                    });
                    let Some((c, r, member)) = candidate else {
                        return Err(format!(
                            "composite {id:?} is missing group member {original}; implicit bank slicing is unsupported"
                        ));
                    };
                    members.push(selected_member(c, r, Some(member))?);
                }
                members
            };
            // All selected contributions must agree on execution scope. Inventory-only rows
            // remain in components but are never promoted into executable tensor selections.
            if candidates.iter().any(|(c, _)| c.scope.permits(&id)) {
                if candidates.iter().any(|(c, _)| !c.scope.permits(&id)) {
                    return Err(format!("composite scope disagrees for {id:?}"));
                }
                selected.insert(
                    id,
                    CompositeTensorSelection {
                        owner,
                        member_ids: selected_ids,
                        members,
                    },
                );
            } else {
                let physical_bytes = members.iter().try_fold(0u64, |total, member| {
                    total
                        .checked_add(member.record.entry.physical_bytes)
                        .ok_or_else(|| format!("composite inventory charge overflows for {id:?}"))
                })?;
                charges.push(crate::source::BoundTensorCharge {
                    id,
                    owner,
                    physical_bytes,
                    execution_selected: false,
                });
            }
        }
        // Independent GGUF auxiliary rows follow their selected owning weight. A lower scale
        // must not survive replacement of the weight it describes, even if its semantic ID is
        // otherwise the first available auxiliary. Folded HF planes already travel in the row.
        let mut shadowed_auxiliaries = Vec::new();
        for (id, selection) in &selected {
            let TensorId::QuantAux { tensor, .. } = id else {
                continue;
            };
            let parent = selected.get(tensor.as_ref()).ok_or_else(|| {
                format!("selected auxiliary {id:?} has no selected owning weight")
            })?;
            if selection.members.len() != 1 || parent.members.len() != 1 {
                return Err(format!(
                    "independent composite auxiliary groups for {tensor:?} are unsupported"
                ));
            }
            if selection.members[0].component_path != parent.members[0].component_path {
                shadowed_auxiliaries.push(id.clone());
            }
        }
        for id in shadowed_auxiliaries {
            selected.remove(&id);
        }
        for (id, selection) in &selected {
            let physical_bytes = selection.members.iter().try_fold(0u64, |total, member| {
                total
                    .checked_add(member.record.entry.physical_bytes)
                    .ok_or_else(|| format!("composite selected charge overflows for {id:?}"))
            })?;
            charges.push(crate::source::BoundTensorCharge {
                id: id.clone(),
                owner: selection.owner,
                physical_bytes,
                execution_selected: true,
            });
        }
        if requested == LoadScope::Text {
            plan.vision = None;
            plan.multimodal = None;
        }
        let mut hash = Sha256::new();
        hash.update(b"memra-composite-metadata-v2\0");
        for value in [
            format!("{plan:?}"),
            format!("{components:?}"),
            format!("{selected:?}"),
        ] {
            hash.update((value.len() as u64).to_le_bytes());
            hash.update(value.as_bytes());
        }
        Ok(Self {
            config,
            plan,
            active_experts,
            components,
            selected,
            charges,
            digest: format!("{:x}", hash.finalize()),
        })
    }

    pub fn plan(&self) -> &ModelPlan {
        &self.plan
    }
    pub fn components(&self) -> &[CompositeComponentCatalog] {
        &self.components
    }
    pub fn selected(&self) -> &BTreeMap<TensorId, CompositeTensorSelection> {
        &self.selected
    }
    /// Semantic metadata digest, not an opened-byte artifact or rewrite identity.
    pub fn binding_sha256(&self) -> &str {
        &self.digest
    }
}

fn selected_member(
    component: &CompositeComponentCatalog,
    requirement: &TensorRequirement,
    member: Option<usize>,
) -> Result<CompositeTensorMember, String> {
    let bound = &component.binding.tensors[&requirement.id];
    let name = match member {
        None => &bound.checkpoint_names[0],
        Some(index) => &requirement.names[index],
    };
    let index = component.by_name.get(name).ok_or_else(|| {
        format!(
            "selected component {:?} is missing {name}",
            component.component_path
        )
    })?;
    let record = &component.census.tensors[*index];
    Ok(CompositeTensorMember {
        component_path: component.component_path.clone(),
        dialect: component.census.dialect,
        record: record.clone(),
        transform: bound.transform,
    })
}

fn validate_physical_records(census: &TensorCensus) -> Result<(), String> {
    let mut physical = BTreeSet::new();
    for row in &census.tensors {
        for name in std::iter::once(&row.physical_name)
            .chain(row.auxiliaries.iter().map(|a| &a.physical_name))
        {
            if !physical.insert(name) {
                return Err(format!("duplicate physical tensor {name}"));
            }
        }
        let auxiliaries: Vec<_> = row
            .auxiliaries
            .iter()
            .map(|aux| match census.dialect {
                CheckpointDialect::Gguf => aux.physical_name.clone(),
                CheckpointDialect::HfSafetensors => {
                    crate::source::canonical_hf_name(&aux.physical_name)
                }
            })
            .collect();
        if auxiliaries != row.entry.auxiliaries {
            return Err(format!(
                "tensor {} auxiliary census disagrees with its physical records",
                row.entry.name
            ));
        }
    }
    Ok(())
}
