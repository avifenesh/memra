//! Immutable checkpoint binding for the runtime loader migration.
//!
//! Construction binds the complete metadata census before any tensor materialization. A request
//! can only be made from this compiler-produced bundle; callers cannot pair arbitrary config,
//! plans, physical targets or transforms. Returned views borrow the original opened source.

use std::collections::BTreeMap;

pub mod composite;
pub mod draft_pair;

use sha2::{Digest, Sha256};

use crate::bound_disk::{BoundDiskCache, BoundDiskView};
use crate::config::ModelConfig;
use crate::model_packs;
use crate::model_plan::ModelPlan;
use crate::source::{
    BoundSourceInterpretation, DiskExtent, Fp8Native, Fp8StackedNative, Nvfp4Native,
    Nvfp4StackedNative, TensorCensus, TensorCensusRecord, TensorSource, TensorView,
};
use crate::surface_catalog::{CatalogScope, LoadScope};
use crate::tensor_contract::{
    BoundTensorContract, CheckpointDialect, ContractOptions, OutputHead, QuantAuxTensor,
    TensorContract, TensorId, TensorTransform,
};

/// A sealed request for one physical member of a semantic tensor. Public so source implementations
/// can consume it, but it is only constructed by `BoundTensorSource`.
pub struct BoundTensorRequest<'a> {
    pub(crate) id: &'a TensorId,
    pub(crate) record: &'a TensorCensusRecord,
    pub(crate) dialect: CheckpointDialect,
    pub(crate) transform: TensorTransform,
    pub(crate) view: BoundTensorView,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundTensorView {
    Whole,
    EncodedBank,
    MlaKey,
    MlaValue,
}

impl BoundTensorRequest<'_> {
    pub fn id(&self) -> &TensorId {
        self.id
    }
    pub fn record(&self) -> &TensorCensusRecord {
        self.record
    }
    pub fn dialect(&self) -> CheckpointDialect {
        self.dialect
    }
    pub fn transform(&self) -> TensorTransform {
        self.transform
    }
    pub fn view(&self) -> BoundTensorView {
        self.view
    }
    pub fn error(&self, message: impl std::fmt::Display) -> String {
        format!(
            "tensor {:?} ({:?}, physical {:?}, {:?}): {message}",
            self.id, self.record.entry.name, self.record.physical_name, self.transform
        )
    }
}

/// The two independent identity components and their framed composite. All values come from
/// the opened source and sealed binding; callers never supply a manifest or replacement hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundArtifactIdentity {
    pub opened_source_sha256: String,
    pub semantic_scope_sha256: String,
    pub artifact_sha256: String,
}

/// Compiler-private preflight authority. It refers to one complete sealed bundle, not a
/// caller-assembled config/plan pair. External TensorSource implementations cannot name this
/// return type and therefore cannot override the compiler-only trait hook.
pub(crate) enum BoundProgramRef<'a> {
    Single(&'a BoundTensorSource<'a>),
    Composite(&'a composite::BoundCompositeSource<'a>),
    ExternalDraft(&'a BoundTensorSource<'a>),
}
impl BoundProgramRef<'_> {
    pub(crate) fn cloned_pair(&self) -> (ModelConfig, ModelPlan) {
        match self {
            Self::Single(bound) | Self::ExternalDraft(bound) => {
                (bound.config.clone(), bound.plan.clone())
            }
            Self::Composite(bound) => (bound.config().clone(), bound.plan().clone()),
        }
    }
    pub(crate) fn output_head(&self) -> OutputHead {
        match self {
            Self::Single(bound) | Self::ExternalDraft(bound) => bound.output_head(),
            Self::Composite(bound) => {
                if bound
                    .catalog()
                    .selected()
                    .contains_key(&TensorId::OutputProjection)
                {
                    OutputHead::Separate
                } else {
                    OutputHead::TiedToEmbedding
                }
            }
        }
    }

    pub(crate) fn consumer_tensors(&self) -> Vec<BoundConsumerTensor> {
        match self {
            Self::Single(bound) | Self::ExternalDraft(bound) => bound
                .binding()
                .tensors
                .iter()
                .filter(|(id, _)| bound.scope().permits(id))
                .map(|(id, tensor)| BoundConsumerTensor {
                    id: id.clone(),
                    owner: tensor.owner,
                    checkpoint_names: tensor.checkpoint_names.clone(),
                })
                .collect(),
            Self::Composite(bound) => bound
                .catalog()
                .selected()
                .iter()
                .map(|(id, tensor)| BoundConsumerTensor {
                    id: id.clone(),
                    owner: tensor.owner,
                    checkpoint_names: tensor
                        .members
                        .iter()
                        .map(|member| {
                            format!(
                                "component {:?} {:?}: {}",
                                member.component_path, member.dialect, member.record.entry.name
                            )
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    pub(crate) fn consumed_ids(
        &self,
        requested: &std::collections::BTreeSet<String>,
    ) -> Result<std::collections::BTreeSet<TensorId>, String> {
        fn record(
            access: abi::Access,
            tied: bool,
            ids: &mut std::collections::BTreeSet<TensorId>,
        ) -> TensorId {
            let id = match access {
                abi::Access::Tensor(id)
                | abi::Access::Member(id, _)
                | abi::Access::Derived(id, _) => id,
                abi::Access::Auxiliary(parent, kind) => {
                    let owner = record(*parent, tied, ids);
                    TensorId::QuantAux {
                        tensor: Box::new(owner),
                        kind,
                    }
                }
            };
            let id = if tied && id == TensorId::OutputProjection {
                TensorId::TokenEmbedding
            } else {
                id
            };
            ids.insert(id.clone());
            id
        }
        let (_, plan) = self.cloned_pair();
        let options = ContractOptions {
            output_head: self.output_head(),
        };
        let abi = abi::RuntimeAbi::new(&plan, options)?;
        let mut ids = std::collections::BTreeSet::new();
        for name in requested {
            let access = match self {
                Self::Single(bound) | Self::ExternalDraft(bound) => abi.access(*bound, name)?,
                Self::Composite(bound) => abi.access(*bound, name)?,
            };
            record(
                access,
                options.output_head == OutputHead::TiedToEmbedding,
                &mut ids,
            );
        }
        Ok(ids)
    }
}

/// Read-only consumption metadata. Composite members retain their component and dialect labels;
/// this does not flatten a physical census or convey tensor materialization authority.
pub(crate) struct BoundConsumerTensor {
    pub id: TensorId,
    pub owner: crate::tensor_contract::TensorOwner,
    pub checkpoint_names: Vec<String>,
}

/// Owns the semantic interpretation, borrows the exact opened artifact. No mutable accessors.
///
/// An ordinary external source cannot install a caller-built preflight tuple:
/// ```compile_fail
/// use memra_gguf::{config::ModelConfig, model_plan::ModelPlan,
///     source::{TensorSource, TensorView}};
/// struct Unbound { config: ModelConfig, plan: ModelPlan }
/// impl TensorSource for Unbound {
///     fn config(&self) -> ModelConfig { self.config.clone() }
///     fn find(&self, _: &str) -> Option<TensorView<'_>> { None }
///     fn bound_program(&self) -> Option<(&ModelConfig, &ModelPlan)> {
///         Some((&self.config, &self.plan))
///     }
/// }
/// ```
/// Nor can it forward preflight authority from a different sealed source:
/// ```compile_fail
/// use memra_gguf::{bound_source::{BoundProgramRef, BoundRuntimeSource},
///     config::ModelConfig, source::{TensorSource, TensorView}};
/// struct Swapped<'a, 'b, 'c> { config: ModelConfig, other: &'a BoundRuntimeSource<'b, 'c> }
/// impl TensorSource for Swapped<'_, '_, '_> {
///     fn config(&self) -> ModelConfig { self.config.clone() }
///     fn find(&self, _: &str) -> Option<TensorView<'_>> { None }
///     fn bound_program(&self) -> Option<BoundProgramRef<'_>> { self.other.bound_program() }
/// }
/// ```
pub struct BoundTensorSource<'a> {
    source: &'a dyn TensorSource,
    config: ModelConfig,
    plan: ModelPlan,
    contract: TensorContract,
    options: ContractOptions,
    binding: BoundTensorContract,
    census: TensorCensus,
    by_name: BTreeMap<String, usize>,
    digest: String,
    interpretation: BoundSourceInterpretation,
    runtime_metadata: crate::source::RuntimeSourceMetadata,
    active_experts: BTreeMap<u32, Vec<bool>>,
    scope: CatalogScope,
    disk_cache: BoundDiskCache,
}

impl<'a> BoundTensorSource<'a> {
    pub fn compile(source: &'a dyn TensorSource) -> Result<Self, Box<dyn std::error::Error>> {
        Self::compile_for_scope(source, LoadScope::Full)
    }
    pub fn compile_for_scope(
        source: &'a dyn TensorSource,
        requested: LoadScope,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut config = source.try_config().map_err(std::io::Error::other)?;
        let interpretation = source
            .bound_interpretation()
            .map_err(std::io::Error::other)?;
        let runtime_metadata = source.runtime_metadata().map_err(std::io::Error::other)?;
        let mut plan = model_packs::compile_for_load(&config)?;
        let active_experts = bind_retained_experts(&mut plan, &interpretation)?;
        for layer in plan
            .layers
            .iter()
            .chain(plan.mtp_blocks.iter().map(|b| &b.layer))
        {
            if source.active_experts(layer.index)
                != active_experts.get(&layer.index).map(Vec::as_slice)
            {
                return Err(format!(
                    "layer {} expert mask differs from its declared bound program",
                    layer.index
                )
                .into());
            }
        }
        let pack = model_packs::for_config(&config).ok_or("compiled model has no model pack")?;
        let mut census = source.tensor_census().map_err(std::io::Error::other)?;
        // Row order is not part of checkpoint semantics. Preserve group-member order in binding.
        census
            .tensors
            .sort_by(|a, b| a.entry.name.cmp(&b.entry.name));
        for row in &census.tensors {
            let physical_auxiliaries: Vec<_> = row
                .auxiliaries
                .iter()
                .map(|aux| {
                    if census.dialect == CheckpointDialect::HfSafetensors {
                        crate::source::canonical_hf_name(&aux.physical_name)
                    } else {
                        aux.physical_name.clone()
                    }
                })
                .collect();
            if physical_auxiliaries != row.entry.auxiliaries {
                return Err(format!(
                    "tensor {} auxiliary census disagrees with its physical records",
                    row.entry.name
                )
                .into());
            }
        }
        // Retain the incoming pack refusal for a forbidden tied head, then bind the
        // prepared program's declared ownership. Tensor absence cannot change that program.
        let options = ContractOptions {
            output_head: crate::checkpoint_binding::declared_output_head_for(
                pack,
                &config,
                census.dialect,
            )?,
        };
        let mut contract = pack.compile_tensor_contract(&config, &plan, census.dialect, options)?;
        let inventory =
            pack.additional_inventory(&config, census.dialect, source.raw_config_json())?;
        let scope = CatalogScope::compile(requested, &plan, &mut contract, inventory)?;
        let entries: Vec<_> = census.tensors.iter().map(|row| row.entry.clone()).collect();
        let binding = contract.bind(&entries)?;
        let by_name: BTreeMap<_, _> = census
            .tensors
            .iter()
            .enumerate()
            .map(|(i, row)| (row.entry.name.clone(), i))
            .collect();
        for (id, bound) in &binding.tensors {
            for name in &bound.checkpoint_names {
                let record = &census.tensors[by_name[name]];
                source
                    .validate_bound_metadata(&BoundTensorRequest {
                        id,
                        record,
                        dialect: census.dialect,
                        transform: bound.transform,
                        view: BoundTensorView::Whole,
                    })
                    .map_err(std::io::Error::other)?;
            }
        }
        // Keep #537's numerical factor validation after the complete metadata boundary. The
        // contract has already checked the factor's identity, storage and accepted shape.
        model_packs::prepare_bound_rope_factors(&mut config, &plan, source)?;
        if requested == LoadScope::Text {
            plan.vision = None;
            plan.multimodal = None;
        }
        let digest = binding_digest(
            &config,
            &plan,
            &contract,
            options,
            &binding,
            &census,
            &interpretation,
            &runtime_metadata,
            &scope,
        );
        Ok(Self {
            source,
            config,
            plan,
            contract,
            options,
            binding,
            census,
            by_name,
            digest,
            interpretation,
            runtime_metadata,
            active_experts,
            scope,
            disk_cache: BoundDiskCache::default(),
        })
    }

    pub fn artifact_identity(&self) -> Result<BoundArtifactIdentity, String> {
        let opened_source_sha256 = self.source.artifact_sha256()?;
        if opened_source_sha256.len() != 64
            || !opened_source_sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err("opened source returned an invalid SHA-256 identity".into());
        }
        let semantic_scope_sha256 = self.digest.clone();
        let mut hash = Sha256::new();
        hash.update(b"memra-bound-runtime-artifact-v1\0");
        for (tag, value) in [
            (
                b"opened-source-sha256".as_slice(),
                opened_source_sha256.as_bytes(),
            ),
            (
                b"semantic-scope-sha256".as_slice(),
                semantic_scope_sha256.as_bytes(),
            ),
        ] {
            hash.update((tag.len() as u64).to_le_bytes());
            hash.update(tag);
            hash.update((value.len() as u64).to_le_bytes());
            hash.update(value);
        }
        Ok(BoundArtifactIdentity {
            opened_source_sha256,
            semantic_scope_sha256,
            artifact_sha256: format!("{:x}", hash.finalize()),
        })
    }

    pub fn scope(&self) -> &CatalogScope {
        &self.scope
    }

    pub fn interpretation(&self) -> &BoundSourceInterpretation {
        &self.interpretation
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    pub fn plan(&self) -> &ModelPlan {
        &self.plan
    }
    pub fn contract(&self) -> &TensorContract {
        &self.contract
    }
    pub fn binding(&self) -> &BoundTensorContract {
        &self.binding
    }
    pub fn census(&self) -> &TensorCensus {
        &self.census
    }
    pub fn output_head(&self) -> OutputHead {
        self.options.output_head
    }
    /// Interpretation identity only. This is NOT an artifact hash or a rewrite receipt.
    pub fn binding_sha256(&self) -> &str {
        &self.digest
    }

    fn resolved_id<'b>(&'b self, id: &'b TensorId) -> &'b TensorId {
        if *id == TensorId::OutputProjection && self.output_head() == OutputHead::TiedToEmbedding {
            &TensorId::TokenEmbedding
        } else {
            id
        }
    }

    pub fn contains(&self, id: &TensorId) -> bool {
        self.binding.tensors.contains_key(self.resolved_id(id))
    }

    fn request<'b>(
        &'b self,
        id: &'b TensorId,
        member: Option<usize>,
    ) -> Result<BoundTensorRequest<'b>, String> {
        let id = self.resolved_id(id);
        self.scope.authorize(id)?;
        let bound = self
            .binding
            .tensors
            .get(id)
            .ok_or_else(|| format!("semantic tensor {id:?} is not bound"))?;
        let (index, transform) = match (bound.transform, member) {
            (TensorTransform::StackExperts, Some(index)) => {
                let original = index;
                let index = if let TensorId::Layer {
                    index: layer_index, ..
                } = id
                {
                    let layer = self
                        .plan
                        .layers
                        .iter()
                        .chain(self.plan.mtp_blocks.iter().map(|b| &b.layer))
                        .find(|layer| layer.index == *layer_index)
                        .ok_or_else(|| format!("{id:?}: layer is outside the bound plan"))?;
                    if let crate::model_plan::MlpPlan::Moe(moe) = &layer.mlp
                        && moe.retained_experts.is_some()
                    {
                        let original = u32::try_from(original)
                            .map_err(|_| format!("{id:?}: expert ID overflows u32"))?;
                        moe.expert_bank_row(original).ok_or_else(|| {
                            format!("{id:?}: expert {original} is pruned or outside the router")
                        })?
                    } else {
                        index
                    }
                } else {
                    index
                };
                (index, TensorTransform::Identity)
            }
            (_, None) if bound.checkpoint_names.len() == 1 => (0, bound.transform),
            (TensorTransform::StackExperts, None) => {
                return Err(format!("{id:?}: expert bank requires an explicit member"));
            }
            (_, Some(_)) => {
                return Err(format!(
                    "{id:?}: member selection requires a StackExperts contract"
                ));
            }
            _ => {
                return Err(format!(
                    "{id:?}: unsupported multi-target transform {:?}",
                    bound.transform
                ));
            }
        };
        let name = bound
            .checkpoint_names
            .get(index)
            .ok_or_else(|| format!("{id:?}: member {index} out of bounds"))?;
        let record = &self.census.tensors[self.by_name[name]];
        Ok(BoundTensorRequest {
            id,
            record,
            dialect: self.census.dialect,
            transform,
            view: BoundTensorView::Whole,
        })
    }

    pub fn tensor(&self, id: &TensorId) -> Result<TensorView<'_>, String> {
        self.source.read_bound(&self.request(id, None)?)
    }
    /// Expert members use their original router IDs, including for retained compact banks.
    pub fn member_tensor(&self, id: &TensorId, member: usize) -> Result<TensorView<'_>, String> {
        self.source.read_bound(&self.request(id, Some(member))?)
    }
    pub fn nvfp4(&self, id: &TensorId) -> Result<Option<Nvfp4Native<'_>>, String> {
        self.source.nvfp4_bound(&self.request(id, None)?)
    }
    pub fn fp8(&self, id: &TensorId) -> Result<Option<Fp8Native<'_>>, String> {
        self.source.fp8_bound(&self.request(id, None)?)
    }
    pub fn fp8_stacked(&self, id: &TensorId) -> Result<Option<Fp8StackedNative<'_>>, String> {
        self.source.fp8_stacked_bound(&self.request(id, None)?)
    }
    pub fn nvfp4_stacked(&self, id: &TensorId) -> Result<Option<Nvfp4StackedNative<'_>>, String> {
        self.source.nvfp4_stacked_bound(&self.request(id, None)?)
    }
    pub fn disk(&self, id: &TensorId) -> Result<Option<BoundDiskView>, String> {
        self.source
            .disk_bound(&self.request(id, None)?)?
            .map(|extent| {
                self.disk_cache
                    .view(extent, std::sync::Arc::from(self.digest.as_str()))
            })
            .transpose()
    }
    pub fn auxiliary(
        &self,
        id: &TensorId,
        kind: QuantAuxTensor,
    ) -> Result<Option<TensorView<'_>>, String> {
        let owner = self.resolved_id(id).clone();
        let aux = TensorId::QuantAux {
            tensor: Box::new(owner),
            kind,
        };
        if self.binding.tensors.contains_key(&aux) {
            return self.tensor(&aux).map(Some);
        }
        self.source.auxiliary_bound(&self.request(id, None)?, kind)
    }
}

fn bind_retained_experts(
    plan: &mut ModelPlan,
    interpretation: &BoundSourceInterpretation,
) -> Result<BTreeMap<u32, Vec<bool>>, String> {
    let BoundSourceInterpretation::RetainedRepack { active_experts, .. } = interpretation else {
        return Ok(BTreeMap::new());
    };
    for (&index, mask) in active_experts {
        let layer = plan
            .layers
            .iter_mut()
            .chain(plan.mtp_blocks.iter_mut().map(|b| &mut b.layer))
            .find(|layer| layer.index == index)
            .ok_or_else(|| format!("retained expert layer {index} is outside the compiled plan"))?;
        let crate::model_plan::MlpPlan::Moe(moe) = &mut layer.mlp else {
            return Err(format!(
                "retained expert layer {index} has no routed MoE operation"
            ));
        };
        if mask.len() != moe.expert_count as usize {
            return Err(format!(
                "retained expert layer {index} mask width disagrees with the router"
            ));
        }
        moe.retained_experts = Some(
            mask.iter()
                .enumerate()
                .filter_map(|(id, &active)| active.then_some(id as u32))
                .collect(),
        );
        moe.validate_expert_set()
            .map_err(|e| format!("retained expert layer {index}: {e}"))?;
    }
    Ok(active_experts.clone())
}

#[allow(clippy::too_many_arguments)] // allow: the sealed compiler bundle's identity inputs stay explicit, including scope and source interpretation
fn binding_digest(
    config: &ModelConfig,
    plan: &ModelPlan,
    contract: &TensorContract,
    options: ContractOptions,
    binding: &BoundTensorContract,
    census: &TensorCensus,
    interpretation: &BoundSourceInterpretation,
    runtime_metadata: &crate::source::RuntimeSourceMetadata,
    scope: &CatalogScope,
) -> String {
    // Versioned canonical debug encoding, matching the plan serialization used by onboarding.
    // These types contain only ordered structs/enums/vectors/BTreeMaps, no pointers or HashMaps.
    // Length-prefix each section; include the contract as well as effective bindings (optional
    // absence and head ownership matter). Paths of shards and memory addresses never enter it.
    let mut hash = Sha256::new();
    hash.update(b"memra-bound-tensor-source-v4\0");
    for section in [
        format!("{config:?}"),
        format!("{plan:?}"),
        format!("{contract:?}"),
        format!("{options:?}"),
        format!("{binding:?}"),
        format!("{census:?}"),
        format!("{interpretation:?}"),
        format!("{runtime_metadata:?}"),
        format!("{scope:?}"),
    ] {
        hash.update((section.len() as u64).to_le_bytes());
        hash.update(section.as_bytes());
    }
    format!("{:x}", hash.finalize())
}

#[cfg(test)]
mod tests;

mod abi;
mod canonical_output;
pub mod consumer;
pub mod draft;
pub mod head_trim;
mod prepare;
pub mod ranks;
mod runtime;
pub(crate) use prepare::CompositeInput;
pub use prepare::PreparedModelSource;
pub use runtime::BoundRuntimeSource;
