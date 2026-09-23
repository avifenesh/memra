//! Ordered opened GGUF components for one external draft program. No model-root authority.
use super::*;
use crate::{GgufFile, source::GgufSource};
use std::path::Path;

/// Whole opened-source ingredients. These hashes cover complete components (including split
/// shards and shadowed data); they are separate from selected tensors and head/rank receipts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeDraftSourceIdentity {
    component_sha256: Vec<String>,
    opened_source_sha256: String,
}
impl CompositeDraftSourceIdentity {
    pub fn components(&self) -> &[String] {
        &self.component_sha256
    }
    pub fn opened_source_sha256(&self) -> &str {
        &self.opened_source_sha256
    }
}

/// Explicit priority order, highest first. Components use the existing standalone GGUF
/// dialect and must fit one common program/geometry. This is not a new manifest syntax.
pub struct CompositeExternalDraftInput {
    components: Vec<GgufFile>,
}
impl CompositeExternalDraftInput {
    pub fn open<P: AsRef<Path>>(paths: &[P]) -> Result<Self, String> {
        if !(2..=64).contains(&paths.len()) {
            return Err("composite external draft requires 2..=64 explicit GGUF components".into());
        }
        let components = paths
            .iter()
            .enumerate()
            .map(|(index, path)| {
                GgufFile::open(path).map_err(|e| format!("draft component {index}: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { components })
    }
    pub fn physical_inventory(&self) -> crate::source::PhysicalTensorInventory {
        crate::source::PhysicalTensorInventory {
            components: self
                .components
                .iter()
                .enumerate()
                .map(|(index, source)| crate::source::TensorComponentInventory {
                    component_path: vec![index as u32],
                    census: crate::source::census_from_gguf(source),
                })
                .collect(),
        }
    }
    pub fn source_identity(&self) -> Result<CompositeDraftSourceIdentity, String> {
        let components = self
            .components
            .iter()
            .map(|g| GgufSource(g).artifact_sha256())
            .collect::<Result<Vec<_>, _>>()?;
        let mut hash = Sha256::new();
        hash.update(b"memra-opened-composite-draft-v1\0");
        hash.update((components.len() as u64).to_le_bytes());
        for (index, digest) in components.iter().enumerate() {
            hash.update((index as u64).to_le_bytes());
            hash.update((digest.len() as u64).to_le_bytes());
            hash.update(digest.as_bytes());
        }
        Ok(CompositeDraftSourceIdentity {
            component_sha256: components,
            opened_source_sha256: format!("{:x}", hash.finalize()),
        })
    }
    pub fn with_runtime<T>(
        &self,
        target: &ModelConfig,
        load: impl FnOnce(&PreparedExternalDraftSource<'_>, &dyn TensorSource) -> T,
    ) -> Result<T, String> {
        let source = LayeredDraftSource::new(self)?;
        let mut prepared =
            PreparedExternalDraftSource::compile_with_inventory(&source, target, |contract| {
                source.validate_inventory(contract)
            })?;
        // Include every physical component's metadata and exact selected owner, not just the
        // flattened selected census. This changes no standalone draft identity.
        let mut hash = Sha256::new();
        hash.update(b"memra-composite-draft-binding-v1\0");
        for field in [
            prepared.bound.digest.clone(),
            format!("{:?}", source.censuses),
            format!("{:?}", source.owners),
        ] {
            hash.update((field.len() as u64).to_le_bytes());
            hash.update(field.as_bytes());
        }
        prepared.bound.digest = format!("{:x}", hash.finalize());
        prepared.with_runtime(|runtime| load(&prepared, runtime))
    }
}

struct LayeredDraftSource<'a> {
    input: &'a CompositeExternalDraftInput,
    sources: Vec<GgufSource<'a>>,
    config: ModelConfig,
    censuses: Vec<TensorCensus>,
    owners: BTreeMap<String, usize>,
    selected: TensorCensus,
}
impl<'a> LayeredDraftSource<'a> {
    fn new(input: &'a CompositeExternalDraftInput) -> Result<Self, String> {
        let configs = input
            .components
            .iter()
            .enumerate()
            .map(|(i, g)| {
                GgufSource(g)
                    .try_config()
                    .map_err(|e| format!("draft component {i}: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let config = configs[0].clone();
        for (i, cfg) in configs.iter().enumerate().skip(1) {
            if format!("{cfg:?}") != format!("{config:?}") {
                return Err(format!(
                    "draft component {i} does not declare the same complete normalized config"
                ));
            }
        }
        let mut censuses = input
            .components
            .iter()
            .map(crate::source::census_from_gguf)
            .collect::<Vec<_>>();
        for census in &mut censuses {
            census
                .tensors
                .sort_by(|a, b| a.entry.name.cmp(&b.entry.name));
        }
        let mut owners = BTreeMap::new();
        for (i, census) in censuses.iter().enumerate() {
            let mut names = BTreeSet::new();
            for row in &census.tensors {
                if !names.insert(&row.entry.name) {
                    return Err(format!(
                        "draft component {i} has duplicate tensor {}",
                        row.entry.name
                    ));
                }
                owners.entry(row.entry.name.clone()).or_insert(i);
            }
            for row in &census.tensors {
                if let Some(owner) = aux_owner(&row.entry.name)
                    && !names.contains(&owner)
                {
                    return Err(format!(
                        "draft component {i}: {} has no component-owned weight",
                        row.entry.name
                    ));
                }
            }
        }
        // A replacement weight cannot inherit a lower component's macro or activation scale.
        owners.retain(|name, component| {
            aux_owner(name).is_none_or(|owner| {
                censuses[*component]
                    .tensors
                    .iter()
                    .any(|r| r.entry.name == owner)
                    && input_owner(&censuses, &owner) == Some(*component)
            })
        });
        let first = config
            .n_layer
            .checked_sub(config.nextn_predict_layers)
            .ok_or("draft block count underflow")?;
        let head_names = [
            format!("blk.{first}.nextn.shared_head_head.weight"),
            format!("blk.{first}.nextn.shared_head.weight"),
            "output.weight".into(),
        ];
        let selected_head = head_names.iter().find_map(|name| owners.get(name).copied());
        // d2t is the selected head's row interpretation, not an independent lower fallback.
        for (i, census) in censuses.iter().enumerate() {
            if census.tensors.iter().any(|r| r.entry.name == "d2t")
                && !head_names
                    .iter()
                    .any(|name| census.tensors.iter().any(|r| &r.entry.name == name))
            {
                return Err(format!(
                    "draft component {i}: d2t has no component-owned draft head"
                ));
            }
        }
        owners.remove("d2t");
        if let Some(component) = selected_head
            && censuses[component]
                .tensors
                .iter()
                .any(|r| r.entry.name == "d2t")
        {
            owners.insert("d2t".into(), component);
        }
        let tensors = owners
            .iter()
            .map(|(name, &i)| {
                censuses[i]
                    .tensors
                    .iter()
                    .find(|r| &r.entry.name == name)
                    .unwrap()
                    .clone()
            })
            .collect();
        Ok(Self {
            input,
            sources: input.components.iter().map(GgufSource).collect(),
            config,
            censuses,
            owners,
            selected: TensorCensus {
                dialect: CheckpointDialect::Gguf,
                tensors,
            },
        })
    }
    fn validate_inventory(&self, contract: &TensorContract) -> Result<(), String> {
        for (index, census) in self.censuses.iter().enumerate() {
            let entries = census
                .tensors
                .iter()
                .map(|r| r.entry.clone())
                .collect::<Vec<_>>();
            let binding = contract
                .bind_fragment(&entries)
                .map_err(|e| format!("draft component {index}: {e}"))?;
            for (id, bound) in &binding.tensors {
                for name in &bound.checkpoint_names {
                    let record = census
                        .tensors
                        .iter()
                        .find(|r| &r.entry.name == name)
                        .unwrap();
                    GgufSource(&self.input.components[index]).validate_bound_metadata(
                        &BoundTensorRequest {
                            id,
                            record,
                            dialect: census.dialect,
                            transform: bound.transform,
                            view: BoundTensorView::Whole,
                        },
                    )?;
                }
            }
        }
        Ok(())
    }
    fn component(&self, request: &BoundTensorRequest<'_>) -> Result<&GgufSource<'a>, String> {
        let owner = *self
            .owners
            .get(&request.record.entry.name)
            .ok_or_else(|| request.error("no selected component owner"))?;
        let selected = self
            .selected
            .tensors
            .iter()
            .find(|r| r.entry.name == request.record.entry.name)
            .ok_or("selected draft row missing")?;
        if selected != request.record {
            return Err(request.error("request differs from selected composite record"));
        }
        Ok(&self.sources[owner])
    }
}
fn aux_owner(name: &str) -> Option<String> {
    [".input_scale", ".scale"].iter().find_map(|suffix| {
        name.strip_suffix(suffix)
            .map(|stem| format!("{stem}.weight"))
    })
}
fn input_owner(censuses: &[TensorCensus], name: &str) -> Option<usize> {
    censuses
        .iter()
        .position(|c| c.tensors.iter().any(|r| r.entry.name == name))
}
impl TensorSource for LayeredDraftSource<'_> {
    fn config(&self) -> ModelConfig {
        self.config.clone()
    }
    fn find(&self, _: &str) -> Option<TensorView<'_>> {
        panic!("composite external draft requires bound reads")
    }
    fn bound_interpretation(&self) -> Result<BoundSourceInterpretation, String> {
        Ok(BoundSourceInterpretation::Gguf)
    }
    fn tensor_census(&self) -> Result<TensorCensus, String> {
        Ok(self.selected.clone())
    }
    fn artifact_sha256(&self) -> Result<String, String> {
        Ok(self.input.source_identity()?.opened_source_sha256)
    }
    fn validate_bound_metadata(&self, r: &BoundTensorRequest<'_>) -> Result<(), String> {
        self.component(r)?.validate_bound_metadata(r)
    }
    fn read_bound(&self, r: &BoundTensorRequest<'_>) -> Result<TensorView<'_>, String> {
        self.component(r)?.read_bound(r)
    }
    fn auxiliary_bound(
        &self,
        r: &BoundTensorRequest<'_>,
        kind: QuantAuxTensor,
    ) -> Result<Option<TensorView<'_>>, String> {
        self.component(r)?.auxiliary_bound(r, kind)
    }
    fn nvfp4_bound(&self, r: &BoundTensorRequest<'_>) -> Result<Option<Nvfp4Native<'_>>, String> {
        self.component(r)?.nvfp4_bound(r)
    }
    fn fp8_bound(&self, r: &BoundTensorRequest<'_>) -> Result<Option<Fp8Native<'_>>, String> {
        self.component(r)?.fp8_bound(r)
    }
    fn fp8_stacked_bound(
        &self,
        r: &BoundTensorRequest<'_>,
    ) -> Result<Option<Fp8StackedNative<'_>>, String> {
        self.component(r)?.fp8_stacked_bound(r)
    }
    fn nvfp4_stacked_bound(
        &self,
        r: &BoundTensorRequest<'_>,
    ) -> Result<Option<Nvfp4StackedNative<'_>>, String> {
        self.component(r)?.nvfp4_stacked_bound(r)
    }
    fn disk_bound(&self, r: &BoundTensorRequest<'_>) -> Result<Option<DiskExtent>, String> {
        self.component(r)?.disk_bound(r)
    }
}
