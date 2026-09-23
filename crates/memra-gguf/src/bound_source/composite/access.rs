//! Explicit source-bound access. No TensorSource implementation or root-loader admission.
use super::*;
use crate::source::composite::OpenedComponent;
use std::sync::Arc;

pub struct BoundCompositeSource<'a> {
    pub(super) catalog: CompositeTensorCatalog,
    pub(super) components: Vec<OpenedComponent<'a>>,
    pub(super) config: ModelConfig,
    pub(super) semantic_sha256: String,
    pub(super) disk_cache: BoundDiskCache,
    pub(super) runtime_metadata: crate::source::RuntimeSourceMetadata,
}

impl<'a> BoundCompositeSource<'a> {
    pub fn canonical_nvfp4(
        &self,
        id: &TensorId,
        original: Option<u32>,
    ) -> Result<Option<BoundDiskView>, String> {
        let (component, request) = self.request(id, original, BoundTensorView::Whole)?;
        let Some(native) = self.components[component].nvfp4(&request)? else {
            return Ok(None);
        };
        let (out_f, in_f) = (native.out_f, native.in_f);
        drop(native);
        let root = self.components[component]
            .output_root(&request)?
            .ok_or_else(|| {
                request.error("component has no retained canonical output capability")
            })?;
        crate::bound_output::repack(
            root,
            1,
            out_f,
            in_f,
            Arc::from(self.semantic_sha256.as_str()),
            |_| {
                self.components[component].nvfp4(&request)?.ok_or_else(|| {
                    request.error("native operand disappeared during canonical output")
                })
            },
        )
        .map(Some)
    }

    pub fn canonical_nvfp4_bank(&self, id: &TensorId) -> Result<Option<BoundDiskView>, String> {
        let selection = self
            .catalog
            .selected
            .get(id)
            .ok_or_else(|| format!("{id:?}: absent or unselected composite bank"))?;
        if let Some(ids) = &selection.member_ids {
            let first = *ids.first().ok_or("canonical bank is empty")?;
            let (component, request) = self.request(id, Some(first), BoundTensorView::Whole)?;
            let Some(native) = self.components[component].nvfp4(&request)? else {
                return Ok(None);
            };
            let (out_f, in_f) = (native.out_f, native.in_f);
            drop(native);
            let root = self.components[component]
                .output_root(&request)?
                .ok_or_else(|| {
                    request.error("component has no retained canonical output capability")
                })?;
            crate::bound_output::repack(
                root,
                ids.len(),
                out_f,
                in_f,
                Arc::from(self.semantic_sha256.as_str()),
                |index| {
                    let (component, request) =
                        self.request(id, Some(ids[index]), BoundTensorView::Whole)?;
                    self.components[component]
                        .nvfp4(&request)?
                        .ok_or_else(|| request.error("bank member is not native NVFP4"))
                },
            )
            .map(Some)
        } else {
            let (component, request) = self.request(id, None, BoundTensorView::Whole)?;
            let Some(bank) = self.components[component].nvfp4_stacked(&request)? else {
                return Ok(None);
            };
            let root = self.components[component]
                .output_root(&request)?
                .ok_or_else(|| {
                    request.error("component has no retained canonical output capability")
                })?;
            crate::bound_output::repack_stacked(
                root,
                &bank,
                Arc::from(self.semantic_sha256.as_str()),
            )
            .map(Some)
        }
    }

    pub fn compile(source: &'a Hy3RepackSource, scope: LoadScope) -> Result<Self, String> {
        let catalog = CompositeTensorCatalog::compile(source, scope)?;
        let components = source.binding_components();
        // Validate every auxiliary value before any optional native representation is chosen,
        // including planes in shadowed and unselected components. No model-sized conversion.
        for (index, component) in catalog.components.iter().enumerate() {
            for (id, bound) in &component.binding.tensors {
                for name in &bound.checkpoint_names {
                    components[index].validate_values(&BoundTensorRequest {
                        id,
                        record: &component.census.tensors[component.by_name[name]],
                        dialect: component.census.dialect,
                        transform: bound.transform,
                        view: BoundTensorView::Whole,
                    })?;
                }
            }
        }
        let mut result = Self {
            config: catalog.config.clone(),
            catalog,
            components,
            semantic_sha256: String::new(),
            disk_cache: BoundDiskCache::default(),
            runtime_metadata: crate::source::RuntimeSourceMetadata {
                is_gguf: false,
                is_safetensors: false,
                expert_activation_precision: source.expert_activation_precision(),
                preserve_expert_encodings: true,
                nvfp4_cache_tag: source.nvfp4_cache_tag(),
            },
        };
        // Capture factors through the selected physical binding first. A failed read cannot
        // become absence in the legacy Option-shaped preflight interface.
        let factors = if result.catalog.selected.contains_key(&TensorId::RopeFactors) {
            Some(result.tensor(&TensorId::RopeFactors)?)
        } else {
            None
        };
        let mut config = result.config.clone();
        model_packs::prepare_bound_rope_factors(
            &mut config,
            &result.catalog.plan,
            &FactorSource {
                config: &result.config,
                factors,
            },
        )
        .map_err(|error| error.to_string())?;
        result.config = config;
        let mut hash = Sha256::new();
        hash.update(b"memra-composite-source-program-v1\0");
        for bytes in [
            result.catalog.digest.as_bytes(),
            format!("{:?}", result.config).as_bytes(),
        ] {
            hash.update((bytes.len() as u64).to_le_bytes());
            hash.update(bytes);
        }
        result.semantic_sha256 = format!("{:x}", hash.finalize());
        Ok(result)
    }

    pub fn catalog(&self) -> &CompositeTensorCatalog {
        &self.catalog
    }
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    pub fn plan(&self) -> &ModelPlan {
        &self.catalog.plan
    }
    /// Compiled interpretation only; not the complete opened artifact identity.
    pub(crate) fn binding_sha256(&self) -> &str {
        &self.semantic_sha256
    }
    pub fn active_experts(&self, layer: u32) -> Option<&[bool]> {
        self.catalog.active_experts.get(&layer).map(Vec::as_slice)
    }

    pub(super) fn request<'b>(
        &'b self,
        id: &TensorId,
        original: Option<u32>,
        view: BoundTensorView,
    ) -> Result<(usize, BoundTensorRequest<'b>), String> {
        let id = if *id == TensorId::OutputProjection
            && model_packs::for_config(&self.config)
                .ok_or("bound model pack disappeared")?
                .contract_options(&self.config)
                .output_head
                == OutputHead::TiedToEmbedding
        {
            &TensorId::TokenEmbedding
        } else {
            id
        };
        let (id, selection) = self.catalog.selected.get_key_value(id).ok_or_else(|| {
            format!("tensor {id:?} is absent or outside selected composite scope")
        })?;
        let (index, transform) = match (&selection.member_ids, original) {
            (Some(ids), Some(original)) => {
                let index = ids.iter().position(|&i| i == original).ok_or_else(|| {
                    format!("{id:?}: expert {original} is pruned or outside the router")
                })?;
                (index, TensorTransform::Identity)
            }
            (Some(_), None) => {
                return Err(format!(
                    "{id:?}: group requires an explicit original member ID"
                ));
            }
            (None, Some(_)) => {
                return Err(format!(
                    "{id:?}: whole-bank slicing requires a separate explicit contract"
                ));
            }
            (None, None) => (0, selection.members[0].transform),
        };
        let member = &selection.members[index];
        let component = member.component_path.len();
        if self.catalog.components[component].component_path != member.component_path {
            return Err(format!("{id:?}: bound component path is inconsistent"));
        }
        Ok((
            component,
            BoundTensorRequest {
                id,
                record: &member.record,
                dialect: member.dialect,
                transform,
                view,
            },
        ))
    }

    pub fn tensor(&self, id: &TensorId) -> Result<TensorView<'_>, String> {
        self.tensor_view(id, BoundTensorView::Whole)
    }
    pub fn tensor_view(
        &self,
        id: &TensorId,
        view: BoundTensorView,
    ) -> Result<TensorView<'_>, String> {
        let (component, request) = self.request(id, None, view)?;
        self.components[component].read(&request)
    }
    pub fn member_tensor(&self, id: &TensorId, original: u32) -> Result<TensorView<'_>, String> {
        let (component, request) = self.request(id, Some(original), BoundTensorView::Whole)?;
        self.components[component].read(&request)
    }
    pub fn fp8(
        &self,
        id: &TensorId,
        original: Option<u32>,
    ) -> Result<Option<Fp8Native<'_>>, String> {
        let (component, request) = self.request(id, original, BoundTensorView::Whole)?;
        self.components[component].fp8(&request)
    }
    pub fn nvfp4(
        &self,
        id: &TensorId,
        original: Option<u32>,
    ) -> Result<Option<Nvfp4Native<'_>>, String> {
        let (component, request) = self.request(id, original, BoundTensorView::Whole)?;
        self.components[component].nvfp4(&request)
    }
    pub fn fp8_stacked(&self, id: &TensorId) -> Result<Option<Fp8StackedNative<'_>>, String> {
        let (component, request) = self.request(id, None, BoundTensorView::Whole)?;
        self.components[component].fp8_stacked(&request)
    }
    pub fn nvfp4_stacked(&self, id: &TensorId) -> Result<Option<Nvfp4StackedNative<'_>>, String> {
        let (component, request) = self.request(id, None, BoundTensorView::Whole)?;
        self.components[component].nvfp4_stacked(&request)
    }
    pub fn auxiliary(
        &self,
        id: &TensorId,
        original: Option<u32>,
        kind: QuantAuxTensor,
    ) -> Result<Option<TensorView<'_>>, String> {
        let id = if *id == TensorId::OutputProjection
            && model_packs::for_config(&self.config)
                .ok_or("bound model pack disappeared")?
                .contract_options(&self.config)
                .output_head
                == OutputHead::TiedToEmbedding
        {
            &TensorId::TokenEmbedding
        } else {
            id
        };
        let aux = TensorId::QuantAux {
            tensor: Box::new(id.clone()),
            kind,
        };
        if original.is_none() && self.catalog.selected.contains_key(&aux) {
            return self.tensor(&aux).map(Some);
        }
        let (component, request) = self.request(id, original, BoundTensorView::Whole)?;
        self.components[component].auxiliary(&request, kind)
    }
    pub fn disk(
        &self,
        id: &TensorId,
        original: Option<u32>,
    ) -> Result<Option<BoundDiskView>, String> {
        let (component, request) = self.request(id, original, BoundTensorView::Whole)?;
        self.components[component]
            .disk(&request)?
            .map(|extent| {
                self.disk_cache
                    .view(extent, Arc::from(self.semantic_sha256.as_str()))
            })
            .transpose()
    }

    /// Hash every opened component, including unused rows and shard padding, without reopening
    /// any path. Component order and manifest-to-shard assignment remain part of identity.
    pub fn artifact_identity(&self) -> Result<BoundArtifactIdentity, String> {
        let mut hash = Sha256::new();
        hash.update(b"memra-opened-composite-artifact-v1\0");
        hash.update((self.components.len() as u64).to_le_bytes());
        for (index, component) in self.components.iter().enumerate() {
            let digest = component.artifact_sha256()?;
            if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(format!(
                    "component {index} returned an invalid source SHA-256"
                ));
            }
            hash.update((index as u64).to_le_bytes());
            hash.update(digest.as_bytes());
        }
        let opened_source_sha256 = format!("{:x}", hash.finalize());
        let semantic_scope_sha256 = self.semantic_sha256.clone();
        let mut hash = Sha256::new();
        hash.update(b"memra-bound-composite-artifact-v1\0");
        hash.update(opened_source_sha256.as_bytes());
        hash.update(semantic_scope_sha256.as_bytes());
        Ok(BoundArtifactIdentity {
            opened_source_sha256,
            semantic_scope_sha256,
            artifact_sha256: format!("{:x}", hash.finalize()),
        })
    }
}

struct FactorSource<'a> {
    config: &'a ModelConfig,
    factors: Option<TensorView<'a>>,
}
impl TensorSource for FactorSource<'_> {
    fn config(&self) -> ModelConfig {
        self.config.clone()
    }
    fn find(&self, name: &str) -> Option<TensorView<'_>> {
        if name != "rope_freqs.weight" {
            return None;
        }
        self.factors.as_ref().map(|view| TensorView {
            bytes: std::borrow::Cow::Borrowed(&view.bytes),
            ggml_type: view.ggml_type,
            ne: view.ne.clone(),
        })
    }
}
