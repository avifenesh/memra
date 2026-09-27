//! Engine ABI adapter over the exact composite selection, with no legacy source fallback.
use super::super::abi::{Access, RuntimeAbi, RuntimeSemantics};
use super::*;
use crate::source::{ExpertActivationPrecision, PhysicalTensorInventory, TensorComponentInventory};

pub struct CompositeRuntimeSource<'a, 'source> {
    bound: &'a BoundCompositeSource<'source>,
    abi: RuntimeAbi,
}

impl RuntimeSemantics for BoundCompositeSource<'_> {
    fn plan(&self) -> &ModelPlan {
        self.plan()
    }
    fn contains(&self, id: &TensorId) -> bool {
        let id = resolved(self, id);
        self.catalog
            .components
            .iter()
            .any(|c| c.binding.tensors.contains_key(id))
    }
    fn transform(&self, id: &TensorId) -> Option<TensorTransform> {
        let selection = self.catalog.selected.get(resolved(self, id))?;
        if selection.member_ids.is_some() {
            Some(TensorTransform::StackExperts)
        } else {
            selection.members.first().map(|m| m.transform)
        }
    }
}

fn resolved<'a>(bound: &BoundCompositeSource<'_>, id: &'a TensorId) -> &'a TensorId {
    if *id == TensorId::OutputProjection
        && model_packs::for_config(bound.config())
            .unwrap()
            .contract_options(bound.config())
            .output_head
            == OutputHead::TiedToEmbedding
    {
        &TensorId::TokenEmbedding
    } else {
        id
    }
}

impl<'source> BoundCompositeSource<'source> {
    pub fn runtime(&self) -> Result<CompositeRuntimeSource<'_, 'source>, String> {
        let options = model_packs::for_config(self.config())
            .ok_or("bound model pack disappeared")?
            .contract_options(self.config());
        Ok(CompositeRuntimeSource {
            bound: self,
            abi: RuntimeAbi::new(self.plan(), options)?,
        })
    }
}

impl CompositeRuntimeSource<'_, '_> {
    fn access(&self, name: &str) -> Result<Access, String> {
        self.abi.access(self.bound, name)
    }
    fn request(&self, access: &Access) -> Result<Option<(usize, BoundTensorRequest<'_>)>, String> {
        let (id, member, view) = match access {
            Access::Tensor(id) => (id, None, BoundTensorView::Whole),
            Access::Member(id, member) => (
                id,
                Some(u32::try_from(*member).map_err(|_| "expert ID overflows u32")?),
                BoundTensorView::Whole,
            ),
            Access::Derived(id, view) => (id, None, *view),
            Access::Auxiliary(..) => {
                return Err("auxiliary is not an independent native operand".into());
            }
        };
        if !self.bound.contains(id) {
            return Ok(None);
        }
        let id = resolved(self.bound, id);
        let selection = self
            .bound
            .catalog
            .selected
            .get(id)
            .ok_or_else(|| format!("{id:?} is outside selected composite scope"))?;
        if member.is_none() && selection.member_ids.is_some() {
            return Ok(None);
        }
        self.bound.request(id, member, view).map(Some)
    }
    fn auxiliary(
        &self,
        access: &Access,
        kind: QuantAuxTensor,
    ) -> Result<Option<TensorView<'_>>, String> {
        let (id, member) = match access {
            Access::Tensor(id) | Access::Derived(id, _) => (id, None),
            Access::Member(id, index) => (
                id,
                Some(u32::try_from(*index).map_err(|_| "expert ID overflows u32")?),
            ),
            Access::Auxiliary(..) => return Err("nested auxiliary requests are unsupported".into()),
        };
        if !self.bound.contains(id) {
            return Ok(None);
        }
        // Group-wide optional scale probes are absent for separate members; their own scales
        // are resolved with original member IDs by the gather path.
        if member.is_none()
            && self
                .bound
                .catalog
                .selected
                .get(resolved(self.bound, id))
                .is_some_and(|s| s.member_ids.is_some())
        {
            return Ok(None);
        }
        self.bound.auxiliary(id, member, kind)
    }
    fn read(&self, access: &Access) -> Result<Option<TensorView<'_>>, String> {
        if let Access::Auxiliary(owner, kind) = access {
            return self.auxiliary(owner, *kind);
        }
        let Some((component, request)) = self.request(access)? else {
            return Ok(None);
        };
        self.bound.components[component].read(&request).map(Some)
    }
    fn disk(&self, access: &Access) -> Result<Option<crate::bound_disk::ExpertDiskView>, String> {
        if matches!(access, Access::Auxiliary(..)) {
            self.read(access)?;
            return Ok(None);
        }
        let Some((component, request)) = self.request(access)? else {
            return Ok(None);
        };
        self.bound.components[component]
            .disk(&request)?
            .map(|extent| {
                self.bound
                    .disk_cache
                    .view(
                        extent,
                        std::sync::Arc::from(self.bound.semantic_sha256.as_str()),
                    )
                    .map(crate::bound_disk::ExpertDiskView::Bound)
            })
            .transpose()
    }
}

macro_rules! native_reader {
    ($method:ident,$kind:ident,$delegate:ident) => {
        fn $method(&self, name: &str) -> Result<Option<$kind<'_>>, String> {
            let access = self.access(name)?;
            if matches!(access, Access::Auxiliary(..)) {
                self.read(&access)?;
                return Ok(None);
            }
            let Some((component, request)) = self.request(&access)? else {
                return Ok(None);
            };
            self.bound.components[component].$delegate(&request)
        }
    };
}

impl TensorSource for CompositeRuntimeSource<'_, '_> {
    fn bound_tensor_charges(
        &self,
    ) -> Result<Option<Vec<crate::source::BoundTensorCharge>>, String> {
        Ok(Some(self.bound.catalog.charges.clone()))
    }
    fn config(&self) -> ModelConfig {
        self.bound.config.clone()
    }
    #[allow(private_interfaces)] // allow: only the compiler-owned adapter can return this preflight authority
    fn bound_program(&self) -> Option<BoundProgramRef<'_>> {
        Some(BoundProgramRef::Composite(self.bound))
    }
    fn runtime_metadata(&self) -> Result<crate::source::RuntimeSourceMetadata, String> {
        Ok(self.bound.runtime_metadata)
    }
    fn physical_tensor_inventory(&self) -> Result<PhysicalTensorInventory, String> {
        Ok(PhysicalTensorInventory {
            components: self
                .bound
                .catalog
                .components
                .iter()
                .map(|c| TensorComponentInventory {
                    component_path: c.component_path.clone(),
                    census: c.census.clone(),
                })
                .collect(),
        })
    }
    fn tensor_census(&self) -> Result<TensorCensus, String> {
        Err(
            "composite runtime has component inventories, not one flattened checkpoint dialect"
                .into(),
        )
    }
    fn gguf_tensor_metadata(
        &self,
    ) -> Result<Option<Vec<crate::source::GgufTensorMetadata>>, String> {
        Ok(None)
    }
    fn artifact_sha256(&self) -> Result<String, String> {
        Ok(self.bound.artifact_identity()?.artifact_sha256)
    }
    fn find(&self, _: &str) -> Option<TensorView<'_>> {
        panic!("bound runtime access requires try_find")
    }
    fn has(&self, _: &str) -> bool {
        panic!("bound runtime access requires try_has")
    }
    fn gguf(&self) -> Option<&crate::GgufFile> {
        panic!("composite runtime never exports a source file")
    }
    fn st_dir(&self) -> Option<&std::path::Path> {
        self.bound.components.first().and_then(|c| c.st_dir())
    }
    fn expert_activation_precision(&self) -> ExpertActivationPrecision {
        self.bound.runtime_metadata.expert_activation_precision
    }
    fn preserve_expert_encodings(&self) -> bool {
        true
    }
    fn nvfp4_cache_tag(&self) -> &'static str {
        self.bound.runtime_metadata.nvfp4_cache_tag
    }
    fn active_experts(&self, layer: u32) -> Option<&[bool]> {
        self.bound.active_experts(layer)
    }
    fn try_find(&self, name: &str) -> Result<Option<TensorView<'_>>, String> {
        self.read(&self.access(name)?)
    }
    fn try_has(&self, name: &str) -> Result<bool, String> {
        let access = self.access(name)?;
        if let Access::Auxiliary(owner, kind) = &access {
            return self.auxiliary(owner, *kind).map(|v| v.is_some());
        }
        Ok(self.request(&access)?.is_some())
    }
    fn try_find_expert_disk(
        &self,
        name: &str,
    ) -> Result<Option<crate::bound_disk::ExpertDiskView>, String> {
        self.disk(&self.access(name)?)
    }
    fn try_find_gguf_disk(
        &self,
        name: &str,
    ) -> Result<Option<crate::bound_disk::ExpertDiskView>, String> {
        let access = self.access(name)?;
        if matches!(access, Access::Auxiliary(..)) {
            self.read(&access)?;
        } else {
            self.request(&access)?;
        }
        Ok(None)
    }
    fn try_has_expert_mmap(&self, name: &str) -> Result<bool, String> {
        self.try_find_expert_disk(name).map(|v| v.is_some())
    }
    fn find_expert_mmap(&self, _: &str) -> Option<(std::sync::Arc<memmap2::Mmap>, usize, usize)> {
        panic!("bound runtime never exports mmap handles")
    }
    fn find_expert_disk(&self, _: &str) -> Option<DiskExtent> {
        panic!("bound runtime access requires try_find_expert_disk")
    }
    fn find_nvfp4_native(&self, _: &str) -> Option<Nvfp4Native<'_>> {
        panic!("bound runtime access requires try_find_nvfp4_native")
    }
    fn find_fp8_native(&self, _: &str) -> Option<Fp8Native<'_>> {
        panic!("bound runtime access requires try_find_fp8_native")
    }
    fn find_fp8_stacked_native(&self, _: &str) -> Option<Fp8StackedNative<'_>> {
        panic!("bound runtime access requires try_find_fp8_stacked_native")
    }
    fn find_nvfp4_stacked_native(&self, _: &str) -> Option<Nvfp4StackedNative<'_>> {
        panic!("bound runtime access requires try_find_nvfp4_stacked_native")
    }
    native_reader!(try_find_nvfp4_native, Nvfp4Native, nvfp4);
    native_reader!(try_find_fp8_native, Fp8Native, fp8);
    native_reader!(try_find_fp8_stacked_native, Fp8StackedNative, fp8_stacked);
    native_reader!(
        try_find_nvfp4_stacked_native,
        Nvfp4StackedNative,
        nvfp4_stacked
    );
    fn try_canonical_nvfp4_bank(&self, name: &str) -> Result<Option<BoundDiskView>, String> {
        let view = match self.access(name)? {
            Access::Tensor(id) | Access::Derived(id, _) => self.bound.canonical_nvfp4_bank(&id),
            Access::Member(id, index) => self.bound.canonical_nvfp4(
                &id,
                Some(u32::try_from(index).map_err(|_| "expert ID overflows u32")?),
            ),
            Access::Auxiliary(..) => Err("a scale is not an independent canonical bank".into()),
        }?;
        view.map(Some)
            .ok_or_else(|| format!("bound bank {name} has no canonical native NVFP4 producer"))
    }
}
