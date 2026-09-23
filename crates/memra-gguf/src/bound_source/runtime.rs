//! Adapter for the engine's ggml ABI. Names select semantic IDs, never checkpoint names.
use super::abi::{Access, RuntimeAbi, RuntimeSemantics};
use super::*;
use crate::source::ExpertActivationPrecision;

pub struct BoundRuntimeSource<'a, 'source> {
    bound: &'a BoundTensorSource<'source>,
    abi: RuntimeAbi,
    external_draft: bool,
}

impl RuntimeSemantics for BoundTensorSource<'_> {
    fn plan(&self) -> &ModelPlan {
        &self.plan
    }
    fn contains(&self, id: &TensorId) -> bool {
        BoundTensorSource::contains(self, id)
    }
    fn transform(&self, id: &TensorId) -> Option<TensorTransform> {
        self.binding.tensors.get(id).map(|b| b.transform)
    }
}
impl<'source> BoundTensorSource<'source> {
    pub fn runtime(&self) -> Result<BoundRuntimeSource<'_, 'source>, String> {
        Ok(BoundRuntimeSource {
            bound: self,
            abi: RuntimeAbi::new(&self.plan, self.options)?,
            external_draft: false,
        })
    }

    pub(super) fn external_draft_runtime(&self) -> Result<BoundRuntimeSource<'_, 'source>, String> {
        Ok(BoundRuntimeSource {
            bound: self,
            abi: RuntimeAbi::external_draft(&self.plan, self.options, &self.contract)?,
            external_draft: true,
        })
    }
}

impl BoundRuntimeSource<'_, '_> {
    fn access(&self, name: &str) -> Result<Access, String> {
        self.abi.access(self.bound, name)
    }

    fn request<'a>(&'a self, access: &'a Access) -> Result<Option<BoundTensorRequest<'a>>, String> {
        let (id, member, view) = match access {
            Access::Tensor(id) => (id, None, BoundTensorView::Whole),
            Access::Member(id, n) => (id, Some(*n), BoundTensorView::Whole),
            Access::Derived(id, view) => (id, None, *view),
            Access::Auxiliary(..) => {
                return Err("auxiliary is not an independent native operand".into());
            }
        };
        if !self.bound.contains(id) {
            return Ok(None);
        }
        let id = self.bound.resolved_id(id);
        let binding = &self.bound.binding.tensors[id];
        if member.is_none() && binding.transform == TensorTransform::StackExperts {
            return Ok(None); // separate members, not a physical stacked bank
        }
        let mut r = self.bound.request(id, member)?;
        r.view = view;
        Ok(Some(r))
    }

    fn read(&self, access: &Access) -> Result<Option<TensorView<'_>>, String> {
        if let Access::Auxiliary(owner, kind) = access {
            return self.auxiliary(owner, *kind);
        }
        let Some(r) = self.request(access)? else {
            return Ok(None);
        };
        self.bound.source.read_bound(&r).map(Some)
    }
    fn auxiliary(
        &self,
        access: &Access,
        kind: QuantAuxTensor,
    ) -> Result<Option<TensorView<'_>>, String> {
        if let Access::Tensor(id) | Access::Derived(id, _) = access {
            if !self.bound.contains(id) {
                return Ok(None);
            };
            return self.bound.auxiliary(id, kind);
        }
        let Some(r) = self.request(access)? else {
            return Ok(None);
        };
        self.bound.source.auxiliary_bound(&r, kind)
    }
    fn present(&self, access: &Access) -> Result<bool, String> {
        if let Access::Auxiliary(owner, kind) = access {
            if let Access::Tensor(id) | Access::Derived(id, _) = owner.as_ref() {
                let aux = TensorId::QuantAux {
                    tensor: Box::new(self.bound.resolved_id(id).clone()),
                    kind: *kind,
                };
                if self.bound.binding.tensors.contains_key(&aux) {
                    return Ok(true);
                }
            }
            let Some(r) = self.request(owner)? else {
                return Ok(false);
            };
            let suffixes: &[&str] = match kind {
                QuantAuxTensor::WeightScale => &[".weight_scale_2", ".weight_global_scale"],
                QuantAuxTensor::InputScale => &[".input_scale"],
                QuantAuxTensor::PreQuantScale => &[".pre_quant_scale"],
            };
            return Ok(r
                .record
                .auxiliaries
                .iter()
                .any(|a| suffixes.iter().any(|s| a.physical_name.ends_with(s))));
        }
        Ok(self.request(access)?.is_some())
    }
}

impl TensorSource for BoundRuntimeSource<'_, '_> {
    fn bound_tensor_charges(
        &self,
    ) -> Result<Option<Vec<crate::source::BoundTensorCharge>>, String> {
        Ok(Some(
            self.bound
                .binding
                .tensors
                .iter()
                .map(|(id, tensor)| crate::source::BoundTensorCharge {
                    id: id.clone(),
                    owner: tensor.owner,
                    physical_bytes: tensor.physical_bytes,
                    execution_selected: self.bound.scope.permits(id),
                })
                .collect(),
        ))
    }
    fn try_canonical_nvfp4_bank(
        &self,
        name: &str,
    ) -> Result<Option<crate::bound_disk::BoundDiskView>, String> {
        let view = match self.access(name)? {
            Access::Tensor(id) | Access::Derived(id, _) => {
                if !self.bound.contains(&id) {
                    return Err(format!(
                        "canonical output requested for missing bound bank {name}"
                    ));
                }
                self.bound.canonical_nvfp4_bank(&id)
            }
            Access::Member(id, member) => self.bound.canonical_nvfp4(&id, Some(member)),
            Access::Auxiliary(..) => Err("a scale is not an independent canonical bank".into()),
        }?;
        view.map(Some)
            .ok_or_else(|| format!("bound bank {name} has no canonical native NVFP4 producer"))
    }
    fn raw_config_json(&self) -> Option<&str> {
        self.bound.source.raw_config_json()
    }
    fn bound_interpretation(&self) -> Result<BoundSourceInterpretation, String> {
        Ok(self.bound.interpretation.clone())
    }

    fn artifact_sha256(&self) -> Result<String, String> {
        Ok(self.bound.artifact_identity()?.artifact_sha256)
    }
    fn runtime_metadata(&self) -> Result<crate::source::RuntimeSourceMetadata, String> {
        Ok(self.bound.runtime_metadata)
    }
    #[allow(private_interfaces)] // allow: only this compiler-owned adapter can issue the private preflight authority
    fn bound_program(&self) -> Option<BoundProgramRef<'_>> {
        Some(if self.external_draft {
            BoundProgramRef::ExternalDraft(self.bound)
        } else {
            BoundProgramRef::Single(self.bound)
        })
    }
    fn gguf_tensor_metadata(
        &self,
    ) -> Result<Option<Vec<crate::source::GgufTensorMetadata>>, String> {
        if !self.bound.runtime_metadata.is_gguf {
            return Ok(None);
        }
        let mut names = std::collections::BTreeSet::new();
        for (id, tensor) in &self.bound.binding.tensors {
            if self.bound.scope.permits(id) {
                names.extend(tensor.checkpoint_names.iter());
            }
        }
        let mut metadata = Vec::with_capacity(names.len());
        for name in names {
            let index = self
                .bound
                .by_name
                .get(name)
                .ok_or_else(|| format!("bound GGUF metadata target {name} is missing"))?;
            let row = &self.bound.census.tensors[*index];
            metadata.push(crate::source::GgufTensorMetadata {
                name: row.physical_name.clone(),
                physical_bytes: row.entry.physical_bytes,
            });
        }
        Ok(Some(metadata))
    }
    fn config(&self) -> ModelConfig {
        self.bound.config.clone()
    }
    fn tensor_census(&self) -> Result<TensorCensus, String> {
        Ok(self.bound.census.clone())
    }
    fn find(&self, _: &str) -> Option<TensorView<'_>> {
        panic!("bound runtime access requires try_find")
    }
    fn has(&self, _: &str) -> bool {
        panic!("bound runtime access requires try_has")
    }
    fn try_find(&self, name: &str) -> Result<Option<TensorView<'_>>, String> {
        self.read(&self.access(name)?)
    }
    fn try_has(&self, name: &str) -> Result<bool, String> {
        self.present(&self.access(name)?)
    }
    fn gguf(&self) -> Option<&crate::GgufFile> {
        panic!(
            "bound runtime does not expose unrestricted source files; use authorized metadata/extent APIs"
        )
    }
    fn st_dir(&self) -> Option<&std::path::Path> {
        self.bound.source.st_dir()
    }
    fn expert_activation_precision(&self) -> ExpertActivationPrecision {
        self.bound.runtime_metadata.expert_activation_precision
    }
    fn nvfp4_cache_tag(&self) -> &'static str {
        self.bound.runtime_metadata.nvfp4_cache_tag
    }
    fn preserve_expert_encodings(&self) -> bool {
        self.bound.runtime_metadata.preserve_expert_encodings
    }
    fn active_experts(&self, layer: u32) -> Option<&[bool]> {
        self.bound.active_experts.get(&layer).map(Vec::as_slice)
    }
    fn find_expert_mmap(&self, _: &str) -> Option<(std::sync::Arc<memmap2::Mmap>, usize, usize)> {
        panic!("bound runtime never exports unrestricted mmap handles")
    }
    fn try_has_expert_mmap(&self, name: &str) -> Result<bool, String> {
        Ok(self.try_find_expert_disk(name)?.is_some())
    }
    fn bound_gguf_name<'a>(&'a self, name: &'a str) -> Result<&'a str, String> {
        let access = self.access(name)?;
        let Some(r) = self.request(&access)? else {
            return Err(format!("missing bound GGUF target {name}"));
        };
        if r.dialect != CheckpointDialect::Gguf
            || r.transform != TensorTransform::Identity
            || r.view != BoundTensorView::Whole
        {
            return Err(r.error("raw GGUF access requires an identity physical binding"));
        }
        // The name belongs to the immutable census, not the temporary access descriptor.
        Ok(&self.bound.census.tensors[self.bound.by_name[&r.record.entry.name]].physical_name)
    }
    fn find_nvfp4_native(&self, _: &str) -> Option<Nvfp4Native<'_>> {
        panic!("bound runtime access requires try_find_nvfp4_native")
    }
    fn try_find_nvfp4_native(&self, name: &str) -> Result<Option<Nvfp4Native<'_>>, String> {
        let access = self.access(name)?;
        if matches!(access, Access::Auxiliary(..)) {
            self.read(&access)?;
            return Ok(None);
        }
        let Some(r) = self.request(&access)? else {
            return Ok(None);
        };
        self.bound.source.nvfp4_bound(&r)
    }
    fn find_fp8_native(&self, _: &str) -> Option<Fp8Native<'_>> {
        panic!("bound runtime access requires try_find_fp8_native")
    }
    fn try_find_fp8_native(&self, name: &str) -> Result<Option<Fp8Native<'_>>, String> {
        let access = self.access(name)?;
        if matches!(access, Access::Auxiliary(..)) {
            self.read(&access)?;
            return Ok(None);
        }
        let Some(r) = self.request(&access)? else {
            return Ok(None);
        };
        self.bound.source.fp8_bound(&r)
    }
    fn find_fp8_stacked_native(&self, _: &str) -> Option<Fp8StackedNative<'_>> {
        panic!("bound runtime access requires try_find_fp8_stacked_native")
    }
    fn try_find_fp8_stacked_native(
        &self,
        name: &str,
    ) -> Result<Option<Fp8StackedNative<'_>>, String> {
        let access = self.access(name)?;
        if matches!(access, Access::Auxiliary(..)) {
            self.read(&access)?;
            return Ok(None);
        }
        let Some(r) = self.request(&access)? else {
            return Ok(None);
        };
        self.bound.source.fp8_stacked_bound(&r)
    }
    fn find_nvfp4_stacked_native(&self, _: &str) -> Option<Nvfp4StackedNative<'_>> {
        panic!("bound runtime access requires try_find_nvfp4_stacked_native")
    }
    fn try_find_nvfp4_stacked_native(
        &self,
        name: &str,
    ) -> Result<Option<Nvfp4StackedNative<'_>>, String> {
        let access = self.access(name)?;
        if matches!(access, Access::Auxiliary(..)) {
            self.read(&access)?;
            return Ok(None);
        }
        let Some(r) = self.request(&access)? else {
            return Ok(None);
        };
        self.bound.source.nvfp4_stacked_bound(&r)
    }
    fn try_find_gguf_disk(
        &self,
        name: &str,
    ) -> Result<Option<crate::bound_disk::ExpertDiskView>, String> {
        let access = self.access(name)?;
        if matches!(access, Access::Auxiliary(..)) {
            self.read(&access)?;
            return Ok(None);
        }
        let Some(r) = self.request(&access)? else {
            return Ok(None);
        };
        if !self.bound.runtime_metadata.is_gguf {
            return Ok(None);
        }
        self.bound
            .source
            .disk_bound(&r)?
            .map(|extent| {
                self.bound
                    .disk_cache
                    .view(extent, std::sync::Arc::from(self.bound.digest.as_str()))
                    .map(crate::bound_disk::ExpertDiskView::Bound)
            })
            .transpose()
    }
    fn find_expert_disk(&self, _: &str) -> Option<DiskExtent> {
        panic!("bound runtime access requires try_find_expert_disk")
    }
    fn try_find_expert_disk(
        &self,
        name: &str,
    ) -> Result<Option<crate::bound_disk::ExpertDiskView>, String> {
        let access = self.access(name)?;
        if matches!(access, Access::Auxiliary(..)) {
            self.read(&access)?;
            return Ok(None);
        }
        let Some(r) = self.request(&access)? else {
            return Ok(None);
        };
        if self.bound.runtime_metadata.is_gguf {
            // Authorize and validate the requested physical view, but do not interpret available
            // disk backing as an instruction to replace the ordinary pinned/pageable load.
            self.bound.source.validate_bound_metadata(&r)?;
            return Ok(None);
        }
        self.bound
            .source
            .disk_bound(&r)?
            .map(|extent| {
                self.bound
                    .disk_cache
                    .view(extent, std::sync::Arc::from(self.bound.digest.as_str()))
                    .map(crate::bound_disk::ExpertDiskView::Bound)
            })
            .transpose()
    }
}
