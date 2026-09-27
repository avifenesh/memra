//! Canonical NVFP4 output is derived exclusively from the bound source operands.
use super::*;
use std::sync::Arc;

impl BoundTensorSource<'_> {
    /// Repack one native matrix/member to block_nvfp4. Macro/input scales remain available
    /// through the same bound auxiliary API; this operation never folds or replaces them.
    /// External adapters cannot forge or forward the output-directory capability:
    /// ```compile_fail
    /// use memra_gguf::{bound_output::RetainedOutputRoot,
    ///     bound_source::BoundTensorRequest, config::ModelConfig, source::{TensorSource, TensorView}};
    /// struct Forward<'a>(&'a dyn TensorSource);
    /// impl TensorSource for Forward<'_> {
    ///     fn config(&self) -> ModelConfig { self.0.config() }
    ///     fn find(&self, _: &str) -> Option<TensorView<'_>> { None }
    ///     fn bound_output_root(&self, r: &BoundTensorRequest<'_>)
    ///         -> Result<Option<&RetainedOutputRoot>, String> { self.0.bound_output_root(r) }
    /// }
    /// ```
    pub fn canonical_nvfp4(
        &self,
        id: &TensorId,
        member: Option<usize>,
    ) -> Result<Option<BoundDiskView>, String> {
        let request = self.request(id, member)?;
        let Some(native) = self.source.nvfp4_bound(&request)? else {
            return Ok(None);
        };
        let (out_f, in_f) = (native.out_f, native.in_f);
        drop(native);
        let root = self
            .source
            .bound_output_root(&request)?
            .ok_or_else(|| request.error("source has no retained canonical output capability"))?;
        crate::bound_output::repack(
            root,
            1,
            out_f,
            in_f,
            Arc::from(self.digest.as_str()),
            |_| {
                self.source.nvfp4_bound(&request)?.ok_or_else(|| {
                    request.error("native operand disappeared during canonical output")
                })
            },
        )
        .map(Some)
    }

    /// Gather an entire native bank in canonical member order, streaming one expert at a time.
    pub fn canonical_nvfp4_bank(&self, id: &TensorId) -> Result<Option<BoundDiskView>, String> {
        let resolved = self.resolved_id(id);
        let tensor = self
            .binding
            .tensors
            .get(resolved)
            .ok_or_else(|| format!("{id:?}: missing bound bank"))?;
        if tensor.transform == TensorTransform::StackExperts {
            let ids = crate::tensor_contract::expert_member_ids(&self.plan, resolved)
                .ok_or_else(|| format!("{id:?}: bank has no canonical original IDs"))?;
            let first = *ids.first().ok_or("canonical bank is empty")?;
            let request = self.request(id, Some(first as usize))?;
            let Some(native) = self.source.nvfp4_bound(&request)? else {
                return Ok(None);
            };
            let (out_f, in_f) = (native.out_f, native.in_f);
            drop(native);
            let root = self.source.bound_output_root(&request)?.ok_or_else(|| {
                request.error("source has no retained canonical output capability")
            })?;
            crate::bound_output::repack(
                root,
                ids.len(),
                out_f,
                in_f,
                Arc::from(self.digest.as_str()),
                |index| {
                    let r = self.request(id, Some(ids[index] as usize))?;
                    self.source
                        .nvfp4_bound(&r)?
                        .ok_or_else(|| r.error("bank member is not native NVFP4"))
                },
            )
            .map(Some)
        } else {
            let request = self.request(id, None)?;
            let Some(bank) = self.source.nvfp4_stacked_bound(&request)? else {
                return Ok(None);
            };
            let root = self.source.bound_output_root(&request)?.ok_or_else(|| {
                request.error("source has no retained canonical output capability")
            })?;
            crate::bound_output::repack_stacked(root, &bank, Arc::from(self.digest.as_str()))
                .map(Some)
        }
    }
}
