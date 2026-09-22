//! Physical access for compiler-bound requests. No legacy ggml-to-HF name resolution here.
use super::*;
use crate::bound_source::{BoundTensorRequest, BoundTensorView};
use crate::hf_mapping::{HfTarget, TransformKind};
use crate::tensor_contract::{LayerTensor, TensorId, TensorTransform};

impl GgufSource<'_> {
    pub(super) fn validate_bound(&self, r: &BoundTensorRequest<'_>) -> Result<(), String> {
        if r.dialect != CheckpointDialect::Gguf {
            return Err(r.error("expected GGUF binding"));
        }
        let t = self
            .0
            .find(&r.record.physical_name)
            .ok_or_else(|| r.error("bound physical tensor is missing"))?;
        if t.ne != r.record.entry.shape
            || ggml_storage(t.ggml_type) != r.record.entry.storage
            || t.n_bytes != r.record.entry.physical_bytes
        {
            return Err(r.error("opened tensor metadata differs from its bound census"));
        }
        Ok(())
    }
    pub(super) fn materialize_bound(
        &self,
        r: &BoundTensorRequest<'_>,
    ) -> Result<TensorView<'_>, String> {
        self.validate_bound(r)?;
        if r.transform != TensorTransform::Identity {
            return Err(r.error("this view requires an explicit derived-plane consumer"));
        }
        self.find(&r.record.physical_name)
            .ok_or_else(|| r.error("bound physical tensor is missing"))
    }
}

impl Hy3RepackSource {
    pub(super) fn validate_complete_bound_repack(&self) -> Result<(), String> {
        if self.fallback.is_some() {
            return Err("bound repack fallback overlays require a composite tensor contract before activation".into());
        }
        Ok(())
    }
    pub(super) fn validate_bound_repack(&self, r: &BoundTensorRequest<'_>) -> Result<(), String> {
        self.validate_complete_bound_repack()
            .map_err(|e| r.error(e))?;
        self.validate_own_bound_repack(r)
    }

    // Metadata-only composite validation may inspect a shadowed own row. This does not open
    // the fallback materialization path: every existing bound reader keeps the guard above.
    pub(super) fn validate_own_bound_repack(
        &self,
        r: &BoundTensorRequest<'_>,
    ) -> Result<(), String> {
        if r.dialect != CheckpointDialect::Gguf
            || !matches!(
                r.transform,
                TensorTransform::Identity | TensorTransform::StackExperts
            )
            || !matches!(
                r.view,
                BoundTensorView::Whole | BoundTensorView::EncodedBank
            )
        {
            return Err(r.error(
                "complete repack access requires an identity or expert-group GGUF binding",
            ));
        }
        let t = self
            .tensors
            .get(&r.record.physical_name)
            .ok_or_else(|| r.error("bound repack tensor is missing"))?;
        if t.ne != r.record.entry.shape
            || ggml_storage(t.ggml_type) != r.record.entry.storage
            || t.bytes as u64 != r.record.entry.physical_bytes
            || r.record.dtype != format!("{:?}", t.ggml_type)
            || !r.record.auxiliaries.is_empty()
            || !r.record.entry.auxiliaries.is_empty()
        {
            return Err(r.error("opened repack tensor differs from its bound census"));
        }
        Ok(())
    }
}

impl SafetensorsSource {
    pub(super) fn materialize_bound_auxiliary(
        &self,
        r: &BoundTensorRequest<'_>,
        kind: crate::tensor_contract::QuantAuxTensor,
    ) -> Result<Option<TensorView<'_>>, String> {
        self.validate_bound(r)?;
        self.validate_bound_values(r)?;
        use crate::tensor_contract::QuantAuxTensor as Q;
        let suffixes: &[&str] = match kind {
            Q::WeightScale => &[".weight_scale_2", ".weight_global_scale"],
            Q::InputScale => &[".input_scale"],
            Q::PreQuantScale => &[".pre_quant_scale"],
        };
        let mut matches = r
            .record
            .auxiliaries
            .iter()
            .filter(|aux| suffixes.iter().any(|s| aux.physical_name.ends_with(s)));
        let Some(aux) = matches.next() else {
            return Ok(None);
        };
        if matches.next().is_some() {
            return Err(r.error("ambiguous auxiliary planes"));
        }
        let (info, bytes) = self
            .model
            .raw(&aux.physical_name)
            .ok_or_else(|| r.error("bound auxiliary payload is missing"))?;
        let dtype = info.ggml_type().map_err(|e| r.error(e))?;
        if aux.physical_name.ends_with(".weight_global_scale") {
            let element_bytes = match dtype {
                GgmlType::F32 => 4,
                GgmlType::BF16 | GgmlType::F16 => 2,
                _ => return Err(r.error("auxiliary must be floating point")),
            };
            let values = crate::dequant::dequantize(dtype, bytes, bytes.len() / element_bytes);
            return Ok(Some(TensorView {
                bytes: Cow::Owned(
                    values
                        .iter()
                        .flat_map(|v| (1.0 / v).to_le_bytes())
                        .collect(),
                ),
                ggml_type: GgmlType::F32,
                ne: info.ne(),
            }));
        }
        if kind == Q::PreQuantScale && r.transform == TensorTransform::OutReorderColumns {
            let element_bytes = match dtype {
                GgmlType::F32 => 4,
                GgmlType::F16 | GgmlType::BF16 => 2,
                _ => unreachable!(),
            };
            let reordered =
                crate::hf_mapping::reorder_input_scale_bytes(bytes, element_bytes, &self.cfg)
                    .map_err(|e| r.error(e))?;
            return Ok(Some(TensorView {
                bytes: Cow::Owned(reordered),
                ggml_type: dtype,
                ne: info.ne(),
            }));
        }
        Ok(Some(TensorView {
            bytes: Cow::Borrowed(bytes),
            ggml_type: dtype,
            ne: info.ne(),
        }))
    }
    pub(super) fn validate_bound(&self, r: &BoundTensorRequest<'_>) -> Result<(), String> {
        if r.dialect != CheckpointDialect::HfSafetensors {
            return Err(r.error("expected safetensors binding"));
        }
        let physical = &r.record.physical_name;
        let info = self
            .model
            .info(physical)
            .ok_or_else(|| r.error("bound physical header is missing"))?;
        if info.dtype != r.record.dtype {
            return Err(r.error("opened dtype differs from bound census"));
        }
        let aux_names: Vec<_> = r
            .record
            .auxiliaries
            .iter()
            .map(|aux| aux.physical_name.clone())
            .collect();
        let (shape, _) = safetensors_storage(info, &aux_names).map_err(|e| r.error(e))?;
        if shape != r.record.entry.shape {
            return Err(r.error("opened shape differs from bound census"));
        }
        let mut total = info_physical_bytes(physical, info).map_err(|e| r.error(e))?;
        for aux in &r.record.auxiliaries {
            let header = self
                .model
                .info(&aux.physical_name)
                .ok_or_else(|| r.error(format!("missing auxiliary {}", aux.physical_name)))?;
            if header.dtype != aux.dtype
                || header.shape != aux.shape
                || info_physical_bytes(&aux.physical_name, header)? != aux.physical_bytes
            {
                return Err(r.error(format!(
                    "auxiliary {} differs from bound census",
                    aux.physical_name
                )));
            }
            total = total
                .checked_add(aux.physical_bytes)
                .ok_or_else(|| r.error("auxiliary byte sum overflow"))?;
        }
        if total != r.record.entry.physical_bytes {
            return Err(r.error("bound byte count differs from opened source"));
        }
        let stem = physical
            .strip_suffix(".weight_packed")
            .or_else(|| physical.strip_suffix(".weight"));
        let Some(stem) = stem else {
            return Ok(());
        };
        let auxiliary = |suffix: &str| {
            r.record
                .auxiliaries
                .iter()
                .find(|a| a.physical_name == format!("{stem}.{suffix}"))
        };
        match info.dtype.as_str() {
            "U8" => {
                let scale = auxiliary("weight_scale")
                    .ok_or_else(|| r.error("NVFP4 is missing weight_scale"))?;
                if shape.len() != 2 && shape.len() != 3 {
                    return Err(r.error("NVFP4 requires a matrix or expert bank"));
                }
                let input = *shape.last().unwrap();
                if input == 0 || !input.is_multiple_of(16) {
                    return Err(r.error("NVFP4 input width must be a positive multiple of 16"));
                }
                let mut expected = shape.clone();
                *expected.last_mut().unwrap() = input / 16;
                let padded = if shape.len() == 2
                    && self.nvfp4_scale_layout == Nvfp4ScaleLayout::Swizzle32x4x4
                {
                    Some(vec![
                        shape[0].div_ceil(128) * 128,
                        (input / 16).div_ceil(4) * 4,
                    ])
                } else {
                    None
                };
                if scale.dtype != "F8_E4M3"
                    || (scale.shape != expected && padded.as_ref() != Some(&scale.shape))
                {
                    return Err(r.error("NVFP4 weight_scale shape/dtype is incompatible"));
                }
                let packed = physical.ends_with(".weight_packed");
                let macro_name = if packed {
                    "weight_global_scale"
                } else {
                    "weight_scale_2"
                };
                if auxiliary(if packed {
                    "weight_scale_2"
                } else {
                    "weight_global_scale"
                })
                .is_some()
                {
                    return Err(r.error("conflicting NVFP4 macro-scale dialects"));
                }
                if let Some(macro_scale) = auxiliary(macro_name) {
                    let valid_shape = if shape.len() == 2 {
                        macro_scale.shape.is_empty() || macro_scale.shape == [1]
                    } else {
                        macro_scale.shape == [shape[0]]
                    };
                    if macro_scale.dtype != "F32" || !valid_shape {
                        return Err(r.error("NVFP4 macro-scale shape/dtype is incompatible"));
                    }
                } else if shape.len() == 2 || packed {
                    return Err(r.error(format!("NVFP4 is missing {macro_name}")));
                }
            }
            "F8_E4M3" => {
                let scale = match (auxiliary("weight_scale"), auxiliary("weight_scale_inv")) {
                    (Some(_), Some(_)) => return Err(r.error("ambiguous FP8 weight scale planes")),
                    (Some(s), None) | (None, Some(s)) => s,
                    _ => return Err(r.error("FP8 weight is missing a scale plane")),
                };
                if !matches!(scale.dtype.as_str(), "F32" | "BF16") {
                    return Err(r.error("FP8 scale must be F32 or BF16"));
                }
                let valid = match shape.as_slice() {
                    [out, input] => {
                        scale.shape.is_empty()
                            || scale.shape == [1]
                            || scale.shape == [1, 1]
                            || scale.shape == [*out]
                            || scale.shape == [*out, 1]
                            || scale.shape == [out.div_ceil(128), input.div_ceil(128)]
                    }
                    [experts, out, input] => {
                        scale.shape == [*experts, out.div_ceil(128), input.div_ceil(128)]
                    }
                    _ => false,
                };
                if !valid {
                    return Err(r.error("FP8 scale shape is incompatible"));
                }
            }
            _ => {}
        }
        if let Some(scale) = auxiliary("pre_quant_scale")
            && (scale.shape
                != [*shape
                    .last()
                    .ok_or_else(|| r.error("AWQ weight has no input axis"))?]
                || !matches!(scale.dtype.as_str(), "F32" | "F16" | "BF16"))
        {
            return Err(r.error("AWQ pre_quant_scale must match the input axis"));
        }
        if let Some(scale) = auxiliary("input_scale") {
            let scalar = scale.shape.is_empty() || scale.shape == [1];
            let per_expert = shape.len() == 3 && scale.shape == [shape[0]];
            if !matches!(scale.dtype.as_str(), "F32" | "F16" | "BF16") || (!scalar && !per_expert) {
                return Err(
                    r.error("input_scale must be a floating scalar or one scale per expert")
                );
            }
        }
        Ok(())
    }

    /// Validate all folded planes before selecting any optional representation. In particular,
    /// an unsupported native transform must not hide an invalid macro/input/AWQ scale.
    pub(super) fn validate_bound_values(&self, r: &BoundTensorRequest<'_>) -> Result<(), String> {
        for aux in &r.record.auxiliaries {
            let (info, bytes) = self
                .model
                .raw(&aux.physical_name)
                .ok_or_else(|| r.error(format!("missing bound auxiliary {}", aux.physical_name)))?;
            match info.dtype.as_str() {
                "F32" => validate_scale_values(
                    r,
                    aux,
                    bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
                )?,
                "F16" => validate_scale_values(
                    r,
                    aux,
                    bytes.chunks_exact(2).map(|b| {
                        crate::dequant::fp16_to_f32(u16::from_le_bytes(b.try_into().unwrap()))
                    }),
                )?,
                "BF16" => validate_scale_values(
                    r,
                    aux,
                    bytes.chunks_exact(2).map(|b| {
                        crate::dequant::bf16_to_f32(u16::from_le_bytes(b.try_into().unwrap()))
                    }),
                )?,
                "F8_E4M3" => validate_scale_values(
                    r,
                    aux,
                    bytes
                        .iter()
                        .map(|&b| crate::nvfp4_repack::fp8_e4m3_to_f32(b)),
                )?,
                _ => {
                    return Err(r.error(format!(
                        "auxiliary {} has unsupported dtype {}",
                        aux.physical_name, info.dtype
                    )));
                }
            }
        }
        Ok(())
    }

    fn bound_target(&self, r: &BoundTensorRequest<'_>) -> Result<HfTarget, String> {
        self.validate_bound(r)?;
        self.validate_bound_values(r)?;
        let hf = r
            .record
            .physical_name
            .strip_suffix(".weight_packed")
            .map(|stem| format!("{stem}.weight"))
            .unwrap_or_else(|| r.record.physical_name.clone());
        match (r.view, r.transform) {
            (BoundTensorView::MlaKey, TensorTransform::SplitMlaKv) => {
                return Ok(HfTarget::Transform {
                    hf,
                    kind: TransformKind::MlaKeyUpSplit,
                });
            }
            (BoundTensorView::MlaValue, TensorTransform::SplitMlaKv) => {
                return Ok(HfTarget::Transform {
                    hf,
                    kind: TransformKind::MlaValueUpSplit,
                });
            }
            (BoundTensorView::EncodedBank, TensorTransform::SplitExpertGateUp) => {
                return Ok(HfTarget::Plain(hf));
            }
            (BoundTensorView::Whole, _) => {}
            _ => return Err(r.error("derived view is incompatible with the bound transform")),
        }
        use TensorTransform as T;
        let kind = match r.transform {
            T::Identity => return Ok(HfTarget::Plain(hf)),
            T::NormAddOne => TransformKind::NormPlusOne,
            T::QkvVReorderRows => TransformKind::QkvVReorderRows,
            T::ZReorderRows => TransformKind::ZReorderRows,
            T::AbReorderRows => TransformKind::AbReorderRows,
            T::NegExpReorderHeads => TransformKind::NegExpReorderHeads,
            T::ReorderHeads => TransformKind::ReorderHeads,
            T::Conv1dSqueezeReorder => TransformKind::Conv1dSqueezeReorder,
            T::OutReorderColumns => TransformKind::OutReorderCols,
            T::StackExperts | T::SplitExpertGateUp | T::SplitMlaKv => {
                return Err(
                    r.error("this view requires an explicit member or derived-plane consumer")
                );
            }
        };
        Ok(HfTarget::Transform { hf, kind })
    }

    pub(super) fn materialize_bound(
        &self,
        r: &BoundTensorRequest<'_>,
    ) -> Result<TensorView<'_>, String> {
        let target = self.bound_target(r)?;
        if matches!(r.id, TensorId::Vision { .. }) {
            if r.transform != TensorTransform::Identity {
                return Err(r.error("vision tensor transform has no qualified materializer"));
            }
            let (info, bytes) = self
                .model
                .raw(&r.record.physical_name)
                .ok_or_else(|| r.error("bound vision payload is missing"))?;
            return Ok(TensorView {
                bytes: Cow::Borrowed(bytes),
                ggml_type: info.ggml_type().map_err(|e| r.error(e))?,
                ne: info.ne(),
            });
        }
        let router = matches!(
            r.id,
            TensorId::Layer {
                tensor: LayerTensor::MoeRouter,
                ..
            }
        );
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.materialize_target(target, router, *r.id == TensorId::OutputProjection)
        }))
        .map_err(|_| r.error("tensor materializer rejected the bound payload"))?
        .ok_or_else(|| r.error("bound tensor cannot be materialized in the declared layout"))
    }
    pub(super) fn bound_nvfp4(
        &self,
        r: &BoundTensorRequest<'_>,
    ) -> Result<Option<Nvfp4Native<'_>>, String> {
        let target = self.bound_target(r)?;
        if r.record.dtype != "U8"
            || r.record.entry.shape.len() != 2
            || r.transform != TensorTransform::Identity
        {
            return Ok(None);
        }
        self.find_nvfp4_native_target(target)
            .map(Some)
            .ok_or_else(|| r.error("invalid native NVFP4 payload or scale"))
    }
    pub(super) fn bound_fp8(
        &self,
        r: &BoundTensorRequest<'_>,
    ) -> Result<Option<Fp8Native<'_>>, String> {
        let target = self.bound_target(r)?;
        if r.record.dtype != "F8_E4M3" || r.record.entry.shape.len() != 2 {
            return Ok(None);
        }
        // Check values even when the native consumer cannot represent per-row scales or a
        // transformed block grid. An invalid plane must not turn into an optional native miss.
        let physical = &r.record.physical_name;
        let stem = physical
            .strip_suffix(".weight")
            .ok_or_else(|| r.error("FP8 target is not a weight"))?;
        let (info, bytes) = self
            .f8_scale_sibling(stem)
            .ok_or_else(|| r.error("missing FP8 scale"))?;
        let shape = &r.record.entry.shape;
        f8_scales(info, bytes, shape[0] as usize, shape[1] as usize)
            .ok_or_else(|| r.error("invalid FP8 scale values"))?;
        Ok(self.find_fp8_native_target(target))
    }
    pub(super) fn bound_fp8_stacked(
        &self,
        r: &BoundTensorRequest<'_>,
    ) -> Result<Option<Fp8StackedNative<'_>>, String> {
        let target = self.bound_target(r)?;
        if r.record.dtype != "F8_E4M3"
            || r.record.entry.shape.len() != 3
            || r.transform != TensorTransform::Identity
        {
            return Ok(None);
        }
        self.find_fp8_stacked_native_target(target)
            .map(Some)
            .ok_or_else(|| r.error("invalid native FP8 expert bank or scales"))
    }
    pub(super) fn bound_nvfp4_stacked(
        &self,
        r: &BoundTensorRequest<'_>,
    ) -> Result<Option<Nvfp4StackedNative<'_>>, String> {
        let target = self.bound_target(r)?;
        if r.record.dtype != "U8"
            || r.record.entry.shape.len() != 3
            || r.transform != TensorTransform::Identity
        {
            return Ok(None);
        }
        self.find_nvfp4_stacked_native_target(target)
            .map(Some)
            .ok_or_else(|| r.error("invalid native NVFP4 expert bank or scales"))
    }
    pub(super) fn bound_disk(
        &self,
        r: &BoundTensorRequest<'_>,
    ) -> Result<Option<DiskExtent>, String> {
        let target = self.bound_target(r)?;
        if r.record.dtype != "F8_E4M3"
            || r.record.entry.shape.len() != 3
            || r.transform != TensorTransform::Identity
        {
            return Ok(None);
        }
        self.find_expert_disk_target(target)
            .map(Some)
            .ok_or_else(|| r.error("bound expert disk extent disappeared"))
    }
}

/// Stream through scale planes: validation must not allocate another full expert-bank scale copy.
fn validate_scale_values(
    r: &BoundTensorRequest<'_>,
    aux: &TensorAuxiliaryRecord,
    values: impl IntoIterator<Item = f32>,
) -> Result<(), String> {
    let allow_zero = aux.dtype == "F8_E4M3" && r.record.dtype == "U8";
    let invert = aux.physical_name.ends_with(".weight_global_scale");
    for value in values {
        if !value.is_finite()
            || if allow_zero {
                value < 0.0
            } else {
                value <= 0.0
            }
        {
            return Err(r.error(format!(
                "invalid auxiliary scale values in {}",
                aux.physical_name
            )));
        }
        if invert && !(1.0 / value).is_finite() {
            return Err(r.error("inverse global scale overflows"));
        }
    }
    Ok(())
}
