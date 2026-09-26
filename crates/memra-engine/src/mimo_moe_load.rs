//! Source-weight acquisition for one pinned MiMo text MLP layer.
//! Views borrow checkpoint bytes; this module does not place weights or execute an MLP.

use memra_gguf::GgmlType;
use memra_gguf::checkpoint_binding::CheckpointBinding;
use memra_gguf::config::Arch;
use memra_gguf::model_plan::{
    ActivationPlan, MlpPlan, ModelPlan, NormKind, ResidualTopology, RouterPlan, WeightTransform,
};
use memra_gguf::source::{Fp8Native, MimoMxfp4Native, TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ExpertTensor, FloatType, LayerTensor, StorageLayout, TensorId,
};

type Fail = Box<dyn std::error::Error>;

const HIDDEN: usize = 4096;
const EXPERTS: usize = 256;
const TOP_K: usize = 8;
const EXPERT_WIDTH: usize = 2048;
const DENSE_WIDTH: usize = 16384;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiMoMlpGeometry {
    Dense {
        intermediate: usize,
    },
    Routed {
        experts: usize,
        top_k: usize,
        intermediate: usize,
    },
}

impl MiMoMlpGeometry {
    pub fn from_plan(plan: &ModelPlan, index: usize) -> Result<Self, &'static str> {
        if plan.arch != Arch::MiMoV2
            || plan.hidden_size as usize != HIDDEN
            || plan.layers.len() != 48
        {
            return Err("MiMo MLP loader requires the pinned 48-layer text trunk");
        }
        let layer = plan
            .layers
            .get(index)
            .ok_or("MiMo MLP layer is out of range")?;
        if layer.index as usize != index
            || layer.residual != ResidualTopology::Serial
            || layer.sparse_overlay.is_some()
            || layer.ple.is_some()
            || layer.pre_mlp_norm.kind != NormKind::Rms
            || layer.pre_mlp_norm.weight_transform != WeightTransform::Identity
            || layer.pre_mlp_norm.epsilon.to_bits() != 1e-6f32.to_bits()
        {
            return Err("MiMo MLP layer differs from the pinned text trunk");
        }
        match (index, &layer.mlp) {
            (0, MlpPlan::Dense(dense))
                if dense.intermediate_size as usize == DENSE_WIDTH
                    && dense.activation == ActivationPlan::Silu =>
            {
                Ok(Self::Dense {
                    intermediate: DENSE_WIDTH,
                })
            }
            (0, _) => Err("MiMo layer 0 must use the dense FP8 MLP"),
            (_, MlpPlan::Moe(moe))
                if moe.expert_count as usize == EXPERTS
                    && moe.experts_per_token as usize == TOP_K
                    && moe.expert_intermediate_size as usize == EXPERT_WIDTH
                    && moe.shared.is_none()
                    && moe.activation == ActivationPlan::Silu
                    && matches!(
                        moe.router,
                        RouterPlan::Sigmoid {
                            normalize_selected: true,
                            scaling_factor: 1.0,
                            selection_bias: true
                        }
                    ) =>
            {
                Ok(Self::Routed {
                    experts: EXPERTS,
                    top_k: TOP_K,
                    intermediate: EXPERT_WIDTH,
                })
            }
            (_, _) => Err("MiMo routed MLP math differs from the pinned source"),
        }
    }
}

/// Layer 0 has three FP8 block-128 projections and no router or experts.
pub struct MiMoDenseSource<'a> {
    pub gate: Fp8Native<'a>,
    pub up: Fp8Native<'a>,
    pub down: Fp8Native<'a>,
}

/// Each projection keeps its source E2M1 nibbles and E8M0 scale bytes.
pub struct MiMoExpertSource<'a> {
    pub gate: MimoMxfp4Native<'a>,
    pub up: MimoMxfp4Native<'a>,
    pub down: MimoMxfp4Native<'a>,
}

/// Routed layer weights stay on the source until the placement owner requests an expert.
pub struct MiMoRoutedSource<'a> {
    pub router: TensorView<'a>,
    pub correction_bias: TensorView<'a>,
    source: &'a dyn TensorSource,
    binding: &'a CheckpointBinding,
    layer: u32,
}

pub enum MiMoMlpSource<'a> {
    Dense(MiMoDenseSource<'a>),
    Routed(MiMoRoutedSource<'a>),
}

fn bound_name(
    binding: &CheckpointBinding,
    id: &TensorId,
    shape: &[u64],
    storage: impl Fn(&StorageLayout) -> bool,
) -> Result<String, Fail> {
    let name = binding.require_ggml(id)?;
    let bound = binding
        .tensor(id)
        .ok_or_else(|| format!("{id:?}: no bound source tensor"))?;
    if bound.shapes.len() != 1
        || bound.shapes[0].as_slice() != shape
        || bound.storage.len() != 1
        || !storage(&bound.storage[0])
    {
        return Err(format!("{name}: bound shape or storage differs from pinned source").into());
    }
    Ok(name)
}

fn bf16_matrix<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    out: usize,
    input: usize,
) -> Result<TensorView<'a>, Fail> {
    let view = source
        .find_mimo_bf16_ggml(name)
        .ok_or_else(|| format!("{name}: raw MiMo BF16 source view unavailable"))?;
    if view.ggml_type != GgmlType::BF16
        || view.ne != [input as u64, out as u64]
        || view.bytes.len() != out * input * size_of::<u16>()
    {
        return Err(format!("{name}: raw BF16 source geometry changed").into());
    }
    Ok(view)
}

fn dense_projection<'a>(
    source: &'a dyn TensorSource,
    binding: &CheckpointBinding,
    tensor: LayerTensor,
    out: usize,
    input: usize,
) -> Result<Fp8Native<'a>, Fail> {
    let id = TensorId::Layer { index: 0, tensor };
    let name = bound_name(binding, &id, &[out as u64, input as u64], |storage| {
        matches!(
            storage,
            StorageLayout::Quantized(layout)
                if layout.format == "FP8_E4M3" && layout.block_shape == [128, 128]
        )
    })?;
    let bound = binding.tensor(&id).expect("bound_name checked this id");
    let StorageLayout::Quantized(layout) = &bound.storage[0] else {
        unreachable!("bound_name checked FP8 storage")
    };
    let stem = bound
        .checkpoint_names
        .first()
        .ok_or_else(|| format!("{name}: dense FP8 checkpoint name missing"))?
        .strip_suffix(".weight")
        .ok_or_else(|| format!("{name}: dense FP8 checkpoint name changed"))?;
    if layout.auxiliaries != [format!("{stem}.weight_scale_inv")] {
        return Err(format!("{name}: dense FP8 scale binding changed").into());
    }
    let view = source
        .find_fp8_native(&name)
        .ok_or_else(|| format!("{name}: native MiMo dense FP8 source view unavailable"))?;
    let grid = view
        .blk
        .as_ref()
        .ok_or_else(|| format!("{name}: dense FP8 block scales unavailable"))?;
    if view.out_f != out
        || view.in_f != input
        || view.bytes.len() != out * input
        || view.scale.to_bits() != 1.0f32.to_bits()
        || grid.rows != out.div_ceil(128)
        || grid.cols != input.div_ceil(128)
        || grid.scales.len() != grid.rows * grid.cols
    {
        return Err(format!("{name}: dense FP8 code or scale geometry changed").into());
    }
    Ok(view)
}

impl<'a> MiMoMlpSource<'a> {
    /// The caller must supply a source-profile binding for the same pinned checkpoint.
    /// A RecordingSource preserves consumption evidence for the caller's later audit.
    pub fn acquire(
        source: &'a dyn TensorSource,
        binding: &'a CheckpointBinding,
        plan: &ModelPlan,
        index: usize,
    ) -> Result<Self, Fail> {
        let geometry = MiMoMlpGeometry::from_plan(plan, index)?;
        if binding.family() != "mimo_v2_source"
            || binding.dialect != CheckpointDialect::HfSafetensors
        {
            return Err("MiMo MLP needs the HF source-pack binding".into());
        }
        match geometry {
            MiMoMlpGeometry::Dense { .. } => {
                let gate =
                    dense_projection(source, binding, LayerTensor::MlpGate, DENSE_WIDTH, HIDDEN)?;
                let up =
                    dense_projection(source, binding, LayerTensor::MlpUp, DENSE_WIDTH, HIDDEN)?;
                let down =
                    dense_projection(source, binding, LayerTensor::MlpDown, HIDDEN, DENSE_WIDTH)?;
                Ok(Self::Dense(MiMoDenseSource { gate, up, down }))
            }
            MiMoMlpGeometry::Routed { .. } => {
                let layer = index as u32;
                let router_id = TensorId::Layer {
                    index: layer,
                    tensor: LayerTensor::MoeRouter,
                };
                let router_name = bound_name(
                    binding,
                    &router_id,
                    &[EXPERTS as u64, HIDDEN as u64],
                    |storage| *storage == StorageLayout::Float(FloatType::Bf16),
                )?;
                let router = bf16_matrix(source, &router_name, EXPERTS, HIDDEN)?;
                let bias_id = TensorId::Layer {
                    index: layer,
                    tensor: LayerTensor::MoeRouterBias,
                };
                let bias_name = bound_name(binding, &bias_id, &[EXPERTS as u64], |storage| {
                    *storage == StorageLayout::Float(FloatType::F32)
                })?;
                let correction_bias = source
                    .find(&bias_name)
                    .ok_or_else(|| format!("{bias_name}: MiMo correction bias missing"))?;
                if correction_bias.ggml_type != GgmlType::F32
                    || correction_bias.ne != [EXPERTS as u64]
                    || correction_bias.bytes.len() != EXPERTS * size_of::<f32>()
                    || correction_bias
                        .bytes
                        .chunks_exact(4)
                        .any(|bytes| !f32::from_le_bytes(bytes.try_into().unwrap()).is_finite())
                {
                    return Err(format!("{bias_name}: MiMo correction bias changed").into());
                }
                Ok(Self::Routed(MiMoRoutedSource {
                    router,
                    correction_bias,
                    source,
                    binding,
                    layer,
                }))
            }
        }
    }
}

impl<'a> MiMoRoutedSource<'a> {
    /// Borrow one expert's three projection planes without allocating device memory.
    pub fn acquire_expert(&self, expert: usize) -> Result<MiMoExpertSource<'a>, Fail> {
        if expert >= EXPERTS {
            return Err(format!("MiMo expert {expert} is outside 0..{EXPERTS}").into());
        }
        let gate = self.projection(expert, ExpertTensor::Gate, EXPERT_WIDTH, HIDDEN)?;
        let up = self.projection(expert, ExpertTensor::Up, EXPERT_WIDTH, HIDDEN)?;
        let down = self.projection(expert, ExpertTensor::Down, HIDDEN, EXPERT_WIDTH)?;
        Ok(MiMoExpertSource { gate, up, down })
    }

    fn projection(
        &self,
        expert: usize,
        tensor: ExpertTensor,
        out: usize,
        input: usize,
    ) -> Result<MimoMxfp4Native<'a>, Fail> {
        let id = TensorId::Expert {
            layer: self.layer,
            expert: expert as u32,
            tensor,
        };
        let name = bound_name(self.binding, &id, &[out as u64, input as u64], |storage| {
            matches!(
                storage,
                StorageLayout::Quantized(layout)
                    if layout.format == "MXFP4" && layout.block_shape == [32]
            )
        })?;
        let bound = self
            .binding
            .tensor(&id)
            .expect("bound_name checked this id");
        let StorageLayout::Quantized(layout) = &bound.storage[0] else {
            unreachable!("bound_name checked MXFP4 storage")
        };
        let stem = bound
            .checkpoint_names
            .first()
            .ok_or_else(|| format!("{name}: MXFP4 checkpoint name missing"))?
            .strip_suffix(".weight")
            .ok_or_else(|| format!("{name}: MXFP4 checkpoint name changed"))?;
        if layout.auxiliaries != [format!("{stem}.weight_scale")] {
            return Err(format!("{name}: MXFP4 scale binding changed").into());
        }
        let view = self
            .source
            .find_mimo_mxfp4_expert_ggml(&name)
            .ok_or_else(|| format!("{name}: raw MiMo MXFP4 expert unavailable"))?;
        if view.out_f != out
            || view.in_f != input
            || view.weight.len() != out * input / 2
            || view.scales.len() != out * input / 32
        {
            return Err(format!("{name}: MXFP4 code or scale geometry changed").into());
        }
        Ok(view)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::checkpoint_binding::bind_census;
    use memra_gguf::config::{HfConfig, ModelConfig};
    use memra_gguf::model_packs::mimo_v2::SOURCE_PROFILE;
    use memra_gguf::source::{F8BlockGrid, TensorCensus, TensorCensusRecord};
    use memra_gguf::tensor_contract::{
        ContractOptions, IntegerType, QuantConstraint, QuantLayout, TensorCensusEntry, TensorMatch,
    };
    use std::borrow::Cow;

    fn pinned_config() -> ModelConfig {
        ModelConfig::from_hf(&HfConfig::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ))))
    }

    fn pinned_plan() -> ModelPlan {
        ModelPlan::compile(&pinned_config()).unwrap()
    }

    // A metadata-only fixture exercises the actual source binding and name map
    // without reading the 170 GB checkpoint or claiming checkpoint parity.
    fn metadata_binding() -> CheckpointBinding {
        let config = pinned_config();
        let plan = ModelPlan::compile(&config).unwrap();
        let contract = SOURCE_PROFILE
            .compile_tensor_contract(
                &config,
                &plan,
                CheckpointDialect::HfSafetensors,
                ContractOptions::default(),
            )
            .unwrap();
        let mut tensors = Vec::new();
        for requirement in contract.requirements.iter().filter(|row| row.required) {
            let storage = match requirement.quant {
                QuantConstraint::Mxfp4 | QuantConstraint::Nvfp4 | QuantConstraint::Fp8Block128 => {
                    let (format, block_shape) = match requirement.quant {
                        QuantConstraint::Mxfp4 => ("MXFP4", vec![32]),
                        QuantConstraint::Nvfp4 => ("NVFP4", vec![16]),
                        _ => ("FP8_E4M3", vec![128, 128]),
                    };
                    StorageLayout::Quantized(QuantLayout {
                        format: format.into(),
                        block_shape,
                        auxiliaries: requirement.auxiliaries.clone().unwrap_or_default(),
                    })
                }
                QuantConstraint::I64 => StorageLayout::Integer(IntegerType::I64),
                QuantConstraint::ExactFloat(dtype) => StorageLayout::Float(dtype),
                _ => match &requirement.id {
                    TensorId::Layer {
                        index: 0,
                        tensor: LayerTensor::MlpGate | LayerTensor::MlpUp | LayerTensor::MlpDown,
                    } => StorageLayout::Quantized(QuantLayout {
                        format: "FP8_E4M3".into(),
                        block_shape: vec![128, 128],
                        auxiliaries: vec![format!(
                            "{}.weight_scale_inv",
                            requirement.names[0].strip_suffix(".weight").unwrap()
                        )],
                    }),
                    TensorId::Layer {
                        tensor: LayerTensor::MoeRouterBias,
                        ..
                    } => StorageLayout::Float(FloatType::F32),
                    _ => StorageLayout::Float(FloatType::Bf16),
                },
            };
            let names = match requirement.match_mode {
                TensorMatch::OneOf => &requirement.names[..1],
                TensorMatch::All => &requirement.names[..],
            };
            for name in names {
                tensors.push(TensorCensusRecord {
                    physical_name: name.clone(),
                    dtype: String::new(),
                    entry: TensorCensusEntry {
                        name: name.clone(),
                        shape: requirement.shape.clone(),
                        storage: storage.clone(),
                        physical_bytes: 1,
                    },
                });
            }
        }
        bind_census(
            Some(&SOURCE_PROFILE),
            &config,
            &plan,
            &TensorCensus {
                dialect: CheckpointDialect::HfSafetensors,
                tensors,
            },
        )
        .unwrap()
    }

    struct OneExpertSource {
        router: Vec<u8>,
        bias: Vec<u8>,
        codes: Vec<u8>,
        scales: Vec<u8>,
    }

    impl OneExpertSource {
        fn new() -> Self {
            Self {
                router: vec![0; EXPERTS * HIDDEN * 2],
                bias: vec![0; EXPERTS * 4],
                codes: vec![0x11; EXPERT_WIDTH * HIDDEN / 2],
                scales: vec![127; EXPERT_WIDTH * HIDDEN / 32],
            }
        }
    }

    impl TensorSource for OneExpertSource {
        fn config(&self) -> ModelConfig {
            pinned_config()
        }

        fn find(&self, ggml_name: &str) -> Option<TensorView<'_>> {
            (ggml_name == "blk.1.exp_probs_b.bias").then(|| TensorView {
                bytes: Cow::Borrowed(&self.bias),
                ggml_type: GgmlType::F32,
                ne: vec![EXPERTS as u64],
            })
        }

        fn find_mimo_bf16_ggml(&self, ggml_name: &str) -> Option<TensorView<'_>> {
            (ggml_name == "blk.1.ffn_gate_inp.weight").then(|| TensorView {
                bytes: Cow::Borrowed(&self.router),
                ggml_type: GgmlType::BF16,
                ne: vec![HIDDEN as u64, EXPERTS as u64],
            })
        }

        fn find_mimo_mxfp4_expert_ggml(&self, ggml_name: &str) -> Option<MimoMxfp4Native<'_>> {
            let (out_f, in_f) = match ggml_name {
                "blk.1.ffn_gate_exps.0.weight" | "blk.1.ffn_up_exps.0.weight" => {
                    (EXPERT_WIDTH, HIDDEN)
                }
                "blk.1.ffn_down_exps.0.weight" => (HIDDEN, EXPERT_WIDTH),
                _ => return None,
            };
            Some(MimoMxfp4Native {
                weight: &self.codes,
                scales: &self.scales,
                out_f,
                in_f,
            })
        }
    }

    struct DenseSource {
        codes: Vec<u8>,
    }

    impl TensorSource for DenseSource {
        fn config(&self) -> ModelConfig {
            pinned_config()
        }

        fn find(&self, _ggml_name: &str) -> Option<TensorView<'_>> {
            None
        }

        fn find_fp8_native(&self, ggml_name: &str) -> Option<Fp8Native<'_>> {
            let (out_f, in_f) = match ggml_name {
                "blk.0.ffn_gate.weight" | "blk.0.ffn_up.weight" => (DENSE_WIDTH, HIDDEN),
                "blk.0.ffn_down.weight" => (HIDDEN, DENSE_WIDTH),
                _ => return None,
            };
            let (rows, cols) = (out_f / 128, in_f / 128);
            Some(Fp8Native {
                bytes: Cow::Borrowed(&self.codes),
                scale: 1.0,
                blk: Some(F8BlockGrid {
                    scales: vec![1.0; rows * cols],
                    rows,
                    cols,
                }),
                out_f,
                in_f,
            })
        }
    }

    #[test]
    fn dense_first_and_routed_geometry_refuse_drift() {
        let plan = pinned_plan();
        assert_eq!(
            MiMoMlpGeometry::from_plan(&plan, 0),
            Ok(MiMoMlpGeometry::Dense {
                intermediate: DENSE_WIDTH
            })
        );
        assert_eq!(
            MiMoMlpGeometry::from_plan(&plan, 1),
            Ok(MiMoMlpGeometry::Routed {
                experts: EXPERTS,
                top_k: TOP_K,
                intermediate: EXPERT_WIDTH
            })
        );
        assert_eq!(
            MiMoMlpGeometry::from_plan(&plan, 47),
            MiMoMlpGeometry::from_plan(&plan, 1)
        );
        assert!(MiMoMlpGeometry::from_plan(&plan, 48).is_err());

        let mut wrong = plan.clone();
        wrong.layers[0].mlp = wrong.layers[1].mlp.clone();
        assert!(MiMoMlpGeometry::from_plan(&wrong, 0).is_err());
        let mut wrong = plan.clone();
        if let MlpPlan::Moe(moe) = &mut wrong.layers[1].mlp {
            moe.experts_per_token = 7;
        }
        assert!(MiMoMlpGeometry::from_plan(&wrong, 1).is_err());
        if let MlpPlan::Moe(moe) = &mut wrong.layers[1].mlp {
            moe.experts_per_token = TOP_K as u32;
            moe.router = RouterPlan::Softmax;
        }
        assert!(MiMoMlpGeometry::from_plan(&wrong, 1).is_err());
    }

    #[test]
    fn bound_router_and_expert_names_fail_closed_when_missing() {
        let mut binding = metadata_binding();
        assert_eq!(
            bound_name(
                &binding,
                &TensorId::Layer {
                    index: 1,
                    tensor: LayerTensor::MoeRouter,
                },
                &[EXPERTS as u64, HIDDEN as u64],
                |storage| *storage == StorageLayout::Float(FloatType::Bf16),
            )
            .unwrap(),
            "blk.1.ffn_gate_inp.weight"
        );
        let expert = TensorId::Expert {
            layer: 1,
            expert: 0,
            tensor: ExpertTensor::Gate,
        };
        assert_eq!(
            binding.require_ggml(&expert).unwrap(),
            "blk.1.ffn_gate_exps.0.weight"
        );
        binding.bound.tensors.remove(&expert);
        let error = binding.require_ggml(&expert).unwrap_err();
        assert!(error.contains("binds no"), "{error}");
        let source = OneExpertSource::new();
        let plan = pinned_plan();
        let MiMoMlpSource::Routed(layer) =
            MiMoMlpSource::acquire(&source, &binding, &plan, 1).unwrap()
        else {
            panic!("layer 1 should route")
        };
        let error = layer.acquire_expert(0).err().unwrap().to_string();
        assert!(error.contains("binds no"), "{error}");
    }

    #[test]
    fn routed_source_borrows_one_expert_and_rejects_bad_scale_extent() {
        let plan = pinned_plan();
        let binding = metadata_binding();
        let mut source = OneExpertSource::new();
        {
            let MiMoMlpSource::Routed(layer) =
                MiMoMlpSource::acquire(&source, &binding, &plan, 1).unwrap()
            else {
                panic!("layer 1 should route")
            };
            assert_eq!(layer.router.bytes.len(), EXPERTS * HIDDEN * 2);
            assert_eq!(layer.correction_bias.bytes.len(), EXPERTS * 4);
            let expert = layer.acquire_expert(0).unwrap();
            assert_eq!(expert.gate.weight.as_ptr(), source.codes.as_ptr());
            assert_eq!(expert.up.scales.as_ptr(), source.scales.as_ptr());
            assert_eq!(
                (expert.down.out_f, expert.down.in_f),
                (HIDDEN, EXPERT_WIDTH)
            );
            assert!(layer.acquire_expert(EXPERTS).is_err());
        }
        source.scales.pop();
        let MiMoMlpSource::Routed(layer) =
            MiMoMlpSource::acquire(&source, &binding, &plan, 1).unwrap()
        else {
            panic!("layer 1 should route")
        };
        assert!(layer.acquire_expert(0).is_err());
    }

    #[test]
    fn dense_layer_zero_keeps_fp8_codes_and_block_scales() {
        let plan = pinned_plan();
        let binding = metadata_binding();
        let source = DenseSource {
            codes: vec![0; DENSE_WIDTH * HIDDEN],
        };
        let MiMoMlpSource::Dense(layer) =
            MiMoMlpSource::acquire(&source, &binding, &plan, 0).unwrap()
        else {
            panic!("layer 0 should be dense")
        };
        assert_eq!(layer.gate.bytes.as_ptr(), source.codes.as_ptr());
        assert_eq!((layer.up.out_f, layer.up.in_f), (DENSE_WIDTH, HIDDEN));
        assert_eq!((layer.down.out_f, layer.down.in_f), (HIDDEN, DENSE_WIDTH));
        assert_eq!(layer.down.blk.unwrap().scales.len(), 4096);
    }
}
