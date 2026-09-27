//! Bound MiMo vision weights. This module uploads the pinned source tensors but
//! does not execute the tower or register it for serving.

use std::collections::BTreeSet;

use cudarc::driver::CudaSlice;
use memra_gguf::GgmlType;
use memra_gguf::checkpoint_binding::{CheckpointBinding, RecordingSource};
use memra_gguf::config::ModelConfig;
use memra_gguf::model_packs::mimo_v2::read_bound_modal_bf16;
use memra_gguf::model_packs::mimo_v2::vision::{MiMoVisionAttentionPlan, pinned_attention_plan};
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{TensorId, VisionTensor};

use crate::Engine;
use crate::model::GpuTensor;

type Fail = Box<dyn std::error::Error>;

const BLOCK_TENSORS: [VisionTensor; 12] = [
    VisionTensor::AttentionOutputBias,
    VisionTensor::AttentionOutput,
    VisionTensor::FusedQkvBias,
    VisionTensor::FusedQkv,
    VisionTensor::MlpDownBias,
    VisionTensor::MlpDown,
    VisionTensor::MlpGateBias,
    VisionTensor::MlpGate,
    VisionTensor::MlpUpBias,
    VisionTensor::MlpUp,
    VisionTensor::InputNorm,
    VisionTensor::PreMlpNorm,
];

pub struct MiMoVisionBlock {
    pub plan: MiMoVisionAttentionPlan,
    pub norm1: CudaSlice<f32>,
    pub qkv: GpuTensor,
    pub qkv_bias: CudaSlice<f32>,
    pub first_key_bias: Option<CudaSlice<f32>>,
    pub attention_output: GpuTensor,
    pub attention_output_bias: CudaSlice<f32>,
    pub norm2: CudaSlice<f32>,
    pub mlp_gate: GpuTensor,
    pub mlp_gate_bias: CudaSlice<f32>,
    pub mlp_up: GpuTensor,
    pub mlp_up_bias: CudaSlice<f32>,
    pub mlp_down: GpuTensor,
    pub mlp_down_bias: CudaSlice<f32>,
}

pub struct MiMoVisionMerger {
    /// The pinned publisher applies LayerNorm with epsilon 1e-6. The absent
    /// `ln_q.bias` is zero after Transformers 5.3 missing-key initialization.
    pub norm_weight: CudaSlice<f32>,
    pub mlp_0: GpuTensor,
    pub mlp_2: GpuTensor,
    // The two absent Linear biases are also initialized to zero.
}

pub struct MiMoVisionWeights {
    /// The Conv3D checkpoint weight [1280, 3, 2, 16, 16] is kept in its
    /// original row-major BF16 bytes as the [1280, 1536] matrix operand.
    pub patch_projection: GpuTensor,
    pub blocks: Vec<MiMoVisionBlock>,
    pub merger: MiMoVisionMerger,
}

fn block_id(layer: u32, tensor: VisionTensor) -> TensorId {
    TensorId::Vision {
        layer: Some(layer),
        tensor,
    }
}

fn patch_id() -> TensorId {
    TensorId::Vision {
        layer: None,
        tensor: VisionTensor::PatchProjection,
    }
}

fn family_id(name: String) -> TensorId {
    TensorId::Family {
        family: "mimo_v2_vision",
        key: name,
    }
}

fn is_vision_id(id: &TensorId) -> bool {
    matches!(
        id,
        TensorId::Vision { .. }
            | TensorId::Family {
                family: "mimo_v2_vision",
                ..
            }
    )
}

fn pinned_plans(config: &ModelConfig) -> Result<Vec<MiMoVisionAttentionPlan>, Fail> {
    (0..28)
        .map(|layer| pinned_attention_plan(config, layer).map_err(Into::into))
        .collect()
}

fn expected_ids(plans: &[MiMoVisionAttentionPlan]) -> BTreeSet<TensorId> {
    let mut ids = BTreeSet::new();
    for plan in plans {
        for tensor in BLOCK_TENSORS {
            ids.insert(block_id(plan.layer, tensor));
        }
        if plan.sink_first_key {
            ids.insert(family_id(format!(
                "visual.blocks.{}.attn.sinks",
                plan.layer
            )));
        }
    }
    ids.insert(patch_id());
    for name in [
        "visual.merger.ln_q.weight",
        "visual.merger.mlp.0.weight",
        "visual.merger.mlp.2.weight",
    ] {
        ids.insert(family_id(name.to_owned()));
    }
    ids
}

fn validate_view(view: &TensorView<'_>, shape: &[u64]) -> Result<(), &'static str> {
    let elements = shape
        .iter()
        .try_fold(1usize, |total, &dim| {
            usize::try_from(dim)
                .ok()
                .and_then(|dim| total.checked_mul(dim))
        })
        .ok_or("BF16 shape extent overflows")?;
    if shape.is_empty()
        || shape.contains(&0)
        || view.ggml_type != GgmlType::BF16
        || view.ne != shape.iter().rev().copied().collect::<Vec<_>>()
        || view.bytes.len()
            != elements
                .checked_mul(2)
                .ok_or("BF16 byte extent overflows")?
    {
        return Err("BF16 source shape, type, or byte count differs from pinned tensor");
    }
    if view
        .bytes
        .chunks_exact(2)
        .any(|pair| u16::from_le_bytes([pair[0], pair[1]]) & 0x7f80 == 0x7f80)
    {
        return Err("BF16 source contains a non-finite value");
    }
    Ok(())
}

fn checked_view<'a>(
    source: &'a dyn TensorSource,
    binding: &CheckpointBinding,
    id: &TensorId,
    shape: &[u64],
) -> Result<TensorView<'a>, Fail> {
    let view = read_bound_modal_bf16(source, binding, id)?;
    validate_view(&view, shape).map_err(|reason| format!("{id:?}: {reason}"))?;
    Ok(view)
}

fn upload_matrix(
    engine: &Engine,
    source: &dyn TensorSource,
    binding: &CheckpointBinding,
    id: &TensorId,
    output: u64,
    input: u64,
) -> Result<GpuTensor, Fail> {
    let view = checked_view(source, binding, id, &[output, input])?;
    let tensor = GpuTensor::FloatBf16 {
        data: engine.htod_bytes(view.bytes.as_ref())?,
        ne: view.ne,
    };
    if tensor.ordinal() != engine.stream().context().ordinal() {
        return Err(format!("{id:?}: BF16 matrix was uploaded to the wrong GPU").into());
    }
    Ok(tensor)
}

fn upload_vector(
    engine: &Engine,
    source: &dyn TensorSource,
    binding: &CheckpointBinding,
    id: &TensorId,
    width: u64,
) -> Result<CudaSlice<f32>, Fail> {
    let view = checked_view(source, binding, id, &[width])?;
    let values = view
        .bytes
        .chunks_exact(2)
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
        .collect::<Vec<_>>();
    let data = engine.htod(&values)?;
    if data.ordinal() != engine.stream().context().ordinal() {
        return Err(format!("{id:?}: BF16 vector was uploaded to the wrong GPU").into());
    }
    Ok(data)
}

fn flattened_patch_ne(shape: &[u64]) -> Result<Vec<u64>, &'static str> {
    if shape.len() != 5 {
        return Err("Conv3D patch shape must have five dimensions");
    }
    let input = shape[1..]
        .iter()
        .try_fold(1u64, |total, &dim| total.checked_mul(dim))
        .ok_or("Conv3D patch row extent overflows")?;
    Ok(vec![input, shape[0]])
}

fn upload_patch(
    engine: &Engine,
    source: &dyn TensorSource,
    binding: &CheckpointBinding,
) -> Result<GpuTensor, Fail> {
    const SHAPE: [u64; 5] = [1280, 3, 2, 16, 16];
    let view = checked_view(source, binding, &patch_id(), &SHAPE)?;
    let tensor = GpuTensor::FloatBf16 {
        data: engine.htod_bytes(view.bytes.as_ref())?,
        ne: flattened_patch_ne(&SHAPE)?,
    };
    if tensor.ordinal() != engine.stream().context().ordinal() || tensor.ne() != [1536, 1280] {
        return Err("MiMo patch operand shape or GPU changed".into());
    }
    Ok(tensor)
}

impl MiMoVisionBlock {
    fn load(
        engine: &Engine,
        source: &dyn TensorSource,
        binding: &CheckpointBinding,
        plan: MiMoVisionAttentionPlan,
    ) -> Result<Self, Fail> {
        let layer = plan.layer;
        let id = |tensor| block_id(layer, tensor);
        let first_key_bias = if plan.sink_first_key {
            Some(upload_vector(
                engine,
                source,
                binding,
                &family_id(format!("visual.blocks.{layer}.attn.sinks")),
                32,
            )?)
        } else {
            None
        };
        Ok(Self {
            plan,
            norm1: upload_vector(engine, source, binding, &id(VisionTensor::InputNorm), 1280)?,
            qkv: upload_matrix(
                engine,
                source,
                binding,
                &id(VisionTensor::FusedQkv),
                3072,
                1280,
            )?,
            qkv_bias: upload_vector(
                engine,
                source,
                binding,
                &id(VisionTensor::FusedQkvBias),
                3072,
            )?,
            first_key_bias,
            attention_output: upload_matrix(
                engine,
                source,
                binding,
                &id(VisionTensor::AttentionOutput),
                1280,
                2048,
            )?,
            attention_output_bias: upload_vector(
                engine,
                source,
                binding,
                &id(VisionTensor::AttentionOutputBias),
                1280,
            )?,
            norm2: upload_vector(engine, source, binding, &id(VisionTensor::PreMlpNorm), 1280)?,
            mlp_gate: upload_matrix(
                engine,
                source,
                binding,
                &id(VisionTensor::MlpGate),
                4608,
                1280,
            )?,
            mlp_gate_bias: upload_vector(
                engine,
                source,
                binding,
                &id(VisionTensor::MlpGateBias),
                4608,
            )?,
            mlp_up: upload_matrix(
                engine,
                source,
                binding,
                &id(VisionTensor::MlpUp),
                4608,
                1280,
            )?,
            mlp_up_bias: upload_vector(
                engine,
                source,
                binding,
                &id(VisionTensor::MlpUpBias),
                4608,
            )?,
            mlp_down: upload_matrix(
                engine,
                source,
                binding,
                &id(VisionTensor::MlpDown),
                1280,
                4608,
            )?,
            mlp_down_bias: upload_vector(
                engine,
                source,
                binding,
                &id(VisionTensor::MlpDownBias),
                1280,
            )?,
        })
    }
}

impl MiMoVisionWeights {
    /// Upload exactly the source checkpoint's 364 bound `visual.*` tensors.
    /// The caller remains responsible for checkpoint custody and model-wide
    /// auditing. No vision forward path is registered here.
    pub fn load(
        engine: &Engine,
        source: &dyn TensorSource,
        binding: &CheckpointBinding,
        config: &ModelConfig,
    ) -> Result<Self, Fail> {
        let plans = pinned_plans(config)?;
        if binding.family() != "mimo_v2_source" {
            return Err("MiMo vision needs the pinned source binding".into());
        }
        let expected = expected_ids(&plans);
        let actual = binding
            .bound
            .tensors
            .keys()
            .filter(|id| is_vision_id(id))
            .cloned()
            .collect::<BTreeSet<_>>();
        if expected.len() != 364 || actual != expected {
            return Err("MiMo vision binding differs from the pinned 364 tensor IDs".into());
        }

        engine.gpu.ctx.bind_to_thread()?;
        let recording = RecordingSource::new(source);
        let recorded: &dyn TensorSource = &recording;
        let patch_projection = upload_patch(engine, recorded, binding)?;
        let mut blocks = Vec::with_capacity(plans.len());
        for plan in plans {
            blocks.push(MiMoVisionBlock::load(engine, recorded, binding, plan)?);
        }
        let merger = MiMoVisionMerger {
            norm_weight: upload_vector(
                engine,
                recorded,
                binding,
                &family_id("visual.merger.ln_q.weight".to_owned()),
                1280,
            )?,
            mlp_0: upload_matrix(
                engine,
                recorded,
                binding,
                &family_id("visual.merger.mlp.0.weight".to_owned()),
                5120,
                5120,
            )?,
            mlp_2: upload_matrix(
                engine,
                recorded,
                binding,
                &family_id("visual.merger.mlp.2.weight".to_owned()),
                4096,
                5120,
            )?,
        };
        let unread =
            binding.audit_consumption(&recording.requested(), config, |id, _| !is_vision_id(id));
        if !unread.is_empty() || recording.requested().len() != 364 {
            return Err(format!(
                "MiMo vision consumption audit failed: {} unread, {} recorded",
                unread.len(),
                recording.requested().len()
            )
            .into());
        }
        Ok(Self {
            patch_projection,
            blocks,
            merger,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use memra_gguf::config::{HfConfig, ModelConfig};
    use memra_gguf::model_packs::mimo_v2::SOURCE_PROFILE;
    use memra_gguf::tensor_contract::{CheckpointDialect, ContractOptions};

    use super::*;

    fn pinned_config() -> ModelConfig {
        ModelConfig::from_hf(&HfConfig::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ))))
    }

    #[test]
    fn loader_ids_match_all_and_only_pinned_source_vision_rows() {
        let config = pinned_config();
        let plans = pinned_plans(&config).unwrap();
        let expected = expected_ids(&plans);
        let model_plan = SOURCE_PROFILE.compile_plan(&config).unwrap();
        let contract = SOURCE_PROFILE
            .compile_tensor_contract(
                &config,
                &model_plan,
                CheckpointDialect::HfSafetensors,
                ContractOptions::default(),
            )
            .unwrap();
        let schema = contract
            .requirements
            .iter()
            .filter(|row| is_vision_id(&row.id))
            .map(|row| row.id.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(expected.len(), 364);
        assert_eq!(expected, schema);
        assert_eq!(plans.iter().filter(|plan| plan.sink_first_key).count(), 24);
        assert!(!expected.contains(&family_id("visual.merger.mlp.0.bias".to_owned())));
    }

    #[test]
    fn bf16_validation_checks_extent_type_and_finiteness() {
        let finite = [0x80, 0x3f, 0x00, 0x40];
        let mut view = TensorView {
            bytes: Cow::Borrowed(&finite),
            ggml_type: GgmlType::BF16,
            ne: vec![2],
        };
        assert!(validate_view(&view, &[2]).is_ok());
        assert!(validate_view(&view, &[1, 2]).is_err());
        assert!(validate_view(&view, &[3]).is_err());
        view.ggml_type = GgmlType::F16;
        assert!(validate_view(&view, &[2]).is_err());
        let nonfinite = [0x80, 0x7f, 0x00, 0x40];
        view = TensorView {
            bytes: Cow::Borrowed(&nonfinite),
            ggml_type: GgmlType::BF16,
            ne: vec![2],
        };
        assert!(validate_view(&view, &[2]).is_err());
    }

    #[test]
    fn conv3d_flatten_preserves_row_major_operand_shape() {
        assert_eq!(
            flattened_patch_ne(&[1280, 3, 2, 16, 16]).unwrap(),
            [1536, 1280]
        );
        assert_eq!(flattened_patch_ne(&[2, 1, 2, 2, 2]).unwrap(), [8, 2]);
        assert!(flattened_patch_ne(&[2, 8]).is_err());
        assert!(flattened_patch_ne(&[2, u64::MAX, 2, 1, 1]).is_err());
    }
}
