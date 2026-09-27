//! Explicit two-card MiMo source text weight residency.
//! This is a diagnostic load path. It has no prefill, decode, or serving door.

use cudarc::driver::CudaSlice;
use memra_gguf::GgmlType;
use memra_gguf::checkpoint_binding::{CheckpointBinding, RecordingSource};
use memra_gguf::config::{Arch, ModelConfig};
use memra_gguf::model_packs::mimo_v2::bind_pinned_text_source;
use memra_gguf::model_plan::{MlpPlan, ModelPlan};
use memra_gguf::source::{SafetensorsSource, TensorSource};
use memra_gguf::tensor_contract::{LayerTensor, TensorId};
use std::sync::Arc;

use crate::Engine;
use crate::QT_F8_E4M3_BLK;
use crate::mimo_attn_load::{MiMoAttentionGeometry, MiMoSourceAttention};
use crate::mimo_moe_load::{MiMoMlpGeometry, MiMoMlpSource};
use crate::mimo_source_moe::GroupedMiMoMoeLayer;
use crate::model::GpuTensor;

type Fail = Box<dyn std::error::Error>;
const STAGE_CUT: usize = 24;
const LAYERS: usize = 48;
const HIDDEN: usize = 4096;
const VOCAB: usize = 152_576;

pub struct MiMoDenseWeights {
    pub gate: GpuTensor,
    pub up: GpuTensor,
    pub down: GpuTensor,
}

pub struct MiMoTextLayerWeights {
    pub attention_norm: CudaSlice<f32>,
    pub attention: MiMoSourceAttention,
    pub mlp_norm: CudaSlice<f32>,
    pub dense: Option<MiMoDenseWeights>,
}

pub struct MiMoTextWeights {
    pub config: ModelConfig,
    pub plan: ModelPlan,
    pub layers: Vec<MiMoTextLayerWeights>,
    pub routed: Vec<Option<GroupedMiMoMoeLayer>>,
    pub output_norm: CudaSlice<f32>,
    pub output_head: GpuTensor,
    source: Arc<SafetensorsSource>,
    embedding_name: String,
}

fn validate_plan(plan: &ModelPlan) -> Result<(), &'static str> {
    if plan.arch != Arch::MiMoV2
        || plan.hidden_size as usize != HIDDEN
        || plan.vocab_size as usize != VOCAB
        || plan.layers.len() != LAYERS
    {
        return Err("MiMo text residency requires the pinned 48-layer source trunk");
    }
    for index in 0..LAYERS {
        MiMoAttentionGeometry::from_plan(plan, index)?;
        MiMoMlpGeometry::from_plan(plan, index)?;
    }
    Ok(())
}

pub fn stage_for_layer(index: usize) -> Result<usize, &'static str> {
    if index >= LAYERS {
        return Err("MiMo text stage layer is out of range");
    }
    Ok(usize::from(index >= STAGE_CUT))
}

fn norm_vector(
    engine: &Engine,
    source: &dyn TensorSource,
    binding: &CheckpointBinding,
    id: &TensorId,
) -> Result<CudaSlice<f32>, Fail> {
    let name = binding.require_ggml(id)?;
    let view = source
        .find_mimo_bf16_ggml(&name)
        .ok_or_else(|| format!("{name}: original MiMo norm is unavailable"))?;
    if view.ggml_type != GgmlType::BF16
        || view.ne != [HIDDEN as u64]
        || view.bytes.len() != HIDDEN * 2
    {
        return Err(format!("{name}: MiMo norm shape or storage changed").into());
    }
    let values = view
        .bytes
        .chunks_exact(2)
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name}: MiMo norm contains a non-finite value").into());
    }
    engine.htod(&values)
}

fn bf16_matrix(
    engine: &Engine,
    source: &dyn TensorSource,
    binding: &CheckpointBinding,
    id: &TensorId,
    input: usize,
    out: usize,
) -> Result<GpuTensor, Fail> {
    let name = binding.require_ggml(id)?;
    let view = source
        .find_mimo_bf16_ggml(&name)
        .ok_or_else(|| format!("{name}: original MiMo matrix is unavailable"))?;
    if view.ggml_type != GgmlType::BF16
        || view.ne != [input as u64, out as u64]
        || view.bytes.len() != input * out * 2
    {
        return Err(format!("{name}: MiMo matrix shape or storage changed").into());
    }
    Ok(GpuTensor::FloatBf16 {
        data: engine.htod_bytes(view.bytes.as_ref())?,
        ne: view.ne,
    })
}

fn dense_matrix(
    engine: &Engine,
    source: &dyn TensorSource,
    binding: &CheckpointBinding,
    tensor: LayerTensor,
    input: usize,
    out: usize,
) -> Result<GpuTensor, Fail> {
    let name = binding.require_ggml(&TensorId::Layer { index: 0, tensor })?;
    let weight = GpuTensor::load_from_source(engine, source, &name)?;
    if weight.in_features() != input
        || weight.out_features() != out
        || !matches!(
            &weight,
            GpuTensor::Quant {
                qtype: QT_F8_E4M3_BLK,
                ..
            }
        )
    {
        return Err(format!("{name}: MiMo dense weight lost native block FP8").into());
    }
    Ok(weight)
}

fn skip_modal_tensor(id: &TensorId) -> bool {
    matches!(
        id,
        TensorId::Vision { .. }
            | TensorId::Mtp { .. }
            | TensorId::Family {
                family: "mimo_v2_vision" | "mimo_v2_audio",
                ..
            }
    )
}

impl MiMoTextWeights {
    pub(crate) fn shares_source(&self, source: &Arc<SafetensorsSource>) -> bool {
        Arc::ptr_eq(&self.source, source)
    }

    /// Bind the full pinned source before uploading any text tensor, then
    /// settle every text-trunk read. Modality and draft tensors keep their
    /// own future load/audit owners.
    pub fn load(
        engines: [&Engine; 2],
        source_handle: Arc<SafetensorsSource>,
    ) -> Result<Self, Fail> {
        let (config, plan, binding) = bind_pinned_text_source(source_handle.as_ref())?;
        validate_plan(&plan)?;
        let recording = RecordingSource::new(source_handle.as_ref());
        let source: &dyn TensorSource = &recording;

        let mut routed = std::iter::repeat_with(|| None)
            .take(LAYERS)
            .collect::<Vec<Option<GroupedMiMoMoeLayer>>>();
        for (index, layer) in plan.layers.iter().enumerate().skip(1) {
            let MlpPlan::Moe(moe) = &layer.mlp else {
                return Err(format!("MiMo layer {index} lost its routed MLP").into());
            };
            let MiMoMlpSource::Routed(bound) =
                MiMoMlpSource::acquire(source, &binding, &plan, index)?
            else {
                return Err(format!("MiMo layer {index} has no bound routed source").into());
            };
            let engine = engines[stage_for_layer(index)?];
            engine.gpu.ctx.bind_to_thread()?;
            routed[index] = Some(GroupedMiMoMoeLayer::load_bound(
                engine, &config, &plan, index, moe, &bound,
            )?);
        }

        let mut layers = Vec::with_capacity(LAYERS);
        for index in 0..LAYERS {
            let engine = engines[stage_for_layer(index)?];
            engine.gpu.ctx.bind_to_thread()?;
            let layer = index as u32;
            let attention_norm = norm_vector(
                engine,
                source,
                &binding,
                &TensorId::Layer {
                    index: layer,
                    tensor: LayerTensor::PreAttentionNorm,
                },
            )?;
            let attention = MiMoSourceAttention::load(engine, source, &binding, &plan, index)?;
            let mlp_norm = norm_vector(
                engine,
                source,
                &binding,
                &TensorId::Layer {
                    index: layer,
                    tensor: LayerTensor::PreMlpNorm,
                },
            )?;
            let dense = if index == 0 {
                if !matches!(
                    MiMoMlpSource::acquire(source, &binding, &plan, index)?,
                    MiMoMlpSource::Dense(_)
                ) {
                    return Err("MiMo layer 0 has no bound dense MLP".into());
                }
                Some(MiMoDenseWeights {
                    gate: dense_matrix(
                        engine,
                        source,
                        &binding,
                        LayerTensor::MlpGate,
                        HIDDEN,
                        16_384,
                    )?,
                    up: dense_matrix(engine, source, &binding, LayerTensor::MlpUp, HIDDEN, 16_384)?,
                    down: dense_matrix(
                        engine,
                        source,
                        &binding,
                        LayerTensor::MlpDown,
                        16_384,
                        HIDDEN,
                    )?,
                })
            } else {
                None
            };
            layers.push(MiMoTextLayerWeights {
                attention_norm,
                attention,
                mlp_norm,
                dense,
            });
        }

        let last = engines[1];
        last.gpu.ctx.bind_to_thread()?;
        let output_norm = norm_vector(last, source, &binding, &TensorId::OutputNorm)?;
        let output_head = bf16_matrix(
            last,
            source,
            &binding,
            &TensorId::OutputProjection,
            HIDDEN,
            VOCAB,
        )?;
        let embedding_name = binding.require_ggml(&TensorId::TokenEmbedding)?;
        let embedding = source
            .find_mimo_bf16_ggml(&embedding_name)
            .ok_or("MiMo source token embedding is missing")?;
        if embedding.ggml_type != GgmlType::BF16
            || embedding.ne != [HIDDEN as u64, VOCAB as u64]
            || embedding.bytes.len() != HIDDEN * VOCAB * 2
        {
            return Err("MiMo source token embedding geometry changed".into());
        }
        drop(embedding);
        let unread = binding.audit_consumption(&recording.requested(), &config, |id, _| {
            skip_modal_tensor(id)
        });
        if !unread.is_empty() {
            return Err(format!(
                "MiMo text loader left {} bound tensors unread: {:?}",
                unread.len(),
                &unread[..unread.len().min(4)]
            )
            .into());
        }
        drop(recording);
        Ok(Self {
            config,
            plan,
            layers,
            routed,
            output_norm,
            output_head,
            source: source_handle,
            embedding_name,
        })
    }

    /// A borrowed source row becomes an exact f32 activation for the first
    /// GPU stage; the full 1.25 GB embedding matrix remains source-mapped.
    pub fn embedding_row(&self, token: u32) -> Result<Vec<f32>, Fail> {
        let index = token as usize;
        if index >= VOCAB {
            return Err(format!("MiMo token {token} exceeds vocabulary {VOCAB}").into());
        }
        let view = self
            .source
            .find_mimo_bf16_ggml(&self.embedding_name)
            .ok_or("MiMo source embedding is unavailable")?;
        if view.ggml_type != GgmlType::BF16
            || view.ne != [HIDDEN as u64, VOCAB as u64]
            || view.bytes.len() != HIDDEN * VOCAB * 2
        {
            return Err("MiMo source embedding changed after load".into());
        }
        let start = index * HIDDEN * 2;
        let values = view.bytes[start..start + HIDDEN * 2]
            .chunks_exact(2)
            .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
            .collect::<Vec<_>>();
        if values.iter().any(|value| !value.is_finite()) {
            return Err("MiMo source embedding row contains a non-finite value".into());
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::HfConfig;
    use memra_gguf::model_packs::mimo_v2::SOURCE_PROFILE;
    use memra_gguf::tensor_contract::{CheckpointDialect, ContractOptions};

    #[test]
    fn pinned_text_consumption_has_only_modal_and_draft_exclusions() {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ))));
        let plan = SOURCE_PROFILE.compile_plan(&config).unwrap();
        validate_plan(&plan).unwrap();
        assert_eq!(stage_for_layer(0), Ok(0));
        assert_eq!(stage_for_layer(23), Ok(0));
        assert_eq!(stage_for_layer(24), Ok(1));
        assert_eq!(stage_for_layer(47), Ok(1));
        assert!(stage_for_layer(48).is_err());
        let contract = SOURCE_PROFILE
            .compile_tensor_contract(
                &config,
                &plan,
                CheckpointDialect::HfSafetensors,
                ContractOptions::default(),
            )
            .unwrap();
        let required = contract
            .requirements
            .iter()
            .filter(|row| row.required)
            .count();
        let modal = contract
            .requirements
            .iter()
            .filter(|row| row.required && skip_modal_tensor(&row.id))
            .count();
        assert_eq!(modal, 495);
        assert_eq!(required, 36_922);
        assert_eq!(required - modal, 36_427);
    }
}
