//! Bound MiMo audio patch weight upload. No audio forward or serving path.

use cudarc::driver::{CudaSlice, DevicePtr};
use memra_gguf::checkpoint_binding::{CheckpointBinding, RecordingSource};
use memra_gguf::config::ModelConfig;
use memra_gguf::model_packs::mimo_v2::audio::{MiMoAudioPatchPlan, pinned_patch_plan};
use memra_gguf::model_packs::mimo_v2::read_bound_modal_bf16;
use memra_gguf::source::TensorSource;
use memra_gguf::tensor_contract::{TensorId, TensorOwner};

use crate::Engine;
use crate::model::GpuTensor;

type Fail = Box<dyn std::error::Error>;

const FAMILY: &str = "mimo_v2_audio";

pub struct MiMoAudioLocalLayer {
    pub input_norm: CudaSlice<f32>,
    pub post_attention_norm: CudaSlice<f32>,
    pub query: GpuTensor,
    pub query_bias: CudaSlice<f32>,
    pub key: GpuTensor,
    pub key_bias: CudaSlice<f32>,
    pub value: GpuTensor,
    pub value_bias: CudaSlice<f32>,
    pub attention_output: GpuTensor,
    pub mlp_gate: GpuTensor,
    pub mlp_up: GpuTensor,
    pub mlp_down: GpuTensor,
}

pub struct MiMoAudioPatchWeights {
    pub plan: MiMoAudioPatchPlan,
    pub speech_embeddings: Vec<GpuTensor>,
    pub speech_ptrs: CudaSlice<u64>,
    pub local_layers: Vec<MiMoAudioLocalLayer>,
    pub final_norm: CudaSlice<f32>,
    pub projection_in: GpuTensor,
    pub projection_out: GpuTensor,
}

fn audio_id(name: &str) -> TensorId {
    TensorId::Family {
        family: FAMILY,
        key: name.to_string(),
    }
}

fn matrix(
    engine: &Engine,
    source: &dyn TensorSource,
    binding: &CheckpointBinding,
    name: &str,
    out: usize,
    input: usize,
) -> Result<GpuTensor, Fail> {
    let view = read_bound_modal_bf16(source, binding, &audio_id(name))?;
    if view.ne != [input as u64, out as u64]
        || view.bytes.len()
            != out
                .checked_mul(input)
                .and_then(|elements| elements.checked_mul(2))
                .ok_or("MiMo audio matrix extent overflows")?
    {
        return Err(format!("{name}: pinned BF16 matrix extent changed").into());
    }
    Ok(GpuTensor::FloatBf16 {
        data: engine.htod_bytes(view.bytes.as_ref())?,
        ne: view.ne,
    })
}

fn vector(
    engine: &Engine,
    source: &dyn TensorSource,
    binding: &CheckpointBinding,
    name: &str,
    width: usize,
) -> Result<CudaSlice<f32>, Fail> {
    let view = read_bound_modal_bf16(source, binding, &audio_id(name))?;
    if view.ne != [width as u64] || view.bytes.len() != width * 2 {
        return Err(format!("{name}: pinned BF16 vector extent changed").into());
    }
    let values = view
        .bytes
        .chunks_exact(2)
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name}: source vector contains a non-finite value").into());
    }
    engine.htod(&values)
}

impl MiMoAudioLocalLayer {
    fn load(
        engine: &Engine,
        source: &dyn TensorSource,
        binding: &CheckpointBinding,
        layer: usize,
        plan: &MiMoAudioPatchPlan,
    ) -> Result<Self, Fail> {
        let prefix = format!("audio_encoder.input_local_transformer.layers.{layer}");
        let hidden = plan.local_hidden;
        let ff = plan.local_intermediate;
        Ok(Self {
            input_norm: vector(
                engine,
                source,
                binding,
                &format!("{prefix}.input_layernorm.weight"),
                hidden,
            )?,
            post_attention_norm: vector(
                engine,
                source,
                binding,
                &format!("{prefix}.post_attention_layernorm.weight"),
                hidden,
            )?,
            query: matrix(
                engine,
                source,
                binding,
                &format!("{prefix}.self_attn.q_proj.weight"),
                hidden,
                hidden,
            )?,
            query_bias: vector(
                engine,
                source,
                binding,
                &format!("{prefix}.self_attn.q_proj.bias"),
                hidden,
            )?,
            key: matrix(
                engine,
                source,
                binding,
                &format!("{prefix}.self_attn.k_proj.weight"),
                hidden,
                hidden,
            )?,
            key_bias: vector(
                engine,
                source,
                binding,
                &format!("{prefix}.self_attn.k_proj.bias"),
                hidden,
            )?,
            value: matrix(
                engine,
                source,
                binding,
                &format!("{prefix}.self_attn.v_proj.weight"),
                hidden,
                hidden,
            )?,
            value_bias: vector(
                engine,
                source,
                binding,
                &format!("{prefix}.self_attn.v_proj.bias"),
                hidden,
            )?,
            attention_output: matrix(
                engine,
                source,
                binding,
                &format!("{prefix}.self_attn.o_proj.weight"),
                hidden,
                hidden,
            )?,
            mlp_gate: matrix(
                engine,
                source,
                binding,
                &format!("{prefix}.mlp.gate_proj.weight"),
                ff,
                hidden,
            )?,
            mlp_up: matrix(
                engine,
                source,
                binding,
                &format!("{prefix}.mlp.up_proj.weight"),
                ff,
                hidden,
            )?,
            mlp_down: matrix(
                engine,
                source,
                binding,
                &format!("{prefix}.mlp.down_proj.weight"),
                hidden,
                ff,
            )?,
        })
    }
}

impl MiMoAudioPatchWeights {
    /// Upload the exact 95 bound audio-patch tensors to the selected GPU.
    /// The caller owns the source-profile bind and any model-wide audit.
    pub fn load(
        engine: &Engine,
        source: &dyn TensorSource,
        binding: &CheckpointBinding,
        config: &ModelConfig,
    ) -> Result<Self, Fail> {
        let plan = pinned_patch_plan(config)?;
        if binding.family() != "mimo_v2_source" {
            return Err("MiMo audio patch needs the pinned source binding".into());
        }
        engine.gpu.ctx.bind_to_thread()?;
        let recording = RecordingSource::new(source);
        let source: &dyn TensorSource = &recording;
        let mut speech_embeddings = Vec::with_capacity(plan.code_channels);
        for channel in 0..plan.code_channels {
            speech_embeddings.push(matrix(
                engine,
                source,
                binding,
                &format!("speech_embeddings.{channel}.weight"),
                plan.code_vocab,
                plan.local_hidden,
            )?);
        }
        let stream = engine.stream();
        let mut pointers = Vec::with_capacity(plan.code_channels);
        for table in &speech_embeddings {
            let GpuTensor::FloatBf16 { data, ne } = table else {
                return Err("MiMo speech table changed resident format".into());
            };
            if ne != &[plan.local_hidden as u64, plan.code_vocab as u64]
                || data.ordinal() != stream.context().ordinal()
            {
                return Err("MiMo speech table changed shape or GPU".into());
            }
            pointers.push(data.device_ptr(&stream).0);
        }
        let speech_ptrs = engine.htod_u64(&pointers)?;
        let mut local_layers = Vec::with_capacity(plan.local_layers);
        for layer in 0..plan.local_layers {
            local_layers.push(MiMoAudioLocalLayer::load(
                engine, source, binding, layer, &plan,
            )?);
        }
        let final_norm = vector(
            engine,
            source,
            binding,
            "audio_encoder.input_local_transformer.norm.weight",
            plan.local_hidden,
        )?;
        let projection_in = matrix(
            engine,
            source,
            binding,
            "audio_encoder.projection.mlp.0.weight",
            plan.projection_intermediate,
            plan.projection_input,
        )?;
        let projection_out = matrix(
            engine,
            source,
            binding,
            "audio_encoder.projection.mlp.2.weight",
            plan.output_hidden,
            plan.projection_intermediate,
        )?;
        let unread = binding.audit_consumption(&recording.requested(), config, |id, bound| {
            !matches!(id, TensorId::Family { family: FAMILY, .. })
                || bound.owner != TensorOwner::Global
        });
        if !unread.is_empty() {
            return Err(format!(
                "MiMo audio patch left {} bound tensors unread: {:?}",
                unread.len(),
                &unread[..unread.len().min(4)]
            )
            .into());
        }
        Ok(Self {
            plan,
            speech_embeddings,
            speech_ptrs,
            local_layers,
            final_norm,
            projection_in,
            projection_out,
        })
    }
}
