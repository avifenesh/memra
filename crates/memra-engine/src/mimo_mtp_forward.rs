//! Bounded, model-owned one-token MiMo MTP3 draft execution.
//!
//! SGLang `mimo_v2_nextn.py` at 35b369eb and `mimo_v2.py` at 26a3214c
//! define embedding then target-hidden RMS normalization, concatenation,
//! fusion, one SWA dense block, final norm and a shared head.
//! The separate checkpoint stores Q/K/V interleaved in four QKV shards. Each
//! draft depth has independent KV state. This path does not schedule speculative
//! verification or admit a serving request.

use std::error::Error;

use cudarc::driver::CudaSlice;
use memra_gguf::config::{Arch, ModelConfig};
use memra_gguf::model_plan::{AttentionPlan, ModelPlan, RopeFactors, TensorPresence};
use memra_gguf::tensor_contract::MtpTensor;

use crate::Engine;
use crate::QT_F8_E4M3_BLK;
use crate::mimo_mtp_weights::{Mtp3Layer, Mtp3Weights, MtpBf16, MtpResidentTensor};
use crate::mimo_text_forward::{validate_forward_plan, validate_residency};
use crate::mimo_text_weights::MiMoTextWeights;
use crate::model::GpuTensor;

type Fail = Box<dyn Error>;
const DEPTHS: usize = 3;
const HIDDEN: usize = 4096;
const INTERMEDIATE: usize = 16_384;
const VOCAB: usize = 152_576;
const QUERY_HEADS: usize = 64;
const KV_HEADS: usize = 8;
const QK: usize = 192;
const VALUE: usize = 128;
const SHARDS: usize = 4;
const Q_ROWS: usize = QUERY_HEADS * QK / SHARDS;
const K_ROWS: usize = KV_HEADS * QK / SHARDS;
const V_ROWS: usize = KV_HEADS * VALUE / SHARDS;
const SHARD_ROWS: usize = Q_ROWS + K_ROWS + V_ROWS;
const QKV_ROWS: usize = SHARDS * SHARD_ROWS;
const OUTPUT_WIDTH: usize = QUERY_HEADS * VALUE;

/// F32 KV bounds this diagnostic path to 256 positions per draft depth.
pub const MAX_MTP3_CONTEXT_TOKENS: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QkvPart {
    Query,
    Key,
    Value,
}

fn qkv_segment(shard: usize, part: QkvPart) -> Result<(usize, usize, usize), &'static str> {
    if shard >= SHARDS {
        return Err("MiMo MTP3 QKV shard is out of range");
    }
    let (within, destination, length) = match part {
        QkvPart::Query => (0, shard * Q_ROWS, Q_ROWS),
        QkvPart::Key => (Q_ROWS, shard * K_ROWS, K_ROWS),
        QkvPart::Value => (Q_ROWS + K_ROWS, shard * V_ROWS, V_ROWS),
    };
    Ok((shard * SHARD_ROWS + within, destination, length))
}

fn validate_contract(config: &ModelConfig, plan: &ModelPlan) -> Result<(), &'static str> {
    let mimo = config
        .mimo
        .as_ref()
        .ok_or("MiMo MTP3 has no source-family config")?;
    let Some(AttentionPlan::SlidingWindow { attention, window }) =
        plan.layers.get(1).map(|layer| &layer.attention)
    else {
        return Err("MiMo MTP3 requires the source SWA attention plan");
    };
    let math = attention
        .mimo_math
        .ok_or("MiMo MTP3 SWA has no source-family attention math")?;
    if config.arch != Arch::MiMoV2
        || config.n_embd as usize != HIDDEN
        || config.n_vocab as usize != VOCAB
        || config.rms_eps.to_bits() != 1e-6f32.to_bits()
        || config.tie_word_embeddings != Some(false)
        || mimo.separate_mtp_layers != Some(DEPTHS as u32)
        || mimo.swa_num_attention_heads != Some(QUERY_HEADS as u32)
        || mimo.swa_num_key_value_heads != Some(KV_HEADS as u32)
        || mimo.swa_head_dim != Some(QK as u32)
        || mimo.swa_v_head_dim != Some(VALUE as u32)
        || mimo.swa_rope_theta.map(f32::to_bits) != Some(10_000f32.to_bits())
        || mimo.attention_value_scale.map(f32::to_bits) != Some(0.707f32.to_bits())
        || mimo.add_swa_attention_sink_bias != Some(true)
        || *window != 128
        || attention.query_heads as usize != QUERY_HEADS
        || attention.kv_heads as usize != KV_HEADS
        || attention.key_head_dim as usize != QK
        || attention.value_head_dim as usize != VALUE
        || attention.rope.dimensions != 64
        || attention.rope.base.to_bits() != 10_000f32.to_bits()
        || attention.rope.factors != RopeFactors::None
        || math.sink != TensorPresence::Required
        || math.fused_qkv_checkpoint_shards != Some(SHARDS as u32)
        || math.value_scale_before_cache.to_bits() != 0.707f32.to_bits()
    {
        return Err("MiMo MTP3 config or attention geometry differs from the pinned source");
    }
    Ok(())
}

fn vector(layer: &Mtp3Layer, tensor: MtpTensor, width: usize) -> Result<&CudaSlice<f32>, Fail> {
    let Some(MtpResidentTensor::Bf16(MtpBf16::Vector { raw, values })) = layer.tensors.get(&tensor)
    else {
        return Err(format!(
            "MiMo MTP3 depth {} lacks BF16 vector {tensor:?}",
            layer.depth
        )
        .into());
    };
    if raw.len() != width * 2 || values.len() != width {
        return Err(format!(
            "MiMo MTP3 depth {} {tensor:?} vector width changed",
            layer.depth
        )
        .into());
    }
    Ok(values)
}

fn bf16_matrix(
    layer: &Mtp3Layer,
    tensor: MtpTensor,
    input: usize,
    output: usize,
) -> Result<&GpuTensor, Fail> {
    let Some(MtpResidentTensor::Bf16(MtpBf16::Matrix(weight))) = layer.tensors.get(&tensor) else {
        return Err(format!(
            "MiMo MTP3 depth {} lacks BF16 matrix {tensor:?}",
            layer.depth
        )
        .into());
    };
    let GpuTensor::FloatBf16 { data, ne } = weight.as_ref() else {
        return Err(format!(
            "MiMo MTP3 depth {} {tensor:?} lost BF16 matrix storage",
            layer.depth
        )
        .into());
    };
    if ne.as_slice() != [input as u64, output as u64] || data.len() != input * output * 2 {
        return Err(format!(
            "MiMo MTP3 depth {} {tensor:?} matrix shape changed",
            layer.depth
        )
        .into());
    }
    Ok(weight.as_ref())
}

fn fp8_matrix(
    layer: &Mtp3Layer,
    tensor: MtpTensor,
    input: usize,
    output: usize,
) -> Result<&GpuTensor, Fail> {
    let Some(MtpResidentTensor::Fp8(weight)) = layer.tensors.get(&tensor) else {
        return Err(format!(
            "MiMo MTP3 depth {} lacks block-FP8 matrix {tensor:?}",
            layer.depth
        )
        .into());
    };
    let GpuTensor::Quant {
        bytes,
        qtype: QT_F8_E4M3_BLK,
        row_bytes,
        ne,
        scale,
        rp: false,
        fp8: None,
        rp4: None,
        blk: Some(grid),
        f16: None,
        a4: None,
        ..
    } = weight.weight.as_ref()
    else {
        return Err(format!(
            "MiMo MTP3 depth {} {tensor:?} lost native block-FP8",
            layer.depth
        )
        .into());
    };
    if weight.shape != [output as u64, input as u64]
        || weight.inverse_scale_shape != [output.div_ceil(128) as u64, input.div_ceil(128) as u64]
        || ne != &[input as u64, output as u64]
        || *row_bytes != input
        || bytes.len() != input * output
        || scale.to_bits() != 1.0f32.to_bits()
        || grid.rows != output.div_ceil(128)
        || grid.cols != input.div_ceil(128)
        || grid.scales.len() != grid.rows * grid.cols
    {
        return Err(format!(
            "MiMo MTP3 depth {} {tensor:?} FP8 grid or shape changed",
            layer.depth
        )
        .into());
    }
    Ok(weight.weight.as_ref())
}

fn validate_layer(layer: &Mtp3Layer, depth: usize) -> Result<(), Fail> {
    if layer.depth as usize != depth || layer.tensors.len() != 12 {
        return Err(format!("MiMo MTP3 draft depth {depth} has an incomplete layer").into());
    }
    for tensor in [
        MtpTensor::EmbeddingNorm,
        MtpTensor::HiddenNorm,
        MtpTensor::PreAttentionNorm,
        MtpTensor::PreMlpNorm,
        MtpTensor::OutputNorm,
    ] {
        vector(layer, tensor, HIDDEN)?;
    }
    vector(layer, MtpTensor::AttentionSink, QUERY_HEADS)?;
    bf16_matrix(layer, MtpTensor::FusionProjection, 2 * HIDDEN, HIDDEN)?;
    bf16_matrix(layer, MtpTensor::AttentionOutput, OUTPUT_WIDTH, HIDDEN)?;
    fp8_matrix(layer, MtpTensor::FusedQkv, HIDDEN, QKV_ROWS)?;
    fp8_matrix(layer, MtpTensor::MlpGate, HIDDEN, INTERMEDIATE)?;
    fp8_matrix(layer, MtpTensor::MlpUp, HIDDEN, INTERMEDIATE)?;
    fp8_matrix(layer, MtpTensor::MlpDown, INTERMEDIATE, HIDDEN)?;
    Ok(())
}

fn validate_position(
    expected: usize,
    supplied: usize,
    cached: Option<usize>,
) -> Result<(), &'static str> {
    if expected >= MAX_MTP3_CONTEXT_TOKENS {
        return Err("MiMo MTP3 draft reached its 256-token context cap");
    }
    if supplied != expected || cached != (expected > 0).then_some(expected) {
        return Err("MiMo MTP3 draft position or KV cursor drifted");
    }
    Ok(())
}

fn validate_placement(
    draft_weight_device: usize,
    draft_engine_device: usize,
    head_weight_device: usize,
    head_engine_device: usize,
) -> Result<(), &'static str> {
    if draft_weight_device != draft_engine_device {
        return Err("MiMo MTP3 draft weights and engine are on different GPUs");
    }
    if head_weight_device != head_engine_device {
        return Err("MiMo MTP3 shared head and engine are on different GPUs");
    }
    if draft_engine_device != 0 || head_engine_device != 1 {
        return Err("MiMo MTP3 requires draft stage GPU 0 and shared head GPU 1");
    }
    Ok(())
}

fn draft_embedding_token(token: u32) -> u32 {
    // NextN clamps multimodal pad sentinels before reading the shared table.
    token.min((VOCAB - 1) as u32)
}

struct KvState {
    key: CudaSlice<f32>,
    value: CudaSlice<f32>,
    tokens: usize,
}

struct DepthState {
    kv: Option<KvState>,
    position: usize,
}

fn append_kv(
    engine: &Engine,
    state: &mut DepthState,
    key: CudaSlice<f32>,
    value: CudaSlice<f32>,
) -> Result<(), Fail> {
    validate_position(
        state.position,
        state.position,
        state.kv.as_ref().map(|kv| kv.tokens),
    )?;
    let device = engine.stream().context().ordinal();
    let key_width = KV_HEADS * QK;
    let value_width = KV_HEADS * VALUE;
    if key.len() != key_width
        || value.len() != value_width
        || key.ordinal() != device
        || value.ordinal() != device
    {
        return Err("MiMo MTP3 projected KV shape or device changed".into());
    }
    let (keys, values) = if let Some(prior) = &state.kv {
        if prior.key.len() != state.position * key_width
            || prior.value.len() != state.position * value_width
            || prior.key.ordinal() != device
            || prior.value.ordinal() != device
        {
            return Err("MiMo MTP3 retained KV shape or device changed".into());
        }
        let mut keys = engine.uninit((state.position + 1) * key_width)?;
        let mut values = engine.uninit((state.position + 1) * value_width)?;
        engine.dtod_copy_into(&prior.key, &mut keys, 0)?;
        engine.dtod_copy_into(&key, &mut keys, state.position * key_width)?;
        engine.dtod_copy_into(&prior.value, &mut values, 0)?;
        engine.dtod_copy_into(&value, &mut values, state.position * value_width)?;
        (keys, values)
    } else {
        (key, value)
    };
    state.kv = Some(KvState {
        key: keys,
        value: values,
        tokens: state.position + 1,
    });
    Ok(())
}

struct Qkv {
    query: CudaSlice<f32>,
    key: CudaSlice<f32>,
    value: CudaSlice<f32>,
}

fn split_interleaved_qkv(engine: &Engine, projected: &CudaSlice<f32>) -> Result<Qkv, Fail> {
    if projected.len() != QKV_ROWS || projected.ordinal() != engine.stream().context().ordinal() {
        return Err("MiMo MTP3 fused QKV projection shape or device changed".into());
    }
    let mut query = engine.uninit(QUERY_HEADS * QK)?;
    let mut key = engine.uninit(KV_HEADS * QK)?;
    let mut value = engine.uninit(KV_HEADS * VALUE)?;
    let stream = engine.stream();
    for shard in 0..SHARDS {
        for part in [QkvPart::Query, QkvPart::Key, QkvPart::Value] {
            let (source, destination, length) = qkv_segment(shard, part)?;
            let output = match part {
                QkvPart::Query => &mut query,
                QkvPart::Key => &mut key,
                QkvPart::Value => &mut value,
            };
            stream.memcpy_dtod(
                &projected.slice(source..source + length),
                &mut output.slice_mut(destination..destination + length),
            )?;
        }
    }
    Ok(Qkv { query, key, value })
}

/// One draft depth's pre-final-norm hidden state and shared-head logits.
pub struct MiMoMtp3Step {
    pub depth: usize,
    pub position: usize,
    pub hidden_before_norm: Vec<f32>,
    pub logits: Vec<f32>,
}

/// Three independently advancing MTP cache lanes for one pinned text model.
/// The caller supplies target pre-final-norm hidden states and token IDs.
/// Each depth starts at position zero and must be primed in order.
pub struct MiMoMtp3Forward<'a> {
    draft: &'a Mtp3Weights,
    text: &'a MiMoTextWeights,
    draft_engine: &'a Engine,
    head_engine: &'a Engine,
    states: [DepthState; DEPTHS],
    failed: bool,
}

impl Mtp3Weights {
    /// Join resident MTP blocks on stage GPU 0 with the pinned text source and
    /// its shared output head on stage GPU 1. No head copy is made.
    pub fn draft_forward<'a>(
        &'a self,
        text: &'a MiMoTextWeights,
        draft_engine: &'a Engine,
        head_engine: &'a Engine,
    ) -> Result<MiMoMtp3Forward<'a>, Fail> {
        validate_forward_plan(&text.config, &text.plan)?;
        validate_contract(&text.config, &text.plan)?;
        if !text.shares_source(&self.source) {
            return Err("MiMo MTP3 draft and text weights have different source owners".into());
        }
        let GpuTensor::FloatBf16 { data, ne } = &text.output_head else {
            return Err("MiMo MTP3 needs the text model's BF16 shared output head".into());
        };
        if ne.as_slice() != [HIDDEN as u64, VOCAB as u64] || data.len() != HIDDEN * VOCAB * 2 {
            return Err("MiMo MTP3 needs the text model's resident shared output head".into());
        }
        validate_placement(
            self.device_ordinal,
            draft_engine.stream().context().ordinal(),
            data.ordinal(),
            head_engine.stream().context().ordinal(),
        )?;
        self.check_device(draft_engine)?;
        validate_residency(text, [draft_engine, head_engine])?;
        for (depth, layer) in self.layers.iter().enumerate() {
            validate_layer(layer, depth)?;
        }
        Ok(MiMoMtp3Forward {
            draft: self,
            text,
            draft_engine,
            head_engine,
            states: std::array::from_fn(|_| DepthState {
                kv: None,
                position: 0,
            }),
            failed: false,
        })
    }
}

impl MiMoMtp3Forward<'_> {
    /// Current number of fully executed steps for one draft depth.
    pub fn position(&self, depth: usize) -> Result<usize, &'static str> {
        Ok(self
            .states
            .get(depth)
            .ok_or("MiMo MTP3 draft depth is out of range")?
            .position)
    }

    /// Execute one draft token at `position` in the selected depth. The hidden
    /// input is the target or prior-draft hidden state before final RMS norm.
    /// A partial GPU failure poisons the whole draft object.
    pub fn token(
        &mut self,
        depth: usize,
        position: usize,
        token: u32,
        target_hidden_before_norm: &[f32],
    ) -> Result<MiMoMtp3Step, Fail> {
        if self.failed {
            return Err("MiMo MTP3 draft was poisoned by a failed GPU step".into());
        }
        let state = self
            .states
            .get(depth)
            .ok_or("MiMo MTP3 draft depth is out of range")?;
        validate_position(
            state.position,
            position,
            state.kv.as_ref().map(|kv| kv.tokens),
        )?;
        if target_hidden_before_norm.len() != HIDDEN
            || target_hidden_before_norm
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(
                "MiMo MTP3 target hidden state has wrong width or non-finite values".into(),
            );
        }
        let embedding = self.text.embedding_row(draft_embedding_token(token))?;
        self.failed = true;
        let (hidden_before_norm, logits) =
            self.token_inner(depth, position, &embedding, target_hidden_before_norm)?;
        self.states[depth].position += 1;
        self.failed = false;
        Ok(MiMoMtp3Step {
            depth,
            position,
            hidden_before_norm,
            logits,
        })
    }

    fn token_inner(
        &mut self,
        depth: usize,
        position: usize,
        embedding: &[f32],
        target_hidden_before_norm: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), Fail> {
        self.draft_engine.gpu.ctx.bind_to_thread()?;
        let layer = &self.draft.layers[depth];
        let epsilon = self.text.config.rms_eps;
        let embedded = self.draft_engine.htod(embedding)?;
        let target_hidden = self.draft_engine.htod(target_hidden_before_norm)?;
        let mut normalized_embedding = self.draft_engine.uninit(HIDDEN)?;
        self.draft_engine.rms_norm(
            &embedded,
            vector(layer, MtpTensor::EmbeddingNorm, HIDDEN)?,
            &mut normalized_embedding,
            HIDDEN,
            1,
            epsilon,
        )?;
        let mut normalized_hidden = self.draft_engine.uninit(HIDDEN)?;
        self.draft_engine.rms_norm(
            &target_hidden,
            vector(layer, MtpTensor::HiddenNorm, HIDDEN)?,
            &mut normalized_hidden,
            HIDDEN,
            1,
            epsilon,
        )?;
        let mut fused_input = self.draft_engine.uninit(2 * HIDDEN)?;
        self.draft_engine
            .dtod_copy_into(&normalized_embedding, &mut fused_input, 0)?;
        self.draft_engine
            .dtod_copy_into(&normalized_hidden, &mut fused_input, HIDDEN)?;
        let fused = self.draft_engine.matmul(
            bf16_matrix(layer, MtpTensor::FusionProjection, 2 * HIDDEN, HIDDEN)?,
            &fused_input,
            1,
        )?;
        let mut attn_input = self.draft_engine.uninit(HIDDEN)?;
        self.draft_engine.rms_norm(
            &fused,
            vector(layer, MtpTensor::PreAttentionNorm, HIDDEN)?,
            &mut attn_input,
            HIDDEN,
            1,
            epsilon,
        )?;
        let projected = self.draft_engine.matmul(
            fp8_matrix(layer, MtpTensor::FusedQkv, HIDDEN, QKV_ROWS)?,
            &attn_input,
            1,
        )?;
        let mut qkv = split_interleaved_qkv(self.draft_engine, &projected)?;
        let gpu_position = self.draft_engine.htod_i32(&[position as i32])?;
        let AttentionPlan::SlidingWindow { attention, .. } = &self.text.plan.layers[1].attention
        else {
            return Err("MiMo MTP3 lost its SWA attention plan".into());
        };
        self.draft_engine.rope_neox(
            &mut qkv.query,
            &gpu_position,
            QK,
            attention.rope.dimensions as usize,
            QUERY_HEADS,
            1,
            attention.rope.base,
            1.0,
        )?;
        self.draft_engine.rope_neox(
            &mut qkv.key,
            &gpu_position,
            QK,
            attention.rope.dimensions as usize,
            KV_HEADS,
            1,
            attention.rope.base,
            1.0,
        )?;
        // The source applies this before the value enters the attention cache.
        self.draft_engine
            .scale_inplace(&mut qkv.value, 0.707, KV_HEADS * VALUE)?;
        append_kv(
            self.draft_engine,
            &mut self.states[depth],
            qkv.key,
            qkv.value,
        )?;
        let cache = self.states[depth]
            .kv
            .as_ref()
            .ok_or("MiMo MTP3 lost its appended KV")?;
        let context = self.draft_engine.mimo_sink_decode(
            &qkv.query,
            &cache.key,
            &cache.value,
            Some(vector(layer, MtpTensor::AttentionSink, QUERY_HEADS)?),
            position + 1,
            &self.text.plan.layers[1].attention,
        )?;
        let attn_output = self.draft_engine.matmul(
            bf16_matrix(layer, MtpTensor::AttentionOutput, OUTPUT_WIDTH, HIDDEN)?,
            &context,
            1,
        )?;
        let mut after_attention = self.draft_engine.uninit(HIDDEN)?;
        self.draft_engine
            .add(&fused, &attn_output, &mut after_attention, HIDDEN)?;
        let mut mlp_input = self.draft_engine.uninit(HIDDEN)?;
        self.draft_engine.rms_norm(
            &after_attention,
            vector(layer, MtpTensor::PreMlpNorm, HIDDEN)?,
            &mut mlp_input,
            HIDDEN,
            1,
            epsilon,
        )?;
        let gate = self.draft_engine.matmul(
            fp8_matrix(layer, MtpTensor::MlpGate, HIDDEN, INTERMEDIATE)?,
            &mlp_input,
            1,
        )?;
        let up = self.draft_engine.matmul(
            fp8_matrix(layer, MtpTensor::MlpUp, HIDDEN, INTERMEDIATE)?,
            &mlp_input,
            1,
        )?;
        let mut activated = self.draft_engine.uninit(INTERMEDIATE)?;
        self.draft_engine
            .silu_mul(&gate, &up, &mut activated, INTERMEDIATE)?;
        let mlp_output = self.draft_engine.matmul(
            fp8_matrix(layer, MtpTensor::MlpDown, INTERMEDIATE, HIDDEN)?,
            &activated,
            1,
        )?;
        let mut before_norm = self.draft_engine.uninit(HIDDEN)?;
        self.draft_engine
            .add(&after_attention, &mlp_output, &mut before_norm, HIDDEN)?;
        let mut final_hidden = self.draft_engine.uninit(HIDDEN)?;
        self.draft_engine.rms_norm(
            &before_norm,
            vector(layer, MtpTensor::OutputNorm, HIDDEN)?,
            &mut final_hidden,
            HIDDEN,
            1,
            epsilon,
        )?;
        // Both rows cross the draft stream's completion boundary together.
        // Only the final 4096-value row is uploaded to the head device.
        let (hidden_before_norm, final_hidden_host) =
            self.draft_engine.dtoh_pair(&before_norm, &final_hidden)?;
        if hidden_before_norm.len() != HIDDEN
            || final_hidden_host.len() != HIDDEN
            || hidden_before_norm.iter().any(|value| !value.is_finite())
            || final_hidden_host.iter().any(|value| !value.is_finite())
        {
            return Err("MiMo MTP3 draft hidden row width or finite check failed".into());
        }
        drop((
            embedded,
            target_hidden,
            normalized_embedding,
            normalized_hidden,
            fused_input,
            fused,
            attn_input,
            projected,
            qkv.query,
            gpu_position,
            context,
            attn_output,
            after_attention,
            mlp_input,
            gate,
            up,
            activated,
            mlp_output,
            before_norm,
            final_hidden,
        ));
        self.head_engine.gpu.ctx.bind_to_thread()?;
        let head_input = self.head_engine.htod(&final_hidden_host)?;
        if head_input.len() != HIDDEN
            || head_input.ordinal() != self.head_engine.stream().context().ordinal()
        {
            return Err("MiMo MTP3 hidden row transfer crossed GPU devices".into());
        }
        let logits_gpu = self
            .head_engine
            .matmul(&self.text.output_head, &head_input, 1)?;
        let logits = self.head_engine.dtoh(&logits_gpu)?;
        if logits.len() != VOCAB || logits.iter().any(|value| !value.is_finite()) {
            return Err("MiMo MTP3 draft output width or finite check failed".into());
        }
        Ok((hidden_before_norm, logits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::HfConfig;
    use memra_gguf::model_packs::mimo_v2::SOURCE_PROFILE;

    fn pinned() -> (ModelConfig, ModelPlan) {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ))));
        let plan = SOURCE_PROFILE.compile_plan(&config).unwrap();
        (config, plan)
    }

    #[test]
    fn pinned_mtp3_draft_contract_rejects_depth_and_swa_drift() {
        let (mut config, mut plan) = pinned();
        validate_contract(&config, &plan).unwrap();
        config.mimo.as_mut().unwrap().separate_mtp_layers = Some(2);
        assert!(validate_contract(&config, &plan).is_err());
        config.mimo.as_mut().unwrap().separate_mtp_layers = Some(3);
        let AttentionPlan::SlidingWindow { window, .. } = &mut plan.layers[1].attention else {
            panic!("pinned layer 1 must slide");
        };
        *window = 256;
        assert!(validate_contract(&config, &plan).is_err());
    }

    #[test]
    fn qkv_checkpoint_interleave_and_scale_grid_have_exact_rows() {
        assert_eq!(QKV_ROWS, 14_848);
        assert_eq!(Q_ROWS, 3_072);
        assert_eq!(K_ROWS, 384);
        assert_eq!(V_ROWS, 256);
        assert_eq!(SHARD_ROWS % 128, 0);
        assert_eq!(QKV_ROWS.div_ceil(128), 116);
        let input: Vec<_> = (0..QKV_ROWS).collect();
        let mut q = vec![0; QUERY_HEADS * QK];
        let mut k = vec![0; KV_HEADS * QK];
        let mut v = vec![0; KV_HEADS * VALUE];
        for shard in 0..SHARDS {
            for part in [QkvPart::Query, QkvPart::Key, QkvPart::Value] {
                let (source, destination, length) = qkv_segment(shard, part).unwrap();
                let output = match part {
                    QkvPart::Query => &mut q,
                    QkvPart::Key => &mut k,
                    QkvPart::Value => &mut v,
                };
                output[destination..destination + length]
                    .copy_from_slice(&input[source..source + length]);
            }
        }
        assert_eq!(q[Q_ROWS], SHARD_ROWS);
        assert_eq!(k[0], Q_ROWS);
        assert_eq!(k[K_ROWS], SHARD_ROWS + Q_ROWS);
        assert_eq!(v[0], Q_ROWS + K_ROWS);
        assert_eq!(v[V_ROWS], SHARD_ROWS + Q_ROWS + K_ROWS);
        assert!(qkv_segment(4, QkvPart::Query).is_err());
    }

    #[test]
    fn each_draft_depth_requires_ordered_bounded_kv_positions() {
        assert!(validate_position(0, 0, None).is_ok());
        assert!(validate_position(1, 1, Some(1)).is_ok());
        assert!(validate_position(1, 0, Some(1)).is_err());
        assert!(validate_position(1, 1, None).is_err());
        assert!(
            validate_position(
                MAX_MTP3_CONTEXT_TOKENS,
                MAX_MTP3_CONTEXT_TOKENS,
                Some(MAX_MTP3_CONTEXT_TOKENS)
            )
            .is_err()
        );
    }

    #[test]
    fn mtp3_stage_and_shared_head_placement_refuses_wrong_devices() {
        assert!(validate_placement(0, 0, 1, 1).is_ok());
        assert!(validate_placement(1, 0, 1, 1).is_err());
        assert!(validate_placement(0, 0, 0, 1).is_err());
        assert!(validate_placement(0, 0, 0, 0).is_err());
        assert!(validate_placement(1, 1, 0, 0).is_err());
    }

    #[test]
    fn nextn_clamps_out_of_vocab_pad_sentinels() {
        assert_eq!(draft_embedding_token(42), 42);
        assert_eq!(draft_embedding_token(VOCAB as u32), (VOCAB - 1) as u32);
        assert_eq!(draft_embedding_token(u32::MAX), (VOCAB - 1) as u32);
    }
}
