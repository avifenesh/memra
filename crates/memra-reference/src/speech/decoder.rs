//! Native single-token Whisper decoder with per-session self KV and fixed encoder cross KV.
//! This is a reference operation, not a server route or a complete transcription policy.

use super::encoder::{
    Linear, Norm, WhisperNumeric, apply_attention, apply_gelu, apply_linear, apply_norm,
    load_linear, load_norm, tensor,
};
use crate::ReferenceTensor;
use memra_gguf::model_plan::speech::{AudioPositionKind, SpeechActivation, WhisperPlan};
use memra_gguf::model_plan::{NormKind, WeightTransform};
use memra_gguf::safetensors::StModel;
use memra_gguf::tensor_contract::{CheckpointDialect, FloatType, StorageLayout, TensorCensusEntry};

struct DecoderAttention {
    norm: Norm,
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
}

struct DecoderBlock {
    self_attention: DecoderAttention,
    cross_attention: DecoderAttention,
    mlp_norm: Norm,
    fc1: Linear,
    fc2: Linear,
}

pub struct WhisperDecoder {
    plan: WhisperPlan,
    numeric: WhisperNumeric,
    embeddings: Linear,
    positions: Vec<f32>,
    blocks: Vec<DecoderBlock>,
    output_norm: Norm,
}

struct Kv {
    key: Vec<f32>,
    value: Vec<f32>,
}

pub struct WhisperDecoderSession<'a> {
    decoder: &'a WhisperDecoder,
    encoder_rows: usize,
    position: usize,
    self_kv: Vec<Kv>,
    cross_kv: Vec<Kv>,
}

pub struct WhisperDecoderStep {
    pub position: usize,
    pub hidden: Vec<f32>,
    pub logits: Vec<f32>,
}

impl WhisperDecoder {
    pub fn load(
        plan: &WhisperPlan,
        source: &StModel,
        numeric: WhisperNumeric,
    ) -> Result<Self, String> {
        let h = plan.hidden_size as usize;
        if h == 0
            || h > 1280
            || plan.decoder_heads == 0
            || plan.decoder_heads > 20
            || !plan.hidden_size.is_multiple_of(plan.decoder_heads)
            || plan.decoder_layers == 0
            || plan.decoder_layers > 32
            || plan.decoder_ffn == 0
            || plan.decoder_ffn > 5120
            || plan.state.max_decoder_tokens == 0
            || plan.target_positions == 0
            || plan.target_positions > 448
            || plan.vocab_size == 0
            || plan.vocab_size > 51866
            || plan.norm.kind != NormKind::LayerNorm
            || plan.norm.epsilon != 1e-5
            || plan.norm.weight_transform != WeightTransform::Identity
            || plan.activation != SpeechActivation::GeluErf
            || plan.decoder_position != AudioPositionKind::LearnedAbsolute
            || !plan.attention.decoder_causal
            || !plan.attention.cross_attention
            || !plan.attention.query_bias
            || plan.attention.key_bias
            || !plan.attention.value_bias
            || !plan.attention.output_bias
            || !plan.attention.pre_norm_serial_residual
            || !plan.attention.tied_output_embedding
            || !plan.state.decoder_self_kv_per_layer
            || !plan.state.encoder_cross_kv_per_decoder_layer
            || plan.attention.qk_scale
                != 1.0 / ((plan.hidden_size / plan.decoder_heads) as f32).sqrt()
        {
            return Err("unsupported Whisper decoder program or geometry".into());
        }
        let mut contract = memra_gguf::model_packs::whisper::tensor_contract(
            plan,
            CheckpointDialect::HfSafetensors,
        )
        .map_err(|e| e.to_string())?;
        contract
            .requirements
            .retain(|r| r.names.iter().all(|n| n.starts_with("model.decoder.")));
        let census = source
            .names()
            .filter(|n| n.starts_with("model.decoder."))
            .map(|name| {
                let info = source.info(name).unwrap();
                if info.dtype != "F32" {
                    return Err(format!("decoder source tensor {name} must be F32"));
                }
                Ok(TensorCensusEntry {
                    name: name.clone(),
                    shape: info.shape.clone(),
                    storage: StorageLayout::Float(FloatType::F32),
                    physical_bytes: (info.data_offsets[1] - info.data_offsets[0]) as u64,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        contract.bind(&census).map_err(|e| e.to_string())?;
        let attention = |prefix: &str| -> Result<DecoderAttention, String> {
            Ok(DecoderAttention {
                norm: load_norm(source, &format!("{prefix}_layer_norm"), h, numeric)?,
                query: load_linear(source, &format!("{prefix}.q_proj"), h, h, true, numeric)?,
                key: load_linear(source, &format!("{prefix}.k_proj"), h, h, false, numeric)?,
                value: load_linear(source, &format!("{prefix}.v_proj"), h, h, true, numeric)?,
                out: load_linear(source, &format!("{prefix}.out_proj"), h, h, true, numeric)?,
            })
        };
        let mut blocks = Vec::new();
        for i in 0..plan.decoder_layers {
            let p = format!("model.decoder.layers.{i}");
            blocks.push(DecoderBlock {
                self_attention: attention(&format!("{p}.self_attn"))?,
                cross_attention: attention(&format!("{p}.encoder_attn"))?,
                mlp_norm: load_norm(source, &format!("{p}.final_layer_norm"), h, numeric)?,
                fc1: load_linear(
                    source,
                    &format!("{p}.fc1"),
                    h,
                    plan.decoder_ffn as usize,
                    true,
                    numeric,
                )?,
                fc2: load_linear(
                    source,
                    &format!("{p}.fc2"),
                    plan.decoder_ffn as usize,
                    h,
                    true,
                    numeric,
                )?,
            });
        }
        Ok(Self {
            plan: plan.clone(),
            numeric,
            blocks,
            embeddings: Linear {
                input: h,
                output: plan.vocab_size as usize,
                bias: None,
                weight: tensor(
                    source,
                    "model.decoder.embed_tokens.weight",
                    &[plan.vocab_size as usize, h],
                    numeric,
                )?,
            },
            positions: tensor(
                source,
                "model.decoder.embed_positions.weight",
                &[plan.target_positions as usize, h],
                numeric,
            )?,
            output_norm: load_norm(source, "model.decoder.layer_norm", h, numeric)?,
        })
    }

    /// Cross K/V belongs to this exact encoder result and this session. Starting a different
    /// utterance creates new caches; no request state is attached to shared model weights.
    pub fn start(&self, encoded: &ReferenceTensor) -> Result<WhisperDecoderSession<'_>, String> {
        let h = self.plan.hidden_size as usize;
        if encoded.shape.len() != 2
            || encoded.shape[1] != h
            || encoded.shape[0] == 0
            || encoded.shape[0] > self.plan.source_positions as usize
            || encoded.data.len() != encoded.shape[0] * h
        {
            return Err("invalid Whisper encoder result for cross attention".into());
        }
        let rows = encoded.shape[0];
        let input: Vec<_> = encoded
            .data
            .iter()
            .map(|&x| self.numeric.round(x))
            .collect();
        if input.iter().any(|x| !x.is_finite()) {
            return Err("cross-attention input is non-finite in the selected numeric class".into());
        }
        let mut cross_kv = Vec::new();
        let mut self_kv = Vec::new();
        for block in &self.blocks {
            let key = apply_linear(&input, rows, &block.cross_attention.key, self.numeric);
            let value = apply_linear(&input, rows, &block.cross_attention.value, self.numeric);
            if key.iter().chain(&value).any(|x| !x.is_finite()) {
                return Err("non-finite encoder cross KV".into());
            }
            cross_kv.push(Kv { key, value });
            self_kv.push(Kv {
                key: Vec::new(),
                value: Vec::new(),
            });
        }
        Ok(WhisperDecoderSession {
            decoder: self,
            encoder_rows: rows,
            position: 0,
            self_kv,
            cross_kv,
        })
    }
}

impl WhisperDecoderSession<'_> {
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn self_kv(&self, layer: usize) -> Option<(&[f32], &[f32])> {
        self.self_kv
            .get(layer)
            .map(|kv| (kv.key.as_slice(), kv.value.as_slice()))
    }

    pub fn cross_kv(&self, layer: usize) -> Option<(&[f32], &[f32])> {
        self.cross_kv
            .get(layer)
            .map(|kv| (kv.key.as_slice(), kv.value.as_slice()))
    }

    /// One causal token. Pending KV rows commit only after all layers/logits are finite,
    /// so a rejected token or failed numerical step cannot partially advance the session.
    pub fn step(&mut self, token: u32) -> Result<WhisperDecoderStep, String> {
        let model = self.decoder;
        let plan = &model.plan;
        let numeric = model.numeric;
        let h = plan.hidden_size as usize;
        if token >= plan.vocab_size {
            return Err("Whisper token outside vocabulary".into());
        }
        if self.position >= plan.target_positions.min(plan.state.max_decoder_tokens) as usize {
            return Err("Whisper decoder position limit reached".into());
        }
        let mut x: Vec<_> = model.embeddings.weight[token as usize * h..(token as usize + 1) * h]
            .iter()
            .zip(&model.positions[self.position * h..(self.position + 1) * h])
            .map(|(&a, &b)| numeric.round(a + b))
            .collect();
        let mut pending = Vec::with_capacity(model.blocks.len());
        for (i, block) in model.blocks.iter().enumerate() {
            let normalized = apply_norm(
                &x,
                1,
                &block.self_attention.norm,
                numeric,
                plan.norm.epsilon,
            );
            let mut query = apply_linear(&normalized, 1, &block.self_attention.query, numeric);
            for q in &mut query {
                *q = numeric.round(*q * plan.attention.qk_scale);
            }
            let new_key = apply_linear(&normalized, 1, &block.self_attention.key, numeric);
            let new_value = apply_linear(&normalized, 1, &block.self_attention.value, numeric);
            let mut keys = self.self_kv[i].key.clone();
            keys.extend_from_slice(&new_key);
            let mut values = self.self_kv[i].value.clone();
            values.extend_from_slice(&new_value);
            let context = apply_attention(
                &query,
                &keys,
                &values,
                1,
                self.position + 1,
                h,
                plan.decoder_heads as usize,
                numeric,
            );
            let residual = apply_linear(&context, 1, &block.self_attention.out, numeric);
            for (value, r) in x.iter_mut().zip(residual) {
                *value = numeric.round(*value + r);
            }
            let normalized = apply_norm(
                &x,
                1,
                &block.cross_attention.norm,
                numeric,
                plan.norm.epsilon,
            );
            let mut query = apply_linear(&normalized, 1, &block.cross_attention.query, numeric);
            for q in &mut query {
                *q = numeric.round(*q * plan.attention.qk_scale);
            }
            let context = apply_attention(
                &query,
                &self.cross_kv[i].key,
                &self.cross_kv[i].value,
                1,
                self.encoder_rows,
                h,
                plan.decoder_heads as usize,
                numeric,
            );
            let residual = apply_linear(&context, 1, &block.cross_attention.out, numeric);
            for (value, r) in x.iter_mut().zip(residual) {
                *value = numeric.round(*value + r);
            }
            let normalized = apply_norm(&x, 1, &block.mlp_norm, numeric, plan.norm.epsilon);
            let mut ff = apply_linear(&normalized, 1, &block.fc1, numeric);
            apply_gelu(&mut ff, numeric);
            let ff = apply_linear(&ff, 1, &block.fc2, numeric);
            for (value, r) in x.iter_mut().zip(ff) {
                *value = numeric.round(*value + r);
            }
            if x.iter().any(|x| !x.is_finite()) {
                return Err(format!("non-finite Whisper decoder layer {i}"));
            }
            pending.push(Kv {
                key: new_key,
                value: new_value,
            });
        }
        let hidden = apply_norm(&x, 1, &model.output_norm, numeric, plan.norm.epsilon);
        let logits = apply_linear(&hidden, 1, &model.embeddings, numeric);
        if logits.iter().any(|x| !x.is_finite()) {
            return Err("non-finite Whisper decoder logits".into());
        }
        for (cache, new) in self.self_kv.iter_mut().zip(pending) {
            cache.key.extend(new.key);
            cache.value.extend(new.value);
        }
        let position = self.position;
        self.position += 1;
        Ok(WhisperDecoderStep {
            position,
            hidden,
            logits,
        })
    }
}

#[cfg(test)]
mod tests;
