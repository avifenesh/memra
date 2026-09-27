//! One-token MiMo text forward over the pinned, two-card resident weights.
//! This is a bounded f32-KV oracle path, not a prefill or serving backend.

use std::error::Error;

use cudarc::driver::CudaSlice;
use memra_gguf::config::ModelConfig;
use memra_gguf::model_plan::{
    ActivationPlan, AttentionPlan, FullAttentionPlan, LayerPlan, MlpPlan, ModelPlan, NormKind,
    WeightTransform,
};

use crate::Engine;
use crate::mimo_attn_load::MiMoAttentionGeometry;
use crate::mimo_text_weights::{
    MiMoDenseWeights, MiMoTextLayerWeights, MiMoTextWeights, stage_for_layer,
};

type Fail = Box<dyn Error>;
const HIDDEN: usize = 4096;
const VOCAB: usize = 152_576;
const LAYERS: usize = 48;
const QK: usize = 192;
const VALUE: usize = 128;
const QUERY_HEADS: usize = 64;
const STAGE_CUT: usize = 24;

/// The f32 KV oracle retains every token, including tokens outside a sliding
/// layer's 128-token attention window. It copies the prior rows on append.
pub const MAX_TEXT_CONTEXT_TOKENS: usize = 256;

struct KvState {
    key: CudaSlice<f32>,
    value: CudaSlice<f32>,
    tokens: usize,
}

/// One continuing text sequence. Every cache row remains on its layer's GPU.
/// A failed GPU step poisons the sequence because earlier layers may have
/// already appended the token.
pub struct MiMoTextForward<'a> {
    weights: &'a MiMoTextWeights,
    engines: [&'a Engine; 2],
    kv: Vec<Option<KvState>>,
    position: usize,
    failed: bool,
}

pub(crate) fn validate_forward_plan(config: &ModelConfig, plan: &ModelPlan) -> Result<(), Fail> {
    if ModelPlan::compile(config)? != *plan
        || plan.hidden_size as usize != HIDDEN
        || plan.vocab_size as usize != VOCAB
        || plan.layers.len() != LAYERS
        || plan.embedding_scale.to_bits() != 1.0f32.to_bits()
        || !plan.logits.is_empty()
        || plan.output_norm.kind != NormKind::Rms
        || plan.output_norm.weight_transform != WeightTransform::Identity
        || !plan.partition_boundaries.contains(&STAGE_CUT)
    {
        return Err("MiMo text forward requires the pinned source plan".into());
    }
    for (index, layer) in plan.layers.iter().enumerate() {
        if layer.index as usize != index
            || layer.pre_attention_norm.kind != NormKind::Rms
            || layer.pre_attention_norm.weight_transform != WeightTransform::Identity
            || layer.pre_mlp_norm.kind != NormKind::Rms
            || layer.pre_mlp_norm.weight_transform != WeightTransform::Identity
        {
            return Err(format!("MiMo layer {index} norm or index changed").into());
        }
        MiMoAttentionGeometry::from_plan(plan, index)?;
        match (&layer.mlp, index) {
            (MlpPlan::Dense(dense), 0)
                if dense.intermediate_size == 16_384
                    && dense.activation == ActivationPlan::Silu => {}
            (MlpPlan::Moe(moe), 1..)
                if moe.expert_count == 256
                    && moe.experts_per_token == 8
                    && moe.expert_intermediate_size == 2048
                    && moe.activation == ActivationPlan::Silu
                    && moe.shared.is_none() => {}
            _ => return Err(format!("MiMo layer {index} MLP changed").into()),
        }
    }
    Ok(())
}

pub(crate) fn validate_residency(
    weights: &MiMoTextWeights,
    engines: [&Engine; 2],
) -> Result<(), Fail> {
    let devices = [
        engines[0].stream().context().ordinal(),
        engines[1].stream().context().ordinal(),
    ];
    if devices[0] == devices[1] {
        return Err("MiMo text forward requires two distinct GPU devices".into());
    }
    if weights.layers.len() != LAYERS || weights.routed.len() != LAYERS {
        return Err("MiMo text forward has an incomplete resident layer table".into());
    }
    for index in 0..LAYERS {
        let device = devices[stage_for_layer(index)?];
        let row = &weights.layers[index];
        let geometry = MiMoAttentionGeometry::from_plan(&weights.plan, index)?;
        if row.attention.geometry != geometry
            || row.attention_norm.len() != HIDDEN
            || row.attention_norm.ordinal() != device
            || row.mlp_norm.len() != HIDDEN
            || row.mlp_norm.ordinal() != device
            || row.attention.qkv.len() != 4
            || row.attention.qkv.iter().any(|shard| {
                shard.in_features() != HIDDEN
                    || shard.out_features() != geometry.shard_rows
                    || shard.ordinal() != device
            })
            || row.attention.output.in_features() != geometry.output_width
            || row.attention.output.out_features() != HIDDEN
            || row.attention.output.ordinal() != device
            || row.attention.sink.is_some() != (geometry.window != 0)
            || row
                .attention
                .sink
                .as_ref()
                .is_some_and(|sink| sink.len() != QUERY_HEADS || sink.ordinal() != device)
            || row.dense.is_some() != (index == 0)
            || weights.routed[index].is_some() != (index != 0)
        {
            return Err(format!("MiMo layer {index} residency or geometry changed").into());
        }
        if let Some(dense) = &row.dense
            && (dense.gate.in_features() != HIDDEN
                || dense.gate.out_features() != 16_384
                || dense.up.in_features() != HIDDEN
                || dense.up.out_features() != 16_384
                || dense.down.in_features() != 16_384
                || dense.down.out_features() != HIDDEN
                || [
                    dense.gate.ordinal(),
                    dense.up.ordinal(),
                    dense.down.ordinal(),
                ]
                .into_iter()
                .any(|ordinal| ordinal != device))
        {
            return Err("MiMo layer 0 dense residency changed".into());
        }
    }
    if weights.output_norm.len() != HIDDEN
        || weights.output_norm.ordinal() != devices[1]
        || weights.output_head.in_features() != HIDDEN
        || weights.output_head.out_features() != VOCAB
        || weights.output_head.ordinal() != devices[1]
    {
        return Err("MiMo text output residency changed".into());
    }
    Ok(())
}

fn validate_cache_position(position: usize, cached: Option<usize>) -> Result<(), &'static str> {
    if position >= MAX_TEXT_CONTEXT_TOKENS {
        return Err("MiMo f32 KV oracle reached its 256-token context cap");
    }
    if cached != (position > 0).then_some(position) {
        return Err("MiMo text KV does not match the current token position");
    }
    Ok(())
}

fn append_kv(
    engine: &Engine,
    slot: &mut Option<KvState>,
    position: usize,
    key: CudaSlice<f32>,
    value: CudaSlice<f32>,
    kv_heads: usize,
) -> Result<(), Fail> {
    validate_cache_position(position, slot.as_ref().map(|state| state.tokens))?;
    let key_width = kv_heads * QK;
    let value_width = kv_heads * VALUE;
    let device = engine.stream().context().ordinal();
    if key.len() != key_width
        || value.len() != value_width
        || key.ordinal() != device
        || value.ordinal() != device
    {
        return Err("MiMo current-token KV geometry or device changed".into());
    }
    let (keys, values) = if let Some(prior) = slot.as_ref() {
        if prior.key.len() != position * key_width
            || prior.value.len() != position * value_width
            || prior.key.ordinal() != device
            || prior.value.ordinal() != device
        {
            return Err("MiMo retained KV geometry or device changed".into());
        }
        let mut keys = engine.uninit((position + 1) * key_width)?;
        let mut values = engine.uninit((position + 1) * value_width)?;
        engine.dtod_copy_into(&prior.key, &mut keys, 0)?;
        engine.dtod_copy_into(&key, &mut keys, position * key_width)?;
        engine.dtod_copy_into(&prior.value, &mut values, 0)?;
        engine.dtod_copy_into(&value, &mut values, position * value_width)?;
        (keys, values)
    } else {
        (key, value)
    };
    *slot = Some(KvState {
        key: keys,
        value: values,
        tokens: position + 1,
    });
    Ok(())
}

pub(crate) fn normalized(
    engine: &Engine,
    input: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    epsilon: f32,
) -> Result<CudaSlice<f32>, Fail> {
    let mut result = engine.uninit(HIDDEN)?;
    engine.rms_norm(input, weight, &mut result, HIDDEN, 1, epsilon)?;
    Ok(result)
}

fn full_attention(plan: &LayerPlan) -> Result<&FullAttentionPlan, Fail> {
    match &plan.attention {
        AttentionPlan::Full(attention) | AttentionPlan::SlidingWindow { attention, .. } => {
            Ok(attention)
        }
        _ => Err("MiMo text layer lost its full or sliding attention".into()),
    }
}

fn attention_token(
    engine: &Engine,
    row: &MiMoTextLayerWeights,
    plan: &LayerPlan,
    hidden: &CudaSlice<f32>,
    slot: &mut Option<KvState>,
    position: usize,
) -> Result<CudaSlice<f32>, Fail> {
    let attention = full_attention(plan)?;
    let norm = normalized(
        engine,
        hidden,
        &row.attention_norm,
        plan.pre_attention_norm.epsilon,
    )?;
    let projections = row
        .attention
        .qkv
        .iter()
        .map(|shard| engine.matmul(shard, &norm, 1))
        .collect::<Result<Vec<_>, _>>()?;
    let mut qkv = engine.mimo_gather_qkv(
        [
            &projections[0],
            &projections[1],
            &projections[2],
            &projections[3],
        ],
        1,
        attention,
    )?;
    let gpu_position = engine.htod_i32(&[position as i32])?;
    engine.rope_neox(
        &mut qkv.query,
        &gpu_position,
        QK,
        64,
        QUERY_HEADS,
        1,
        attention.rope.base,
        1.0,
    )?;
    engine.rope_neox(
        &mut qkv.key,
        &gpu_position,
        QK,
        64,
        row.attention.geometry.kv_heads,
        1,
        attention.rope.base,
        1.0,
    )?;
    append_kv(
        engine,
        slot,
        position,
        qkv.key,
        qkv.value,
        row.attention.geometry.kv_heads,
    )?;
    let cache = slot.as_ref().ok_or("MiMo attention lost its appended KV")?;
    let context = engine.mimo_sink_decode(
        &qkv.query,
        &cache.key,
        &cache.value,
        row.attention.sink.as_ref(),
        position + 1,
        &plan.attention,
    )?;
    engine.matmul(&row.attention.output, &context, 1)
}

pub(crate) fn dense_token(
    engine: &Engine,
    input: &CudaSlice<f32>,
    weights: &MiMoDenseWeights,
) -> Result<CudaSlice<f32>, Fail> {
    let gate = engine.matmul(&weights.gate, input, 1)?;
    let up = engine.matmul(&weights.up, input, 1)?;
    let mut activated = engine.uninit(16_384)?;
    engine.silu_mul(&gate, &up, &mut activated, 16_384)?;
    engine.matmul(&weights.down, &activated, 1)
}

impl MiMoTextWeights {
    /// Start a separate f32-KV oracle sequence over this resident model.
    pub fn text_forward<'a>(
        &'a self,
        engines: [&'a Engine; 2],
    ) -> Result<MiMoTextForward<'a>, Fail> {
        MiMoTextForward::new(self, engines)
    }
}

impl<'a> MiMoTextForward<'a> {
    fn new(weights: &'a MiMoTextWeights, engines: [&'a Engine; 2]) -> Result<Self, Fail> {
        validate_forward_plan(&weights.config, &weights.plan)?;
        validate_residency(weights, engines)?;
        Ok(Self {
            weights,
            engines,
            kv: std::iter::repeat_with(|| None).take(LAYERS).collect(),
            position: 0,
            failed: false,
        })
    }

    /// Number of fully committed token steps. Each successful call advances
    /// all 48 native KV states once.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Process one source token and return all 152,576 f32 logits. The token
    /// must be a text-embedding row; modality placeholders need a separate
    /// injection path. This diagnostic path has no sampler or batching.
    pub fn token(&mut self, token: u32) -> Result<Vec<f32>, Fail> {
        if self.failed {
            return Err("MiMo text sequence was poisoned by a failed GPU step".into());
        }
        for slot in &self.kv {
            validate_cache_position(self.position, slot.as_ref().map(|state| state.tokens))?;
        }
        // Source-row and token validation happen before any GPU or KV mutation.
        let initial = self.weights.embedding_row(token)?;
        self.failed = true;
        let logits = self.token_inner(&initial)?;
        self.position += 1;
        self.failed = false;
        Ok(logits)
    }

    fn token_inner(&mut self, initial: &[f32]) -> Result<Vec<f32>, Fail> {
        self.engines[0].gpu.ctx.bind_to_thread()?;
        let mut hidden = self.engines[0].htod(initial)?;
        for index in 0..LAYERS {
            if index == STAGE_CUT {
                let values = self.engines[0].dtoh(&hidden)?;
                if values.len() != HIDDEN || values.iter().any(|value| !value.is_finite()) {
                    return Err("MiMo stage transfer carried invalid hidden values".into());
                }
                self.engines[1].gpu.ctx.bind_to_thread()?;
                hidden = self.engines[1].htod(&values)?;
            }
            let stage = stage_for_layer(index)?;
            let engine = self.engines[stage];
            engine.gpu.ctx.bind_to_thread()?;
            let plan = &self.weights.plan.layers[index];
            let row = &self.weights.layers[index];
            let attention = attention_token(
                engine,
                row,
                plan,
                &hidden,
                &mut self.kv[index],
                self.position,
            )?;
            let mut after_attention = engine.uninit(HIDDEN)?;
            engine.add(&hidden, &attention, &mut after_attention, HIDDEN)?;
            let mlp_input = normalized(
                engine,
                &after_attention,
                &row.mlp_norm,
                plan.pre_mlp_norm.epsilon,
            )?;
            let mlp = match &plan.mlp {
                MlpPlan::Dense(_) if index == 0 => dense_token(
                    engine,
                    &mlp_input,
                    row.dense
                        .as_ref()
                        .ok_or("MiMo layer 0 dense weights missing")?,
                )?,
                MlpPlan::Moe(moe) if index > 0 => {
                    self.weights.routed[index]
                        .as_ref()
                        .ok_or("MiMo routed weights missing")?
                        .token_bound(
                            engine,
                            &mlp_input,
                            moe,
                            &self.weights.config,
                            &self.weights.plan,
                        )?
                        .output
                }
                _ => return Err(format!("MiMo layer {index} MLP changed").into()),
            };
            let mut after_mlp = engine.uninit(HIDDEN)?;
            engine.add(&after_attention, &mlp, &mut after_mlp, HIDDEN)?;
            hidden = after_mlp;
        }
        let last = self.engines[1];
        let final_norm = normalized(
            last,
            &hidden,
            &self.weights.output_norm,
            self.weights.plan.output_norm.epsilon,
        )?;
        let logits_gpu = last.matmul(&self.weights.output_head, &final_norm, 1)?;
        let logits = last.dtoh(&logits_gpu)?;
        if logits.len() != VOCAB || logits.iter().any(|value| !value.is_finite()) {
            return Err("MiMo text output logits are non-finite or wrong width".into());
        }
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::HfConfig;
    use memra_gguf::model_packs::mimo_v2::SOURCE_PROFILE;

    fn pinned_plan() -> (ModelConfig, ModelPlan) {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ))));
        let plan = SOURCE_PROFILE.compile_plan(&config).unwrap();
        (config, plan)
    }

    #[test]
    fn pinned_forward_plan_has_exact_stage_and_attention_boundaries() {
        let (config, plan) = pinned_plan();
        validate_forward_plan(&config, &plan).unwrap();
        assert_eq!(stage_for_layer(0), Ok(0));
        assert_eq!(stage_for_layer(23), Ok(0));
        assert_eq!(stage_for_layer(24), Ok(1));
        assert_eq!(stage_for_layer(47), Ok(1));
        assert!(stage_for_layer(48).is_err());
        assert_eq!(
            MiMoAttentionGeometry::from_plan(&plan, 23).unwrap().window,
            0
        );
        assert_eq!(
            MiMoAttentionGeometry::from_plan(&plan, 24).unwrap().window,
            128
        );
    }

    #[test]
    fn changed_forward_plan_is_rejected() {
        let (config, mut plan) = pinned_plan();
        plan.partition_boundaries.clear();
        assert!(validate_forward_plan(&config, &plan).is_err());
        let (_, mut plan) = pinned_plan();
        plan.layers[24].pre_mlp_norm.weight_transform = WeightTransform::AddOne;
        assert!(validate_forward_plan(&config, &plan).is_err());
    }

    #[test]
    fn f32_kv_requires_every_prior_layer_row_and_honors_cap() {
        assert!(validate_cache_position(0, None).is_ok());
        assert!(validate_cache_position(0, Some(0)).is_err());
        assert!(validate_cache_position(1, Some(1)).is_ok());
        assert!(validate_cache_position(1, None).is_err());
        assert!(validate_cache_position(24, Some(23)).is_err());
        assert!(validate_cache_position(MAX_TEXT_CONTEXT_TOKENS - 1, Some(255)).is_ok());
        assert!(validate_cache_position(MAX_TEXT_CONTEXT_TOKENS, Some(256)).is_err());
    }
}
