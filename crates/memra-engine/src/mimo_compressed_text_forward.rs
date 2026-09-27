//! One-token MiMo text forward with model-owned compressed KV.
//! This diagnostic executor has no modality, batching, or serving dispatch.

use std::error::Error;

use memra_gguf::model_plan::{AttentionPlan, MlpPlan};

use crate::Engine;
use crate::mimo_compressed_kv::{MAX_COMPRESSED_CONTEXT_TOKENS, MiMoCompressedKv};
use crate::mimo_text_forward::{dense_token, normalized};
use crate::mimo_text_weights::{MiMoTextWeights, stage_for_layer};

type Fail = Box<dyn Error>;
const HIDDEN: usize = 4096;
const VOCAB: usize = 152_576;
const LAYERS: usize = 48;
const STAGE_CUT: usize = 24;

fn check_step(
    position: usize,
    kv_position: usize,
    max_tokens: usize,
    failed: bool,
) -> Result<(), &'static str> {
    if failed {
        return Err("MiMo compressed text sequence was poisoned by a failed GPU step");
    }
    if !(1..=MAX_COMPRESSED_CONTEXT_TOKENS).contains(&max_tokens) || position >= max_tokens {
        return Err("MiMo compressed text reached its admitted context cap");
    }
    if kv_position != position {
        return Err("MiMo compressed text KV position drifted");
    }
    Ok(())
}

/// One continuing source-text sequence. `position` counts completed logits
/// steps, while the KV cursor may advance before a later layer or head fails.
pub struct MiMoCompressedTextForward<'a> {
    weights: &'a MiMoTextWeights,
    engines: [&'a Engine; 2],
    kv: MiMoCompressedKv<'a>,
    max_tokens: usize,
    position: usize,
    failed: bool,
}

impl MiMoTextWeights {
    /// Admit a text-only compressed-KV sequence on the resident two-card model.
    /// Input and generated tokens share `max_tokens`. The KV constructor
    /// enforces its capacity and caller-supplied per-card free headroom.
    pub fn compressed_text_forward<'a>(
        &'a self,
        engines: [&'a Engine; 2],
        max_tokens: usize,
        min_free_after: [usize; 2],
    ) -> Result<MiMoCompressedTextForward<'a>, Fail> {
        MiMoCompressedTextForward::new(self, engines, max_tokens, min_free_after)
    }
}

impl<'a> MiMoCompressedTextForward<'a> {
    fn new(
        weights: &'a MiMoTextWeights,
        engines: [&'a Engine; 2],
        max_tokens: usize,
        min_free_after: [usize; 2],
    ) -> Result<Self, Fail> {
        let kv = weights.compressed_text_kv(engines, max_tokens, min_free_after)?;
        Ok(Self {
            weights,
            engines,
            kv,
            max_tokens,
            position: 0,
            failed: false,
        })
    }

    /// Number of complete 48-layer token and logits steps.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Process one source text token and return all 152,576 f32 logits.
    /// A failed GPU, transfer, or head step poisons this sequence because its
    /// 48 per-layer KV writes cannot be rolled back.
    pub fn token(&mut self, token: u32) -> Result<Vec<f32>, Fail> {
        check_step(
            self.position,
            self.kv.position(),
            self.max_tokens,
            self.failed,
        )?;
        // Validate the source row before any GPU or KV mutation.
        let initial = self.weights.embedding_row(token)?;
        self.failed = true;
        let logits = self.token_inner(&initial)?;
        if self.kv.position() != self.position + 1 {
            return Err("MiMo compressed text did not commit all 48 KV rows".into());
        }
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
            let attention = match &plan.attention {
                AttentionPlan::Full(attention) | AttentionPlan::SlidingWindow { attention, .. } => {
                    attention
                }
                _ => return Err("MiMo text layer lost its full or sliding attention".into()),
            };
            let norm = normalized(
                engine,
                &hidden,
                &row.attention_norm,
                plan.pre_attention_norm.epsilon,
            )?;
            let projections = row
                .attention
                .qkv
                .iter()
                .map(|shard| engine.matmul(shard, &norm, 1))
                .collect::<Result<Vec<_>, _>>()?;
            // The gather kernel applies the pinned 0.707 V scale before
            // either compressed or f32 KV append. Do not scale V again.
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
            let gpu_position = engine.htod_i32(&[self.position as i32])?;
            engine.rope_neox(
                &mut qkv.query,
                &gpu_position,
                192,
                64,
                64,
                1,
                attention.rope.base,
                1.0,
            )?;
            engine.rope_neox(
                &mut qkv.key,
                &gpu_position,
                192,
                64,
                row.attention.geometry.kv_heads,
                1,
                attention.rope.base,
                1.0,
            )?;
            let context = self
                .kv
                .append_and_attend(index, &qkv.query, &qkv.key, &qkv.value)?;
            // append_and_attend synchronizes attention before these async
            // projection and RoPE inputs are released.
            drop((qkv, gpu_position, projections, norm));
            let attention = engine.matmul(&row.attention.output, &context, 1)?;
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
    use crate::mimo_compressed_kv::MiMoCompressedKvBudget;
    use memra_gguf::config::{HfConfig, ModelConfig};
    use memra_gguf::model_packs::mimo_v2::SOURCE_PROFILE;
    use memra_gguf::model_plan::AttentionPlan;

    #[test]
    fn compressed_forward_plan_keeps_source_value_scale_and_stage_census() {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ))));
        let plan = SOURCE_PROFILE.compile_plan(&config).unwrap();
        MiMoCompressedKvBudget::for_plan(&config, &plan, MAX_COMPRESSED_CONTEXT_TOKENS).unwrap();
        let mut full = [0; 2];
        let mut local = [0; 2];
        for (index, layer) in plan.layers.iter().enumerate() {
            let stage = stage_for_layer(index).unwrap();
            let (attention, is_full) = match &layer.attention {
                AttentionPlan::Full(attention) => (attention, true),
                AttentionPlan::SlidingWindow { attention, .. } => (attention, false),
                _ => panic!("MiMo text layer {index} lost full or sliding attention"),
            };
            assert_eq!(
                attention
                    .mimo_math
                    .unwrap()
                    .value_scale_before_cache
                    .to_bits(),
                0.707f32.to_bits()
            );
            if is_full {
                full[stage] += 1;
            } else {
                local[stage] += 1;
            }
        }
        assert_eq!(full, [5, 4]);
        assert_eq!(local, [19, 20]);
    }

    #[test]
    fn compressed_forward_rejects_cap_kv_drift_and_poison() {
        assert!(check_step(0, 0, 1, false).is_ok());
        assert!(check_step(1, 1, 1, false).is_err());
        assert!(check_step(0, 1, 2, false).is_err());
        assert!(check_step(0, 0, 2, true).is_err());
        assert_eq!(
            check_step(0, 1, 2, true).unwrap_err(),
            "MiMo compressed text sequence was poisoned by a failed GPU step"
        );
        assert!(check_step(0, 0, 0, false).is_err());
        assert!(check_step(0, 0, MAX_COMPRESSED_CONTEXT_TOKENS + 1, false).is_err());
        assert!(
            check_step(
                MAX_COMPRESSED_CONTEXT_TOKENS - 1,
                MAX_COMPRESSED_CONTEXT_TOKENS - 1,
                MAX_COMPRESSED_CONTEXT_TOKENS,
                false,
            )
            .is_ok()
        );
        assert!(
            check_step(
                MAX_COMPRESSED_CONTEXT_TOKENS,
                MAX_COMPRESSED_CONTEXT_TOKENS,
                MAX_COMPRESSED_CONTEXT_TOKENS,
                false,
            )
            .is_err()
        );
    }
}
