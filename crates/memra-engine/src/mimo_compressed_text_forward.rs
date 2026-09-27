//! One-token MiMo text forward with model-owned compressed KV.
//! Prepared modal chunks may enter a fresh sequence; there is no raw-modal
//! decoder, batched prefill, payload-keyed KV reuse, or serving dispatch.

use std::error::Error;

use cudarc::driver::CudaSlice;
use memra_gguf::model_plan::{AttentionPlan, MlpPlan};

use crate::Engine;
use crate::mimo_compressed_kv::{MAX_COMPRESSED_CONTEXT_TOKENS, MiMoCompressedKv};
use crate::mimo_modal_overlay::MiMoGpuEmbeddingChunk;
use crate::mimo_text_forward::{MiMoTextStep, dense_token, normalized, read_text_output};
use crate::mimo_text_weights::{MiMoDenseWeights, MiMoTextWeights, stage_for_layer};
use crate::model::GpuTensor;

type Fail = Box<dyn Error>;
const HIDDEN: usize = 4096;
const VOCAB: usize = 152_576;
const LAYERS: usize = 48;
const STAGE_CUT: usize = 24;
const MAX_FIRST_BATCH_CHUNK: usize = 128;

fn first_batch_rows(tokens: usize) -> Result<usize, &'static str> {
    if !(1..=MAX_FIRST_BATCH_CHUNK).contains(&tokens) {
        return Err("MiMo first batch chunk must contain 1..=128 tokens");
    }
    Ok(tokens)
}

fn normalized_rows(
    engine: &Engine,
    input: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    rows: usize,
    epsilon: f32,
) -> Result<CudaSlice<f32>, Fail> {
    if first_batch_rows(rows).is_err()
        || input.len() != rows * HIDDEN
        || weight.len() != HIDDEN
        || input.ordinal() != engine.stream().context().ordinal()
        || weight.ordinal() != input.ordinal()
    {
        return Err("MiMo batch RMS input width, rows, or GPU changed".into());
    }
    let mut output = engine.uninit(rows * HIDDEN)?;
    engine.rms_norm(input, weight, &mut output, HIDDEN, rows, epsilon)?;
    Ok(output)
}

fn dense_rows(
    engine: &Engine,
    input: &CudaSlice<f32>,
    weights: &MiMoDenseWeights,
    rows: usize,
) -> Result<CudaSlice<f32>, Fail> {
    let gate = engine.matmul(&weights.gate, input, rows)?;
    let up = engine.matmul(&weights.up, input, rows)?;
    let mut activated = engine.uninit(rows * 16_384)?;
    engine.silu_mul(&gate, &up, &mut activated, rows * 16_384)?;
    engine.matmul(&weights.down, &activated, rows)
}

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

fn check_chunk_admission(
    position: usize,
    kv_position: usize,
    max_tokens: usize,
    failed: bool,
    tokens: usize,
    elements: usize,
    stages: [usize; 2],
) -> Result<(), &'static str> {
    check_step(position, kv_position, max_tokens, failed)?;
    if position != 0
        || tokens == 0
        || tokens > max_tokens
        || elements != tokens * HIDDEN
        || stages[0] != stages[1]
    {
        return Err("MiMo embedding chunk must fit a fresh stage-0 text sequence");
    }
    Ok(())
}

/// One continuing source-text sequence with optional prepared modal rows.
/// `position` counts completed logits
/// steps, while the KV cursor may advance before a later layer or head fails.
pub struct MiMoCompressedTextForward<'a> {
    weights: &'a MiMoTextWeights,
    engines: [&'a Engine; 2],
    kv: MiMoCompressedKv<'a>,
    max_tokens: usize,
    position: usize,
    failed: bool,
    has_modal_payload: bool,
    #[cfg(test)]
    trace_position: Option<usize>,
    #[cfg(test)]
    trace_rows: Vec<(usize, Vec<f32>)>,
}

impl MiMoTextWeights {
    /// Admit a compressed-KV sequence on the resident two-card model.
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
            has_modal_payload: false,
            #[cfg(test)]
            trace_position: None,
            #[cfg(test)]
            trace_rows: Vec::new(),
        })
    }

    /// Number of complete 48-layer token and logits steps.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Whether this sequence consumed prepared modal embeddings. Any future
    /// external KV key must include the payload, not only placeholder IDs.
    pub fn has_modal_payload(&self) -> bool {
        self.has_modal_payload
    }

    /// Process one source text token and return all 152,576 f32 logits.
    /// A failed GPU, transfer, or head step poisons this sequence because its
    /// 48 per-layer KV writes cannot be rolled back.
    pub fn token(&mut self, token: u32) -> Result<Vec<f32>, Fail> {
        Ok(self.token_step::<false>(token)?.0)
    }

    /// Process one source text token and return complete logits plus its
    /// pre-final-norm hidden row for the bounded MTP3 drafter.
    pub fn token_with_hidden(&mut self, token: u32) -> Result<MiMoTextStep, Fail> {
        let position = self.position;
        let (logits, hidden) = self.token_step::<true>(token)?;
        Ok(MiMoTextStep {
            position,
            logits,
            hidden_before_norm: hidden.ok_or("MiMo compressed text hidden capture was omitted")?,
        })
    }

    /// Consume one source-ordered, stage-0 prepared embedding chunk on a
    /// fresh sequence. Each row executes the pinned 48-layer token path and
    /// commits its model-owned KV. Return final logits and the hidden row
    /// needed by the separate MTP3 drafter. This is serial diagnostic prefill.
    pub fn consume_embedding_chunk(
        &mut self,
        chunk: &MiMoGpuEmbeddingChunk,
    ) -> Result<MiMoTextStep, Fail> {
        check_chunk_admission(
            self.position,
            self.kv.position(),
            self.max_tokens,
            self.failed,
            chunk.token_count(),
            chunk.embeddings().len(),
            [
                chunk.embeddings().ordinal(),
                self.engines[0].stream().context().ordinal(),
            ],
        )?;
        let tokens = chunk.token_count();
        for index in 0..tokens - 1 {
            self.chunk_row::<false>(chunk, index)?;
        }
        let position = self.position;
        let (logits, hidden) = self.chunk_row::<true>(chunk, tokens - 1)?;
        self.has_modal_payload = chunk.requires_payload_identity();
        Ok(MiMoTextStep {
            position,
            logits,
            hidden_before_norm: hidden.ok_or("MiMo modal final hidden capture was omitted")?,
        })
    }

    /// Execute one fresh source-ordered chunk with row-batched projections,
    /// attention, and resident MoE. Only the first 1..=128 positions are
    /// admitted: the current chunk attention has no preceding cache input.
    /// Each layer stores Q8_0 K / NVFP4 V or local F32 rows for later decode.
    ///
    /// The fresh chunk attends with pre-quantized F32 K/V, so this is a
    /// separately qualified arithmetic path. It is never selected by the
    /// ordinary serial `consume_embedding_chunk` or customer serving.
    pub fn consume_embedding_chunk_batched(
        &mut self,
        chunk: &MiMoGpuEmbeddingChunk,
    ) -> Result<MiMoTextStep, Fail> {
        let tokens = first_batch_rows(chunk.token_count())?;
        check_chunk_admission(
            self.position,
            self.kv.position(),
            self.max_tokens,
            self.failed,
            tokens,
            chunk.embeddings().len(),
            [
                chunk.embeddings().ordinal(),
                self.engines[0].stream().context().ordinal(),
            ],
        )?;
        self.kv.begin_prefill_batch(tokens)?;
        self.failed = true;
        let positions = (0..tokens)
            .map(|position| position as i32)
            .collect::<Vec<_>>();
        let position_ids = [
            self.engines[0].htod_i32(&positions)?,
            self.engines[1].htod_i32(&positions)?,
        ];
        let first = self.engines[0];
        first.gpu.ctx.bind_to_thread()?;
        let mut hidden = first.uninit(tokens * HIDDEN)?;
        first
            .stream()
            .memcpy_dtod(chunk.embeddings(), &mut hidden)?;

        for index in 0..LAYERS {
            if index == STAGE_CUT {
                let values = first.dtoh(&hidden)?;
                if values.len() != tokens * HIDDEN || values.iter().any(|value| !value.is_finite())
                {
                    return Err("MiMo batch stage transfer carried invalid hidden values".into());
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
                _ => return Err("MiMo batch layer lost its full or sliding attention".into()),
            };
            let norm = normalized_rows(
                engine,
                &hidden,
                &row.attention_norm,
                tokens,
                plan.pre_attention_norm.epsilon,
            )?;
            let projections = row
                .attention
                .qkv
                .iter()
                .map(|shard| engine.matmul(shard, &norm, tokens))
                .collect::<Result<Vec<_>, _>>()?;
            let mut qkv = engine.mimo_gather_qkv(
                [
                    &projections[0],
                    &projections[1],
                    &projections[2],
                    &projections[3],
                ],
                tokens,
                attention,
            )?;
            engine.rope_neox(
                &mut qkv.query,
                &position_ids[stage],
                192,
                64,
                64,
                tokens,
                attention.rope.base,
                1.0,
            )?;
            engine.rope_neox(
                &mut qkv.key,
                &position_ids[stage],
                192,
                64,
                row.attention.geometry.kv_heads,
                tokens,
                attention.rope.base,
                1.0,
            )?;
            let context = engine.mimo_text_chunk_attention(
                &plan.attention,
                &qkv.query,
                &qkv.key,
                &qkv.value,
                row.attention.sink.as_ref(),
                tokens,
            )?;
            self.kv.append_prefill_layer(index, &qkv.key, &qkv.value)?;
            drop((qkv, projections, norm));
            let attention_output = engine.matmul(&row.attention.output, &context, tokens)?;
            let mut after_attention = engine.uninit(tokens * HIDDEN)?;
            engine.add(
                &hidden,
                &attention_output,
                &mut after_attention,
                tokens * HIDDEN,
            )?;
            let mlp_input = normalized_rows(
                engine,
                &after_attention,
                &row.mlp_norm,
                tokens,
                plan.pre_mlp_norm.epsilon,
            )?;
            let mlp = match &plan.mlp {
                MlpPlan::Dense(_) if index == 0 => dense_rows(
                    engine,
                    &mlp_input,
                    row.dense
                        .as_ref()
                        .ok_or("MiMo batch layer 0 dense weights missing")?,
                    tokens,
                )?,
                MlpPlan::Moe(moe) if index > 0 => {
                    self.weights.routed[index]
                        .as_ref()
                        .ok_or("MiMo batch routed weights missing")?
                        .batch_bound(
                            engine,
                            &mlp_input,
                            tokens,
                            moe,
                            &self.weights.config,
                            &self.weights.plan,
                        )?
                        .output
                }
                _ => return Err(format!("MiMo batch layer {index} MLP changed").into()),
            };
            let mut after_mlp = engine.uninit(tokens * HIDDEN)?;
            engine.add(&after_attention, &mlp, &mut after_mlp, tokens * HIDDEN)?;
            hidden = after_mlp;
            #[cfg(test)]
            if let Some(position) = self.trace_position.filter(|&position| position < tokens) {
                let values =
                    engine.dtoh_view(&hidden.slice(position * HIDDEN..(position + 1) * HIDDEN))?;
                self.trace_rows.push((index, values));
            }
        }
        self.kv.finish_prefill_batch()?;
        if self.kv.position() != tokens {
            return Err("MiMo batch text KV cursor did not commit all rows".into());
        }
        let last = self.engines[1];
        last.gpu.ctx.bind_to_thread()?;
        let mut final_hidden = last.uninit(HIDDEN)?;
        last.dtod_copy_view(
            &hidden.slice((tokens - 1) * HIDDEN..tokens * HIDDEN),
            &mut final_hidden,
        )?;
        let final_norm = normalized(
            last,
            &final_hidden,
            &self.weights.output_norm,
            self.weights.plan.output_norm.epsilon,
        )?;
        let GpuTensor::FloatBf16 { data, ne } = &self.weights.output_head else {
            return Err("MiMo batch head lost the source BF16 output projection".into());
        };
        if ne.as_slice() != [HIDDEN as u64, VOCAB as u64]
            || data.len() != HIDDEN * VOCAB * 2
            || data.ordinal() != last.stream().context().ordinal()
        {
            return Err("MiMo batch head source shape or GPU changed".into());
        }
        let mut logits_gpu = last.uninit(VOCAB)?;
        last.matvec_bf16_rows_into(data, &final_norm, &mut logits_gpu, HIDDEN, VOCAB, 1)?;
        let (logits, final_hidden) = read_text_output::<true>(last, &final_hidden, &logits_gpu)?;
        self.position = tokens;
        self.failed = false;
        self.has_modal_payload = chunk.requires_payload_identity();
        Ok(MiMoTextStep {
            position: tokens - 1,
            logits,
            hidden_before_norm: final_hidden.ok_or("MiMo batch text hidden capture was omitted")?,
        })
    }

    fn chunk_row<const CAPTURE_HIDDEN: bool>(
        &mut self,
        chunk: &MiMoGpuEmbeddingChunk,
        index: usize,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>), Fail> {
        check_step(
            self.position,
            self.kv.position(),
            self.max_tokens,
            self.failed,
        )?;
        self.failed = true;
        let initial = chunk.owned_row(self.engines[0], index)?;
        self.finish_gpu_row::<CAPTURE_HIDDEN>(initial)
    }

    fn token_step<const CAPTURE_HIDDEN: bool>(
        &mut self,
        token: u32,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>), Fail> {
        check_step(
            self.position,
            self.kv.position(),
            self.max_tokens,
            self.failed,
        )?;
        // Validate the source row before any GPU or KV mutation.
        let initial = self.weights.embedding_row(token)?;
        self.failed = true;
        self.engines[0].gpu.ctx.bind_to_thread()?;
        let initial = self.engines[0].htod(&initial)?;
        self.finish_gpu_row::<CAPTURE_HIDDEN>(initial)
    }

    fn finish_gpu_row<const CAPTURE_HIDDEN: bool>(
        &mut self,
        initial: CudaSlice<f32>,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>), Fail> {
        let output = self.token_inner_gpu::<CAPTURE_HIDDEN>(initial)?;
        if self.kv.position() != self.position + 1 {
            return Err("MiMo compressed text did not commit all 48 KV rows".into());
        }
        if CAPTURE_HIDDEN && output.1.is_none() {
            return Err("MiMo compressed text hidden capture was omitted".into());
        }
        self.position += 1;
        self.failed = false;
        Ok(output)
    }

    fn token_inner_gpu<const CAPTURE_HIDDEN: bool>(
        &mut self,
        initial: CudaSlice<f32>,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>), Fail> {
        self.engines[0].gpu.ctx.bind_to_thread()?;
        let mut hidden = initial;
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
            #[cfg(test)]
            if self.trace_position == Some(self.position) {
                self.trace_rows.push((index, engine.dtoh(&hidden)?));
            }
        }
        let last = self.engines[1];
        let final_norm = normalized(
            last,
            &hidden,
            &self.weights.output_norm,
            self.weights.plan.output_norm.epsilon,
        )?;
        let GpuTensor::FloatBf16 { data, ne } = &self.weights.output_head else {
            return Err("MiMo head-only BF16 candidate lost the source output head".into());
        };
        if ne.as_slice() != [HIDDEN as u64, VOCAB as u64]
            || data.len() != HIDDEN * VOCAB * 2
            || data.ordinal() != last.stream().context().ordinal()
        {
            return Err("MiMo head-only BF16 candidate head shape or stage changed".into());
        }
        let mut logits_gpu = last.uninit(VOCAB)?;
        last.matvec_bf16_rows_into(data, &final_norm, &mut logits_gpu, HIDDEN, VOCAB, 1)?;
        read_text_output::<CAPTURE_HIDDEN>(last, &hidden, &logits_gpu)
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
    fn modal_chunk_refuses_nonfresh_or_cross_stage_kv_admission() {
        assert!(check_chunk_admission(0, 0, 4, false, 4, 4 * HIDDEN, [0, 0]).is_ok());
        assert!(check_chunk_admission(1, 1, 4, false, 3, 3 * HIDDEN, [0, 0]).is_err());
        assert!(check_chunk_admission(0, 1, 4, false, 4, 4 * HIDDEN, [0, 0]).is_err());
        assert!(check_chunk_admission(0, 0, 4, true, 4, 4 * HIDDEN, [0, 0]).is_err());
        assert!(check_chunk_admission(0, 0, 4, false, 5, 5 * HIDDEN, [0, 0]).is_err());
        assert!(check_chunk_admission(0, 0, 4, false, 0, 0, [0, 0]).is_err());
        assert!(check_chunk_admission(0, 0, 4, false, 4, 4 * HIDDEN - 1, [0, 0]).is_err());
        assert!(check_chunk_admission(0, 0, 4, false, 4, 4 * HIDDEN, [1, 0]).is_err());
    }

    #[test]
    fn first_batch_chunk_has_explicit_qkv_and_fresh_sequence_bound() {
        assert_eq!(first_batch_rows(1).unwrap(), 1);
        assert_eq!(first_batch_rows(20).unwrap(), 20);
        assert_eq!(first_batch_rows(128).unwrap(), 128);
        assert!(first_batch_rows(0).is_err());
        assert!(first_batch_rows(129).is_err());
        assert!(check_chunk_admission(0, 0, 128, false, 128, 128 * HIDDEN, [0, 0]).is_ok());
        assert!(check_chunk_admission(1, 1, 128, false, 20, 20 * HIDDEN, [0, 0]).is_err());
    }

    #[test]
    #[ignore = "requires pinned MiMo source and a dedicated two-card GPU lane"]
    fn source_modal_chunk_runs_text_kv_and_preserves_text_token_bits() -> Result<(), Fail> {
        use std::path::Path;
        use std::sync::Arc;

        use memra_gguf::source::SafetensorsSource;
        use memra_reference::mimo_modal_overlay::{AUDIO_TOKEN_ID, IMAGE_TOKEN_ID, VIDEO_TOKEN_ID};

        const FOUR_GIB: usize = 4 * 1024 * 1024 * 1024;
        let root = std::env::var("MIMO_PINNED_SOURCE_ROOT")?;
        let source = Arc::new(SafetensorsSource::open(Path::new(&root))?);
        let cards = [Engine::new(0)?, Engine::new(1)?];
        let engines = [&cards[0], &cards[1]];
        let text = MiMoTextWeights::load(engines, source)?;

        let baseline_logits = {
            let mut sequence = text.compressed_text_forward(engines, 1, [FOUR_GIB; 2])?;
            sequence.token(42)?
        };
        let prepared_text = text.modal_embedding_gpu_chunk(&cards[0], &[42], &[], &[], &[])?;
        let text_step = {
            let mut sequence = text.compressed_text_forward(engines, 1, [FOUR_GIB; 2])?;
            let step = sequence.consume_embedding_chunk(&prepared_text)?;
            assert_eq!(sequence.position(), 1);
            assert!(!sequence.has_modal_payload());
            step
        };
        assert_eq!(text_step.position, 0);
        assert_eq!(text_step.logits.len(), 152_576);
        assert!(
            text_step
                .logits
                .iter()
                .zip(baseline_logits)
                .all(|(got, want)| got.to_bits() == want.to_bits())
        );

        let tokens = [42, IMAGE_TOKEN_ID, VIDEO_TOKEN_ID, AUDIO_TOKEN_ID];
        let image = vec![vec![0.125; HIDDEN]];
        let video = vec![vec![-0.25; HIDDEN]];
        let audio = vec![vec![0.5; HIDDEN]];
        let prepared =
            text.modal_embedding_gpu_chunk(&cards[0], &tokens, &image, &video, &audio)?;
        let mut sequence = text.compressed_text_forward(engines, 5, [FOUR_GIB; 2])?;
        let step = sequence.consume_embedding_chunk(&prepared)?;
        assert_eq!(step.position, 3);
        assert_eq!(sequence.position(), 4);
        assert!(sequence.has_modal_payload());
        assert_eq!(step.logits.len(), 152_576);
        assert_eq!(step.hidden_before_norm.len(), HIDDEN);
        assert!(step.logits.iter().all(|value| value.is_finite()));
        assert!(
            step.hidden_before_norm
                .iter()
                .all(|value| value.is_finite())
        );
        let continuation = sequence.token(220)?;
        assert_eq!(sequence.position(), 5);
        assert_eq!(continuation.len(), 152_576);
        assert!(continuation.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned MiMo source and a dedicated two-card GPU lane"]
    fn first_batch_chunk_replays_and_hands_off_to_ordinary_decode() -> Result<(), Fail> {
        use std::path::Path;
        use std::sync::Arc;

        use memra_gguf::source::SafetensorsSource;
        use memra_reference::mimo_modal_overlay::{AUDIO_TOKEN_ID, IMAGE_TOKEN_ID, VIDEO_TOKEN_ID};

        const FOUR_GIB: usize = 4 * 1024 * 1024 * 1024;
        let root = std::env::var("MIMO_PINNED_SOURCE_ROOT")?;
        let source = Arc::new(SafetensorsSource::open(Path::new(&root))?);
        let cards = [Engine::new(0)?, Engine::new(1)?];
        let engines = [&cards[0], &cards[1]];
        let text = MiMoTextWeights::load(engines, source)?;
        let tokens = [42, IMAGE_TOKEN_ID, VIDEO_TOKEN_ID, AUDIO_TOKEN_ID];
        let image = vec![vec![0.125; HIDDEN]];
        let video = vec![vec![-0.25; HIDDEN]];
        let audio = vec![vec![0.5; HIDDEN]];
        let prepared =
            text.modal_embedding_gpu_chunk(&cards[0], &tokens, &image, &video, &audio)?;

        let run_batch = || -> Result<(MiMoTextStep, Vec<f32>), Fail> {
            let mut sequence = text.compressed_text_forward(engines, 5, [FOUR_GIB; 2])?;
            let step = sequence.consume_embedding_chunk_batched(&prepared)?;
            if step.position != 3
                || sequence.position() != 4
                || !sequence.has_modal_payload()
                || step.hidden_before_norm.len() != HIDDEN
                || step.logits.len() != VOCAB
            {
                return Err("MiMo first batch chunk returned an incomplete source step".into());
            }
            let continuing = sequence.token(220)?;
            if sequence.position() != 5 || continuing.len() != VOCAB {
                return Err("MiMo first batch chunk did not hand off to decode".into());
            }
            Ok((step, continuing))
        };
        let (first, first_next) = run_batch()?;
        let (second, second_next) = run_batch()?;
        if first
            .logits
            .iter()
            .zip(&second.logits)
            .any(|(a, b)| a.to_bits() != b.to_bits())
            || first
                .hidden_before_norm
                .iter()
                .zip(&second.hidden_before_norm)
                .any(|(a, b)| a.to_bits() != b.to_bits())
            || first_next
                .iter()
                .zip(&second_next)
                .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err("MiMo first batch chunk changed on byte-identical GPU replay".into());
        }
        let mut serial = text.compressed_text_forward(engines, 5, [FOUR_GIB; 2])?;
        let serial_step = serial.consume_embedding_chunk(&prepared)?;
        let serial_next = serial.token(220)?;
        let relative_l2 = |candidate: &[f32], control: &[f32]| {
            let (diff, base) = candidate.iter().zip(control).fold(
                (0.0f64, 0.0f64),
                |(diff, base), (&got, &want)| {
                    (
                        diff + f64::from(got - want).powi(2),
                        base + f64::from(want).powi(2),
                    )
                },
            );
            (diff / base).sqrt()
        };
        let argmax = |logits: &[f32]| {
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .unwrap()
        };
        println!(
            "mimo_first_chunk_batch\tlast_rel_l2={:.9e}\tdecode_rel_l2={:.9e}\tlast_argmax={}\tserial_argmax={}",
            relative_l2(&first.logits, &serial_step.logits),
            relative_l2(&first_next, &serial_next),
            argmax(&first.logits),
            argmax(&serial_step.logits),
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned MiMo source and a dedicated two-card GPU lane"]
    fn first_batch_chunk_layerwise_diagnostic() -> Result<(), Fail> {
        use std::path::Path;
        use std::sync::Arc;

        use memra_gguf::source::SafetensorsSource;
        use memra_reference::mimo_modal_overlay::{AUDIO_TOKEN_ID, IMAGE_TOKEN_ID, VIDEO_TOKEN_ID};

        const FOUR_GIB: usize = 4 * 1024 * 1024 * 1024;
        let root = std::env::var("MIMO_PINNED_SOURCE_ROOT")?;
        let source = Arc::new(SafetensorsSource::open(Path::new(&root))?);
        let cards = [Engine::new(0)?, Engine::new(1)?];
        let engines = [&cards[0], &cards[1]];
        let text = MiMoTextWeights::load(engines, source)?;
        let tokens = [42, IMAGE_TOKEN_ID, VIDEO_TOKEN_ID, AUDIO_TOKEN_ID];
        let image = vec![vec![0.125; HIDDEN]];
        let video = vec![vec![-0.25; HIDDEN]];
        let audio = vec![vec![0.5; HIDDEN]];
        let prepared =
            text.modal_embedding_gpu_chunk(&cards[0], &tokens, &image, &video, &audio)?;

        let (batched, batched_rows) = {
            let mut sequence = text.compressed_text_forward(engines, 4, [FOUR_GIB; 2])?;
            sequence.trace_position = Some(3);
            let step = sequence.consume_embedding_chunk_batched(&prepared)?;
            (step, sequence.trace_rows)
        };
        let (serial, serial_rows) = {
            let mut sequence = text.compressed_text_forward(engines, 4, [FOUR_GIB; 2])?;
            sequence.trace_position = Some(3);
            let step = sequence.consume_embedding_chunk(&prepared)?;
            (step, sequence.trace_rows)
        };
        if batched_rows.len() != LAYERS || serial_rows.len() != LAYERS {
            return Err("MiMo layerwise diagnostic did not capture all layers".into());
        }
        for ((batch_layer, batch), (serial_layer, control)) in batched_rows.iter().zip(&serial_rows)
        {
            if batch_layer != serial_layer || batch.len() != HIDDEN || control.len() != HIDDEN {
                return Err("MiMo layerwise diagnostic rows are misaligned".into());
            }
            let mut diff_squared = 0.0f64;
            let mut base_squared = 0.0f64;
            let mut max_abs = 0.0f32;
            let mut matching_bits = 0usize;
            for (&got, &want) in batch.iter().zip(control) {
                let delta = (got - want).abs();
                max_abs = max_abs.max(delta);
                diff_squared += f64::from(delta).powi(2);
                base_squared += f64::from(want).powi(2);
                matching_bits += usize::from(got.to_bits() == want.to_bits());
            }
            println!(
                "mimo_batch_layer\tlayer={batch_layer}\trel_l2={:.9e}\tmax_abs={max_abs:.9e}\tmatching_bits={matching_bits}",
                (diff_squared / base_squared).sqrt(),
            );
        }
        let argmax = |logits: &[f32]| {
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .unwrap()
        };
        println!(
            "mimo_batch_layer\tfinal_batch_argmax={}\tfinal_serial_argmax={}",
            argmax(&batched.logits),
            argmax(&serial.logits)
        );
        Ok(())
    }

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
