//! Source-backed MiMo embeddings for one bounded fresh multimodal chunk.
//! The payload identity must travel with any future KV cache key.

use std::error::Error;

use cudarc::driver::CudaSlice;
use memra_gguf::config::Arch;
use memra_reference::mimo_modal_overlay::{
    HIDDEN, MiMoHostEmbeddingChunk, MiMoModalCounts, overlay_pinned_chunk,
    validate_pinned_modal_request,
};

use crate::Engine;
use crate::mimo_text_weights::MiMoTextWeights;

type Fail = Box<dyn Error>;

pub struct MiMoGpuEmbeddingChunk {
    embeddings: CudaSlice<f32>,
    tokens: usize,
    counts: MiMoModalCounts,
}

impl MiMoGpuEmbeddingChunk {
    pub fn embeddings(&self) -> &CudaSlice<f32> {
        &self.embeddings
    }

    pub fn token_count(&self) -> usize {
        self.tokens
    }

    pub fn modal_counts(&self) -> MiMoModalCounts {
        self.counts
    }

    /// A chunk with modal rows cannot use a KV cache keyed only by token IDs.
    /// This component creates no KV entry or cache lookup.
    pub fn requires_payload_identity(&self) -> bool {
        self.counts.has_payload()
    }

    pub(crate) fn owned_row(&self, stage0: &Engine, index: usize) -> Result<CudaSlice<f32>, Fail> {
        let device = stage0.stream().context().ordinal();
        if index >= self.tokens
            || self.embeddings.len() != self.tokens * HIDDEN
            || self.embeddings.ordinal() != device
        {
            return Err("MiMo modal embedding row index, extent, or stage changed".into());
        }
        stage0.gpu.ctx.bind_to_thread()?;
        let mut row = stage0.uninit(HIDDEN)?;
        stage0.dtod_copy_view(
            &self.embeddings.slice(index * HIDDEN..(index + 1) * HIDDEN),
            &mut row,
        )?;
        Ok(row)
    }
}

fn validate_stage_layout(
    requested: usize,
    stage0: usize,
    stage1: usize,
) -> Result<(), &'static str> {
    if requested != stage0 || stage0 == stage1 {
        return Err("MiMo modal embedding upload requires the distinct stage-0 GPU");
    }
    Ok(())
}

impl MiMoTextWeights {
    /// Read the pinned text source row for every token, then replace each modal
    /// placeholder with the corresponding supplied encoder row in source order.
    /// The result remains a fresh chunk and carries modal-presence metadata.
    pub fn modal_embedding_host_chunk(
        &self,
        token_ids: &[u32],
        image_rows: &[Vec<f32>],
        video_rows: &[Vec<f32>],
        audio_rows: &[Vec<f32>],
    ) -> Result<MiMoHostEmbeddingChunk, Fail> {
        self.verify_source_config()?;
        if self.plan.arch != Arch::MiMoV2
            || self.plan.hidden_size as usize != HIDDEN
            || self.plan.vocab_size != self.config.n_vocab
            || self.plan.layers.len() != 48
        {
            return Err("MiMo modal embedding plan changed after binding".into());
        }
        validate_pinned_modal_request(&self.config, token_ids, image_rows, video_rows, audio_rows)?;
        let mut text = Vec::with_capacity(token_ids.len() * HIDDEN);
        for &token in token_ids {
            text.extend(self.embedding_row(token)?);
        }
        Ok(overlay_pinned_chunk(
            &self.config,
            token_ids,
            &text,
            image_rows,
            video_rows,
            audio_rows,
        )?)
    }

    /// Upload a complete `[tokens,4096]` f32 chunk of BF16-valued embeddings
    /// to the text trunk's first GPU. This upload is synchronized before return.
    pub fn modal_embedding_gpu_chunk(
        &self,
        stage0: &Engine,
        token_ids: &[u32],
        image_rows: &[Vec<f32>],
        video_rows: &[Vec<f32>],
        audio_rows: &[Vec<f32>],
    ) -> Result<MiMoGpuEmbeddingChunk, Fail> {
        if self.layers.len() != 48 {
            return Err("MiMo modal embedding text stages are incomplete".into());
        }
        let requested = stage0.stream().context().ordinal();
        let first_stage = self.layers[0].attention_norm.ordinal();
        let second_stage = self.layers[24].attention_norm.ordinal();
        validate_stage_layout(requested, first_stage, second_stage)?;
        if self.layers[..24]
            .iter()
            .any(|layer| layer.attention_norm.ordinal() != first_stage)
            || self.layers[24..]
                .iter()
                .any(|layer| layer.attention_norm.ordinal() != second_stage)
        {
            return Err("MiMo modal embedding weights crossed text stages".into());
        }
        let host =
            self.modal_embedding_host_chunk(token_ids, image_rows, video_rows, audio_rows)?;
        let (rows, tokens, counts) = host.into_parts();
        stage0.gpu.ctx.bind_to_thread()?;
        let embeddings = stage0.htod(&rows)?;
        stage0.stream().synchronize()?;
        if embeddings.ordinal() != first_stage || embeddings.len() != tokens * HIDDEN {
            return Err("MiMo modal embedding upload landed on the wrong stage".into());
        }
        Ok(MiMoGpuEmbeddingChunk {
            embeddings,
            tokens,
            counts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modal_upload_refuses_wrong_or_aliased_stage() {
        assert!(validate_stage_layout(0, 0, 1).is_ok());
        assert!(validate_stage_layout(1, 0, 1).is_err());
        assert!(validate_stage_layout(0, 0, 0).is_err());
        assert!(validate_stage_layout(2, 1, 2).is_err());
    }

    #[test]
    #[ignore = "requires the pinned MiMo source and a dedicated two-card GPU lane"]
    fn pinned_modal_upload_matches_bound_source_rows() -> Result<(), Fail> {
        use std::path::Path;
        use std::sync::Arc;

        use memra_gguf::source::SafetensorsSource;
        use memra_reference::mimo_modal_overlay::{AUDIO_TOKEN_ID, IMAGE_TOKEN_ID, VIDEO_TOKEN_ID};

        let root = std::env::var("MIMO_PINNED_SOURCE_ROOT")?;
        let source = Arc::new(SafetensorsSource::open(Path::new(&root))?);
        let cards = [Engine::new(0)?, Engine::new(1)?];
        let text = MiMoTextWeights::load([&cards[0], &cards[1]], source)?;
        let tokens = [42, IMAGE_TOKEN_ID, AUDIO_TOKEN_ID, VIDEO_TOKEN_ID, 43];
        let image = vec![vec![1.007_812_5; HIDDEN]];
        let audio = vec![vec![-0.531_25; HIDDEN]];
        let video = vec![vec![0.328_125; HIDDEN]];
        let expected = text.modal_embedding_host_chunk(&tokens, &image, &video, &audio)?;
        let actual = text.modal_embedding_gpu_chunk(&cards[0], &tokens, &image, &video, &audio)?;
        assert_eq!(actual.token_count(), tokens.len());
        assert!(actual.requires_payload_identity());
        let uploaded = cards[0].dtoh(actual.embeddings())?;
        assert_eq!(uploaded.len(), expected.rows().len());
        assert!(
            uploaded
                .iter()
                .zip(expected.rows())
                .all(|(got, want)| got.to_bits() == want.to_bits())
        );
        Ok(())
    }
}
