//! Pinned MiMo placeholder replacement for one fresh text embedding chunk.
//! This is a CPU oracle for supplied encoder rows, not a decoder or KV cache.

use memra_gguf::config::{Arch, ModelConfig};

pub const IMAGE_TOKEN_ID: u32 = 151_655;
pub const VIDEO_TOKEN_ID: u32 = 151_656;
pub const AUDIO_TOKEN_ID: u32 = 151_669;
pub const HIDDEN: usize = 4096;
pub const VOCAB: u32 = 152_576;
pub const MAX_CHUNK: usize = 256;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MiMoModalCounts {
    pub image: usize,
    pub video: usize,
    pub audio: usize,
}

impl MiMoModalCounts {
    pub fn has_payload(self) -> bool {
        self.image + self.video + self.audio != 0
    }
}

/// Flat row-major BF16-valued f32 embeddings. The counts travel with the
/// chunk so a future prefill path cannot treat modal slots as token-only KV.
pub struct MiMoHostEmbeddingChunk {
    rows: Vec<f32>,
    tokens: usize,
    counts: MiMoModalCounts,
}

impl MiMoHostEmbeddingChunk {
    pub fn rows(&self) -> &[f32] {
        &self.rows
    }

    pub fn token_count(&self) -> usize {
        self.tokens
    }

    pub fn modal_counts(&self) -> MiMoModalCounts {
        self.counts
    }

    pub fn has_modal_payload(&self) -> bool {
        self.counts.has_payload()
    }

    pub fn into_parts(self) -> (Vec<f32>, usize, MiMoModalCounts) {
        (self.rows, self.tokens, self.counts)
    }
}

pub fn validate_pinned_modal_config(config: &ModelConfig) -> Result<(), String> {
    let mimo = config
        .mimo
        .as_ref()
        .ok_or("MiMo modal overlay requires its pinned model config")?;
    if config.arch != Arch::MiMoV2
        || config.n_embd as usize != HIDDEN
        || config.n_vocab != VOCAB
        || config.n_layer != 48
        || mimo.image_token_id != Some(IMAGE_TOKEN_ID)
        || mimo.video_token_id != Some(VIDEO_TOKEN_ID)
        || mimo.audio_token_id != Some(AUDIO_TOKEN_ID)
        || mimo.vision_config.as_ref().map(|v| v.out_hidden_size) != Some(HIDDEN as u32)
        || mimo.audio_config.as_ref().map(|a| a.out_hidden_size) != Some(HIDDEN as u32)
    {
        return Err("MiMo modal overlay config differs from pinned source".into());
    }
    Ok(())
}

fn validate_modal_rows(label: &str, rows: &[Vec<f32>], expected: usize) -> Result<(), String> {
    if rows.len() != expected {
        return Err(format!(
            "MiMo {label} placeholder count {expected} differs from supplied rows {}",
            rows.len()
        ));
    }
    for (index, row) in rows.iter().enumerate() {
        if row.len() != HIDDEN {
            return Err(format!(
                "MiMo {label} row {index} width {} differs from {HIDDEN}",
                row.len()
            ));
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(format!("MiMo {label} row {index} has a non-finite value"));
        }
    }
    Ok(())
}

/// Check every input before the model-owned text source is read or GPU memory
/// is allocated. Slots are counted independently for image, video, and audio.
pub fn validate_pinned_modal_request(
    config: &ModelConfig,
    token_ids: &[u32],
    image_rows: &[Vec<f32>],
    video_rows: &[Vec<f32>],
    audio_rows: &[Vec<f32>],
) -> Result<MiMoModalCounts, String> {
    validate_pinned_modal_config(config)?;
    if !(1..=MAX_CHUNK).contains(&token_ids.len()) {
        return Err("MiMo modal overlay requires 1..=256 tokens".into());
    }
    if let Some(&token) = token_ids.iter().find(|&&token| token >= VOCAB) {
        return Err(format!(
            "MiMo modal overlay token {token} exceeds vocabulary"
        ));
    }
    let counts = MiMoModalCounts {
        image: token_ids.iter().filter(|&&id| id == IMAGE_TOKEN_ID).count(),
        video: token_ids.iter().filter(|&&id| id == VIDEO_TOKEN_ID).count(),
        audio: token_ids.iter().filter(|&&id| id == AUDIO_TOKEN_ID).count(),
    };
    validate_modal_rows("image", image_rows, counts.image)?;
    validate_modal_rows("video", video_rows, counts.video)?;
    validate_modal_rows("audio", audio_rows, counts.audio)?;
    Ok(counts)
}

fn round_bf16(value: f32) -> Result<f32, String> {
    let bits = value.to_bits();
    let tie_to_even = 0x7fff + ((bits >> 16) & 1);
    let rounded = f32::from_bits((bits.wrapping_add(tie_to_even)) & 0xffff_0000);
    if !rounded.is_finite() {
        return Err("MiMo modal row overflows the BF16 text embedding dtype".into());
    }
    Ok(rounded)
}

/// Publisher order is per modality and then placeholder order within the
/// token sequence. Distinct modalities may interleave in a message chunk.
pub fn overlay_pinned_chunk(
    config: &ModelConfig,
    token_ids: &[u32],
    text_embeddings: &[f32],
    image_rows: &[Vec<f32>],
    video_rows: &[Vec<f32>],
    audio_rows: &[Vec<f32>],
) -> Result<MiMoHostEmbeddingChunk, String> {
    let counts =
        validate_pinned_modal_request(config, token_ids, image_rows, video_rows, audio_rows)?;
    if text_embeddings.len() != token_ids.len() * HIDDEN {
        return Err("MiMo text embedding chunk width or count changed".into());
    }
    if text_embeddings
        .iter()
        .any(|value| !value.is_finite() || value.to_bits() & 0xffff != 0)
    {
        return Err("MiMo source text embedding is not a finite BF16 value".into());
    }
    let mut rows = text_embeddings.to_vec();
    let mut image = 0;
    let mut video = 0;
    let mut audio = 0;
    for (position, &token) in token_ids.iter().enumerate() {
        let supplied = match token {
            IMAGE_TOKEN_ID => {
                let row = &image_rows[image];
                image += 1;
                Some(row)
            }
            VIDEO_TOKEN_ID => {
                let row = &video_rows[video];
                video += 1;
                Some(row)
            }
            AUDIO_TOKEN_ID => {
                let row = &audio_rows[audio];
                audio += 1;
                Some(row)
            }
            _ => None,
        };
        if let Some(row) = supplied {
            for (dst, &value) in rows[position * HIDDEN..(position + 1) * HIDDEN]
                .iter_mut()
                .zip(row)
            {
                *dst = round_bf16(value)?;
            }
        }
    }
    Ok(MiMoHostEmbeddingChunk {
        rows,
        tokens: token_ids.len(),
        counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::HfConfig;

    fn config() -> ModelConfig {
        ModelConfig::from_hf(&HfConfig::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ))))
    }

    fn row(value: f32) -> Vec<f32> {
        vec![value; HIDDEN]
    }

    #[test]
    fn source_order_replaces_interleaved_modal_slots_and_rounds_to_bf16() {
        let tokens = [
            11,
            IMAGE_TOKEN_ID,
            AUDIO_TOKEN_ID,
            VIDEO_TOKEN_ID,
            IMAGE_TOKEN_ID,
            12,
        ];
        let mut text = vec![0.0; tokens.len() * HIDDEN];
        text[..HIDDEN].fill(11.0);
        text[5 * HIDDEN..].fill(12.0);
        let image = vec![row(f32::from_bits(0x3f80_8000)), row(3.25)];
        let video = vec![row(f32::from_bits(0x3f81_8000))];
        let audio = vec![row(5.75)];
        let chunk =
            overlay_pinned_chunk(&config(), &tokens, &text, &image, &video, &audio).unwrap();
        assert_eq!(chunk.token_count(), 6);
        assert_eq!(
            chunk.modal_counts(),
            MiMoModalCounts {
                image: 2,
                video: 1,
                audio: 1
            }
        );
        assert!(chunk.has_modal_payload());
        let first = chunk
            .rows()
            .chunks_exact(HIDDEN)
            .map(|row| row[0])
            .collect::<Vec<_>>();
        assert_eq!(first, [11.0, 1.0, 5.75, 1.015_625, 3.25, 12.0]);
        assert!(
            chunk
                .rows()
                .iter()
                .all(|value| value.to_bits() & 0xffff == 0)
        );
    }

    #[test]
    fn text_only_chunk_keeps_source_rows_and_has_no_modal_payload() {
        let text = [row(0.25), row(-2.0)].concat();
        let chunk = overlay_pinned_chunk(&config(), &[11, 12], &text, &[], &[], &[]).unwrap();
        assert_eq!(chunk.rows(), text);
        assert_eq!(chunk.modal_counts(), MiMoModalCounts::default());
        assert!(!chunk.has_modal_payload());
    }

    #[test]
    fn every_placeholder_and_supplied_row_must_match() {
        let tokens = [IMAGE_TOKEN_ID, AUDIO_TOKEN_ID, IMAGE_TOKEN_ID];
        let text = vec![0.0; tokens.len() * HIDDEN];
        let image = vec![row(1.0), row(2.0)];
        let audio = vec![row(3.0)];
        assert!(overlay_pinned_chunk(&config(), &tokens, &text, &image, &[], &audio).is_ok());
        assert!(overlay_pinned_chunk(&config(), &tokens, &text, &image[..1], &[], &audio).is_err());
        assert!(overlay_pinned_chunk(&config(), &tokens, &text, &image, &[], &[]).is_err());
        assert!(
            overlay_pinned_chunk(&config(), &tokens, &text, &image, &[row(4.0)], &audio).is_err()
        );
        assert!(overlay_pinned_chunk(&config(), &[11], &text[..HIDDEN], &image, &[], &[]).is_err());
    }

    #[test]
    fn rejects_geometry_nonfinite_token_and_config_drift() {
        let config = config();
        let token = [IMAGE_TOKEN_ID];
        let text = row(0.0);
        assert!(
            overlay_pinned_chunk(&config, &token, &text, &[vec![1.0; HIDDEN - 1]], &[], &[])
                .is_err()
        );
        assert!(overlay_pinned_chunk(&config, &token, &text, &[row(f32::NAN)], &[], &[]).is_err());
        assert!(overlay_pinned_chunk(&config, &token, &text, &[row(f32::MAX)], &[], &[]).is_err());
        assert!(overlay_pinned_chunk(&config, &[VOCAB], &text, &[], &[], &[]).is_err());
        assert!(overlay_pinned_chunk(&config, &[], &[], &[], &[], &[]).is_err());
        assert!(
            overlay_pinned_chunk(&config, &vec![11; MAX_CHUNK + 1], &[], &[], &[], &[]).is_err()
        );
        let mut bad_text = text.clone();
        bad_text[0] = 1.001;
        assert!(overlay_pinned_chunk(&config, &[11], &bad_text, &[], &[], &[]).is_err());
        assert!(overlay_pinned_chunk(&config, &[11], &text[..HIDDEN - 1], &[], &[], &[]).is_err());
        let mut changed = config.clone();
        changed.mimo.as_mut().unwrap().image_token_id = Some(11);
        assert!(overlay_pinned_chunk(&changed, &token, &text, &[row(1.0)], &[], &[]).is_err());
        changed = config.clone();
        changed.mimo.as_mut().unwrap().video_token_id = None;
        assert!(overlay_pinned_chunk(&changed, &token, &text, &[row(1.0)], &[], &[]).is_err());
        changed = config;
        changed.mimo.as_mut().unwrap().audio_token_id = Some(11);
        assert!(overlay_pinned_chunk(&changed, &token, &text, &[row(1.0)], &[], &[]).is_err());
    }
}
