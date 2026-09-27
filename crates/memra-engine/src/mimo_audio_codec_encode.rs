//! Bounded model-owned MiMo audio tokenizer encode from prepared log mel.

use std::error::Error;

use cudarc::driver::CudaSlice;
use memra_reference::mimo_audio::GroupedAudioCodes;
use memra_reference::mimo_audio_codec_layer::MAX_COMPONENT_TOKENS;

use crate::Engine;
use crate::mimo_audio_codec_weights::MiMoAudioCodecEncoderWeights;

type Fail = Box<dyn Error>;

const MEL_CHANNELS: usize = 128;
const HIDDEN: usize = 1_024;
const MAX_MEL_FRAMES: usize = MAX_COMPONENT_TOKENS * 2;

/// Complete codec IDs for one prepared mel plane. `code_ids` is token-major
/// `[tokens,20]`; `grouped_patch_codes` is `[ceil(tokens/4),4,20]` with the
/// final row repeated as needed.
pub struct MiMoPreparedMelAudioCodes {
    pub code_ids: Vec<u16>,
    pub tokens: usize,
    pub grouped_patch_codes: GroupedAudioCodes,
}

impl MiMoAudioCodecEncoderWeights {
    /// Encode one finite, contiguous F32 `[mel_frames,128]` prepared mel plane.
    ///
    /// Source order is conv1/GELU, conv2/GELU, the 24-layer transformer stack
    /// and final norm, post-stack stride-two downsample and norm, then the
    /// twenty F32 RVQ depths. This path accepts 1..=512 mel frames so the
    /// transformer never exceeds 256 rows. It does not prepare mel from PCM.
    pub fn encode_prepared_mel_codes(
        &self,
        engine: &Engine,
        mel: &CudaSlice<f32>,
        mel_frames: usize,
    ) -> Result<MiMoPreparedMelAudioCodes, Fail> {
        if !(1..=MAX_MEL_FRAMES).contains(&mel_frames)
            || mel_frames.checked_mul(MEL_CHANNELS) != Some(mel.len())
        {
            return Err("MiMo codec prepared mel frame count or F32 extent changed".into());
        }
        engine.gpu.ctx.bind_to_thread()?;
        self.check_device(engine)?;
        if mel.ordinal() != self.device_ordinal {
            return Err("MiMo codec prepared mel belongs to another GPU".into());
        }

        let transformer_tokens = mel_frames.div_ceil(2);
        let frontend = self.encode_prepared_mel_conv(engine, mel, mel_frames)?;
        if frontend.len() != transformer_tokens * HIDDEN {
            return Err("MiMo codec frontend returned wrong token extent".into());
        }
        let encoded = self.encode_transformer_stack(engine, &frontend, transformer_tokens)?;
        let post_stack = self.downsample_post_stack(engine, &encoded, transformer_tokens)?;
        let tokens = transformer_tokens.div_ceil(2);
        if post_stack.len() != tokens * HIDDEN {
            return Err("MiMo codec downsample returned wrong token extent".into());
        }
        let rvq = self.encode_20_rvq(engine, &post_stack, tokens)?;
        let grouped_patch_codes = rvq.grouped_patch_codes()?;
        Ok(MiMoPreparedMelAudioCodes {
            code_ids: rvq.code_ids,
            tokens: rvq.tokens,
            grouped_patch_codes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::model_packs::mimo_v2::audio_tokenizer::{LFS_WEIGHT_SHA256, SOURCE};
    use sha2::{Digest, Sha256};
    use std::path::Path;

    // Frozen from audio-publisher-codes.json (SHA256
    // b619f47caf72eed27f904281efbbf9da51186b90eccc0c9e0bc0d7d410b7d8c7),
    // produced by the independent pinned V2.6 CPU publisher oracle.
    const PUBLISHER_TOKEN_MAJOR: [u16; 60] = [
        892, 542, 112, 40, 86, 47, 38, 118, 18, 53, 99, 124, 43, 24, 87, 75, 69, 26, 58, 91, 54,
        61, 119, 28, 59, 86, 90, 38, 102, 35, 72, 17, 29, 37, 55, 24, 123, 127, 17, 18, 992, 285,
        49, 126, 107, 96, 114, 107, 88, 83, 89, 5, 48, 24, 37, 56, 16, 64, 29, 44,
    ];

    #[test]
    #[ignore = "requires pinned bundled weights and a dedicated target GPU"]
    fn prepared_mel_codes_match_pinned_cpu_publisher() -> Result<(), Fail> {
        assert_eq!(
            SOURCE,
            "XiaomiMiMo/MiMo-V2.6-Flash-RL@3b38d063180c3e4aed9691fdc735f3d10b266ee4"
        );
        assert_eq!(
            LFS_WEIGHT_SHA256,
            "077033345d80eef3a315e8d394e0589667e80e4cdaba9bc5a7488410c6657265"
        );
        let mel_bytes = include_bytes!(
            "../../memra-reference/src/fixtures/mimo-pcm-mel-voiced-2048.logmel.f32"
        );
        assert_eq!(mel_bytes.len(), 9 * MEL_CHANNELS * 4);
        assert_eq!(
            format!("{:x}", Sha256::digest(mel_bytes)),
            "0def2591fe32a564c2bda9a8444c7b34896d1129fa05c99d71dddd518e842aeb"
        );
        let mel = mel_bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().expect("four-byte F32")))
            .collect::<Vec<_>>();

        let root = std::env::var("MIMO_PINNED_SOURCE_ROOT")?;
        let gpu = std::env::var("MEMRA_MIMO_COMPONENT_GPU")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        let engine = Engine::new(gpu)?;
        // The resident loader hashes the full pinned payload before upload.
        let weights = MiMoAudioCodecEncoderWeights::load(&engine, Path::new(&root))?;
        let got = weights.encode_prepared_mel_codes(&engine, &engine.htod(&mel)?, 9)?;

        assert_eq!(got.tokens, 3);
        assert_eq!(&got.code_ids, &PUBLISHER_TOKEN_MAJOR);
        assert_eq!(got.grouped_patch_codes.groups, 1);
        assert_eq!(got.grouped_patch_codes.codes.len(), 4 * 20);
        assert_eq!(&got.grouped_patch_codes.codes[..60], &PUBLISHER_TOKEN_MAJOR);
        assert_eq!(
            &got.grouped_patch_codes.codes[60..],
            &PUBLISHER_TOKEN_MAJOR[40..]
        );
        Ok(())
    }
}
