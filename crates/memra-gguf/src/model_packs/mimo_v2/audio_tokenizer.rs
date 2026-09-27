//! Metadata-only contract for MiMo V2.6's separately bundled audio tokenizer.
//!
//! The pinned `audio_tokenizer/config.json` and safetensors header sidecar are
//! enough to describe the codec geometry and tensor layout. The LFS identity
//! and file length are source metadata, not a hash of payload bytes read here.
//! This module does not decode audio or enable audio or serving support.

use crate::safetensors::{StInfo, parse_header_json_checked};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

pub const SOURCE: &str = "XiaomiMiMo/MiMo-V2.6-Flash-RL@3b38d063180c3e4aed9691fdc735f3d10b266ee4";
pub const CONFIG_SHA256: &str = "e0702adae37947e0c980c38bae58ffa0d48bd492d523afe814aa4d73f008c7d1";
pub const HEADER_SHA256: &str = "45e94c498c6ae5214525ad1060da99dd2ef9be428c1eaf226987739984b33e53";
/// LFS pointer identity. This checker does not read or hash the weight payload.
pub const LFS_WEIGHT_SHA256: &str =
    "077033345d80eef3a315e8d394e0589667e80e4cdaba9bc5a7488410c6657265";

const TENSOR_ROWS: usize = 828;
const F32_ROWS: usize = 459;
const BF16_ROWS: usize = 369;
const TENSOR_BYTES: usize = 1_872_525_336;
const LFS_FILE_BYTES: usize = 1_872_618_384;
const SAFETENSORS_PREFIX_BYTES: usize = 8;
// The fixture is the exact raw safetensors header after the 8-byte prefix.
// Prefix + header + final tensor offset equals the pinned LFS file length.
const IN_FILE_HEADER_BYTES: usize = 93_040;
const CODEBOOK_SIZES: [usize; 20] = [
    1_024, 1_024, 256, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    128, 128,
];

/// Geometry and header receipt. No field represents executable codec support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioTokenizerAuxiliaryContract {
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub hidden_size: usize,
    pub attention_heads: usize,
    pub ffn_size: usize,
    pub quantizers: usize,
    pub codebook_sizes: [usize; 20],
    pub sampling_rate: usize,
    pub mel_bins: usize,
    pub fft_size: usize,
    pub hop_length: usize,
    pub window_size: usize,
    pub hybrid_attention: bool,
    pub hybrid_block_size: usize,
    pub swa_per_block: usize,
    pub tensor_rows: usize,
    pub f32_rows: usize,
    pub bf16_rows: usize,
    pub tensor_bytes: usize,
    pub file_bytes: usize,
    pub in_file_header_bytes: usize,
}

fn check_hash(bytes: &[u8], expected: &str, which: &str) -> Result<(), String> {
    let found = format!("{:x}", Sha256::digest(bytes));
    if found != expected {
        return Err(format!(
            "MiMo audio tokenizer {which} changed: got {found}, expected {expected}"
        ));
    }
    Ok(())
}

fn check_config_fields(config: &Value) -> Result<(), String> {
    if config.as_object().map(|fields| fields.len()) != Some(39) {
        return Err("MiMo audio tokenizer config field set differs from pin".into());
    }
    for (path, expected) in [
        ("/max_audio_seconds", json!(300)),
        ("/stride_size", json!(2)),
        ("/avg_pooler", json!(2)),
        ("/d_model", json!(1024)),
        ("/scale_embedding", json!(false)),
        ("/kernel_size", json!(3)),
        ("/activation_function", json!("gelu")),
        ("/encoder_layers", json!(24)),
        ("/encoder_skip_layer_id", json!(3)),
        ("/encoder_attention_heads", json!(16)),
        ("/encoder_ffn_dim", json!(4096)),
        ("/encoder_causal", json!(true)),
        ("/encoder_attn_window_size", json!([128, 0])),
        ("/decoder_layers", json!(24)),
        ("/decoder_attention_heads", json!(16)),
        ("/decoder_ffn_dim", json!(4096)),
        ("/decoder_kernel_size", json!(3)),
        ("/decoder_stride_size", json!(2)),
        ("/decoder_causal", json!(true)),
        ("/decoder_attn_window_size", json!([128, 0])),
        ("/nfft", json!(960)),
        ("/n_mels", json!(128)),
        ("/sampling_rate", json!(24000)),
        ("/hop_length", json!(240)),
        ("/window_size", json!(960)),
        ("/vocoder_padding", json!("same")),
        ("/fmin", json!(0)),
        ("/fmax", json!(null)),
        ("/num_quantizers", json!(20)),
        ("/codebook_size", json!(CODEBOOK_SIZES)),
        ("/threshold_ema_dead_code", json!(2)),
        ("/position_embedding_type", json!("rope")),
        ("/rope_theta", json!(10000)),
        ("/rope_type", json!("default")),
        ("/ln_type", json!("LayerNorm")),
        ("/use_istft_only", json!(true)),
        ("/hybrid_attention", json!(true)),
        ("/hybrid_block_size", json!(8)),
        ("/swa_per_block", json!(2)),
    ] {
        if config.pointer(path) != Some(&expected) {
            return Err(format!(
                "MiMo audio tokenizer config field {path} differs from pin"
            ));
        }
    }
    Ok(())
}

fn check_header_rows(headers: &HashMap<String, StInfo>) -> Result<(), String> {
    // Retain the exact sidecar as a fixture so a same-sized tensor offset swap
    // cannot pass a mere contiguous-span check. Its digest is checked at runtime.
    let pinned_bytes = include_bytes!("fixtures/audio-tokenizer-header.json");
    check_hash(pinned_bytes, HEADER_SHA256, "embedded header")?;
    let pinned = parse_header_json_checked(
        std::str::from_utf8(pinned_bytes)
            .map_err(|error| format!("MiMo audio tokenizer embedded header UTF-8: {error}"))?,
    )?;
    if pinned.len() != TENSOR_ROWS || headers.len() != TENSOR_ROWS {
        return Err(format!(
            "MiMo audio tokenizer header has {} tensors, expected {TENSOR_ROWS}",
            headers.len()
        ));
    }

    let mut f32_rows = 0;
    let mut bf16_rows = 0;
    let mut offsets = Vec::with_capacity(TENSOR_ROWS);
    for (name, info) in headers {
        let expected = pinned
            .get(name)
            .ok_or_else(|| format!("MiMo audio tokenizer header has unexpected tensor {name}"))?;
        if info.dtype != expected.dtype
            || info.shape != expected.shape
            || info.data_offsets != expected.data_offsets
        {
            return Err(format!(
                "MiMo audio tokenizer {name} dtype, shape, or offsets differ from pin"
            ));
        }
        let element_bytes = match info.dtype.as_str() {
            "F32" => {
                f32_rows += 1;
                4usize
            }
            "BF16" => {
                bf16_rows += 1;
                2usize
            }
            other => return Err(format!("MiMo audio tokenizer {name} has dtype {other}")),
        };
        let bytes = info
            .shape
            .iter()
            .try_fold(1usize, |count, &dim| {
                usize::try_from(dim)
                    .ok()
                    .and_then(|dim| count.checked_mul(dim))
            })
            .and_then(|elements| elements.checked_mul(element_bytes))
            .ok_or_else(|| format!("MiMo audio tokenizer {name} byte extent overflows"))?;
        let [start, end] = info.data_offsets;
        if end.checked_sub(start) != Some(bytes) {
            return Err(format!(
                "MiMo audio tokenizer {name} byte extent differs from shape"
            ));
        }
        offsets.push((start, end, name));
    }
    if f32_rows != F32_ROWS || bf16_rows != BF16_ROWS {
        return Err(format!(
            "MiMo audio tokenizer header dtype counts {f32_rows} F32, {bf16_rows} BF16 differ"
        ));
    }
    offsets.sort_by_key(|(start, _, _)| *start);
    let mut cursor = 0;
    for (start, end, name) in offsets {
        if start != cursor {
            return Err(format!(
                "MiMo audio tokenizer {name} has overlapping or missing bytes"
            ));
        }
        cursor = end;
    }
    if cursor != TENSOR_BYTES {
        return Err(format!(
            "MiMo audio tokenizer tensor span {cursor} differs from {TENSOR_BYTES}"
        ));
    }
    Ok(())
}

/// Verify pinned source metadata without reading the model payload.
///
/// `lfs_oid` and `lfs_file_bytes` must come from the model file's LFS pointer
/// or another trusted source manifest. Matching them does not verify payload
/// contents. The supplied header is the raw in-file JSON header.
pub fn verify_pinned_auxiliary(
    source: &str,
    config_bytes: &[u8],
    header_bytes: &[u8],
    lfs_oid: &str,
    lfs_file_bytes: usize,
) -> Result<AudioTokenizerAuxiliaryContract, String> {
    if source != SOURCE {
        return Err(format!(
            "MiMo audio tokenizer source {source:?} is not {SOURCE}"
        ));
    }
    check_hash(config_bytes, CONFIG_SHA256, "config")?;
    let config: Value = serde_json::from_slice(config_bytes)
        .map_err(|error| format!("MiMo audio tokenizer config JSON: {error}"))?;
    check_config_fields(&config)?;
    check_hash(header_bytes, HEADER_SHA256, "header")?;
    let headers = parse_header_json_checked(
        std::str::from_utf8(header_bytes)
            .map_err(|error| format!("MiMo audio tokenizer header UTF-8: {error}"))?,
    )?;
    check_header_rows(&headers)?;
    if lfs_oid != LFS_WEIGHT_SHA256 {
        return Err("MiMo audio tokenizer LFS weight SHA256 differs from pin".into());
    }
    if lfs_file_bytes != LFS_FILE_BYTES
        || header_bytes.len() != IN_FILE_HEADER_BYTES
        || lfs_file_bytes.checked_sub(TENSOR_BYTES + SAFETENSORS_PREFIX_BYTES)
            != Some(IN_FILE_HEADER_BYTES)
    {
        return Err("MiMo audio tokenizer LFS file length differs from pin".into());
    }
    Ok(AudioTokenizerAuxiliaryContract {
        encoder_layers: 24,
        decoder_layers: 24,
        hidden_size: 1_024,
        attention_heads: 16,
        ffn_size: 4_096,
        quantizers: 20,
        codebook_sizes: CODEBOOK_SIZES,
        sampling_rate: 24_000,
        mel_bins: 128,
        fft_size: 960,
        hop_length: 240,
        window_size: 960,
        hybrid_attention: true,
        hybrid_block_size: 8,
        swa_per_block: 2,
        tensor_rows: TENSOR_ROWS,
        f32_rows: F32_ROWS,
        bf16_rows: BF16_ROWS,
        tensor_bytes: TENSOR_BYTES,
        file_bytes: LFS_FILE_BYTES,
        in_file_header_bytes: IN_FILE_HEADER_BYTES,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixtures() -> (&'static [u8], &'static [u8]) {
        (
            include_bytes!("fixtures/audio-tokenizer-config.json"),
            include_bytes!("fixtures/audio-tokenizer-header.json"),
        )
    }

    fn parsed_header() -> HashMap<String, StInfo> {
        let (_, header) = fixtures();
        parse_header_json_checked(std::str::from_utf8(header).unwrap()).unwrap()
    }

    #[test]
    fn pinned_metadata_receipt_covers_all_codec_rows() {
        let (config, header) = fixtures();
        let receipt =
            verify_pinned_auxiliary(SOURCE, config, header, LFS_WEIGHT_SHA256, LFS_FILE_BYTES)
                .unwrap();
        assert_eq!(receipt.encoder_layers, 24);
        assert_eq!(receipt.decoder_layers, 24);
        assert_eq!(receipt.hidden_size, 1_024);
        assert_eq!(receipt.codebook_sizes, CODEBOOK_SIZES);
        assert_eq!(receipt.sampling_rate, 24_000);
        assert_eq!(receipt.mel_bins, 128);
        assert!(receipt.hybrid_attention);
        assert_eq!(receipt.tensor_rows, 828);
        assert_eq!(receipt.f32_rows, 459);
        assert_eq!(receipt.bf16_rows, 369);
        assert_eq!(receipt.tensor_bytes, 1_872_525_336);
        assert_eq!(receipt.file_bytes, 1_872_618_384);
        assert_eq!(receipt.in_file_header_bytes, 93_040);
        assert_eq!(header.len(), receipt.in_file_header_bytes);
    }

    #[test]
    fn source_config_header_and_lfs_mutations_are_refused() {
        let (config, header) = fixtures();
        assert!(
            verify_pinned_auxiliary(
                "other/revision",
                config,
                header,
                LFS_WEIGHT_SHA256,
                LFS_FILE_BYTES
            )
            .is_err()
        );
        let mut changed_config = config.to_vec();
        changed_config[0] = b'x';
        assert!(
            verify_pinned_auxiliary(
                SOURCE,
                &changed_config,
                header,
                LFS_WEIGHT_SHA256,
                LFS_FILE_BYTES
            )
            .is_err()
        );
        let mut parsed: Value = serde_json::from_slice(config).unwrap();
        parsed["encoder_layers"] = json!(23);
        assert!(check_config_fields(&parsed).is_err());
        let mut changed_header = header.to_vec();
        changed_header[0] = b'x';
        assert!(
            verify_pinned_auxiliary(
                SOURCE,
                config,
                &changed_header,
                LFS_WEIGHT_SHA256,
                LFS_FILE_BYTES
            )
            .is_err()
        );
        assert!(verify_pinned_auxiliary(SOURCE, config, header, "other", LFS_FILE_BYTES).is_err());
        assert!(
            verify_pinned_auxiliary(
                SOURCE,
                config,
                header,
                LFS_WEIGHT_SHA256,
                LFS_FILE_BYTES - 1
            )
            .is_err()
        );
    }

    #[test]
    fn tensor_name_dtype_shape_and_offsets_are_refused() {
        let mut rows = parsed_header();
        rows.remove("encoder.conv1.bias");
        assert!(check_header_rows(&rows).is_err());

        let mut rows = parsed_header();
        let row = rows.remove("encoder.conv1.bias").unwrap();
        rows.insert("encoder.other.bias".into(), row);
        assert!(check_header_rows(&rows).is_err());

        let mut rows = parsed_header();
        rows.get_mut("encoder.conv1.bias").unwrap().dtype = "F32".into();
        assert!(check_header_rows(&rows).is_err());

        let mut rows = parsed_header();
        rows.get_mut("encoder.conv1.bias").unwrap().shape[0] += 1;
        assert!(check_header_rows(&rows).is_err());

        let mut rows = parsed_header();
        rows.get_mut("encoder.conv1.bias").unwrap().data_offsets[1] -= 2;
        assert!(check_header_rows(&rows).is_err());

        let mut rows = parsed_header();
        let first = rows["decoder.dconv1.conv.bias"].data_offsets;
        let second = rows["decoder.dconv1.norm.bias"].data_offsets;
        rows.get_mut("decoder.dconv1.conv.bias")
            .unwrap()
            .data_offsets = second;
        rows.get_mut("decoder.dconv1.norm.bias")
            .unwrap()
            .data_offsets = first;
        assert!(check_header_rows(&rows).is_err());
    }
}
