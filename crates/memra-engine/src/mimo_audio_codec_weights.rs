//! Resident weights for the pinned bundled MiMo audio-tokenizer encoder.
//! This loads original BF16/F32 bytes only. It does not run the codec.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::path::Path;

use cudarc::driver::CudaSlice;
use memmap2::Mmap;
use memra_gguf::model_packs::mimo_v2::audio_tokenizer::{
    AudioTokenizerAuxiliaryContract, LFS_WEIGHT_SHA256, SOURCE, verify_pinned_auxiliary,
};
use memra_gguf::safetensors::{StInfo, parse_header_json_checked};
use sha2::{Digest, Sha256};

use crate::Engine;

type Fail = Box<dyn std::error::Error>;

const FILE_BYTES: usize = 1_872_618_384;
const CONFIG_BYTES: u64 = 1_215;
const HEADER_BYTES: usize = 93_040;
const ENCODER_ROWS: usize = 449;
const DECODER_ROWS: usize = 379;
const ENCODER_BYTES: usize = 652_572_240;
const ENCODER_F32_ROWS: usize = 80;
const ENCODER_BF16_ROWS: usize = 369;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecEncoderDtype {
    Bf16,
    F32,
}

impl CodecEncoderDtype {
    fn from_header(name: &str, dtype: &str) -> Result<Self, String> {
        match dtype {
            "BF16" => Ok(Self::Bf16),
            "F32" => Ok(Self::F32),
            other => Err(format!(
                "MiMo codec encoder {name} has unsupported dtype {other}"
            )),
        }
    }

    fn byte_width(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Debug, Clone)]
struct EncoderRow {
    name: String,
    dtype: CodecEncoderDtype,
    shape: Vec<u64>,
    offsets: [usize; 2],
}

impl EncoderRow {
    fn byte_len(&self) -> Result<usize, String> {
        self.shape
            .iter()
            .try_fold(self.dtype.byte_width(), |bytes, &dimension| {
                usize::try_from(dimension)
                    .ok()
                    .and_then(|dimension| bytes.checked_mul(dimension))
            })
            .ok_or_else(|| format!("MiMo codec encoder {} byte extent overflows", self.name))
    }
}

/// Original checkpoint bytes on one GPU, with their source dtype and row-major shape.
pub struct CodecEncoderTensor {
    pub bytes: CudaSlice<u8>,
    pub dtype: CodecEncoderDtype,
    pub shape: Vec<u64>,
    pub data_offsets: [usize; 2],
}

/// Exactly the encoder half of `audio_tokenizer/model.safetensors`.
pub struct MiMoAudioCodecEncoderWeights {
    tensors: BTreeMap<String, CodecEncoderTensor>,
    pub contract: AudioTokenizerAuxiliaryContract,
    pub device_ordinal: usize,
}

/// A validated BF16 Conv1D weight and bias pair from the pinned encoder.
///
/// The byte slices contain little-endian BF16 values. The weight is contiguous
/// `[out_channels, in_channels, 3]`; the bias is `[out_channels]`.
pub struct CodecEncoderBf16Conv1d<'a> {
    weight: &'a CudaSlice<u8>,
    bias: &'a CudaSlice<u8>,
    in_channels: usize,
    out_channels: usize,
    stride: usize,
}

/// Borrowed, geometry-checked source BF16 matrix and optional bias.
pub struct CodecEncoderBf16Linear<'a> {
    pub weight: &'a CudaSlice<u8>,
    pub bias: Option<&'a CudaSlice<u8>>,
    pub input: usize,
    pub output: usize,
}

/// Source LayerNorm affine vectors, both BF16.
pub struct CodecEncoderBf16Norm<'a> {
    pub weight: &'a CudaSlice<u8>,
    pub bias: &'a CudaSlice<u8>,
}

/// One of the 24 pinned encoder layers. No codec decoder tensors are exposed.
pub struct CodecEncoderBf16Layer<'a> {
    pub attention_norm: CodecEncoderBf16Norm<'a>,
    pub query: CodecEncoderBf16Linear<'a>,
    pub key: CodecEncoderBf16Linear<'a>,
    pub value: CodecEncoderBf16Linear<'a>,
    pub attention_output: CodecEncoderBf16Linear<'a>,
    pub final_norm: CodecEncoderBf16Norm<'a>,
    pub fc1: CodecEncoderBf16Linear<'a>,
    pub fc2: CodecEncoderBf16Linear<'a>,
    pub layer_index: usize,
}

impl CodecEncoderBf16Layer<'_> {
    /// The bundled hybrid plan alternates 128-window and full causal layers.
    pub fn attention_window(&self) -> Option<usize> {
        if self.layer_index.is_multiple_of(2) {
            Some(128)
        } else {
            None
        }
    }
}

impl CodecEncoderBf16Conv1d<'_> {
    pub fn weight(&self) -> &CudaSlice<u8> {
        self.weight
    }

    pub fn bias(&self) -> &CudaSlice<u8> {
        self.bias
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    pub fn stride(&self) -> usize {
        self.stride
    }
}

fn check_conv_row(
    name: &str,
    dtype: CodecEncoderDtype,
    shape: &[u64],
    byte_len: usize,
    expected_shape: &[u64],
) -> Result<(), String> {
    let expected_bytes = expected_shape.iter().try_fold(2usize, |bytes, &dimension| {
        usize::try_from(dimension)
            .ok()
            .and_then(|dimension| bytes.checked_mul(dimension))
    });
    if dtype != CodecEncoderDtype::Bf16
        || shape != expected_shape
        || expected_bytes != Some(byte_len)
    {
        return Err(format!(
            "MiMo codec {name} BF16 shape or byte extent changed"
        ));
    }
    Ok(())
}

// Drop synchronizes before its tensor fields are freed. The source mapping is
// declared before this owner and remains alive until the same synchronization.
struct PendingUpload<'a> {
    engine: &'a Engine,
    tensors: BTreeMap<String, CodecEncoderTensor>,
    active: bool,
}

impl Drop for PendingUpload<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.engine.stream().synchronize();
        }
    }
}

impl PendingUpload<'_> {
    fn finish(mut self) -> Result<BTreeMap<String, CodecEncoderTensor>, Fail> {
        self.engine.stream().synchronize()?;
        self.active = false;
        Ok(std::mem::take(&mut self.tensors))
    }
}

fn verify_payload_hash(bytes: &[u8], expected: &str) -> Result<(), String> {
    let found = format!("{:x}", Sha256::digest(bytes));
    if found != expected {
        return Err(format!(
            "MiMo codec weight payload SHA256 changed: got {found}, expected {expected}"
        ));
    }
    Ok(())
}

fn in_file_header(bytes: &[u8]) -> Result<&[u8], String> {
    let prefix: [u8; 8] = bytes
        .get(..8)
        .ok_or("MiMo codec weight file has no safetensors prefix")?
        .try_into()
        .map_err(|_| "MiMo codec weight prefix has wrong length")?;
    let length = usize::try_from(u64::from_le_bytes(prefix))
        .map_err(|_| "MiMo codec header length overflows usize")?;
    if length != HEADER_BYTES {
        return Err(format!(
            "MiMo codec in-file header length {length} differs from pinned {HEADER_BYTES}"
        ));
    }
    bytes
        .get(8..8 + length)
        .ok_or_else(|| "MiMo codec in-file header is truncated".into())
}

fn select_encoder_rows(headers: &HashMap<String, StInfo>) -> Result<Vec<EncoderRow>, String> {
    let mut selected = Vec::with_capacity(ENCODER_ROWS);
    let mut decoder_rows = 0;
    let mut total_bytes = 0usize;
    let mut f32_rows = 0;
    let mut bf16_rows = 0;
    for (name, info) in headers {
        if name.starts_with("decoder.") {
            decoder_rows += 1;
            continue;
        }
        if !name.starts_with("encoder.") {
            return Err(format!("MiMo codec has unexpected tensor {name}"));
        }
        let dtype = CodecEncoderDtype::from_header(name, &info.dtype)?;
        let row = EncoderRow {
            name: name.clone(),
            dtype,
            shape: info.shape.clone(),
            offsets: info.data_offsets,
        };
        let bytes = row.byte_len()?;
        if row.offsets[1].checked_sub(row.offsets[0]) != Some(bytes) {
            return Err(format!(
                "MiMo codec encoder {name} shape and byte offsets disagree"
            ));
        }
        total_bytes = total_bytes
            .checked_add(bytes)
            .ok_or("MiMo codec encoder byte total overflows")?;
        match dtype {
            CodecEncoderDtype::F32 => f32_rows += 1,
            CodecEncoderDtype::Bf16 => bf16_rows += 1,
        }
        selected.push(row);
    }
    if selected.len() != ENCODER_ROWS
        || decoder_rows != DECODER_ROWS
        || total_bytes != ENCODER_BYTES
        || f32_rows != ENCODER_F32_ROWS
        || bf16_rows != ENCODER_BF16_ROWS
    {
        return Err(format!(
            "MiMo codec encoder selection differs from pin: {} encoder rows, {decoder_rows} \
             decoder rows, {total_bytes} encoder bytes, {f32_rows} F32, {bf16_rows} BF16",
            selected.len()
        ));
    }
    selected.sort_unstable_by(|a, b| a.offsets[0].cmp(&b.offsets[0]).then(a.name.cmp(&b.name)));
    Ok(selected)
}

fn audit_selected_rows(rows: &[EncoderRow], uploaded: &BTreeSet<String>) -> Result<(), String> {
    let expected: BTreeSet<_> = rows.iter().map(|row| row.name.clone()).collect();
    if *uploaded != expected {
        return Err(format!(
            "MiMo codec encoder source audit differs: unread {:?}, extra {:?}",
            expected.difference(uploaded).next(),
            uploaded.difference(&expected).next()
        ));
    }
    Ok(())
}

fn finite_bf16_payload(bytes: &[u8]) -> bool {
    bytes.len().is_multiple_of(2)
        && bytes
            .chunks_exact(2)
            .all(|pair| u16::from_le_bytes([pair[0], pair[1]]) & 0x7f80 != 0x7f80)
}

impl MiMoAudioCodecEncoderWeights {
    pub fn tensor(&self, name: &str) -> Option<&CodecEncoderTensor> {
        self.tensors.get(name)
    }

    pub fn tensors(&self) -> &BTreeMap<String, CodecEncoderTensor> {
        &self.tensors
    }

    fn layer_row(&self, name: &str, shape: &[u64]) -> Result<&CudaSlice<u8>, String> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("MiMo codec encoder missing {name}"))?;
        check_conv_row(name, tensor.dtype, &tensor.shape, tensor.bytes.len(), shape)?;
        if tensor.bytes.ordinal() != self.device_ordinal {
            return Err(format!("MiMo codec encoder {name} crossed GPU devices"));
        }
        Ok(&tensor.bytes)
    }

    fn layer_norm_bf16(&self, stem: &str) -> Result<CodecEncoderBf16Norm<'_>, String> {
        Ok(CodecEncoderBf16Norm {
            weight: self.layer_row(&format!("{stem}.weight"), &[1_024])?,
            bias: self.layer_row(&format!("{stem}.bias"), &[1_024])?,
        })
    }

    fn layer_linear_bf16(
        &self,
        stem: &str,
        input: usize,
        output: usize,
        has_bias: bool,
    ) -> Result<CodecEncoderBf16Linear<'_>, String> {
        let weight = self.layer_row(&format!("{stem}.weight"), &[output as u64, input as u64])?;
        let bias = if has_bias {
            Some(self.layer_row(&format!("{stem}.bias"), &[output as u64])?)
        } else {
            if self.tensors.contains_key(&format!("{stem}.bias")) {
                return Err(format!(
                    "MiMo codec encoder {stem} gained an unsupported bias"
                ));
            }
            None
        };
        Ok(CodecEncoderBf16Linear {
            weight,
            bias,
            input,
            output,
        })
    }

    /// Bind all 15 physical BF16 rows for one pinned transformer layer.
    pub fn encoder_layer_bf16(&self, layer: usize) -> Result<CodecEncoderBf16Layer<'_>, String> {
        if layer >= 24
            || self.contract.encoder_layers != 24
            || self.contract.hidden_size != 1_024
            || self.contract.attention_heads != 16
            || self.contract.ffn_size != 4_096
            || !self.contract.hybrid_attention
            || self.contract.swa_per_block != 2
        {
            return Err("MiMo codec encoder layer or pinned plan changed".into());
        }
        let stem = format!("encoder.layers.{layer}");
        let attention = format!("{stem}.self_attn");
        let bound = CodecEncoderBf16Layer {
            attention_norm: self.layer_norm_bf16(&format!("{stem}.self_attn_layer_norm"))?,
            query: self.layer_linear_bf16(&format!("{attention}.q_proj"), 1_024, 1_024, true)?,
            key: self.layer_linear_bf16(&format!("{attention}.k_proj"), 1_024, 1_024, false)?,
            value: self.layer_linear_bf16(&format!("{attention}.v_proj"), 1_024, 1_024, true)?,
            attention_output: self.layer_linear_bf16(
                &format!("{attention}.out_proj"),
                1_024,
                1_024,
                true,
            )?,
            final_norm: self.layer_norm_bf16(&format!("{stem}.final_layer_norm"))?,
            fc1: self.layer_linear_bf16(&format!("{stem}.fc1"), 1_024, 4_096, true)?,
            fc2: self.layer_linear_bf16(&format!("{stem}.fc2"), 4_096, 1_024, true)?,
            layer_index: layer,
        };
        Ok(bound)
    }

    /// Bind the source `encoder.layer_norm` after all 24 transformer layers.
    /// `load` has verified the exact source config and weight payload before
    /// these BF16 vectors can be borrowed.
    pub fn encoder_final_norm_bf16(&self) -> Result<CodecEncoderBf16Norm<'_>, String> {
        if self.contract.encoder_layers != 24 || self.contract.hidden_size != 1_024 {
            return Err("MiMo codec encoder final LayerNorm plan changed".into());
        }
        self.layer_norm_bf16("encoder.layer_norm")
    }

    fn conv_bf16(
        &self,
        stem: &str,
        in_channels: usize,
        stride: usize,
    ) -> Result<CodecEncoderBf16Conv1d<'_>, String> {
        const OUT: usize = 1_024;
        let weight_name = format!("{stem}.weight");
        let bias_name = format!("{stem}.bias");
        let weight = self
            .tensors
            .get(&weight_name)
            .ok_or_else(|| format!("MiMo codec missing {weight_name}"))?;
        let bias = self
            .tensors
            .get(&bias_name)
            .ok_or_else(|| format!("MiMo codec missing {bias_name}"))?;
        check_conv_row(
            &weight_name,
            weight.dtype,
            &weight.shape,
            weight.bytes.len(),
            &[OUT as u64, in_channels as u64, 3],
        )?;
        check_conv_row(
            &bias_name,
            bias.dtype,
            &bias.shape,
            bias.bytes.len(),
            &[OUT as u64],
        )?;
        if weight.bytes.ordinal() != self.device_ordinal
            || bias.bytes.ordinal() != self.device_ordinal
        {
            return Err(format!("MiMo codec {stem} crossed GPU devices"));
        }
        Ok(CodecEncoderBf16Conv1d {
            weight: &weight.bytes,
            bias: &bias.bytes,
            in_channels,
            out_channels: OUT,
            stride,
        })
    }

    pub fn conv1_bf16(&self) -> Result<CodecEncoderBf16Conv1d<'_>, String> {
        self.conv_bf16("encoder.conv1", 128, 1)
    }

    pub fn conv2_bf16(&self) -> Result<CodecEncoderBf16Conv1d<'_>, String> {
        self.conv_bf16("encoder.conv2", 1_024, 2)
    }

    /// Verify the bundled config, actual in-file header, file length and full
    /// payload SHA256, then upload exactly 449 encoder rows to `engine`'s GPU.
    ///
    /// `repo_root` is the pinned MiMo repository snapshot root. HF cache
    /// symlinks are accepted; the weight file is opened once and the same mmap
    /// is hashed and used for every queued device copy.
    pub fn load(engine: &Engine, repo_root: &Path) -> Result<Self, Fail> {
        let config_path = repo_root.join("audio_tokenizer/config.json");
        if std::fs::metadata(&config_path)?.len() != CONFIG_BYTES {
            return Err("MiMo codec config length differs from pin".into());
        }
        let config = std::fs::read(config_path)?;
        let path = repo_root.join("audio_tokenizer/model.safetensors");
        let file = File::open(&path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.len() != FILE_BYTES as u64 {
            return Err(format!(
                "MiMo codec weight file is not the pinned regular {FILE_BYTES}-byte file"
            )
            .into());
        }
        // SAFETY: `file` stays open until after `mmap` is dropped. The weight
        // artifact is immutable for the duration of this load.
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() != FILE_BYTES {
            return Err("MiMo codec mapped weight length changed".into());
        }
        let header_bytes = in_file_header(&mmap)?;
        let contract =
            verify_pinned_auxiliary(SOURCE, &config, header_bytes, LFS_WEIGHT_SHA256, mmap.len())?;
        verify_payload_hash(&mmap, LFS_WEIGHT_SHA256)?;
        let header = std::str::from_utf8(header_bytes)?;
        let headers = parse_header_json_checked(header)?;
        let rows = select_encoder_rows(&headers)?;
        let data_base = 8 + header_bytes.len();

        engine.gpu.ctx.bind_to_thread()?;
        let ordinal = engine.stream().context().ordinal();
        let mut pending = PendingUpload {
            engine,
            tensors: BTreeMap::new(),
            active: true,
        };
        let mut uploaded = BTreeSet::new();
        for row in &rows {
            let start = data_base
                .checked_add(row.offsets[0])
                .ok_or("MiMo codec encoder start offset overflows")?;
            let end = data_base
                .checked_add(row.offsets[1])
                .ok_or("MiMo codec encoder end offset overflows")?;
            let bytes = mmap
                .get(start..end)
                .ok_or_else(|| format!("MiMo codec encoder {} exceeds mapped file", row.name))?;
            if bytes.len() != row.byte_len()? {
                return Err(format!("MiMo codec encoder {} byte extent changed", row.name).into());
            }
            if row.dtype == CodecEncoderDtype::Bf16 && !finite_bf16_payload(bytes) {
                return Err(format!(
                    "MiMo codec encoder {} has non-finite BF16 weights",
                    row.name
                )
                .into());
            }
            let tensor = CodecEncoderTensor {
                bytes: engine.htod_bytes(bytes)?,
                dtype: row.dtype,
                shape: row.shape.clone(),
                data_offsets: row.offsets,
            };
            if pending.tensors.insert(row.name.clone(), tensor).is_some() {
                return Err(format!("MiMo codec encoder {} was uploaded twice", row.name).into());
            }
            if pending.tensors[&row.name].bytes.ordinal() != ordinal {
                return Err(
                    format!("MiMo codec encoder {} landed on another GPU", row.name).into(),
                );
            }
            uploaded.insert(row.name.clone());
        }
        audit_selected_rows(&rows, &uploaded)?;
        let tensors = pending.finish()?;
        let weights = Self {
            tensors,
            contract,
            device_ordinal: ordinal,
        };
        weights.check_device(engine)?;
        Ok(weights)
    }

    /// Refuse use from a GPU other than the one that owns all encoder rows.
    pub fn check_device(&self, engine: &Engine) -> Result<(), String> {
        let ordinal = engine.stream().context().ordinal();
        if ordinal != self.device_ordinal
            || self
                .tensors
                .values()
                .any(|tensor| tensor.bytes.ordinal() != ordinal)
        {
            return Err(format!(
                "MiMo codec encoder belongs to GPU {}, not GPU {ordinal}",
                self.device_ordinal
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!(
        "../../memra-gguf/src/model_packs/mimo_v2/fixtures/audio-tokenizer-header.json"
    );

    fn fixture_rows() -> HashMap<String, StInfo> {
        parse_header_json_checked(FIXTURE).expect("pinned header fixture parses")
    }

    #[test]
    fn exact_encoder_partition_and_decoder_exclusion() {
        let rows = select_encoder_rows(&fixture_rows()).expect("pinned encoder partition");
        assert_eq!(rows.len(), ENCODER_ROWS);
        assert_eq!(
            rows.iter()
                .map(|row| row.byte_len().unwrap())
                .sum::<usize>(),
            ENCODER_BYTES
        );
        assert!(rows.iter().all(|row| row.name.starts_with("encoder.")));
        assert!(
            rows.windows(2)
                .all(|pair| pair[0].offsets[0] <= pair[1].offsets[0])
        );
        let read: BTreeSet<_> = rows.iter().map(|row| row.name.clone()).collect();
        audit_selected_rows(&rows, &read).expect("all encoder rows consumed");
        let mut unread = read.clone();
        unread.remove(&rows[0].name);
        assert!(audit_selected_rows(&rows, &unread).is_err());
        let mut extra = read;
        extra.insert("decoder.dconv1.conv.bias".into());
        assert!(audit_selected_rows(&rows, &extra).is_err());
    }

    #[test]
    fn encoder_schema_mutations_are_refused() {
        let mut rows = fixture_rows();
        rows.remove("encoder.conv1.bias");
        assert!(select_encoder_rows(&rows).is_err());

        let mut rows = fixture_rows();
        rows.get_mut("encoder.conv1.bias").unwrap().dtype = "F32".into();
        assert!(select_encoder_rows(&rows).is_err());

        let mut rows = fixture_rows();
        rows.get_mut("encoder.conv1.bias").unwrap().data_offsets[1] -= 2;
        assert!(select_encoder_rows(&rows).is_err());

        let mut rows = fixture_rows();
        let row = rows.remove("encoder.conv1.bias").unwrap();
        rows.insert("unexpected.conv1.bias".into(), row);
        assert!(select_encoder_rows(&rows).is_err());
    }

    #[test]
    fn bf16_payload_finiteness_is_checked_once_before_upload() {
        assert!(finite_bf16_payload(&[0x80, 0x3f, 0x00, 0xc0]));
        assert!(!finite_bf16_payload(&[0x80, 0x7f]));
        assert!(!finite_bf16_payload(&[0xc1, 0xff]));
        assert!(!finite_bf16_payload(&[0x80]));
    }

    #[test]
    fn in_file_prefix_and_payload_mutations_are_refused() {
        let digest = format!("{:x}", Sha256::digest(b"original"));
        verify_payload_hash(b"original", &digest).unwrap();
        assert!(verify_payload_hash(b"changed", &digest).is_err());

        let mut file = Vec::from((HEADER_BYTES as u64).to_le_bytes());
        file.extend(std::iter::repeat_n(0, HEADER_BYTES));
        assert_eq!(in_file_header(&file).unwrap().len(), HEADER_BYTES);
        file[0] ^= 1;
        assert!(in_file_header(&file).is_err());
        file.clear();
        assert!(in_file_header(&file).is_err());
    }

    #[test]
    fn pinned_conv_rows_have_exact_bf16_shapes_and_extents() {
        let rows = fixture_rows();
        for (name, shape) in [
            ("encoder.conv1.weight", vec![1_024, 128, 3]),
            ("encoder.conv1.bias", vec![1_024]),
            ("encoder.conv2.weight", vec![1_024, 1_024, 3]),
            ("encoder.conv2.bias", vec![1_024]),
        ] {
            let row = rows.get(name).unwrap();
            let bytes = row.data_offsets[1] - row.data_offsets[0];
            check_conv_row(
                name,
                CodecEncoderDtype::from_header(name, &row.dtype).unwrap(),
                &row.shape,
                bytes,
                &shape,
            )
            .unwrap();
            assert!(
                check_conv_row(name, CodecEncoderDtype::F32, &row.shape, bytes, &shape).is_err()
            );
            assert!(
                check_conv_row(name, CodecEncoderDtype::Bf16, &row.shape, bytes - 2, &shape)
                    .is_err()
            );
        }
    }

    #[test]
    fn pinned_layer_rows_match_projection_bias_and_norm_contract() {
        let rows = fixture_rows();
        for layer in 0..24 {
            let stem = format!("encoder.layers.{layer}");
            let selected: BTreeSet<_> = rows
                .keys()
                .filter(|name| name.starts_with(&format!("{stem}.")))
                .cloned()
                .collect();
            assert_eq!(selected.len(), 15);
            for (suffix, shape) in [
                ("self_attn_layer_norm.weight", vec![1_024]),
                ("self_attn_layer_norm.bias", vec![1_024]),
                ("self_attn.q_proj.weight", vec![1_024, 1_024]),
                ("self_attn.q_proj.bias", vec![1_024]),
                ("self_attn.k_proj.weight", vec![1_024, 1_024]),
                ("self_attn.v_proj.weight", vec![1_024, 1_024]),
                ("self_attn.v_proj.bias", vec![1_024]),
                ("self_attn.out_proj.weight", vec![1_024, 1_024]),
                ("self_attn.out_proj.bias", vec![1_024]),
                ("final_layer_norm.weight", vec![1_024]),
                ("final_layer_norm.bias", vec![1_024]),
                ("fc1.weight", vec![4_096, 1_024]),
                ("fc1.bias", vec![4_096]),
                ("fc2.weight", vec![1_024, 4_096]),
                ("fc2.bias", vec![1_024]),
            ] {
                let name = format!("{stem}.{suffix}");
                let row = &rows[&name];
                check_conv_row(
                    &name,
                    CodecEncoderDtype::from_header(&name, &row.dtype).unwrap(),
                    &row.shape,
                    row.data_offsets[1] - row.data_offsets[0],
                    &shape,
                )
                .unwrap();
            }
            assert!(!selected.contains(&format!("{stem}.self_attn.k_proj.bias")));
        }
    }

    #[test]
    fn pinned_encoder_final_norm_is_exact_bf16_affine() {
        let rows = fixture_rows();
        for suffix in ["weight", "bias"] {
            let name = format!("encoder.layer_norm.{suffix}");
            let row = &rows[&name];
            let bytes = row.data_offsets[1] - row.data_offsets[0];
            check_conv_row(
                &name,
                CodecEncoderDtype::from_header(&name, &row.dtype).unwrap(),
                &row.shape,
                bytes,
                &[1_024],
            )
            .unwrap();
            assert!(
                check_conv_row(&name, CodecEncoderDtype::F32, &row.shape, bytes, &[1_024]).is_err()
            );
            assert!(
                check_conv_row(&name, CodecEncoderDtype::Bf16, &row.shape, bytes, &[1_023])
                    .is_err()
            );
        }
        let config = include_bytes!(
            "../../memra-gguf/src/model_packs/mimo_v2/fixtures/audio-tokenizer-config.json"
        );
        let contract = verify_pinned_auxiliary(
            SOURCE,
            config,
            FIXTURE.as_bytes(),
            LFS_WEIGHT_SHA256,
            FILE_BYTES,
        )
        .unwrap();
        assert_eq!(contract.encoder_layers, 24);
        assert_eq!(contract.hidden_size, 1_024);
    }
}
