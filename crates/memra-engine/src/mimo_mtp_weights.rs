//! Resident weights for the main checkpoint's three MiMo MTP blocks.
//! `model_mtp.safetensors` is distinct from the five-layer DFlash auxiliary.
//! This module only binds and uploads weights; it does not execute a draft.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use memra_gguf::checkpoint_binding::CheckpointBinding;
use memra_gguf::config::{Arch, ModelConfig};
use memra_gguf::model_packs::mimo_v2::bind_pinned_text_source;
use memra_gguf::safetensors::{StInfo, StModel};
use memra_gguf::source::SafetensorsSource;
use memra_gguf::tensor_contract::{
    BoundTensorContract, CheckpointDialect, FloatType, MtpTensor, StorageLayout, TensorId,
    TensorOwner, TensorTransform,
};
use sha2::{Digest, Sha256};

use crate::Engine;

type Fail = Box<dyn std::error::Error>;

const LAYERS: usize = 3;
const INDEX_BYTES: usize = 6_900_294;
const INDEX_SHA256: &str = "09d9b96a77ed9765fa4e02a6a45f92797702eef432da426efc6b45c1131b1812";
const HEADER_BYTES: usize = 5_504;
const PAYLOAD_BYTES: usize = 1_189_400_448;
const HEADER_SHA256: &str = "f0f3e77b85200e06acdcbb2146b8d3cf3c3c04f846eef9d60dc14a6c1e6f0cb0";

#[derive(Clone, Copy)]
struct WeightSpec {
    tensor: MtpTensor,
    suffix: &'static str,
    shape: &'static [u64],
    fp8: bool,
}

const SPECS: [WeightSpec; 12] = [
    WeightSpec {
        tensor: MtpTensor::FusionProjection,
        suffix: "eh_proj.weight",
        shape: &[4_096, 8_192],
        fp8: false,
    },
    WeightSpec {
        tensor: MtpTensor::EmbeddingNorm,
        suffix: "enorm.weight",
        shape: &[4_096],
        fp8: false,
    },
    WeightSpec {
        tensor: MtpTensor::OutputNorm,
        suffix: "final_layernorm.weight",
        shape: &[4_096],
        fp8: false,
    },
    WeightSpec {
        tensor: MtpTensor::HiddenNorm,
        suffix: "hnorm.weight",
        shape: &[4_096],
        fp8: false,
    },
    WeightSpec {
        tensor: MtpTensor::PreAttentionNorm,
        suffix: "input_layernorm.weight",
        shape: &[4_096],
        fp8: false,
    },
    WeightSpec {
        tensor: MtpTensor::MlpDown,
        suffix: "mlp.down_proj.weight",
        shape: &[4_096, 16_384],
        fp8: true,
    },
    WeightSpec {
        tensor: MtpTensor::MlpGate,
        suffix: "mlp.gate_proj.weight",
        shape: &[16_384, 4_096],
        fp8: true,
    },
    WeightSpec {
        tensor: MtpTensor::MlpUp,
        suffix: "mlp.up_proj.weight",
        shape: &[16_384, 4_096],
        fp8: true,
    },
    WeightSpec {
        tensor: MtpTensor::PreMlpNorm,
        suffix: "pre_mlp_layernorm.weight",
        shape: &[4_096],
        fp8: false,
    },
    WeightSpec {
        tensor: MtpTensor::AttentionSink,
        suffix: "self_attn.attention_sink_bias",
        shape: &[64],
        fp8: false,
    },
    WeightSpec {
        tensor: MtpTensor::AttentionOutput,
        suffix: "self_attn.o_proj.weight",
        shape: &[4_096, 8_192],
        fp8: false,
    },
    WeightSpec {
        tensor: MtpTensor::FusedQkv,
        suffix: "self_attn.qkv_proj.weight",
        shape: &[14_848, 4_096],
        fp8: true,
    },
];

/// Original BF16 bytes in checkpoint row-major shape, including norms and sinks.
pub struct MtpBf16 {
    pub data: CudaSlice<u8>,
    pub shape: Vec<u64>,
}

/// Original E4M3 codes and F32 block-128 scales in checkpoint row-major order.
pub struct MtpFp8 {
    pub codes: CudaSlice<u8>,
    pub scales: CudaSlice<f32>,
    pub shape: [u64; 2],
    pub scale_shape: [u64; 2],
}

pub enum MtpResidentTensor {
    Bf16(MtpBf16),
    Fp8(MtpFp8),
}

impl MtpResidentTensor {
    fn on_device(&self, ordinal: usize) -> bool {
        match self {
            Self::Bf16(weight) => weight.data.ordinal() == ordinal,
            Self::Fp8(weight) => {
                weight.codes.ordinal() == ordinal && weight.scales.ordinal() == ordinal
            }
        }
    }
}

pub struct Mtp3Layer {
    pub depth: u32,
    pub tensors: BTreeMap<MtpTensor, MtpResidentTensor>,
}

/// All three blocks on one selected GPU. The source handle remains owned so
/// the pinned binding and backing checkpoint live with the resident weights.
pub struct Mtp3Weights {
    pub layers: [Mtp3Layer; LAYERS],
    pub source: Arc<SafetensorsSource>,
    pub device_ordinal: usize,
}

// The source mmap and temporary F32 scale buffers must outlive queued copies,
// including when a later tensor fails validation or upload.
struct UploadDrain<'a> {
    engine: &'a Engine,
    active: bool,
}

impl Drop for UploadDrain<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.engine.stream().synchronize();
        }
    }
}

impl Mtp3Weights {
    /// Rebind the complete pinned main source, then validate and upload its
    /// separate MTP3 file. A different config, header, owner, or device fails.
    pub fn load(engine: &Engine, source: Arc<SafetensorsSource>) -> Result<Self, Fail> {
        let dir = memra_gguf::source::TensorSource::st_dir(source.as_ref())
            .ok_or("MiMo MTP3 source has no checkpoint directory")?;
        verify_pinned_index(&dir.join("model.safetensors.index.json"))?;
        let (config, _, binding) = bind_pinned_text_source(&source)?;
        if binding.family() != "mimo_v2_source"
            || binding.dialect != CheckpointDialect::HfSafetensors
        {
            return Err("MiMo MTP3 requires the pinned HF source binding".into());
        }
        let path = dir.join("model_mtp.safetensors");
        verify_pinned_file_header(&path)?;
        let sidecar = StModel::open(&path)?;
        let headers = sidecar
            .names()
            .map(|name| {
                sidecar
                    .info(name)
                    .map(|info| (name.clone(), info.clone()))
                    .ok_or_else(|| format!("MiMo MTP3 sidecar lost {name}"))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        check_schema(&config, &binding, &headers)?;

        engine.gpu.ctx.bind_to_thread()?;
        let ordinal = engine.stream().context().ordinal();
        let mut read = BTreeSet::new();
        // cudarc queues host-to-device copies on the stream. Retain the
        // decoded scale vectors and mapped sidecar until the final sync.
        let mut host_scales = Vec::with_capacity(12);
        let mut drain = UploadDrain {
            engine,
            active: true,
        };
        let mut layers = Vec::with_capacity(LAYERS);
        for depth in 0..LAYERS {
            let mut tensors = BTreeMap::new();
            for spec in SPECS {
                let name = tensor_name(depth, spec);
                let tensor = if spec.fp8 {
                    let scale_name = scale_name(&name)?;
                    let (_, codes) = sidecar
                        .raw(&name)
                        .ok_or_else(|| format!("MiMo MTP3 lost {name} during upload"))?;
                    let (_, scale_bytes) = sidecar
                        .raw(&scale_name)
                        .ok_or_else(|| format!("MiMo MTP3 lost {scale_name} during upload"))?;
                    check_fp8_codes(&name, codes)?;
                    let scales = decode_scales(&scale_name, scale_bytes)?;
                    let [out, input] = spec.shape else {
                        return Err(format!("{name}: FP8 weight is not a matrix").into());
                    };
                    let weight = MtpFp8 {
                        codes: engine.htod_bytes(codes)?,
                        scales: engine.htod(&scales)?,
                        shape: [*out, *input],
                        scale_shape: [out.div_ceil(128), input.div_ceil(128)],
                    };
                    host_scales.push(scales);
                    read.insert(scale_name);
                    MtpResidentTensor::Fp8(weight)
                } else {
                    let (_, bytes) = sidecar
                        .raw(&name)
                        .ok_or_else(|| format!("MiMo MTP3 lost {name} during upload"))?;
                    check_bf16(&name, bytes)?;
                    MtpResidentTensor::Bf16(MtpBf16 {
                        data: engine.htod_bytes(bytes)?,
                        shape: spec.shape.to_vec(),
                    })
                };
                if !tensor.on_device(ordinal) {
                    return Err(format!("{name}: upload landed on a different GPU").into());
                }
                read.insert(name);
                if tensors.insert(spec.tensor, tensor).is_some() {
                    return Err("MiMo MTP3 repeated a semantic tensor".into());
                }
            }
            layers.push(Mtp3Layer {
                depth: depth as u32,
                tensors,
            });
        }
        let expected: BTreeSet<_> = sidecar.names().cloned().collect();
        if read != expected {
            return Err(format!(
                "MiMo MTP3 source audit differs: unread {:?}, extra {:?}",
                expected.difference(&read).next(),
                read.difference(&expected).next()
            )
            .into());
        }
        engine.stream().synchronize()?;
        drain.active = false;
        let layers = layers
            .try_into()
            .map_err(|_: Vec<Mtp3Layer>| "MiMo MTP3 did not load exactly three blocks")?;
        let weights = Self {
            layers,
            source,
            device_ordinal: ordinal,
        };
        weights.check_device(engine)?;
        Ok(weights)
    }

    /// Refuse a forward owner on a different card, including a split code/scale pair.
    pub fn check_device(&self, engine: &Engine) -> Result<(), String> {
        let ordinal = engine.stream().context().ordinal();
        if ordinal != self.device_ordinal
            || self.layers.iter().any(|layer| {
                layer
                    .tensors
                    .values()
                    .any(|tensor| !tensor.on_device(ordinal))
            })
        {
            return Err(format!(
                "MiMo MTP3 belongs to GPU {}, not GPU {ordinal}",
                self.device_ordinal
            ));
        }
        Ok(())
    }
}

fn tensor_name(depth: usize, spec: WeightSpec) -> String {
    format!("model.mtp.layers.{depth}.{}", spec.suffix)
}

fn scale_name(name: &str) -> Result<String, String> {
    name.strip_suffix(".weight")
        .map(|stem| format!("{stem}.weight_scale_inv"))
        .ok_or_else(|| format!("{name}: FP8 name has no weight suffix"))
}

fn byte_len(shape: &[u64], item_bytes: usize) -> Result<usize, String> {
    shape
        .iter()
        .try_fold(item_bytes, |bytes, &dim| {
            usize::try_from(dim)
                .ok()
                .and_then(|dim| bytes.checked_mul(dim))
        })
        .ok_or_else(|| format!("{shape:?}: byte extent overflows"))
}

fn check_header_row(
    headers: &BTreeMap<String, StInfo>,
    name: &str,
    dtype: &str,
    shape: &[u64],
    item_bytes: usize,
) -> Result<usize, String> {
    let info = headers
        .get(name)
        .ok_or_else(|| format!("MiMo MTP3 is missing {name}"))?;
    let bytes = byte_len(shape, item_bytes)?;
    if info.dtype != dtype
        || info.shape != shape
        || info.data_offsets[1].checked_sub(info.data_offsets[0]) != Some(bytes)
    {
        return Err(format!(
            "MiMo MTP3 {name}: expected {dtype} {shape:?} and {bytes} bytes, got {} {:?} {:?}",
            info.dtype, info.shape, info.data_offsets
        ));
    }
    Ok(bytes)
}

fn check_schema(
    config: &ModelConfig,
    binding: &CheckpointBinding,
    headers: &BTreeMap<String, StInfo>,
) -> Result<(), String> {
    check_bound_schema(config, &binding.bound, headers)
}

fn check_bound_schema(
    config: &ModelConfig,
    bound: &BoundTensorContract,
    headers: &BTreeMap<String, StInfo>,
) -> Result<(), String> {
    if config.arch != Arch::MiMoV2
        || config.n_embd != 4_096
        || config.n_layer != 48
        || config.mimo.as_ref().and_then(|m| m.separate_mtp_layers) != Some(3)
    {
        return Err("MiMo MTP3 config differs from pinned main checkpoint".into());
    }
    if bound
        .tensors
        .keys()
        .filter(|id| matches!(id, TensorId::Mtp { .. }))
        .count()
        != LAYERS * SPECS.len()
    {
        return Err("MiMo MTP3 semantic binding has missing or extra tensors".into());
    }
    let mut expected_names = BTreeSet::new();
    for depth in 0..LAYERS {
        for spec in SPECS {
            let name = tensor_name(depth, spec);
            let id = TensorId::Mtp {
                depth: depth as u32,
                tensor: spec.tensor,
            };
            let tensor = bound
                .tensors
                .get(&id)
                .ok_or_else(|| format!("MiMo MTP3 source binds no {id:?}"))?;
            if tensor.owner != TensorOwner::Mtp(depth as u32)
                || tensor.transform != TensorTransform::Identity
                || tensor.checkpoint_names != [name.clone()]
                || tensor.shapes != [spec.shape.to_vec()]
                || tensor.storage.len() != 1
            {
                return Err(format!(
                    "{name}: semantic owner, shape, or source name changed"
                ));
            }
            let weight_bytes = check_header_row(
                headers,
                &name,
                if spec.fp8 { "F8_E4M3" } else { "BF16" },
                spec.shape,
                if spec.fp8 { 1 } else { 2 },
            )?;
            expected_names.insert(name.clone());
            let total_bytes = if spec.fp8 {
                let [out, input] = spec.shape else {
                    return Err(format!("{name}: FP8 shape is not rank two"));
                };
                let scale = scale_name(&name)?;
                let scale_shape = [out.div_ceil(128), input.div_ceil(128)];
                let scale_bytes = check_header_row(headers, &scale, "F32", &scale_shape, 4)?;
                expected_names.insert(scale.clone());
                if !matches!(&tensor.storage[0], StorageLayout::Quantized(layout)
                    if layout.format == "FP8_E4M3"
                        && layout.block_shape == [128, 128]
                        && layout.auxiliaries == [scale])
                {
                    return Err(format!("{name}: FP8 block scale binding changed"));
                }
                weight_bytes + scale_bytes
            } else {
                if tensor.storage[0] != StorageLayout::Float(FloatType::Bf16) {
                    return Err(format!("{name}: BF16 binding changed"));
                }
                weight_bytes
            };
            if tensor.physical_bytes != total_bytes as u64 {
                return Err(format!("{name}: bound byte extent changed"));
            }
        }
    }
    let actual_names: BTreeSet<_> = headers.keys().cloned().collect();
    if expected_names != actual_names || headers.len() != 48 {
        return Err(format!(
            "MiMo MTP3 namespace differs: missing {:?}, extra {:?}",
            expected_names.difference(&actual_names).next(),
            actual_names.difference(&expected_names).next()
        ));
    }
    let mut offsets: Vec<_> = headers
        .iter()
        .map(|(name, info)| (info.data_offsets[0], info.data_offsets[1], name))
        .collect();
    offsets.sort_unstable();
    let mut cursor = 0;
    for (start, end, name) in offsets {
        if start != cursor {
            return Err(format!("MiMo MTP3 {name} overlaps or leaves a byte gap"));
        }
        cursor = end;
    }
    if cursor != PAYLOAD_BYTES {
        return Err(format!(
            "MiMo MTP3 payload spans {cursor} bytes, expected {PAYLOAD_BYTES}"
        ));
    }
    Ok(())
}

fn verify_pinned_file_header(path: &Path) -> Result<(), Fail> {
    let mut file = std::fs::File::open(path)?;
    let mut length = [0u8; 8];
    file.read_exact(&mut length)?;
    if u64::from_le_bytes(length) != HEADER_BYTES as u64
        || file.metadata()?.len() != (8 + HEADER_BYTES + PAYLOAD_BYTES) as u64
    {
        return Err("MiMo MTP3 sidecar file extent differs from pinned artifact".into());
    }
    let mut header = vec![0u8; HEADER_BYTES];
    file.read_exact(&mut header)?;
    let digest = format!("{:x}", Sha256::digest(&header));
    if digest != HEADER_SHA256 {
        return Err(
            format!("MiMo MTP3 header changed: got {digest}, expected {HEADER_SHA256}").into(),
        );
    }
    Ok(())
}

fn verify_pinned_index(path: &Path) -> Result<(), Fail> {
    if std::fs::metadata(path)?.len() != INDEX_BYTES as u64 {
        return Err("MiMo MTP3 source index length changed".into());
    }
    let digest = format!("{:x}", Sha256::digest(std::fs::read(path)?));
    if digest != INDEX_SHA256 {
        return Err(format!(
            "MiMo MTP3 source index changed: got {digest}, expected {INDEX_SHA256}"
        )
        .into());
    }
    Ok(())
}

fn check_fp8_codes(name: &str, bytes: &[u8]) -> Result<(), String> {
    if bytes.iter().any(|code| code & 0x7f == 0x7f) {
        return Err(format!("{name}: FP8 weight contains a NaN code"));
    }
    Ok(())
}

fn decode_scales(name: &str, bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err(format!("{name}: F32 scale plane is not aligned"));
    }
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let scale = f32::from_le_bytes(chunk.try_into().expect("four-byte chunk"));
            if !scale.is_finite() || scale <= 0.0 {
                Err(format!("{name}: invalid FP8 block scale"))
            } else {
                Ok(scale)
            }
        })
        .collect()
}

fn check_bf16(name: &str, bytes: &[u8]) -> Result<(), String> {
    if !bytes.len().is_multiple_of(2)
        || bytes.chunks_exact(2).any(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            bits & 0x7f80 == 0x7f80
        })
    {
        return Err(format!("{name}: BF16 weight is non-finite or misaligned"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::HfConfig;
    use memra_gguf::model_packs::mimo_v2::SOURCE_PROFILE;
    use memra_gguf::source::census_from_safetensors_headers;
    use memra_gguf::tensor_contract::{ContractOptions, TensorContract};

    fn fixture() -> (ModelConfig, BoundTensorContract, BTreeMap<String, StInfo>) {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ))));
        let plan = SOURCE_PROFILE.compile_plan(&config).unwrap();
        let mut contract = SOURCE_PROFILE
            .compile_tensor_contract(
                &config,
                &plan,
                CheckpointDialect::HfSafetensors,
                ContractOptions::default(),
            )
            .unwrap();
        contract
            .requirements
            .retain(|row| matches!(row.id, TensorId::Mtp { .. }));
        let mut headers = BTreeMap::new();
        let mut cursor = 0;
        for depth in 0..LAYERS {
            for spec in SPECS {
                let name = tensor_name(depth, spec);
                let bytes = byte_len(spec.shape, if spec.fp8 { 1 } else { 2 }).unwrap();
                headers.insert(
                    name.clone(),
                    StInfo {
                        dtype: if spec.fp8 { "F8_E4M3" } else { "BF16" }.into(),
                        shape: spec.shape.to_vec(),
                        data_offsets: [cursor, cursor + bytes],
                    },
                );
                cursor += bytes;
                if spec.fp8 {
                    let scale = scale_name(&name).unwrap();
                    let shape = vec![spec.shape[0].div_ceil(128), spec.shape[1].div_ceil(128)];
                    let bytes = byte_len(&shape, 4).unwrap();
                    headers.insert(
                        scale,
                        StInfo {
                            dtype: "F32".into(),
                            shape,
                            data_offsets: [cursor, cursor + bytes],
                        },
                    );
                    cursor += bytes;
                }
            }
        }
        assert_eq!(cursor, PAYLOAD_BYTES);
        let census = census_from_safetensors_headers(&headers).unwrap();
        let contract = TensorContract {
            dialect: CheckpointDialect::HfSafetensors,
            requirements: contract.requirements,
        };
        let bound = contract
            .bind(
                &census
                    .tensors
                    .into_iter()
                    .map(|row| row.entry)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        (config, bound, headers)
    }

    #[test]
    fn pinned_mtp3_binding_covers_every_weight_and_scale() {
        let (config, bound, headers) = fixture();
        check_bound_schema(&config, &bound, &headers).unwrap();
        assert_eq!(bound.tensors.len(), 36);
        assert_eq!(headers.len(), 48);
    }

    #[test]
    fn owner_scale_and_namespace_drift_are_refused() {
        let (config, mut bound, mut headers) = fixture();
        let id = TensorId::Mtp {
            depth: 0,
            tensor: MtpTensor::FusedQkv,
        };
        bound.tensors.get_mut(&id).unwrap().owner = TensorOwner::Mtp(1);
        assert!(check_bound_schema(&config, &bound, &headers).is_err());
        bound.tensors.get_mut(&id).unwrap().owner = TensorOwner::Mtp(0);
        let scale = "model.mtp.layers.0.self_attn.qkv_proj.weight_scale_inv";
        headers.get_mut(scale).unwrap().shape[0] = 1;
        assert!(check_bound_schema(&config, &bound, &headers).is_err());
        headers.get_mut(scale).unwrap().shape[0] = 116;
        headers.insert(
            "model.mtp.layers.3.eh_proj.weight".into(),
            StInfo {
                dtype: "BF16".into(),
                shape: vec![1],
                data_offsets: [PAYLOAD_BYTES, PAYLOAD_BYTES + 2],
            },
        );
        assert!(check_bound_schema(&config, &bound, &headers).is_err());
    }

    #[test]
    fn depth_offsets_and_native_payload_errors_are_refused() {
        let (mut config, bound, mut headers) = fixture();
        config.mimo.as_mut().unwrap().separate_mtp_layers = Some(2);
        assert!(check_bound_schema(&config, &bound, &headers).is_err());
        config.mimo.as_mut().unwrap().separate_mtp_layers = Some(3);
        headers
            .get_mut("model.mtp.layers.2.enorm.weight")
            .unwrap()
            .data_offsets[0] += 2;
        assert!(check_bound_schema(&config, &bound, &headers).is_err());
        assert!(check_fp8_codes("qkv", &[0, 0x7f]).is_err());
        assert!(decode_scales("qkv_scale", &0.0f32.to_le_bytes()).is_err());
        assert!(check_bf16("norm", &0x7f80u16.to_le_bytes()).is_err());
        assert_eq!(
            decode_scales("scale", &1.25f32.to_le_bytes()).unwrap(),
            [1.25]
        );
    }
}
