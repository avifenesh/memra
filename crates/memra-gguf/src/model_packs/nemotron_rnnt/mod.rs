//! Nemotron 3.5 streaming RNNT pack: archive layout, tensor contract and streaming state
//! contract. Native execution is unsupported; this binds a checkpoint and refuses to run it.
//!
//! The contract is pack-local rather than the shared `TensorContract`. That type is keyed by
//! `CheckpointDialect`, which knows GGUF and HF safetensors, and folding a torch-zip dialect
//! into it means touching every text builder that matches on the dialect. That is the next
//! stage's change, not a thing to smuggle in behind a speech census.

use crate::nemo::{
    NemoDtype, NemoError, NemoTensor, pickle_census, shared_storage_groups, tar_members,
    zip_entries,
};
use std::collections::BTreeMap;
use std::fmt;

/// Base publisher artifact this architecture was read from.
pub const SOURCE_REPOSITORY: &str = "nvidia/nemotron-3.5-asr-streaming-0.6b";
pub const SOURCE_REVISION: &str = "1c8deaecc64b91f034d73e08dd8b64625eb3395d";
/// Base `.nemo` archive, verified against the publisher hash on 2026-09-09.
pub const BASE_ARCHIVE_SHA256: &str =
    "210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74";
/// The Hebrew fine-tuned successor this lane binds. A successor inherits no receipt from its
/// base: this hash, and the census taken from this file, are its own evidence.
pub const HEBREW_SUCCESSOR_SHA256: &str =
    "585c40f214a53e8b5ba565c6176aa6cb548c7c8ab2f7ec5dc6dd45dbdff46ee4";
pub const HEBREW_SUCCESSOR_BYTES: u64 = 2_553_098_240;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RnntError {
    Archive(NemoError),
    MissingMember(&'static str),
    MissingTensor(String),
    UnexpectedTensor(String),
    Shape {
        name: String,
        expected: Vec<u64>,
        found: Vec<u64>,
    },
    Dtype {
        name: String,
        found: NemoDtype,
    },
    NonContiguous(String),
    AliasedStorage(String),
}

impl fmt::Display for RnntError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Archive(e) => write!(f, "{e}"),
            Self::MissingMember(m) => write!(f, "archive has no {m}"),
            Self::MissingTensor(n) => write!(f, "checkpoint is missing {n}"),
            Self::UnexpectedTensor(n) => write!(f, "checkpoint carries unexpected {n}"),
            Self::Shape {
                name,
                expected,
                found,
            } => write!(f, "{name}: expected shape {expected:?}, found {found:?}"),
            Self::Dtype { name, found } => write!(f, "{name}: unsupported storage {found:?}"),
            Self::NonContiguous(n) => write!(f, "{n} does not read its storage densely"),
            Self::AliasedStorage(k) => write!(f, "storage {k} is read by more than one tensor"),
        }
    }
}

impl std::error::Error for RnntError {}

/// Geometry the contract is generated from. Every field is a census-checked fact about the
/// pinned artifact, not a default carried over from a text config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RnntGeometry {
    pub mel_bins: u32,
    pub fft_bins: u32,
    pub window_length: u32,
    pub subsample_channels: u32,
    pub subsample_factor: u32,
    /// Frequency bins the three stride-2 subsampling stages leave. Measured from the pinned
    /// projection, 4352 = 256 x 17, not computed from a padding formula.
    pub subsample_output_bins: u32,
    pub encoder_layers: u32,
    pub encoder_width: u32,
    pub encoder_heads: u32,
    pub encoder_ffn: u32,
    pub conv_kernel: u32,
    pub predictor_width: u32,
    pub predictor_layers: u32,
    pub joint_width: u32,
    pub vocabulary: u32,
    pub prompt_slots: u32,
    pub prompt_hidden: u32,
    pub prompt_input: u32,
}

pub const HEBREW_GEOMETRY: RnntGeometry = RnntGeometry {
    mel_bins: 128,
    fft_bins: 257,
    window_length: 400,
    subsample_channels: 256,
    subsample_factor: 8,
    subsample_output_bins: 17,
    encoder_layers: 24,
    encoder_width: 1024,
    encoder_heads: 8,
    encoder_ffn: 4096,
    conv_kernel: 9,
    predictor_width: 640,
    predictor_layers: 2,
    joint_width: 640,
    vocabulary: 13088,
    prompt_slots: 128,
    prompt_hidden: 2048,
    prompt_input: 1152,
};

/// The streaming arm this lane qualifies. `[left, right]` are attention context chunks, so
/// `[56, 0]` is 56 chunks of left context and no future audio: nothing is decoded from samples
/// the session does not own yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionContext {
    pub left: u32,
    pub right: u32,
}

pub const QUALIFIED_CONTEXT: AttentionContext = AttentionContext { left: 56, right: 0 };
/// Arms the state schema must be able to express but that this lane does not admit: each one
/// buys accuracy with future audio, which is latency the product contract does not have.
pub const EXPRESSIBLE_CONTEXTS: &[AttentionContext] = &[
    AttentionContext { left: 56, right: 0 },
    AttentionContext { left: 56, right: 3 },
    AttentionContext { left: 56, right: 6 },
    AttentionContext {
        left: 56,
        right: 13,
    },
];

/// Per-session cache extents for one streaming step. Nothing here is allocated or executed:
/// it is the shape contract a cache-aware FastConformer encoder must honour, so the first
/// executable step cannot silently invent a cache layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingStateContract {
    pub context: AttentionContext,
    /// Attention history per layer, in encoder frames. The reference caches the normalized
    /// layer input and recomputes keys and values from it, so this is one tensor per layer and
    /// not a key and a value tensor: the captured cache is `[layers, batch, 56, width]`.
    pub last_channel_frames: u32,
    /// Causal depthwise convolution history per layer, in encoder frames.
    pub last_time_frames: u32,
    pub layers: u32,
    pub width: u32,
    /// Encoder frames a chunk emits after subsampling.
    pub chunk_frames: u32,
    /// Predictor hidden and cell state, per LSTM layer.
    pub predictor_state_width: u32,
    pub predictor_layers: u32,
    /// Mel frames the first step consumes. It is not a normal step: it carries no history and
    /// still produces one encoder frame.
    pub first_chunk_mel_frames: u32,
    /// Mel frames every later step consumes on top of the carry.
    pub chunk_mel_frames: u32,
    /// Mel frames of the previous chunk each later step re-reads so the subsampling stem sees
    /// the context its kernels need.
    pub pre_encode_carry_mel_frames: u32,
    /// Pre-encoded rows each later step drops, because the carry already produced them.
    pub drop_extra_pre_encoded: u32,
}

impl StreamingStateContract {
    /// Frames one 80 ms audio chunk becomes: 80 ms at hop 10 ms is 8 frontend frames, and the
    /// subsampler divides by 8.
    pub const CHUNK_MILLISECONDS: u32 = 80;

    pub fn new(geometry: RnntGeometry, context: AttentionContext) -> Result<Self, RnntError> {
        if !EXPRESSIBLE_CONTEXTS.contains(&context) {
            return Err(RnntError::UnexpectedTensor(format!(
                "attention context [{}, {}] is not an expressible arm",
                context.left, context.right
            )));
        }
        let chunk_frames = Self::CHUNK_MILLISECONDS / 10 / geometry.subsample_factor;
        Ok(Self {
            context,
            last_channel_frames: context.left,
            last_time_frames: geometry.conv_kernel - 1,
            layers: geometry.encoder_layers,
            width: geometry.encoder_width,
            chunk_frames: chunk_frames.max(1),
            predictor_state_width: geometry.predictor_width,
            predictor_layers: geometry.predictor_layers,
            // Measured from the reference's own streaming configuration for this arm:
            // chunk_size [1, 8], pre_encode_cache_size [0, 9], drop_extra_pre_encoded 2.
            first_chunk_mel_frames: 1,
            chunk_mel_frames: geometry.subsample_factor,
            pre_encode_carry_mel_frames: geometry.subsample_factor + 1,
            drop_extra_pre_encoded: 2,
        })
    }

    /// Elements one session holds between chunks, excluding the frontend carry.
    ///
    /// The attention term counts one row per cached frame, not two. An earlier version of this
    /// contract doubled it for a key and a value tensor; the pinned capture shows the reference
    /// caches the layer input alone, `[24, 1, 56, 1024]`.
    pub fn state_elements(&self) -> u64 {
        let attention =
            u64::from(self.layers) * u64::from(self.last_channel_frames) * u64::from(self.width);
        let convolution =
            u64::from(self.layers) * u64::from(self.last_time_frames) * u64::from(self.width);
        let predictor =
            u64::from(self.predictor_layers) * u64::from(self.predictor_state_width) * 2;
        attention + convolution + predictor
    }

    /// True when the arm never reads a sample the session does not own yet.
    pub fn is_causal(&self) -> bool {
        self.context.right == 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RnntTensorRequirement {
    pub name: String,
    pub shape: Vec<u64>,
}

/// Every tensor the pinned architecture must carry, in checkpoint order.
pub fn tensor_contract(geometry: RnntGeometry) -> Vec<RnntTensorRequirement> {
    let width = u64::from(geometry.encoder_width);
    let ffn = u64::from(geometry.encoder_ffn);
    let channels = u64::from(geometry.subsample_channels);
    let head = width / u64::from(geometry.encoder_heads);
    let predictor = u64::from(geometry.predictor_width);
    let joint = u64::from(geometry.joint_width);
    let vocabulary = u64::from(geometry.vocabulary);
    let mut want = Vec::new();
    let mut add = |name: String, shape: Vec<u64>| {
        want.push(RnntTensorRequirement { name, shape });
    };
    add(
        "preprocessor.featurizer.window".into(),
        vec![u64::from(geometry.window_length)],
    );
    add(
        "preprocessor.featurizer.fb".into(),
        vec![
            1,
            u64::from(geometry.mel_bins),
            u64::from(geometry.fft_bins),
        ],
    );
    // Subsampling projects the flattened frequency-channel field, not the raw mel field.
    add(
        "encoder.pre_encode.out.weight".into(),
        vec![width, channels * u64::from(geometry.subsample_output_bins)],
    );
    add("encoder.pre_encode.out.bias".into(), vec![width]);
    for (index, depthwise) in [(0u32, true), (2, true), (3, false), (5, true), (6, false)] {
        let shape = if depthwise {
            vec![channels, 1, 3, 3]
        } else {
            vec![channels, channels, 1, 1]
        };
        add(format!("encoder.pre_encode.conv.{index}.weight"), shape);
        add(
            format!("encoder.pre_encode.conv.{index}.bias"),
            vec![channels],
        );
    }
    for layer in 0..geometry.encoder_layers {
        let p = format!("encoder.layers.{layer}");
        for norm in [
            "norm_feed_forward1",
            "norm_conv",
            "norm_self_att",
            "norm_feed_forward2",
            "norm_out",
        ] {
            add(format!("{p}.{norm}.weight"), vec![width]);
            add(format!("{p}.{norm}.bias"), vec![width]);
        }
        for feed_forward in ["feed_forward1", "feed_forward2"] {
            add(
                format!("{p}.{feed_forward}.linear1.weight"),
                vec![ffn, width],
            );
            add(
                format!("{p}.{feed_forward}.linear2.weight"),
                vec![width, ffn],
            );
        }
        add(
            format!("{p}.conv.pointwise_conv1.weight"),
            vec![2 * width, width, 1],
        );
        add(
            format!("{p}.conv.depthwise_conv.weight"),
            vec![width, 1, u64::from(geometry.conv_kernel)],
        );
        // Named batch_norm in the checkpoint; the config declares LayerNorm.
        add(format!("{p}.conv.batch_norm.weight"), vec![width]);
        add(format!("{p}.conv.batch_norm.bias"), vec![width]);
        add(
            format!("{p}.conv.pointwise_conv2.weight"),
            vec![width, width, 1],
        );
        for bias in ["pos_bias_u", "pos_bias_v"] {
            add(
                format!("{p}.self_attn.{bias}"),
                vec![u64::from(geometry.encoder_heads), head],
            );
        }
        for projection in [
            "linear_q",
            "linear_k",
            "linear_v",
            "linear_out",
            "linear_pos",
        ] {
            add(
                format!("{p}.self_attn.{projection}.weight"),
                vec![width, width],
            );
        }
    }
    add(
        "decoder.prediction.embed.weight".into(),
        vec![vocabulary, predictor],
    );
    for layer in 0..geometry.predictor_layers {
        for gate in ["weight_ih", "weight_hh"] {
            add(
                format!("decoder.prediction.dec_rnn.lstm.{gate}_l{layer}"),
                vec![4 * predictor, predictor],
            );
        }
        for gate in ["bias_ih", "bias_hh"] {
            add(
                format!("decoder.prediction.dec_rnn.lstm.{gate}_l{layer}"),
                vec![4 * predictor],
            );
        }
    }
    add("joint.pred.weight".into(), vec![joint, predictor]);
    add("joint.pred.bias".into(), vec![joint]);
    add("joint.enc.weight".into(), vec![joint, width]);
    add("joint.enc.bias".into(), vec![joint]);
    add("joint.joint_net.2.weight".into(), vec![vocabulary, joint]);
    add("joint.joint_net.2.bias".into(), vec![vocabulary]);
    add(
        "prompt_kernel.0.weight".into(),
        vec![
            u64::from(geometry.prompt_hidden),
            u64::from(geometry.prompt_input),
        ],
    );
    add(
        "prompt_kernel.0.bias".into(),
        vec![u64::from(geometry.prompt_hidden)],
    );
    add(
        "prompt_kernel.2.weight".into(),
        vec![width, u64::from(geometry.prompt_hidden)],
    );
    add("prompt_kernel.2.bias".into(), vec![width]);
    want
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundRnnt {
    /// Contract name to the checkpoint tensor that satisfied it.
    pub tensors: BTreeMap<String, NemoTensor>,
    pub elements: u64,
    pub payload_bytes: u64,
}

/// Bind a census against the contract: nothing missing, nothing extra, every shape and dtype
/// exact, every tensor dense in its own storage.
pub fn bind(geometry: RnntGeometry, census: &[NemoTensor]) -> Result<BoundRnnt, RnntError> {
    let aliased = shared_storage_groups(census);
    if let Some(key) = aliased.keys().next() {
        return Err(RnntError::AliasedStorage(key.clone()));
    }
    let mut by_name: BTreeMap<&str, &NemoTensor> = BTreeMap::new();
    for tensor in census {
        by_name.insert(tensor.name.as_str(), tensor);
    }
    let mut bound = BTreeMap::new();
    let mut elements = 0u64;
    let mut payload_bytes = 0u64;
    for requirement in tensor_contract(geometry) {
        let found = by_name
            .remove(requirement.name.as_str())
            .ok_or_else(|| RnntError::MissingTensor(requirement.name.clone()))?;
        if found.shape != requirement.shape {
            return Err(RnntError::Shape {
                name: requirement.name,
                expected: requirement.shape,
                found: found.shape.clone(),
            });
        }
        if found.dtype != NemoDtype::F32 {
            return Err(RnntError::Dtype {
                name: requirement.name,
                found: found.dtype,
            });
        }
        if !found.is_contiguous() {
            return Err(RnntError::NonContiguous(requirement.name));
        }
        elements += found.elements();
        payload_bytes += found.elements() * found.dtype.bytes();
        bound.insert(requirement.name, found.clone());
    }
    if let Some(extra) = by_name.keys().next() {
        return Err(RnntError::UnexpectedTensor((*extra).to_owned()));
    }
    Ok(BoundRnnt {
        tensors: bound,
        elements,
        payload_bytes,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NemoCheckpoint {
    /// Config, tokenizer and vocabulary members, by archive name.
    pub members: Vec<(String, usize, usize)>,
    pub config_yaml: (usize, usize),
    pub census: Vec<NemoTensor>,
    /// Zip member name to (offset within the archive, byte length) for every storage.
    pub storages: BTreeMap<String, (usize, usize)>,
}

/// Read an archive's layout and census without materializing one tensor.
///
/// Offsets are absolute in the archive, so a later loader can map the file once and slice.
pub fn read_archive(archive: &[u8]) -> Result<NemoCheckpoint, RnntError> {
    let members = tar_members(archive).map_err(RnntError::Archive)?;
    let find = |suffix: &str| {
        members
            .iter()
            .find(|m| m.name.ends_with(suffix))
            .map(|m| (m.offset, m.size))
    };
    let config_yaml =
        find("model_config.yaml").ok_or(RnntError::MissingMember("model_config.yaml"))?;
    let (weights_at, weights_len) =
        find("model_weights.ckpt").ok_or(RnntError::MissingMember("model_weights.ckpt"))?;
    let zip = &archive[weights_at..weights_at + weights_len];
    let entries = zip_entries(zip).map_err(RnntError::Archive)?;
    let (pickle_at, pickle_len) = entries
        .iter()
        .find(|(name, _)| name.ends_with("/data.pkl"))
        .map(|(_, extent)| *extent)
        .ok_or(RnntError::MissingMember("data.pkl"))?;
    let census =
        pickle_census(&zip[pickle_at..pickle_at + pickle_len]).map_err(RnntError::Archive)?;
    let storages = entries
        .iter()
        .filter(|(name, _)| name.contains("/data/"))
        .map(|(name, (at, len))| {
            (
                name.rsplit('/').next().unwrap_or(name).to_owned(),
                (weights_at + at, *len),
            )
        })
        .collect();
    Ok(NemoCheckpoint {
        members: members
            .into_iter()
            .map(|m| (m.name, m.offset, m.size))
            .collect(),
        config_yaml,
        census,
        storages,
    })
}

/// A memory-mapped archive with its census already taken.
///
/// The map is held for the lifetime of the value so a loader can slice tensor payloads out of
/// it without a 2.5 GB copy.
pub struct MappedNemo {
    map: memmap2::Mmap,
    pub checkpoint: NemoCheckpoint,
}

impl MappedNemo {
    pub fn open(path: &std::path::Path) -> Result<Self, RnntError> {
        let file = std::fs::File::open(path)
            .map_err(|e| RnntError::Archive(NemoError::Archive(e.to_string())))?;
        // SAFETY: the archive is a read-only checkpoint; a concurrent writer would be a
        // deployment error, not a case this reader tries to survive.
        let map = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| RnntError::Archive(NemoError::Archive(e.to_string())))?;
        let checkpoint = read_archive(&map)?;
        Ok(Self { map, checkpoint })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.map
    }
}

#[cfg(test)]
mod tests;
