//! Speech semantics, independent of a kernel backend. No executor is admitted by this schema.

use super::{ModelPlan, NormKind, NormPlan, OperationKind, WeightTransform};

#[derive(Debug, Clone, PartialEq)]
pub struct WhisperPlan {
    pub frontend: LogMelPlan,
    pub conv_stem: [AudioConv1dPlan; 2],
    pub hidden_size: u32,
    pub encoder_layers: u32,
    pub decoder_layers: u32,
    pub encoder_heads: u32,
    pub decoder_heads: u32,
    pub encoder_ffn: u32,
    pub decoder_ffn: u32,
    pub source_positions: u32,
    pub target_positions: u32,
    pub vocab_size: u32,
    pub norm: NormPlan,
    pub activation: SpeechActivation,
    pub encoder_position: AudioPositionKind,
    pub decoder_position: AudioPositionKind,
    pub attention: WhisperAttentionPlan,
    pub state: WhisperStatePlan,
    pub decode: AsrDecodePlan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechActivation {
    GeluErf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioPositionKind {
    CheckpointSinusoidal,
    LearnedAbsolute,
}

/// Mono f32 PCM at the declared rate; resampling is a separately qualified input operation.
#[derive(Debug, Clone, PartialEq)]
pub struct LogMelPlan {
    pub sample_rate: u32,
    pub fft_size: u32,
    pub hop_length: u32,
    pub mel_bins: u32,
    pub max_samples: u32,
    pub max_frames: u32,
    pub periodic_hann: bool,
    pub centered_reflect_padding: bool,
    pub drop_last_stft_frame: bool,
    pub slaney_mel_filters: bool,
    pub power: u32,
    pub log10_floor: f32,
    pub dynamic_range: f32,
    pub normalization_offset: f32,
    pub normalization_divisor: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioConv1dPlan {
    pub input_channels: u32,
    pub output_channels: u32,
    pub kernel: u32,
    pub stride: u32,
    pub padding: u32,
    pub bias: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WhisperAttentionPlan {
    pub encoder_bidirectional: bool,
    pub decoder_causal: bool,
    pub cross_attention: bool,
    pub query_bias: bool,
    pub key_bias: bool,
    pub value_bias: bool,
    pub output_bias: bool,
    pub qk_scale: f32,
    pub pre_norm_serial_residual: bool,
    pub tied_output_embedding: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhisperStatePlan {
    /// Recompute the bidirectional encoder for each complete bounded waveform.
    pub encoder_recomputed_per_utterance: bool,
    pub decoder_self_kv_per_layer: bool,
    /// Immutable for this waveform; never carry across utterances or sessions.
    pub encoder_cross_kv_per_decoder_layer: bool,
    pub max_decoder_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsrDecodePlan {
    pub beam_size: u32,
    pub deterministic: bool,
    pub start_token: u32,
    pub language_token: u32,
    pub task_token: u32,
    pub no_timestamps_token: u32,
    pub eos_token: u32,
    /// Decode policy, suppression and timestamp handling still need oracle qualification.
    pub generation_policy_qualified: bool,
}

impl WhisperPlan {
    pub fn operations(&self) -> Vec<OperationKind> {
        vec![
            OperationKind::AudioLogMel,
            OperationKind::AudioStridedConv,
            OperationKind::AudioPositionEmbedding,
            OperationKind::BiasedLayerNorm,
            OperationKind::GeluErfActivation,
            OperationKind::AudioEncoderAttention,
            OperationKind::AudioDecoderSelfAttention,
            OperationKind::AudioCrossAttention,
            OperationKind::AudioCrossKvState,
            OperationKind::AsrDeterministicDecode,
            OperationKind::DenseMlp,
            OperationKind::KvState,
            OperationKind::OutputProjection,
        ]
    }

    pub fn into_model_plan(self) -> ModelPlan {
        ModelPlan {
            arch: crate::config::Arch::Other("whisper".into()),
            hidden_size: self.hidden_size,
            vocab_size: self.vocab_size,
            context_length: self.target_positions,
            embedding_scale: 1.0,
            output_norm: NormPlan {
                kind: NormKind::LayerNorm,
                epsilon: 1e-5,
                weight_transform: WeightTransform::Identity,
            },
            speech: Some(self),
            vision: None,
            multimodal: None,
            layers: Vec::new(),
            exit_mixer: None,
            logits: Vec::new(),
            mtp_blocks: Vec::new(),
            drafter: None,
            draft_source: super::DraftSourcePlan::None,
            sampling_defaults: None,
            partition_boundaries: Vec::new(),
        }
    }
}
