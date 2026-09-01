//! Portable, deliberately unfused executor for canonical `ModelPlan` operations.
//!
//! This crate is a correctness oracle, not a serving backend. It has no CUDA dependency and no
//! external engine fallback. Unsupported canonical operations return a named error.

pub mod hidden_trace;

use memra_gguf::config::AttentionGateKind;
use memra_gguf::model_plan::{
    ActivationPlan, AttentionPlan, AttentionScale, GemmaLayerScale, HcCollapse, LogitsTransform,
    MlpPlan, ModelPlan, ResidualTopology, ValueNorm, ValueProjection,
};
use memra_gguf::tensor_contract::{DsparkTensor, LayerTensor, MtpTensor, TensorId, VisionTensor};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceTensor {
    /// Logical row-major shape, outermost dimension first.
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl ReferenceTensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, ReferenceError> {
        let expected = shape.iter().product();
        if data.len() != expected {
            return Err(ReferenceError::TensorShape {
                id: None,
                expected: shape,
                actual_elements: data.len(),
            });
        }
        Ok(Self { shape, data })
    }
}

pub type ReferenceWeights = BTreeMap<TensorId, ReferenceTensor>;

#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceFixture {
    pub token_ids: Vec<u32>,
    pub weights: ReferenceWeights,
    pub vision: Option<ReferenceVisionInput>,
    pub multimodal_token_ids: Option<Vec<u32>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceVisionInput {
    /// Raw patch pixels in `[0, 1]`, row-major `[patches, 3 * patch_size^2]`.
    pub patches: ReferenceTensor,
    pub positions: Vec<[u32; 2]>,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceVisionOutput {
    pub encoder_hidden: Vec<f32>,
    pub pooled_hidden: Vec<f32>,
    pub projected_hidden: Vec<f32>,
    pub patch_count: usize,
    pub output_tokens: usize,
    pub hidden_size: usize,
    pub projection_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceMultimodalOutput {
    pub language: ReferenceOutput,
    pub vision: ReferenceVisionOutput,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceState {
    pub layers: Vec<ReferenceLayerState>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReferenceLayerState {
    Kv {
        key: Vec<f32>,
        value: Vec<f32>,
        tokens: usize,
        kv_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        window: Option<usize>,
    },
    Recurrent {
        conv: Vec<f32>,
        matrix: Vec<f32>,
        value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        conv_width: usize,
    },
    LatentKv {
        rows: Vec<f32>,
        tokens: usize,
        width: usize,
    },
    CompressedAttention {
        rows: Vec<f32>,
        tokens: usize,
        width: usize,
        window: usize,
        compressed_tokens: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceOutput {
    /// `[tokens, vocab]`, row-major.
    pub logits: Vec<f32>,
    pub tokens: usize,
    pub vocab: usize,
    pub state: ReferenceState,
    pub mtp: Vec<ReferenceMtpOutput>,
    pub draft: Option<ReferenceDraftOutput>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceMtpOutput {
    pub depth: u32,
    pub logits: Vec<f32>,
    pub hidden: Vec<f32>,
    pub state: ReferenceLayerState,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceDraftOutput {
    pub input_token: u32,
    pub output_ids: Vec<u32>,
    pub confidence: Vec<f32>,
    pub logits: Vec<f32>,
    pub hidden: Vec<f32>,
    pub block_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReferenceError {
    EmptyInput,
    TokenOutOfRange {
        token: u32,
        vocab: usize,
    },
    MissingTensor(TensorId),
    TensorShape {
        id: Option<TensorId>,
        expected: Vec<usize>,
        actual_elements: usize,
    },
    UnsupportedOperation {
        layer: Option<u32>,
        operation: &'static str,
    },
    InvalidPlan {
        layer: Option<u32>,
        reason: &'static str,
    },
}

impl std::fmt::Display for ReferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "reference executor requires at least one token"),
            Self::TokenOutOfRange { token, vocab } => {
                write!(f, "token {token} is outside vocabulary size {vocab}")
            }
            Self::MissingTensor(id) => write!(f, "missing reference tensor {id:?}"),
            Self::TensorShape {
                id,
                expected,
                actual_elements,
            } => write!(
                f,
                "reference tensor {id:?} expected shape {expected:?}, got {actual_elements} elements"
            ),
            Self::UnsupportedOperation { layer, operation } => {
                write!(
                    f,
                    "unsupported reference operation {operation} at layer {layer:?}"
                )
            }
            Self::InvalidPlan { layer, reason } => {
                write!(f, "invalid model plan at layer {layer:?}: {reason}")
            }
        }
    }
}

impl std::error::Error for ReferenceError {}

pub fn deterministic_fixture(plan: &ModelPlan) -> Result<ReferenceFixture, ReferenceError> {
    let hidden = plan.hidden_size as usize;
    let vocab = plan.vocab_size as usize;
    if hidden == 0 || vocab < 2 || hidden > 256 || vocab > 262_144 {
        return Err(ReferenceError::InvalidPlan {
            layer: None,
            reason: "reference fixture requires hidden<=256 and 2<=vocab<=262144",
        });
    }
    let mut executable_layers: Vec<_> = plan
        .layers
        .iter()
        .chain(plan.mtp_blocks.iter().map(|block| &block.layer))
        .collect();
    if let Some(memra_gguf::model_plan::DrafterPlan::Dspark(dspark)) = plan.drafter.as_ref() {
        executable_layers.extend(dspark.blocks.iter());
    }
    let mut weights = ReferenceWeights::new();
    weights.insert(
        TensorId::TokenEmbedding,
        generated_tensor(&[vocab, hidden], 1, 0.2)?,
    );
    let vision = if let Some(vision) = plan.vision.as_ref() {
        Some(add_vision_fixture(
            &mut weights,
            vision,
            plan.multimodal.map(|injection| injection.tokens_per_image),
        )?)
    } else {
        None
    };
    weights.insert(
        TensorId::OutputNorm,
        ReferenceTensor::new(vec![hidden], vec![1.0; hidden])?,
    );
    let checkpoint_factor_width = executable_layers
        .iter()
        .copied()
        .filter_map(|layer| match &layer.attention {
            AttentionPlan::Full(attention) | AttentionPlan::SlidingWindow { attention, .. } => {
                matches!(
                    attention.rope.factors,
                    memra_gguf::model_plan::RopeFactors::Checkpoint
                )
                .then_some(attention.rope.dimensions as usize / 2)
            }
            _ => None,
        })
        .max();
    if let Some(width) = checkpoint_factor_width {
        weights.insert(
            TensorId::RopeFactors,
            ReferenceTensor::new(vec![width], vec![1.0; width])?,
        );
    }
    if let Some((streams, epsilon, sinkhorn_iterations, collapse)) = hyper_topology(plan)? {
        // The mean collapse has no learned head tensors (Glm5NextTextHyperHead).
        if collapse == HcCollapse::GatedHead {
            add_hyper_head_fixture(&mut weights, streams, hidden)?;
        }
        if epsilon <= 0.0 || sinkhorn_iterations == 0 {
            return Err(ReferenceError::InvalidPlan {
                layer: None,
                reason: "HyperConnections require positive epsilon and Sinkhorn iterations",
            });
        }
    }
    for layer in executable_layers {
        match layer.residual {
            ResidualTopology::Serial => {}
            ResidualTopology::Gemma { parallel_moe, .. } => {
                for tensor in [LayerTensor::PostAttentionNorm, LayerTensor::PostMlpNorm] {
                    weights.insert(
                        layer_id(layer.index, tensor),
                        ReferenceTensor::new(vec![hidden], vec![1.0; hidden])?,
                    );
                }
                weights.insert(
                    layer_id(layer.index, LayerTensor::LayerScale),
                    ReferenceTensor::new(vec![1], vec![0.9])?,
                );
                if parallel_moe.is_some() {
                    for tensor in [
                        LayerTensor::PostSharedMlpNorm,
                        LayerTensor::PreRoutedMlpNorm,
                        LayerTensor::PostRoutedMlpNorm,
                    ] {
                        weights.insert(
                            layer_id(layer.index, tensor),
                            ReferenceTensor::new(vec![hidden], vec![1.0; hidden])?,
                        );
                    }
                }
            }
            ResidualTopology::HyperConnections { streams, .. } => {
                add_hyper_fixture(&mut weights, layer.index, streams as usize, hidden)?;
            }
        }
        for tensor in [LayerTensor::PreAttentionNorm, LayerTensor::PreMlpNorm] {
            weights.insert(
                layer_id(layer.index, tensor),
                ReferenceTensor::new(vec![hidden], vec![1.0; hidden])?,
            );
        }
        match &layer.attention {
            AttentionPlan::Full(attention) | AttentionPlan::SlidingWindow { attention, .. } => {
                add_full_attention_fixture(&mut weights, layer.index, attention, hidden)?;
            }
            AttentionPlan::GatedDeltaNet(gdn) => {
                add_gdn_fixture(&mut weights, layer.index, gdn, hidden)?;
            }
            AttentionPlan::Mla(mla) => {
                add_mla_fixture(&mut weights, layer.index, mla, hidden)?;
            }
            AttentionPlan::KimiDeltaNet(kda) => {
                add_kda_fixture(&mut weights, layer.index, kda, hidden)?;
            }
        }
        match &layer.mlp {
            MlpPlan::Dense(mlp) => {
                add_dense_mlp_fixture(&mut weights, layer.index, mlp, hidden)?;
            }
            MlpPlan::Moe(moe) => {
                add_moe_fixture(&mut weights, layer.index, moe, hidden, vocab)?;
                if matches!(
                    layer.residual,
                    ResidualTopology::Gemma {
                        parallel_moe: Some(_),
                        ..
                    }
                ) {
                    add_gemma_parallel_moe_fixture(&mut weights, layer.index, moe, hidden)?;
                }
            }
        }
    }
    for block in &plan.mtp_blocks {
        for tensor in [MtpTensor::EmbeddingNorm, MtpTensor::HiddenNorm] {
            weights.insert(
                TensorId::Mtp {
                    depth: block.depth,
                    tensor,
                },
                ReferenceTensor::new(vec![hidden], vec![1.0; hidden])?,
            );
        }
        weights.insert(
            TensorId::Mtp {
                depth: block.depth,
                tensor: MtpTensor::FusionProjection,
            },
            generated_tensor(
                &[hidden, 2 * hidden],
                100 + block.depth as u64,
                1.0 / ((2 * hidden) as f32).sqrt(),
            )?,
        );
    }
    if let Some(memra_gguf::model_plan::DrafterPlan::Dspark(dspark)) = plan.drafter.as_ref() {
        add_dspark_fixture(&mut weights, dspark, hidden, vocab)?;
    }
    let token_ids = (1..=3.min(vocab - 1)).map(|token| token as u32).collect();
    let multimodal_token_ids = plan.multimodal.map(|injection| {
        let mut tokens = Vec::with_capacity(injection.tokens_per_image as usize + 2);
        tokens.push(1);
        tokens.extend(std::iter::repeat_n(
            injection.placeholder_token_id,
            injection.tokens_per_image as usize,
        ));
        tokens.push(if injection.placeholder_token_id == 2 {
            3
        } else {
            2
        });
        tokens
    });
    Ok(ReferenceFixture {
        token_ids,
        weights,
        vision,
        multimodal_token_ids,
    })
}

fn add_dspark_fixture(
    weights: &mut ReferenceWeights,
    plan: &memra_gguf::model_plan::DsparkPlan,
    hidden: usize,
    vocab: usize,
) -> Result<(), ReferenceError> {
    if plan.blocks.is_empty()
        || plan.block_size == 0
        || plan.markov_rank == 0
        || plan.target_layer_ids.is_empty()
        || plan.noise_token_id as usize >= vocab
    {
        return Err(ReferenceError::InvalidPlan {
            layer: None,
            reason: "DSpark fixture requires blocks, targets, rank, block size, and valid noise token",
        });
    }
    let streams = match plan.blocks[0].residual {
        ResidualTopology::HyperConnections { streams, .. } if streams > 0 => streams as usize,
        _ => {
            return Err(ReferenceError::InvalidPlan {
                layer: Some(plan.blocks[0].index),
                reason: "DSpark blocks require HyperConnections",
            });
        }
    };
    weights.insert(
        TensorId::Dspark(DsparkTensor::MainProjection),
        generated_tensor(
            &[hidden, plan.target_layer_ids.len() * hidden],
            140,
            1.0 / ((plan.target_layer_ids.len() * hidden) as f32).sqrt(),
        )?,
    );
    weights.insert(
        TensorId::Dspark(DsparkTensor::MainNorm),
        ReferenceTensor::new(vec![hidden], vec![1.0; hidden])?,
    );
    weights.insert(
        TensorId::Dspark(DsparkTensor::OutputNorm),
        ReferenceTensor::new(vec![hidden], vec![1.0; hidden])?,
    );
    let rank = plan.markov_rank as usize;
    weights.insert(
        TensorId::Dspark(DsparkTensor::MarkovEmbedding),
        generated_tensor(&[vocab, rank], 141, 0.1)?,
    );
    weights.insert(
        TensorId::Dspark(DsparkTensor::MarkovOutput),
        generated_tensor(&[vocab, rank], 142, 0.1)?,
    );
    weights.insert(
        TensorId::Dspark(DsparkTensor::ConfidenceProjection),
        generated_tensor(&[1, hidden + rank], 143, 0.1)?,
    );
    weights.insert(
        TensorId::Dspark(DsparkTensor::HeadHyperFunction),
        generated_tensor(&[streams, streams * hidden], 144, 0.1)?,
    );
    weights.insert(
        TensorId::Dspark(DsparkTensor::HeadHyperBase),
        generated_tensor(&[streams], 145, 0.05)?,
    );
    weights.insert(
        TensorId::Dspark(DsparkTensor::HeadHyperScale),
        ReferenceTensor::new(vec![1], vec![0.2])?,
    );
    Ok(())
}

fn add_vision_fixture(
    weights: &mut ReferenceWeights,
    plan: &memra_gguf::model_plan::VisionEncoderPlan,
    output_tokens: Option<u32>,
) -> Result<ReferenceVisionInput, ReferenceError> {
    let hidden = plan.hidden_size as usize;
    let patch_width =
        (plan.patch.channels * plan.patch.patch_size * plan.patch.patch_size) as usize;
    let axes = plan.patch.position_axes as usize;
    let positions = plan.patch.position_embedding_size as usize;
    weights.insert(
        TensorId::Vision {
            layer: None,
            tensor: VisionTensor::PatchProjection,
        },
        generated_tensor(
            &[hidden, patch_width],
            150,
            1.0 / (patch_width as f32).sqrt(),
        )?,
    );
    weights.insert(
        TensorId::Vision {
            layer: None,
            tensor: VisionTensor::PositionEmbedding,
        },
        generated_tensor(&[axes, positions, hidden], 151, 0.05)?,
    );
    if plan.standardize {
        weights.insert(
            TensorId::Vision {
                layer: None,
                tensor: VisionTensor::StandardizeBias,
            },
            generated_tensor(&[hidden], 152, 0.05)?,
        );
        weights.insert(
            TensorId::Vision {
                layer: None,
                tensor: VisionTensor::StandardizeScale,
            },
            ReferenceTensor::new(vec![hidden], vec![0.5; hidden])?,
        );
    }
    weights.insert(
        TensorId::Vision {
            layer: None,
            tensor: VisionTensor::OutputProjection,
        },
        generated_tensor(
            &[plan.projection_output_size as usize, hidden],
            153,
            1.0 / (hidden as f32).sqrt(),
        )?,
    );
    for layer in &plan.layers {
        let layer_id = Some(layer.index);
        for tensor in [
            VisionTensor::InputNorm,
            VisionTensor::PostAttentionNorm,
            VisionTensor::PreMlpNorm,
            VisionTensor::PostMlpNorm,
        ] {
            weights.insert(
                TensorId::Vision {
                    layer: layer_id,
                    tensor,
                },
                ReferenceTensor::new(vec![hidden], vec![1.0; hidden])?,
            );
        }
        let query_width = (layer.attention.query_heads * layer.attention.head_dim) as usize;
        let kv_width = (layer.attention.kv_heads * layer.attention.head_dim) as usize;
        for (tensor, shape, input, salt) in [
            (VisionTensor::Query, vec![query_width, hidden], hidden, 160),
            (VisionTensor::Key, vec![kv_width, hidden], hidden, 161),
            (VisionTensor::Value, vec![kv_width, hidden], hidden, 162),
            (
                VisionTensor::AttentionOutput,
                vec![hidden, query_width],
                query_width,
                163,
            ),
            (
                VisionTensor::MlpGate,
                vec![layer.mlp.intermediate_size as usize, hidden],
                hidden,
                164,
            ),
            (
                VisionTensor::MlpUp,
                vec![layer.mlp.intermediate_size as usize, hidden],
                hidden,
                165,
            ),
            (
                VisionTensor::MlpDown,
                vec![hidden, layer.mlp.intermediate_size as usize],
                layer.mlp.intermediate_size as usize,
                166,
            ),
        ] {
            weights.insert(
                TensorId::Vision {
                    layer: layer_id,
                    tensor,
                },
                generated_tensor(
                    &shape,
                    salt + layer.index as u64 * 17,
                    1.0 / (input as f32).sqrt(),
                )?,
            );
        }
        for tensor in [VisionTensor::QueryNorm, VisionTensor::KeyNorm] {
            weights.insert(
                TensorId::Vision {
                    layer: layer_id,
                    tensor,
                },
                ReferenceTensor::new(
                    vec![layer.attention.head_dim as usize],
                    vec![1.0; layer.attention.head_dim as usize],
                )?,
            );
        }
    }
    let side = plan.pooling_kernel_size.max(1) as usize;
    let output_tokens = output_tokens.unwrap_or(1) as usize;
    let patch_count = side * side * output_tokens;
    let mut patches = generated_tensor(&[patch_count, patch_width], 170, 0.5)?;
    for value in &mut patches.data {
        *value += 0.5;
    }
    let mut patch_positions = Vec::with_capacity(patch_count);
    for y in 0..side {
        for x in 0..side * output_tokens {
            patch_positions.push([x as u32, y as u32]);
        }
    }
    Ok(ReferenceVisionInput {
        patches,
        positions: patch_positions,
        output_tokens,
    })
}

fn add_hyper_head_fixture(
    weights: &mut ReferenceWeights,
    streams: usize,
    hidden: usize,
) -> Result<(), ReferenceError> {
    if streams == 0 {
        return Err(ReferenceError::InvalidPlan {
            layer: None,
            reason: "HyperConnections require at least one stream",
        });
    }
    weights.insert(
        TensorId::HyperHeadFunction,
        generated_tensor(&[streams, streams * hidden], 90, 0.1)?,
    );
    weights.insert(
        TensorId::HyperHeadBase,
        generated_tensor(&[streams], 91, 0.05)?,
    );
    weights.insert(
        TensorId::HyperHeadScale,
        ReferenceTensor::new(vec![1], vec![0.2])?,
    );
    Ok(())
}

fn add_hyper_fixture(
    weights: &mut ReferenceWeights,
    layer: u32,
    streams: usize,
    hidden: usize,
) -> Result<(), ReferenceError> {
    if streams == 0 {
        return Err(ReferenceError::InvalidPlan {
            layer: Some(layer),
            reason: "HyperConnections require at least one stream",
        });
    }
    let rows = (2 + streams) * streams;
    for (function, base, scale, salt) in [
        (
            LayerTensor::HyperAttentionFunction,
            LayerTensor::HyperAttentionBase,
            LayerTensor::HyperAttentionScale,
            92,
        ),
        (
            LayerTensor::HyperMlpFunction,
            LayerTensor::HyperMlpBase,
            LayerTensor::HyperMlpScale,
            95,
        ),
    ] {
        weights.insert(
            layer_id(layer, function),
            generated_tensor(&[rows, streams * hidden], salt + layer as u64 * 101, 0.1)?,
        );
        weights.insert(
            layer_id(layer, base),
            generated_tensor(&[rows], salt + 1 + layer as u64 * 101, 0.05)?,
        );
        weights.insert(
            layer_id(layer, scale),
            ReferenceTensor::new(vec![3], vec![0.2, 0.2, 0.2])?,
        );
    }
    Ok(())
}

#[allow(clippy::unusual_byte_groupings)] // allow: mnemonic grouping of a pinned seed/magic constant
fn generated_tensor(
    shape: &[usize],
    salt: u64,
    scale: f32,
) -> Result<ReferenceTensor, ReferenceError> {
    let elements = shape.iter().product();
    let data = (0..elements)
        .map(|index| {
            let mut value = index as u64 ^ salt.wrapping_mul(0x9e37_79b9);
            value ^= value >> 16;
            value = value.wrapping_mul(0x45d9_f3b);
            value ^= value >> 16;
            let unit = (value as u32) as f32 / u32::MAX as f32;
            (2.0 * unit - 1.0) * scale
        })
        .collect();
    ReferenceTensor::new(shape.to_vec(), data)
}

fn add_full_attention_fixture(
    weights: &mut ReferenceWeights,
    layer: u32,
    attention: &memra_gguf::model_plan::FullAttentionPlan,
    hidden: usize,
) -> Result<(), ReferenceError> {
    let query_heads = attention.query_heads as usize;
    let kv_heads = attention.kv_heads as usize;
    let key_dim = attention.key_head_dim as usize;
    let value_dim = attention.value_head_dim as usize;
    let q_width = query_heads
        * key_dim
        * if attention.output_gate == AttentionGateKind::FusedQ {
            2
        } else {
            1
        };
    for (tensor, output, input, salt) in [
        (LayerTensor::Query, q_width, hidden, 10),
        (LayerTensor::Key, kv_heads * key_dim, hidden, 11),
        (
            LayerTensor::AttentionOutput,
            hidden,
            query_heads * value_dim,
            13,
        ),
    ] {
        weights.insert(
            layer_id(layer, tensor),
            generated_tensor(
                &[output, input],
                salt + layer as u64 * 31,
                1.0 / (input as f32).sqrt(),
            )?,
        );
    }
    if attention.value_projection == ValueProjection::Separate {
        weights.insert(
            layer_id(layer, LayerTensor::Value),
            generated_tensor(
                &[kv_heads * value_dim, hidden],
                12 + layer as u64 * 31,
                1.0 / (hidden as f32).sqrt(),
            )?,
        );
    }
    if attention.qk_norm != memra_gguf::model_plan::TensorPresence::Absent {
        for tensor in [LayerTensor::QueryNorm, LayerTensor::KeyNorm] {
            weights.insert(
                layer_id(layer, tensor),
                ReferenceTensor::new(vec![key_dim], vec![1.0; key_dim])?,
            );
        }
    }
    if attention.output_gate == AttentionGateKind::SeparateHead {
        weights.insert(
            layer_id(layer, LayerTensor::AttentionGate),
            generated_tensor(
                &[query_heads, hidden],
                14 + layer as u64 * 31,
                1.0 / (hidden as f32).sqrt(),
            )?,
        );
    }
    Ok(())
}

fn add_gdn_fixture(
    weights: &mut ReferenceWeights,
    layer: u32,
    gdn: &memra_gguf::model_plan::GatedDeltaNetPlan,
    hidden: usize,
) -> Result<(), ReferenceError> {
    let key_heads = gdn.key_heads as usize;
    let value_heads = gdn.value_heads as usize;
    let key_dim = gdn.key_head_dim as usize;
    let value_dim = gdn.value_head_dim as usize;
    let conv_width = 2 * key_heads * key_dim + value_heads * value_dim;
    for (tensor, output, input, salt) in [
        (LayerTensor::GdnQkv, conv_width, hidden, 40),
        (LayerTensor::GdnGate, value_heads * value_dim, hidden, 41),
        (LayerTensor::GdnBeta, value_heads, hidden, 42),
        (LayerTensor::GdnAlpha, value_heads, hidden, 43),
        (LayerTensor::GdnOutput, hidden, value_heads * value_dim, 44),
    ] {
        weights.insert(
            layer_id(layer, tensor),
            generated_tensor(
                &[output, input],
                salt + layer as u64 * 47,
                1.0 / (input as f32).sqrt(),
            )?,
        );
    }
    weights.insert(
        layer_id(layer, LayerTensor::GdnA),
        ReferenceTensor::new(vec![value_heads], vec![-0.5; value_heads])?,
    );
    weights.insert(
        layer_id(layer, LayerTensor::GdnDtBias),
        ReferenceTensor::new(vec![value_heads], vec![0.0; value_heads])?,
    );
    weights.insert(
        layer_id(layer, LayerTensor::GdnNorm),
        ReferenceTensor::new(vec![value_dim], vec![1.0; value_dim])?,
    );
    weights.insert(
        layer_id(layer, LayerTensor::GdnConv1d),
        generated_tensor(
            &[conv_width, gdn.conv_kernel as usize],
            45 + layer as u64 * 47,
            1.0 / (gdn.conv_kernel as f32).sqrt(),
        )?,
    );
    Ok(())
}

fn add_kda_fixture(
    weights: &mut ReferenceWeights,
    layer: u32,
    kda: &memra_gguf::model_plan::KimiDeltaNetPlan,
    hidden: usize,
) -> Result<(), ReferenceError> {
    let heads = kda.num_heads as usize;
    let head_dim = kda.head_dim as usize;
    let kernel = kda.conv_kernel as usize;
    let qkv = heads * head_dim;
    for (tensor, output, input, salt) in [
        (LayerTensor::KdaQuery, qkv, hidden, 140),
        (LayerTensor::KdaKey, qkv, hidden, 141),
        (LayerTensor::KdaValue, qkv, hidden, 142),
        (LayerTensor::KdaForgetDown, head_dim, hidden, 143),
        (LayerTensor::KdaForgetUp, qkv, head_dim, 144),
        (LayerTensor::KdaGateDown, head_dim, hidden, 145),
        (LayerTensor::KdaGateUp, qkv, head_dim, 146),
        (LayerTensor::KdaBeta, heads, hidden, 147),
        (LayerTensor::KdaOutput, hidden, qkv, 148),
    ] {
        weights.insert(
            layer_id(layer, tensor),
            generated_tensor(
                &[output, input],
                salt + layer as u64 * 163,
                1.0 / (input as f32).sqrt(),
            )?,
        );
    }
    for (tensor, salt) in [
        (LayerTensor::KdaQueryConv, 149),
        (LayerTensor::KdaKeyConv, 150),
        (LayerTensor::KdaValueConv, 151),
    ] {
        weights.insert(
            layer_id(layer, tensor),
            generated_tensor(
                &[qkv, kernel],
                salt + layer as u64 * 163,
                1.0 / (kernel as f32).sqrt(),
            )?,
        );
    }
    weights.insert(
        layer_id(layer, LayerTensor::KdaALog),
        generated_tensor(&[heads], 152 + layer as u64 * 163, 0.1)?,
    );
    // dt_bias is per-CHANNEL (width qkv), unlike GDN's per-head bias.
    weights.insert(
        layer_id(layer, LayerTensor::KdaDtBias),
        generated_tensor(&[qkv], 153 + layer as u64 * 163, 0.1)?,
    );
    weights.insert(
        layer_id(layer, LayerTensor::KdaOutputNorm),
        ReferenceTensor::new(vec![head_dim], vec![1.0; head_dim])?,
    );
    Ok(())
}

fn add_mla_fixture(
    weights: &mut ReferenceWeights,
    layer: u32,
    mla: &memra_gguf::model_plan::MlaAttentionPlan,
    hidden: usize,
) -> Result<(), ReferenceError> {
    if let memra_gguf::model_plan::MlaAttentionPlan::CompressedKv { .. } = mla {
        return add_compressed_mla_fixture(weights, layer, mla, hidden);
    }
    let memra_gguf::model_plan::MlaAttentionPlan::LatentKv {
        query_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_head_dim,
        rope_head_dim,
        value_head_dim,
        sparse_index,
        ..
    } = mla.clone()
    else {
        return Err(ReferenceError::UnsupportedOperation {
            layer: Some(layer),
            operation: "compressed-KV MLA fixture",
        });
    };
    let heads = query_heads as usize;
    let q_rank = q_lora_rank as usize;
    let kv_rank = kv_lora_rank as usize;
    let qk_dim = qk_head_dim as usize;
    let rope_dim = rope_head_dim as usize;
    let nope_dim = qk_dim - rope_dim;
    let value_dim = value_head_dim as usize;
    for (tensor, shape, input, salt) in [
        (LayerTensor::MlaQueryDown, vec![q_rank, hidden], hidden, 80),
        (
            LayerTensor::MlaQueryUp,
            vec![heads * qk_dim, q_rank],
            q_rank,
            81,
        ),
        (
            LayerTensor::MlaKvDown,
            vec![kv_rank + rope_dim, hidden],
            hidden,
            82,
        ),
        // `[head][rank][nope]`, NOT the checkpoint's own `[head][nope][rank]`. This is the
        // `TensorId::MlaKeyUp` layout the tensor contract declares (GGUF ne
        // `[nope, kv_rank, heads]`, fastest axis first), which is what llama.cpp's `attn_k_b`
        // mint and `hf_mapping::split_mla_kv_plane` both emit and what the engine's absorb GEMM
        // reads. One TensorId, one byte order: a fixture minted the other way round mis-strides
        // every engine-vs-reference MLA comparison while preserving element counts.
        (
            LayerTensor::MlaKeyUp,
            vec![heads, kv_rank, nope_dim],
            kv_rank,
            83,
        ),
        (
            LayerTensor::MlaValueUp,
            vec![heads, value_dim, kv_rank],
            kv_rank,
            84,
        ),
        (
            LayerTensor::MlaOutput,
            vec![hidden, heads * value_dim],
            heads * value_dim,
            85,
        ),
    ] {
        weights.insert(
            layer_id(layer, tensor),
            generated_tensor(
                &shape,
                salt + layer as u64 * 71,
                1.0 / (input as f32).sqrt(),
            )?,
        );
    }
    weights.insert(
        layer_id(layer, LayerTensor::MlaQueryDownNorm),
        ReferenceTensor::new(vec![q_rank], vec![1.0; q_rank])?,
    );
    weights.insert(
        layer_id(layer, LayerTensor::MlaKvDownNorm),
        ReferenceTensor::new(vec![kv_rank], vec![1.0; kv_rank])?,
    );
    // Only the k-pool indexer (glm5_next) owns tensors on the LatentKv path; the
    // per-token variant executes through full-selection equivalence without them.
    if let memra_gguf::model_plan::SparseIndexPlan::Own {
        heads: index_heads,
        head_dim: index_dim,
        top_k: _,
        kpool: Some(kpool),
    } = sparse_index
    {
        let index_heads = index_heads as usize;
        let index_dim = index_dim as usize;
        let pool = kpool.pool as usize;
        for (tensor, shape, input, salt) in [
            (
                LayerTensor::SparseQuery,
                vec![index_heads * index_dim, q_rank],
                q_rank,
                120,
            ),
            (LayerTensor::SparseKey, vec![index_dim, hidden], hidden, 121),
            (
                LayerTensor::SparseProjection,
                vec![index_heads, hidden],
                hidden,
                122,
            ),
            (
                LayerTensor::SparseCompressorGate,
                vec![index_dim, hidden],
                hidden,
                123,
            ),
            (
                LayerTensor::SparseCompressorPosition,
                vec![pool, index_dim],
                index_dim,
                124,
            ),
        ] {
            weights.insert(
                layer_id(layer, tensor),
                generated_tensor(
                    &shape,
                    salt + layer as u64 * 71,
                    1.0 / (input as f32).sqrt(),
                )?,
            );
        }
        weights.insert(
            layer_id(layer, LayerTensor::SparseKeyNorm),
            ReferenceTensor::new(vec![index_dim], vec![1.0; index_dim])?,
        );
        // LayerNorm bias (nonzero so the bias path is exercised).
        weights.insert(
            layer_id(layer, LayerTensor::SparseKeyNormBias),
            generated_tensor(&[index_dim], 125 + layer as u64 * 71, 0.05)?,
        );
    }
    Ok(())
}

#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn add_compressed_mla_fixture(
    weights: &mut ReferenceWeights,
    layer: u32,
    mla: &memra_gguf::model_plan::MlaAttentionPlan,
    hidden: usize,
) -> Result<(), ReferenceError> {
    use memra_gguf::model_plan::{MlaAttentionPlan, SparseIndexPlan};

    let MlaAttentionPlan::CompressedKv {
        query_heads,
        q_lora_rank,
        latent_head_dim,
        rope_head_dim,
        output_lora_rank,
        output_groups,
        compressor,
        sparse_index,
        ..
    } = mla
    else {
        unreachable!()
    };
    let heads = *query_heads as usize;
    let q_rank = *q_lora_rank as usize;
    let head_dim = *latent_head_dim as usize;
    let rope_dim = *rope_head_dim as usize;
    let output_rank = *output_lora_rank as usize;
    let groups = *output_groups as usize;
    if groups == 0 || heads % groups != 0 || rope_dim > head_dim {
        return Err(ReferenceError::InvalidPlan {
            layer: Some(layer),
            reason: "compressed attention has invalid head or output-group geometry",
        });
    }
    let group_width = heads / groups * head_dim;
    for (tensor, shape, input, salt) in [
        (LayerTensor::MlaQueryDown, vec![q_rank, hidden], hidden, 110),
        (
            LayerTensor::MlaQueryUp,
            vec![heads * head_dim, q_rank],
            q_rank,
            111,
        ),
        (LayerTensor::MlaKvDown, vec![head_dim, hidden], hidden, 112),
        (
            LayerTensor::MlaOutputDown,
            vec![groups * output_rank, group_width],
            group_width,
            113,
        ),
        (
            LayerTensor::MlaOutput,
            vec![hidden, groups * output_rank],
            groups * output_rank,
            114,
        ),
    ] {
        weights.insert(
            layer_id(layer, tensor),
            generated_tensor(
                &shape,
                salt + layer as u64 * 131,
                1.0 / (input as f32).sqrt(),
            )?,
        );
    }
    weights.insert(
        layer_id(layer, LayerTensor::MlaQueryDownNorm),
        ReferenceTensor::new(vec![q_rank], vec![1.0; q_rank])?,
    );
    weights.insert(
        layer_id(layer, LayerTensor::MlaKvDownNorm),
        ReferenceTensor::new(vec![head_dim], vec![1.0; head_dim])?,
    );
    weights.insert(
        layer_id(layer, LayerTensor::AttentionSink),
        generated_tensor(&[heads], 115 + layer as u64 * 131, 0.05)?,
    );
    if let Some(compressor) = compressor {
        add_compressor_fixture(
            weights,
            layer,
            hidden,
            head_dim,
            compressor.ratio as usize,
            compressor.latent_dim as usize,
            false,
        )?;
    }
    match sparse_index {
        SparseIndexPlan::None => {}
        SparseIndexPlan::Own {
            heads, head_dim, ..
        } => {
            let Some(compressor) = compressor else {
                return Err(ReferenceError::InvalidPlan {
                    layer: Some(layer),
                    reason: "compressed sparse index requires a compressor ratio",
                });
            };
            let index_heads = *heads as usize;
            let index_dim = *head_dim as usize;
            weights.insert(
                layer_id(layer, LayerTensor::SparseQuery),
                generated_tensor(
                    &[index_heads * index_dim, q_rank],
                    116 + layer as u64 * 131,
                    1.0 / (q_rank as f32).sqrt(),
                )?,
            );
            weights.insert(
                layer_id(layer, LayerTensor::SparseProjection),
                generated_tensor(
                    &[index_heads, hidden],
                    117 + layer as u64 * 131,
                    1.0 / (hidden as f32).sqrt(),
                )?,
            );
            add_compressor_fixture(
                weights,
                layer,
                hidden,
                index_dim,
                compressor.ratio as usize,
                2 * index_dim,
                true,
            )?;
        }
        SparseIndexPlan::SharedFromPrevious { .. } => {
            return Err(ReferenceError::UnsupportedOperation {
                layer: Some(layer),
                operation: "shared compressed sparse-index fixture",
            });
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn add_compressor_fixture(
    weights: &mut ReferenceWeights,
    layer: u32,
    hidden: usize,
    output_dim: usize,
    ratio: usize,
    latent: usize,
    sparse: bool,
) -> Result<(), ReferenceError> {
    let (key_value, gate, norm, position, salt) = if sparse {
        (
            LayerTensor::SparseCompressorKeyValue,
            LayerTensor::SparseCompressorGate,
            LayerTensor::SparseCompressorNorm,
            LayerTensor::SparseCompressorPosition,
            121,
        )
    } else {
        (
            LayerTensor::KvCompressorKeyValue,
            LayerTensor::KvCompressorGate,
            LayerTensor::KvCompressorNorm,
            LayerTensor::KvCompressorPosition,
            118,
        )
    };
    for (tensor, offset) in [(key_value, 0), (gate, 1)] {
        weights.insert(
            layer_id(layer, tensor),
            generated_tensor(
                &[latent, hidden],
                salt + offset + layer as u64 * 131,
                1.0 / (hidden as f32).sqrt(),
            )?,
        );
    }
    weights.insert(
        layer_id(layer, norm),
        ReferenceTensor::new(vec![output_dim], vec![1.0; output_dim])?,
    );
    weights.insert(
        layer_id(layer, position),
        generated_tensor(&[ratio, latent], salt + 2 + layer as u64 * 131, 0.05)?,
    );
    Ok(())
}

fn add_dense_mlp_fixture(
    weights: &mut ReferenceWeights,
    layer: u32,
    mlp: &memra_gguf::model_plan::DenseMlpPlan,
    hidden: usize,
) -> Result<(), ReferenceError> {
    let intermediate = mlp.intermediate_size as usize;
    for (tensor, output, input, salt) in [
        (LayerTensor::MlpGate, intermediate, hidden, 20),
        (LayerTensor::MlpUp, intermediate, hidden, 21),
        (LayerTensor::MlpDown, hidden, intermediate, 22),
    ] {
        weights.insert(
            layer_id(layer, tensor),
            generated_tensor(
                &[output, input],
                salt + layer as u64 * 31,
                1.0 / (input as f32).sqrt(),
            )?,
        );
    }
    Ok(())
}

fn add_moe_fixture(
    weights: &mut ReferenceWeights,
    layer: u32,
    moe: &memra_gguf::model_plan::MoeMlpPlan,
    hidden: usize,
    vocab: usize,
) -> Result<(), ReferenceError> {
    let experts = moe.expert_count as usize;
    let selected = moe.experts_per_token as usize;
    let intermediate = moe.expert_intermediate_size as usize;
    if matches!(
        moe.router,
        memra_gguf::model_plan::RouterPlan::TokenIdHash { .. }
    ) {
        let mut table = Vec::with_capacity(vocab * selected);
        for token in 0..vocab {
            for rank in 0..selected {
                table.push(((token + rank) % experts) as f32);
            }
        }
        weights.insert(
            layer_id(layer, LayerTensor::MoeTokenToExpert),
            ReferenceTensor::new(vec![vocab, selected], table)?,
        );
    }
    weights.insert(
        layer_id(layer, LayerTensor::MoeRouter),
        generated_tensor(
            &[experts, hidden],
            60 + layer as u64 * 59,
            1.0 / (hidden as f32).sqrt(),
        )?,
    );
    if router_has_selection_bias(&moe.router) {
        weights.insert(
            layer_id(layer, LayerTensor::MoeRouterBias),
            generated_tensor(&[experts], 61 + layer as u64 * 59, 0.05)?,
        );
    }
    for (tensor, shape, input, salt) in [
        (
            LayerTensor::MoeExpertGateBank,
            vec![experts, intermediate, hidden],
            hidden,
            62,
        ),
        (
            LayerTensor::MoeExpertUpBank,
            vec![experts, intermediate, hidden],
            hidden,
            63,
        ),
        (
            LayerTensor::MoeExpertDownBank,
            vec![experts, hidden, intermediate],
            intermediate,
            64,
        ),
    ] {
        weights.insert(
            layer_id(layer, tensor),
            generated_tensor(
                &shape,
                salt + layer as u64 * 59,
                1.0 / (input as f32).sqrt(),
            )?,
        );
    }
    if let Some(shared) = moe.shared.as_ref() {
        let intermediate = shared.intermediate_size as usize;
        for (tensor, output, input, salt) in [
            (LayerTensor::SharedMlpGate, intermediate, hidden, 65),
            (LayerTensor::SharedMlpUp, intermediate, hidden, 66),
            (LayerTensor::SharedMlpDown, hidden, intermediate, 67),
        ] {
            weights.insert(
                layer_id(layer, tensor),
                generated_tensor(
                    &[output, input],
                    salt + layer as u64 * 59,
                    1.0 / (input as f32).sqrt(),
                )?,
            );
        }
        if shared.gated {
            weights.insert(
                layer_id(layer, LayerTensor::SharedMlpInputGate),
                generated_tensor(&[hidden], 68 + layer as u64 * 59, 0.2)?,
            );
        }
    }
    Ok(())
}

fn add_gemma_parallel_moe_fixture(
    weights: &mut ReferenceWeights,
    layer: u32,
    moe: &memra_gguf::model_plan::MoeMlpPlan,
    hidden: usize,
) -> Result<(), ReferenceError> {
    let experts = moe.expert_count as usize;
    let intermediate = moe.expert_intermediate_size as usize;
    weights.insert(
        layer_id(layer, LayerTensor::MoeExpertGateUpBank),
        generated_tensor(
            &[experts, 2 * intermediate, hidden],
            180 + layer as u64 * 19,
            1.0 / (hidden as f32).sqrt(),
        )?,
    );
    weights.insert(
        layer_id(layer, LayerTensor::MoeRouterScale),
        ReferenceTensor::new(vec![hidden], vec![1.0; hidden])?,
    );
    weights.insert(
        layer_id(layer, LayerTensor::MoeExpertOutputScale),
        generated_tensor(&[experts], 181 + layer as u64 * 19, 0.2)?,
    );
    Ok(())
}

pub fn execute(
    plan: &ModelPlan,
    weights: &ReferenceWeights,
    token_ids: &[u32],
) -> Result<ReferenceOutput, ReferenceError> {
    if token_ids.is_empty() {
        return Err(ReferenceError::EmptyInput);
    }
    let hidden = plan.hidden_size as usize;
    let vocab = plan.vocab_size as usize;
    let embedding = tensor(weights, &TensorId::TokenEmbedding, &[vocab, hidden])?;
    let embedded = embed_token_rows(plan, embedding, token_ids, vocab, hidden)?;
    execute_embedded(plan, weights, token_ids, embedding, embedded)
}

pub fn execute_multimodal(
    plan: &ModelPlan,
    weights: &ReferenceWeights,
    token_ids: &[u32],
    vision_input: &ReferenceVisionInput,
) -> Result<ReferenceMultimodalOutput, ReferenceError> {
    if token_ids.is_empty() {
        return Err(ReferenceError::EmptyInput);
    }
    let injection = plan.multimodal.ok_or(ReferenceError::InvalidPlan {
        layer: None,
        reason: "multimodal input requires a vision-token injection plan",
    })?;
    let vision = execute_vision(plan, weights, vision_input)?;
    if vision.output_tokens != injection.tokens_per_image as usize {
        return Err(ReferenceError::InvalidPlan {
            layer: None,
            reason: "vision output token count does not match the injection plan",
        });
    }
    let placeholder_count = token_ids
        .iter()
        .filter(|&&token| token == injection.placeholder_token_id)
        .count();
    if placeholder_count != vision.output_tokens {
        return Err(ReferenceError::InvalidPlan {
            layer: None,
            reason: "image placeholder count does not match projected vision tokens",
        });
    }
    let hidden = plan.hidden_size as usize;
    let vocab = plan.vocab_size as usize;
    let embedding = tensor(weights, &TensorId::TokenEmbedding, &[vocab, hidden])?;
    let mut embedded = embed_token_rows(plan, embedding, token_ids, vocab, hidden)?;
    let mut vision_row = 0;
    for (position, &token) in token_ids.iter().enumerate() {
        if token == injection.placeholder_token_id {
            embedded[position * hidden..(position + 1) * hidden].copy_from_slice(
                &vision.projected_hidden[vision_row * hidden..(vision_row + 1) * hidden],
            );
            vision_row += 1;
        }
    }
    let language = execute_embedded(plan, weights, token_ids, embedding, embedded)?;
    Ok(ReferenceMultimodalOutput { language, vision })
}

fn embed_token_rows(
    plan: &ModelPlan,
    embedding: &[f32],
    token_ids: &[u32],
    vocab: usize,
    hidden: usize,
) -> Result<Vec<f32>, ReferenceError> {
    let mut embedded = vec![0.0; token_ids.len() * hidden];
    for (position, &token) in token_ids.iter().enumerate() {
        let token = token as usize;
        if token >= vocab {
            return Err(ReferenceError::TokenOutOfRange {
                token: token as u32,
                vocab,
            });
        }
        embedded[position * hidden..(position + 1) * hidden]
            .copy_from_slice(&embedding[token * hidden..(token + 1) * hidden]);
        if plan.embedding_scale != 1.0 {
            for value in &mut embedded[position * hidden..(position + 1) * hidden] {
                *value *= plan.embedding_scale;
            }
        }
    }
    Ok(embedded)
}

fn execute_embedded(
    plan: &ModelPlan,
    weights: &ReferenceWeights,
    token_ids: &[u32],
    embedding: &[f32],
    embedded: Vec<f32>,
) -> Result<ReferenceOutput, ReferenceError> {
    let tokens = token_ids.len();
    let hidden = plan.hidden_size as usize;
    let vocab = plan.vocab_size as usize;
    if embedded.len() != tokens * hidden {
        return Err(ReferenceError::InvalidPlan {
            layer: None,
            reason: "embedded language input does not match tokens x hidden",
        });
    }
    let hyper = hyper_topology(plan)?;
    let mut x = hyper.map_or(embedded.clone(), |(streams, _, _, _)| {
        memra_gguf::dsv4_forward::hc_expand(&embedded, tokens, streams, hidden)
    });

    let mut state = Vec::with_capacity(plan.layers.len());
    let dspark = plan.drafter.as_ref().map(|drafter| match drafter {
        memra_gguf::model_plan::DrafterPlan::Dspark(plan) => plan,
    });
    let mut draft_taps = dspark.map(|plan| vec![None; plan.target_layer_ids.len()]);
    for layer in &plan.layers {
        let (next, layer_state) =
            execute_layer(layer, weights, &x, token_ids, tokens, hidden, vocab)?;
        x = next;
        if let (Some(dspark), Some(taps)) = (dspark, draft_taps.as_mut())
            && let Some(target) = dspark
                .target_layer_ids
                .iter()
                .position(|&target| target == layer.index)
        {
            taps[target] = Some(collapse_stream_mean(
                &x,
                tokens,
                hidden,
                hyper.map(|topology| topology.0),
            )?);
        }
        state.push(layer_state);
    }
    let trunk_hidden = x;
    let logits = project_trunk_logits(
        plan,
        weights,
        &trunk_hidden,
        tokens,
        hidden,
        vocab,
        embedding,
    )?;
    let output = weights
        .get(&TensorId::OutputProjection)
        .map(|tensor| tensor_checked(&TensorId::OutputProjection, tensor, &[vocab, hidden]))
        .transpose()?
        .unwrap_or(embedding);
    let draft = match (dspark, draft_taps) {
        (Some(dspark), Some(taps)) => Some(execute_dspark(
            dspark,
            weights,
            token_ids,
            embedding,
            output,
            &plan.logits,
            plan.output_norm.epsilon,
            hidden,
            vocab,
            taps,
        )?),
        _ => None,
    };
    let mtp = execute_mtp(
        plan,
        weights,
        token_ids,
        embedding,
        &trunk_hidden,
        tokens,
        hidden,
        vocab,
        output,
    )?;
    Ok(ReferenceOutput {
        logits,
        tokens,
        vocab,
        state: ReferenceState { layers: state },
        mtp,
        draft,
    })
}

/// Trunk exit shared by [`execute`] and the streamed driver: stream collapse, output
/// norm, LM-head projection, and logits transforms. `embedding` is the tied-head
/// fallback when `OutputProjection` is absent. Kept as one function so the two paths
/// cannot drift; the checkpoint runner's `--self-test` pins them bit-for-bit.
fn project_trunk_logits(
    plan: &ModelPlan,
    weights: &ReferenceWeights,
    trunk_hidden: &[f32],
    tokens: usize,
    hidden: usize,
    vocab: usize,
    embedding: &[f32],
) -> Result<Vec<f32>, ReferenceError> {
    let hyper = hyper_topology(plan)?;
    let collapsed: Vec<f32>;
    let x: &[f32] = if let Some((streams, epsilon, _, collapse)) = hyper {
        collapsed = match collapse {
            HcCollapse::GatedHead => collapse_hyper_head(
                weights,
                trunk_hidden,
                tokens,
                streams,
                hidden,
                plan,
                epsilon,
            )?,
            HcCollapse::Mean => collapse_stream_mean(trunk_hidden, tokens, hidden, Some(streams))?,
        };
        &collapsed
    } else {
        trunk_hidden
    };
    if crate::hidden_trace::enabled() {
        crate::hidden_trace::emit_last_row("collapse", -1, tokens, hidden, x);
    }
    let x = rms_norm(
        x,
        tokens,
        hidden,
        tensor(weights, &TensorId::OutputNorm, &[hidden])?,
        plan.output_norm.epsilon,
    );
    let output = weights
        .get(&TensorId::OutputProjection)
        .map(|tensor| tensor_checked(&TensorId::OutputProjection, tensor, &[vocab, hidden]))
        .transpose()?
        .unwrap_or(embedding);
    let mut logits = linear(&x, output, tokens, hidden, vocab);
    apply_logits_transforms(&mut logits, vocab, &plan.logits);
    Ok(logits)
}

/// Layer-at-a-time trunk execution over the exact per-layer math of [`execute`], for
/// checkpoint-scale runs where all weights cannot be resident at once. The driver
/// materializes only the current layer's tensors, calls [`StreamedTrunkExecution::step`],
/// and frees them before the next layer.
///
/// `begin` needs `TokenEmbedding` in `globals`; `finish` needs `OutputNorm` plus
/// `OutputProjection` (falling back to the embedding for tied heads) and, for a
/// gated-head collapse, the `HyperHead*` tensors.
///
/// Deliberate scope: trunk + final norm + LM head only. MTP blocks are not executed
/// (`mtp` stays empty) and drafter plans are refused at `begin` — the streamed path has
/// no per-layer tap capture. The glm5 checkpoint runner's `--self-test` mode pins this
/// path against [`execute`] bit-for-bit.
pub struct StreamedTrunkExecution<'a> {
    plan: &'a ModelPlan,
    token_ids: Vec<u32>,
    x: Vec<f32>,
    tokens: usize,
    hidden: usize,
    vocab: usize,
    next: usize,
    states: Vec<ReferenceLayerState>,
}

impl<'a> StreamedTrunkExecution<'a> {
    pub fn begin(
        plan: &'a ModelPlan,
        globals: &ReferenceWeights,
        token_ids: &[u32],
    ) -> Result<Self, ReferenceError> {
        if token_ids.is_empty() {
            return Err(ReferenceError::EmptyInput);
        }
        if plan.drafter.is_some() {
            return Err(ReferenceError::UnsupportedOperation {
                layer: None,
                operation: "streamed drafter execution",
            });
        }
        let tokens = token_ids.len();
        let hidden = plan.hidden_size as usize;
        let vocab = plan.vocab_size as usize;
        let embedding = tensor(globals, &TensorId::TokenEmbedding, &[vocab, hidden])?;
        let embedded = embed_token_rows(plan, embedding, token_ids, vocab, hidden)?;
        let x = match hyper_topology(plan)? {
            Some((streams, _, _, _)) => {
                memra_gguf::dsv4_forward::hc_expand(&embedded, tokens, streams, hidden)
            }
            None => embedded,
        };
        if crate::hidden_trace::enabled() {
            crate::hidden_trace::emit_tokens(token_ids);
            let width = x.len() / tokens;
            crate::hidden_trace::emit_last_row("expand", -1, tokens, width, &x);
        }
        Ok(Self {
            plan,
            token_ids: token_ids.to_vec(),
            x,
            tokens,
            hidden,
            vocab,
            next: 0,
            states: Vec::with_capacity(plan.layers.len()),
        })
    }

    /// The plan layer the next [`Self::step`] call will execute, or `None` when the
    /// trunk is fully executed.
    pub fn next_layer(&self) -> Option<&'a memra_gguf::model_plan::LayerPlan> {
        self.plan.layers.get(self.next)
    }

    /// Execute the next trunk layer using only that layer's tensors. Returns the
    /// executed layer's plan index.
    pub fn step(&mut self, weights: &ReferenceWeights) -> Result<u32, ReferenceError> {
        let layer = self
            .plan
            .layers
            .get(self.next)
            .ok_or(ReferenceError::InvalidPlan {
                layer: None,
                reason: "streamed trunk stepped past the final layer",
            })?;
        let (next, layer_state) = execute_layer(
            layer,
            weights,
            &self.x,
            &self.token_ids,
            self.tokens,
            self.hidden,
            self.vocab,
        )?;
        self.x = next;
        self.states.push(layer_state);
        self.next += 1;
        Ok(layer.index)
    }

    /// Collapse, final-norm, and project the trunk. MTP blocks are skipped by design.
    pub fn finish(self, globals: &ReferenceWeights) -> Result<ReferenceOutput, ReferenceError> {
        if self.next != self.plan.layers.len() {
            return Err(ReferenceError::InvalidPlan {
                layer: None,
                reason: "streamed trunk finished before executing every layer",
            });
        }
        let embedding = tensor(
            globals,
            &TensorId::TokenEmbedding,
            &[self.vocab, self.hidden],
        )?;
        let logits = project_trunk_logits(
            self.plan,
            globals,
            &self.x,
            self.tokens,
            self.hidden,
            self.vocab,
            embedding,
        )?;
        Ok(ReferenceOutput {
            logits,
            tokens: self.tokens,
            vocab: self.vocab,
            state: ReferenceState {
                layers: self.states,
            },
            mtp: Vec::new(),
            draft: None,
        })
    }
}

pub fn execute_vision(
    plan: &ModelPlan,
    weights: &ReferenceWeights,
    input: &ReferenceVisionInput,
) -> Result<ReferenceVisionOutput, ReferenceError> {
    let Some(vision) = plan.vision.as_ref() else {
        return Err(ReferenceError::InvalidPlan {
            layer: None,
            reason: "vision input requires a vision subplan",
        });
    };
    if vision.clipped_linears {
        return Err(ReferenceError::UnsupportedOperation {
            layer: None,
            operation: "clipped vision linears",
        });
    }
    let patches = input.positions.len();
    let hidden = vision.hidden_size as usize;
    let patch_width =
        (vision.patch.channels * vision.patch.patch_size * vision.patch.patch_size) as usize;
    if input.patches.shape != [patches, patch_width]
        || input.output_tokens == 0
        || input.output_tokens > patches
    {
        return Err(ReferenceError::InvalidPlan {
            layer: None,
            reason: "vision patch input shape or output-token count is invalid",
        });
    }
    let mut normalized_patches = input.patches.data.clone();
    for value in &mut normalized_patches {
        *value = 2.0 * (*value - 0.5);
    }
    let mut x = linear(
        &normalized_patches,
        tensor(
            weights,
            &TensorId::Vision {
                layer: None,
                tensor: VisionTensor::PatchProjection,
            },
            &[hidden, patch_width],
        )?,
        patches,
        patch_width,
        hidden,
    );
    let position_table = tensor(
        weights,
        &TensorId::Vision {
            layer: None,
            tensor: VisionTensor::PositionEmbedding,
        },
        &[
            vision.patch.position_axes as usize,
            vision.patch.position_embedding_size as usize,
            hidden,
        ],
    )?;
    for (patch, position) in input.positions.iter().enumerate() {
        for (axis, &coordinate) in position.iter().enumerate() {
            let coordinate = coordinate as usize;
            if axis >= vision.patch.position_axes as usize
                || coordinate >= vision.patch.position_embedding_size as usize
            {
                return Err(ReferenceError::InvalidPlan {
                    layer: None,
                    reason: "vision patch position is outside the embedding table",
                });
            }
            let source =
                (axis * vision.patch.position_embedding_size as usize + coordinate) * hidden;
            for column in 0..hidden {
                x[patch * hidden + column] += position_table[source + column];
            }
        }
    }
    for layer in &vision.layers {
        x = execute_vision_layer(layer, weights, &x, &input.positions, patches, hidden)?;
    }
    let encoder_hidden = x.clone();
    let pooled_hidden = vision_pool(&x, &input.positions, patches, input.output_tokens, hidden)?;
    let mut standardized = pooled_hidden.clone();
    if vision.standardize {
        let bias = tensor(
            weights,
            &TensorId::Vision {
                layer: None,
                tensor: VisionTensor::StandardizeBias,
            },
            &[hidden],
        )?;
        let scale = tensor(
            weights,
            &TensorId::Vision {
                layer: None,
                tensor: VisionTensor::StandardizeScale,
            },
            &[hidden],
        )?;
        for row in standardized.chunks_exact_mut(hidden) {
            for column in 0..hidden {
                row[column] = (row[column] - bias[column]) * scale[column];
            }
        }
    }
    let standardized = rms_norm(
        &standardized,
        input.output_tokens,
        hidden,
        &vec![1.0; hidden],
        vision.layers[0].input_norm.epsilon,
    );
    let projection_size = vision.projection_output_size as usize;
    let projected_hidden = linear(
        &standardized,
        tensor(
            weights,
            &TensorId::Vision {
                layer: None,
                tensor: VisionTensor::OutputProjection,
            },
            &[projection_size, hidden],
        )?,
        input.output_tokens,
        hidden,
        projection_size,
    );
    Ok(ReferenceVisionOutput {
        encoder_hidden,
        pooled_hidden,
        projected_hidden,
        patch_count: patches,
        output_tokens: input.output_tokens,
        hidden_size: hidden,
        projection_size,
    })
}

#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn execute_vision_layer(
    plan: &memra_gguf::model_plan::VisionLayerPlan,
    weights: &ReferenceWeights,
    input: &[f32],
    positions: &[[u32; 2]],
    tokens: usize,
    hidden: usize,
) -> Result<Vec<f32>, ReferenceError> {
    let id = |tensor| TensorId::Vision {
        layer: Some(plan.index),
        tensor,
    };
    let attention_input = rms_norm(
        input,
        tokens,
        hidden,
        tensor(weights, &id(VisionTensor::InputNorm), &[hidden])?,
        plan.input_norm.epsilon,
    );
    let query_heads = plan.attention.query_heads as usize;
    let kv_heads = plan.attention.kv_heads as usize;
    let head_dim = plan.attention.head_dim as usize;
    if query_heads == 0 || kv_heads == 0 || query_heads % kv_heads != 0 {
        return Err(ReferenceError::InvalidPlan {
            layer: Some(plan.index),
            reason: "vision attention has invalid query/KV head grouping",
        });
    }
    let mut query = linear(
        &attention_input,
        tensor(
            weights,
            &id(VisionTensor::Query),
            &[query_heads * head_dim, hidden],
        )?,
        tokens,
        hidden,
        query_heads * head_dim,
    );
    let mut key = linear(
        &attention_input,
        tensor(
            weights,
            &id(VisionTensor::Key),
            &[kv_heads * head_dim, hidden],
        )?,
        tokens,
        hidden,
        kv_heads * head_dim,
    );
    let mut value = linear(
        &attention_input,
        tensor(
            weights,
            &id(VisionTensor::Value),
            &[kv_heads * head_dim, hidden],
        )?,
        tokens,
        hidden,
        kv_heads * head_dim,
    );
    apply_optional_head_norm(
        weights,
        id(VisionTensor::QueryNorm),
        &mut query,
        tokens * query_heads,
        head_dim,
        memra_gguf::model_plan::TensorPresence::Required,
        plan.input_norm.epsilon,
    )?;
    apply_optional_head_norm(
        weights,
        id(VisionTensor::KeyNorm),
        &mut key,
        tokens * kv_heads,
        head_dim,
        memra_gguf::model_plan::TensorPresence::Required,
        plan.input_norm.epsilon,
    )?;
    value = rms_norm(
        &value,
        tokens * kv_heads,
        head_dim,
        &vec![1.0; head_dim],
        plan.input_norm.epsilon,
    );
    apply_vision_rope(
        &mut query,
        tokens,
        query_heads,
        head_dim,
        positions,
        plan.attention.rope.base,
    )?;
    apply_vision_rope(
        &mut key,
        tokens,
        kv_heads,
        head_dim,
        positions,
        plan.attention.rope.base,
    )?;
    let repeat = query_heads / kv_heads;
    let mut attended = vec![0.0; tokens * query_heads * head_dim];
    for token in 0..tokens {
        for head in 0..query_heads {
            let kv_head = head / repeat;
            let mut scores = Vec::with_capacity(tokens);
            for source in 0..tokens {
                let mut score = 0.0;
                for column in 0..head_dim {
                    score += query[(token * query_heads + head) * head_dim + column]
                        * key[(source * kv_heads + kv_head) * head_dim + column];
                }
                scores.push(score);
            }
            softmax_in_place(&mut scores);
            for (source, probability) in scores.into_iter().enumerate() {
                for column in 0..head_dim {
                    attended[(token * query_heads + head) * head_dim + column] +=
                        probability * value[(source * kv_heads + kv_head) * head_dim + column];
                }
            }
        }
    }
    let attention = linear(
        &attended,
        tensor(
            weights,
            &id(VisionTensor::AttentionOutput),
            &[hidden, query_heads * head_dim],
        )?,
        tokens,
        query_heads * head_dim,
        hidden,
    );
    let attention = rms_norm(
        &attention,
        tokens,
        hidden,
        tensor(weights, &id(VisionTensor::PostAttentionNorm), &[hidden])?,
        plan.post_attention_norm.epsilon,
    );
    let mut residual = input.to_vec();
    add_in_place(&mut residual, &attention);
    let mlp_input = rms_norm(
        &residual,
        tokens,
        hidden,
        tensor(weights, &id(VisionTensor::PreMlpNorm), &[hidden])?,
        plan.pre_mlp_norm.epsilon,
    );
    let intermediate = plan.mlp.intermediate_size as usize;
    let gate = linear(
        &mlp_input,
        tensor(weights, &id(VisionTensor::MlpGate), &[intermediate, hidden])?,
        tokens,
        hidden,
        intermediate,
    );
    let up = linear(
        &mlp_input,
        tensor(weights, &id(VisionTensor::MlpUp), &[intermediate, hidden])?,
        tokens,
        hidden,
        intermediate,
    );
    let mut activated = vec![0.0; gate.len()];
    for index in 0..activated.len() {
        activated[index] = activate_pair(&plan.mlp.activation, gate[index], up[index], plan.index)?;
    }
    let mlp = linear(
        &activated,
        tensor(weights, &id(VisionTensor::MlpDown), &[hidden, intermediate])?,
        tokens,
        intermediate,
        hidden,
    );
    let mlp = rms_norm(
        &mlp,
        tokens,
        hidden,
        tensor(weights, &id(VisionTensor::PostMlpNorm), &[hidden])?,
        plan.post_mlp_norm.epsilon,
    );
    add_in_place(&mut residual, &mlp);
    Ok(residual)
}

#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn apply_vision_rope(
    values: &mut [f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    positions: &[[u32; 2]],
    base: f32,
) -> Result<(), ReferenceError> {
    let axes = 2;
    let chunk = head_dim / axes;
    if head_dim % axes != 0 || !chunk.is_multiple_of(2) || positions.len() != tokens {
        return Err(ReferenceError::InvalidPlan {
            layer: None,
            reason: "vision 2D RoPE requires even per-axis head chunks",
        });
    }
    let half = chunk / 2;
    #[allow(clippy::needless_range_loop)]
    // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
    for token in 0..tokens {
        for head in 0..heads {
            let row = (token * heads + head) * head_dim;
            #[allow(clippy::needless_range_loop)]
            // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
            for axis in 0..axes {
                let start = row + axis * chunk;
                let position = positions[token][axis] as f32;
                for pair in 0..half {
                    let angle = position / base.powf((2 * pair) as f32 / chunk as f32);
                    let (sin, cos) = angle.sin_cos();
                    let left = values[start + pair];
                    let right = values[start + half + pair];
                    values[start + pair] = left * cos - right * sin;
                    values[start + half + pair] = left * sin + right * cos;
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn vision_pool(
    hidden_states: &[f32],
    positions: &[[u32; 2]],
    patches: usize,
    output_tokens: usize,
    hidden: usize,
) -> Result<Vec<f32>, ReferenceError> {
    if patches % output_tokens != 0 {
        return Err(ReferenceError::InvalidPlan {
            layer: None,
            reason: "vision pooling ratio must divide the patch count",
        });
    }
    let area = patches / output_tokens;
    let kernel = (area as f32).sqrt() as usize;
    if kernel * kernel != area {
        return Err(ReferenceError::InvalidPlan {
            layer: None,
            reason: "vision pooling ratio must be a square kernel",
        });
    }
    let max_x = positions
        .iter()
        .map(|position| position[0] as usize)
        .max()
        .unwrap_or(0)
        + 1;
    let grid_width = max_x / kernel;
    let mut output = vec![0.0; output_tokens * hidden];
    for patch in 0..patches {
        let target = positions[patch][0] as usize / kernel
            + grid_width * (positions[patch][1] as usize / kernel);
        if target >= output_tokens {
            return Err(ReferenceError::InvalidPlan {
                layer: None,
                reason: "vision patch positions do not fit the pooled grid",
            });
        }
        for column in 0..hidden {
            output[target * hidden + column] +=
                hidden_states[patch * hidden + column] / area as f32;
        }
    }
    let scale = (hidden as f32).sqrt();
    for value in &mut output {
        *value *= scale;
    }
    Ok(output)
}

fn collapse_stream_mean(
    x: &[f32],
    tokens: usize,
    hidden: usize,
    hyper_streams: Option<usize>,
) -> Result<Vec<f32>, ReferenceError> {
    let Some(streams) = hyper_streams else {
        if x.len() != tokens * hidden {
            return Err(ReferenceError::InvalidPlan {
                layer: None,
                reason: "single-stream DSpark tap has invalid shape",
            });
        }
        return Ok(x.to_vec());
    };
    if x.len() != tokens * streams * hidden {
        return Err(ReferenceError::InvalidPlan {
            layer: None,
            reason: "HyperConnections DSpark tap has invalid shape",
        });
    }
    let mut output = vec![0.0; tokens * hidden];
    for token in 0..tokens {
        for stream in 0..streams {
            for column in 0..hidden {
                output[token * hidden + column] +=
                    x[(token * streams + stream) * hidden + column] / streams as f32;
            }
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn execute_dspark(
    plan: &memra_gguf::model_plan::DsparkPlan,
    weights: &ReferenceWeights,
    token_ids: &[u32],
    embedding: &[f32],
    output_projection: &[f32],
    logits_transforms: &[LogitsTransform],
    norm_epsilon: f32,
    hidden: usize,
    vocab: usize,
    taps: Vec<Option<Vec<f32>>>,
) -> Result<ReferenceDraftOutput, ReferenceError> {
    use memra_gguf::dsv4_forward::{hc_expand, hc_head, matmul, rmsnorm};

    let tokens = token_ids.len();
    let block_size = plan.block_size as usize;
    let rank = plan.markov_rank as usize;
    if tokens < 2
        || block_size == 0
        || plan.blocks.is_empty()
        || taps.len() != plan.target_layer_ids.len()
        || plan.noise_token_id as usize >= vocab
    {
        return Err(ReferenceError::InvalidPlan {
            layer: None,
            reason: "DSpark execution requires a primed prompt and valid drafter geometry",
        });
    }
    let streams = match plan.blocks[0].residual {
        ResidualTopology::HyperConnections { streams, .. } if streams > 0 => streams as usize,
        _ => {
            return Err(ReferenceError::InvalidPlan {
                layer: Some(plan.blocks[0].index),
                reason: "DSpark blocks require HyperConnections",
            });
        }
    };
    let mut main_hidden = vec![0.0; tokens * taps.len() * hidden];
    for (target, tap) in taps.into_iter().enumerate() {
        let Some(tap) = tap else {
            return Err(ReferenceError::InvalidPlan {
                layer: None,
                reason: "DSpark target layer was not captured from the trunk",
            });
        };
        if tap.len() != tokens * hidden {
            return Err(ReferenceError::InvalidPlan {
                layer: None,
                reason: "DSpark trunk tap has invalid shape",
            });
        }
        for token in 0..tokens {
            main_hidden[(token * plan.target_layer_ids.len() + target) * hidden
                ..(token * plan.target_layer_ids.len() + target + 1) * hidden]
                .copy_from_slice(&tap[token * hidden..(token + 1) * hidden]);
        }
    }
    let main_x = rmsnorm(
        &matmul(
            &main_hidden,
            tokens,
            plan.target_layer_ids.len() * hidden,
            tensor(
                weights,
                &TensorId::Dspark(DsparkTensor::MainProjection),
                &[hidden, plan.target_layer_ids.len() * hidden],
            )?,
            hidden,
        ),
        tensor(
            weights,
            &TensorId::Dspark(DsparkTensor::MainNorm),
            &[hidden],
        )?,
        norm_epsilon,
    );
    let rings = plan
        .blocks
        .iter()
        .map(|block| dspark_prime_ring(block, weights, &main_x, tokens, hidden, norm_epsilon))
        .collect::<Result<Vec<_>, _>>()?;

    let input_token = *token_ids.last().unwrap();
    let mut draft_ids = vec![plan.noise_token_id; block_size];
    draft_ids[0] = input_token;
    let mut embedded = vec![0.0; block_size * hidden];
    for (position, &token) in draft_ids.iter().enumerate() {
        let token = token as usize;
        embedded[position * hidden..(position + 1) * hidden]
            .copy_from_slice(&embedding[token * hidden..(token + 1) * hidden]);
    }
    let mut draft_hidden = hc_expand(&embedded, block_size, streams, hidden);
    for (block, ring) in plan.blocks.iter().zip(&rings) {
        draft_hidden = execute_dspark_layer(
            block,
            weights,
            &draft_hidden,
            ring,
            tokens - 1,
            block_size,
            hidden,
            vocab,
        )?;
    }
    let head_set = memra_gguf::dsv4_forward::HcSet {
        rows: streams,
        fn_w: tensor(
            weights,
            &TensorId::Dspark(DsparkTensor::HeadHyperFunction),
            &[streams, streams * hidden],
        )?
        .to_vec(),
        base: tensor(
            weights,
            &TensorId::Dspark(DsparkTensor::HeadHyperBase),
            &[streams],
        )?
        .to_vec(),
        scale: tensor(
            weights,
            &TensorId::Dspark(DsparkTensor::HeadHyperScale),
            &[1],
        )?
        .to_vec(),
    };
    let hc_epsilon = match plan.blocks[0].residual {
        ResidualTopology::HyperConnections { epsilon, .. } => epsilon,
        _ => unreachable!(),
    };
    let collapsed = hc_head(
        &draft_hidden,
        block_size,
        streams,
        hidden,
        &head_set,
        norm_epsilon,
        hc_epsilon,
    );
    let normalized = rmsnorm(
        &collapsed,
        tensor(
            weights,
            &TensorId::Dspark(DsparkTensor::OutputNorm),
            &[hidden],
        )?,
        norm_epsilon,
    );
    let mut logits = matmul(&normalized, block_size, hidden, output_projection, vocab);
    apply_logits_transforms(&mut logits, vocab, logits_transforms);

    let markov_embedding = tensor(
        weights,
        &TensorId::Dspark(DsparkTensor::MarkovEmbedding),
        &[vocab, rank],
    )?;
    let markov_output = tensor(
        weights,
        &TensorId::Dspark(DsparkTensor::MarkovOutput),
        &[vocab, rank],
    )?;
    let confidence_weight = tensor(
        weights,
        &TensorId::Dspark(DsparkTensor::ConfidenceProjection),
        &[1, hidden + rank],
    )?;
    let mut output_ids = vec![input_token];
    let mut confidence = Vec::with_capacity(block_size);
    for position in 0..block_size {
        let previous = output_ids[position] as usize;
        let markov = &markov_embedding[previous * rank..(previous + 1) * rank];
        let row = &mut logits[position * vocab..(position + 1) * vocab];
        for token in 0..vocab {
            row[token] += memra_gguf::dsv4_forward::dot(
                markov,
                &markov_output[token * rank..(token + 1) * rank],
            );
        }
        let next = row
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index as u32)
            .unwrap();
        output_ids.push(next);
        let mut confidence_input = Vec::with_capacity(hidden + rank);
        confidence_input.extend_from_slice(&collapsed[position * hidden..(position + 1) * hidden]);
        confidence_input.extend_from_slice(markov);
        confidence.push(memra_gguf::dsv4_forward::dot(
            &confidence_input,
            confidence_weight,
        ));
    }
    Ok(ReferenceDraftOutput {
        input_token,
        output_ids,
        confidence,
        logits,
        hidden: collapsed,
        block_size,
    })
}

fn dspark_prime_ring(
    layer: &memra_gguf::model_plan::LayerPlan,
    weights: &ReferenceWeights,
    main_x: &[f32],
    tokens: usize,
    hidden: usize,
    epsilon: f32,
) -> Result<Vec<f32>, ReferenceError> {
    use memra_gguf::dsv4_forward::{ActQuantVariant, apply_rope, matmul, rmsnorm};
    use memra_gguf::model_plan::{MlaAttentionPlan, RopeFactors, SparseIndexPlan};

    let AttentionPlan::Mla(MlaAttentionPlan::CompressedKv {
        latent_head_dim,
        rope_head_dim,
        window,
        rope,
        compressor: None,
        sparse_index: SparseIndexPlan::None,
        ..
    }) = &layer.attention
    else {
        return Err(ReferenceError::InvalidPlan {
            layer: Some(layer.index),
            reason: "DSpark blocks require uncompressed window-only attention",
        });
    };
    if !matches!(rope.factors, RopeFactors::None) {
        return Err(ReferenceError::InvalidPlan {
            layer: Some(layer.index),
            reason: "DSpark block RoPE must not use scaling factors",
        });
    }
    let head_dim = *latent_head_dim as usize;
    let rope_dim = *rope_head_dim as usize;
    if head_dim <= rope_dim || !(head_dim - rope_dim).is_multiple_of(64) {
        return Err(ReferenceError::InvalidPlan {
            layer: Some(layer.index),
            reason: "DSpark block has invalid KV quantization geometry",
        });
    }
    let frequencies = memra_gguf::dsv4_forward::precompute_freqs_cis(
        rope_dim,
        tokens + 1,
        0,
        rope.base,
        1.0,
        32.0,
        1.0,
    );
    let mut key_value = rmsnorm(
        &matmul(
            main_x,
            tokens,
            hidden,
            tensor(
                weights,
                &layer_id(layer.index, LayerTensor::MlaKvDown),
                &[head_dim, hidden],
            )?,
            head_dim,
        ),
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::MlaKvDownNorm),
            &[head_dim],
        )?,
        epsilon,
    );
    let positions: Vec<_> = (0..tokens).collect();
    apply_rope(
        &mut key_value,
        tokens,
        1,
        head_dim,
        rope_dim,
        &frequencies,
        &positions,
        false,
    );
    for row in key_value.chunks_exact_mut(head_dim) {
        memra_gguf::dsv4_forward::act_quant(
            &mut row[..head_dim - rope_dim],
            64,
            ActQuantVariant::RefFp8Round,
        );
    }
    let window = *window as usize;
    let mut ring = vec![0.0; window * head_dim];
    for position in tokens.saturating_sub(window)..tokens {
        ring[(position % window) * head_dim..(position % window + 1) * head_dim]
            .copy_from_slice(&key_value[position * head_dim..(position + 1) * head_dim]);
    }
    Ok(ring)
}

#[allow(clippy::too_many_arguments)]
fn execute_dspark_layer(
    layer: &memra_gguf::model_plan::LayerPlan,
    weights: &ReferenceWeights,
    input: &[f32],
    ring: &[f32],
    start_position: usize,
    block_size: usize,
    hidden: usize,
    vocab: usize,
) -> Result<Vec<f32>, ReferenceError> {
    let ResidualTopology::HyperConnections {
        streams,
        epsilon,
        sinkhorn_iterations,
        collapse: _,
    } = layer.residual
    else {
        return Err(ReferenceError::InvalidPlan {
            layer: Some(layer.index),
            reason: "DSpark block requires HyperConnections",
        });
    };
    let streams = streams as usize;
    let attention_set = hyper_set(
        weights,
        layer.index,
        streams,
        hidden,
        LayerTensor::HyperAttentionFunction,
        LayerTensor::HyperAttentionBase,
        LayerTensor::HyperAttentionScale,
    )?;
    let (attention_input, post, combination) = memra_gguf::dsv4_forward::hc_pre(
        input,
        block_size,
        streams,
        hidden,
        &attention_set,
        sinkhorn_iterations,
        epsilon,
    );
    let attention_input = rms_norm(
        &attention_input,
        block_size,
        hidden,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::PreAttentionNorm),
            &[hidden],
        )?,
        layer.pre_attention_norm.epsilon,
    );
    let attention = dspark_attention(
        layer,
        weights,
        &attention_input,
        ring,
        start_position,
        block_size,
        hidden,
    )?;
    let attention_residual = memra_gguf::dsv4_forward::hc_post(
        &attention,
        input,
        block_size,
        streams,
        hidden,
        &post,
        &combination,
    );
    let mlp_set = hyper_set(
        weights,
        layer.index,
        streams,
        hidden,
        LayerTensor::HyperMlpFunction,
        LayerTensor::HyperMlpBase,
        LayerTensor::HyperMlpScale,
    )?;
    let (mlp_input, post, combination) = memra_gguf::dsv4_forward::hc_pre(
        &attention_residual,
        block_size,
        streams,
        hidden,
        &mlp_set,
        sinkhorn_iterations,
        epsilon,
    );
    let mlp_input = rms_norm(
        &mlp_input,
        block_size,
        hidden,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::PreMlpNorm),
            &[hidden],
        )?,
        layer.pre_mlp_norm.epsilon,
    );
    let zeros = vec![0; block_size];
    let mlp = match &layer.mlp {
        MlpPlan::Dense(mlp) => {
            dense_mlp(layer.index, mlp, weights, &mlp_input, block_size, hidden)?
        }
        MlpPlan::Moe(moe) => moe_mlp(
            layer.index,
            moe,
            weights,
            &mlp_input,
            &zeros,
            block_size,
            hidden,
            vocab,
        )?,
    };
    Ok(memra_gguf::dsv4_forward::hc_post(
        &mlp,
        &attention_residual,
        block_size,
        streams,
        hidden,
        &post,
        &combination,
    ))
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn dspark_attention(
    layer: &memra_gguf::model_plan::LayerPlan,
    weights: &ReferenceWeights,
    x: &[f32],
    ring: &[f32],
    start_position: usize,
    block_size: usize,
    hidden: usize,
) -> Result<Vec<f32>, ReferenceError> {
    use memra_gguf::dsv4_forward::{ActQuantVariant, apply_rope, matmul, rmsnorm};
    use memra_gguf::model_plan::{MlaAttentionPlan, RopeFactors, SparseIndexPlan};

    let AttentionPlan::Mla(MlaAttentionPlan::CompressedKv {
        query_heads,
        q_lora_rank,
        latent_head_dim,
        rope_head_dim,
        output_lora_rank,
        output_groups,
        window,
        rope,
        compressor: None,
        sparse_index: SparseIndexPlan::None,
    }) = &layer.attention
    else {
        return Err(ReferenceError::InvalidPlan {
            layer: Some(layer.index),
            reason: "DSpark block requires window-only compressed-attention geometry",
        });
    };
    if !matches!(rope.factors, RopeFactors::None) {
        return Err(ReferenceError::InvalidPlan {
            layer: Some(layer.index),
            reason: "DSpark block RoPE must not use scaling factors",
        });
    }
    let heads = *query_heads as usize;
    let q_rank = *q_lora_rank as usize;
    let head_dim = *latent_head_dim as usize;
    let rope_dim = *rope_head_dim as usize;
    let output_rank = *output_lora_rank as usize;
    let groups = *output_groups as usize;
    let window = *window as usize;
    if start_position == 0
        || head_dim <= rope_dim
        || !(head_dim - rope_dim).is_multiple_of(64)
        || groups == 0
        || heads % groups != 0
        || ring.len() != window * head_dim
    {
        return Err(ReferenceError::InvalidPlan {
            layer: Some(layer.index),
            reason: "DSpark attention has invalid geometry or unprimed ring",
        });
    }
    let positions: Vec<_> = (1..=block_size)
        .map(|offset| start_position + offset)
        .collect();
    let frequencies = memra_gguf::dsv4_forward::precompute_freqs_cis(
        rope_dim,
        start_position + block_size + 1,
        0,
        rope.base,
        1.0,
        32.0,
        1.0,
    );
    let query_low_rank = rmsnorm(
        &matmul(
            x,
            block_size,
            hidden,
            tensor(
                weights,
                &layer_id(layer.index, LayerTensor::MlaQueryDown),
                &[q_rank, hidden],
            )?,
            q_rank,
        ),
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::MlaQueryDownNorm),
            &[q_rank],
        )?,
        layer.pre_attention_norm.epsilon,
    );
    let mut query = matmul(
        &query_low_rank,
        block_size,
        q_rank,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::MlaQueryUp),
            &[heads * head_dim, q_rank],
        )?,
        heads * head_dim,
    );
    for head in query.chunks_exact_mut(head_dim) {
        let mean_square = head
            .iter()
            .map(|value| (*value as f64) * (*value as f64))
            .sum::<f64>()
            / head_dim as f64;
        let scale = 1.0 / (mean_square as f32 + layer.pre_attention_norm.epsilon).sqrt();
        for value in head {
            *value *= scale;
        }
    }
    apply_rope(
        &mut query,
        block_size,
        heads,
        head_dim,
        rope_dim,
        &frequencies,
        &positions,
        false,
    );
    let mut key_value = rmsnorm(
        &matmul(
            x,
            block_size,
            hidden,
            tensor(
                weights,
                &layer_id(layer.index, LayerTensor::MlaKvDown),
                &[head_dim, hidden],
            )?,
            head_dim,
        ),
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::MlaKvDownNorm),
            &[head_dim],
        )?,
        layer.pre_attention_norm.epsilon,
    );
    apply_rope(
        &mut key_value,
        block_size,
        1,
        head_dim,
        rope_dim,
        &frequencies,
        &positions,
        false,
    );
    for row in key_value.chunks_exact_mut(head_dim) {
        memra_gguf::dsv4_forward::act_quant(
            &mut row[..head_dim - rope_dim],
            64,
            ActQuantVariant::RefFp8Round,
        );
    }
    let indices = memra_gguf::dsv4_dspark::dspark_topk_idxs(window, block_size, start_position);
    let sink = tensor(
        weights,
        &layer_id(layer.index, LayerTensor::AttentionSink),
        &[heads],
    )?;
    let mut attended = vec![0.0; block_size * heads * head_dim];
    for token in 0..block_size {
        memra_gguf::dsv4_decode::sparse_attn_query(
            &query[token * heads * head_dim..(token + 1) * heads * head_dim],
            heads,
            head_dim,
            &indices,
            |index| {
                if index < window {
                    &ring[index * head_dim..(index + 1) * head_dim]
                } else {
                    let index = index - window;
                    &key_value[index * head_dim..(index + 1) * head_dim]
                }
            },
            sink,
            (head_dim as f64).powf(-0.5) as f32,
            &mut attended[token * heads * head_dim..(token + 1) * heads * head_dim],
        );
    }
    apply_rope(
        &mut attended,
        block_size,
        heads,
        head_dim,
        rope_dim,
        &frequencies,
        &positions,
        true,
    );
    let group_width = heads / groups * head_dim;
    let output_down = tensor(
        weights,
        &layer_id(layer.index, LayerTensor::MlaOutputDown),
        &[groups * output_rank, group_width],
    )?;
    let mut grouped = vec![0.0; block_size * groups * output_rank];
    for token in 0..block_size {
        for group in 0..groups {
            let source = &attended[token * heads * head_dim + group * group_width
                ..token * heads * head_dim + (group + 1) * group_width];
            for rank in 0..output_rank {
                let weight = &output_down[(group * output_rank + rank) * group_width
                    ..(group * output_rank + rank + 1) * group_width];
                grouped[(token * groups + group) * output_rank + rank] =
                    memra_gguf::dsv4_forward::dot(source, weight);
            }
        }
    }
    Ok(matmul(
        &grouped,
        block_size,
        groups * output_rank,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::MlaOutput),
            &[hidden, groups * output_rank],
        )?,
        hidden,
    ))
}

fn hyper_topology(
    plan: &ModelPlan,
) -> Result<Option<(usize, f32, u32, HcCollapse)>, ReferenceError> {
    let topology = plan.layers.iter().find_map(|layer| match layer.residual {
        ResidualTopology::HyperConnections {
            streams,
            epsilon,
            sinkhorn_iterations,
            collapse,
        } => Some((streams as usize, epsilon, sinkhorn_iterations, collapse)),
        _ => None,
    });
    let Some(topology) = topology else {
        return Ok(None);
    };
    if topology.0 == 0 || topology.1 <= 0.0 || topology.2 == 0 {
        return Err(ReferenceError::InvalidPlan {
            layer: None,
            reason: "HyperConnections require streams, epsilon, and Sinkhorn iterations",
        });
    }
    for layer in &plan.layers {
        if layer.residual
            != (ResidualTopology::HyperConnections {
                streams: topology.0 as u32,
                epsilon: topology.1,
                sinkhorn_iterations: topology.2,
                collapse: topology.3,
            })
        {
            return Err(ReferenceError::InvalidPlan {
                layer: Some(layer.index),
                reason: "HyperConnections topology must be consistent across the trunk",
            });
        }
    }
    Ok(Some(topology))
}

fn collapse_hyper_head(
    weights: &ReferenceWeights,
    x: &[f32],
    tokens: usize,
    streams: usize,
    hidden: usize,
    plan: &ModelPlan,
    epsilon: f32,
) -> Result<Vec<f32>, ReferenceError> {
    let set = memra_gguf::dsv4_forward::HcSet {
        rows: streams,
        fn_w: tensor(
            weights,
            &TensorId::HyperHeadFunction,
            &[streams, streams * hidden],
        )?
        .to_vec(),
        base: tensor(weights, &TensorId::HyperHeadBase, &[streams])?.to_vec(),
        scale: tensor(weights, &TensorId::HyperHeadScale, &[1])?.to_vec(),
    };
    Ok(memra_gguf::dsv4_forward::hc_head(
        x,
        tokens,
        streams,
        hidden,
        &set,
        plan.output_norm.epsilon,
        epsilon,
    ))
}

fn apply_logits_transforms(logits: &mut [f32], vocab: usize, transforms: &[LogitsTransform]) {
    for transform in transforms {
        match transform {
            LogitsTransform::Softcap(cap) => {
                for value in logits.iter_mut() {
                    *value = *cap * (*value / *cap).tanh();
                }
            }
            LogitsTransform::SuppressTokens(ids) => {
                for row in logits.chunks_exact_mut(vocab) {
                    for &id in ids {
                        if let Some(value) = row.get_mut(id as usize) {
                            *value = f32::NEG_INFINITY;
                        }
                    }
                }
            }
        }
    }
}

fn execute_layer(
    layer: &memra_gguf::model_plan::LayerPlan,
    weights: &ReferenceWeights,
    input: &[f32],
    token_ids: &[u32],
    tokens: usize,
    hidden: usize,
    vocab: usize,
) -> Result<(Vec<f32>, ReferenceLayerState), ReferenceError> {
    if let ResidualTopology::HyperConnections {
        streams,
        epsilon,
        sinkhorn_iterations,
        // The collapse knob only applies at model exit; per-layer mixing is identical.
        collapse: _,
    } = layer.residual
    {
        return execute_hyper_layer(
            layer,
            weights,
            input,
            token_ids,
            tokens,
            hidden,
            vocab,
            streams as usize,
            epsilon,
            sinkhorn_iterations,
        );
    }
    if let ResidualTopology::Gemma {
        parallel_moe: Some(parallel),
        ..
    } = layer.residual
    {
        return execute_gemma_parallel_moe_layer(layer, parallel, weights, input, tokens, hidden);
    }
    if let ResidualTopology::Gemma {
        parallel_moe: None, ..
    } = layer.residual
    {
        return execute_gemma_dense_layer(layer, weights, input, tokens, hidden);
    }
    if layer.residual != ResidualTopology::Serial {
        return Err(ReferenceError::UnsupportedOperation {
            layer: Some(layer.index),
            operation: "non-serial residual",
        });
    }
    let pre_attn = rms_norm(
        input,
        tokens,
        hidden,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::PreAttentionNorm),
            &[hidden],
        )?,
        layer.pre_attention_norm.epsilon,
    );
    let (attention, layer_state) = match &layer.attention {
        AttentionPlan::Full(attention) => full_attention(
            layer.index,
            attention,
            None,
            layer.pre_attention_norm.epsilon,
            weights,
            &pre_attn,
            tokens,
            hidden,
        )?,
        AttentionPlan::SlidingWindow { attention, window } => full_attention(
            layer.index,
            attention,
            Some(*window as usize),
            layer.pre_attention_norm.epsilon,
            weights,
            &pre_attn,
            tokens,
            hidden,
        )?,
        AttentionPlan::Mla(mla) => mla_attention(
            layer.index,
            mla,
            layer.pre_attention_norm.epsilon,
            weights,
            &pre_attn,
            tokens,
            hidden,
        )?,
        AttentionPlan::GatedDeltaNet(gdn) => gated_delta_net(
            layer.index,
            gdn,
            layer.pre_attention_norm.epsilon,
            weights,
            &pre_attn,
            tokens,
            hidden,
        )?,
        AttentionPlan::KimiDeltaNet(kda) => kimi_delta_net(
            layer.index,
            kda,
            layer.pre_attention_norm.epsilon,
            weights,
            &pre_attn,
            tokens,
            hidden,
        )?,
    };
    let mut output = input.to_vec();
    add_in_place(&mut output, &attention);
    let pre_mlp = rms_norm(
        &output,
        tokens,
        hidden,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::PreMlpNorm),
            &[hidden],
        )?,
        layer.pre_mlp_norm.epsilon,
    );
    let mlp = match &layer.mlp {
        MlpPlan::Dense(mlp) => dense_mlp(layer.index, mlp, weights, &pre_mlp, tokens, hidden)?,
        MlpPlan::Moe(moe) => moe_mlp(
            layer.index,
            moe,
            weights,
            &pre_mlp,
            token_ids,
            tokens,
            hidden,
            vocab,
        )?,
    };
    add_in_place(&mut output, &mlp);
    Ok((output, layer_state))
}

fn execute_gemma_parallel_moe_layer(
    layer: &memra_gguf::model_plan::LayerPlan,
    parallel: memra_gguf::model_plan::GemmaParallelMoePlan,
    weights: &ReferenceWeights,
    input: &[f32],
    tokens: usize,
    hidden: usize,
) -> Result<(Vec<f32>, ReferenceLayerState), ReferenceError> {
    let ResidualTopology::Gemma {
        post_attention_norm,
        post_mlp_norm,
        layer_scale,
        parallel_moe: Some(_),
    } = layer.residual
    else {
        unreachable!()
    };
    let pre_attention = rms_norm(
        input,
        tokens,
        hidden,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::PreAttentionNorm),
            &[hidden],
        )?,
        layer.pre_attention_norm.epsilon,
    );
    let (attention, state) = match &layer.attention {
        AttentionPlan::Full(attention) => full_attention(
            layer.index,
            attention,
            None,
            layer.pre_attention_norm.epsilon,
            weights,
            &pre_attention,
            tokens,
            hidden,
        )?,
        AttentionPlan::SlidingWindow { attention, window } => full_attention(
            layer.index,
            attention,
            Some(*window as usize),
            layer.pre_attention_norm.epsilon,
            weights,
            &pre_attention,
            tokens,
            hidden,
        )?,
        _ => {
            return Err(ReferenceError::UnsupportedOperation {
                layer: Some(layer.index),
                operation: "gemma parallel MoE non-softmax attention",
            });
        }
    };
    let attention = rms_norm(
        &attention,
        tokens,
        hidden,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::PostAttentionNorm),
            &[hidden],
        )?,
        post_attention_norm.epsilon,
    );
    let mut attention_residual = input.to_vec();
    add_in_place(&mut attention_residual, &attention);

    let MlpPlan::Moe(moe) = &layer.mlp else {
        return Err(ReferenceError::InvalidPlan {
            layer: Some(layer.index),
            reason: "gemma parallel MoE residual requires an MoE plan",
        });
    };
    let shared_plan = moe.shared.as_ref().ok_or(ReferenceError::InvalidPlan {
        layer: Some(layer.index),
        reason: "gemma parallel MoE requires a shared MLP branch",
    })?;
    let shared_input = rms_norm(
        &attention_residual,
        tokens,
        hidden,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::PreMlpNorm),
            &[hidden],
        )?,
        layer.pre_mlp_norm.epsilon,
    );
    let shared_intermediate = shared_plan.intermediate_size as usize;
    let shared_gate = linear(
        &shared_input,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::SharedMlpGate),
            &[shared_intermediate, hidden],
        )?,
        tokens,
        hidden,
        shared_intermediate,
    );
    let shared_up = linear(
        &shared_input,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::SharedMlpUp),
            &[shared_intermediate, hidden],
        )?,
        tokens,
        hidden,
        shared_intermediate,
    );
    let mut shared_activated = vec![0.0; shared_gate.len()];
    for index in 0..shared_activated.len() {
        shared_activated[index] = activate_pair(
            &moe.activation,
            shared_gate[index],
            shared_up[index],
            layer.index,
        )?;
    }
    let shared = linear(
        &shared_activated,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::SharedMlpDown),
            &[hidden, shared_intermediate],
        )?,
        tokens,
        shared_intermediate,
        hidden,
    );
    let shared = rms_norm(
        &shared,
        tokens,
        hidden,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::PostSharedMlpNorm),
            &[hidden],
        )?,
        parallel.shared_post_norm.epsilon,
    );

    let routed_input = rms_norm(
        &attention_residual,
        tokens,
        hidden,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::PreRoutedMlpNorm),
            &[hidden],
        )?,
        parallel.routed_pre_norm.epsilon,
    );
    let router_scale = tensor(
        weights,
        &layer_id(layer.index, LayerTensor::MoeRouterScale),
        &[hidden],
    )?;
    let router_weight: Vec<_> = router_scale
        .iter()
        .map(|value| *value / (hidden as f32).sqrt())
        .collect();
    let router_input = rms_norm(
        &attention_residual,
        tokens,
        hidden,
        &router_weight,
        layer.pre_mlp_norm.epsilon,
    );
    let experts = moe.expert_count as usize;
    let selected = moe.experts_per_token as usize;
    let intermediate = moe.expert_intermediate_size as usize;
    let router_logits = linear(
        &router_input,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::MoeRouter),
            &[experts, hidden],
        )?,
        tokens,
        hidden,
        experts,
    );
    let gate_up = tensor(
        weights,
        &layer_id(layer.index, LayerTensor::MoeExpertGateUpBank),
        &[experts, 2 * intermediate, hidden],
    )?;
    let down = tensor(
        weights,
        &layer_id(layer.index, LayerTensor::MoeExpertDownBank),
        &[experts, hidden, intermediate],
    )?;
    let expert_scale = tensor(
        weights,
        &layer_id(layer.index, LayerTensor::MoeExpertOutputScale),
        &[experts],
    )?;
    let mut routed = vec![0.0; tokens * hidden];
    for token in 0..tokens {
        let routes = route_experts(
            &moe.router,
            &router_logits[token * experts..(token + 1) * experts],
            None,
            selected,
            None,
            layer.index,
        )?;
        let row = &routed_input[token * hidden..(token + 1) * hidden];
        for (expert, route_weight) in routes {
            let expert_offset = expert * 2 * intermediate * hidden;
            let mut activated = vec![0.0; intermediate];
            for output in 0..intermediate {
                let gate = memra_gguf::dsv4_forward::dot(
                    row,
                    &gate_up
                        [expert_offset + output * hidden..expert_offset + (output + 1) * hidden],
                );
                let up_offset = expert_offset + (intermediate + output) * hidden;
                let up =
                    memra_gguf::dsv4_forward::dot(row, &gate_up[up_offset..up_offset + hidden]);
                activated[output] = activate_pair(&moe.activation, gate, up, layer.index)?;
            }
            let down_offset = expert * hidden * intermediate;
            for output in 0..hidden {
                routed[token * hidden + output] += route_weight
                    * expert_scale[expert]
                    * memra_gguf::dsv4_forward::dot(
                        &activated,
                        &down[down_offset + output * intermediate
                            ..down_offset + (output + 1) * intermediate],
                    );
            }
        }
    }
    let routed = rms_norm(
        &routed,
        tokens,
        hidden,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::PostRoutedMlpNorm),
            &[hidden],
        )?,
        parallel.routed_post_norm.epsilon,
    );
    let mut combined = shared;
    add_in_place(&mut combined, &routed);
    let combined = rms_norm(
        &combined,
        tokens,
        hidden,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::PostMlpNorm),
            &[hidden],
        )?,
        post_mlp_norm.epsilon,
    );
    add_in_place(&mut attention_residual, &combined);
    let scale = match layer_scale {
        GemmaLayerScale::Learned => tensor(
            weights,
            &layer_id(layer.index, LayerTensor::LayerScale),
            &[1],
        )?[0],
    };
    for value in &mut attention_residual {
        *value *= scale;
    }
    Ok((attention_residual, state))
}

#[allow(clippy::too_many_arguments)]
fn execute_hyper_layer(
    layer: &memra_gguf::model_plan::LayerPlan,
    weights: &ReferenceWeights,
    input: &[f32],
    token_ids: &[u32],
    tokens: usize,
    hidden: usize,
    vocab: usize,
    streams: usize,
    epsilon: f32,
    sinkhorn_iterations: u32,
) -> Result<(Vec<f32>, ReferenceLayerState), ReferenceError> {
    if input.len() != tokens * streams * hidden {
        return Err(ReferenceError::InvalidPlan {
            layer: Some(layer.index),
            reason: "HyperConnections input does not match tokens x streams x hidden",
        });
    }
    let attention_set = hyper_set(
        weights,
        layer.index,
        streams,
        hidden,
        LayerTensor::HyperAttentionFunction,
        LayerTensor::HyperAttentionBase,
        LayerTensor::HyperAttentionScale,
    )?;
    let (attention_input, post, combination) = memra_gguf::dsv4_forward::hc_pre(
        input,
        tokens,
        streams,
        hidden,
        &attention_set,
        sinkhorn_iterations,
        epsilon,
    );
    let attention_input = rms_norm(
        &attention_input,
        tokens,
        hidden,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::PreAttentionNorm),
            &[hidden],
        )?,
        layer.pre_attention_norm.epsilon,
    );
    let (attention, state) = match &layer.attention {
        AttentionPlan::Full(attention) => full_attention(
            layer.index,
            attention,
            None,
            layer.pre_attention_norm.epsilon,
            weights,
            &attention_input,
            tokens,
            hidden,
        )?,
        AttentionPlan::SlidingWindow { attention, window } => full_attention(
            layer.index,
            attention,
            Some(*window as usize),
            layer.pre_attention_norm.epsilon,
            weights,
            &attention_input,
            tokens,
            hidden,
        )?,
        AttentionPlan::Mla(mla) => mla_attention(
            layer.index,
            mla,
            layer.pre_attention_norm.epsilon,
            weights,
            &attention_input,
            tokens,
            hidden,
        )?,
        AttentionPlan::GatedDeltaNet(gdn) => gated_delta_net(
            layer.index,
            gdn,
            layer.pre_attention_norm.epsilon,
            weights,
            &attention_input,
            tokens,
            hidden,
        )?,
        AttentionPlan::KimiDeltaNet(kda) => kimi_delta_net(
            layer.index,
            kda,
            layer.pre_attention_norm.epsilon,
            weights,
            &attention_input,
            tokens,
            hidden,
        )?,
    };
    let attention_residual = memra_gguf::dsv4_forward::hc_post(
        &attention,
        input,
        tokens,
        streams,
        hidden,
        &post,
        &combination,
    );
    if crate::hidden_trace::enabled() {
        let index = layer.index as i64;
        crate::hidden_trace::emit_last_row("mixer", index, tokens, hidden, &attention);
        crate::hidden_trace::emit_last_row(
            "attn",
            index,
            tokens,
            streams * hidden,
            &attention_residual,
        );
    }

    let mlp_set = hyper_set(
        weights,
        layer.index,
        streams,
        hidden,
        LayerTensor::HyperMlpFunction,
        LayerTensor::HyperMlpBase,
        LayerTensor::HyperMlpScale,
    )?;
    let (mlp_input, post, combination) = memra_gguf::dsv4_forward::hc_pre(
        &attention_residual,
        tokens,
        streams,
        hidden,
        &mlp_set,
        sinkhorn_iterations,
        epsilon,
    );
    let mlp_input = rms_norm(
        &mlp_input,
        tokens,
        hidden,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::PreMlpNorm),
            &[hidden],
        )?,
        layer.pre_mlp_norm.epsilon,
    );
    let mlp = match &layer.mlp {
        MlpPlan::Dense(mlp) => dense_mlp(layer.index, mlp, weights, &mlp_input, tokens, hidden)?,
        MlpPlan::Moe(moe) => moe_mlp(
            layer.index,
            moe,
            weights,
            &mlp_input,
            token_ids,
            tokens,
            hidden,
            vocab,
        )?,
    };
    let output = memra_gguf::dsv4_forward::hc_post(
        &mlp,
        &attention_residual,
        tokens,
        streams,
        hidden,
        &post,
        &combination,
    );
    if crate::hidden_trace::enabled() {
        let index = layer.index as i64;
        crate::hidden_trace::emit_last_row("ffn", index, tokens, hidden, &mlp);
        crate::hidden_trace::emit_last_row("layer", index, tokens, streams * hidden, &output);
    }
    Ok((output, state))
}

#[allow(clippy::too_many_arguments)]
fn hyper_set(
    weights: &ReferenceWeights,
    layer: u32,
    streams: usize,
    hidden: usize,
    function: LayerTensor,
    base: LayerTensor,
    scale: LayerTensor,
) -> Result<memra_gguf::dsv4_forward::HcSet, ReferenceError> {
    let rows = (2 + streams) * streams;
    Ok(memra_gguf::dsv4_forward::HcSet {
        rows,
        fn_w: tensor(
            weights,
            &layer_id(layer, function),
            &[rows, streams * hidden],
        )?
        .to_vec(),
        base: tensor(weights, &layer_id(layer, base), &[rows])?.to_vec(),
        scale: tensor(weights, &layer_id(layer, scale), &[3])?.to_vec(),
    })
}

fn execute_gemma_dense_layer(
    layer: &memra_gguf::model_plan::LayerPlan,
    weights: &ReferenceWeights,
    input: &[f32],
    tokens: usize,
    hidden: usize,
) -> Result<(Vec<f32>, ReferenceLayerState), ReferenceError> {
    let ResidualTopology::Gemma {
        post_attention_norm,
        post_mlp_norm,
        layer_scale,
        parallel_moe: None,
    } = layer.residual
    else {
        return Err(ReferenceError::UnsupportedOperation {
            layer: Some(layer.index),
            operation: "gemma parallel MoE residual",
        });
    };
    let pre_attn = rms_norm(
        input,
        tokens,
        hidden,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::PreAttentionNorm),
            &[hidden],
        )?,
        layer.pre_attention_norm.epsilon,
    );
    let (attention, state) = match &layer.attention {
        AttentionPlan::Full(attention) => full_attention(
            layer.index,
            attention,
            None,
            layer.pre_attention_norm.epsilon,
            weights,
            &pre_attn,
            tokens,
            hidden,
        )?,
        AttentionPlan::SlidingWindow { attention, window } => full_attention(
            layer.index,
            attention,
            Some(*window as usize),
            layer.pre_attention_norm.epsilon,
            weights,
            &pre_attn,
            tokens,
            hidden,
        )?,
        _ => {
            return Err(ReferenceError::UnsupportedOperation {
                layer: Some(layer.index),
                operation: "gemma non-softmax attention",
            });
        }
    };
    let post_attention = rms_norm(
        &attention,
        tokens,
        hidden,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::PostAttentionNorm),
            &[hidden],
        )?,
        post_attention_norm.epsilon,
    );
    let mut attention_residual = input.to_vec();
    add_in_place(&mut attention_residual, &post_attention);
    let pre_mlp = rms_norm(
        &attention_residual,
        tokens,
        hidden,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::PreMlpNorm),
            &[hidden],
        )?,
        layer.pre_mlp_norm.epsilon,
    );
    let MlpPlan::Dense(mlp) = &layer.mlp else {
        return Err(ReferenceError::UnsupportedOperation {
            layer: Some(layer.index),
            operation: "gemma parallel MoE residual",
        });
    };
    let mlp = dense_mlp(layer.index, mlp, weights, &pre_mlp, tokens, hidden)?;
    let mlp = rms_norm(
        &mlp,
        tokens,
        hidden,
        tensor(
            weights,
            &layer_id(layer.index, LayerTensor::PostMlpNorm),
            &[hidden],
        )?,
        post_mlp_norm.epsilon,
    );
    let scale = match layer_scale {
        GemmaLayerScale::Learned => tensor(
            weights,
            &layer_id(layer.index, LayerTensor::LayerScale),
            &[1],
        )?[0],
    };
    let mut output = attention_residual;
    add_in_place(&mut output, &mlp);
    for value in &mut output {
        *value *= scale;
    }
    Ok((output, state))
}

#[allow(clippy::too_many_arguments)]
fn execute_mtp(
    plan: &ModelPlan,
    weights: &ReferenceWeights,
    token_ids: &[u32],
    embedding: &[f32],
    trunk_hidden: &[f32],
    tokens: usize,
    hidden: usize,
    vocab: usize,
    model_output: &[f32],
) -> Result<Vec<ReferenceMtpOutput>, ReferenceError> {
    if plan.mtp_blocks.is_empty() {
        return Ok(Vec::new());
    }
    if trunk_hidden.len() != tokens * hidden {
        return Err(ReferenceError::UnsupportedOperation {
            layer: None,
            operation: "HyperConnections MTP fusion",
        });
    }
    let mut embedded = vec![0.0; tokens * hidden];
    for (position, &token) in token_ids.iter().enumerate() {
        let token = token as usize;
        embedded[position * hidden..(position + 1) * hidden]
            .copy_from_slice(&embedding[token * hidden..(token + 1) * hidden]);
    }
    let mut source_hidden = trunk_hidden.to_vec();
    let mut outputs = Vec::with_capacity(plan.mtp_blocks.len());
    for block in &plan.mtp_blocks {
        let embedding_norm = rms_norm(
            &embedded,
            tokens,
            hidden,
            tensor(
                weights,
                &TensorId::Mtp {
                    depth: block.depth,
                    tensor: MtpTensor::EmbeddingNorm,
                },
                &[hidden],
            )?,
            block.input.embedding_norm.epsilon,
        );
        let hidden_norm = rms_norm(
            &source_hidden,
            tokens,
            hidden,
            tensor(
                weights,
                &TensorId::Mtp {
                    depth: block.depth,
                    tensor: MtpTensor::HiddenNorm,
                },
                &[hidden],
            )?,
            block.input.hidden_norm.epsilon,
        );
        let mut concatenated = vec![0.0; tokens * 2 * hidden];
        for token in 0..tokens {
            concatenated[token * 2 * hidden..token * 2 * hidden + hidden]
                .copy_from_slice(&embedding_norm[token * hidden..(token + 1) * hidden]);
            concatenated[token * 2 * hidden + hidden..(token + 1) * 2 * hidden]
                .copy_from_slice(&hidden_norm[token * hidden..(token + 1) * hidden]);
        }
        let fused = linear(
            &concatenated,
            tensor(
                weights,
                &TensorId::Mtp {
                    depth: block.depth,
                    tensor: MtpTensor::FusionProjection,
                },
                &[hidden, 2 * hidden],
            )?,
            tokens,
            2 * hidden,
            hidden,
        );
        let (hidden_next, state) = execute_layer(
            &block.layer,
            weights,
            &fused,
            token_ids,
            tokens,
            hidden,
            vocab,
        )?;
        let norm_id = TensorId::Mtp {
            depth: block.depth,
            tensor: MtpTensor::OutputNorm,
        };
        let norm = match weights.get(&norm_id) {
            Some(tensor) => tensor_checked(&norm_id, tensor, &[hidden])?,
            None => tensor(weights, &TensorId::OutputNorm, &[hidden])?,
        };
        let final_hidden = rms_norm(&hidden_next, tokens, hidden, norm, plan.output_norm.epsilon);
        let head_id = TensorId::Mtp {
            depth: block.depth,
            tensor: MtpTensor::OutputProjection,
        };
        let head = match weights.get(&head_id) {
            Some(tensor) => tensor_checked(&head_id, tensor, &[vocab, hidden])?,
            None => model_output,
        };
        let mut logits = linear(&final_hidden, head, tokens, hidden, vocab);
        apply_logits_transforms(&mut logits, vocab, &plan.logits);
        source_hidden = hidden_next.clone();
        outputs.push(ReferenceMtpOutput {
            depth: block.depth,
            logits,
            hidden: hidden_next,
            state,
        });
    }
    Ok(outputs)
}

fn mla_attention(
    layer: u32,
    plan: &memra_gguf::model_plan::MlaAttentionPlan,
    epsilon: f32,
    weights: &ReferenceWeights,
    x: &[f32],
    tokens: usize,
    hidden: usize,
) -> Result<(Vec<f32>, ReferenceLayerState), ReferenceError> {
    if let memra_gguf::model_plan::MlaAttentionPlan::CompressedKv { .. } = plan {
        return compressed_mla_attention(layer, plan, epsilon, weights, x, tokens, hidden);
    }
    let memra_gguf::model_plan::MlaAttentionPlan::LatentKv {
        query_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_head_dim,
        rope_head_dim,
        value_head_dim,
        rope,
        sparse_index,
    } = plan.clone()
    else {
        return Err(ReferenceError::UnsupportedOperation {
            layer: Some(layer),
            operation: "compressed-KV MLA",
        });
    };
    // Per-token indexers execute only through full-selection equivalence; the k-pool
    // indexer (glm5_next) selects for real and is scored after q_resid exists.
    let plain_sparse_top_k = match &sparse_index {
        memra_gguf::model_plan::SparseIndexPlan::None
        | memra_gguf::model_plan::SparseIndexPlan::Own { kpool: Some(_), .. } => None,
        memra_gguf::model_plan::SparseIndexPlan::Own {
            top_k, kpool: None, ..
        }
        | memra_gguf::model_plan::SparseIndexPlan::SharedFromPrevious { top_k } => {
            Some(*top_k as usize)
        }
    };
    if plain_sparse_top_k.is_some_and(|top_k| tokens > top_k) {
        return Err(ReferenceError::UnsupportedOperation {
            layer: Some(layer),
            operation: "sparse MLA selection beyond full-selection equivalence",
        });
    }
    let heads = query_heads as usize;
    let q_rank = q_lora_rank as usize;
    let kv_rank = kv_lora_rank as usize;
    let qk_dim = qk_head_dim as usize;
    let rope_dim = rope_head_dim as usize;
    let nope_dim = qk_dim - rope_dim;
    let value_dim = value_head_dim as usize;
    let latent_dim = kv_rank + rope_dim;

    let q_down = linear(
        x,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::MlaQueryDown),
            &[q_rank, hidden],
        )?,
        tokens,
        hidden,
        q_rank,
    );
    let q_down = rms_norm(
        &q_down,
        tokens,
        q_rank,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::MlaQueryDownNorm),
            &[q_rank],
        )?,
        epsilon,
    );
    // q_down is q_resid = q_a_layernorm(q_a_proj(x)): it feeds both the MLA query
    // up-projection and the k-pool indexer.
    let allowed_mask = match &sparse_index {
        memra_gguf::model_plan::SparseIndexPlan::Own {
            heads: index_heads,
            head_dim: index_dim,
            top_k,
            kpool: Some(kpool),
        } => {
            let allowed = kpool_allowed_tokens(
                layer,
                *index_heads as usize,
                *index_dim as usize,
                *top_k as usize,
                kpool,
                weights,
                x,
                &q_down,
                tokens,
                hidden,
                q_rank,
            )?;
            let mut mask = vec![false; tokens * tokens];
            for (token, sources) in allowed.iter().enumerate() {
                for &source in sources {
                    mask[token * tokens + source] = true;
                }
            }
            Some(mask)
        }
        _ => None,
    };
    let query = linear(
        &q_down,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::MlaQueryUp),
            &[heads * qk_dim, q_rank],
        )?,
        tokens,
        q_rank,
        heads * qk_dim,
    );
    let latent_raw = linear(
        x,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::MlaKvDown),
            &[latent_dim, hidden],
        )?,
        tokens,
        hidden,
        latent_dim,
    );
    let kv_norm = tensor(
        weights,
        &layer_id(layer, LayerTensor::MlaKvDownNorm),
        &[kv_rank],
    )?;
    let mut latent = latent_raw;
    for token in 0..tokens {
        let offset = token * latent_dim;
        let normalized = rms_norm(
            &latent[offset..offset + kv_rank],
            1,
            kv_rank,
            kv_norm,
            epsilon,
        );
        latent[offset..offset + kv_rank].copy_from_slice(&normalized);
    }

    let mut query_nope = vec![0.0; tokens * heads * nope_dim];
    let mut query_rope = vec![0.0; tokens * heads * rope_dim];
    for token in 0..tokens {
        for head in 0..heads {
            let source = (token * heads + head) * qk_dim;
            let nope_target = (token * heads + head) * nope_dim;
            let rope_target = (token * heads + head) * rope_dim;
            query_nope[nope_target..nope_target + nope_dim]
                .copy_from_slice(&query[source..source + nope_dim]);
            query_rope[rope_target..rope_target + rope_dim]
                .copy_from_slice(&query[source + nope_dim..source + qk_dim]);
        }
    }
    let rope_factors = rope_factor_values(&rope, weights)?;
    apply_rope(
        &mut query_rope,
        tokens,
        heads,
        rope_dim,
        rope.dimensions as usize,
        rope.base,
        rope_factors.as_deref(),
    );
    let mut key_rope = vec![0.0; tokens * rope_dim];
    for token in 0..tokens {
        key_rope[token * rope_dim..(token + 1) * rope_dim]
            .copy_from_slice(&latent[token * latent_dim + kv_rank..(token + 1) * latent_dim]);
    }
    apply_rope(
        &mut key_rope,
        tokens,
        1,
        rope_dim,
        rope.dimensions as usize,
        rope.base,
        rope_factors.as_deref(),
    );
    for token in 0..tokens {
        latent[token * latent_dim + kv_rank..(token + 1) * latent_dim]
            .copy_from_slice(&key_rope[token * rope_dim..(token + 1) * rope_dim]);
    }

    // Contract layout: [head][kv_rank][nope] (see `deterministic_fixture`).
    let key_weight = tensor(
        weights,
        &layer_id(layer, LayerTensor::MlaKeyUp),
        &[heads, kv_rank, nope_dim],
    )?;
    let value_weight = tensor(
        weights,
        &layer_id(layer, LayerTensor::MlaValueUp),
        &[heads, value_dim, kv_rank],
    )?;
    let mut key_nope = vec![0.0; tokens * heads * nope_dim];
    let mut value = vec![0.0; tokens * heads * value_dim];
    for token in 0..tokens {
        let latent_row = &latent[token * latent_dim..token * latent_dim + kv_rank];
        for head in 0..heads {
            for out in 0..nope_dim {
                for rank in 0..kv_rank {
                    key_nope[(token * heads + head) * nope_dim + out] +=
                        latent_row[rank] * key_weight[(head * kv_rank + rank) * nope_dim + out];
                }
            }
            for out in 0..value_dim {
                for rank in 0..kv_rank {
                    value[(token * heads + head) * value_dim + out] +=
                        latent_row[rank] * value_weight[(head * value_dim + out) * kv_rank + rank];
                }
            }
        }
    }
    let mut attended = vec![0.0; tokens * heads * value_dim];
    let scale = 1.0 / (qk_dim as f32).sqrt();
    for token in 0..tokens {
        for head in 0..heads {
            let mut scores = Vec::with_capacity(token + 1);
            for source in 0..=token {
                // The indexer's allowed set masks keys exactly like the eager
                // additive -inf mask built from topk_indices.
                if allowed_mask
                    .as_ref()
                    .is_some_and(|mask| !mask[token * tokens + source])
                {
                    scores.push(f32::NEG_INFINITY);
                    continue;
                }
                let mut score = 0.0;
                for dim in 0..nope_dim {
                    score += query_nope[(token * heads + head) * nope_dim + dim]
                        * key_nope[(source * heads + head) * nope_dim + dim];
                }
                for dim in 0..rope_dim {
                    score += query_rope[(token * heads + head) * rope_dim + dim]
                        * key_rope[source * rope_dim + dim];
                }
                scores.push(score * scale);
            }
            softmax_in_place(&mut scores);
            for (source, probability) in scores.into_iter().enumerate() {
                for dim in 0..value_dim {
                    attended[(token * heads + head) * value_dim + dim] +=
                        probability * value[(source * heads + head) * value_dim + dim];
                }
            }
        }
    }
    let output = linear(
        &attended,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::MlaOutput),
            &[hidden, heads * value_dim],
        )?,
        tokens,
        heads * value_dim,
        hidden,
    );
    Ok((
        output,
        ReferenceLayerState::LatentKv {
            rows: latent,
            tokens,
            width: latent_dim,
        },
    ))
}

/// K-pool compressed indexer selection (Glm5NextTextIndexer.forward), single-sequence
/// causal case: every token is a valid key, so pooling starts at index 0 and only
/// causality masks candidates. Returns the allowed source-token set per query,
/// sorted ascending.
///
/// PUBLIC because it is the CUDA k-pool indexer's oracle: `memra-engine`'s
/// `tests/glm5_kpool_indexer_gpu.rs` compares the device's selected index sets against this
/// function on identical inputs. Its scope line above is part of the contract — a padded or
/// batched caller is outside it.
#[allow(clippy::too_many_arguments)]
pub fn kpool_allowed_tokens(
    layer: u32,
    index_heads: usize,
    index_dim: usize,
    top_k: usize,
    kpool: &memra_gguf::model_plan::KpoolPlan,
    weights: &ReferenceWeights,
    x: &[f32],
    q_resid: &[f32],
    tokens: usize,
    hidden: usize,
    q_rank: usize,
) -> Result<Vec<Vec<usize>>, ReferenceError> {
    let pool = kpool.pool as usize;
    if index_heads == 0 || index_dim == 0 || pool == 0 {
        return Err(ReferenceError::InvalidPlan {
            layer: Some(layer),
            reason: "k-pool sparse index requires positive heads, head_dim, and pool",
        });
    }
    let q = linear(
        q_resid,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::SparseQuery),
            &[index_heads * index_dim, q_rank],
        )?,
        tokens,
        q_rank,
        index_heads * index_dim,
    );
    let key = layer_norm(
        &linear(
            x,
            tensor(
                weights,
                &layer_id(layer, LayerTensor::SparseKey),
                &[index_dim, hidden],
            )?,
            tokens,
            hidden,
            index_dim,
        ),
        tokens,
        index_dim,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::SparseKeyNorm),
            &[index_dim],
        )?,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::SparseKeyNormBias),
            &[index_dim],
        )?,
    );
    let gate_scores = linear(
        x,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::SparseCompressorGate),
            &[index_dim, hidden],
        )?,
        tokens,
        hidden,
        index_dim,
    );
    let ape = tensor(
        weights,
        &layer_id(layer, LayerTensor::SparseCompressorPosition),
        &[pool, index_dim],
    )?;
    // Only COMPLETE pools are candidates; the incomplete tail never scores. Each
    // channel takes its own softmax over the pool members (gate score + APE).
    let pools = tokens / pool;
    let mut pool_keys = vec![0.0f32; pools * index_dim];
    for pool_index in 0..pools {
        for channel in 0..index_dim {
            let mut logits = Vec::with_capacity(pool);
            for slot in 0..pool {
                logits.push(
                    gate_scores[(pool_index * pool + slot) * index_dim + channel]
                        + ape[slot * index_dim + channel],
                );
            }
            softmax_in_place(&mut logits);
            let mut pooled = 0.0;
            for slot in 0..pool {
                pooled += logits[slot] * key[(pool_index * pool + slot) * index_dim + channel];
            }
            pool_keys[pool_index * index_dim + channel] = pooled;
        }
    }
    let mut head_weights = linear(
        x,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::SparseProjection),
            &[index_heads, hidden],
        )?,
        tokens,
        hidden,
        index_heads,
    );
    let head_scale = (index_heads as f32).powf(-0.5);
    for value in &mut head_weights {
        *value *= head_scale;
    }
    // Same scale convention as the per-token DSA indexer: relu(q . k * hd^-0.5).
    let softmax_scale = (index_dim as f32).powf(-0.5);
    let select_k = (top_k / pool).min(pools);
    let mut allowed = Vec::with_capacity(tokens);
    for token in 0..tokens {
        // A pool is selectable only when its final token index is <= the query.
        let visible_pools = ((token + 1) / pool).min(pools);
        let mut scored: Vec<(usize, f32)> = (0..visible_pools)
            .map(|pool_index| {
                let mut score = 0.0f32;
                for head in 0..index_heads {
                    let mut dot = 0.0f32;
                    for dim in 0..index_dim {
                        dot += q[(token * index_heads + head) * index_dim + dim]
                            * pool_keys[pool_index * index_dim + dim];
                    }
                    score +=
                        (dot * softmax_scale).max(0.0) * head_weights[token * index_heads + head];
                }
                (pool_index, score)
            })
            .collect();
        scored.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(left.0.cmp(&right.0))
        });
        let mut selected: Vec<usize> = Vec::new();
        for &(pool_index, _) in scored.iter().take(select_k) {
            selected.extend(pool_index * pool..(pool_index + 1) * pool);
        }
        if kpool.always_select_tail {
            // The current incomplete tail: the visible tokens past the last complete
            // visible pool (at most pool - 1 of them), always <= the query index.
            let visible = token + 1;
            let tail = visible % pool;
            selected.extend(visible - tail..visible);
        }
        if selected.is_empty() {
            // always_select_tail=false leaves early queries (before the first
            // complete pool) with no candidates; the reference would emit NaN rows.
            return Err(ReferenceError::InvalidPlan {
                layer: Some(layer),
                reason: "k-pool selection produced an empty candidate set for a query",
            });
        }
        selected.sort_unstable();
        allowed.push(selected);
    }
    Ok(allowed)
}

#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn compressed_mla_attention(
    layer: u32,
    plan: &memra_gguf::model_plan::MlaAttentionPlan,
    epsilon: f32,
    weights: &ReferenceWeights,
    x: &[f32],
    tokens: usize,
    hidden: usize,
) -> Result<(Vec<f32>, ReferenceLayerState), ReferenceError> {
    use memra_gguf::dsv4_forward::{
        ActQuantVariant, IndexerW, apply_rope as apply_dsv4_rope, matmul, precompute_freqs_cis,
        rmsnorm,
    };
    use memra_gguf::model_plan::{MlaAttentionPlan, RopeFactors, SparseIndexPlan};

    let MlaAttentionPlan::CompressedKv {
        query_heads,
        q_lora_rank,
        latent_head_dim,
        rope_head_dim,
        output_lora_rank,
        output_groups,
        window,
        rope,
        compressor,
        sparse_index,
    } = plan
    else {
        unreachable!()
    };
    let heads = *query_heads as usize;
    let q_rank = *q_lora_rank as usize;
    let head_dim = *latent_head_dim as usize;
    let rope_dim = *rope_head_dim as usize;
    let output_rank = *output_lora_rank as usize;
    let groups = *output_groups as usize;
    let window = *window as usize;
    if heads == 0
        || q_rank == 0
        || head_dim == 0
        || rope_dim == 0
        || rope_dim > head_dim
        || !(head_dim - rope_dim).is_multiple_of(64)
        || groups == 0
        || heads % groups != 0
        || window == 0
    {
        return Err(ReferenceError::InvalidPlan {
            layer: Some(layer),
            reason: "compressed attention has invalid reference geometry",
        });
    }
    let (original_context, factor, beta_fast, beta_slow) = match rope.factors {
        RopeFactors::None => (0, 1.0, 32.0, 1.0),
        RopeFactors::Yarn {
            factor,
            original_context,
            beta_fast,
            beta_slow,
        } => (original_context, factor, beta_fast, beta_slow),
        _ => {
            return Err(ReferenceError::InvalidPlan {
                layer: Some(layer),
                reason: "compressed attention requires plain or YaRN RoPE",
            });
        }
    };
    let frequencies = precompute_freqs_cis(
        rope_dim,
        tokens.max(1),
        original_context,
        rope.base,
        factor,
        beta_fast,
        beta_slow,
    );
    let positions: Vec<usize> = (0..tokens).collect();

    let query_low_rank = rmsnorm(
        &matmul(
            x,
            tokens,
            hidden,
            tensor(
                weights,
                &layer_id(layer, LayerTensor::MlaQueryDown),
                &[q_rank, hidden],
            )?,
            q_rank,
        ),
        tensor(
            weights,
            &layer_id(layer, LayerTensor::MlaQueryDownNorm),
            &[q_rank],
        )?,
        epsilon,
    );
    let mut query = matmul(
        &query_low_rank,
        tokens,
        q_rank,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::MlaQueryUp),
            &[heads * head_dim, q_rank],
        )?,
        heads * head_dim,
    );
    for head in query.chunks_exact_mut(head_dim) {
        let mean_square = head
            .iter()
            .map(|value| (*value as f64) * (*value as f64))
            .sum::<f64>()
            / head_dim as f64;
        let scale = 1.0 / (mean_square as f32 + epsilon).sqrt();
        for value in head {
            *value *= scale;
        }
    }
    apply_dsv4_rope(
        &mut query,
        tokens,
        heads,
        head_dim,
        rope_dim,
        &frequencies,
        &positions,
        false,
    );

    let mut key_value = rmsnorm(
        &matmul(
            x,
            tokens,
            hidden,
            tensor(
                weights,
                &layer_id(layer, LayerTensor::MlaKvDown),
                &[head_dim, hidden],
            )?,
            head_dim,
        ),
        tensor(
            weights,
            &layer_id(layer, LayerTensor::MlaKvDownNorm),
            &[head_dim],
        )?,
        epsilon,
    );
    apply_dsv4_rope(
        &mut key_value,
        tokens,
        1,
        head_dim,
        rope_dim,
        &frequencies,
        &positions,
        false,
    );
    for row in key_value.chunks_exact_mut(head_dim) {
        memra_gguf::dsv4_forward::act_quant(
            &mut row[..head_dim - rope_dim],
            64,
            ActQuantVariant::RefFp8Round,
        );
    }

    let (mut indices, mut slots) = memra_gguf::dsv4_forward::window_topk_idxs(window, tokens);
    let mut key_value_rows = tokens;
    let mut compressed_tokens = 0;
    if let Some(compressor_plan) = compressor {
        let ratio = compressor_plan.ratio as usize;
        let compressor = reference_compressor(
            weights,
            layer,
            hidden,
            head_dim,
            ratio,
            compressor_plan.latent_dim as usize,
            false,
        )?;
        let (compressed_indices, compressed_slots) = match sparse_index {
            SparseIndexPlan::None => {
                memra_gguf::dsv4_forward::compress_topk_idxs(ratio, tokens, tokens)
            }
            SparseIndexPlan::Own {
                heads: index_heads,
                head_dim: index_dim,
                top_k,
                kpool,
            } => {
                // K-pool scoring is a LatentKv (glm5_next) program; dsv4 compiles None.
                if kpool.is_some() {
                    return Err(ReferenceError::UnsupportedOperation {
                        layer: Some(layer),
                        operation: "k-pool sparse index on compressed attention",
                    });
                }
                let index_heads = *index_heads as usize;
                let index_dim = *index_dim as usize;
                if index_dim < rope_dim
                    || !index_dim.is_multiple_of(32)
                    || !index_dim.is_power_of_two()
                {
                    return Err(ReferenceError::InvalidPlan {
                        layer: Some(layer),
                        reason: "compressed sparse index has invalid head geometry",
                    });
                }
                let indexer = IndexerW {
                    wq_b: tensor(
                        weights,
                        &layer_id(layer, LayerTensor::SparseQuery),
                        &[index_heads * index_dim, q_rank],
                    )?
                    .to_vec(),
                    weights_proj: tensor(
                        weights,
                        &layer_id(layer, LayerTensor::SparseProjection),
                        &[index_heads, hidden],
                    )?
                    .to_vec(),
                    compressor: reference_compressor(
                        weights,
                        layer,
                        hidden,
                        index_dim,
                        ratio,
                        2 * index_dim,
                        true,
                    )?,
                    heads: index_heads,
                    hd: index_dim,
                    topk: *top_k as usize,
                };
                let output = indexer.forward(
                    x,
                    &query_low_rank,
                    tokens,
                    hidden,
                    q_rank,
                    tokens,
                    &frequencies,
                    rope_dim,
                    epsilon,
                    ActQuantVariant::RefFp8Round,
                    false,
                );
                (output.idxs, output.slots)
            }
            SparseIndexPlan::SharedFromPrevious { .. } => {
                return Err(ReferenceError::UnsupportedOperation {
                    layer: Some(layer),
                    operation: "shared compressed sparse-index execution",
                });
            }
        };
        if compressed_slots > 0 {
            let mut merged = vec![-1; tokens * (slots + compressed_slots)];
            for token in 0..tokens {
                merged[token * (slots + compressed_slots)
                    ..token * (slots + compressed_slots) + slots]
                    .copy_from_slice(&indices[token * slots..(token + 1) * slots]);
                merged[token * (slots + compressed_slots) + slots
                    ..(token + 1) * (slots + compressed_slots)]
                    .copy_from_slice(
                        &compressed_indices
                            [token * compressed_slots..(token + 1) * compressed_slots],
                    );
            }
            indices = merged;
            slots += compressed_slots;
        }
        if let Some((compressed, count)) = compressor.forward(
            x,
            tokens,
            hidden,
            &frequencies,
            rope_dim,
            epsilon,
            ActQuantVariant::RefFp8Round,
        ) {
            key_value.extend_from_slice(&compressed);
            key_value_rows += count;
            compressed_tokens = count;
        }
    }

    let sink = tensor(
        weights,
        &layer_id(layer, LayerTensor::AttentionSink),
        &[heads],
    )?;
    let attention_scale = (head_dim as f64).powf(-0.5) as f32;
    let mut attended = vec![0.0; tokens * heads * head_dim];
    for token in 0..tokens {
        let selected = &indices[token * slots..(token + 1) * slots];
        memra_gguf::dsv4_decode::sparse_attn_query(
            &query[token * heads * head_dim..(token + 1) * heads * head_dim],
            heads,
            head_dim,
            selected,
            |index| &key_value[index * head_dim..(index + 1) * head_dim],
            sink,
            attention_scale,
            &mut attended[token * heads * head_dim..(token + 1) * heads * head_dim],
        );
    }
    apply_dsv4_rope(
        &mut attended,
        tokens,
        heads,
        head_dim,
        rope_dim,
        &frequencies,
        &positions,
        true,
    );

    let group_width = heads / groups * head_dim;
    let output_down = tensor(
        weights,
        &layer_id(layer, LayerTensor::MlaOutputDown),
        &[groups * output_rank, group_width],
    )?;
    let mut grouped = vec![0.0; tokens * groups * output_rank];
    for token in 0..tokens {
        for group in 0..groups {
            let source = &attended[token * heads * head_dim + group * group_width
                ..token * heads * head_dim + (group + 1) * group_width];
            let group_weight = &output_down
                [group * output_rank * group_width..(group + 1) * output_rank * group_width];
            for rank in 0..output_rank {
                grouped[(token * groups + group) * output_rank + rank] =
                    memra_gguf::dsv4_forward::dot(
                        source,
                        &group_weight[rank * group_width..(rank + 1) * group_width],
                    );
            }
        }
    }
    let output = matmul(
        &grouped,
        tokens,
        groups * output_rank,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::MlaOutput),
            &[hidden, groups * output_rank],
        )?,
        hidden,
    );
    Ok((
        output,
        ReferenceLayerState::CompressedAttention {
            rows: key_value,
            tokens: key_value_rows,
            width: head_dim,
            window,
            compressed_tokens,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn reference_compressor(
    weights: &ReferenceWeights,
    layer: u32,
    hidden: usize,
    output_dim: usize,
    ratio: usize,
    latent: usize,
    sparse: bool,
) -> Result<memra_gguf::dsv4_forward::CompressorW, ReferenceError> {
    let (key_value, gate, norm, position) = if sparse {
        (
            LayerTensor::SparseCompressorKeyValue,
            LayerTensor::SparseCompressorGate,
            LayerTensor::SparseCompressorNorm,
            LayerTensor::SparseCompressorPosition,
        )
    } else {
        (
            LayerTensor::KvCompressorKeyValue,
            LayerTensor::KvCompressorGate,
            LayerTensor::KvCompressorNorm,
            LayerTensor::KvCompressorPosition,
        )
    };
    Ok(memra_gguf::dsv4_forward::CompressorW {
        ratio,
        d: output_dim,
        latent,
        overlap: ratio == 4,
        rotate: sparse,
        wkv: tensor(weights, &layer_id(layer, key_value), &[latent, hidden])?.to_vec(),
        wgate: tensor(weights, &layer_id(layer, gate), &[latent, hidden])?.to_vec(),
        norm_w: tensor(weights, &layer_id(layer, norm), &[output_dim])?.to_vec(),
        ape: tensor(weights, &layer_id(layer, position), &[ratio, latent])?.to_vec(),
    })
}

fn gated_delta_net(
    layer: u32,
    plan: &memra_gguf::model_plan::GatedDeltaNetPlan,
    epsilon: f32,
    weights: &ReferenceWeights,
    x: &[f32],
    tokens: usize,
    hidden: usize,
) -> Result<(Vec<f32>, ReferenceLayerState), ReferenceError> {
    let key_heads = plan.key_heads as usize;
    let value_heads = plan.value_heads as usize;
    let key_dim = plan.key_head_dim as usize;
    let value_dim = plan.value_head_dim as usize;
    let kernel = plan.conv_kernel as usize;
    if key_heads == 0 || value_heads == 0 || key_dim == 0 || value_dim == 0 || kernel == 0 {
        return Err(ReferenceError::InvalidPlan {
            layer: Some(layer),
            reason: "GDN dimensions must be positive",
        });
    }
    let key_width = key_heads * key_dim;
    let value_width = value_heads * value_dim;
    let conv_width = 2 * key_width + value_width;
    let qkv = linear(
        x,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::GdnQkv),
            &[conv_width, hidden],
        )?,
        tokens,
        hidden,
        conv_width,
    );
    let gate = linear(
        x,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::GdnGate),
            &[value_width, hidden],
        )?,
        tokens,
        hidden,
        value_width,
    );
    let beta_raw = linear(
        x,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::GdnBeta),
            &[value_heads, hidden],
        )?,
        tokens,
        hidden,
        value_heads,
    );
    let alpha = linear(
        x,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::GdnAlpha),
            &[value_heads, hidden],
        )?,
        tokens,
        hidden,
        value_heads,
    );
    let conv_weight = tensor(
        weights,
        &layer_id(layer, LayerTensor::GdnConv1d),
        &[conv_width, kernel],
    )?;
    let mut conv = vec![0.0; tokens * conv_width];
    let pad = kernel - 1;
    for token in 0..tokens {
        for channel in 0..conv_width {
            let mut sum = 0.0;
            for tap in 0..kernel {
                let source = token as isize - pad as isize + tap as isize;
                if source >= 0 {
                    sum += qkv[source as usize * conv_width + channel]
                        * conv_weight[channel * kernel + tap];
                }
            }
            conv[token * conv_width + channel] = silu(sum);
        }
    }

    let mut query = vec![0.0; tokens * value_heads * key_dim];
    let mut key = vec![0.0; tokens * value_heads * key_dim];
    let mut value = vec![0.0; tokens * value_width];
    for token in 0..tokens {
        for value_head in 0..value_heads {
            let key_head = value_head % key_heads;
            let q_source = token * conv_width + key_head * key_dim;
            let k_source = token * conv_width + key_width + key_head * key_dim;
            let v_source = token * conv_width + 2 * key_width + value_head * value_dim;
            let q_target = (token * value_heads + value_head) * key_dim;
            let v_target = (token * value_heads + value_head) * value_dim;
            query[q_target..q_target + key_dim]
                .copy_from_slice(&conv[q_source..q_source + key_dim]);
            key[q_target..q_target + key_dim].copy_from_slice(&conv[k_source..k_source + key_dim]);
            value[v_target..v_target + value_dim]
                .copy_from_slice(&conv[v_source..v_source + value_dim]);
        }
    }
    l2_normalize_rows(&mut query, tokens * value_heads, key_dim, epsilon);
    l2_normalize_rows(&mut key, tokens * value_heads, key_dim, epsilon);

    let a = tensor(weights, &layer_id(layer, LayerTensor::GdnA), &[value_heads])?;
    let dt = tensor(
        weights,
        &layer_id(layer, LayerTensor::GdnDtBias),
        &[value_heads],
    )?;
    let mut matrix = vec![0.0; value_heads * value_dim * key_dim];
    let mut mixed = vec![0.0; tokens * value_width];
    let scale = 1.0 / (key_dim as f32).sqrt();
    for token in 0..tokens {
        for head in 0..value_heads {
            let beta = sigmoid(beta_raw[token * value_heads + head]);
            let decay = (a[head] * softplus(alpha[token * value_heads + head] + dt[head])).exp();
            let q_offset = (token * value_heads + head) * key_dim;
            let v_offset = (token * value_heads + head) * value_dim;
            let state_offset = head * value_dim * key_dim;
            let mut next = matrix[state_offset..state_offset + value_dim * key_dim].to_vec();
            for value_index in 0..value_dim {
                let row = state_offset + value_index * key_dim;
                let mut state_key = 0.0;
                for key_index in 0..key_dim {
                    state_key += matrix[row + key_index] * key[q_offset + key_index];
                }
                let delta = (value[v_offset + value_index] - decay * state_key) * beta;
                let mut attended = 0.0;
                for key_index in 0..key_dim {
                    let updated =
                        decay * matrix[row + key_index] + key[q_offset + key_index] * delta;
                    next[value_index * key_dim + key_index] = updated;
                    attended += updated * query[q_offset + key_index];
                }
                mixed[v_offset + value_index] = attended * scale;
            }
            matrix[state_offset..state_offset + value_dim * key_dim].copy_from_slice(&next);
        }
    }

    let norm = tensor(
        weights,
        &layer_id(layer, LayerTensor::GdnNorm),
        &[value_dim],
    )?;
    let normalized = rms_norm(&mixed, tokens * value_heads, value_dim, norm, epsilon);
    let mut gated = normalized;
    for index in 0..gated.len() {
        gated[index] *= silu(gate[index]);
    }
    let output = linear(
        &gated,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::GdnOutput),
            &[hidden, value_width],
        )?,
        tokens,
        value_width,
        hidden,
    );
    let mut conv_state = vec![0.0; conv_width * pad];
    for channel in 0..conv_width {
        for index in 0..pad {
            let source = tokens as isize - pad as isize + index as isize;
            if source >= 0 {
                conv_state[channel * pad + index] = qkv[source as usize * conv_width + channel];
            }
        }
    }
    Ok((
        output,
        ReferenceLayerState::Recurrent {
            conv: conv_state,
            matrix,
            value_heads,
            key_head_dim: key_dim,
            value_head_dim: value_dim,
            conv_width,
        },
    ))
}

/// Kimi Delta Attention (recurrent_kimi_delta_attention + Glm5NextTextLinearAttention),
/// all f32, sequential over tokens. Only the lower-bound forget-gate branch exists:
/// GLM-5.3-Flash always configures `gate_lower_bound`, so the softplus branch of
/// Glm5NextTextForgetGate is dead for this model and deliberately not implemented.
/// GPU-parity seam: run ONE KDA layer's mixer over `x` (`[tokens, hidden]`, already
/// pre-attention-normed) and return its output plus the recurrent state it leaves behind.
///
/// This is the very `kimi_delta_net` the trunk executor dispatches — exposed so
/// `crates/memra-engine/tests/kda_fixture_gpu.rs` can gate the CUDA mixer against the pinned
/// reference without standing up a whole model (glm5_next's residual topology and MLA layers
/// are a different surface, and a mixer gate must not depend on them).
pub fn kimi_delta_net_layer(
    layer: u32,
    plan: &memra_gguf::model_plan::KimiDeltaNetPlan,
    epsilon: f32,
    weights: &ReferenceWeights,
    x: &[f32],
    tokens: usize,
    hidden: usize,
) -> Result<(Vec<f32>, ReferenceLayerState), ReferenceError> {
    kimi_delta_net(layer, plan, epsilon, weights, x, tokens, hidden)
}

fn kimi_delta_net(
    layer: u32,
    plan: &memra_gguf::model_plan::KimiDeltaNetPlan,
    epsilon: f32,
    weights: &ReferenceWeights,
    x: &[f32],
    tokens: usize,
    hidden: usize,
) -> Result<(Vec<f32>, ReferenceLayerState), ReferenceError> {
    let heads = plan.num_heads as usize;
    let head_dim = plan.head_dim as usize;
    let kernel = plan.conv_kernel as usize;
    if heads == 0 || head_dim == 0 || kernel == 0 {
        return Err(ReferenceError::InvalidPlan {
            layer: Some(layer),
            reason: "KDA dimensions must be positive",
        });
    }
    let qkv = heads * head_dim;
    let conv_width = 3 * qkv;
    let project_and_convolve = |projection: LayerTensor,
                                conv: LayerTensor|
     -> Result<(Vec<f32>, Vec<f32>), ReferenceError> {
        let projected = linear(
            x,
            tensor(weights, &layer_id(layer, projection), &[qkv, hidden])?,
            tokens,
            hidden,
            qkv,
        );
        // The checkpoint splits the grouped causal conv into three per-plane
        // convs; applying each to its own plane is the fused conv exactly.
        let conv_weight = tensor(weights, &layer_id(layer, conv), &[qkv, kernel])?;
        let mut convolved = vec![0.0; tokens * qkv];
        for token in 0..tokens {
            for channel in 0..qkv {
                let mut sum = 0.0;
                for tap in 0..kernel {
                    let source = token as isize - (kernel - 1) as isize + tap as isize;
                    if source >= 0 {
                        sum += projected[source as usize * qkv + channel]
                            * conv_weight[channel * kernel + tap];
                    }
                }
                convolved[token * qkv + channel] = silu(sum);
            }
        }
        Ok((projected, convolved))
    };
    let (q_raw, mut query) =
        project_and_convolve(LayerTensor::KdaQuery, LayerTensor::KdaQueryConv)?;
    let (k_raw, mut key) = project_and_convolve(LayerTensor::KdaKey, LayerTensor::KdaKeyConv)?;
    let (v_raw, value) = project_and_convolve(LayerTensor::KdaValue, LayerTensor::KdaValueConv)?;
    // FLA l2norm: x / sqrt(sum(x^2) + 1e-6) — the epsilon sits INSIDE the sqrt and is
    // fixed at 1e-6, independent of the layer epsilon.
    l2_normalize_rows(&mut query, tokens * heads, head_dim, 1e-6);
    l2_normalize_rows(&mut key, tokens * heads, head_dim, 1e-6);
    // Query scale head_dim^-0.5 applies AFTER the l2norm.
    let query_scale = 1.0 / (head_dim as f32).sqrt();
    for entry in &mut query {
        *entry *= query_scale;
    }

    // Forget gate: g = gate_lower_bound * sigmoid(exp(A_log[head]) * (f_b(f_a(x)) + dt_bias)),
    // per channel (dt_bias has width qkv).
    let forget_down = linear(
        x,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::KdaForgetDown),
            &[head_dim, hidden],
        )?,
        tokens,
        hidden,
        head_dim,
    );
    let mut forget = linear(
        &forget_down,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::KdaForgetUp),
            &[qkv, head_dim],
        )?,
        tokens,
        head_dim,
        qkv,
    );
    let dt_bias = tensor(weights, &layer_id(layer, LayerTensor::KdaDtBias), &[qkv])?;
    let a_log = tensor(weights, &layer_id(layer, LayerTensor::KdaALog), &[heads])?;
    for token in 0..tokens {
        #[allow(clippy::needless_range_loop)]
        // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
        for head in 0..heads {
            let decay_rate = a_log[head].exp();
            for dim in 0..head_dim {
                let channel = head * head_dim + dim;
                let raw = forget[token * qkv + channel] + dt_bias[channel];
                forget[token * qkv + channel] = plan.gate_lower_bound * sigmoid(decay_rate * raw);
            }
        }
    }
    let beta_raw = linear(
        x,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::KdaBeta),
            &[heads, hidden],
        )?,
        tokens,
        hidden,
        heads,
    );

    // Recurrence (recurrent_kimi_delta_attention:477-489): state [heads, k_dim, v_dim];
    // exp(g) decays along the K dimension.
    let mut matrix = vec![0.0; heads * head_dim * head_dim];
    let mut core = vec![0.0; tokens * qkv];
    for token in 0..tokens {
        for head in 0..heads {
            let beta = sigmoid(beta_raw[token * heads + head]);
            let row_offset = (token * heads + head) * head_dim;
            let state_offset = head * head_dim * head_dim;
            for key_index in 0..head_dim {
                let decay = forget[token * qkv + head * head_dim + key_index].exp();
                let state_row = state_offset + key_index * head_dim;
                for value_index in 0..head_dim {
                    matrix[state_row + value_index] *= decay;
                }
            }
            let mut delta = vec![0.0; head_dim];
            for value_index in 0..head_dim {
                let mut memory = 0.0;
                for key_index in 0..head_dim {
                    memory += matrix[state_offset + key_index * head_dim + value_index]
                        * key[row_offset + key_index];
                }
                delta[value_index] = (value[row_offset + value_index] - memory) * beta;
            }
            for key_index in 0..head_dim {
                let state_row = state_offset + key_index * head_dim;
                for value_index in 0..head_dim {
                    matrix[state_row + value_index] +=
                        key[row_offset + key_index] * delta[value_index];
                }
            }
            for value_index in 0..head_dim {
                let mut attended = 0.0;
                for key_index in 0..head_dim {
                    attended += matrix[state_offset + key_index * head_dim + value_index]
                        * query[row_offset + key_index];
                }
                core[row_offset + value_index] = attended;
            }
        }
    }

    // Output: sigmoid-gated fp32 RMSNorm over head_dim (o_norm uses the layer's
    // rms_norm_eps), gate = g_b(g_a(x)); then o_proj.
    let gate_down = linear(
        x,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::KdaGateDown),
            &[head_dim, hidden],
        )?,
        tokens,
        hidden,
        head_dim,
    );
    let gate = linear(
        &gate_down,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::KdaGateUp),
            &[qkv, head_dim],
        )?,
        tokens,
        head_dim,
        qkv,
    );
    let norm_weight = tensor(
        weights,
        &layer_id(layer, LayerTensor::KdaOutputNorm),
        &[head_dim],
    )?;
    let mut gated = rms_norm(&core, tokens * heads, head_dim, norm_weight, epsilon);
    for index in 0..gated.len() {
        gated[index] *= sigmoid(gate[index]);
    }
    let output = linear(
        &gated,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::KdaOutput),
            &[hidden, qkv],
        )?,
        tokens,
        qkv,
        hidden,
    );

    // Conv state stores the raw fused [q|k|v] pre-conv planes for the trailing
    // kernel-1 positions, mirroring the GDN layout.
    let pad = kernel - 1;
    let mut conv_state = vec![0.0; conv_width * pad];
    let planes = [&q_raw, &k_raw, &v_raw];
    for channel in 0..conv_width {
        let plane = channel / qkv;
        let plane_channel = channel % qkv;
        for index in 0..pad {
            let source = tokens as isize - pad as isize + index as isize;
            if source >= 0 {
                conv_state[channel * pad + index] =
                    planes[plane][source as usize * qkv + plane_channel];
            }
        }
    }
    Ok((
        output,
        ReferenceLayerState::Recurrent {
            conv: conv_state,
            matrix,
            value_heads: heads,
            key_head_dim: head_dim,
            value_head_dim: head_dim,
            conv_width,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
// allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn full_attention(
    layer: u32,
    plan: &memra_gguf::model_plan::FullAttentionPlan,
    window: Option<usize>,
    norm_epsilon: f32,
    weights: &ReferenceWeights,
    x: &[f32],
    tokens: usize,
    hidden: usize,
) -> Result<(Vec<f32>, ReferenceLayerState), ReferenceError> {
    let query_heads = plan.query_heads as usize;
    let kv_heads = plan.kv_heads as usize;
    let key_dim = plan.key_head_dim as usize;
    let value_dim = plan.value_head_dim as usize;
    if query_heads == 0 || kv_heads == 0 || query_heads % kv_heads != 0 {
        return Err(ReferenceError::InvalidPlan {
            layer: Some(layer),
            reason: "query heads must be a positive multiple of KV heads",
        });
    }
    let fused = plan.output_gate == AttentionGateKind::FusedQ;
    let q_width = query_heads * key_dim;
    let q_projection_width = q_width * if fused { 2 } else { 1 };
    let k_width = kv_heads * key_dim;
    let v_width = kv_heads * value_dim;
    let q_weight = tensor(
        weights,
        &layer_id(layer, LayerTensor::Query),
        &[q_projection_width, hidden],
    )?;
    let k_weight = tensor(
        weights,
        &layer_id(layer, LayerTensor::Key),
        &[k_width, hidden],
    )?;
    let output_weight = tensor(
        weights,
        &layer_id(layer, LayerTensor::AttentionOutput),
        &[hidden, query_heads * value_dim],
    )?;
    let q_projected = linear(x, q_weight, tokens, hidden, q_projection_width);
    let mut query = vec![0.0; tokens * q_width];
    let mut fused_gate = None;
    if fused {
        let mut gate = vec![0.0; tokens * q_width];
        for token in 0..tokens {
            for head in 0..query_heads {
                let projected = token * q_projection_width + head * 2 * key_dim;
                let canonical = (token * query_heads + head) * key_dim;
                query[canonical..canonical + key_dim]
                    .copy_from_slice(&q_projected[projected..projected + key_dim]);
                gate[canonical..canonical + key_dim]
                    .copy_from_slice(&q_projected[projected + key_dim..projected + 2 * key_dim]);
            }
        }
        fused_gate = Some(gate);
    } else {
        query.copy_from_slice(&q_projected);
    }
    let mut key = linear(x, k_weight, tokens, hidden, k_width);
    let mut value = match plan.value_projection {
        ValueProjection::Separate => linear(
            x,
            tensor(
                weights,
                &layer_id(layer, LayerTensor::Value),
                &[v_width, hidden],
            )?,
            tokens,
            hidden,
            v_width,
        ),
        ValueProjection::ReuseKey => {
            if value_dim != key_dim {
                return Err(ReferenceError::InvalidPlan {
                    layer: Some(layer),
                    reason: "K-as-V requires equal key/value head widths",
                });
            }
            key.clone()
        }
    };
    apply_optional_head_norm(
        weights,
        layer_id(layer, LayerTensor::QueryNorm),
        &mut query,
        tokens * query_heads,
        key_dim,
        plan.qk_norm,
        norm_epsilon,
    )?;
    if plan.value_norm == ValueNorm::WeightlessRms {
        let ones = vec![1.0; value_dim];
        value = rms_norm(&value, tokens * kv_heads, value_dim, &ones, norm_epsilon);
    }
    apply_optional_head_norm(
        weights,
        layer_id(layer, LayerTensor::KeyNorm),
        &mut key,
        tokens * kv_heads,
        key_dim,
        plan.qk_norm,
        norm_epsilon,
    )?;
    let rope_factors = rope_factor_values(&plan.rope, weights)?;
    apply_rope(
        &mut query,
        tokens,
        query_heads,
        key_dim,
        plan.rope.dimensions as usize,
        plan.rope.base,
        rope_factors.as_deref(),
    );
    apply_rope(
        &mut key,
        tokens,
        kv_heads,
        key_dim,
        plan.rope.dimensions as usize,
        plan.rope.base,
        rope_factors.as_deref(),
    );

    let mut attended = vec![0.0; tokens * query_heads * value_dim];
    let scale = match plan.scale {
        AttentionScale::InverseSqrtKeyDim => 1.0 / (key_dim as f32).sqrt(),
        AttentionScale::Fixed(scale) => scale,
    };
    for token in 0..tokens {
        for head in 0..query_heads {
            let kv_head = head * kv_heads / query_heads;
            let first_source = window
                .map(|window| (token + 1).saturating_sub(window))
                .unwrap_or(0);
            let mut scores = Vec::with_capacity(token + 1 - first_source);
            for source in first_source..=token {
                let mut score = 0.0;
                for dim in 0..key_dim {
                    score += query[(token * query_heads + head) * key_dim + dim]
                        * key[(source * kv_heads + kv_head) * key_dim + dim];
                }
                scores.push(score * scale);
            }
            softmax_in_place(&mut scores);
            for (offset, probability) in scores.into_iter().enumerate() {
                let source = first_source + offset;
                for dim in 0..value_dim {
                    attended[(token * query_heads + head) * value_dim + dim] +=
                        probability * value[(source * kv_heads + kv_head) * value_dim + dim];
                }
            }
        }
    }
    if let Some(gate) = fused_gate {
        for token in 0..tokens {
            for head in 0..query_heads {
                for dim in 0..value_dim {
                    if dim >= key_dim {
                        return Err(ReferenceError::InvalidPlan {
                            layer: Some(layer),
                            reason: "fused attention gate requires value_dim <= key_dim",
                        });
                    }
                    attended[(token * query_heads + head) * value_dim + dim] *=
                        sigmoid(gate[(token * query_heads + head) * key_dim + dim]);
                }
            }
        }
    } else if plan.output_gate == AttentionGateKind::SeparateHead {
        let gate_weight = tensor(
            weights,
            &layer_id(layer, LayerTensor::AttentionGate),
            &[query_heads, hidden],
        )?;
        let gates = linear(x, gate_weight, tokens, hidden, query_heads);
        for token in 0..tokens {
            for head in 0..query_heads {
                let gate = sigmoid(gates[token * query_heads + head]);
                for dim in 0..value_dim {
                    attended[(token * query_heads + head) * value_dim + dim] *= gate;
                }
            }
        }
    }
    let state_start = window
        .map(|window| tokens.saturating_sub(window))
        .unwrap_or(0);
    let state_tokens = tokens - state_start;
    let state_key = key[state_start * k_width..].to_vec();
    let state_value = value[state_start * v_width..].to_vec();
    Ok((
        linear(
            &attended,
            output_weight,
            tokens,
            query_heads * value_dim,
            hidden,
        ),
        ReferenceLayerState::Kv {
            key: state_key,
            value: state_value,
            tokens: state_tokens,
            kv_heads,
            key_head_dim: key_dim,
            value_head_dim: value_dim,
            window,
        },
    ))
}

fn dense_mlp(
    layer: u32,
    plan: &memra_gguf::model_plan::DenseMlpPlan,
    weights: &ReferenceWeights,
    x: &[f32],
    tokens: usize,
    hidden: usize,
) -> Result<Vec<f32>, ReferenceError> {
    let intermediate = plan.intermediate_size as usize;
    let gate = linear(
        x,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::MlpGate),
            &[intermediate, hidden],
        )?,
        tokens,
        hidden,
        intermediate,
    );
    let up = linear(
        x,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::MlpUp),
            &[intermediate, hidden],
        )?,
        tokens,
        hidden,
        intermediate,
    );
    let mut activated = vec![0.0; gate.len()];
    for index in 0..gate.len() {
        activated[index] = activate_pair(&plan.activation, gate[index], up[index], layer)?;
    }
    Ok(linear(
        &activated,
        tensor(
            weights,
            &layer_id(layer, LayerTensor::MlpDown),
            &[hidden, intermediate],
        )?,
        tokens,
        intermediate,
        hidden,
    ))
}

#[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
fn moe_mlp(
    layer: u32,
    plan: &memra_gguf::model_plan::MoeMlpPlan,
    weights: &ReferenceWeights,
    x: &[f32],
    token_ids: &[u32],
    tokens: usize,
    hidden: usize,
    vocab: usize,
) -> Result<Vec<f32>, ReferenceError> {
    let experts = plan.expert_count as usize;
    let selected = plan.experts_per_token as usize;
    let intermediate = plan.expert_intermediate_size as usize;
    if selected == 0 || selected > experts {
        return Err(ReferenceError::InvalidPlan {
            layer: Some(layer),
            reason: "MoE top-k must be in 1..=expert_count",
        });
    }
    let router = tensor(
        weights,
        &layer_id(layer, LayerTensor::MoeRouter),
        &[experts, hidden],
    )?;
    let logits = linear(x, router, tokens, hidden, experts);
    let bias = if router_has_selection_bias(&plan.router) {
        Some(tensor(
            weights,
            &layer_id(layer, LayerTensor::MoeRouterBias),
            &[experts],
        )?)
    } else {
        None
    };
    let token_to_expert = if matches!(
        plan.router,
        memra_gguf::model_plan::RouterPlan::TokenIdHash { .. }
    ) {
        Some(tensor(
            weights,
            &layer_id(layer, LayerTensor::MoeTokenToExpert),
            &[vocab, selected],
        )?)
    } else {
        None
    };
    let gate_bank = tensor(
        weights,
        &layer_id(layer, LayerTensor::MoeExpertGateBank),
        &[experts, intermediate, hidden],
    )?;
    let up_bank = tensor(
        weights,
        &layer_id(layer, LayerTensor::MoeExpertUpBank),
        &[experts, intermediate, hidden],
    )?;
    let down_bank = tensor(
        weights,
        &layer_id(layer, LayerTensor::MoeExpertDownBank),
        &[experts, hidden, intermediate],
    )?;
    let mut output = vec![0.0; tokens * hidden];
    for token in 0..tokens {
        let forced_routes = token_to_expert
            .map(|table| {
                let token_id = token_ids[token] as usize;
                &table[token_id * selected..(token_id + 1) * selected]
            })
            .map(|row| {
                row.iter()
                    .map(|&value| {
                        if !value.is_finite()
                            || value < 0.0
                            || value.fract() != 0.0
                            || value as usize >= experts
                        {
                            return Err(ReferenceError::InvalidPlan {
                                layer: Some(layer),
                                reason: "token-id expert table contains an invalid expert id",
                            });
                        }
                        Ok(value as usize)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        let routes = route_experts(
            &plan.router,
            &logits[token * experts..(token + 1) * experts],
            bias,
            selected,
            forced_routes.as_deref(),
            layer,
        )?;
        if crate::hidden_trace::enabled() && token + 1 == tokens {
            crate::hidden_trace::emit_last_row(
                "router",
                layer as i64,
                1,
                experts,
                &logits[token * experts..(token + 1) * experts],
            );
            let mut route = Vec::with_capacity(routes.len() * 2);
            for (expert, weight) in &routes {
                route.push(*expert as f32);
                route.push(*weight);
            }
            crate::hidden_trace::emit_last_row("route", layer as i64, 1, route.len(), &route);
        }
        let input = &x[token * hidden..(token + 1) * hidden];
        for (expert, route_weight) in routes {
            let gate_offset = expert * intermediate * hidden;
            let down_offset = expert * hidden * intermediate;
            let mut activated = vec![0.0; intermediate];
            for row in 0..intermediate {
                let mut gate = 0.0;
                let mut up = 0.0;
                for column in 0..hidden {
                    gate += input[column] * gate_bank[gate_offset + row * hidden + column];
                    up += input[column] * up_bank[gate_offset + row * hidden + column];
                }
                activated[row] = activate_pair(&plan.activation, gate, up, layer)?;
            }
            for row in 0..hidden {
                let mut value = 0.0;
                for column in 0..intermediate {
                    value +=
                        activated[column] * down_bank[down_offset + row * intermediate + column];
                }
                output[token * hidden + row] += route_weight * value;
            }
        }
    }

    if crate::hidden_trace::enabled() {
        crate::hidden_trace::emit_last_row("routed", layer as i64, tokens, hidden, &output);
    }

    if let Some(shared) = plan.shared.as_ref() {
        let intermediate = shared.intermediate_size as usize;
        let gate = linear(
            x,
            tensor(
                weights,
                &layer_id(layer, LayerTensor::SharedMlpGate),
                &[intermediate, hidden],
            )?,
            tokens,
            hidden,
            intermediate,
        );
        let up = linear(
            x,
            tensor(
                weights,
                &layer_id(layer, LayerTensor::SharedMlpUp),
                &[intermediate, hidden],
            )?,
            tokens,
            hidden,
            intermediate,
        );
        let mut activated = vec![0.0; gate.len()];
        for index in 0..gate.len() {
            activated[index] = activate_pair(&plan.activation, gate[index], up[index], layer)?;
        }
        let mut shared_output = linear(
            &activated,
            tensor(
                weights,
                &layer_id(layer, LayerTensor::SharedMlpDown),
                &[hidden, intermediate],
            )?,
            tokens,
            intermediate,
            hidden,
        );
        if shared.gated {
            let gate_weight = tensor(
                weights,
                &layer_id(layer, LayerTensor::SharedMlpInputGate),
                &[hidden],
            )?;
            for token in 0..tokens {
                let mut gate = 0.0;
                for column in 0..hidden {
                    gate += x[token * hidden + column] * gate_weight[column];
                }
                let gate = sigmoid(gate);
                for column in 0..hidden {
                    shared_output[token * hidden + column] *= gate;
                }
            }
        }
        add_in_place(&mut output, &shared_output);
    }
    Ok(output)
}

fn route_experts(
    router: &memra_gguf::model_plan::RouterPlan,
    logits: &[f32],
    bias: Option<&[f32]>,
    selected: usize,
    forced_indices: Option<&[usize]>,
    layer: u32,
) -> Result<Vec<(usize, f32)>, ReferenceError> {
    use memra_gguf::model_plan::{RouterPlan, RouterScorePlan};

    let mut weights = match router {
        RouterPlan::Softmax => {
            let mut probabilities = logits.to_vec();
            softmax_in_place(&mut probabilities);
            probabilities
        }
        RouterPlan::Sigmoid { .. } => logits.iter().map(|&value| sigmoid(value)).collect(),
        RouterPlan::SqrtSoftplus { .. } => {
            logits.iter().map(|&value| softplus(value).sqrt()).collect()
        }
        RouterPlan::TokenIdHash { score, .. } => match score {
            RouterScorePlan::Softmax => {
                let mut probabilities = logits.to_vec();
                softmax_in_place(&mut probabilities);
                probabilities
            }
            RouterScorePlan::Sigmoid => logits.iter().map(|&value| sigmoid(value)).collect(),
            RouterScorePlan::SqrtSoftplus => {
                logits.iter().map(|&value| softplus(value).sqrt()).collect()
            }
        },
    };
    let selection_scores: Vec<f32> = weights
        .iter()
        .enumerate()
        .map(|(index, &weight)| weight + bias.map_or(0.0, |bias| bias[index]))
        .collect();
    let indices = if let RouterPlan::TokenIdHash { .. } = router {
        let Some(forced) = forced_indices else {
            return Err(ReferenceError::InvalidPlan {
                layer: Some(layer),
                reason: "token-id hash router requires a token-to-expert row",
            });
        };
        if forced.len() != selected {
            return Err(ReferenceError::InvalidPlan {
                layer: Some(layer),
                reason: "token-id expert row width does not match MoE top-k",
            });
        }
        let mut seen = std::collections::BTreeSet::new();
        for &index in forced {
            if index >= logits.len() || !seen.insert(index) {
                return Err(ReferenceError::InvalidPlan {
                    layer: Some(layer),
                    reason: "token-id expert row contains an out-of-range or duplicate expert",
                });
            }
        }
        forced.to_vec()
    } else {
        if forced_indices.is_some() {
            return Err(ReferenceError::InvalidPlan {
                layer: Some(layer),
                reason: "score-selected router received forced expert indices",
            });
        }
        let mut indices: Vec<usize> = (0..logits.len()).collect();
        indices.sort_by(|&left, &right| {
            selection_scores[right]
                .total_cmp(&selection_scores[left])
                .then(left.cmp(&right))
        });
        indices.truncate(selected);
        indices
    };
    let (normalize, scaling) = match router {
        RouterPlan::Softmax => (true, 1.0),
        RouterPlan::Sigmoid {
            normalize_selected,
            scaling_factor,
            ..
        }
        | RouterPlan::SqrtSoftplus {
            normalize_selected,
            scaling_factor,
            ..
        } => (*normalize_selected, *scaling_factor),
        RouterPlan::TokenIdHash {
            normalize_selected,
            scaling_factor,
            ..
        } => (*normalize_selected, *scaling_factor),
    };
    if normalize {
        let denominator = indices
            .iter()
            .map(|&index| weights[index])
            .sum::<f32>()
            .max(if matches!(router, RouterPlan::Softmax) {
                6.103_515_6e-5
            } else {
                1e-20
            });
        for weight in &mut weights {
            *weight = *weight / denominator * scaling;
        }
    } else {
        for weight in &mut weights {
            *weight *= scaling;
        }
    }
    Ok(indices
        .into_iter()
        .map(|index| (index, weights[index]))
        .collect())
}

fn router_has_selection_bias(router: &memra_gguf::model_plan::RouterPlan) -> bool {
    matches!(
        router,
        memra_gguf::model_plan::RouterPlan::Sigmoid {
            selection_bias: true,
            ..
        } | memra_gguf::model_plan::RouterPlan::SqrtSoftplus {
            selection_bias: true,
            ..
        }
    )
}

fn activate_pair(
    activation: &ActivationPlan,
    gate: f32,
    up: f32,
    layer: u32,
) -> Result<f32, ReferenceError> {
    Ok(match activation {
        ActivationPlan::Silu => silu(gate) * up,
        ActivationPlan::GeluTanh => gelu_tanh(gate) * up,
        ActivationPlan::SwiGluOai { alpha, limit } => {
            (gate * sigmoid(*alpha * gate)).min(*limit) * up.clamp(-*limit, *limit)
        }
        ActivationPlan::SwiGluClamped { limit } => {
            silu(gate).min(*limit) * up.clamp(-*limit, *limit)
        }
        // glm5_next: the gate clamp is PRE-silu and one-sided (no lower bound).
        ActivationPlan::SwiGluPreClamped { limit } => {
            silu(gate.min(*limit)) * up.clamp(-*limit, *limit)
        }
        ActivationPlan::Named(_) => {
            return Err(ReferenceError::UnsupportedOperation {
                layer: Some(layer),
                operation: "named MLP activation",
            });
        }
    })
}

fn tensor<'a>(
    weights: &'a ReferenceWeights,
    id: &TensorId,
    expected: &[usize],
) -> Result<&'a [f32], ReferenceError> {
    let tensor = weights
        .get(id)
        .ok_or_else(|| ReferenceError::MissingTensor(id.clone()))?;
    tensor_checked(id, tensor, expected)
}

fn tensor_checked<'a>(
    id: &TensorId,
    tensor: &'a ReferenceTensor,
    expected: &[usize],
) -> Result<&'a [f32], ReferenceError> {
    if tensor.shape != expected {
        return Err(ReferenceError::TensorShape {
            id: Some(id.clone()),
            expected: expected.to_vec(),
            actual_elements: tensor.data.len(),
        });
    }
    Ok(&tensor.data)
}

fn layer_id(layer: u32, tensor: LayerTensor) -> TensorId {
    TensorId::Layer {
        index: layer,
        tensor,
    }
}

fn linear(x: &[f32], weight: &[f32], rows: usize, input: usize, output: usize) -> Vec<f32> {
    let mut result = vec![0.0; rows * output];
    for row in 0..rows {
        for out in 0..output {
            let mut sum = 0.0;
            for inner in 0..input {
                sum += x[row * input + inner] * weight[out * input + inner];
            }
            result[row * output + out] = sum;
        }
    }
    result
}

fn rms_norm(x: &[f32], rows: usize, width: usize, weight: &[f32], epsilon: f32) -> Vec<f32> {
    let mut result = vec![0.0; x.len()];
    for row in 0..rows {
        let input = &x[row * width..(row + 1) * width];
        let mean_square = input.iter().map(|value| value * value).sum::<f32>() / width as f32;
        let inverse = 1.0 / (mean_square + epsilon).sqrt();
        for index in 0..width {
            result[row * width + index] = input[index] * inverse * weight[index];
        }
    }
    result
}

/// LayerNorm WITH bias (indexer k_norm). The epsilon is nn.LayerNorm's default and
/// does NOT track rms_norm_eps.
fn layer_norm(x: &[f32], rows: usize, width: usize, weight: &[f32], bias: &[f32]) -> Vec<f32> {
    const EPSILON: f32 = 1e-5;
    let mut result = vec![0.0; x.len()];
    for row in 0..rows {
        let input = &x[row * width..(row + 1) * width];
        let mean = input.iter().sum::<f32>() / width as f32;
        let variance = input
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f32>()
            / width as f32;
        let inverse = 1.0 / (variance + EPSILON).sqrt();
        for index in 0..width {
            result[row * width + index] =
                (input[index] - mean) * inverse * weight[index] + bias[index];
        }
    }
    result
}

fn l2_normalize_rows(values: &mut [f32], rows: usize, width: usize, epsilon: f32) {
    for row in 0..rows {
        let offset = row * width;
        let sum = values[offset..offset + width]
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        let inverse = 1.0 / (sum + epsilon).sqrt();
        for value in &mut values[offset..offset + width] {
            *value *= inverse;
        }
    }
}

fn apply_optional_head_norm(
    weights: &ReferenceWeights,
    id: TensorId,
    values: &mut [f32],
    rows: usize,
    width: usize,
    presence: memra_gguf::model_plan::TensorPresence,
    epsilon: f32,
) -> Result<(), ReferenceError> {
    let Some(weight) = weights.get(&id) else {
        return if presence == memra_gguf::model_plan::TensorPresence::Required {
            Err(ReferenceError::MissingTensor(id))
        } else {
            Ok(())
        };
    };
    let normalized = rms_norm(
        values,
        rows,
        width,
        tensor_checked(&id, weight, &[width])?,
        epsilon,
    );
    values.copy_from_slice(&normalized);
    Ok(())
}

fn rope_factor_values(
    plan: &memra_gguf::model_plan::RopePlan,
    weights: &ReferenceWeights,
) -> Result<Option<Vec<f32>>, ReferenceError> {
    use memra_gguf::model_plan::RopeFactors;

    let width = plan.dimensions as usize / 2;
    Ok(match plan.factors {
        RopeFactors::None => None,
        RopeFactors::PartialRotary { factor } => {
            let keep = (width as f32 * factor.clamp(0.0, 1.0)).round() as usize;
            Some(
                (0..width)
                    .map(|index| if index < keep { 1.0 } else { 1.0e30 })
                    .collect(),
            )
        }
        RopeFactors::Checkpoint => {
            let tensor = weights
                .get(&TensorId::RopeFactors)
                .ok_or(ReferenceError::MissingTensor(TensorId::RopeFactors))?;
            if tensor.shape.len() != 1 || tensor.data.len() < width {
                return Err(ReferenceError::TensorShape {
                    id: Some(TensorId::RopeFactors),
                    expected: vec![width],
                    actual_elements: tensor.data.len(),
                });
            }
            Some(tensor.data[..width].to_vec())
        }
        RopeFactors::Yarn { .. } => {
            return Err(ReferenceError::UnsupportedOperation {
                layer: None,
                operation: "YaRN on non-compressed attention",
            });
        }
    })
}

fn apply_rope(
    values: &mut [f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    dimensions: usize,
    base: f32,
    factors: Option<&[f32]>,
) {
    let dimensions = dimensions.min(head_dim) / 2 * 2;
    let half = dimensions / 2;
    for token in 0..tokens {
        for head in 0..heads {
            let offset = (token * heads + head) * head_dim;
            for index in 0..half {
                let factor = factors.map_or(1.0, |factors| factors[index]);
                let frequency = base.powf(-2.0 * index as f32 / dimensions as f32) / factor;
                let angle = token as f32 * frequency;
                let (sin, cos) = angle.sin_cos();
                let first = values[offset + index];
                let second = values[offset + index + half];
                values[offset + index] = first * cos - second * sin;
                values[offset + index + half] = first * sin + second * cos;
            }
        }
    }
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

fn add_in_place(target: &mut [f32], addend: &[f32]) {
    for (target, addend) in target.iter_mut().zip(addend) {
        *target += addend;
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

fn gelu_tanh(value: f32) -> f32 {
    0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * value * value * value)).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::{HfConfig, ModelConfig};

    fn weight(shape: &[usize], data: &[f32]) -> ReferenceTensor {
        ReferenceTensor::new(shape.to_vec(), data.to_vec()).unwrap()
    }

    #[test]
    fn one_token_dense_plan_matches_hand_derived_logits_and_emits_kv_state() {
        let config = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":2,
            "num_attention_heads":1,"num_key_value_heads":1,"head_dim":2,
            "intermediate_size":2,"vocab_size":3,"max_position_embeddings":8,
            "rms_norm_eps":0.000001}"#,
        ));
        let plan = ModelPlan::compile(&config).unwrap();
        let identity = [1.0, 0.0, 0.0, 1.0];
        let zero = [0.0; 4];
        let mut weights = ReferenceWeights::new();
        weights.insert(
            TensorId::TokenEmbedding,
            weight(&[3, 2], &[1.0, 0.0, 0.0, 1.0, -1.0, 0.0]),
        );
        weights.insert(TensorId::OutputNorm, weight(&[2], &[1.0, 1.0]));
        for tensor in [LayerTensor::PreAttentionNorm, LayerTensor::PreMlpNorm] {
            weights.insert(layer_id(0, tensor), weight(&[2], &[1.0, 1.0]));
        }
        for tensor in [
            LayerTensor::Query,
            LayerTensor::Key,
            LayerTensor::Value,
            LayerTensor::AttentionOutput,
        ] {
            weights.insert(layer_id(0, tensor), weight(&[2, 2], &identity));
        }
        for tensor in [
            LayerTensor::MlpGate,
            LayerTensor::MlpUp,
            LayerTensor::MlpDown,
        ] {
            weights.insert(layer_id(0, tensor), weight(&[2, 2], &zero));
        }

        let output = execute(&plan, &weights, &[0]).unwrap();
        let root_two = 2.0f32.sqrt();
        assert_eq!((output.tokens, output.vocab), (1, 3));
        assert!((output.logits[0] - root_two).abs() < 2e-5);
        assert!(output.logits[1].abs() < 2e-5);
        assert!((output.logits[2] + root_two).abs() < 2e-5);
        let ReferenceLayerState::Kv {
            tokens, key, value, ..
        } = &output.state.layers[0]
        else {
            panic!("expected KV state");
        };
        assert_eq!(*tokens, 1);
        assert_eq!(key.len(), 2);
        assert_eq!(value.len(), 2);
    }

    #[test]
    fn hyperconnections_execute_stream_state_and_head_collapse() {
        let config = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":2,
            "num_attention_heads":1,"num_key_value_heads":1,"head_dim":2,
            "intermediate_size":2,"vocab_size":3,"max_position_embeddings":8}"#,
        ));
        let mut plan = ModelPlan::compile(&config).unwrap();
        plan.layers[0].residual = ResidualTopology::HyperConnections {
            streams: 2,
            epsilon: 1e-6,
            sinkhorn_iterations: 2,
            collapse: HcCollapse::GatedHead,
        };
        let fixture = deterministic_fixture(&plan).unwrap();
        assert_eq!(
            fixture.weights[&TensorId::HyperHeadFunction].shape,
            vec![2, 4]
        );
        assert_eq!(
            fixture.weights[&layer_id(0, LayerTensor::HyperAttentionFunction)].shape,
            vec![8, 4]
        );
        let output = execute(&plan, &fixture.weights, &fixture.token_ids).unwrap();
        assert!(output.logits.iter().all(|value| value.is_finite()));
        assert!(matches!(
            output.state.layers[0],
            ReferenceLayerState::Kv { .. }
        ));
    }

    #[test]
    fn generated_tiny_fixture_is_deterministic_and_executable() {
        let config = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3","num_hidden_layers":2,"hidden_size":8,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32}"#,
        ));
        let plan = ModelPlan::compile(&config).unwrap();
        let first = deterministic_fixture(&plan).unwrap();
        let second = deterministic_fixture(&plan).unwrap();
        assert_eq!(first, second);
        let output = execute(&plan, &first.weights, &first.token_ids).unwrap();
        assert_eq!(output.logits.len(), first.token_ids.len() * 32);
        assert!(output.logits.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn qwen35_fixture_executes_mixed_gdn_and_full_attention_state() {
        let config = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3_5","num_hidden_layers":4,"hidden_size":8,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32,
            "rms_norm_eps":0.000001,"full_attention_interval":2,
            "linear_conv_kernel_dim":3,"linear_key_head_dim":4,
            "linear_value_head_dim":4,"linear_num_key_heads":1,
            "linear_num_value_heads":2}"#,
        ));
        let plan = ModelPlan::compile(&config).unwrap();
        let fixture = deterministic_fixture(&plan).unwrap();
        let output = execute(&plan, &fixture.weights, &fixture.token_ids).unwrap();
        assert_eq!(output.state.layers.len(), 4);
        assert!(matches!(
            output.state.layers[0],
            ReferenceLayerState::Recurrent { .. }
        ));
        assert!(matches!(
            output.state.layers[1],
            ReferenceLayerState::Kv { .. }
        ));
        assert!(matches!(
            output.state.layers[2],
            ReferenceLayerState::Recurrent { .. }
        ));
        assert!(matches!(
            output.state.layers[3],
            ReferenceLayerState::Kv { .. }
        ));
        assert!(output.logits.iter().all(|value| value.is_finite()));
        assert_eq!(
            output.logits[..8]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![
                3_182_242_076,
                1_053_299_392,
                3_199_800_546,
                3_198_737_445,
                3_180_184_136,
                3_187_768_631,
                1_057_556_100,
                1_035_812_924,
            ]
        );
    }

    #[test]
    fn router_laws_pin_stable_ties_and_selection_only_bias() {
        use memra_gguf::model_plan::{RouterPlan, RouterScorePlan};

        assert_eq!(
            route_experts(&RouterPlan::Softmax, &[0.0, 0.0, 0.0], None, 2, None, 0,).unwrap(),
            vec![(0, 0.5), (1, 0.5)]
        );
        assert_eq!(
            route_experts(
                &RouterPlan::Sigmoid {
                    normalize_selected: true,
                    scaling_factor: 2.0,
                    selection_bias: true,
                },
                &[0.0, 0.0],
                Some(&[-1.0, 1.0]),
                1,
                None,
                0,
            )
            .unwrap(),
            vec![(1, 2.0)]
        );
        assert_eq!(
            route_experts(
                &RouterPlan::TokenIdHash {
                    score: RouterScorePlan::SqrtSoftplus,
                    normalize_selected: true,
                    scaling_factor: 1.5,
                },
                &[0.0, 0.0, 0.0],
                None,
                2,
                Some(&[2, 0]),
                0,
            )
            .unwrap(),
            vec![(2, 0.75), (0, 0.75)]
        );
        assert!(matches!(
            route_experts(
                &RouterPlan::TokenIdHash {
                    score: RouterScorePlan::SqrtSoftplus,
                    normalize_selected: true,
                    scaling_factor: 1.5,
                },
                &[0.0, 0.0, 0.0],
                None,
                2,
                Some(&[1, 1]),
                0,
            ),
            Err(ReferenceError::InvalidPlan {
                reason: "token-id expert row contains an out-of-range or duplicate expert",
                ..
            })
        ));
    }

    #[test]
    fn token_hash_moe_fixture_executes_from_semantic_token_table() {
        use memra_gguf::model_plan::{RouterPlan, RouterScorePlan};

        let config = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3_moe","num_hidden_layers":1,"hidden_size":8,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "intermediate_size":16,"vocab_size":16,"max_position_embeddings":32,
            "num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":8}"#,
        ));
        let mut plan = ModelPlan::compile(&config).unwrap();
        let MlpPlan::Moe(moe) = &mut plan.layers[0].mlp else {
            unreachable!()
        };
        moe.router = RouterPlan::TokenIdHash {
            score: RouterScorePlan::SqrtSoftplus,
            normalize_selected: true,
            scaling_factor: 1.5,
        };
        let fixture = deterministic_fixture(&plan).unwrap();
        let table_id = layer_id(0, LayerTensor::MoeTokenToExpert);
        assert_eq!(fixture.weights[&table_id].shape, vec![16, 2]);
        let output = execute(&plan, &fixture.weights, &fixture.token_ids).unwrap();
        assert!(output.logits.iter().all(|value| value.is_finite()));

        let mut alternate = fixture.weights.clone();
        alternate.get_mut(&table_id).unwrap().data.fill(3.0);
        for row in alternate
            .get_mut(&table_id)
            .unwrap()
            .data
            .chunks_exact_mut(2)
        {
            row[1] = 2.0;
        }
        let alternate = execute(&plan, &alternate, &fixture.token_ids).unwrap();
        assert_ne!(output.logits, alternate.logits);
    }

    #[test]
    fn qwen3_moe_fixture_executes_routed_and_shared_branches() {
        let config = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3_moe","num_hidden_layers":2,"hidden_size":8,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32,
            "num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":8,
            "shared_expert_intermediate_size":8}"#,
        ));
        let plan = ModelPlan::compile(&config).unwrap();
        let fixture = deterministic_fixture(&plan).unwrap();
        let output = execute(&plan, &fixture.weights, &fixture.token_ids).unwrap();
        assert!(output.logits.iter().all(|value| value.is_finite()));
        assert_eq!(
            output.logits[..8]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![
                3_205_834_204,
                1_034_800_117,
                1_053_917_366,
                3_190_866_844,
                984_171_488,
                3_182_514_784,
                3_154_736_064,
                3_175_624_690,
            ]
        );
    }

    #[test]
    fn sliding_window_limits_attention_and_trims_reference_state() {
        let config = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":8,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32}"#,
        ));
        let mut plan = ModelPlan::compile(&config).unwrap();
        let AttentionPlan::Full(attention) = plan.layers[0].attention.clone() else {
            unreachable!()
        };
        plan.layers[0].attention = AttentionPlan::SlidingWindow {
            attention,
            window: 2,
        };
        let fixture = deterministic_fixture(&plan).unwrap();
        let output = execute(&plan, &fixture.weights, &fixture.token_ids).unwrap();
        let ReferenceLayerState::Kv { tokens, window, .. } = output.state.layers[0] else {
            panic!("expected sliding KV state");
        };
        assert_eq!(tokens, 2);
        assert_eq!(window, Some(2));
    }

    #[test]
    fn mla_fixture_emits_latent_state_and_sparse_overflow_refuses() {
        use memra_gguf::model_plan::{
            MlaAttentionPlan, RopeFactors, RopePlan, SparseIndexPlan, StatePlan,
        };

        let config = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":8,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32}"#,
        ));
        let mut plan = ModelPlan::compile(&config).unwrap();
        let mla = MlaAttentionPlan::LatentKv {
            query_heads: 2,
            q_lora_rank: 4,
            kv_lora_rank: 4,
            qk_head_dim: 4,
            rope_head_dim: 2,
            value_head_dim: 4,
            rope: RopePlan {
                dimensions: 2,
                base: 10_000.0,
                factors: RopeFactors::None,
            },
            sparse_index: SparseIndexPlan::None,
        };
        plan.layers[0].attention = AttentionPlan::Mla(mla.clone());
        plan.layers[0].state = StatePlan::LatentKvCache {
            width: 6,
            index_width: 0,
        };
        let fixture = deterministic_fixture(&plan).unwrap();
        let output = execute(&plan, &fixture.weights, &fixture.token_ids).unwrap();
        let ReferenceLayerState::LatentKv { tokens, width, .. } = output.state.layers[0] else {
            panic!("expected latent KV state");
        };
        assert_eq!((tokens, width), (3, 6));
        assert_eq!(
            output.logits[..4]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![1_035_177_220, 1_055_447_641, 3_201_478_680, 3_199_508_856]
        );

        let MlaAttentionPlan::LatentKv {
            query_heads,
            q_lora_rank,
            kv_lora_rank,
            qk_head_dim,
            rope_head_dim,
            value_head_dim,
            rope,
            ..
        } = mla
        else {
            unreachable!()
        };
        plan.layers[0].attention = AttentionPlan::Mla(MlaAttentionPlan::LatentKv {
            query_heads,
            q_lora_rank,
            kv_lora_rank,
            qk_head_dim,
            rope_head_dim,
            value_head_dim,
            rope,
            sparse_index: SparseIndexPlan::Own {
                heads: 1,
                head_dim: 2,
                top_k: 2,
                kpool: None,
            },
        });
        let error = execute(&plan, &fixture.weights, &fixture.token_ids).unwrap_err();
        assert!(matches!(
            error,
            ReferenceError::UnsupportedOperation {
                operation: "sparse MLA selection beyond full-selection equivalence",
                ..
            }
        ));
    }

    #[test]
    fn compressed_mla_executes_window_compressor_indexer_and_grouped_output() {
        use memra_gguf::model_plan::{
            KvCompressorPlan, MlaAttentionPlan, RopeFactors, RopePlan, SparseIndexPlan, StatePlan,
        };

        let config = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":128,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":64,
            "intermediate_size":256,"vocab_size":32,"max_position_embeddings":64,
            "rms_norm_eps":0.000001}"#,
        ));
        let mut plan = ModelPlan::compile(&config).unwrap();
        plan.layers[0].attention = AttentionPlan::Mla(MlaAttentionPlan::CompressedKv {
            query_heads: 2,
            q_lora_rank: 64,
            latent_head_dim: 128,
            rope_head_dim: 64,
            output_lora_rank: 64,
            output_groups: 1,
            window: 4,
            rope: RopePlan {
                dimensions: 64,
                base: 160_000.0,
                factors: RopeFactors::Yarn {
                    factor: 2.0,
                    original_context: 32,
                    beta_fast: 32.0,
                    beta_slow: 1.0,
                },
            },
            compressor: Some(KvCompressorPlan {
                ratio: 4,
                latent_dim: 256,
            }),
            sparse_index: SparseIndexPlan::Own {
                heads: 2,
                head_dim: 128,
                top_k: 2,
                kpool: None,
            },
        });
        plan.layers[0].state = StatePlan::CompressedAttention {
            window: 4,
            head_dim: 128,
            compressor_ratio: Some(4),
            sparse_top_k: Some(2),
        };
        let fixture = deterministic_fixture(&plan).unwrap();
        let output = execute(&plan, &fixture.weights, &[1, 2, 3, 4]).unwrap();
        let ReferenceLayerState::CompressedAttention {
            tokens,
            width,
            window,
            compressed_tokens,
            ..
        } = output.state.layers[0]
        else {
            panic!("expected compressed attention state")
        };
        assert_eq!((tokens, width, window, compressed_tokens), (5, 128, 4, 1));
        assert!(output.logits.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn dsv4_shaped_trunk_executes_one_canonical_plan() {
        let config = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"deepseek_v4","num_hidden_layers":2,"hidden_size":128,
            "num_attention_heads":1,"num_key_value_heads":1,"head_dim":128,
            "intermediate_size":256,"vocab_size":128,"max_position_embeddings":1024,
            "rms_norm_eps":0.000001,"rope_theta":10000,"n_routed_experts":4,
            "n_shared_experts":1,"num_experts_per_tok":2,"moe_intermediate_size":128,
            "norm_topk_prob":true,"num_hash_layers":1,"num_nextn_predict_layers":1,
            "scoring_func":"sqrtsoftplus","topk_method":"noaux_tc",
            "routed_scaling_factor":1.5,"hc_eps":0.000001,"hc_mult":2,
            "hc_sinkhorn_iters":4,"q_lora_rank":128,"qk_rope_head_dim":64,
            "o_lora_rank":128,"o_groups":1,"index_n_heads":1,"index_head_dim":128,
            "index_topk":16,"compress_ratios":[0,4,0],"compress_rope_theta":160000,
            "sliding_window":128,"swiglu_limit":10.0,
            "rope_scaling":{"factor":4,"beta_fast":32,"beta_slow":1,
            "original_max_position_embeddings":1024}}"#,
        ));
        let mut plan = ModelPlan::compile(&config).unwrap();
        assert_eq!(plan.layers.len(), 2);
        plan.mtp_blocks.clear();
        let fixture = deterministic_fixture(&plan).unwrap();
        let output = execute(&plan, &fixture.weights, &[1, 2, 3, 4]).unwrap();
        assert_eq!(output.state.layers.len(), 2);
        assert!(
            output
                .state
                .layers
                .iter()
                .all(|state| matches!(state, ReferenceLayerState::CompressedAttention { .. }))
        );
        assert!(
            fixture
                .weights
                .contains_key(&layer_id(0, LayerTensor::MoeTokenToExpert))
        );
        assert!(
            fixture
                .weights
                .contains_key(&layer_id(1, LayerTensor::MoeRouterBias))
        );
        assert!(output.logits.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn dspark_executes_trunk_tap_ring_blocks_markov_and_confidence() {
        use memra_gguf::model_plan::{DrafterPlan, DsparkPlan};

        let config = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"deepseek_v4","num_hidden_layers":2,"hidden_size":128,
            "num_attention_heads":1,"num_key_value_heads":1,"head_dim":128,
            "intermediate_size":256,"vocab_size":128,"max_position_embeddings":1024,
            "rms_norm_eps":0.000001,"rope_theta":10000,"n_routed_experts":4,
            "n_shared_experts":1,"num_experts_per_tok":2,"moe_intermediate_size":128,
            "norm_topk_prob":true,"num_hash_layers":1,"num_nextn_predict_layers":1,
            "scoring_func":"sqrtsoftplus","topk_method":"noaux_tc",
            "routed_scaling_factor":1.5,"hc_eps":0.000001,"hc_mult":2,
            "hc_sinkhorn_iters":4,"q_lora_rank":128,"qk_rope_head_dim":64,
            "o_lora_rank":128,"o_groups":1,"index_n_heads":1,"index_head_dim":128,
            "index_topk":16,"compress_ratios":[0,4,0],"compress_rope_theta":160000,
            "sliding_window":128,"swiglu_limit":10.0,
            "rope_scaling":{"factor":4,"beta_fast":32,"beta_slow":1,
            "original_max_position_embeddings":1024}}"#,
        ));
        let mut plan = ModelPlan::compile(&config).unwrap();
        let block = plan.mtp_blocks.remove(0).layer;
        plan.drafter = Some(DrafterPlan::Dspark(DsparkPlan {
            block_size: 3,
            noise_token_id: 31,
            target_layer_ids: vec![1],
            markov_rank: 8,
            blocks: vec![block],
        }));
        let fixture = deterministic_fixture(&plan).unwrap();
        let output = execute(&plan, &fixture.weights, &[1, 2, 3, 4]).unwrap();
        let draft = output.draft.expect("DSpark output");
        assert_eq!(draft.input_token, 4);
        assert_eq!(draft.output_ids.len(), 4);
        assert_eq!(draft.confidence.len(), 3);
        assert_eq!(draft.logits.len(), 3 * 128);
        assert!(draft.logits.iter().all(|value| value.is_finite()));
        assert!(draft.confidence.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn gemma4_vision_executes_patch_rope_pool_standardize_and_projection() {
        let config = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"gemma4","image_token_id":31,"vision_soft_tokens_per_image":1,
            "text_config":{"model_type":"gemma4_text",
            "num_hidden_layers":2,"hidden_size":8,"num_attention_heads":2,
            "num_key_value_heads":1,"num_global_key_value_heads":1,"head_dim":4,
            "global_head_dim":4,"intermediate_size":16,"vocab_size":32,
            "max_position_embeddings":64,"rms_norm_eps":0.000001,"sliding_window":8,
            "layer_types":["sliding_attention","full_attention"],
            "rope_parameters":{"full_attention":{"rope_theta":10000,
            "partial_rotary_factor":0.5},"sliding_attention":{"rope_theta":10000}}},
            "vision_config":{"hidden_size":8,"intermediate_size":16,
            "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
            "head_dim":4,"max_position_embeddings":64,"patch_size":2,
            "position_embedding_size":16,"pooling_kernel_size":2,
            "rms_norm_eps":0.000001,"standardize":true,"use_clipped_linears":false,
            "hidden_activation":"gelu_pytorch_tanh","rope_parameters":{"rope_theta":100}}}"#,
        ));
        let plan = ModelPlan::compile(&config).unwrap();
        let fixture = deterministic_fixture(&plan).unwrap();
        let input = fixture.vision.as_ref().expect("vision fixture");
        let first = execute_vision(&plan, &fixture.weights, input).unwrap();
        let second = execute_vision(&plan, &fixture.weights, input).unwrap();
        assert_eq!(first, second);
        assert_eq!((first.patch_count, first.output_tokens), (4, 1));
        assert_eq!((first.hidden_size, first.projection_size), (8, 8));
        assert_eq!(first.encoder_hidden.len(), 4 * 8);
        assert_eq!(first.pooled_hidden.len(), 8);
        assert_eq!(first.projected_hidden.len(), 8);
        assert!(first.projected_hidden.iter().all(|value| value.is_finite()));
        let multimodal = execute_multimodal(&plan, &fixture.weights, &[1, 31, 2], input).unwrap();
        let text_only = execute(&plan, &fixture.weights, &[1, 31, 2]).unwrap();
        assert_eq!(multimodal.vision, first);
        assert_ne!(multimodal.language.logits, text_only.logits);
        assert!(
            plan.operations()
                .contains(&memra_gguf::model_plan::OperationKind::VisionTokenInjection)
        );
    }

    #[test]
    fn gemma4_parallel_moe_executes_shared_routed_and_scaled_residual_branches() {
        let config = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"gemma4","text_config":{"model_type":"gemma4_text",
            "num_hidden_layers":2,"hidden_size":8,"num_attention_heads":2,
            "num_key_value_heads":1,"num_global_key_value_heads":1,"head_dim":4,
            "global_head_dim":4,"intermediate_size":16,"moe_intermediate_size":8,
            "num_experts":4,"top_k_experts":2,"vocab_size":32,
            "max_position_embeddings":64,"rms_norm_eps":0.000001,"sliding_window":8,
            "layer_types":["sliding_attention","full_attention"],
            "rope_parameters":{"full_attention":{"rope_theta":10000,
            "partial_rotary_factor":0.5},"sliding_attention":{"rope_theta":10000}}}}"#,
        ));
        let plan = ModelPlan::compile(&config).unwrap();
        let MlpPlan::Moe(moe) = &plan.layers[0].mlp else {
            panic!("expected Gemma MoE")
        };
        assert_eq!(moe.experts_per_token, 2);
        assert_eq!(moe.shared.as_ref().unwrap().intermediate_size, 16);
        assert!(matches!(
            plan.layers[0].residual,
            ResidualTopology::Gemma {
                parallel_moe: Some(_),
                ..
            }
        ));
        let fixture = deterministic_fixture(&plan).unwrap();
        let output = execute(&plan, &fixture.weights, &fixture.token_ids).unwrap();
        assert!(output.logits.iter().all(|value| value.is_finite()));
        assert!(
            plan.operations()
                .contains(&memra_gguf::model_plan::OperationKind::GemmaParallelMoeResidual)
        );
    }

    #[test]
    fn embedded_mtp_executes_typed_fusion_block_and_fallback_head() {
        let config = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3_5","num_hidden_layers":2,
            "num_nextn_predict_layers":1,"hidden_size":8,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32,
            "rms_norm_eps":0.000001,"full_attention_interval":2,
            "linear_conv_kernel_dim":3,"linear_key_head_dim":4,
            "linear_value_head_dim":4,"linear_num_key_heads":1,
            "linear_num_value_heads":2}"#,
        ));
        let plan = ModelPlan::compile(&config).unwrap();
        assert_eq!(plan.mtp_blocks.len(), 1);
        let fixture = deterministic_fixture(&plan).unwrap();
        let output = execute(&plan, &fixture.weights, &fixture.token_ids).unwrap();
        assert_eq!(output.mtp.len(), 1);
        assert_eq!(output.mtp[0].depth, 0);
        assert_eq!(output.mtp[0].hidden.len(), fixture.token_ids.len() * 8);
        assert_eq!(output.mtp[0].logits.len(), fixture.token_ids.len() * 32);
        assert!(output.mtp[0].logits.iter().all(|value| value.is_finite()));
        assert_eq!(
            output.mtp[0].logits[..4]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![1_042_962_358, 1_044_718_512, 3_171_782_004, 3_189_261_409]
        );
    }

    #[test]
    fn multi_depth_mtp_threads_hidden_through_every_typed_block() {
        let config = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3_5","num_hidden_layers":2,
            "num_nextn_predict_layers":2,"hidden_size":8,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32,
            "rms_norm_eps":0.000001,"full_attention_interval":2,
            "linear_conv_kernel_dim":3,"linear_key_head_dim":4,
            "linear_value_head_dim":4,"linear_num_key_heads":1,
            "linear_num_value_heads":2}"#,
        ));
        let plan = ModelPlan::compile(&config).unwrap();
        assert_eq!(plan.mtp_blocks.len(), 2);
        let fixture = deterministic_fixture(&plan).unwrap();
        let output = execute(&plan, &fixture.weights, &fixture.token_ids).unwrap();
        assert_eq!(
            output
                .mtp
                .iter()
                .map(|block| block.depth)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(
            output
                .mtp
                .iter()
                .flat_map(|block| &block.logits)
                .all(|value| value.is_finite())
        );
        assert_ne!(output.mtp[0].hidden, output.mtp[1].hidden);
    }

    #[test]
    fn rope_uses_neox_split_half_pairs() {
        use memra_gguf::model_plan::{RopeFactors, RopePlan};

        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        apply_rope(&mut values, 1, 1, 4, 4, 10_000.0, None);
        // Position zero is deliberately unchanged.
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);

        let mut values = vec![0.0; 8];
        values[4..].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        apply_rope(&mut values, 2, 1, 4, 4, 10_000.0, None);
        let (sin0, cos0) = 1.0f32.sin_cos();
        let (sin1, cos1) = 0.01f32.sin_cos();
        let row = &values[4..];
        assert!((row[0] - (cos0 - 3.0 * sin0)).abs() < 1e-6);
        assert!((row[2] - (sin0 + 3.0 * cos0)).abs() < 1e-6);
        assert!((row[1] - (2.0 * cos1 - 4.0 * sin1)).abs() < 1e-6);
        assert!((row[3] - (2.0 * sin1 + 4.0 * cos1)).abs() < 1e-6);
        assert_eq!(
            rope_factor_values(
                &RopePlan {
                    dimensions: 4,
                    base: 10_000.0,
                    factors: RopeFactors::PartialRotary { factor: 0.5 },
                },
                &ReferenceWeights::new(),
            )
            .unwrap()
            .unwrap(),
            vec![1.0, 1.0e30]
        );
    }

    #[test]
    fn dense_gemma_executes_scaled_parallel_residual_and_k_as_v() {
        let config = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"gemma4","num_hidden_layers":2,"hidden_size":8,
            "num_attention_heads":2,"num_key_value_heads":1,
            "num_global_key_value_heads":1,"head_dim":4,"global_head_dim":4,
            "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32,
            "rms_norm_eps":0.000001,"sliding_window":2,
            "final_logit_softcapping":30,
            "layer_types":["sliding_attention","full_attention"],
            "rope_parameters":{"full_attention":{"rope_theta":1000000,
            "partial_rotary_factor":0.5},"sliding_attention":{"rope_theta":10000}}}"#,
        ));
        let plan = ModelPlan::compile(&config).unwrap();
        assert_eq!(plan.embedding_scale, 8.0f32.sqrt());
        let fixture = deterministic_fixture(&plan).unwrap();
        assert!(
            !fixture
                .weights
                .contains_key(&layer_id(1, LayerTensor::Value))
        );
        let output = execute(&plan, &fixture.weights, &fixture.token_ids).unwrap();
        assert!(output.logits.iter().all(|value| value.is_finite()));
        let ReferenceLayerState::Kv { window, .. } = output.state.layers[0] else {
            panic!("expected SWA state");
        };
        assert_eq!(window, Some(2));
        let ReferenceLayerState::Kv { window, .. } = output.state.layers[1] else {
            panic!("expected global state");
        };
        assert_eq!(window, None);
        assert_eq!(
            output.logits[..4]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![3_198_203_366, 1_057_194_687, 3_185_247_713, 3_204_119_266]
        );
    }

    /// ONE TensorId, ONE byte order. `deterministic_fixture` mints the reference's weights and
    /// `TensorContract` names the same ids for the engine's loader; any engine-vs-reference gate
    /// serves ONE set of bytes under both. A shape disagreement preserves element counts, so
    /// nothing else catches it — `MlaKeyUp` was minted `[head][nope][rank]` against the
    /// contract's `[head][rank][nope]` and every MLA parity comparison silently mis-strided the
    /// absorb operand (glm53-flash lane, 2026-08-28: 1.24e-1 relative on a micro fixture that
    /// drops to 6.9e-7 once the layouts agree).
    #[test]
    fn the_mla_fixture_shapes_match_the_tensor_contract() {
        use memra_gguf::tensor_contract::{
            CheckpointDialect, ContractOptions, OutputHead, TensorContract,
        };

        use memra_gguf::model_plan::{MlaAttentionPlan, StatePlan};

        // EVERY MLA extent DISTINCT (heads 2, q_lora 3, kv_rank 6, nope 4, v 5). The shared tiny
        // plan runs 2/4/4/4/4, where a transposed key plane has the SAME shape as a correct one
        // and this pin would be a tautology.
        let mut plan = kpool_mla_reference_plan();
        let AttentionPlan::Mla(MlaAttentionPlan::LatentKv {
            q_lora_rank,
            kv_lora_rank,
            qk_head_dim,
            value_head_dim,
            ..
        }) = &mut plan.layers[1].attention
        else {
            panic!("layer 1 of the tiny plan must be MLA LatentKv");
        };
        *q_lora_rank = 3;
        *kv_lora_rank = 6;
        *qk_head_dim = 4;
        *value_head_dim = 5;
        plan.layers[1].state = StatePlan::LatentKvCache {
            width: 6,
            index_width: 8,
        };
        let fixture = deterministic_fixture(&plan).unwrap();
        let contract = TensorContract::for_plan(
            &plan,
            CheckpointDialect::Gguf,
            ContractOptions {
                output_head: OutputHead::TiedToEmbedding,
            },
        )
        .unwrap();
        let mut checked = 0;
        for requirement in &contract.requirements {
            let Some(tensor) = fixture.weights.get(&requirement.id) else {
                continue;
            };
            // GGUF `ne` is fastest-axis-first; the fixture states row-major shapes.
            let mut wanted: Vec<usize> = requirement.shape.iter().map(|&d| d as usize).collect();
            wanted.reverse();
            let TensorId::Layer { tensor: kind, .. } = requirement.id else {
                continue;
            };
            if !matches!(
                kind,
                LayerTensor::MlaKeyUp | LayerTensor::MlaValueUp | LayerTensor::MlaQueryUp
            ) {
                continue;
            }
            assert_eq!(
                tensor.shape, wanted,
                "{:?}: fixture shape {:?} but the contract declares ne {:?}",
                requirement.id, tensor.shape, requirement.shape
            );
            checked += 1;
        }
        assert!(checked >= 3, "the plan must exercise the MLA planes");
    }

    /// The glm5_next-shaped tiny plan: one KDA layer, one MLA+k-pool-indexer layer, sigmoid MoE,
    /// hyper-connections with the mean collapse. Shared by the execution gate and the
    /// fixture-vs-contract shape pin so both describe the SAME plan.
    fn kpool_mla_reference_plan() -> ModelPlan {
        use memra_gguf::model_plan::{
            DenseMlpPlan, KimiDeltaNetPlan, KpoolPlan, MlaAttentionPlan, MoeMlpPlan, RopeFactors,
            RopePlan, RouterPlan, SharedMlpPlan, SparseIndexPlan, StatePlan,
        };

        let config = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3","num_hidden_layers":2,"hidden_size":8,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32,
            "rms_norm_eps":0.00001}"#,
        ));
        let mut plan = ModelPlan::compile(&config).unwrap();
        plan.layers[0].attention = AttentionPlan::KimiDeltaNet(KimiDeltaNetPlan {
            num_heads: 2,
            head_dim: 4,
            conv_kernel: 3,
            gate_lower_bound: -5.0,
        });
        plan.layers[0].state = StatePlan::Recurrent {
            conv_width: 24,
            conv_kernel: 3,
            state_width: 32,
        };
        plan.layers[0].mlp = MlpPlan::Dense(DenseMlpPlan {
            intermediate_size: 16,
            activation: ActivationPlan::SwiGluPreClamped { limit: 10.0 },
        });
        plan.layers[1].attention = AttentionPlan::Mla(MlaAttentionPlan::LatentKv {
            query_heads: 2,
            q_lora_rank: 4,
            kv_lora_rank: 4,
            qk_head_dim: 4,
            rope_head_dim: 0,
            value_head_dim: 4,
            rope: RopePlan {
                dimensions: 0,
                base: 10_000.0,
                factors: RopeFactors::None,
            },
            sparse_index: SparseIndexPlan::Own {
                heads: 2,
                head_dim: 4,
                top_k: 4,
                kpool: Some(KpoolPlan {
                    pool: 2,
                    always_select_tail: true,
                }),
            },
        });
        plan.layers[1].state = StatePlan::LatentKvCache {
            width: 4,
            index_width: 8,
        };
        plan.layers[1].mlp = MlpPlan::Moe(MoeMlpPlan {
            expert_count: 4,
            experts_per_token: 2,
            expert_intermediate_size: 4,
            router: RouterPlan::Sigmoid {
                normalize_selected: true,
                scaling_factor: 2.5,
                selection_bias: true,
            },
            shared: Some(SharedMlpPlan {
                intermediate_size: 4,
                gated: false,
            }),
            activation: ActivationPlan::SwiGluPreClamped { limit: 10.0 },
        });
        for layer in &mut plan.layers {
            layer.residual = ResidualTopology::HyperConnections {
                streams: 2,
                epsilon: 1e-6,
                sinkhorn_iterations: 2,
                collapse: HcCollapse::Mean,
            };
        }
        plan
    }

    #[test]
    fn glm5_shaped_tiny_plan_executes_kda_kpool_mla_and_mean_collapse_deterministically() {
        let plan = kpool_mla_reference_plan();
        let fixture = deterministic_fixture(&plan).unwrap();
        // The mean collapse owns no learned head tensors.
        assert!(!fixture.weights.contains_key(&TensorId::HyperHeadFunction));
        assert!(
            fixture
                .weights
                .contains_key(&layer_id(1, LayerTensor::SparseCompressorGate))
        );
        let output = execute(&plan, &fixture.weights, &fixture.token_ids).unwrap();
        assert_eq!(output.logits.len(), fixture.token_ids.len() * 32);
        assert!(output.logits.iter().all(|value| value.is_finite()));
        assert!(matches!(
            output.state.layers[0],
            ReferenceLayerState::Recurrent { conv_width: 24, .. }
        ));
        assert!(matches!(
            output.state.layers[1],
            ReferenceLayerState::LatentKv { width: 4, .. }
        ));
        let second = execute(&plan, &fixture.weights, &fixture.token_ids).unwrap();
        assert_eq!(
            output
                .logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            second
                .logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn kimi_delta_net_matches_hand_derived_three_token_recurrence() {
        use memra_gguf::model_plan::KimiDeltaNetPlan;

        let plan = KimiDeltaNetPlan {
            num_heads: 1,
            head_dim: 2,
            conv_kernel: 2,
            gate_lower_bound: -5.0,
        };
        let x = [[0.5f32, -0.3], [0.1, 0.8], [-0.6, 0.2]];
        let wq = [[0.7f32, -0.2], [0.3, 0.5]];
        let wk = [[0.4f32, 0.1], [-0.3, 0.6]];
        let wv = [[0.9f32, 0.2], [-0.1, 0.8]];
        let q_conv = [[0.3f32, 0.7], [-0.2, 0.9]];
        let k_conv = [[0.5f32, 0.5], [0.1, 0.8]];
        let v_conv = [[0.2f32, 0.6], [0.4, 0.4]];
        let f_a = [[0.6f32, -0.4], [0.2, 0.3]];
        let f_b = [[0.5f32, 0.1], [-0.2, 0.7]];
        let dt_bias = [0.05f32, -0.1];
        let a_log = [0.2f32];
        let b_proj = [[0.4f32, -0.6]];
        let g_a = [[0.3f32, 0.2], [-0.5, 0.4]];
        let g_b = [[0.6f32, -0.3], [0.2, 0.5]];
        let o_norm = [1.0f32, 1.5];
        let wo = [[0.8f32, -0.4], [0.3, 0.9]];

        let mut weights = ReferenceWeights::new();
        let flat = |rows: &[[f32; 2]]| -> Vec<f32> { rows.iter().flatten().copied().collect() };
        weights.insert(
            layer_id(0, LayerTensor::KdaQuery),
            weight(&[2, 2], &flat(&wq)),
        );
        weights.insert(
            layer_id(0, LayerTensor::KdaKey),
            weight(&[2, 2], &flat(&wk)),
        );
        weights.insert(
            layer_id(0, LayerTensor::KdaValue),
            weight(&[2, 2], &flat(&wv)),
        );
        weights.insert(
            layer_id(0, LayerTensor::KdaQueryConv),
            weight(&[2, 2], &flat(&q_conv)),
        );
        weights.insert(
            layer_id(0, LayerTensor::KdaKeyConv),
            weight(&[2, 2], &flat(&k_conv)),
        );
        weights.insert(
            layer_id(0, LayerTensor::KdaValueConv),
            weight(&[2, 2], &flat(&v_conv)),
        );
        weights.insert(
            layer_id(0, LayerTensor::KdaForgetDown),
            weight(&[2, 2], &flat(&f_a)),
        );
        weights.insert(
            layer_id(0, LayerTensor::KdaForgetUp),
            weight(&[2, 2], &flat(&f_b)),
        );
        weights.insert(layer_id(0, LayerTensor::KdaDtBias), weight(&[2], &dt_bias));
        weights.insert(layer_id(0, LayerTensor::KdaALog), weight(&[1], &a_log));
        weights.insert(
            layer_id(0, LayerTensor::KdaBeta),
            weight(&[1, 2], &flat(&b_proj)),
        );
        weights.insert(
            layer_id(0, LayerTensor::KdaGateDown),
            weight(&[2, 2], &flat(&g_a)),
        );
        weights.insert(
            layer_id(0, LayerTensor::KdaGateUp),
            weight(&[2, 2], &flat(&g_b)),
        );
        weights.insert(
            layer_id(0, LayerTensor::KdaOutputNorm),
            weight(&[2], &o_norm),
        );
        weights.insert(
            layer_id(0, LayerTensor::KdaOutput),
            weight(&[2, 2], &flat(&wo)),
        );

        let x_flat: Vec<f32> = x.iter().flatten().copied().collect();
        let (output, _) = kimi_delta_net(0, &plan, 1e-5, &weights, &x_flat, 3, 2).unwrap();

        // Duplicated arithmetic, written independently of the operator.
        let sig = |value: f32| 1.0 / (1.0 + (-value).exp());
        let act = |value: f32| value * (1.0 / (1.0 + (-value).exp()));
        let mat2 = |m: &[[f32; 2]; 2], v: [f32; 2]| {
            [
                m[0][0] * v[0] + m[0][1] * v[1],
                m[1][0] * v[0] + m[1][1] * v[1],
            ]
        };
        let mut q_proj = [[0.0f32; 2]; 3];
        let mut k_proj = [[0.0f32; 2]; 3];
        let mut v_proj = [[0.0f32; 2]; 3];
        for token in 0..3 {
            q_proj[token] = mat2(&wq, x[token]);
            k_proj[token] = mat2(&wk, x[token]);
            v_proj[token] = mat2(&wv, x[token]);
        }
        let causal_conv = |proj: &[[f32; 2]; 3], conv: &[[f32; 2]; 2]| {
            let mut out = [[0.0f32; 2]; 3];
            for token in 0..3 {
                for channel in 0..2 {
                    let previous = if token == 0 {
                        0.0
                    } else {
                        proj[token - 1][channel]
                    };
                    out[token][channel] =
                        act(conv[channel][0] * previous + conv[channel][1] * proj[token][channel]);
                }
            }
            out
        };
        let mut q = causal_conv(&q_proj, &q_conv);
        let mut k = causal_conv(&k_proj, &k_conv);
        let v = causal_conv(&v_proj, &v_conv);
        for token in 0..3 {
            let q_inv = 1.0 / (q[token][0] * q[token][0] + q[token][1] * q[token][1] + 1e-6).sqrt();
            let k_inv = 1.0 / (k[token][0] * k[token][0] + k[token][1] * k[token][1] + 1e-6).sqrt();
            for channel in 0..2 {
                q[token][channel] *= q_inv * (1.0 / 2.0f32.sqrt());
                k[token][channel] *= k_inv;
            }
        }
        let decay_rate = a_log[0].exp();
        let mut expected = Vec::new();
        let mut state = [[0.0f32; 2]; 2];
        for token in 0..3 {
            let f_lin = mat2(&f_b, mat2(&f_a, x[token]));
            let g = [
                -5.0 * sig(decay_rate * (f_lin[0] + dt_bias[0])),
                -5.0 * sig(decay_rate * (f_lin[1] + dt_bias[1])),
            ];
            let beta = sig(b_proj[0][0] * x[token][0] + b_proj[0][1] * x[token][1]);
            for key_index in 0..2 {
                #[allow(clippy::needless_range_loop)]
                // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
                for value_index in 0..2 {
                    state[key_index][value_index] *= g[key_index].exp();
                }
            }
            let mut core = [0.0f32; 2];
            for value_index in 0..2 {
                let memory =
                    state[0][value_index] * k[token][0] + state[1][value_index] * k[token][1];
                let delta = (v[token][value_index] - memory) * beta;
                state[0][value_index] += k[token][0] * delta;
                state[1][value_index] += k[token][1] * delta;
            }
            for value_index in 0..2 {
                core[value_index] =
                    state[0][value_index] * q[token][0] + state[1][value_index] * q[token][1];
            }
            let gate = mat2(&g_b, mat2(&g_a, x[token]));
            let mean_square = (core[0] * core[0] + core[1] * core[1]) / 2.0;
            let inverse = 1.0 / (mean_square + 1e-5).sqrt();
            let gated = [
                core[0] * inverse * o_norm[0] * sig(gate[0]),
                core[1] * inverse * o_norm[1] * sig(gate[1]),
            ];
            let final_row = mat2(&wo, gated);
            expected.extend_from_slice(&final_row);
        }
        assert_eq!(output.len(), expected.len());
        for (index, (actual, wanted)) in output.iter().zip(&expected).enumerate() {
            assert!(
                (actual - wanted).abs() < 1e-5,
                "output[{index}] = {actual}, expected {wanted}"
            );
        }
    }

    #[test]
    fn kpool_indexer_selects_causal_pools_and_appends_visible_tail() {
        use memra_gguf::model_plan::KpoolPlan;

        let tokens = 8;
        let hidden = 2;
        let q_rank = 2;
        let identity = [1.0f32, 0.0, 0.0, 1.0];
        let mut weights = ReferenceWeights::new();
        weights.insert(
            layer_id(0, LayerTensor::SparseQuery),
            weight(&[2, 2], &identity),
        );
        weights.insert(
            layer_id(0, LayerTensor::SparseKey),
            weight(&[2, 2], &identity),
        );
        weights.insert(
            layer_id(0, LayerTensor::SparseKeyNorm),
            weight(&[2], &[1.0, 1.0]),
        );
        weights.insert(
            layer_id(0, LayerTensor::SparseKeyNormBias),
            weight(&[2], &[0.0, 0.0]),
        );
        weights.insert(
            layer_id(0, LayerTensor::SparseProjection),
            weight(&[1, 2], &[1.0, 1.0]),
        );
        weights.insert(
            layer_id(0, LayerTensor::SparseCompressorGate),
            weight(&[2, 2], &[0.3, -0.2, 0.1, 0.4]),
        );
        weights.insert(
            layer_id(0, LayerTensor::SparseCompressorPosition),
            weight(&[4, 2], &[0.1, 0.0, -0.1, 0.2, 0.05, -0.05, 0.0, 0.1]),
        );
        let x: Vec<f32> = (0..tokens * hidden)
            .map(|index| ((index % 5) as f32 - 2.0) * 0.3)
            .collect();
        let q_resid = x.clone();

        // top_k 8 / pool 4 = a 2-pool budget, so every causally visible pool selects.
        let kpool = KpoolPlan {
            pool: 4,
            always_select_tail: true,
        };
        let allowed = kpool_allowed_tokens(
            0, 1, 2, 8, &kpool, &weights, &x, &q_resid, tokens, hidden, q_rank,
        )
        .unwrap();
        // Query 7 sees both complete pools; 8 % 4 == 0 leaves no tail.
        assert_eq!(allowed[7], (0..8).collect::<Vec<_>>());
        // Query 6: pool [4..=7] ends past the query, so only [0..=3] plus tail [4,5,6].
        assert_eq!(allowed[6], vec![0, 1, 2, 3, 4, 5, 6]);
        // Query 2 precedes any complete pool: tail only.
        assert_eq!(allowed[2], vec![0, 1, 2]);

        // Without the tail, queries before the first complete pool have no candidates.
        let no_tail = KpoolPlan {
            pool: 4,
            always_select_tail: false,
        };
        let error = kpool_allowed_tokens(
            0, 1, 2, 8, &no_tail, &weights, &x, &q_resid, tokens, hidden, q_rank,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ReferenceError::InvalidPlan {
                reason: "k-pool selection produced an empty candidate set for a query",
                ..
            }
        ));
    }
}
