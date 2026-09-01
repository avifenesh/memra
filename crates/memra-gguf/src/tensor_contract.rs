//! Strict checkpoint-to-semantic tensor binding.
//!
//! Contracts are built from `ModelPlan`, then bound against a cheap tensor census before any
//! tensor bytes are uploaded. Every physical tensor must bind exactly once; missing, extra, and
//! ambiguous names are compile errors.

use std::collections::{BTreeMap, BTreeSet};

use crate::config::{Arch, AttentionGateKind};
use crate::model_plan::{
    AttentionPlan, LayerPlan, MlaAttentionPlan, MlpPlan, ModelPlan, MoeMlpPlan, ResidualTopology,
    RopeFactors, RouterPlan, SparseIndexPlan, TensorPresence, ValueProjection, WeightTransform,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointDialect {
    Gguf,
    HfSafetensors,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputHead {
    Separate,
    TiedToEmbedding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContractOptions {
    pub output_head: OutputHead,
}

impl Default for ContractOptions {
    fn default() -> Self {
        Self {
            output_head: OutputHead::Separate,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TensorId {
    TokenEmbedding,
    RopeFactors,
    HyperHeadBase,
    HyperHeadFunction,
    HyperHeadScale,
    OutputNorm,
    OutputProjection,
    Layer {
        index: u32,
        tensor: LayerTensor,
    },
    Expert {
        layer: u32,
        expert: u32,
        tensor: ExpertTensor,
    },
    Mtp {
        depth: u32,
        tensor: MtpTensor,
    },
    Dspark(DsparkTensor),
    Vision {
        layer: Option<u32>,
        tensor: VisionTensor,
    },
    QuantAux {
        tensor: Box<TensorId>,
        kind: QuantAuxTensor,
    },
    Family {
        family: &'static str,
        key: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QuantAuxTensor {
    WeightScale,
    InputScale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExpertTensor {
    Gate,
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VisionTensor {
    PatchProjection,
    PositionEmbedding,
    StandardizeBias,
    StandardizeScale,
    OutputProjection,
    InputNorm,
    PostAttentionNorm,
    PreMlpNorm,
    PostMlpNorm,
    Query,
    Key,
    Value,
    AttentionOutput,
    QueryNorm,
    KeyNorm,
    MlpGate,
    MlpUp,
    MlpDown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DsparkTensor {
    MainProjection,
    MainNorm,
    OutputNorm,
    MarkovEmbedding,
    MarkovOutput,
    ConfidenceProjection,
    HeadHyperBase,
    HeadHyperFunction,
    HeadHyperScale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MtpTensor {
    EmbeddingNorm,
    HiddenNorm,
    FusionProjection,
    OutputNorm,
    OutputProjection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LayerTensor {
    PreAttentionNorm,
    PostAttentionNorm,
    Query,
    Key,
    Value,
    AttentionOutput,
    QueryNorm,
    KeyNorm,
    AttentionGate,
    HyperAttentionBase,
    HyperAttentionFunction,
    HyperAttentionScale,
    GdnQkv,
    GdnGate,
    GdnBeta,
    GdnAlpha,
    GdnA,
    GdnDtBias,
    GdnConv1d,
    GdnNorm,
    GdnOutput,
    MlaQueryDown,
    MlaQueryDownNorm,
    MlaQueryUp,
    MlaKvDown,
    MlaKvDownNorm,
    MlaKeyUp,
    MlaValueUp,
    MlaKvSource,
    MlaOutputDown,
    MlaOutput,
    AttentionSink,
    KvCompressorKeyValue,
    KvCompressorGate,
    KvCompressorNorm,
    KvCompressorPosition,
    SparseQuery,
    SparseKey,
    SparseKeyNorm,
    SparseKeyNormBias,
    SparseProjection,
    SparseCompressorKeyValue,
    SparseCompressorGate,
    SparseCompressorNorm,
    SparseCompressorPosition,
    PreMlpNorm,
    HyperMlpBase,
    HyperMlpFunction,
    HyperMlpScale,
    PostMlpNorm,
    PostSharedMlpNorm,
    PreRoutedMlpNorm,
    PostRoutedMlpNorm,
    LayerScale,
    MlpGate,
    MlpUp,
    MlpDown,
    MoeRouter,
    MoeRouterBias,
    MoeRouterScale,
    MoeTokenToExpert,
    MoeExpertGateUpBank,
    MoeExpertGateBank,
    MoeExpertUpBank,
    MoeExpertDownBank,
    MoeExpertOutputScale,
    SharedMlpGate,
    SharedMlpUp,
    SharedMlpDown,
    SharedMlpInputGate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorOwner {
    Global,
    Layer(u32),
    Mtp(u32),
    Vision(Option<u32>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorTransform {
    Identity,
    NormAddOne,
    QkvVReorderRows,
    ZReorderRows,
    AbReorderRows,
    NegExpReorderHeads,
    ReorderHeads,
    Conv1dSqueezeReorder,
    OutReorderColumns,
    StackExperts,
    SplitExpertGateUp,
    SplitMlaKv,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRequirement {
    pub id: TensorId,
    pub names: Vec<String>,
    pub match_mode: TensorMatch,
    pub shape: Vec<u64>,
    pub owner: TensorOwner,
    pub transform: TensorTransform,
    pub quant: QuantConstraint,
    pub auxiliaries: Option<Vec<String>>,
    pub required: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorMatch {
    /// Exactly one of the accepted aliases may exist.
    OneOf,
    /// Every listed physical tensor is one semantic tensor bank.
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantConstraint {
    FloatOnly,
    Weight,
    ExactFloat(FloatType),
    Nvfp4,
    Mxfp4,
    Fp8Block128,
    I64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorCensusEntry {
    pub name: String,
    /// Checkpoint-native dimension order: safetensors outer-to-inner, GGUF inner-to-outer.
    pub shape: Vec<u64>,
    pub storage: StorageLayout,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageLayout {
    Float(FloatType),
    Quantized(QuantLayout),
    Integer(IntegerType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatType {
    F32,
    F16,
    Bf16,
    Fp8E4m3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerType {
    I64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantLayout {
    pub format: String,
    pub block_shape: Vec<u32>,
    /// Semantic names of scale/zero-point planes required to decode the weight.
    pub auxiliaries: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorContract {
    pub dialect: CheckpointDialect,
    pub requirements: Vec<TensorRequirement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundTensorContract {
    pub tensors: BTreeMap<TensorId, BoundTensor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundTensor {
    pub checkpoint_names: Vec<String>,
    pub shapes: Vec<Vec<u64>>,
    pub storage: Vec<StorageLayout>,
    pub owner: TensorOwner,
    pub transform: TensorTransform,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorContractError {
    UnsupportedPlanOperation {
        operation: &'static str,
    },
    DuplicateCensusName {
        name: String,
    },
    DuplicateTensorId {
        id: TensorId,
    },
    Missing {
        id: TensorId,
        accepted_names: Vec<String>,
    },
    Ambiguous {
        id: TensorId,
        matched_names: Vec<String>,
    },
    ClaimedByMultipleIds {
        name: String,
        first: TensorId,
        second: TensorId,
    },
    ShapeMismatch {
        id: TensorId,
        name: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    QuantLayoutMismatch {
        id: TensorId,
        name: String,
        expected: QuantConstraint,
        actual: StorageLayout,
    },
    AuxiliaryLayoutMismatch {
        id: TensorId,
        name: String,
        expected: Vec<String>,
        actual: Vec<String>,
    },
    Extra {
        names: Vec<String>,
    },
}

impl std::fmt::Display for TensorContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedPlanOperation { operation } => {
                write!(
                    f,
                    "no semantic tensor schema for plan operation {operation}"
                )
            }
            Self::DuplicateCensusName { name } => write!(f, "duplicate census tensor {name}"),
            Self::DuplicateTensorId { id } => write!(f, "duplicate semantic tensor id {id:?}"),
            Self::Missing { id, accepted_names } => {
                write!(
                    f,
                    "missing tensor {id:?}; accepted names: {accepted_names:?}"
                )
            }
            Self::Ambiguous { id, matched_names } => {
                write!(f, "ambiguous tensor {id:?}; matched: {matched_names:?}")
            }
            Self::ClaimedByMultipleIds {
                name,
                first,
                second,
            } => write!(
                f,
                "checkpoint tensor {name} is claimed by both {first:?} and {second:?}"
            ),
            Self::ShapeMismatch {
                id,
                name,
                expected,
                actual,
            } => write!(
                f,
                "tensor {id:?} ({name}) shape mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::QuantLayoutMismatch {
                id,
                name,
                expected,
                actual,
            } => write!(
                f,
                "tensor {id:?} ({name}) storage mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::AuxiliaryLayoutMismatch {
                id,
                name,
                expected,
                actual,
            } => write!(
                f,
                "tensor {id:?} ({name}) auxiliary planes mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::Extra { names } => write!(f, "extra checkpoint tensors: {names:?}"),
        }
    }
}

impl std::error::Error for TensorContractError {}

impl TensorContract {
    pub fn for_plan(
        plan: &ModelPlan,
        dialect: CheckpointDialect,
        options: ContractOptions,
    ) -> Result<Self, TensorContractError> {
        let mut builder = ContractBuilder::new(dialect);
        builder.weight(
            TensorId::TokenEmbedding,
            top_name(dialect, "token_embd.weight", "model.embed_tokens.weight"),
            matrix_shape(dialect, plan.vocab_size, plan.hidden_size),
            TensorOwner::Global,
            TensorTransform::Identity,
        );
        if dialect == CheckpointDialect::Gguf {
            if let Some(width) = rope_factor_width(plan) {
                builder.float(
                    TensorId::RopeFactors,
                    "rope_freqs.weight".to_string(),
                    vec![width as u64],
                    TensorOwner::Global,
                    TensorTransform::Identity,
                    true,
                );
            }
        }
        builder.float(
            TensorId::OutputNorm,
            top_name(dialect, "output_norm.weight", "model.norm.weight"),
            vec![plan.hidden_size as u64],
            TensorOwner::Global,
            norm_transform(plan.output_norm.weight_transform),
            true,
        );
        if options.output_head == OutputHead::Separate {
            builder.weight(
                TensorId::OutputProjection,
                top_name(dialect, "output.weight", "lm_head.weight"),
                matrix_shape(dialect, plan.vocab_size, plan.hidden_size),
                TensorOwner::Global,
                TensorTransform::Identity,
            );
        }

        for layer in &plan.layers {
            add_layer(&mut builder, plan, layer)?;
        }
        for block in &plan.mtp_blocks {
            if plan.arch == Arch::DeepSeekV4 {
                return Err(TensorContractError::UnsupportedPlanOperation {
                    operation: "deepseek_v4 MTP",
                });
            }
            let first = builder.requirements.len();
            add_layer(&mut builder, plan, &block.layer)?;
            rewrite_mtp_namespace(
                &mut builder.requirements[first..],
                plan,
                dialect,
                block.depth,
                block.layer.index,
            );
            add_mtp_glue(&mut builder, plan, block);
        }

        Ok(Self {
            dialect,
            requirements: builder.finish()?,
        })
    }

    pub fn bind(
        &self,
        census: &[TensorCensusEntry],
    ) -> Result<BoundTensorContract, TensorContractError> {
        let mut by_name = BTreeMap::new();
        for entry in census {
            if by_name.insert(entry.name.as_str(), entry).is_some() {
                return Err(TensorContractError::DuplicateCensusName {
                    name: entry.name.clone(),
                });
            }
        }

        let mut claims: BTreeMap<&str, TensorId> = BTreeMap::new();
        let mut tensors = BTreeMap::new();
        for requirement in &self.requirements {
            let matched: Vec<&TensorCensusEntry> = requirement
                .names
                .iter()
                .filter_map(|name| by_name.get(name.as_str()).copied())
                .collect();
            let missing_names: Vec<String> = requirement
                .names
                .iter()
                .filter(|name| !by_name.contains_key(name.as_str()))
                .cloned()
                .collect();
            let missing = match requirement.match_mode {
                TensorMatch::OneOf => matched.is_empty(),
                TensorMatch::All => !missing_names.is_empty(),
            };
            if missing {
                if requirement.required {
                    return Err(TensorContractError::Missing {
                        id: requirement.id.clone(),
                        accepted_names: if requirement.match_mode == TensorMatch::All {
                            missing_names
                        } else {
                            requirement.names.clone()
                        },
                    });
                }
                continue;
            }
            if requirement.match_mode == TensorMatch::OneOf && matched.len() > 1 {
                return Err(TensorContractError::Ambiguous {
                    id: requirement.id.clone(),
                    matched_names: matched.iter().map(|entry| entry.name.clone()).collect(),
                });
            }
            for entry in &matched {
                if let Some(first) = claims.insert(entry.name.as_str(), requirement.id.clone()) {
                    return Err(TensorContractError::ClaimedByMultipleIds {
                        name: entry.name.clone(),
                        first,
                        second: requirement.id.clone(),
                    });
                }
                if entry.shape != requirement.shape {
                    return Err(TensorContractError::ShapeMismatch {
                        id: requirement.id.clone(),
                        name: entry.name.clone(),
                        expected: requirement.shape.clone(),
                        actual: entry.shape.clone(),
                    });
                }
                if !requirement.quant.accepts(&entry.storage) {
                    return Err(TensorContractError::QuantLayoutMismatch {
                        id: requirement.id.clone(),
                        name: entry.name.clone(),
                        expected: requirement.quant,
                        actual: entry.storage.clone(),
                    });
                }
                let actual_auxiliaries = match &entry.storage {
                    StorageLayout::Quantized(layout) => layout.auxiliaries.clone(),
                    _ => Vec::new(),
                };
                if let Some(expected) = requirement.auxiliaries.as_ref() {
                    if actual_auxiliaries != *expected {
                        return Err(TensorContractError::AuxiliaryLayoutMismatch {
                            id: requirement.id.clone(),
                            name: entry.name.clone(),
                            expected: expected.clone(),
                            actual: actual_auxiliaries,
                        });
                    }
                }
            }
            tensors.insert(
                requirement.id.clone(),
                BoundTensor {
                    checkpoint_names: matched.iter().map(|entry| entry.name.clone()).collect(),
                    shapes: matched.iter().map(|entry| entry.shape.clone()).collect(),
                    storage: matched.iter().map(|entry| entry.storage.clone()).collect(),
                    owner: requirement.owner,
                    transform: requirement.transform,
                },
            );
        }

        let mut extras: Vec<String> = census
            .iter()
            .filter(|entry| !claims.contains_key(entry.name.as_str()))
            .map(|entry| entry.name.clone())
            .collect();
        extras.sort();
        if !extras.is_empty() {
            return Err(TensorContractError::Extra { names: extras });
        }
        Ok(BoundTensorContract { tensors })
    }
}

fn rope_factor_width(plan: &ModelPlan) -> Option<u32> {
    plan.layers
        .iter()
        .chain(plan.mtp_blocks.iter().map(|block| &block.layer))
        .filter_map(|layer| match &layer.attention {
            AttentionPlan::Full(attention) | AttentionPlan::SlidingWindow { attention, .. } => {
                matches!(
                    attention.rope.factors,
                    RopeFactors::Checkpoint | RopeFactors::PartialRotary { .. }
                )
                .then_some(attention.rope.dimensions / 2)
            }
            AttentionPlan::Mla(mla) => match mla {
                MlaAttentionPlan::LatentKv { rope, .. }
                | MlaAttentionPlan::CompressedKv { rope, .. } => matches!(
                    rope.factors,
                    RopeFactors::Checkpoint | RopeFactors::PartialRotary { .. }
                )
                .then_some(rope.dimensions / 2),
            },
            AttentionPlan::GatedDeltaNet(_) => None,
        })
        .max()
}

impl QuantConstraint {
    fn accepts(self, storage: &StorageLayout) -> bool {
        match self {
            Self::FloatOnly => matches!(storage, StorageLayout::Float(_)),
            Self::Weight => true,
            Self::ExactFloat(expected) => {
                matches!(storage, StorageLayout::Float(actual) if *actual == expected)
            }
            Self::Nvfp4 => matches!(
                storage,
                StorageLayout::Quantized(layout)
                    if layout.format == "NVFP4" && layout.block_shape == [16]
            ),
            Self::Mxfp4 => matches!(
                storage,
                StorageLayout::Quantized(layout)
                    if layout.format == "MXFP4" && layout.block_shape == [32]
            ),
            Self::Fp8Block128 => matches!(
                storage,
                StorageLayout::Quantized(layout)
                    if layout.format == "FP8_E4M3" && layout.block_shape == [128, 128]
            ),
            Self::I64 => matches!(storage, StorageLayout::Integer(IntegerType::I64)),
        }
    }
}

struct ContractBuilder {
    dialect: CheckpointDialect,
    requirements: Vec<TensorRequirement>,
}

impl ContractBuilder {
    fn new(dialect: CheckpointDialect) -> Self {
        Self {
            dialect,
            requirements: Vec::new(),
        }
    }

    fn weight(
        &mut self,
        id: TensorId,
        name: String,
        shape: Vec<u64>,
        owner: TensorOwner,
        transform: TensorTransform,
    ) {
        let aux_id = id.clone();
        let aux_name = name.clone();
        self.requirements.push(TensorRequirement {
            id,
            names: vec![name],
            match_mode: TensorMatch::OneOf,
            shape,
            owner,
            transform: if self.dialect == CheckpointDialect::Gguf {
                TensorTransform::Identity
            } else {
                transform
            },
            quant: QuantConstraint::Weight,
            auxiliaries: None,
            required: true,
        });
        self.gguf_quant_auxiliaries(&aux_id, &[aux_name], owner);
    }

    fn weight_group(
        &mut self,
        id: TensorId,
        names: Vec<String>,
        shape: Vec<u64>,
        owner: TensorOwner,
        transform: TensorTransform,
    ) {
        self.requirements.push(TensorRequirement {
            id,
            names,
            match_mode: TensorMatch::All,
            shape,
            owner,
            transform: if self.dialect == CheckpointDialect::Gguf {
                TensorTransform::Identity
            } else {
                transform
            },
            quant: QuantConstraint::Weight,
            auxiliaries: None,
            required: true,
        });
    }

    fn weight_aliases(
        &mut self,
        id: TensorId,
        names: Vec<String>,
        shape: Vec<u64>,
        owner: TensorOwner,
        transform: TensorTransform,
        required: bool,
    ) {
        let aux_id = id.clone();
        let aux_names = names.clone();
        self.requirements.push(TensorRequirement {
            id,
            names,
            match_mode: TensorMatch::OneOf,
            shape,
            owner,
            transform: if self.dialect == CheckpointDialect::Gguf {
                TensorTransform::Identity
            } else {
                transform
            },
            quant: QuantConstraint::Weight,
            auxiliaries: None,
            required,
        });
        self.gguf_quant_auxiliaries(&aux_id, &aux_names, owner);
    }

    fn gguf_quant_auxiliaries(
        &mut self,
        tensor: &TensorId,
        weight_names: &[String],
        owner: TensorOwner,
    ) {
        for (kind, suffix) in [
            (QuantAuxTensor::WeightScale, ".scale"),
            (QuantAuxTensor::InputScale, ".input_scale"),
        ] {
            let names: Vec<_> = if self.dialect == CheckpointDialect::Gguf {
                weight_names
                    .iter()
                    .filter_map(|name| {
                        name.strip_suffix(".weight")
                            .map(|stem| format!("{stem}{suffix}"))
                    })
                    .collect()
            } else {
                Vec::new()
            };
            self.requirements.push(TensorRequirement {
                id: TensorId::QuantAux {
                    tensor: Box::new(tensor.clone()),
                    kind,
                },
                names,
                match_mode: TensorMatch::OneOf,
                shape: vec![1],
                owner,
                transform: TensorTransform::Identity,
                quant: QuantConstraint::FloatOnly,
                auxiliaries: None,
                required: false,
            });
        }
    }

    fn float(
        &mut self,
        id: TensorId,
        name: String,
        shape: Vec<u64>,
        owner: TensorOwner,
        transform: TensorTransform,
        required: bool,
    ) {
        self.requirements.push(TensorRequirement {
            id,
            names: vec![name],
            match_mode: TensorMatch::OneOf,
            shape,
            owner,
            transform: if self.dialect == CheckpointDialect::Gguf {
                TensorTransform::Identity
            } else {
                transform
            },
            quant: QuantConstraint::FloatOnly,
            auxiliaries: None,
            required,
        });
    }

    fn float_aliases(
        &mut self,
        id: TensorId,
        names: Vec<String>,
        shape: Vec<u64>,
        owner: TensorOwner,
        transform: TensorTransform,
        required: bool,
    ) {
        self.requirements.push(TensorRequirement {
            id,
            names,
            match_mode: TensorMatch::OneOf,
            shape,
            owner,
            transform: if self.dialect == CheckpointDialect::Gguf {
                TensorTransform::Identity
            } else {
                transform
            },
            quant: QuantConstraint::FloatOnly,
            auxiliaries: None,
            required,
        });
    }

    fn finish(self) -> Result<Vec<TensorRequirement>, TensorContractError> {
        let mut ids = BTreeSet::new();
        for requirement in &self.requirements {
            if !ids.insert(requirement.id.clone()) {
                return Err(TensorContractError::DuplicateTensorId {
                    id: requirement.id.clone(),
                });
            }
        }
        Ok(self.requirements)
    }
}

fn add_layer(
    builder: &mut ContractBuilder,
    plan: &ModelPlan,
    layer: &LayerPlan,
) -> Result<(), TensorContractError> {
    let index = layer.index;
    builder.float(
        layer_id(index, LayerTensor::PreAttentionNorm),
        layer_name(
            builder.dialect,
            index,
            "attn_norm.weight",
            "input_layernorm.weight",
        ),
        vec![plan.hidden_size as u64],
        TensorOwner::Layer(index),
        norm_transform(layer.pre_attention_norm.weight_transform),
        true,
    );
    match &layer.attention {
        AttentionPlan::Full(attention) | AttentionPlan::SlidingWindow { attention, .. } => {
            add_full_attention(builder, plan, index, attention);
        }
        AttentionPlan::GatedDeltaNet(gdn) => add_gdn(builder, plan, index, *gdn),
        AttentionPlan::Mla(mla) => add_mla(builder, plan, index, mla)?,
    }
    let gemma = matches!(layer.residual, ResidualTopology::Gemma { .. });
    if let ResidualTopology::Gemma {
        post_attention_norm,
        ..
    } = layer.residual
    {
        builder.float(
            layer_id(index, LayerTensor::PostAttentionNorm),
            layer_name(
                builder.dialect,
                index,
                "post_attention_norm.weight",
                "post_attention_layernorm.weight",
            ),
            vec![plan.hidden_size as u64],
            TensorOwner::Layer(index),
            norm_transform(post_attention_norm.weight_transform),
            true,
        );
    }
    let pre_mlp_names = if builder.dialect == CheckpointDialect::Gguf && !gemma {
        vec![
            format!("blk.{index}.ffn_norm.weight"),
            format!("blk.{index}.post_attention_norm.weight"),
        ]
    } else {
        vec![layer_name(
            builder.dialect,
            index,
            "ffn_norm.weight",
            if gemma {
                "pre_feedforward_layernorm.weight"
            } else {
                "post_attention_layernorm.weight"
            },
        )]
    };
    builder.float_aliases(
        layer_id(index, LayerTensor::PreMlpNorm),
        pre_mlp_names,
        vec![plan.hidden_size as u64],
        TensorOwner::Layer(index),
        norm_transform(layer.pre_mlp_norm.weight_transform),
        true,
    );
    match (&layer.mlp, layer.residual) {
        (MlpPlan::Dense(mlp), _) => add_dense_mlp(builder, plan, index, mlp.intermediate_size),
        (
            MlpPlan::Moe(moe),
            ResidualTopology::Gemma {
                parallel_moe: Some(parallel),
                ..
            },
        ) => add_gemma_parallel_moe(builder, plan, index, moe, parallel)?,
        (MlpPlan::Moe(moe), _) => add_moe_mlp(builder, plan, index, moe)?,
    }
    if let ResidualTopology::Gemma {
        post_mlp_norm,
        layer_scale: _,
        ..
    } = layer.residual
    {
        builder.float(
            layer_id(index, LayerTensor::PostMlpNorm),
            layer_name(
                builder.dialect,
                index,
                "post_ffw_norm.weight",
                "post_feedforward_layernorm.weight",
            ),
            vec![plan.hidden_size as u64],
            TensorOwner::Layer(index),
            norm_transform(post_mlp_norm.weight_transform),
            true,
        );
        builder.float(
            layer_id(index, LayerTensor::LayerScale),
            layer_name(
                builder.dialect,
                index,
                "layer_output_scale.weight",
                "layer_scalar",
            ),
            vec![1],
            TensorOwner::Layer(index),
            TensorTransform::Identity,
            true,
        );
    }
    Ok(())
}

fn add_gemma_parallel_moe(
    builder: &mut ContractBuilder,
    plan: &ModelPlan,
    index: u32,
    moe: &MoeMlpPlan,
    parallel: crate::model_plan::GemmaParallelMoePlan,
) -> Result<(), TensorContractError> {
    if builder.dialect != CheckpointDialect::HfSafetensors {
        return Err(TensorContractError::UnsupportedPlanOperation {
            operation: "gemma parallel MoE non-safetensors schema",
        });
    }
    let owner = TensorOwner::Layer(index);
    let hidden = plan.hidden_size;
    let expert_ff = moe.expert_intermediate_size;
    let shared = moe
        .shared
        .as_ref()
        .ok_or(TensorContractError::UnsupportedPlanOperation {
            operation: "gemma parallel MoE without shared branch",
        })?;
    for (tensor, suffix, output, input) in [
        (
            LayerTensor::SharedMlpGate,
            "mlp.gate_proj.weight",
            shared.intermediate_size,
            hidden,
        ),
        (
            LayerTensor::SharedMlpUp,
            "mlp.up_proj.weight",
            shared.intermediate_size,
            hidden,
        ),
        (
            LayerTensor::SharedMlpDown,
            "mlp.down_proj.weight",
            hidden,
            shared.intermediate_size,
        ),
        (
            LayerTensor::MoeRouter,
            "router.proj.weight",
            moe.expert_count,
            hidden,
        ),
    ] {
        builder.weight(
            layer_id(index, tensor),
            format!("model.layers.{index}.{suffix}"),
            vec![output as u64, input as u64],
            owner,
            TensorTransform::Identity,
        );
    }
    builder.weight(
        layer_id(index, LayerTensor::MoeExpertGateUpBank),
        format!("model.layers.{index}.experts.gate_up_proj"),
        vec![
            moe.expert_count as u64,
            (2 * expert_ff) as u64,
            hidden as u64,
        ],
        owner,
        TensorTransform::SplitExpertGateUp,
    );
    builder.weight(
        layer_id(index, LayerTensor::MoeExpertDownBank),
        format!("model.layers.{index}.experts.down_proj"),
        vec![moe.expert_count as u64, hidden as u64, expert_ff as u64],
        owner,
        TensorTransform::Identity,
    );
    for (tensor, suffix, width, norm) in [
        (
            LayerTensor::PostSharedMlpNorm,
            "post_feedforward_layernorm_1.weight",
            hidden,
            parallel.shared_post_norm,
        ),
        (
            LayerTensor::PreRoutedMlpNorm,
            "pre_feedforward_layernorm_2.weight",
            hidden,
            parallel.routed_pre_norm,
        ),
        (
            LayerTensor::PostRoutedMlpNorm,
            "post_feedforward_layernorm_2.weight",
            hidden,
            parallel.routed_post_norm,
        ),
    ] {
        builder.float(
            layer_id(index, tensor),
            format!("model.layers.{index}.{suffix}"),
            vec![width as u64],
            owner,
            norm_transform(norm.weight_transform),
            true,
        );
    }
    if parallel.router_input_scale {
        builder.float(
            layer_id(index, LayerTensor::MoeRouterScale),
            format!("model.layers.{index}.router.scale"),
            vec![hidden as u64],
            owner,
            TensorTransform::Identity,
            true,
        );
    }
    if parallel.per_expert_output_scale {
        builder.float(
            layer_id(index, LayerTensor::MoeExpertOutputScale),
            format!("model.layers.{index}.router.per_expert_scale"),
            vec![moe.expert_count as u64],
            owner,
            TensorTransform::Identity,
            true,
        );
    }
    Ok(())
}

fn rewrite_mtp_namespace(
    requirements: &mut [TensorRequirement],
    plan: &ModelPlan,
    dialect: CheckpointDialect,
    depth: u32,
    global_index: u32,
) {
    if dialect != CheckpointDialect::HfSafetensors
        || !matches!(plan.arch, Arch::Qwen35 | Arch::Qwen35Moe)
    {
        return;
    }
    let source = format!("model.layers.{global_index}.");
    let target = format!("mtp.layers.{depth}.");
    for requirement in requirements {
        for name in &mut requirement.names {
            if let Some(suffix) = name.strip_prefix(&source) {
                *name = format!("{target}{suffix}");
            }
        }
    }
}

fn add_mtp_glue(
    builder: &mut ContractBuilder,
    plan: &ModelPlan,
    block: &crate::model_plan::MtpBlockPlan,
) {
    let depth = block.depth;
    let global_index = block.layer.index;
    let owner = TensorOwner::Mtp(depth);
    let (enorm, hnorm, fusion, output_norm) = match builder.dialect {
        CheckpointDialect::Gguf => (
            format!("blk.{global_index}.nextn.enorm.weight"),
            format!("blk.{global_index}.nextn.hnorm.weight"),
            format!("blk.{global_index}.nextn.eh_proj.weight"),
            format!("blk.{global_index}.nextn.shared_head_norm.weight"),
        ),
        CheckpointDialect::HfSafetensors if matches!(plan.arch, Arch::Qwen35 | Arch::Qwen35Moe) => {
            (
                "mtp.pre_fc_norm_embedding.weight".to_string(),
                "mtp.pre_fc_norm_hidden.weight".to_string(),
                "mtp.fc.weight".to_string(),
                "mtp.norm.weight".to_string(),
            )
        }
        CheckpointDialect::HfSafetensors => (
            format!("model.layers.{global_index}.enorm.weight"),
            format!("model.layers.{global_index}.hnorm.weight"),
            format!("model.layers.{global_index}.eh_proj.weight"),
            format!("model.layers.{global_index}.final_layernorm.weight"),
        ),
    };
    for (tensor, name) in [
        (MtpTensor::EmbeddingNorm, enorm),
        (MtpTensor::HiddenNorm, hnorm),
    ] {
        builder.float(
            TensorId::Mtp { depth, tensor },
            name,
            vec![plan.hidden_size as u64],
            owner,
            norm_transform(match tensor {
                MtpTensor::EmbeddingNorm => block.input.embedding_norm.weight_transform,
                MtpTensor::HiddenNorm => block.input.hidden_norm.weight_transform,
                _ => unreachable!(),
            }),
            true,
        );
    }
    builder.weight(
        TensorId::Mtp {
            depth,
            tensor: MtpTensor::FusionProjection,
        },
        fusion,
        matrix_shape(builder.dialect, plan.hidden_size, 2 * plan.hidden_size),
        owner,
        TensorTransform::Identity,
    );
    builder.float_aliases(
        TensorId::Mtp {
            depth,
            tensor: MtpTensor::OutputNorm,
        },
        vec![output_norm],
        vec![plan.hidden_size as u64],
        owner,
        norm_transform(plan.output_norm.weight_transform),
        false,
    );
    let head_names = match builder.dialect {
        CheckpointDialect::Gguf => vec![
            format!("blk.{global_index}.nextn.shared_head_head.weight"),
            format!("blk.{global_index}.nextn.shared_head.weight"),
        ],
        CheckpointDialect::HfSafetensors => Vec::new(),
    };
    if !head_names.is_empty() {
        builder.weight_aliases(
            TensorId::Mtp {
                depth,
                tensor: MtpTensor::OutputProjection,
            },
            head_names,
            matrix_shape(builder.dialect, plan.vocab_size, plan.hidden_size),
            owner,
            TensorTransform::Identity,
            false,
        );
    }
}

fn add_full_attention(
    builder: &mut ContractBuilder,
    plan: &ModelPlan,
    index: u32,
    attention: &crate::model_plan::FullAttentionPlan,
) {
    let gate_multiplier = if attention.output_gate == AttentionGateKind::FusedQ {
        2
    } else {
        1
    };
    let query_width = attention.query_heads * attention.key_head_dim * gate_multiplier;
    let key_width = attention.kv_heads * attention.key_head_dim;
    let value_width = attention.kv_heads * attention.value_head_dim;
    let output_width = attention.query_heads * attention.value_head_dim;
    for (tensor, gguf, hf, out, input) in [
        (
            LayerTensor::Query,
            "attn_q.weight",
            "self_attn.q_proj.weight",
            query_width,
            plan.hidden_size,
        ),
        (
            LayerTensor::Key,
            "attn_k.weight",
            "self_attn.k_proj.weight",
            key_width,
            plan.hidden_size,
        ),
        (
            LayerTensor::AttentionOutput,
            "attn_output.weight",
            "self_attn.o_proj.weight",
            plan.hidden_size,
            output_width,
        ),
    ] {
        builder.weight(
            layer_id(index, tensor),
            layer_name(builder.dialect, index, gguf, hf),
            matrix_shape(builder.dialect, out, input),
            TensorOwner::Layer(index),
            TensorTransform::Identity,
        );
    }
    if attention.value_projection == ValueProjection::Separate {
        builder.weight(
            layer_id(index, LayerTensor::Value),
            layer_name(
                builder.dialect,
                index,
                "attn_v.weight",
                "self_attn.v_proj.weight",
            ),
            matrix_shape(builder.dialect, value_width, plan.hidden_size),
            TensorOwner::Layer(index),
            TensorTransform::Identity,
        );
    }
    let qk_required = attention.qk_norm == TensorPresence::Required;
    if attention.qk_norm != TensorPresence::Absent {
        builder.float(
            layer_id(index, LayerTensor::QueryNorm),
            layer_name(
                builder.dialect,
                index,
                "attn_q_norm.weight",
                "self_attn.q_norm.weight",
            ),
            vec![attention.key_head_dim as u64],
            TensorOwner::Layer(index),
            TensorTransform::Identity,
            qk_required,
        );
        builder.float(
            layer_id(index, LayerTensor::KeyNorm),
            layer_name(
                builder.dialect,
                index,
                "attn_k_norm.weight",
                "self_attn.k_norm.weight",
            ),
            vec![attention.key_head_dim as u64],
            TensorOwner::Layer(index),
            TensorTransform::Identity,
            qk_required,
        );
    }
    if attention.output_gate == AttentionGateKind::SeparateHead {
        builder.weight(
            layer_id(index, LayerTensor::AttentionGate),
            layer_name(
                builder.dialect,
                index,
                "attn_gate.weight",
                "self_attn.gate_proj.weight",
            ),
            matrix_shape(builder.dialect, attention.query_heads, plan.hidden_size),
            TensorOwner::Layer(index),
            TensorTransform::Identity,
        );
    }
}

fn add_mla(
    builder: &mut ContractBuilder,
    plan: &ModelPlan,
    index: u32,
    mla: &MlaAttentionPlan,
) -> Result<(), TensorContractError> {
    if builder.dialect != CheckpointDialect::Gguf {
        return Err(TensorContractError::UnsupportedPlanOperation {
            operation: "HF MLA tensor schema",
        });
    }
    let MlaAttentionPlan::LatentKv {
        query_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_head_dim,
        rope_head_dim,
        value_head_dim,
        rope: _,
        sparse_index,
    } = mla.clone()
    else {
        return Err(TensorContractError::UnsupportedPlanOperation {
            operation: "compressed-KV MLA tensor schema",
        });
    };
    let nope_head_dim = qk_head_dim - rope_head_dim;
    let latent_dim = kv_lora_rank + rope_head_dim;
    let owner = TensorOwner::Layer(index);
    for (tensor, suffix, shape) in [
        (
            LayerTensor::MlaQueryDown,
            "attn_q_a.weight",
            vec![plan.hidden_size as u64, q_lora_rank as u64],
        ),
        (
            LayerTensor::MlaQueryUp,
            "attn_q_b.weight",
            vec![q_lora_rank as u64, (query_heads * qk_head_dim) as u64],
        ),
        (
            LayerTensor::MlaKvDown,
            "attn_kv_a_mqa.weight",
            vec![plan.hidden_size as u64, latent_dim as u64],
        ),
        (
            LayerTensor::MlaKeyUp,
            "attn_k_b.weight",
            vec![
                nope_head_dim as u64,
                kv_lora_rank as u64,
                query_heads as u64,
            ],
        ),
        (
            LayerTensor::MlaValueUp,
            "attn_v_b.weight",
            vec![
                kv_lora_rank as u64,
                value_head_dim as u64,
                query_heads as u64,
            ],
        ),
        (
            LayerTensor::MlaOutput,
            "attn_output.weight",
            vec![
                (query_heads * value_head_dim) as u64,
                plan.hidden_size as u64,
            ],
        ),
    ] {
        builder.weight(
            layer_id(index, tensor),
            format!("blk.{index}.{suffix}"),
            shape,
            owner,
            TensorTransform::Identity,
        );
    }
    for (tensor, suffix, width) in [
        (
            LayerTensor::MlaQueryDownNorm,
            "attn_q_a_norm.weight",
            q_lora_rank,
        ),
        (
            LayerTensor::MlaKvDownNorm,
            "attn_kv_a_norm.weight",
            kv_lora_rank,
        ),
    ] {
        builder.float(
            layer_id(index, tensor),
            format!("blk.{index}.{suffix}"),
            vec![width as u64],
            owner,
            TensorTransform::Identity,
            true,
        );
    }
    builder.weight_aliases(
        layer_id(index, LayerTensor::MlaKvSource),
        vec![format!("blk.{index}.attn_kv_b.weight")],
        vec![
            kv_lora_rank as u64,
            (query_heads * (nope_head_dim + value_head_dim)) as u64,
        ],
        owner,
        TensorTransform::SplitMlaKv,
        false,
    );

    if let SparseIndexPlan::Own {
        heads,
        head_dim,
        top_k: _,
    } = sparse_index
    {
        for (tensor, suffix, shape) in [
            (
                LayerTensor::SparseQuery,
                "indexer.attn_q_b.weight",
                vec![q_lora_rank as u64, (heads * head_dim) as u64],
            ),
            (
                LayerTensor::SparseKey,
                "indexer.attn_k.weight",
                vec![plan.hidden_size as u64, head_dim as u64],
            ),
            (
                LayerTensor::SparseProjection,
                "indexer.proj.weight",
                vec![plan.hidden_size as u64, heads as u64],
            ),
        ] {
            builder.weight(
                layer_id(index, tensor),
                format!("blk.{index}.{suffix}"),
                shape,
                owner,
                TensorTransform::Identity,
            );
        }
        for (tensor, suffix) in [
            (LayerTensor::SparseKeyNorm, "indexer.k_norm.weight"),
            (LayerTensor::SparseKeyNormBias, "indexer.k_norm.bias"),
        ] {
            builder.float(
                layer_id(index, tensor),
                format!("blk.{index}.{suffix}"),
                vec![head_dim as u64],
                owner,
                TensorTransform::Identity,
                true,
            );
        }
    }
    Ok(())
}

fn add_gdn(
    builder: &mut ContractBuilder,
    plan: &ModelPlan,
    index: u32,
    gdn: crate::model_plan::GatedDeltaNetPlan,
) {
    let value_width = gdn.value_heads * gdn.value_head_dim;
    let conv_width = 2 * gdn.key_heads * gdn.key_head_dim + value_width;
    let owner = TensorOwner::Layer(index);
    let matrix = |out, input| matrix_shape(builder.dialect, out, input);
    for (tensor, gguf, hf, shape, transform) in [
        (
            LayerTensor::GdnQkv,
            "attn_qkv.weight",
            "linear_attn.in_proj_qkv.weight",
            matrix(conv_width, plan.hidden_size),
            TensorTransform::QkvVReorderRows,
        ),
        (
            LayerTensor::GdnGate,
            "attn_gate.weight",
            "linear_attn.in_proj_z.weight",
            matrix(value_width, plan.hidden_size),
            TensorTransform::ZReorderRows,
        ),
        (
            LayerTensor::GdnBeta,
            "ssm_beta.weight",
            "linear_attn.in_proj_b.weight",
            matrix(gdn.value_heads, plan.hidden_size),
            TensorTransform::AbReorderRows,
        ),
        (
            LayerTensor::GdnAlpha,
            "ssm_alpha.weight",
            "linear_attn.in_proj_a.weight",
            matrix(gdn.value_heads, plan.hidden_size),
            TensorTransform::AbReorderRows,
        ),
        (
            LayerTensor::GdnOutput,
            "ssm_out.weight",
            "linear_attn.out_proj.weight",
            matrix(plan.hidden_size, value_width),
            TensorTransform::OutReorderColumns,
        ),
    ] {
        builder.weight(
            layer_id(index, tensor),
            layer_name(builder.dialect, index, gguf, hf),
            shape,
            owner,
            transform,
        );
    }
    for (tensor, gguf, hf, shape, transform) in [
        (
            LayerTensor::GdnA,
            "ssm_a",
            "linear_attn.A_log",
            vec![gdn.value_heads as u64],
            TensorTransform::NegExpReorderHeads,
        ),
        (
            LayerTensor::GdnDtBias,
            "ssm_dt.bias",
            "linear_attn.dt_bias",
            vec![gdn.value_heads as u64],
            TensorTransform::ReorderHeads,
        ),
        (
            LayerTensor::GdnNorm,
            "ssm_norm.weight",
            "linear_attn.norm.weight",
            vec![gdn.value_head_dim as u64],
            TensorTransform::Identity,
        ),
    ] {
        builder.float(
            layer_id(index, tensor),
            layer_name(builder.dialect, index, gguf, hf),
            shape,
            owner,
            transform,
            true,
        );
    }
    let conv_shape = match builder.dialect {
        CheckpointDialect::Gguf => vec![gdn.conv_kernel as u64, conv_width as u64],
        CheckpointDialect::HfSafetensors => {
            vec![conv_width as u64, 1, gdn.conv_kernel as u64]
        }
    };
    builder.weight(
        layer_id(index, LayerTensor::GdnConv1d),
        layer_name(
            builder.dialect,
            index,
            "ssm_conv1d.weight",
            "linear_attn.conv1d.weight",
        ),
        conv_shape,
        owner,
        TensorTransform::Conv1dSqueezeReorder,
    );
}

fn add_dense_mlp(builder: &mut ContractBuilder, plan: &ModelPlan, index: u32, intermediate: u32) {
    for (tensor, gguf, hf, out, input) in [
        (
            LayerTensor::MlpGate,
            "ffn_gate.weight",
            "mlp.gate_proj.weight",
            intermediate,
            plan.hidden_size,
        ),
        (
            LayerTensor::MlpUp,
            "ffn_up.weight",
            "mlp.up_proj.weight",
            intermediate,
            plan.hidden_size,
        ),
        (
            LayerTensor::MlpDown,
            "ffn_down.weight",
            "mlp.down_proj.weight",
            plan.hidden_size,
            intermediate,
        ),
    ] {
        builder.weight(
            layer_id(index, tensor),
            layer_name(builder.dialect, index, gguf, hf),
            matrix_shape(builder.dialect, out, input),
            TensorOwner::Layer(index),
            TensorTransform::Identity,
        );
    }
}

fn add_moe_mlp(
    builder: &mut ContractBuilder,
    plan: &ModelPlan,
    index: u32,
    moe: &MoeMlpPlan,
) -> Result<(), TensorContractError> {
    if builder.dialect == CheckpointDialect::HfSafetensors
        && matches!(plan.arch, Arch::Gemma4 | Arch::DeepSeekV4 | Arch::Step35)
    {
        return Err(TensorContractError::UnsupportedPlanOperation {
            operation: "family-specific HF MoE bank",
        });
    }

    let owner = TensorOwner::Layer(index);
    let router_hf = match plan.arch {
        Arch::MinimaxM3 => "block_sparse_moe.gate.weight",
        Arch::Hy3 => "mlp.router.gate.weight",
        _ => "mlp.gate.weight",
    };
    builder.weight(
        layer_id(index, LayerTensor::MoeRouter),
        layer_name(builder.dialect, index, "ffn_gate_inp.weight", router_hf),
        matrix_shape(builder.dialect, moe.expert_count, plan.hidden_size),
        owner,
        TensorTransform::Identity,
    );

    let selection_bias = matches!(
        moe.router,
        RouterPlan::Sigmoid {
            selection_bias: true,
            ..
        } | RouterPlan::SqrtSoftplus {
            selection_bias: true,
            ..
        }
    );
    if selection_bias {
        let names = match builder.dialect {
            CheckpointDialect::Gguf => vec![format!("blk.{index}.exp_probs_b.bias")],
            CheckpointDialect::HfSafetensors if plan.arch == Arch::MinimaxM3 => vec![format!(
                "model.layers.{index}.block_sparse_moe.e_score_correction_bias"
            )],
            CheckpointDialect::HfSafetensors if plan.arch == Arch::Hy3 => vec![
                format!("model.layers.{index}.mlp.expert_bias"),
                format!("model.layers.{index}.mlp.router.expert_bias"),
            ],
            CheckpointDialect::HfSafetensors => {
                vec![format!("model.layers.{index}.mlp.e_score_correction_bias")]
            }
        };
        builder.float_aliases(
            layer_id(index, LayerTensor::MoeRouterBias),
            names,
            vec![moe.expert_count as u64],
            owner,
            TensorTransform::Identity,
            true,
        );
    }

    match builder.dialect {
        CheckpointDialect::Gguf => add_gguf_expert_banks(builder, plan, index, moe, owner),
        CheckpointDialect::HfSafetensors if plan.arch == Arch::Hy3 => {
            add_hy3_expert_banks(builder, plan, index, moe, owner)
        }
        CheckpointDialect::HfSafetensors => add_hf_expert_groups(builder, plan, index, moe, owner),
    }

    if let Some(shared) = moe.shared.as_ref() {
        let (gate, up, down) = match plan.arch {
            Arch::MinimaxM3 => (
                "block_sparse_moe.shared_experts.gate_proj.weight",
                "block_sparse_moe.shared_experts.up_proj.weight",
                "block_sparse_moe.shared_experts.down_proj.weight",
            ),
            Arch::Hy3 => (
                "mlp.shared_mlp.gate_proj.weight",
                "mlp.shared_mlp.up_proj.weight",
                "mlp.shared_mlp.down_proj.weight",
            ),
            _ => (
                "mlp.shared_expert.gate_proj.weight",
                "mlp.shared_expert.up_proj.weight",
                "mlp.shared_expert.down_proj.weight",
            ),
        };
        for (tensor, gguf, hf, out, input) in [
            (
                LayerTensor::SharedMlpGate,
                "ffn_gate_shexp.weight",
                gate,
                shared.intermediate_size,
                plan.hidden_size,
            ),
            (
                LayerTensor::SharedMlpUp,
                "ffn_up_shexp.weight",
                up,
                shared.intermediate_size,
                plan.hidden_size,
            ),
            (
                LayerTensor::SharedMlpDown,
                "ffn_down_shexp.weight",
                down,
                plan.hidden_size,
                shared.intermediate_size,
            ),
        ] {
            builder.weight(
                layer_id(index, tensor),
                layer_name(builder.dialect, index, gguf, hf),
                matrix_shape(builder.dialect, out, input),
                owner,
                TensorTransform::Identity,
            );
        }
        if shared.gated {
            builder.float(
                layer_id(index, LayerTensor::SharedMlpInputGate),
                layer_name(
                    builder.dialect,
                    index,
                    "ffn_gate_inp_shexp.weight",
                    "mlp.shared_expert_gate.weight",
                ),
                vec![plan.hidden_size as u64],
                owner,
                TensorTransform::Identity,
                true,
            );
        }
    }
    Ok(())
}

fn add_gguf_expert_banks(
    builder: &mut ContractBuilder,
    plan: &ModelPlan,
    index: u32,
    moe: &MoeMlpPlan,
    owner: TensorOwner,
) {
    for (tensor, suffix, shape) in [
        (
            LayerTensor::MoeExpertGateBank,
            "ffn_gate_exps.weight",
            vec![
                plan.hidden_size as u64,
                moe.expert_intermediate_size as u64,
                moe.expert_count as u64,
            ],
        ),
        (
            LayerTensor::MoeExpertUpBank,
            "ffn_up_exps.weight",
            vec![
                plan.hidden_size as u64,
                moe.expert_intermediate_size as u64,
                moe.expert_count as u64,
            ],
        ),
        (
            LayerTensor::MoeExpertDownBank,
            "ffn_down_exps.weight",
            vec![
                moe.expert_intermediate_size as u64,
                plan.hidden_size as u64,
                moe.expert_count as u64,
            ],
        ),
    ] {
        builder.weight(
            layer_id(index, tensor),
            format!("blk.{index}.{suffix}"),
            shape,
            owner,
            TensorTransform::Identity,
        );
    }
}

fn add_hy3_expert_banks(
    builder: &mut ContractBuilder,
    plan: &ModelPlan,
    index: u32,
    moe: &MoeMlpPlan,
    owner: TensorOwner,
) {
    for (tensor, suffix, shape) in [
        (
            LayerTensor::MoeExpertGateBank,
            "mlp.switch_mlp.gate_proj.weight",
            vec![
                moe.expert_count as u64,
                moe.expert_intermediate_size as u64,
                plan.hidden_size as u64,
            ],
        ),
        (
            LayerTensor::MoeExpertUpBank,
            "mlp.switch_mlp.up_proj.weight",
            vec![
                moe.expert_count as u64,
                moe.expert_intermediate_size as u64,
                plan.hidden_size as u64,
            ],
        ),
        (
            LayerTensor::MoeExpertDownBank,
            "mlp.switch_mlp.down_proj.weight",
            vec![
                moe.expert_count as u64,
                plan.hidden_size as u64,
                moe.expert_intermediate_size as u64,
            ],
        ),
    ] {
        builder.weight(
            layer_id(index, tensor),
            format!("model.layers.{index}.{suffix}"),
            shape,
            owner,
            TensorTransform::Identity,
        );
    }
}

fn add_hf_expert_groups(
    builder: &mut ContractBuilder,
    plan: &ModelPlan,
    index: u32,
    moe: &MoeMlpPlan,
    owner: TensorOwner,
) {
    for (tensor, projection, out, input) in [
        (
            LayerTensor::MoeExpertGateBank,
            "gate",
            moe.expert_intermediate_size,
            plan.hidden_size,
        ),
        (
            LayerTensor::MoeExpertUpBank,
            "up",
            moe.expert_intermediate_size,
            plan.hidden_size,
        ),
        (
            LayerTensor::MoeExpertDownBank,
            "down",
            plan.hidden_size,
            moe.expert_intermediate_size,
        ),
    ] {
        let names = (0..moe.expert_count)
            .map(|expert| crate::hf_mapping::hf_expert_name(index, expert, projection, &plan.arch))
            .collect();
        builder.weight_group(
            layer_id(index, tensor),
            names,
            vec![out as u64, input as u64],
            owner,
            TensorTransform::StackExperts,
        );
    }
}

fn layer_id(index: u32, tensor: LayerTensor) -> TensorId {
    TensorId::Layer { index, tensor }
}

fn top_name(dialect: CheckpointDialect, gguf: &str, hf: &str) -> String {
    match dialect {
        CheckpointDialect::Gguf => gguf.to_string(),
        CheckpointDialect::HfSafetensors => hf.to_string(),
    }
}

fn layer_name(dialect: CheckpointDialect, index: u32, gguf: &str, hf: &str) -> String {
    match dialect {
        CheckpointDialect::Gguf => format!("blk.{index}.{gguf}"),
        CheckpointDialect::HfSafetensors => format!("model.layers.{index}.{hf}"),
    }
}

fn matrix_shape(dialect: CheckpointDialect, out: u32, input: u32) -> Vec<u64> {
    match dialect {
        CheckpointDialect::Gguf => vec![input as u64, out as u64],
        CheckpointDialect::HfSafetensors => vec![out as u64, input as u64],
    }
}

fn norm_transform(transform: WeightTransform) -> TensorTransform {
    match transform {
        WeightTransform::Identity => TensorTransform::Identity,
        WeightTransform::AddOne => TensorTransform::NormAddOne,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HfConfig, ModelConfig};

    fn qwen3_plan() -> ModelPlan {
        let cfg = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":64,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
            "intermediate_size":128,"vocab_size":16,"max_position_embeddings":128}"#,
        ));
        ModelPlan::compile(&cfg).unwrap()
    }

    fn qwen35_plan() -> ModelPlan {
        let cfg = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3_5","num_hidden_layers":4,"hidden_size":256,
            "num_attention_heads":8,"num_key_value_heads":2,"head_dim":32,
            "intermediate_size":512,"vocab_size":1024,"max_position_embeddings":2048,
            "full_attention_interval":4,"linear_conv_kernel_dim":4,
            "linear_key_head_dim":32,"linear_value_head_dim":32,
            "linear_num_key_heads":2,"linear_num_value_heads":4}"#,
        ));
        ModelPlan::compile(&cfg).unwrap()
    }

    fn qwen35_mtp_plan() -> ModelPlan {
        let cfg = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3_5","num_hidden_layers":4,
            "num_nextn_predict_layers":1,"hidden_size":256,
            "num_attention_heads":8,"num_key_value_heads":2,"head_dim":32,
            "intermediate_size":512,"vocab_size":1024,"max_position_embeddings":2048,
            "full_attention_interval":4,"linear_conv_kernel_dim":4,
            "linear_key_head_dim":32,"linear_value_head_dim":32,
            "linear_num_key_heads":2,"linear_num_value_heads":4}"#,
        ));
        ModelPlan::compile(&cfg).unwrap()
    }

    fn qwen3_moe_plan() -> ModelPlan {
        let cfg = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3_moe","num_hidden_layers":1,"hidden_size":64,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
            "intermediate_size":128,"vocab_size":16,"max_position_embeddings":128,
            "num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":32,
            "shared_expert_intermediate_size":48}"#,
        ));
        ModelPlan::compile(&cfg).unwrap()
    }

    fn gemma4_dense_plan() -> ModelPlan {
        let cfg = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"gemma4","num_hidden_layers":2,"hidden_size":8,
            "num_attention_heads":2,"num_key_value_heads":1,
            "num_global_key_value_heads":1,"head_dim":4,"global_head_dim":4,
            "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32,
            "rms_norm_eps":0.000001,"sliding_window":2,
            "layer_types":["sliding_attention","full_attention"],
            "rope_parameters":{"full_attention":{"rope_theta":1000000,
            "partial_rotary_factor":0.5},"sliding_attention":{"rope_theta":10000}}}"#,
        ));
        ModelPlan::compile(&cfg).unwrap()
    }

    fn census_for(contract: &TensorContract) -> Vec<TensorCensusEntry> {
        contract
            .requirements
            .iter()
            .filter(|requirement| requirement.required)
            .flat_map(|requirement| {
                let names = match requirement.match_mode {
                    TensorMatch::OneOf => &requirement.names[..1],
                    TensorMatch::All => requirement.names.as_slice(),
                };
                names.iter().map(|name| TensorCensusEntry {
                    name: name.clone(),
                    shape: requirement.shape.clone(),
                    storage: if requirement.quant == QuantConstraint::FloatOnly {
                        StorageLayout::Float(FloatType::Bf16)
                    } else {
                        StorageLayout::Quantized(QuantLayout {
                            format: "Q4_K".to_string(),
                            block_shape: vec![256],
                            auxiliaries: Vec::new(),
                        })
                    },
                })
            })
            .collect()
    }

    #[test]
    fn dense_hf_contract_binds_every_tensor_to_stable_ids() {
        let contract = TensorContract::for_plan(
            &qwen3_plan(),
            CheckpointDialect::HfSafetensors,
            ContractOptions::default(),
        )
        .unwrap();
        let bound = contract.bind(&census_for(&contract)).unwrap();

        assert_eq!(
            bound.tensors.len(),
            contract
                .requirements
                .iter()
                .filter(|requirement| requirement.required)
                .count()
        );
        assert_eq!(
            bound.tensors[&TensorId::Layer {
                index: 0,
                tensor: LayerTensor::Query,
            }]
                .shapes[0],
            vec![64, 64]
        );
        assert_eq!(
            bound.tensors[&TensorId::OutputNorm].owner,
            TensorOwner::Global
        );
    }

    #[test]
    fn missing_extra_ambiguous_shape_and_layout_all_fail_closed() {
        let contract = TensorContract::for_plan(
            &qwen3_plan(),
            CheckpointDialect::HfSafetensors,
            ContractOptions::default(),
        )
        .unwrap();
        let mut census = census_for(&contract);
        let missing = census.remove(0);
        assert!(matches!(
            contract.bind(&census),
            Err(TensorContractError::Missing { .. })
        ));

        census.push(missing);
        census.push(TensorCensusEntry {
            name: "unexpected.weight".to_string(),
            shape: vec![1],
            storage: StorageLayout::Float(FloatType::F32),
        });
        assert_eq!(
            contract.bind(&census),
            Err(TensorContractError::Extra {
                names: vec!["unexpected.weight".to_string()]
            })
        );

        let mut shape_bad = census_for(&contract);
        shape_bad[0].shape = vec![999];
        assert!(matches!(
            contract.bind(&shape_bad),
            Err(TensorContractError::ShapeMismatch { .. })
        ));

        let norm_name = contract
            .requirements
            .iter()
            .find(|requirement| requirement.id == TensorId::OutputNorm)
            .unwrap()
            .names[0]
            .clone();
        let mut layout_bad = census_for(&contract);
        layout_bad
            .iter_mut()
            .find(|entry| entry.name == norm_name)
            .unwrap()
            .storage = StorageLayout::Quantized(QuantLayout {
            format: "Q4_K".to_string(),
            block_shape: vec![256],
            auxiliaries: Vec::new(),
        });
        assert!(matches!(
            contract.bind(&layout_bad),
            Err(TensorContractError::QuantLayoutMismatch { .. })
        ));

        let ambiguous = TensorContract {
            dialect: CheckpointDialect::HfSafetensors,
            requirements: vec![TensorRequirement {
                id: TensorId::OutputNorm,
                names: vec!["model.norm.weight".into(), "transformer.norm.weight".into()],
                match_mode: TensorMatch::OneOf,
                shape: vec![64],
                owner: TensorOwner::Global,
                transform: TensorTransform::Identity,
                quant: QuantConstraint::FloatOnly,
                auxiliaries: None,
                required: true,
            }],
        };
        let entries = vec![
            TensorCensusEntry {
                name: ambiguous.requirements[0].names[0].clone(),
                shape: vec![64],
                storage: StorageLayout::Float(FloatType::Bf16),
            },
            TensorCensusEntry {
                name: ambiguous.requirements[0].names[1].clone(),
                shape: vec![64],
                storage: StorageLayout::Float(FloatType::Bf16),
            },
        ];
        assert!(matches!(
            ambiguous.bind(&entries),
            Err(TensorContractError::Ambiguous { .. })
        ));
    }

    #[test]
    fn gguf_and_hf_contracts_have_same_semantic_ids_and_transposed_matrix_shapes() {
        let plan = qwen3_plan();
        let gguf =
            TensorContract::for_plan(&plan, CheckpointDialect::Gguf, ContractOptions::default())
                .unwrap();
        let hf = TensorContract::for_plan(
            &plan,
            CheckpointDialect::HfSafetensors,
            ContractOptions::default(),
        )
        .unwrap();
        let gguf_by_id: BTreeMap<_, _> = gguf
            .requirements
            .iter()
            .map(|requirement| (&requirement.id, requirement))
            .collect();
        let hf_by_id: BTreeMap<_, _> = hf
            .requirements
            .iter()
            .map(|requirement| (&requirement.id, requirement))
            .collect();
        assert_eq!(
            gguf_by_id.keys().collect::<Vec<_>>(),
            hf_by_id.keys().collect::<Vec<_>>()
        );

        let id = TensorId::Layer {
            index: 0,
            tensor: LayerTensor::MlpGate,
        };
        assert_eq!(gguf_by_id[&id].shape, vec![64, 128]);
        assert_eq!(hf_by_id[&id].shape, vec![128, 64]);
    }

    #[test]
    fn gdn_contract_pins_hf_shapes_and_value_reorder_transforms() {
        let contract = TensorContract::for_plan(
            &qwen35_plan(),
            CheckpointDialect::HfSafetensors,
            ContractOptions::default(),
        )
        .unwrap();
        let by_id: BTreeMap<_, _> = contract
            .requirements
            .iter()
            .map(|requirement| (&requirement.id, requirement))
            .collect();

        let qkv = &by_id[&TensorId::Layer {
            index: 0,
            tensor: LayerTensor::GdnQkv,
        }];
        assert_eq!(
            qkv.names,
            vec!["model.layers.0.linear_attn.in_proj_qkv.weight"]
        );
        assert_eq!(qkv.shape, vec![256, 256]);
        assert_eq!(qkv.transform, TensorTransform::QkvVReorderRows);

        let conv = &by_id[&TensorId::Layer {
            index: 0,
            tensor: LayerTensor::GdnConv1d,
        }];
        assert_eq!(conv.shape, vec![256, 1, 4]);
        assert_eq!(conv.transform, TensorTransform::Conv1dSqueezeReorder);
        contract.bind(&census_for(&contract)).unwrap();
    }

    #[test]
    fn moe_bank_id_binds_hf_expert_group_and_gguf_stack() {
        let plan = qwen3_moe_plan();
        let hf = TensorContract::for_plan(
            &plan,
            CheckpointDialect::HfSafetensors,
            ContractOptions::default(),
        )
        .unwrap();
        let gguf =
            TensorContract::for_plan(&plan, CheckpointDialect::Gguf, ContractOptions::default())
                .unwrap();
        let gate_id = TensorId::Layer {
            index: 0,
            tensor: LayerTensor::MoeExpertGateBank,
        };
        let hf_gate = hf
            .requirements
            .iter()
            .find(|requirement| requirement.id == gate_id)
            .unwrap();
        let gguf_gate = gguf
            .requirements
            .iter()
            .find(|requirement| requirement.id == gate_id)
            .unwrap();
        assert_eq!(hf_gate.match_mode, TensorMatch::All);
        assert_eq!(hf_gate.names.len(), 4);
        assert_eq!(hf_gate.shape, vec![32, 64]);
        assert_eq!(hf_gate.transform, TensorTransform::StackExperts);
        assert_eq!(gguf_gate.match_mode, TensorMatch::OneOf);
        assert_eq!(gguf_gate.names, vec!["blk.0.ffn_gate_exps.weight"]);
        assert_eq!(gguf_gate.shape, vec![64, 32, 4]);

        let census = census_for(&hf);
        let bound = hf.bind(&census).unwrap();
        assert_eq!(bound.tensors[&gate_id].checkpoint_names.len(), 4);

        let missing_name = hf_gate.names[2].clone();
        let incomplete: Vec<_> = census
            .into_iter()
            .filter(|entry| entry.name != missing_name)
            .collect();
        assert_eq!(
            hf.bind(&incomplete),
            Err(TensorContractError::Missing {
                id: gate_id,
                accepted_names: vec![missing_name],
            })
        );
    }

    #[test]
    fn mtp_contract_reuses_block_schema_and_rewrites_hf_namespace() {
        let plan = qwen35_mtp_plan();
        let contract = TensorContract::for_plan(
            &plan,
            CheckpointDialect::HfSafetensors,
            ContractOptions::default(),
        )
        .unwrap();
        let query = contract
            .requirements
            .iter()
            .find(|requirement| {
                requirement.id
                    == TensorId::Layer {
                        index: 4,
                        tensor: LayerTensor::Query,
                    }
            })
            .unwrap();
        assert_eq!(query.names, vec!["mtp.layers.0.self_attn.q_proj.weight"]);
        let fusion = contract
            .requirements
            .iter()
            .find(|requirement| {
                requirement.id
                    == TensorId::Mtp {
                        depth: 0,
                        tensor: MtpTensor::FusionProjection,
                    }
            })
            .unwrap();
        assert_eq!(fusion.names, vec!["mtp.fc.weight"]);
        assert_eq!(fusion.shape, vec![256, 512]);
        assert_eq!(fusion.owner, TensorOwner::Mtp(0));
        assert_eq!(fusion.transform, TensorTransform::Identity);

        let enorm = contract
            .requirements
            .iter()
            .find(|requirement| {
                requirement.id
                    == TensorId::Mtp {
                        depth: 0,
                        tensor: MtpTensor::EmbeddingNorm,
                    }
            })
            .unwrap();
        assert_eq!(enorm.names, vec!["mtp.pre_fc_norm_embedding.weight"]);
        assert_eq!(enorm.transform, TensorTransform::NormAddOne);
        contract.bind(&census_for(&contract)).unwrap();
    }

    #[test]
    fn glm_mla_moe_mtp_contract_binds_the_existing_micro_fixture() {
        use crate::{GgmlType, GgufFile};

        let path = std::env::temp_dir().join(format!(
            "memra-model-plan-mla-{}-{}.gguf",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        crate::micro_gguf::write_glm_dsa_micro(&path, 0x4d4c_4150).unwrap();
        let gguf = GgufFile::open(&path).unwrap();
        let plan = ModelPlan::compile(&ModelConfig::from_gguf(&gguf)).unwrap();
        let contract =
            TensorContract::for_plan(&plan, CheckpointDialect::Gguf, ContractOptions::default())
                .unwrap();
        let census: Vec<_> = gguf
            .tensors
            .iter()
            .map(|tensor| TensorCensusEntry {
                name: tensor.name.clone(),
                shape: tensor.ne.clone(),
                storage: match tensor.ggml_type {
                    GgmlType::F32 => StorageLayout::Float(FloatType::F32),
                    GgmlType::F16 => StorageLayout::Float(FloatType::F16),
                    GgmlType::BF16 => StorageLayout::Float(FloatType::Bf16),
                    other => StorageLayout::Quantized(QuantLayout {
                        format: format!("{other:?}"),
                        block_shape: Vec::new(),
                        auxiliaries: Vec::new(),
                    }),
                },
            })
            .collect();
        let bound = contract.bind(&census).unwrap();
        let expected_bound = contract
            .requirements
            .iter()
            .filter(|requirement| {
                requirement.required
                    || requirement
                        .names
                        .iter()
                        .any(|name| census.iter().any(|entry| &entry.name == name))
            })
            .count();
        assert_eq!(bound.tensors.len(), expected_bound);
        assert!(bound.tensors.contains_key(&TensorId::Layer {
            index: 0,
            tensor: LayerTensor::SparseQuery,
        }));
        assert!(!bound.tensors.contains_key(&TensorId::Layer {
            index: 1,
            tensor: LayerTensor::SparseQuery,
        }));
        assert!(bound.tensors.contains_key(&TensorId::Mtp {
            depth: 0,
            tensor: MtpTensor::FusionProjection,
        }));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn dense_gemma_contract_pins_norms_scale_factors_and_k_as_v() {
        let plan = gemma4_dense_plan();
        let hf = TensorContract::for_plan(
            &plan,
            CheckpointDialect::HfSafetensors,
            ContractOptions {
                output_head: OutputHead::TiedToEmbedding,
            },
        )
        .unwrap();
        let has = |id: TensorId| {
            hf.requirements
                .iter()
                .any(|requirement| requirement.id == id)
        };
        assert!(has(layer_id(0, LayerTensor::Value)));
        assert!(!has(layer_id(1, LayerTensor::Value)));
        for (tensor, expected) in [
            (
                LayerTensor::PostAttentionNorm,
                "model.layers.0.post_attention_layernorm.weight",
            ),
            (
                LayerTensor::PreMlpNorm,
                "model.layers.0.pre_feedforward_layernorm.weight",
            ),
            (
                LayerTensor::PostMlpNorm,
                "model.layers.0.post_feedforward_layernorm.weight",
            ),
            (LayerTensor::LayerScale, "model.layers.0.layer_scalar"),
        ] {
            let requirement = hf
                .requirements
                .iter()
                .find(|requirement| requirement.id == layer_id(0, tensor))
                .unwrap();
            assert_eq!(requirement.names, vec![expected]);
        }
        assert!(!has(TensorId::RopeFactors));
        hf.bind(&census_for(&hf)).unwrap();

        let gguf = TensorContract::for_plan(
            &plan,
            CheckpointDialect::Gguf,
            ContractOptions {
                output_head: OutputHead::TiedToEmbedding,
            },
        )
        .unwrap();
        let factors = gguf
            .requirements
            .iter()
            .find(|requirement| requirement.id == TensorId::RopeFactors)
            .unwrap();
        assert_eq!(factors.names, vec!["rope_freqs.weight"]);
        assert_eq!(factors.shape, vec![2]);
    }
}
