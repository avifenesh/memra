//! Inkling 2 (`inkling2`) pack: config normalization, the typed 48-slot plan and the
//! config-derived tensor schema for `thinkingmachines/Inkling-2-Mini`. The reference executor
//! (memra-reference `inkling2`) runs the plan; every tuned execution surface refuses it.
//!
//! The checkpoint defines the program. Its `config.json` gives the topology and
//! `hf_quant_config.json` gives the numerics: the routed experts are NVFP4 (layers 0-19) or
//! MXFP8 (layers 20-23) and each group REQUIRES its input quantized the same way. That declared
//! program is the family's default numeric class. Running the experts on float inputs is a
//! second, named bring-up class and is never presented as the first.
//!
//! Checkpoint namespaces (1,298 header tensors):
//!   model.llm.*      trunk: 24 physical layers, loop controllers and queries, embed, unembed
//!   model.visual.*   ViT tower and projector; contract-owned, not executed (text only)
//!   mtp.*            the DSpark drafter of `mtp.safetensors` (a file outside the shard index;
//!                    the prefix is the census's); contract-owned, not executed
//!
//! Layers 6..18 run three passes over shared weights, so the plan has 48 layer entries (cache
//! slots) over 24 weight sets; `ModelPlan::inkling2.loop_plan` maps each slot to its physical
//! layer. Tensor ids are keyed on the physical layer.

use std::collections::BTreeMap;

use super::*;
use crate::config::Inkling2Config;
use crate::dsv4::TensorSpec;
use crate::model_plan::{
    ActivationPlan, AttentionInputs, AttentionPlan, BeyondExtent, BlockAttentionPlan, ClipScale,
    DenseNumericsPlan, DenseTensorClass, DraftAttentionPlan, DraftBlockPlan, DraftFusePlan,
    DraftRnnHeadPlan, DraftRopePlan, DraftRowInput, DraftSourcePlan, DrafterPlan, DrafterTap,
    DsparkContextLayers, DsparkFusion, DsparkLayers, DsparkMarkov, DsparkPlan, DsparkRows,
    ExpertActivationMode, ExpertNumericsPlan, ExpertWeightFormat, Inkling2NumericClass,
    Inkling2Numerics, Inkling2Plan, Inkling2WeightClass, KvCacheFormat, L2ClipPlan, LayerPlan,
    LogPositionScalePlan, LogitsTransform, LoopMessageTarget, LoopPlan, LoopSlotPlan, MlpPlan,
    MoeMlpPlan, NormKind, NormPlan, OutputGatePlan, QuerySinkPlan, RelativeBiasAttentionPlan,
    RelativeBiasPlan, RelativeBiasStatePlan, ResidualTopology, RnnChunk, RnnInput, RouterPlan,
    ShortConvPlan, ShortConvTaps, StatePlan, ValueNorm, WeightTransform,
};
use crate::quant_groups::ComputeDtype;
use crate::tensor_contract::{
    FloatType, LayerTensor, QuantConstraint, TensorId, TensorMatch, TensorOwner, TensorRequirement,
    TensorTransform,
};

const FAMILY: &str = "inkling2";

/// Census prefix of the drafter file's tensors.
pub const DRAFTER_PREFIX: &str = "mtp.";

/// The oracle numeric class of the checkpoint's declared program (NVFP4 W4A4 on layers 0..19,
/// MXFP8 W8A8 on 20..23, everything outside the experts in f32): the class name a
/// `memra-checkpoint-oracle-v1` tape of that program carries, and the key of its parity band.
pub const DECLARED_PARITY_CLASS: &str = "inkling2-declared-activation-quant";

/// The oracle numeric class of the nvfp4-rig program (the declared experts, the minted NVFP4
/// overlay, e4m3 K/V): the class name its tapes carry and the key of its parity band.
pub const NVFP4_RIG_PARITY_CLASS: &str = "inkling2-nvfp4-rig";

/// Epsilon of the L2 clips, the residual mixes and the controller norms. It is a constant of the
/// program, not a config key.
const PROGRAM_EPSILON: f32 = 1e-6;

/// Planes of a stacked expert bank, in the order the safetensors census lists them.
/// Quantization planes, as suffixes of the tensor they belong to: a stacked bank's three, and a
/// minted dense tensor's modelopt `_scale` and `_scale_2` (`nvfp4-rig`).
const BANK_PLANES: [&str; 4] = ["_scale", "_amax", "_outer_scale_denominator", "_scale_2"];

pub static PACK: ModelPack = ModelPack {
    family: "inkling2",
    output_head: OutputHeadContract::SeparateHead,
    tensor_consumption: TensorConsumption::Report,
    auxiliary_files: &[AuxiliaryTensorFile {
        file: "mtp.safetensors",
        prefix: DRAFTER_PREFIX,
    }],
    aliases: &[
        "inkling2",
        "inkling2_loop",
        "inkling2_loop_text",
        "inkling-2-mini",
    ],
    config_layout: ConfigLayout::FlatOrTextConfig,
    tokenizer_sources: &[TokenizerSource::TokenizerJson],
    // The artifact ships `chat_template.jinja`.
    template: TemplateContract::ArtifactRequired,
    // The shipped class is qualified on its GPU route (record inkling2-shipped-gpu-1x-5090:
    // CheckpointParity on the declared class's band and Serve, on the pinned artifact); no tuned
    // rewrite is qualified.
    support: Some(NativeSupport::NativeQualified),
    gates: &[
        Gate::Config,
        Gate::TokenizerTemplate,
        Gate::TensorCensus,
        Gate::TinyParity,
        Gate::CheckpointParity,
        Gate::RewriteParity,
        Gate::Serve,
    ],
    // No default band: an oracle of another class (a float-activation one, say) cannot write a
    // CheckpointParity pass for a pack whose default class is the declared one. The tiny gate does
    // not read this field: its fixtures carry their own comparison (memra-reference `TinyGate`).
    checkpoint_parity: None,
    // The declared class: its band is twice the largest spread between independent backends of
    // the source program on the probe below. Lane E's CPU oracle and lane F's harness (the
    // transformers model with the act-quant experts) on CPU and on the GPU differ pairwise by
    // 1.76 to 1.84 logits at most (far-tail logits; KL under 1e-9, the same argmax), so a native
    // run may sit 3.7 from the oracle and must give its argmax. The route's per-MoE bits, its
    // float trunk and its end-to-end KL on longer text are held separately
    // (docs/decisions/INKLING2-DECLARED-CHECKPOINT-PARITY.md,
    // research/modelplan-onboarding-inkling2mini-20261121/checkpoint-parity/).
    checkpoint_parity_classes: &[ClassParityGate {
        numeric_class: DECLARED_PARITY_CLASS,
        gate: CheckpointParityGate {
            max_abs: 3.7,
            max_rel: 0.0,
            require_argmax: true,
        },
        tape: None,
    }],
    // The probe is a real chat turn: the system turn "Thinking effort level: 0.9" and the user turn
    // "Explain in one sentence why the sky is usually blue." rendered by the checkpoint's template
    // (lane E's p1_chat prompt, 26 tokens), whose next token every backend picks by a margin of
    // 20.3 logits or more. The default 1,2,3,4 is not a sequence the model ever sees.
    checkpoint_probe: Some(&[
        200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13, 24, 200010, 200000, 200004,
        100887, 149881, 150011, 156021, 151661, 149876, 158808, 149884, 152978, 152778, 13, 200010,
    ]),
    checkpoint_requests: &[],
    matches_config: |config| matches!(config.arch, Arch::Inkling2Loop) && config.inkling2.is_some(),
    plan_builder,
    tensor_schema,
    tiny_plan: Some(tiny_plan),
};

/// The tiny-gate config: the family's program at hidden 64 in the checkpoint's own key layout.
/// Eight physical layers in four attention-residual blocks of two (a prelude block, two looped
/// core blocks, a tail block), a full-attention layer closing each block, three passes over
/// layers 2..6 (16 cache slots), a sliding window of 8, a relative extent of 12 and a log-scale
/// floor of 24, so a 40-token request crosses all three. Two core blocks make the loop's message
/// bank hold two states, so the message mix is a real softmax. Top-2 of 8 experts.
pub const TINY_CONFIG: &str = include_str!("fixtures/tiny-config.json");

/// The tiny checkpoint's `hf_quant_config.json`: the real groups over the tiny tensor names,
/// NVFP4 experts on layers 0-5 and MXFP8 on layers 6-7.
pub const TINY_QUANT_CONFIG: &str = include_str!("fixtures/tiny-hf_quant_config.json");

fn tiny_config() -> Result<ModelConfig, PlanCompileError> {
    let mut config = ModelConfig::from_hf(&crate::config::HfConfig::parse(TINY_CONFIG));
    config
        .attach_quant_groups(TINY_QUANT_CONFIG)
        .map_err(|value| PlanCompileError::UnsupportedSemantics {
            field: "hf_quant_config.json",
            value,
        })?;
    Ok(config)
}

/// The tiny gate's plan runs every layer's experts on float inputs. That is the class the gate
/// has an independent oracle for (the float-activation implementation in f64): it pins the
/// trunk, the loop, the exact bank dequantization and the float expert program. The declared
/// quantizer is pinned by its own golden vectors, and [`tiny_declared_plan`] runs as a variant
/// of the gate against a simulation of the activation-quant program.
fn tiny_plan() -> Result<ModelPlan, PlanCompileError> {
    let config = tiny_config()?;
    let layers = inkling(&config)?.num_hidden_layers as usize;
    plan_with_expert_activation(&config, &vec![ExpertActivationMode::Float; layers])
}

/// The tiny config under the family's default class: every layer on its declared quantizer.
pub fn tiny_declared_plan() -> Result<ModelPlan, PlanCompileError> {
    plan_builder(&tiny_config()?)
}

/// The tiny config with the expert input mode chosen per physical layer.
pub fn tiny_plan_with_expert_activation(
    modes: &[ExpertActivationMode],
) -> Result<ModelPlan, PlanCompileError> {
    plan_with_expert_activation(&tiny_config()?, modes)
}

fn inkling(config: &ModelConfig) -> Result<&Inkling2Config, PlanCompileError> {
    config
        .inkling2
        .as_ref()
        .ok_or_else(|| PlanCompileError::UnsupportedSemantics {
            field: "inkling2",
            value: "config carries no inkling2 block".into(),
        })
}

/// The family's default plan: every layer runs its declared activation quantizer.
fn plan_builder(config: &ModelConfig) -> Result<ModelPlan, PlanCompileError> {
    let layers = inkling(config)?.num_hidden_layers as usize;
    plan_with_expert_activation(config, &vec![ExpertActivationMode::Declared; layers])
}

/// The plan with the expert input mode chosen per physical layer. The default plan is all
/// `Declared`; anything else is a measurement plan with its own numeric class and plan hash,
/// which is how a per-layer study of the quantizer's effect names what it ran.
pub fn plan_with_expert_activation(
    config: &ModelConfig,
    modes: &[ExpertActivationMode],
) -> Result<ModelPlan, PlanCompileError> {
    plan_with_classes(config, Inkling2WeightClass::Shipped, modes)
}

/// The `nvfp4-rig` plan: the family's default weight and cache class, every layer on its
/// declared activation quantizer. It binds only a checkpoint merged with its minted overlay.
pub fn nvfp4_rig_plan(config: &ModelConfig) -> Result<ModelPlan, PlanCompileError> {
    let layers = inkling(config)?.num_hidden_layers as usize;
    plan_with_classes(
        config,
        Inkling2WeightClass::Nvfp4Rig,
        &vec![ExpertActivationMode::Declared; layers],
    )
}

/// The measurement class `nvfp4-rig-outside-loop`: `nvfp4-rig` with the loop core's attention
/// read as stored (BF16), every layer on its declared quantizer.
pub fn nvfp4_rig_outside_loop_plan(config: &ModelConfig) -> Result<ModelPlan, PlanCompileError> {
    let layers = inkling(config)?.num_hidden_layers as usize;
    plan_with_classes(
        config,
        Inkling2WeightClass::Nvfp4RigOutsideLoop,
        &vec![ExpertActivationMode::Declared; layers],
    )
}

/// The tiny config under `nvfp4-rig` with the expert input mode chosen per physical layer.
pub fn tiny_nvfp4_rig_plan(modes: &[ExpertActivationMode]) -> Result<ModelPlan, PlanCompileError> {
    plan_with_classes(&tiny_config()?, Inkling2WeightClass::Nvfp4Rig, modes)
}

/// The tiny config (for the mint and the tiny gates of other crates).
pub fn tiny_model_config() -> Result<ModelConfig, PlanCompileError> {
    tiny_config()
}

/// The plan of a weight class with the expert input mode chosen per physical layer.
pub fn plan_with_classes(
    config: &ModelConfig,
    weight_class: Inkling2WeightClass,
    modes: &[ExpertActivationMode],
) -> Result<ModelPlan, PlanCompileError> {
    config.validate_plan_semantics()?;
    let ink = inkling(config)?;
    if modes.len() != ink.num_hidden_layers as usize {
        return Err(PlanCompileError::UnsupportedSemantics {
            field: "expert activation modes",
            value: format!(
                "{} modes for {} physical layers",
                modes.len(),
                ink.num_hidden_layers
            ),
        });
    }
    if ink.num_hidden_layers == 0 {
        return Err(PlanCompileError::EmptyModel);
    }
    let numerics_error = |value: String| PlanCompileError::UnsupportedSemantics {
        field: "hf_quant_config.json",
        value,
    };
    let quant = ink.quant.as_ref().ok_or_else(|| {
        numerics_error(
            "not attached; the expert formats and the required activation quantizers are declared there"
                .into(),
        )
    })?;
    let norm = NormPlan {
        kind: NormKind::Rms,
        epsilon: ink.rms_norm_eps,
        weight_transform: WeightTransform::Identity,
    };
    let rig = weight_class != Inkling2WeightClass::Shipped;
    let loop_attention = weight_class == Inkling2WeightClass::Nvfp4Rig;
    // The NVFP4 program a minted bank runs: the declared quantizer of the shipped NVFP4 layers.
    let nvfp4_declared = (0..ink.num_hidden_layers)
        .filter_map(|physical| ink.expert_quant(physical).ok())
        .find(|q| q.compute == ComputeDtype::Nvfp4)
        .map(|q| q.activation);
    let experts = (0..ink.num_hidden_layers)
        .map(|physical| {
            let mut declared = ink.expert_quant(physical).map_err(&numerics_error)?;
            let minted = rig
                && nvfp4_rig::converts_experts(ink, physical).map_err(&numerics_error)?;
            if minted {
                declared.compute = ComputeDtype::Nvfp4;
                declared.activation = nvfp4_declared.ok_or_else(|| {
                    numerics_error(
                        "nvfp4-rig converts MXFP8 banks to the NVFP4 program, and no layer declares one"
                            .into(),
                    )
                })?;
            }
            Ok(ExpertNumericsPlan {
                minted,
                physical_layer: physical,
                weight: match declared.compute {
                    ComputeDtype::Nvfp4 => ExpertWeightFormat::Nvfp4Amax,
                    ComputeDtype::Mxfp8 => ExpertWeightFormat::Mxfp8Row32,
                    other => {
                        return Err(numerics_error(format!(
                            "layer {physical} experts compute in {other:?}; only nvfp4 and mxfp8 banks are typed"
                        )));
                    }
                },
                declared_activation: declared.activation,
                activation: modes[physical as usize],
            })
        })
        .collect::<Result<Vec<_>, PlanCompileError>>()?;
    let dense = dense_numerics(quant).map_err(&numerics_error)?;
    declaration_covers_census(config, quant).map_err(&numerics_error)?;

    let slots = ink.slots();
    let layers = (0..slots).map(|slot| layer_plan(ink, slot, norm)).collect();
    let loop_slots = (0..slots)
        .map(|slot| {
            let pass = ink.slot_pass(slot);
            LoopSlotPlan {
                slot,
                physical_layer: ink.physical_layer(slot),
                pass,
                message: pass.is_some_and(|pass| pass > 0),
            }
        })
        .collect();
    let execution_order = ink.execution_order();
    let drafter = &ink.drafter;
    let taps = drafter
        .target_layer_ids
        .iter()
        .map(|&visit| match execution_order.get(visit as usize) {
            Some(&slot) => DrafterTap::SlotInput { slot },
            None => DrafterTap::FinalNorm,
        })
        .collect();
    Ok(ModelPlan {
        speech: None,
        arch: config.arch.clone(),
        prefill_activation: None,
        hidden_size: ink.hidden_size,
        // Rows of the embedding and the output projection; `Truncate` drops the padding.
        vocab_size: ink.vocab_size,
        context_length: ink.model_max_length,
        embedding_scale: 1.0,
        // The ViT tower and projector are contract-owned and not part of the text plan.
        vision: None,
        mimo_audio_patch: None,
        multimodal: None,
        per_layer_input: None,
        layers,
        output_norm: norm,
        exit_mixer: None,
        logits: vec![
            LogitsTransform::Softcap(ink.final_logit_softcapping),
            LogitsTransform::Truncate(ink.unpadded_vocab_size),
        ],
        mtp_blocks: Vec::new(),
        // The shipped drafter (`mtp.safetensors`, in the checkpoint) is a DSpark program with
        // its own variants of the rows, fusion, layers and Markov head, and no confidence head.
        drafter: Some(DrafterPlan::Dspark(DsparkPlan {
            block_size: drafter.block_size,
            noise_token_id: 0,
            target_layer_ids: drafter.target_layer_ids.clone(),
            markov_rank: drafter.markov_rank,
            blocks: Vec::new(),
            rows: DsparkRows::Block(DraftBlockPlan {
                rows: drafter.block_size,
                anchor: DraftRowInput::TrunkEmbeddingNormed,
                mask: DraftRowInput::LearnedMask,
                proposals: drafter.block_size - 1,
            }),
            fusion: DsparkFusion::FuseWeights {
                fuse: DraftFusePlan {
                    rows: drafter.num_hidden_layers + 1,
                    taps: drafter.target_layer_ids.len() as u32,
                },
                taps,
            },
            layers: DsparkLayers::Context(Box::new(DsparkContextLayers {
                layers: drafter.num_hidden_layers,
                query_heads: drafter.num_attention_heads,
                kv_heads: drafter.num_key_value_heads,
                head_dim: drafter.head_dim,
                intermediate_size: drafter.intermediate_size,
                hidden_size: drafter.hidden_size,
                epsilon: drafter.rms_norm_eps,
                attention: DraftAttentionPlan {
                    scale: (drafter.head_dim as f32).powf(-0.5),
                    qk_norm: true,
                    rope: DraftRopePlan {
                        theta: drafter.rope_theta,
                    },
                    context_key_norm: true,
                    bidirectional_block: true,
                    local_layers: drafter.local_layer_ids.clone(),
                    local_context: drafter.sliding_window_size,
                },
                mlp_swiglu: true,
            })),
            markov: DsparkMarkov::RnnHidden(DraftRnnHeadPlan {
                rank: drafter.markov_rank,
                inputs: [
                    RnnInput::State,
                    RnnInput::PreviousToken,
                    RnnInput::RowHidden,
                    RnnInput::Feature,
                ],
                chunks: [RnnChunk::Gate, RnnChunk::Candidate, RnnChunk::Out],
                folded_through_output: true,
                softcap: ink.final_logit_softcapping,
            }),
            confidence: None,
        })),
        draft_source: DraftSourcePlan::Embedded,
        sampling_defaults: None,
        // A loop revisits its layers; no pipeline cut is declared.
        partition_boundaries: Vec::new(),
        deepseek_v41: None,
        hy_v4_flash: None,
        inkling2: Some(Inkling2Plan {
            embedding_norm: norm,
            loop_plan: LoopPlan {
                core_start: ink.loop_core_start,
                core_end: ink.loop_core_end,
                passes: ink.loop_num_passes,
                gate_rank: ink.loop_gate_rank,
                // The message gate's norms take the config's RMS epsilon; the residual and
                // message mixes take the program's constant.
                controller_epsilon: ink.rms_norm_eps,
                targets: [
                    LoopMessageTarget::Query,
                    LoopMessageTarget::Key,
                    LoopMessageTarget::Value,
                    LoopMessageTarget::Mlp,
                ],
                slots: loop_slots,
                execution_order,
            },
            mlp_output_conv: ShortConvPlan {
                kernel: ink.sconv_kernel_size,
                channels: ink.hidden_size,
                taps: ShortConvTaps::Static,
            },
            output_queries: 1,
            numerics: Inkling2Numerics {
                default_class: Inkling2NumericClass::DeclaredActivationQuant,
                bring_up_class: Inkling2NumericClass::FloatActivation,
                class: Inkling2Numerics::class_of(&experts),
                default_weight_class: Inkling2WeightClass::Nvfp4Rig,
                weight_class,
                minted: if rig {
                    nvfp4_rig::class_tensors(ink, loop_attention)
                        .map_err(&numerics_error)?
                        .iter()
                        .map(|t| t.name().to_string())
                        .collect()
                } else {
                    Vec::new()
                },
                kv: vec![
                    if rig {
                        KvCacheFormat::E4m3Unit
                    } else {
                        KvCacheFormat::F32
                    };
                    slots as usize
                ],
                experts,
                dense,
                gate_up_interleaved: true,
                routed_weight_before_down: true,
                quant_config_sha256: quant.sha256.clone(),
            },
            vision_executed: false,
        }),
    })
}

/// The declaration and the derived census name the same tensors: every tensor the checkpoint
/// must hold is in a parameter group of its file's part, and every declared tensor is one the
/// checkpoint must hold. Quantization planes are declared through their bank.
fn declaration_covers_census(
    config: &ModelConfig,
    quant: &crate::quant_groups::QuantGroups,
) -> Result<(), String> {
    let expected = expected_census(config)?;
    let mut model = 0usize;
    let mut drafter = 0usize;
    for name in expected.keys() {
        if plane_owner(name, &expected).is_some() {
            continue;
        }
        match name.strip_prefix(DRAFTER_PREFIX) {
            Some(tensor) => {
                drafter += 1;
                if quant.mtp_group(tensor).is_none() {
                    return Err(format!("no mtp parameter group declares {tensor}"));
                }
            }
            None => {
                model += 1;
                if quant.model_group(name).is_none() {
                    return Err(format!("no model parameter group declares {name}"));
                }
            }
        }
    }
    // Names are unique within a part, so equal counts leave no declared tensor unaccounted for;
    // the loops below only name the first stray one.
    for (part, groups, count, prefix) in [
        ("model", &quant.model, model, ""),
        ("mtp", &quant.mtp, drafter, DRAFTER_PREFIX),
    ] {
        if groups
            .iter()
            .map(|group| group.tensors.len())
            .sum::<usize>()
            == count
        {
            continue;
        }
        for tensor in groups.iter().flat_map(|group| &group.tensors) {
            if !expected.contains_key(&format!("{prefix}{tensor}")) {
                return Err(format!(
                    "the {part} parameter groups declare {tensor}, which the config does not derive"
                ));
            }
        }
    }
    Ok(())
}

/// The dense tensor class of a trunk tensor; `None` for expert banks.
fn dense_class(name: &str) -> Option<DenseTensorClass> {
    let rest = name.strip_prefix("model.llm.")?;
    Some(match rest {
        "embed.weight" => DenseTensorClass::Embedding,
        "unembed.weight" => DenseTensorClass::OutputProjection,
        _ if rest.contains(".mlp.experts.") => return None,
        _ if rest.ends_with(".mlp.gate.weight") => DenseTensorClass::Router,
        _ if rest.ends_with(".mlp.gate.bias") => DenseTensorClass::RouterBias,
        _ if rest.contains(".controllers.") => DenseTensorClass::LoopControllers,
        _ if rest.contains("sconv.") => DenseTensorClass::ShortConvs,
        _ if rest.starts_with("attn_res.") || rest.starts_with("middle_cycle_cross_pass_") => {
            DenseTensorClass::ResidualQueries
        }
        _ if rest.ends_with("norm.weight") => DenseTensorClass::Norms,
        _ if rest.contains(".attn.") => DenseTensorClass::AttentionProjections,
        _ => return None,
    })
}

/// Numerics of every dense tensor class, read from the parameter groups. Each trunk tensor
/// that is not an expert bank must fall in a class, and a class must be declared one way.
fn dense_numerics(
    quant: &crate::quant_groups::QuantGroups,
) -> Result<Vec<DenseNumericsPlan>, String> {
    const CLASSES: [DenseTensorClass; 9] = [
        DenseTensorClass::AttentionProjections,
        DenseTensorClass::Router,
        DenseTensorClass::RouterBias,
        DenseTensorClass::LoopControllers,
        DenseTensorClass::ShortConvs,
        DenseTensorClass::Norms,
        DenseTensorClass::ResidualQueries,
        DenseTensorClass::Embedding,
        DenseTensorClass::OutputProjection,
    ];
    let mut found: Vec<Option<DenseNumericsPlan>> = vec![None; CLASSES.len()];
    for group in &quant.model {
        for tensor in &group.tensors {
            if !tensor.starts_with("model.llm.") || tensor.contains(".mlp.experts.") {
                continue;
            }
            let class = dense_class(tensor)
                .ok_or_else(|| format!("{tensor} is in no typed dense tensor class"))?;
            if group.compute_dtype.is_quantized() {
                return Err(format!(
                    "{tensor} is declared {:?}; only the routed experts are quantized in this program",
                    group.compute_dtype
                ));
            }
            let this = DenseNumericsPlan {
                class,
                weight: group.compute_dtype,
                output_f32: match group.fwd_output_dtype.as_deref() {
                    None => false,
                    Some("float32") => true,
                    Some(other) => {
                        return Err(format!("{tensor} declares fwd_output_dtype {other:?}"));
                    }
                },
            };
            let slot = &mut found[CLASSES
                .iter()
                .position(|&c| c == class)
                .expect("listed class")];
            if slot.is_some_and(|first| first != this) {
                return Err(format!(
                    "tensor class {class:?} is declared two ways (at {tensor})"
                ));
            }
            *slot = Some(this);
        }
    }
    CLASSES
        .iter()
        .zip(found)
        .map(|(class, numerics)| {
            numerics.ok_or_else(|| format!("no tensor of class {class:?} is declared"))
        })
        .collect()
}

fn layer_plan(ink: &Inkling2Config, slot: u32, norm: NormPlan) -> LayerPlan {
    let physical = ink.physical_layer(slot);
    let local = ink.is_local(physical);
    let kv_width = ink.num_key_value_heads * ink.head_dim;
    let window = local.then_some(ink.sliding_window_size);
    let kv_conv = ShortConvPlan {
        kernel: ink.sconv_kernel_size,
        channels: kv_width,
        taps: if local {
            ShortConvTaps::Static
        } else {
            ShortConvTaps::StaticPlusDynamic {
                groups: ink.num_key_value_heads,
            }
        },
    };
    LayerPlan {
        index: slot,
        pre_attention_norm: norm,
        attention: AttentionPlan::RelativeBias(RelativeBiasAttentionPlan {
            query_heads: ink.num_attention_heads,
            kv_heads: ink.num_key_value_heads,
            head_dim: ink.head_dim,
            window,
            inputs: if local {
                AttentionInputs::SharedNorm
            } else {
                AttentionInputs::SeparateQkvNorms
            },
            query_clip: L2ClipPlan {
                threshold: ink.qk_l2_clip_threshold,
                epsilon: PROGRAM_EPSILON,
                scale: ClipScale::PerHeadChannel,
            },
            key_clip: L2ClipPlan {
                threshold: ink.qk_l2_clip_threshold,
                epsilon: PROGRAM_EPSILON,
                scale: ClipScale::None,
            },
            value_norm: ValueNorm::WeightlessRms,
            value_norm_epsilon: ink.rms_norm_eps,
            relative: RelativeBiasPlan {
                rank: ink.d_rel,
                // A sliding layer's table spans its window; a full layer's spans rel_extent.
                extent: if local {
                    ink.sliding_window_size
                } else {
                    ink.rel_extent
                },
                clip: L2ClipPlan {
                    threshold: ink.rel_l2_clip_threshold,
                    epsilon: PROGRAM_EPSILON,
                    scale: ClipScale::PerHeadChannel,
                },
                beyond_extent: if local {
                    BeyondExtent::Masked
                } else {
                    BeyondExtent::ZeroBias
                },
            },
            log_scale: (!local).then_some(LogPositionScalePlan {
                floor: ink.log_scaling_n_floor,
                alpha: ink.log_scaling_alpha,
            }),
            sink: QuerySinkPlan::PerQueryProjection,
            output_gate: OutputGatePlan::SeparatePerDimSigmoid,
            key_conv: kv_conv,
            value_conv: kv_conv,
            output_conv: ShortConvPlan {
                kernel: ink.sconv_kernel_size,
                channels: ink.hidden_size,
                taps: ShortConvTaps::Static,
            },
        }),
        pre_mlp_norm: norm,
        mlp: MlpPlan::Moe(MoeMlpPlan {
            expert_count: ink.n_routed_experts,
            experts_per_token: ink.num_experts_per_tok,
            expert_intermediate_size: ink.intermediate_size,
            // Sigmoid scores; `gate.bias` steers the selection only; the selected raw scores are
            // renormalized and scaled by `route_scale`.
            router: RouterPlan::Sigmoid {
                normalize_selected: true,
                scaling_factor: ink.route_scale,
                selection_bias: ink.use_gate_bias,
            },
            shared: None,
            activation: ActivationPlan::SwiGluSoftcapped {
                gate_cap: ink.mlp_gate_softcap,
                up_cap: ink.mlp_up_softcap,
            },
        }),
        residual: ResidualTopology::BlockAttention(BlockAttentionPlan {
            block: ink.n_layers_per_attn_res_block,
            starts_block: ink.starts_block(physical),
            attention_queries: attention_queries(ink, physical),
            normalize_values: ink.attn_res_normalize_values,
            epsilon: PROGRAM_EPSILON,
        }),
        state: StatePlan::RelativeBias(RelativeBiasStatePlan {
            key_width: kv_width,
            value_width: kv_width,
            window,
            conv_history: ink.sconv_kernel_size - 1,
            conv_channels: ink.conv_state_channels(),
        }),
        sparse_overlay: None,
        ple: None,
    }
}

/// Residual queries of a layer's attention sublayer: none on layer 0 (nothing to mix yet), one
/// on a sliding layer, three (Q, K, V) on a full layer.
fn attention_queries(ink: &Inkling2Config, physical: u32) -> u32 {
    if physical == 0 {
        0
    } else if ink.is_local(physical) {
        1
    } else {
        3
    }
}

/// The complete expected header census, derived from config math: every tensor of the indexed
/// shard under its own name and every tensor of the drafter file under [`DRAFTER_PREFIX`].
/// Shapes are the stored safetensors shapes (packed U8 experts carry `in / 2` bytes per row).
pub fn expected_census(config: &ModelConfig) -> Result<BTreeMap<String, TensorSpec>, String> {
    let ink = config
        .inkling2
        .as_ref()
        .ok_or("inkling2 census needs an inkling2 config")?;
    let hidden = u64::from(ink.hidden_size);
    let vocab = u64::from(ink.vocab_size);
    let heads = u64::from(ink.num_attention_heads);
    let head_dim = u64::from(ink.head_dim);
    let q_width = heads * head_dim;
    let kv_heads = u64::from(ink.num_key_value_heads);
    let kv_width = kv_heads * head_dim;
    let rank = u64::from(ink.d_rel);
    let kernel = u64::from(ink.sconv_kernel_size);
    let experts = u64::from(ink.n_routed_experts);
    let inter = u64::from(ink.intermediate_size);
    let gate_rank = u64::from(ink.loop_gate_rank);
    let mut map = BTreeMap::new();
    let mut put = |name: String, dtype: &'static str, shape: Vec<u64>| {
        assert!(
            map.insert(name.clone(), TensorSpec { dtype, shape })
                .is_none(),
            "inkling2 census derives {name} twice"
        );
    };
    let llm = "model.llm.";
    put(format!("{llm}embed.weight"), "BF16", vec![vocab, hidden]);
    put(format!("{llm}embed_norm.weight"), "BF16", vec![hidden]);
    put(format!("{llm}norm.weight"), "BF16", vec![hidden]);
    put(format!("{llm}unembed.weight"), "BF16", vec![vocab, hidden]);
    put(format!("{llm}attn_res.output_query"), "F32", vec![hidden]);
    for name in [
        "middle_cycle_cross_pass_queries",
        "middle_cycle_cross_pass_kv_queries",
    ] {
        // Two message queries per core layer in each tensor.
        put(
            format!("{llm}{name}"),
            "F32",
            vec![2 * u64::from(ink.core_len()), hidden],
        );
    }
    for layer in 0..ink.num_hidden_layers {
        let local = ink.is_local(layer);
        let p = format!("{llm}layers.{layer}.");
        // `attn_queries.N` belongs to layer N + 1.
        match attention_queries(ink, layer) {
            0 => {}
            1 => put(
                format!("{llm}attn_res.attn_queries.{}", layer - 1),
                "F32",
                vec![hidden],
            ),
            n => put(
                format!("{llm}attn_res.attn_queries.{}", layer - 1),
                "F32",
                vec![u64::from(n), hidden],
            ),
        }
        put(
            format!("{llm}attn_res.mlp_queries.{layer}"),
            "F32",
            vec![hidden],
        );
        put(
            format!("{p}attn.wq_du.weight"),
            "BF16",
            vec![q_width, hidden],
        );
        put(
            format!("{p}attn.wk_dv.weight"),
            "BF16",
            vec![kv_width, hidden],
        );
        put(
            format!("{p}attn.wv_dv.weight"),
            "BF16",
            vec![kv_width, hidden],
        );
        put(
            format!("{p}attn.wo_ud.weight"),
            "BF16",
            vec![hidden, q_width],
        );
        put(
            format!("{p}attn.wr_du.weight"),
            "BF16",
            vec![heads * rank, hidden],
        );
        put(
            format!("{p}attn.gate_attn.weight"),
            "BF16",
            vec![q_width, hidden],
        );
        put(
            format!("{p}attn.attn_sink_proj.weight"),
            "BF16",
            vec![heads, hidden],
        );
        put(
            format!("{p}attn.q_l2_clip.scale"),
            "BF16",
            vec![heads, head_dim],
        );
        put(
            format!("{p}attn.rel_l2_clip.scale"),
            "BF16",
            vec![heads, rank],
        );
        put(
            format!("{p}attn.rel_logits_proj.proj"),
            "BF16",
            vec![
                rank,
                u64::from(if local {
                    ink.sliding_window_size
                } else {
                    ink.rel_extent
                }),
            ],
        );
        for conv in ["k_sconv", "v_sconv"] {
            if local {
                put(
                    format!("{p}attn.{conv}.weight"),
                    "BF16",
                    vec![kv_width, 1, kernel],
                );
            } else {
                put(
                    format!("{p}attn.{conv}.static_w"),
                    "BF16",
                    vec![kernel, kv_width],
                );
                put(
                    format!("{p}attn.{conv}.weight_proj"),
                    "BF16",
                    vec![kernel * kv_heads, hidden],
                );
            }
        }
        if local {
            put(format!("{p}attn_norm.weight"), "BF16", vec![hidden]);
        } else {
            for input in ["q", "k", "v"] {
                put(
                    format!("{p}{input}_input_norm.weight"),
                    "BF16",
                    vec![hidden],
                );
            }
        }
        put(
            format!("{p}attn_sconv.weight"),
            "BF16",
            vec![hidden, 1, kernel],
        );
        put(
            format!("{p}mlp_sconv.weight"),
            "BF16",
            vec![hidden, 1, kernel],
        );
        put(format!("{p}mlp_norm.weight"), "BF16", vec![hidden]);
        put(format!("{p}mlp.gate.weight"), "BF16", vec![experts, hidden]);
        put(format!("{p}mlp.gate.bias"), "F32", vec![experts]);
        if (ink.loop_core_start..ink.loop_core_end).contains(&layer) {
            for controller in ["attn", "k", "v", "mlp"] {
                let c = format!("{p}controllers.{controller}.");
                put(
                    format!("{c}down_weight"),
                    "BF16",
                    vec![gate_rank, 2 * hidden],
                );
                put(format!("{c}forget_weight"), "BF16", vec![hidden, gate_rank]);
                put(format!("{c}gain_weight"), "BF16", vec![hidden, gate_rank]);
            }
        }
        // Expert banks: gate and up interleaved in w13 [experts, 2 * inter, hidden], down in w2
        // [experts, hidden, inter].
        let format = ink.expert_quant(layer)?.compute;
        for (bank, rows, inputs) in [
            ("w13_weight", 2 * inter, hidden),
            ("w2_weight", hidden, inter),
        ] {
            let name = format!("{p}mlp.experts.{bank}");
            match format {
                ComputeDtype::Nvfp4 => {
                    put(name.clone(), "U8", vec![experts, rows, inputs / 2]);
                    put(
                        format!("{name}_scale"),
                        "F8_E4M3",
                        vec![experts, rows, inputs / 16],
                    );
                    put(format!("{name}_amax"), "F32", vec![experts]);
                    put(format!("{name}_outer_scale_denominator"), "F32", vec![1]);
                }
                ComputeDtype::Mxfp8 => {
                    put(name.clone(), "F8_E4M3", vec![experts, rows, inputs]);
                    put(
                        format!("{name}_scale"),
                        "U8",
                        vec![experts, rows, inputs / 32],
                    );
                }
                other => {
                    return Err(format!(
                        "layer {layer} experts compute in {other:?}; only nvfp4 and mxfp8 banks are typed"
                    ));
                }
            }
        }
    }

    // ViT tower and projector (contract-owned, not in the text plan).
    let v = &ink.vision;
    let vd = u64::from(v.hidden_size);
    let vmlp = (f64::from(v.hidden_size) * f64::from(v.mlp_ratio)) as u64;
    let patch = u64::from(v.native_patch_size);
    let grid = u64::from(v.posemb_grid_size);
    let projection = u64::from(v.projection_hidden_dim);
    let merge = u64::from(v.adapter_merge_size);
    let vis = "model.visual.";
    let emb = format!("{vis}backbone.embeddings.");
    put(format!("{emb}class_embedding"), "BF16", vec![vd]);
    put(format!("{emb}cls_position_embedding"), "BF16", vec![vd]);
    put(
        format!("{emb}patch_embedding.weight"),
        "BF16",
        vec![vd, 3 * patch * patch],
    );
    put(
        format!("{emb}position_embedding"),
        "BF16",
        vec![grid * grid, vd],
    );
    put(format!("{vis}backbone.ln_pre.weight"), "BF16", vec![vd]);
    put(format!("{vis}backbone.ln_pre.bias"), "BF16", vec![vd]);
    for block in 0..v.num_hidden_layers {
        let p = format!("{vis}backbone.layers.{block}.");
        for norm in ["ln_1", "ln_2"] {
            put(format!("{p}{norm}.weight"), "BF16", vec![vd]);
            put(format!("{p}{norm}.bias"), "BF16", vec![vd]);
        }
        put(format!("{p}ls_1.gamma"), "BF16", vec![vd]);
        put(format!("{p}ls_2.gamma"), "BF16", vec![vd]);
        put(format!("{p}mlp.c_fc.weight"), "BF16", vec![vmlp, vd]);
        put(format!("{p}mlp.c_fc.bias"), "BF16", vec![vmlp]);
        put(format!("{p}mlp.c_proj.weight"), "BF16", vec![vd, vmlp]);
        put(format!("{p}mlp.c_proj.bias"), "BF16", vec![vd]);
        for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            put(
                format!("{p}self_attn.{projection}.weight"),
                "BF16",
                vec![vd, vd],
            );
            put(format!("{p}self_attn.{projection}.bias"), "BF16", vec![vd]);
        }
    }
    put(
        format!("{vis}proj_in.weight"),
        "BF16",
        vec![projection, vd * merge * merge],
    );
    put(
        format!("{vis}proj_out.weight"),
        "BF16",
        vec![u64::from(v.text_hidden_size), projection],
    );

    // DSpark drafter (`mtp.safetensors`; contract-owned, not executed). All BF16.
    let d = &ink.drafter;
    let dq = u64::from(d.num_attention_heads) * u64::from(d.head_dim);
    let dkv = u64::from(d.num_key_value_heads) * u64::from(d.head_dim);
    let dhead = u64::from(d.head_dim);
    let dinter = u64::from(d.intermediate_size);
    let markov = u64::from(d.markov_rank);
    let taps = d.target_layer_ids.len() as u64;
    let m = DRAFTER_PREFIX;
    // One fuse row per drafter block plus one for the head, over the taps.
    put(
        format!("{m}fuse_weight"),
        "BF16",
        vec![u64::from(d.num_hidden_layers) + 1, taps],
    );
    put(format!("{m}mask_embed"), "BF16", vec![hidden]);
    put(format!("{m}norm.weight"), "BF16", vec![hidden]);
    for block in 0..d.num_hidden_layers {
        let p = format!("{m}layers.{block}.");
        for norm in ["dflare_norm", "input_layernorm", "post_attention_layernorm"] {
            put(format!("{p}{norm}.weight"), "BF16", vec![hidden]);
        }
        put(
            format!("{p}self_attn.q_proj.weight"),
            "BF16",
            vec![dq, hidden],
        );
        for projection in ["k_proj", "v_proj", "k_proj_ctx", "v_proj_ctx"] {
            put(
                format!("{p}self_attn.{projection}.weight"),
                "BF16",
                vec![dkv, hidden],
            );
        }
        put(
            format!("{p}self_attn.o_proj.weight"),
            "BF16",
            vec![hidden, dq],
        );
        for norm in ["q_norm", "k_norm", "k_norm_ctx"] {
            put(format!("{p}self_attn.{norm}.weight"), "BF16", vec![dhead]);
        }
        put(
            format!("{p}mlp.gate_proj.weight"),
            "BF16",
            vec![dinter, hidden],
        );
        put(
            format!("{p}mlp.up_proj.weight"),
            "BF16",
            vec![dinter, hidden],
        );
        put(
            format!("{p}mlp.down_proj.weight"),
            "BF16",
            vec![hidden, dinter],
        );
    }
    let head = format!("{m}markov_head.");
    put(format!("{head}dflare_norm.weight"), "BF16", vec![hidden]);
    put(
        format!("{head}markov_w1.weight"),
        "BF16",
        vec![vocab, markov],
    );
    // Recurrent head: [gate | candidate | out] rows over [state | embedding | hidden | feature].
    put(
        format!("{head}rnn_proj.weight"),
        "BF16",
        vec![3 * markov, 2 * markov + 2 * hidden],
    );
    put(format!("{head}rnn_proj.bias"), "BF16", vec![3 * markov]);
    put(
        format!("{head}hidden_out.weight"),
        "BF16",
        vec![hidden, markov],
    );
    put(format!("{head}hidden_out.bias"), "BF16", vec![hidden]);
    Ok(map)
}

/// The bank a quantization plane belongs to, when `name` is one: an expert bank's
/// (`*_weight`) or a minted dense tensor's (`*.weight`).
fn plane_owner<'a>(name: &'a str, expected: &BTreeMap<String, TensorSpec>) -> Option<&'a str> {
    BANK_PLANES.into_iter().find_map(|suffix| {
        name.strip_suffix(suffix).filter(|bank| {
            (bank.ends_with("_weight") || bank.ends_with(".weight")) && expected.contains_key(*bank)
        })
    })
}

/// The census a plan binds: the shipped census, with every tensor the plan reads as minted
/// NVFP4 (`nvfp4-rig`) in the overlay's NVFP4 layout instead of its stored one.
pub fn expected_census_for(
    config: &ModelConfig,
    plan: &ModelPlan,
) -> Result<BTreeMap<String, TensorSpec>, String> {
    let mut map = expected_census(config)?;
    let Some(program) = plan.inkling2.as_ref() else {
        return Ok(map);
    };
    if program.numerics.weight_class == Inkling2WeightClass::Shipped {
        return Ok(map);
    }
    let ink = inkling(config).map_err(|e| format!("{e:?}"))?;
    let set = nvfp4_rig::class_tensors(
        ink,
        program.numerics.weight_class == Inkling2WeightClass::Nvfp4Rig,
    )?;
    let names: Vec<&str> = set.iter().map(|t| t.name()).collect();
    if names
        != program
            .numerics
            .minted
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
    {
        return Err("the plan's minted list is not the nvfp4-rig set of its config".into());
    }
    for t in &set {
        let spec = map
            .get(t.name())
            .ok_or_else(|| format!("{} is not in the shipped census", t.name()))?;
        let mut logical: Vec<usize> = spec.shape.iter().map(|v| *v as usize).collect();
        if spec.dtype == "U8" {
            *logical.last_mut().expect("rank") *= 2;
        }
        map.remove(&format!("{}_scale", t.name()));
        for (plane, dtype, shape) in nvfp4_rig::overlay_planes(t, &logical) {
            map.insert(
                plane,
                TensorSpec {
                    dtype,
                    shape: shape.iter().map(|v| *v as u64).collect(),
                },
            );
        }
    }
    Ok(map)
}

#[allow(clippy::result_large_err)] // allow: the fat error type is the diagnostic contract here; boxing it would change the error surface
fn tensor_schema(
    config: &ModelConfig,
    plan: &ModelPlan,
    dialect: CheckpointDialect,
    _options: ContractOptions,
) -> Result<TensorContract, TensorContractError> {
    if dialect != CheckpointDialect::HfSafetensors {
        return Err(TensorContractError::UnsupportedPlanOperation {
            operation: "inkling2 non-safetensors schema",
        });
    }
    if plan.inkling2.is_none() {
        return Err(TensorContractError::UnsupportedPlanOperation {
            operation: "inkling2 schema without its plan",
        });
    }
    let Some(DrafterPlan::Dspark(DsparkPlan {
        layers: DsparkLayers::Context(draft_layers),
        ..
    })) = plan.drafter.as_ref()
    else {
        return Err(TensorContractError::UnsupportedPlanOperation {
            operation: "inkling2 schema without its DSpark drafter",
        });
    };
    let draft_layers = draft_layers.layers;
    let expected = expected_census_for(config, plan).map_err(|_| {
        TensorContractError::UnsupportedPlanOperation {
            operation: "inkling2 schema without typed expert formats",
        }
    })?;
    let mut requirements = Vec::new();
    for (name, spec) in &expected {
        if plane_owner(name, &expected).is_some() {
            continue;
        }
        let planes: Vec<String> = BANK_PLANES
            .into_iter()
            .map(|suffix| format!("{name}{suffix}"))
            .filter(|plane| expected.contains_key(plane))
            .collect();
        let mut shape = spec.shape.clone();
        let quant = match (spec.dtype, planes.len()) {
            ("BF16", 0) => QuantConstraint::ExactFloat(FloatType::Bf16),
            ("F32", 0) => QuantConstraint::ExactFloat(FloatType::F32),
            ("U8", 3) => {
                // e2m1 pairs: the requirement carries the logical [experts, out, in].
                *shape.last_mut().expect("expert bank rank") *= 2;
                QuantConstraint::Nvfp4Amax
            }
            ("F8_E4M3", 1) => QuantConstraint::Mxfp8Row32,
            ("U8", 2) => {
                // a minted dense tensor (nvfp4-rig): e2m1 pairs, per-16 e4m3 scales, scale_2
                *shape.last_mut().expect("matrix rank") *= 2;
                QuantConstraint::Nvfp4
            }
            _ => {
                return Err(TensorContractError::UnsupportedPlanOperation {
                    operation: "unknown inkling2 dtype or plane pairing",
                });
            }
        };
        requirements.push(TensorRequirement {
            id: semantic_tensor_id(name),
            names: vec![name.clone()],
            match_mode: TensorMatch::OneOf,
            shape,
            owner: tensor_owner(name, draft_layers),
            transform: TensorTransform::Identity,
            quant,
            auxiliaries: Some(planes),
            required: true,
        });
    }
    Ok(TensorContract {
        dialect,
        requirements,
    })
}

fn tensor_owner(name: &str, drafter_blocks: u32) -> TensorOwner {
    let index_after = |prefix: &str| {
        name.strip_prefix(prefix)
            .and_then(|rest| rest.split('.').next())
            .and_then(|index| index.parse::<u32>().ok())
    };
    if let Some(layer) = index_after("model.llm.layers.") {
        TensorOwner::Layer(layer)
    } else if name.starts_with("model.visual.") {
        TensorOwner::Vision(index_after("model.visual.backbone.layers."))
    } else if let Some(rest) = name.strip_prefix(DRAFTER_PREFIX) {
        // Input-side glue belongs to the first block, the head and exit norm to the last.
        TensorOwner::Mtp(index_after("mtp.layers.").unwrap_or(
            if rest == "fuse_weight" || rest == "mask_embed" {
                0
            } else {
                drafter_blocks.saturating_sub(1)
            },
        ))
    } else {
        TensorOwner::Global
    }
}

/// Typed ids where an existing role is exactly this tensor's role; family keys for everything
/// the family alone has. Layer ids are keyed on the PHYSICAL layer: loop slots share them.
pub fn semantic_tensor_id(name: &str) -> TensorId {
    match name {
        "model.llm.embed.weight" => return TensorId::TokenEmbedding,
        "model.llm.unembed.weight" => return TensorId::OutputProjection,
        "model.llm.norm.weight" => return TensorId::OutputNorm,
        _ => {}
    }
    let family = || TensorId::Family {
        family: FAMILY,
        key: name.strip_prefix("model.").unwrap_or(name).to_string(),
    };
    let Some((index, suffix)) = name
        .strip_prefix("model.llm.layers.")
        .and_then(|rest| rest.split_once('.'))
        .and_then(|(index, suffix)| Some((index.parse::<u32>().ok()?, suffix)))
    else {
        return family();
    };
    let tensor = match suffix {
        "mlp_norm.weight" => LayerTensor::PreMlpNorm,
        "mlp.gate.weight" => LayerTensor::MoeRouter,
        "mlp.gate.bias" => LayerTensor::MoeRouterBias,
        "mlp.experts.w13_weight" => LayerTensor::MoeExpertGateUpBank,
        "mlp.experts.w2_weight" => LayerTensor::MoeExpertDownBank,
        _ => return family(),
    };
    TensorId::Layer { index, tensor }
}

#[cfg(test)]
mod tests;

pub mod nvfp4_rig;
