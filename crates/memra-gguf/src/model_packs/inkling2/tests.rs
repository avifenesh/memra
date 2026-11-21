use std::collections::BTreeMap;

use super::*;
use crate::config::HfConfig;
use crate::model_plan::OperationKind;
use crate::quant_groups::{
    ActivationAmax, ActivationQuant, ActivationScale, ComputeDtype, ScaleSelection,
};
use crate::safetensors::StInfo;
use crate::source::{TensorCensus, census_from_safetensors_headers};
use crate::tensor_contract::{StorageLayout, TensorCensusEntry};

/// `config.json` of thinkingmachines/Inkling-2-Mini at ed40e7da625f594fc9c4990f5f66be24873c22d7,
/// byte-identical (sha256 a7a349601f0984c4abf999c480ef775d8567d21f5f84b4e8a3e035b44f288dc2).
const CONFIG: &str = include_str!("fixtures/config.json");
/// `hf_quant_config.json` of the same revision, byte-identical
/// (sha256 7c656207d3ce37879c7be209a036b0dde43bf01f78159bff06880b868ed57143).
const QUANT: &str = include_str!("fixtures/hf_quant_config.json");
/// Every header tensor of the same revision, both weight files, grouped by name class
/// (research/modelplan-onboarding-inkling2mini-20261121/census.py, lossless by construction).
const CENSUS: &str = include_str!("fixtures/census-classes.tsv");

fn bare_config() -> ModelConfig {
    ModelConfig::from_hf(&HfConfig::parse(CONFIG))
}

fn config() -> ModelConfig {
    let mut config = bare_config();
    config.attach_quant_groups(QUANT).unwrap();
    config
}

fn plan() -> ModelPlan {
    PACK.compile_plan(&config()).unwrap()
}

/// Mutate one `"key": value` of the pinned config text.
fn config_with(from: &str, to: &str) -> String {
    assert_eq!(CONFIG.matches(from).count(), 1, "{from} is not unique");
    CONFIG.replace(from, to)
}

#[test]
fn parses_the_pinned_config_and_derives_the_loop_geometry() {
    assert_eq!(
        Arch::from_hf_model_type("inkling2_loop"),
        Arch::Inkling2Loop
    );
    assert_eq!(
        Arch::from_hf_model_type("inkling2_loop_text"),
        Arch::Inkling2Loop
    );
    assert_eq!(
        Arch::Inkling2Loop.attention_gate_kind(),
        Some(crate::config::AttentionGateKind::SeparatePerDim)
    );
    let config = bare_config();
    assert_eq!(config.arch, Arch::Inkling2Loop);
    assert!(config.vision.is_none(), "no generic tower is fabricated");
    assert!(config.multimodal.is_none());
    assert_eq!(config.n_layer, 24);
    assert_eq!(config.n_embd, 2304);
    assert_eq!((config.n_head, config.n_head_kv), (24, 8));
    assert_eq!(config.n_vocab, 201_024);
    assert_eq!(config.context_length, 1_048_576);
    assert!(config.needs_quant_groups());

    let ink = config.inkling2.as_ref().unwrap();
    assert_eq!(ink.unpadded_vocab_size, 200_058);
    assert_eq!(ink.eos_token_id, 200_006);
    assert_eq!(ink.qk_l2_clip_threshold.to_bits(), 22.627_417f32.to_bits());
    assert_eq!(
        (ink.d_rel, ink.rel_extent, ink.sliding_window_size),
        (16, 1024, 1024)
    );
    assert_eq!(
        (ink.loop_core_start, ink.loop_core_end, ink.loop_num_passes),
        (6, 18, 3)
    );
    assert_eq!(ink.slots(), 48);
    assert_eq!(ink.conv_state_channels(), 6656);
    let full: Vec<u32> = (0..24).filter(|&layer| !ink.is_local(layer)).collect();
    assert_eq!(full, [5, 11, 17, 23]);
    // Slots past the physical layers revisit the core.
    assert_eq!(ink.physical_layer(23), 23);
    assert_eq!(ink.physical_layer(24), 6);
    assert_eq!(ink.physical_layer(35), 17);
    assert_eq!(ink.physical_layer(36), 6);
    assert_eq!(ink.physical_layer(47), 17);
    assert_eq!(ink.slot_pass(5), None);
    assert_eq!(ink.slot_pass(6), Some(0));
    assert_eq!(ink.slot_pass(29), Some(1));
    assert_eq!(ink.slot_pass(47), Some(2));
    assert_eq!(ink.slot_pass(18), None);
    let order = ink.execution_order();
    assert_eq!(order.len(), 48);
    assert_eq!(&order[..8], &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(&order[17..20], &[17, 24, 25]);
    assert_eq!(&order[29..31], &[35, 36]);
    assert_eq!(&order[41..], &[47, 18, 19, 20, 21, 22, 23]);
    let mut sorted = order.clone();
    sorted.sort();
    assert_eq!(sorted, (0..48).collect::<Vec<_>>(), "every slot runs once");

    assert_eq!(ink.vision.num_hidden_layers, 23);
    assert_eq!(ink.drafter.target_layer_ids, [5, 23, 41, 48]);
    assert_eq!(ink.drafter.markov_head_type, "rnn_hidden");
}

#[test]
fn a_missing_field_or_an_unrunnable_topology_is_refused_by_name() {
    let refuse = |text: String, needle: &str| {
        let Err(error) = HfConfig::try_parse(&text) else {
            panic!("must refuse");
        };
        assert!(error.contains(needle), "{error}");
    };
    refuse(
        config_with(r#""d_rel": 16,"#, ""),
        "inkling2 config.json missing required field text_config.d_rel",
    );
    refuse(
        config_with(r#""gate_rank": 64,"#, ""),
        "missing required field text_config.loop_config.gate_rank",
    );
    refuse(
        config_with(r#""markov_head_type": "rnn_hidden","#, ""),
        "missing required field mtp_config.markov_head_type",
    );
    refuse(
        config_with(r#""core_start": 6"#, r#""core_start": 5"#),
        "not aligned to the attention-residual block",
    );
    refuse(
        config_with(r#""num_passes": 3"#, r#""num_passes": 1"#),
        "a fixed loop needs at least 2",
    );
    refuse(
        config_with(r#""is_varloop": false"#, r#""is_varloop": true"#),
        "loop_config.is_varloop is true",
    );
    refuse(
        config_with(r#""n_shared_experts": 0"#, r#""n_shared_experts": 1"#),
        "n_shared_experts is 1",
    );
    refuse(
        config_with(r#""swa_head_dim": 128"#, r#""swa_head_dim": 64"#),
        "sliding-window geometry",
    );
    refuse(
        config_with(
            r#""unpadded_vocab_size": 200058"#,
            r#""unpadded_vocab_size": 201025"#,
        ),
        "exceeds vocab_size",
    );
    refuse(
        config_with(
            r#""markov_head_type": "rnn_hidden""#,
            r#""markov_head_type": "bigram""#,
        ),
        "only \"rnn_hidden\" is declared",
    );
    refuse(
        config_with(r#""quant_algo": "NVFP4""#, r#""quant_algo": "FP8""#),
        "the pinned declaration is modelopt/NVFP4",
    );
}

#[test]
fn the_pack_selects_by_config_and_alias_and_only_it_compiles_the_family() {
    let config = config();
    assert_eq!(for_config(&config).unwrap().family, "inkling2");
    for alias in ["inkling2", "inkling2_loop", "inkling-2-mini"] {
        assert_eq!(by_alias(alias).unwrap().family, "inkling2");
    }
    assert_eq!(PACK.support, Some(NativeSupport::NativeQualified));
    assert_eq!(PACK.gates.len(), 7);
    assert_eq!(
        ModelPlan::compile(&config),
        Err(PlanCompileError::UnsupportedSemantics {
            field: "inkling2",
            value: "compiled only by the inkling2 model pack".into(),
        })
    );
}

#[test]
fn the_plan_refuses_to_compile_without_the_quant_declaration() {
    let error = PACK.compile_plan(&bare_config()).unwrap_err();
    assert!(
        matches!(
            &error,
            PlanCompileError::UnsupportedSemantics { field: "hf_quant_config.json", value }
                if value.contains("not attached")
        ),
        "{error:?}"
    );
    // A declaration that leaves a layer's experts on a float input is not this program.
    let mut config = bare_config();
    let stripped = QUANT.replacen(r#""model.llm.layers.0.mlp.experts.w13_weight","#, "", 1);
    assert_ne!(stripped, QUANT);
    let error = config.attach_quant_groups(&stripped).unwrap_err();
    assert!(
        error.contains("declares no group for model.llm.layers.0.mlp.experts.w13_weight"),
        "{error}"
    );
    assert!(
        config.needs_quant_groups(),
        "a refused attach leaves nothing attached"
    );
}

#[test]
fn the_plan_types_the_48_slot_program() {
    let plan = plan();
    assert_eq!(plan.layers.len(), 48);
    assert_eq!((plan.hidden_size, plan.vocab_size), (2304, 201_024));
    assert_eq!(
        plan.logits,
        vec![
            LogitsTransform::Softcap(30.0),
            LogitsTransform::Truncate(200_058)
        ]
    );
    assert!(plan.drafter.is_none() && plan.mtp_blocks.is_empty());
    assert!(plan.partition_boundaries.is_empty());
    let program = plan.inkling2.as_ref().unwrap();
    let loop_plan = &program.loop_plan;
    assert_eq!(loop_plan.slots.len(), 48);
    assert_eq!(loop_plan.execution_order.len(), 48);
    assert_eq!((loop_plan.passes, loop_plan.gate_rank), (3, 64));

    let mut full_slots = Vec::new();
    for (slot, layer) in plan.layers.iter().enumerate() {
        assert_eq!(layer.index as usize, slot);
        let entry = loop_plan.slots[slot];
        let physical = entry.physical_layer;
        let AttentionPlan::RelativeBias(attention) = &layer.attention else {
            panic!("slot {slot} is not relative-bias attention");
        };
        let ResidualTopology::BlockAttention(residual) = layer.residual else {
            panic!("slot {slot} has no block attention residual");
        };
        let StatePlan::RelativeBias(state) = layer.state else {
            panic!("slot {slot} has no relative-bias state");
        };
        assert_eq!(
            (
                attention.query_heads,
                attention.kv_heads,
                attention.head_dim
            ),
            (24, 8, 128)
        );
        assert_eq!(attention.value_norm, ValueNorm::WeightlessRms);
        assert_eq!(attention.query_clip.scale, ClipScale::PerHeadChannel);
        assert_eq!(attention.key_clip.scale, ClipScale::None);
        assert_eq!(attention.relative.clip.threshold, 8.0);
        assert_eq!(state.conv_channels, 6656);
        assert_eq!(state.conv_history, 3);
        assert_eq!(residual.block, 6);
        assert_eq!(residual.starts_block, physical.is_multiple_of(6));
        let full = [5, 11, 17, 23].contains(&physical);
        if full {
            full_slots.push(slot);
            assert_eq!(attention.window, None);
            assert_eq!(attention.inputs, AttentionInputs::SeparateQkvNorms);
            assert_eq!(
                attention.log_scale,
                Some(LogPositionScalePlan {
                    floor: 8192,
                    alpha: 0.1
                })
            );
            assert_eq!(attention.relative.beyond_extent, BeyondExtent::ZeroBias);
            assert_eq!(
                attention.key_conv.taps,
                ShortConvTaps::StaticPlusDynamic { groups: 8 }
            );
            assert_eq!(residual.attention_queries, 3);
            assert_eq!(state.window, None);
        } else {
            assert_eq!(attention.window, Some(1024));
            assert_eq!(attention.inputs, AttentionInputs::SharedNorm);
            assert_eq!(attention.log_scale, None);
            assert_eq!(attention.relative.beyond_extent, BeyondExtent::Masked);
            assert_eq!(attention.key_conv.taps, ShortConvTaps::Static);
            assert_eq!(residual.attention_queries, u32::from(physical != 0));
            assert_eq!(state.window, Some(1024));
        }
        assert_eq!(attention.output_conv.taps, ShortConvTaps::Static);
        // Loop slots share weights with their physical layer and carry a message.
        assert_eq!(entry.message, slot >= 24);
        if slot >= 24 {
            assert_eq!(plan.layers[physical as usize].attention, layer.attention);
            assert_eq!(plan.layers[physical as usize].mlp, layer.mlp);
        }
        let MlpPlan::Moe(moe) = &layer.mlp else {
            panic!("slot {slot} is not MoE");
        };
        assert_eq!((moe.expert_count, moe.experts_per_token), (128, 8));
        assert_eq!(
            moe.router,
            RouterPlan::Sigmoid {
                normalize_selected: true,
                scaling_factor: 8.0,
                selection_bias: true,
            }
        );
        assert_eq!(
            moe.activation,
            ActivationPlan::SwiGluSoftcapped {
                gate_cap: 4.0,
                up_cap: 25.0
            }
        );
        assert!(moe.shared.is_none());
    }
    // 4 full layers, 2 of them in the core and visited 3 times: 8 full slots, 40 sliding.
    assert_eq!(full_slots, [5, 11, 17, 23, 29, 35, 41, 47]);

    // The drafter's taps count layer visits; as slots they are the inputs of 5, 29 and 47.
    assert_eq!(
        program.drafter.taps,
        [
            DrafterTap::SlotInput { slot: 5 },
            DrafterTap::SlotInput { slot: 29 },
            DrafterTap::SlotInput { slot: 47 },
            DrafterTap::FinalNorm,
        ]
    );
    assert_eq!(program.drafter.head, DrafterHead::RnnHidden { rank: 256 });
    assert!(!program.vision_executed);
}

#[test]
fn the_declared_activation_quantizers_are_the_default_numeric_class() {
    let plan = plan();
    let numerics = &plan.inkling2.as_ref().unwrap().numerics;
    assert_eq!(
        numerics.default_class,
        Inkling2NumericClass::DeclaredActivationQuant
    );
    assert_eq!(
        numerics.bring_up_class,
        Inkling2NumericClass::FloatActivation
    );
    assert!(numerics.gate_up_interleaved && numerics.routed_weight_before_down);
    assert_eq!(
        numerics.class,
        Inkling2NumericClass::DeclaredActivationQuant
    );
    // Dense classes run as stored; only the output projection declares an f32 forward.
    assert_eq!(numerics.dense.len(), 9);
    for dense in &numerics.dense {
        let f32_stored = matches!(
            dense.class,
            DenseTensorClass::RouterBias | DenseTensorClass::ResidualQueries
        );
        assert_eq!(
            dense.weight,
            if f32_stored {
                ComputeDtype::Float32
            } else {
                ComputeDtype::Bfloat16
            },
            "{:?}",
            dense.class
        );
        assert_eq!(
            dense.output_f32,
            dense.class == DenseTensorClass::OutputProjection,
            "{:?}",
            dense.class
        );
    }
    assert_eq!(
        numerics.quant_config_sha256,
        "7c656207d3ce37879c7be209a036b0dde43bf01f78159bff06880b868ed57143"
    );
    assert_eq!(numerics.experts.len(), 24);
    for expert in &numerics.experts {
        assert_eq!(expert.activation, ExpertActivationMode::Declared);
        if expert.physical_layer < 20 {
            assert_eq!(expert.weight, ExpertWeightFormat::Nvfp4Amax);
            assert_eq!(
                expert.declared_activation,
                ActivationQuant::Nvfp4E2m1 {
                    block_size: 16,
                    amax: ActivationAmax::PerToken,
                    scale: ActivationScale::E4m3,
                    scale_selection: ScaleSelection::FourOverSixMae,
                }
            );
        } else {
            assert_eq!(expert.weight, ExpertWeightFormat::Mxfp8Row32);
            assert_eq!(
                expert.declared_activation,
                ActivationQuant::Mxfp8E4m3 {
                    block_size: 32,
                    scale: ActivationScale::Ue8m0,
                }
            );
        }
    }
    // The class is part of the plan text, so the plan hash moves with it.
    let text = format!("{plan:#?}");
    assert!(text.contains("default_class: DeclaredActivationQuant"));
    assert!(text.contains("FourOverSixMae"));
}

#[test]
fn every_new_operation_is_named_and_no_surface_accepts_the_plan() {
    let plan = plan();
    let operations = plan.operations();
    for operation in [
        OperationKind::EmbeddingNorm,
        OperationKind::LoopWeightSharing,
        OperationKind::LoopMessageMix,
        OperationKind::LoopController,
        OperationKind::Nvfp4W4A4Experts,
        OperationKind::Mxfp8W8A8Experts,
        OperationKind::RelativeBiasAttention,
        OperationKind::AttentionL2Clip,
        OperationKind::AttentionLogScale,
        OperationKind::PerQuerySink,
        OperationKind::PerDimOutputGate,
        OperationKind::ShortConv,
        OperationKind::DynamicShortConv,
        OperationKind::RelativeBiasState,
        OperationKind::BlockAttentionResidual,
        OperationKind::SoftcappedSwiGluActivation,
        OperationKind::DsparkRnnDrafter,
        OperationKind::LogitsTruncate,
    ] {
        assert!(
            operations.contains(&operation),
            "{operation:?} is not emitted"
        );
        assert!(
            crate::op_registry::surfaces(operation).is_none(),
            "{operation:?} claims a surface"
        );
    }
    // The trunk alone never names the drafter; the draft program always does.
    assert!(
        !plan
            .trunk_operations()
            .contains(&OperationKind::DsparkRnnDrafter)
    );
    assert!(
        plan.draft_operations()
            .unwrap()
            .contains(&OperationKind::DsparkRnnDrafter)
    );
    let rewrites = crate::execution_manifest::execution_rewrites(&plan);
    assert!(!rewrites.is_empty());
    for rewrite in &rewrites {
        assert!(
            !rewrite.eligible(),
            "{} accepts a plan nothing implements",
            rewrite.id
        );
        assert!(
            rewrite.blockers.iter().any(|blocker| {
                matches!(
                    blocker,
                    OperationKind::RelativeBiasAttention
                        | OperationKind::LoopWeightSharing
                        | OperationKind::Nvfp4W4A4Experts
                        | OperationKind::DsparkRnnDrafter
                )
            }),
            "{} does not name an inkling2 blocker: {:?}",
            rewrite.id,
            rewrite.blockers
        );
    }
}

#[test]
fn the_numeric_class_is_set_per_layer_and_a_mixed_plan_is_its_own_class() {
    let config = config();
    let default = plan();
    let mut modes = vec![ExpertActivationMode::Declared; 24];
    assert_eq!(
        plan_with_expert_activation(&config, &modes).unwrap(),
        default
    );

    // One NVFP4 layer on float inputs: a mixed class, another plan text, a float-input operation.
    modes[3] = ExpertActivationMode::Float;
    let mixed = plan_with_expert_activation(&config, &modes).unwrap();
    let numerics = &mixed.inkling2.as_ref().unwrap().numerics;
    assert_eq!(numerics.class, Inkling2NumericClass::PerLayerMixed);
    assert_eq!(
        numerics.default_class,
        Inkling2NumericClass::DeclaredActivationQuant
    );
    assert_eq!(numerics.experts[3].activation, ExpertActivationMode::Float);
    assert_eq!(
        numerics.experts[3].declared_activation,
        default.inkling2.as_ref().unwrap().numerics.experts[3].declared_activation,
        "the declaration does not move with the mode"
    );
    assert_ne!(format!("{mixed:#?}"), format!("{default:#?}"));
    let operations = mixed.operations();
    assert!(operations.contains(&OperationKind::Nvfp4FloatInputExperts));
    assert!(operations.contains(&OperationKind::Nvfp4W4A4Experts));
    assert!(!operations.contains(&OperationKind::Mxfp8FloatInputExperts));
    assert!(
        !default
            .operations()
            .contains(&OperationKind::Nvfp4FloatInputExperts)
    );

    // All layers on float inputs is the bring-up class and runs no quantized-input operation.
    let float = plan_with_expert_activation(&config, &[ExpertActivationMode::Float; 24]).unwrap();
    assert_eq!(
        float.inkling2.as_ref().unwrap().numerics.class,
        Inkling2NumericClass::FloatActivation
    );
    let operations = float.operations();
    assert!(!operations.contains(&OperationKind::Nvfp4W4A4Experts));
    assert!(!operations.contains(&OperationKind::Mxfp8W8A8Experts));
    assert!(operations.contains(&OperationKind::Mxfp8FloatInputExperts));
    for operation in [
        OperationKind::Nvfp4FloatInputExperts,
        OperationKind::Mxfp8FloatInputExperts,
    ] {
        assert!(crate::op_registry::surfaces(operation).is_none());
    }

    assert!(plan_with_expert_activation(&config, &modes[..23]).is_err());
}

/// Census name -> (dtype, stored shape).
type PinnedCensus = BTreeMap<String, (String, Vec<u64>)>;

fn parse_ranges(text: &str) -> Vec<u32> {
    text.split(',')
        .flat_map(|part| {
            let (a, b) = part.split_once('-').unwrap_or((part, part));
            a.parse::<u32>().unwrap()..=b.parse::<u32>().unwrap()
        })
        .collect()
}

/// The pinned header census, expanded: census name -> (dtype, stored shape).
fn pinned_census() -> PinnedCensus {
    let mut names = BTreeMap::new();
    for line in CENSUS.lines().filter(|line| !line.starts_with('#')).skip(1) {
        let fields: Vec<&str> = line.split('\t').collect();
        let [file, pattern, dtype, shape, indices] = fields[..] else {
            panic!("bad fixture row {line:?}");
        };
        let prefix = match file {
            "model-00001-of-00001.safetensors" => "",
            "mtp.safetensors" => DRAFTER_PREFIX,
            other => panic!("unknown weight file {other}"),
        };
        let shape: Vec<u64> = shape.split(',').map(|dim| dim.parse().unwrap()).collect();
        let indices: Vec<Option<u32>> = if indices == "-" {
            vec![None]
        } else {
            parse_ranges(indices).into_iter().map(Some).collect()
        };
        for index in indices {
            let name = match index {
                Some(index) => pattern.replace("{N}", &index.to_string()),
                None => pattern.to_string(),
            };
            assert!(
                names
                    .insert(
                        format!("{prefix}{name}"),
                        (dtype.to_string(), shape.clone())
                    )
                    .is_none(),
                "{name} twice"
            );
        }
    }
    names
}

#[test]
fn the_derived_census_is_the_pinned_header_census() {
    let pinned = pinned_census();
    assert_eq!(pinned.len(), 1298);
    assert_eq!(
        pinned
            .keys()
            .filter(|name| name.starts_with(DRAFTER_PREFIX))
            .count(),
        54
    );
    let derived: PinnedCensus = expected_census(&config())
        .unwrap()
        .into_iter()
        .map(|(name, spec)| (name, (spec.dtype.to_string(), spec.shape)))
        .collect();
    for (name, spec) in &pinned {
        assert_eq!(derived.get(name), Some(spec), "{name}");
    }
    for name in derived.keys() {
        assert!(
            pinned.contains_key(name),
            "derived {name} is not in the checkpoint"
        );
    }
}

fn dtype_bytes(dtype: &str) -> u64 {
    match dtype {
        "U8" | "F8_E4M3" => 1,
        "BF16" => 2,
        "F32" => 4,
        other => panic!("unknown dtype {other}"),
    }
}

fn headers_from(census: &PinnedCensus) -> BTreeMap<String, StInfo> {
    let mut offset = 0usize;
    census
        .iter()
        .map(|(name, (dtype, shape))| {
            let bytes = (shape.iter().product::<u64>() * dtype_bytes(dtype)) as usize;
            let info = StInfo {
                dtype: dtype.clone(),
                shape: shape.clone(),
                data_offsets: [offset, offset + bytes],
            };
            offset += bytes;
            (name.clone(), info)
        })
        .collect()
}

fn bind(census: &TensorCensus) -> Result<usize, String> {
    let config = config();
    let plan = PACK.compile_plan(&config).unwrap();
    crate::checkpoint_binding::bind_census(Some(&PACK), &config, &plan, census)
        .map(|binding| binding.bound.tensors.len())
        .map_err(|error| error.to_string())
}

#[test]
fn the_contract_binds_every_tensor_of_the_pinned_census() {
    let pinned = pinned_census();
    let census = census_from_safetensors_headers(&headers_from(&pinned)).unwrap();
    // 1,298 header tensors: 128 of them are quantization planes folded into the 48 expert
    // banks (three per NVFP4 bank on 20 layers, one per MXFP8 bank on 4).
    assert_eq!(census.tensors.len(), 1298 - 128);
    let bytes: u64 = census
        .tensors
        .iter()
        .map(|row| row.entry.physical_bytes)
        .sum();
    assert_eq!(bytes, 19_525_870_752 + 635_795_744);
    let storage = |name: &str| -> &TensorCensusEntry {
        &census
            .tensors
            .iter()
            .find(|row| row.entry.name == name)
            .unwrap_or_else(|| panic!("{name} is not in the census"))
            .entry
    };
    let nvfp4 = storage("model.llm.layers.0.mlp.experts.w13_weight");
    assert_eq!(
        nvfp4.shape,
        [128, 2304, 2304],
        "logical shape: two codes per byte"
    );
    let StorageLayout::Quantized(layout) = &nvfp4.storage else {
        panic!("NVFP4 bank is not quantized storage");
    };
    assert_eq!(
        (layout.format.as_str(), &layout.block_shape[..]),
        ("NVFP4_AMAX", &[16][..])
    );
    assert_eq!(
        layout.auxiliaries,
        [
            "model.llm.layers.0.mlp.experts.w13_weight_scale",
            "model.llm.layers.0.mlp.experts.w13_weight_amax",
            "model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator",
        ]
    );
    let mxfp8 = storage("model.llm.layers.23.mlp.experts.w2_weight");
    assert_eq!(mxfp8.shape, [128, 2304, 1152]);
    let StorageLayout::Quantized(layout) = &mxfp8.storage else {
        panic!("MXFP8 bank is not quantized storage");
    };
    assert_eq!(
        (layout.format.as_str(), &layout.block_shape[..]),
        ("MXFP8_E4M3", &[1, 32][..])
    );
    assert_eq!(bind(&census), Ok(1298 - 128));

    let config = config();
    let plan = PACK.compile_plan(&config).unwrap();
    let binding =
        crate::checkpoint_binding::bind_census(Some(&PACK), &config, &plan, &census).unwrap();
    // Vision and drafter tensors are owned, by the right owner, though nothing executes them.
    let owners = |want: fn(&TensorOwner) -> bool| {
        binding
            .bound
            .tensors
            .values()
            .filter(|tensor| want(&tensor.owner))
            .count()
    };
    assert_eq!(owners(|owner| matches!(owner, TensorOwner::Vision(_))), 422);
    assert_eq!(owners(|owner| matches!(owner, TensorOwner::Mtp(_))), 54);
    assert!(
        binding
            .bound
            .tensors
            .contains_key(&TensorId::OutputProjection)
    );
    assert!(binding.bound.tensors.contains_key(&TensorId::Layer {
        index: 17,
        tensor: LayerTensor::MoeExpertGateUpBank,
    }));
}

#[test]
fn a_missing_extra_reshaped_or_requantized_tensor_is_refused() {
    let pinned = pinned_census();
    let refuse = |mutate: &dyn Fn(&mut PinnedCensus), needle: &str| {
        let mut census = pinned.clone();
        mutate(&mut census);
        let error = match census_from_safetensors_headers(&headers_from(&census)) {
            Ok(census) => bind(&census).expect_err("must refuse"),
            Err(error) => error,
        };
        assert!(error.contains(needle), "{needle:?} not in {error}");
    };
    // A drafter tensor and a vision tensor are owned like trunk tensors.
    refuse(
        &|census| {
            census.remove("mtp.markov_head.rnn_proj.weight");
        },
        "mtp.markov_head.rnn_proj.weight",
    );
    refuse(
        &|census| {
            census.remove("model.visual.proj_out.weight");
        },
        "model.visual.proj_out.weight",
    );
    refuse(
        &|census| {
            census.insert(
                "model.llm.layers.5.attn_norm.weight".into(),
                ("BF16".into(), vec![2304]),
            );
        },
        "model.llm.layers.5.attn_norm.weight",
    );
    refuse(
        &|census| {
            census
                .get_mut("model.llm.layers.5.attn.rel_logits_proj.proj")
                .unwrap()
                .1 = vec![16, 512];
        },
        "rel_logits_proj.proj",
    );
    // A dense tensor in another float type.
    refuse(
        &|census| {
            census
                .get_mut("model.llm.layers.3.mlp.gate.bias")
                .unwrap()
                .0 = "BF16".into();
        },
        "mlp.gate.bias",
    );
    // An NVFP4 bank without its per-expert amax is another layout.
    refuse(
        &|census| {
            census.remove("model.llm.layers.0.mlp.experts.w2_weight_amax");
        },
        "NVFP4 stacked bank",
    );
    // An MXFP8 layer shipped as NVFP4 (and the reverse) is another numeric program.
    refuse(
        &|census| {
            for bank in ["w13_weight", "w2_weight"] {
                let (from, to) = (
                    format!("model.llm.layers.0.mlp.experts.{bank}"),
                    format!("model.llm.layers.20.mlp.experts.{bank}"),
                );
                for suffix in ["", "_scale", "_amax", "_outer_scale_denominator"] {
                    let value = census[&format!("{from}{suffix}")].clone();
                    census.insert(format!("{to}{suffix}"), value);
                }
            }
        },
        "model.llm.layers.20.mlp.experts",
    );
    // A scale grid that is not one exponent per 32 inputs of a row.
    refuse(
        &|census| {
            census
                .get_mut("model.llm.layers.21.mlp.experts.w13_weight_scale")
                .unwrap()
                .1 = vec![128, 72, 72];
        },
        "MXFP8 stacked bank",
    );
}

/// The pinned quant declaration with one exact substitution.
fn quant_with(from: &str, to: &str) -> String {
    assert!(QUANT.contains(from), "{from} is not in the declaration");
    QUANT.replacen(from, to, 1)
}

#[test]
fn review_an_orphan_quantization_plane_stays_a_census_row_and_is_refused() {
    let pinned = pinned_census();
    let pinned_bytes: u64 = census_from_safetensors_headers(&headers_from(&pinned))
        .unwrap()
        .tensors
        .iter()
        .map(|row| row.entry.physical_bytes)
        .sum();
    // A plane-suffixed tensor beside a `*_weight` tensor that has no `_scale` is not a plane.
    for (orphan, shape) in [
        (
            "model.llm.layers.6.controllers.attn.down_weight_amax",
            vec![64],
        ),
        ("mtp.fuse_weight_outer_scale_denominator", vec![1]),
    ] {
        let mut census = pinned.clone();
        census.insert(orphan.into(), ("F32".into(), shape.clone()));
        let census = census_from_safetensors_headers(&headers_from(&census)).unwrap();
        assert!(
            census.tensors.iter().any(|row| row.entry.name == orphan),
            "{orphan} left the census"
        );
        let bytes: u64 = census
            .tensors
            .iter()
            .map(|row| row.entry.physical_bytes)
            .sum();
        assert_eq!(
            bytes,
            pinned_bytes + 4 * shape[0],
            "{orphan} bytes are not counted"
        );
        let error = bind(&census).expect_err("an orphan plane must be refused");
        assert!(error.contains(orphan), "{error}");
    }
}

#[test]
fn review_declared_weight_fields_are_typed_against_the_stored_layout() {
    // The shipped declaration attaches: NVFP4 is [1, 16], MXFP8 declares [32, 32].
    let mut config = bare_config();
    config.attach_quant_groups(QUANT).unwrap();
    let refuse = |text: String, needle: &str| {
        let mut config = bare_config();
        let error = config
            .attach_quant_groups(&text)
            .expect_err("the declaration must be refused");
        assert!(error.contains(needle), "{needle:?} not in {error}");
    };
    refuse(
        quant_with(
            r#""weight_transform": "nvfp4_nearest""#,
            r#""weight_transform": "bogus""#,
        ),
        "weight_transform",
    );
    refuse(
        quant_with(r#""output_dtype": "uint8""#, r#""output_dtype": "int64""#),
        "output_dtype",
    );
    refuse(
        quant_with(
            r#""weight_transform": "mxfp8_nearest""#,
            r#""weight_transform": "nvfp4_nearest""#,
        ),
        "weight_transform",
    );
    refuse(
        quant_with(
            r#""output_dtype": "float8_e4m3fn""#,
            r#""output_dtype": "bfloat16""#,
        ),
        "output_dtype",
    );
    // Block shapes: the compact text of the two quantized groups.
    let compact = |text: &str| text.split_whitespace().collect::<String>();
    let squeezed = compact(QUANT);
    for (from, to) in [
        (
            r#""weight_block_shape":[1,16]"#,
            r#""weight_block_shape":[7,3]"#,
        ),
        (
            r#""weight_block_shape":[32,32]"#,
            r#""weight_block_shape":[1,32]"#,
        ),
        (
            r#""weight_block_shape":[32,32]"#,
            r#""weight_block_shape":[128,128]"#,
        ),
    ] {
        assert_eq!(squeezed.matches(from).count(), 1, "{from}");
        refuse(squeezed.replacen(from, to, 1), "weight_block_shape");
    }
}

#[test]
fn review_the_loop_pass_count_is_bounded_without_overflow() {
    for passes in ["4294967295", "100000", "85"] {
        let text = config_with(r#""num_passes": 3"#, &format!(r#""num_passes": {passes}"#));
        let Err(error) = HfConfig::try_parse(&text) else {
            panic!("num_passes {passes} must be refused");
        };
        assert!(error.contains("cache slots"), "{error}");
    }
    // The cap is 1,024 slots: 84 passes are 24 + 83 * 12 = 1,020 slots and fit; 85 are 1,032.
    // At 84 the slot bound passes and the next rule speaks: the drafter taps no longer end at
    // the visit count.
    let text = config_with(r#""num_passes": 3"#, r#""num_passes": 84"#);
    let Err(error) = HfConfig::try_parse(&text) else {
        panic!("the pinned taps do not fit an 84-pass loop");
    };
    assert!(
        !error.contains("cache slots") && error.contains("does not end at 1020"),
        "{error}"
    );
}

#[test]
fn review_drafter_taps_end_at_the_final_hidden_and_its_biases_are_refused() {
    let refuse = |from: &str, to: &str, needle: &str| {
        let squeezed: String = CONFIG.split_whitespace().collect();
        assert_eq!(squeezed.matches(from).count(), 1, "{from}");
        let Err(error) = HfConfig::try_parse(&squeezed.replacen(from, to, 1)) else {
            panic!("{to} must be refused");
        };
        assert!(error.contains(needle), "{needle:?} not in {error}");
    };
    let taps = r#""target_layer_ids":[5,23,41,48]"#;
    refuse(
        taps,
        r#""target_layer_ids":[5,23,41,47]"#,
        "the final hidden",
    );
    refuse(taps, r#""target_layer_ids":[48]"#, "at least two");
    refuse(taps, r#""target_layer_ids":[23,5,48]"#, "increasing");
    refuse(
        r#""rms_norm_eps":1e-06,"rope_theta":1000000.0,"local_layer_ids":[0,2]"#,
        r#""rms_norm_eps":1e-06,"rope_theta":1000000.0,"local_layer_ids":[2,0]"#,
        "mtp_config.local_layer_ids",
    );
    refuse(
        r#""torch_dtype":"bfloat16","q_bias":false,"o_bias":false,"rms_norm_eps":1e-06,"rope_theta""#,
        r#""torch_dtype":"bfloat16","q_bias":true,"o_bias":false,"rms_norm_eps":1e-06,"rope_theta""#,
        "mtp_config.q_bias",
    );
    refuse(
        r#""block_size":8"#,
        r#""block_size":0"#,
        "mtp_config.block_size",
    );
}

#[test]
fn review_zero_sizes_and_unaligned_widths_are_refused_at_the_config() {
    let refuse = |pairs: &[(&str, &str)], needle: &str| {
        let mut text: String = CONFIG.split_whitespace().collect();
        for (from, to) in pairs {
            assert_eq!(text.matches(from).count(), 1, "{from}");
            text = text.replacen(from, to, 1);
        }
        let Err(error) = HfConfig::try_parse(&text) else {
            panic!("{pairs:?} must be refused");
        };
        assert!(error.contains(needle), "{needle:?} not in {error}");
    };
    refuse(
        &[
            (
                r#""num_attention_heads":24,"num_key_value_heads":8,"head_dim":128,"d_rel""#,
                r#""num_attention_heads":0,"num_key_value_heads":8,"head_dim":128,"d_rel""#,
            ),
            (
                r#""swa_num_attention_heads":24"#,
                r#""swa_num_attention_heads":0"#,
            ),
        ],
        "num_attention_heads is 0",
    );
    refuse(
        &[(r#""model_max_length":1048576"#, r#""model_max_length":0"#)],
        "model_max_length is 0",
    );
    // Packed and block-scaled widths: hidden and expert width in whole 32-input blocks.
    refuse(
        &[(
            r#""intermediate_size":1152,"mlp_gate_softcap""#,
            r#""intermediate_size":1150,"mlp_gate_softcap""#,
        )],
        "intermediate_size",
    );
    refuse(
        &[
            (
                r#""hidden_size":2304,"num_hidden_layers":24"#,
                r#""hidden_size":2310,"num_hidden_layers":24"#,
            ),
            (r#""text_hidden_size":2304"#, r#""text_hidden_size":2310"#),
            (
                r#""mtp_config":{"hidden_size":2304"#,
                r#""mtp_config":{"hidden_size":2310"#,
            ),
        ],
        "hidden_size 2310",
    );
}

#[test]
fn review_every_census_tensor_is_declared_and_every_declared_tensor_exists() {
    // A dense tensor dropped from its group: its class is still declared, the tensor is not.
    let mut config = bare_config();
    config
        .attach_quant_groups(&quant_with(r#""model.llm.norm.weight","#, ""))
        .unwrap();
    let error = PACK.compile_plan(&config).unwrap_err();
    assert!(
        format!("{error:?}").contains("model.llm.norm.weight"),
        "{error:?}"
    );
    // A vision and a drafter tensor are covered too.
    for dropped in [r#""model.visual.proj_in.weight","#, r#""mask_embed","#] {
        let mut config = bare_config();
        config
            .attach_quant_groups(&quant_with(dropped, ""))
            .unwrap();
        assert!(PACK.compile_plan(&config).is_err(), "{dropped}");
    }
    // A declared tensor the checkpoint does not have.
    let mut config = bare_config();
    config
        .attach_quant_groups(&quant_with(
            r#""model.llm.norm.weight","#,
            r#""model.llm.norm.weight", "model.llm.norm.bias","#,
        ))
        .unwrap();
    let error = PACK.compile_plan(&config).unwrap_err();
    assert!(
        format!("{error:?}").contains("model.llm.norm.bias"),
        "{error:?}"
    );
}

#[test]
fn only_the_declared_class_has_a_checkpoint_band() {
    // A fallback band would let an oracle of any class, the float-activation one included, write
    // a CheckpointParity pass for this pack. The declared class's band is the source program's
    // own spread between backends (docs/decisions/INKLING2-DECLARED-CHECKPOINT-PARITY.md).
    assert_eq!(PACK.checkpoint_parity, None);
    for class in [
        "source-weights-float32-accumulation",
        "inkling2-float-activation",
    ] {
        assert_eq!(PACK.checkpoint_parity_for(class), None, "{class}");
    }
    let gate = PACK.checkpoint_parity_for(DECLARED_PARITY_CLASS).unwrap();
    assert_eq!(
        (gate.max_abs, gate.max_rel, gate.require_argmax),
        (3.7, 0.0, true)
    );
    assert_eq!(PACK.checkpoint_probe_tokens().len(), 26);
}

#[test]
fn loop_controller_norms_take_the_config_rms_epsilon() {
    let text = TINY_CONFIG.replacen("\"rms_norm_eps\": 1e-06", "\"rms_norm_eps\": 1e-05", 1);
    assert_ne!(text, TINY_CONFIG);
    let mut config = ModelConfig::from_hf(&HfConfig::parse(&text));
    config.attach_quant_groups(TINY_QUANT_CONFIG).unwrap();
    let plan = PACK.compile_plan(&config).unwrap();
    assert_eq!(
        plan.inkling2.as_ref().unwrap().loop_plan.controller_epsilon,
        1e-5
    );
    // The depth mixes keep the program's constant.
    let ResidualTopology::BlockAttention(residual) = &plan.layers[3].residual else {
        panic!("block attention residual");
    };
    assert_eq!(residual.epsilon, 1e-6);
}
