use super::*;
use crate::config::HfConfig;
use crate::model_plan::OperationKind;
use crate::model_plan::{HyV4EngramMultiplierBound, HyV4EngramNormEpsilon};
use crate::modelopt_quant::LayerQuantMap;
use crate::safetensors::StInfo;
use crate::source::census_from_safetensors_headers;
use crate::tensor_contract::{StorageLayout, TensorCensusEntry};
use serde_json::Value;

/// `config.json` of Avifenesh/Hy4-flash-NVFP4 at f9eac06cf022cfd3dc2e8ed423922466b2adcd87,
/// byte-identical to the pinned revision (sha256 c88f950d8050be7402a9d969a28fa2af8d09fa46bce04920a3a14ae0dbf4b999).
const CONFIG: &str = include_str!("fixtures/config.json");
/// `hf_quant_config.json` of the same revision (sha256 9b84bf91465f41fbfbcbb4d9ce2ffca8ee6653af894cc792a11e98e80a7dad9f).
const HF_QUANT_CONFIG: &str = include_str!("fixtures/hf_quant_config.json");
/// Every header tensor of the same revision, grouped by name class
/// (research/modelplan-onboarding-hy4flash-20261119/census_classes.py, lossless by construction).
const CENSUS: &str = include_str!("fixtures/census-classes.tsv");

const INDEXED: [u32; 12] = [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41];

fn config() -> ModelConfig {
    ModelConfig::from_hf(&HfConfig::parse(CONFIG))
}

/// The fixture config with `edit` applied to its JSON tree, parsed (or refused) the way a
/// source reader parses it.
fn edited(edit: impl FnOnce(&mut Value)) -> Result<ModelConfig, String> {
    let mut json: Value = serde_json::from_str(CONFIG).unwrap();
    edit(&mut json);
    HfConfig::try_parse(&json.to_string()).map(|c| ModelConfig::from_hf(&c))
}

fn text(json: &mut Value) -> &mut serde_json::Map<String, Value> {
    json["text_config"].as_object_mut().unwrap()
}

#[test]
fn parses_the_pinned_config_and_derives_its_geometry() {
    assert_eq!(Arch::from_hf_model_type("hy_v4_flash"), Arch::HyV4Flash);
    assert!(Arch::HyV4Flash.is_hy_v4_flash());
    let config = config();
    assert_eq!(config.arch, Arch::HyV4Flash);
    assert!(config.dsv41.is_none() && config.dsv4.is_none());
    assert_eq!(config.n_layer, 42);
    assert_eq!(config.n_embd, 3072);
    assert_eq!(config.n_vocab, 120_832);
    let hy = config.hy_v4_flash.as_ref().unwrap();
    let indexed: Vec<u32> = (0..42)
        .filter(|il| hy.layer_kinds[*il as usize] == HyV4LayerKind::Indexed)
        .collect();
    assert_eq!(indexed, INDEXED);
    assert_eq!(hy.dense_layers.iter().filter(|d| **d).count(), 1);
    assert!(hy.dense_layers[0]);
    assert_eq!((hy.indexed.heads, hy.sliding.heads), (32, 64));
    assert_eq!(hy.indexed.rope_theta, 1e7);
    assert_eq!(hy.sliding.rope_theta, 1e4);
    assert_eq!(hy.sliding_window, 512);
    assert_eq!((hy.q_lora_rank, hy.kv_lora_rank), (1024, 512));
    assert_eq!(
        (
            hy.index_n_heads,
            hy.index_head_dim,
            hy.index_topk,
            hy.index_pool
        ),
        (16, 128, 2048, 4)
    );
    assert_eq!((hy.n_routed_experts, hy.num_experts_per_tok), (512, 16));
    assert_eq!(
        (hy.routed_expert_hidden_size, hy.moe_intermediate_size),
        (1536, 1536)
    );
    assert_eq!(hy.routed_scaling_factor, 3.998);
    assert_eq!((hy.hc_mult, hy.hc_magnitude), (4, 2.0));
    assert_eq!(hy.dfly.target_kv_layers, [25, 29, 33, 37, 41]);
    assert_eq!(hy.vision.num_hidden_layers, 27);

    // Engram: 1-based ids, modules in the 0-based layers id - 1, seeds and sizes by the id.
    assert_eq!(hy.engram_layer_ids, [2, 12, 22, 32]);
    let slots: Vec<(u32, usize)> = (0..42)
        .filter_map(|il| hy.engram_slot(il).map(|slot| (il, slot)))
        .collect();
    assert_eq!(slots, [(1, 0), (11, 1), (21, 2), (31, 3)]);
    assert_eq!(
        (0..4)
            .map(|slot| hy.engram_table_rows(slot))
            .collect::<Vec<_>>(),
        [164_994_380, 164_999_008, 165_004_086, 165_008_942]
    );
    assert_eq!(
        (0..4)
            .map(|slot| hy.engram_hash_seed(slot))
            .collect::<Vec<_>>(),
        [20_014, 120_084, 220_154, 320_224]
    );
    assert_eq!((hy.engram_heads(), hy.engram_head_dim()), (16, 96));
    assert_eq!(hy.engram_lookup_table_size, 120_818);
}

/// The two copies of the per-layer map, `config.json quantization_config` and
/// `hf_quant_config.json`, read to the same map: 141 W4A16_NVFP4 and 138 FP8_PB_WO modules.
#[test]
fn both_modelopt_maps_agree_and_name_the_two_weight_only_classes() {
    let from_file = LayerQuantMap::from_hf_quant_config(HF_QUANT_CONFIG).unwrap();
    let config = config();
    let hy = config.hy_v4_flash.as_ref().unwrap();
    assert_eq!(hy.quant, from_file);
    let count = |algo| {
        from_file
            .layers
            .values()
            .filter(|q| {
                q.algo == algo
                    && q.group_size
                        == if algo == ModeloptAlgo::W4a16Nvfp4 {
                            16
                        } else {
                            128
                        }
            })
            .count()
    };
    assert_eq!(count(ModeloptAlgo::W4a16Nvfp4), 141);
    assert_eq!(count(ModeloptAlgo::Fp8PbWo), 138);
    assert_eq!(from_file.layers.len(), 279);
    assert!(from_file.is_excluded("model.layers.0.self_attn.o_proj"));
    assert!(from_file.is_excluded("model.layers.9.mlp.shared_experts.gate_proj"));
    // The engram tables are in neither list.
    let table = "model.layers.1.engram_layer.multi_head_embedding.embedding";
    assert!(from_file.get(table).is_none() && !from_file.is_excluded(table));
}

#[test]
fn plan_is_typed_gated_mla_with_pooled_index_latent_moe_and_engram() {
    let config = config();
    assert!(matches!(
        ModelPlan::compile(&config),
        Err(PlanCompileError::UnsupportedSemantics {
            field: "hy_v4_flash",
            ..
        })
    ));
    let pack = for_config(&config).unwrap();
    assert_eq!(pack.family, "hy_v4_flash");
    assert_eq!(by_alias("hy4-flash").unwrap().family, "hy_v4_flash");
    let plan = pack.compile_plan(&config).unwrap();
    assert_eq!(plan.layers.len(), 42);
    assert_eq!(plan.context_length, 1_048_576);
    assert!(plan.drafter.is_none() && plan.vision.is_none());
    assert_eq!(
        plan.hy_v4_flash,
        Some(HyV4FlashNumerics {
            streams: 4,
            exit_epsilon: 1e-6,
            output_projection_f32: true
        })
    );
    for layer in &plan.layers {
        let il = layer.index;
        let AttentionPlan::HyV4(attention) = layer.attention else {
            panic!("layer {il} is not Hy4 attention");
        };
        let indexed = INDEXED.contains(&il);
        assert_eq!(
            attention.query_heads,
            if indexed { 32 } else { 64 },
            "layer {il}"
        );
        assert_eq!(
            (
                attention.qk_nope_head_dim,
                attention.qk_rope_head_dim,
                attention.value_head_dim
            ),
            (192, 64, 256)
        );
        match attention.kind {
            HyV4AttentionKind::Indexed(index) => {
                assert!(indexed, "layer {il}");
                assert_eq!(
                    index,
                    IndexPoolPlan {
                        heads: 16,
                        head_dim: 128,
                        rope_head_dim: 64,
                        pool: 4,
                        top_k: 2048,
                        ties: GroupTieOrder::LowerIndexFirst,
                    }
                );
                assert_eq!(attention.rope.base, 1e7);
            }
            HyV4AttentionKind::Sliding { window } => {
                assert!(!indexed, "layer {il}");
                assert_eq!(window, 512);
                assert_eq!(attention.rope.base, 1e4);
            }
        }
        assert_eq!(
            attention.projection_weights,
            if il == 0 {
                HyV4WeightFormat::Bf16
            } else {
                HyV4WeightFormat::Fp8Block { block: 128 }
            }
        );
        match &layer.mlp {
            MlpPlan::Dense(dense) => {
                assert_eq!(il, 0);
                assert_eq!(dense.intermediate_size, 13_824);
            }
            MlpPlan::LatentMoe(moe) => {
                assert_ne!(il, 0);
                assert_eq!(moe.latent_size, 1536);
                assert_eq!(
                    moe.expert_weights,
                    HyV4WeightFormat::Nvfp4W4A16 { group: 16 }
                );
                assert_eq!(
                    moe.moe.activation,
                    ActivationPlan::SwiGluPreClamped { limit: 10.0 }
                );
                // The shared expert is the unclamped HYV4FlashMLP.
                assert_eq!(moe.shared_activation, ActivationPlan::Silu);
                assert_eq!(moe.moe.shared.as_ref().unwrap().intermediate_size, 1536);
            }
            other => panic!("layer {il}: {other:?}"),
        }
        let ResidualTopology::HyV4(residual) = layer.residual else {
            panic!("layer {il} residual");
        };
        assert_eq!(residual.streams, 4);
        assert_eq!(residual.post_magnitude, 2.0);
        let engram = residual
            .engram
            .map(|e| (e.layer_id, e.hash_seed, e.table_rows));
        let expected = match il {
            1 => Some((2, 20_014, 164_994_380)),
            11 => Some((12, 120_084, 164_999_008)),
            21 => Some((22, 220_154, 165_004_086)),
            31 => Some((32, 320_224, 165_008_942)),
            _ => None,
        };
        assert_eq!(engram, expected, "layer {il}");
        if let Some(engram) = residual.engram {
            assert_eq!(
                engram.multiplier_bound,
                HyV4EngramMultiplierBound::LookupTableMaxPlusOne
            );
            assert_eq!(engram.norm_epsilon, HyV4EngramNormEpsilon::InputDtype);
            assert_eq!(
                (engram.heads, engram.head_dim, engram.max_ngram),
                (16, 96, 3)
            );
            assert_eq!(engram.lookup_table_size, 120_818);
        }
        let StatePlan::HyV4(state) = layer.state else {
            panic!("layer {il} state")
        };
        assert_eq!((state.latent_width, state.rope_width), (512, 64));
        assert_eq!(state.window, (!indexed).then_some(512));
        assert_eq!(state.index_key_width, indexed.then_some(128));
    }
    let operations = plan.operations();
    for operation in [
        OperationKind::HyV4GatedMlaAttention,
        OperationKind::IndexPool,
        OperationKind::HyV4State,
        OperationKind::LatentMoeMlp,
        OperationKind::IndependentHyperConnections,
        OperationKind::HyV4Engram,
        OperationKind::Fp8BlockWeightOnlyLinear,
        OperationKind::Nvfp4W4A16Linear,
        OperationKind::HyV4StreamExit,
        OperationKind::SigmoidRouter,
        OperationKind::SharedMlp,
    ] {
        assert!(operations.contains(&operation), "{operation:?}");
    }
    assert_eq!(
        operations
            .iter()
            .filter(|op| **op == OperationKind::IndexPool)
            .count(),
        12
    );
    assert_eq!(
        operations
            .iter()
            .filter(|op| **op == OperationKind::HyV4Engram)
            .count(),
        4
    );
    // Nothing executes it: every Hy4 operation is a registry row with no surface.
    for operation in [
        OperationKind::HyV4GatedMlaAttention,
        OperationKind::IndexPool,
        OperationKind::HyV4Engram,
        OperationKind::IndependentHyperConnections,
        OperationKind::LatentMoeMlp,
    ] {
        assert!(
            crate::op_registry::surfaces(operation).is_none(),
            "{operation:?}"
        );
    }
}

/// `{name: (dtype, shape)}` of every header tensor.
type HeaderCensus = BTreeMap<String, (String, Vec<u64>)>;

/// Expand the class fixture into the header census plus its recorded tensor and byte totals.
fn pinned_census() -> (HeaderCensus, usize, u64) {
    let mut names = BTreeMap::new();
    let (mut tensors, mut bytes) = (0, 0);
    for line in CENSUS.lines() {
        if let Some(rest) = line.strip_prefix("# tensors\t") {
            tensors = rest.parse().unwrap();
            continue;
        }
        if let Some(rest) = line.strip_prefix("# bytes\t") {
            bytes = rest.parse().unwrap();
            continue;
        }
        if line.starts_with('#') || line.starts_with("pattern\t") {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        let [pattern, dtype, shape, indices] = cols[..] else {
            panic!("malformed census fixture line {line:?}");
        };
        let shape: Vec<u64> = if shape == "-" {
            Vec::new()
        } else {
            shape.split(',').map(|d| d.parse().unwrap()).collect()
        };
        let layers: Vec<Option<u32>> = if indices == "-" {
            vec![None]
        } else {
            indices
                .split(',')
                .flat_map(|part| {
                    let (a, b) = part.split_once('-').unwrap_or((part, part));
                    a.parse::<u32>().unwrap()..=b.parse::<u32>().unwrap()
                })
                .map(Some)
                .collect()
        };
        for layer in layers {
            let name = match layer {
                Some(layer) => pattern.replace("{L}", &layer.to_string()),
                None => pattern.to_string(),
            };
            assert!(
                names
                    .insert(name.clone(), (dtype.to_string(), shape.clone()))
                    .is_none(),
                "duplicate {name}"
            );
        }
    }
    (names, tensors, bytes)
}

fn dtype_bytes(dtype: &str) -> u64 {
    match dtype {
        "BF16" => 2,
        "F32" | "I32" => 4,
        "F8_E4M3" | "U8" => 1,
        other => panic!("unexpected dtype {other}"),
    }
}

/// Safetensors headers with exact byte extents for a census.
fn headers(artifact: &HeaderCensus) -> (BTreeMap<String, StInfo>, u64) {
    let mut offset = 0usize;
    let headers = artifact
        .iter()
        .map(|(name, (dtype, shape))| {
            let length = (shape.iter().product::<u64>() * dtype_bytes(dtype)) as usize;
            let info = StInfo {
                dtype: dtype.clone(),
                shape: shape.clone(),
                data_offsets: [offset, offset + length],
            };
            offset += length;
            (name.clone(), info)
        })
        .collect();
    (headers, offset as u64)
}

fn entries(artifact: &HeaderCensus) -> Vec<TensorCensusEntry> {
    let (headers, _) = headers(artifact);
    census_from_safetensors_headers(&headers)
        .unwrap()
        .tensors
        .into_iter()
        .map(|row| row.entry)
        .collect()
}

fn contract_for(config: &ModelConfig) -> TensorContract {
    let plan = PACK.compile_plan(config).unwrap();
    PACK.compile_tensor_contract(
        config,
        &plan,
        CheckpointDialect::HfSafetensors,
        ContractOptions::default(),
    )
    .unwrap()
}

/// Census gate: the config-derived schema reproduces all 2,429 header tensors of the pinned
/// artifact (names, dtypes, shapes), and the generated contract binds the census the header
/// reader builds from them with nothing missing, extra, ambiguous or reshaped.
#[test]
fn expected_census_reproduces_the_pinned_artifact_and_the_contract_binds_it() {
    let (artifact, tensors, bytes) = pinned_census();
    assert_eq!(artifact.len(), 2_429);
    assert_eq!(tensors, 2_429);
    let config = config();
    let expected = expected_census(config.hy_v4_flash.as_ref().unwrap()).unwrap();
    let missing: Vec<_> = expected
        .keys()
        .filter(|n| !artifact.contains_key(*n))
        .collect();
    let extra: Vec<_> = artifact
        .keys()
        .filter(|n| !expected.contains_key(*n))
        .collect();
    assert!(
        missing.is_empty(),
        "derived, not in the artifact: {missing:?}"
    );
    assert!(extra.is_empty(), "in the artifact, not derived: {extra:?}");
    for (name, spec) in &expected {
        let (dtype, shape) = &artifact[name];
        assert_eq!(spec.dtype, dtype, "{name} dtype");
        assert_eq!(&spec.shape, shape, "{name} shape");
    }

    let (headers, total) = headers(&artifact);
    assert_eq!(total, bytes, "fixture byte total");
    let census = census_from_safetensors_headers(&headers).unwrap();
    let entries: Vec<_> = census.tensors.iter().map(|row| row.entry.clone()).collect();
    assert_eq!(entries.iter().map(|e| e.physical_bytes).sum::<u64>(), bytes);
    for entry in &entries {
        if let StorageLayout::Quantized(layout) = &entry.storage
            && layout.format == "FP8_E4M3"
        {
            let expect: &[u32] = if entry.name.contains(".engram_layer.") {
                &[1, 96]
            } else {
                &[128, 128]
            };
            assert_eq!(layout.block_shape, expect, "{}", entry.name);
        }
    }

    let plan = PACK.compile_plan(&config).unwrap();
    let contract = contract_for(&config);
    let ids: BTreeSet<_> = contract.requirements.iter().map(|r| r.id.clone()).collect();
    assert_eq!(
        ids.len(),
        contract.requirements.len(),
        "semantic ids are unique"
    );
    let bound = contract.bind(&entries).unwrap();
    assert_eq!(bound.tensors.len(), entries.len());
    assert_eq!(bound.tensors.len(), contract.requirements.len());
    // The loader boundary (head ownership included) accepts it too.
    let source_census = crate::source::TensorCensus {
        dialect: CheckpointDialect::HfSafetensors,
        tensors: census.tensors.clone(),
    };
    crate::checkpoint_binding::bind_census(Some(&PACK), &config, &plan, &source_census).unwrap();

    let requirement = |name: &str| {
        contract
            .requirements
            .iter()
            .find(|r| r.names == [name])
            .unwrap_or_else(|| panic!("no requirement for {name}"))
    };
    let gate = requirement("model.layers.7.mlp.experts.gate_proj.weight");
    assert_eq!(gate.quant, QuantConstraint::Nvfp4);
    assert_eq!(gate.shape, [512, 1536, 1536]);
    assert_eq!(
        gate.auxiliaries.as_deref(),
        Some(
            [
                "model.layers.7.mlp.experts.gate_proj.weight_scale".to_string(),
                "model.layers.7.mlp.experts.gate_proj.weight_scale_2".to_string(),
            ]
            .as_slice()
        )
    );
    assert_eq!(
        gate.id,
        TensorId::Layer {
            index: 7,
            tensor: LayerTensor::MoeExpertGateBank
        }
    );
    let q_b = requirement("model.layers.5.self_attn.q_b_proj.weight");
    assert_eq!(q_b.quant, QuantConstraint::Fp8Block128);
    assert_eq!(q_b.shape, [8192, 1024]);
    assert_eq!(
        requirement("model.layers.0.self_attn.q_b_proj.weight").quant,
        QuantConstraint::ExactFloat(FloatType::Bf16)
    );
    let table = requirement("model.layers.21.engram_layer.multi_head_embedding.embedding.weight");
    assert_eq!(table.quant, QuantConstraint::Fp8RowF32);
    assert_eq!(table.shape, [165_004_086, 96]);
    assert_eq!(
        requirement("model.layers.31.engram_layer.lookup_table").quant,
        QuantConstraint::I32
    );
    assert_eq!(requirement("lm_head.weight").id, TensorId::OutputProjection);
    assert_eq!(
        requirement("model.dfly_layers.3.mlp.up_proj.weight").owner,
        TensorOwner::Mtp(3)
    );
    assert_eq!(
        requirement("vit.layers.26.mlp.dense_4h_to_h.weight").owner,
        TensorOwner::Vision(Some(26))
    );
    assert_eq!(
        requirement("vit.perceive.image_newline").owner,
        TensorOwner::Vision(None)
    );

    // A missing or extra tensor fails the census.
    let mut short = entries.clone();
    short.retain(|entry| {
        entry.name != "model.layers.9.self_attn.compressed_indexer.intra_group_ape"
    });
    assert!(matches!(
        contract.bind(&short),
        Err(TensorContractError::Missing { .. })
    ));
    let mut long = entries.clone();
    let mut stray = long
        .iter()
        .find(|entry| entry.name == "model.layers.9.self_attn.compressed_indexer.intra_group_ape")
        .unwrap()
        .clone();
    stray.name = "model.layers.10.self_attn.compressed_indexer.intra_group_ape".into();
    long.push(stray);
    assert!(matches!(
        contract.bind(&long),
        Err(TensorContractError::Extra { .. })
    ));
}

/// Gate and up are separate banks with separate per-expert second-level scales. A checkpoint
/// without up's own `weight_scale_2`, or with one fused `gate_up_proj` bank, fails the census;
/// a map that quantizes a fused bank fails the config.
#[test]
fn gate_and_up_keep_separate_scales() {
    let (artifact, _, _) = pinned_census();
    let config = config();
    let contract = contract_for(&config);

    // No up_proj.weight_scale_2: up would borrow gate's scale.
    let mut shared = artifact.clone();
    shared.remove("model.layers.4.mlp.experts.up_proj.weight_scale_2");
    match contract.bind(&entries(&shared)) {
        Err(TensorContractError::AuxiliaryLayoutMismatch { name, .. }) => {
            assert_eq!(name, "model.layers.4.mlp.experts.up_proj.weight")
        }
        other => panic!("{other:?}"),
    }

    // One fused gate_up bank, as the BF16 source stores it.
    let mut fused = artifact.clone();
    for suffix in ["weight", "weight_scale", "weight_scale_2"] {
        let (dtype, mut shape) = fused
            .remove(&format!("model.layers.4.mlp.experts.gate_proj.{suffix}"))
            .unwrap();
        fused
            .remove(&format!("model.layers.4.mlp.experts.up_proj.{suffix}"))
            .unwrap();
        if shape.len() == 3 {
            shape[1] *= 2;
        }
        fused.insert(
            format!("model.layers.4.mlp.experts.gate_up_proj.{suffix}"),
            (dtype, shape),
        );
    }
    assert!(matches!(
        contract.bind(&entries(&fused)),
        Err(TensorContractError::Missing { .. })
    ));

    // A map that adds a fused bank beside gate and up names a module the program does not have.
    let config = edited(|json| {
        let layers = json["quantization_config"]["quantized_layers"]
            .as_object_mut()
            .unwrap();
        let gate = layers["model.layers.4.mlp.experts.gate_proj"].clone();
        layers.insert("model.layers.4.mlp.experts.gate_up_proj".into(), gate);
    })
    .unwrap();
    match PACK.compile_plan(&config) {
        Err(PlanCompileError::UnsupportedSemantics { value, .. }) => assert!(
            value.contains("names model.layers.4.mlp.experts.gate_up_proj, which is not"),
            "{value}"
        ),
        other => panic!("{other:?}"),
    }

    // A map that replaces gate and up by the fused bank leaves gate_proj unlisted.
    let error = edited(|json| {
        let layers = json["quantization_config"]["quantized_layers"]
            .as_object_mut()
            .unwrap();
        layers.remove("model.layers.4.mlp.experts.up_proj");
        let gate = layers
            .remove("model.layers.4.mlp.experts.gate_proj")
            .unwrap();
        layers.insert("model.layers.4.mlp.experts.gate_up_proj".into(), gate);
    })
    .map(|config| PACK.compile_plan(&config).map(|_| ()))
    .unwrap();
    match error {
        Err(PlanCompileError::UnsupportedSemantics { value, .. }) => {
            assert!(
                value.starts_with("model.layers.4.mlp.experts.gate_proj is neither"),
                "{value}"
            )
        }
        other => panic!("{other:?}"),
    }
}

/// The engram index offset: `engram_layer_ids` is 1-based. Read as 0-based it would put the
/// memory one layer early and seed its hash with the wrong id; both fail closed.
#[test]
fn the_engram_layer_id_is_one_based() {
    // Ids shifted to the tensor layers while the size table keeps the true ids: refused by name.
    let error = edited(|json| {
        text(json).insert(
            "engram_layer_ids".into(),
            serde_json::json!([1, 11, 21, 31]),
        );
    })
    .unwrap_err();
    assert!(error.contains("engram_layer_head_vocab_sizes.1"), "{error}");

    // Ids and size keys both shifted by `delta`: the memory moves one layer.
    let shifted = |delta: i64| {
        edited(|json| {
            let t = text(json);
            let ids: Vec<i64> = [2, 12, 22, 32].iter().map(|id| id + delta).collect();
            t.insert("engram_layer_ids".into(), serde_json::json!(ids));
            let sizes = t["engram_layer_head_vocab_sizes"]
                .as_object()
                .unwrap()
                .clone();
            let moved: serde_json::Map<String, Value> = sizes
                .into_iter()
                .map(|(k, v)| ((k.parse::<i64>().unwrap() + delta).to_string(), v))
                .collect();
            t.insert("engram_layer_head_vocab_sizes".into(), Value::Object(moved));
        })
        .unwrap()
    };
    // Read 0-based ([1, 11, 21, 31]): the memory lands on layers 0/10/20/30 and seeds with id 1.
    // With the artifact's map, layer 0's `model.layers.0.*` exclusion also covers the table,
    // so the plan refuses it.
    let early = shifted(-1);
    let hy = early.hy_v4_flash.as_ref().unwrap();
    assert_eq!(hy.engram_slot(0), Some(0));
    assert_eq!(
        hy.engram_hash_seed(0),
        10_007,
        "the 0-based reading seeds with id 1"
    );
    match PACK.compile_plan(&early) {
        Err(PlanCompileError::UnsupportedSemantics { value, .. }) => {
            assert!(value.starts_with("model.layers.0.engram_layer."), "{value}")
        }
        other => panic!("{other:?}"),
    }
    // Without that wildcard (layer 0's attention excluded by name) the plan compiles, and the
    // real headers refuse the table on layer 0.
    let narrowed = |json: &mut Value| {
        let ignore = json["quantization_config"]["ignore"]
            .as_array_mut()
            .unwrap();
        let at = ignore.iter().position(|p| p == "model.layers.0.*").unwrap();
        ignore[at] = Value::from("model.layers.0.self_attn.*");
    };
    let early = edited(|json| {
        narrowed(json);
        let t = text(json);
        t.insert(
            "engram_layer_ids".into(),
            serde_json::json!([1, 11, 21, 31]),
        );
        let sizes = t["engram_layer_head_vocab_sizes"]
            .as_object()
            .unwrap()
            .clone();
        let moved: serde_json::Map<String, Value> = sizes
            .into_iter()
            .map(|(k, v)| ((k.parse::<i64>().unwrap() - 1).to_string(), v))
            .collect();
        t.insert("engram_layer_head_vocab_sizes".into(), Value::Object(moved));
    })
    .unwrap();
    let (artifact, _, _) = pinned_census();
    match contract_for(&early).bind(&entries(&artifact)) {
        Err(TensorContractError::Missing { accepted_names, .. }) => assert!(
            accepted_names[0].starts_with("model.layers.0.engram_layer."),
            "{accepted_names:?}"
        ),
        other => panic!("{other:?}"),
    }
    // The narrowed map alone (true ids) still binds: the refusal is the index, not the map.
    let control = edited(narrowed).unwrap();
    contract_for(&control).bind(&entries(&artifact)).unwrap();
    // One layer late ([3, 13, 23, 33]): the plan compiles and the real headers refuse it.
    let late = shifted(1);
    let plan = PACK.compile_plan(&late).unwrap();
    let ResidualTopology::HyV4(residual) = plan.layers[2].residual else {
        panic!()
    };
    assert_eq!(residual.engram.map(|e| e.layer_id), Some(3));
    let (artifact, _, _) = pinned_census();
    match contract_for(&late).bind(&entries(&artifact)) {
        Err(TensorContractError::Missing { accepted_names, .. }) => {
            assert!(
                accepted_names[0].starts_with("model.layers.12.engram_layer."),
                "{accepted_names:?}"
            )
        }
        other => panic!("{other:?}"),
    }
    // The true config seeds layer 1's memory with id 2, not 1.
    let config = config();
    let hy = config.hy_v4_flash.as_ref().unwrap();
    assert_eq!(hy.engram_slot(1), Some(0));
    assert_eq!(hy.engram_slot(0), None);
    assert_eq!(hy.engram_hash_seed(0), 20_014);
}

/// Indexed layers have 32 heads and sliding layers 64; the head count is per layer type, not
/// one model-wide value. A config that gives both types one count fails the census on the real
/// headers, and layer_types and swa_pattern must agree.
#[test]
fn the_two_layer_types_keep_their_own_head_counts() {
    let (artifact, _, _) = pinned_census();
    let census = entries(&artifact);
    for (key, heads) in [("swa_num_attention_heads", 32), ("num_attention_heads", 64)] {
        let config = edited(|json| {
            let t = text(json);
            t.insert(key.into(), serde_json::json!(heads));
            if key == "num_attention_heads" {
                t.insert("num_key_value_heads".into(), serde_json::json!(heads));
            }
        })
        .unwrap();
        match contract_for(&config).bind(&census) {
            Err(TensorContractError::ShapeMismatch { name, .. }) => {
                assert!(name.contains(".self_attn."), "{key}: {name}")
            }
            other => panic!("{key}: {other:?}"),
        }
    }
    let error = edited(|json| {
        let t = text(json);
        let mut pattern = t["swa_pattern"].clone();
        pattern[2] = serde_json::json!(0);
        t.insert("swa_pattern".into(), pattern);
    })
    .unwrap_err();
    assert!(error.contains("swa_pattern[2]"), "{error}");
}

/// Every program switch the pack does not compile refuses by name.
#[test]
fn switches_of_another_program_refuse_by_name() {
    for (key, value) in [
        ("enable_ihc", serde_json::json!(false)),
        ("moe_router_use_sigmoid", serde_json::json!(false)),
        (
            "compressed_indexer_skip_fine_select",
            serde_json::json!(false),
        ),
        ("gating_type", serde_json::json!("headwise")),
        ("learnable_sink", serde_json::json!(false)),
        ("o_lora_rank", serde_json::json!(512)),
        ("engram_isolate_fffd_namespace", serde_json::json!(true)),
        ("compressed_indexer_group_topk", serde_json::json!(256)),
        ("window_size", serde_json::json!([512, 0])),
    ] {
        let error = edited(|json| {
            text(json).insert(key.into(), value.clone());
        })
        .unwrap_err();
        assert!(error.contains(key), "{key}: {error}");
    }
    // A zero budget passes `index_topk % pool == 0` and every alias, but holds no group: the
    // indexer could not recall even the query's own (transformers' `topk(-1)` raises).
    let error = edited(|json| {
        for key in [
            "index_topk",
            "compressed_indexer_candidate_topk",
            "compressed_indexer_group_topk",
        ] {
            text(json).insert(key.into(), serde_json::json!(0));
        }
    })
    .unwrap_err();
    assert!(
        error.contains("compressed_indexer_compress_ratio"),
        "{error}"
    );
    let error = edited(|json| {
        text(json).remove("hc_magnitude");
    })
    .unwrap_err();
    assert!(
        error.contains("missing required field hc_magnitude"),
        "{error}"
    );
    let error = edited(|json| {
        json.as_object_mut().unwrap().remove("quantization_config");
    })
    .unwrap_err();
    assert!(error.contains("quantization_config"), "{error}");
    // A quantizable module the map neither quantizes nor excludes.
    let config = edited(|json| {
        json["quantization_config"]["quantized_layers"]
            .as_object_mut()
            .unwrap()
            .remove("model.layers.6.self_attn.o_proj");
    })
    .unwrap();
    match PACK.compile_plan(&config) {
        Err(PlanCompileError::UnsupportedSemantics { value, .. }) => {
            assert!(value.contains("model.layers.6.self_attn.o_proj"), "{value}")
        }
        other => panic!("{other:?}"),
    }
}

/// The map's FP8 blocks are 128x128 and layer 0's dense MLP is BF16; a map that says otherwise
/// refuses the config rather than compiling a plan that reads another format.
#[test]
fn the_quant_map_cannot_move_a_module_to_another_format() {
    let config = edited(|json| {
        json["quantization_config"]["quantized_layers"]["model.layers.5.self_attn.o_proj"]
            ["group_size"] = Value::from(64);
    })
    .unwrap();
    match PACK.compile_plan(&config) {
        Err(PlanCompileError::UnsupportedSemantics { value, .. }) => {
            assert!(value.contains("FP8_PB_WO group_size 64"), "{value}")
        }
        other => panic!("{other:?}"),
    }
    let config = edited(|json| {
        let ignore = json["quantization_config"]["ignore"]
            .as_array_mut()
            .unwrap();
        let at = ignore.iter().position(|p| p == "model.layers.0.*").unwrap();
        ignore[at] = Value::from("model.layers.0.self_attn.*");
        json["quantization_config"]["quantized_layers"]
            .as_object_mut()
            .unwrap()
            .insert(
                "model.layers.0.mlp.gate_proj".into(),
                serde_json::json!({"quant_algo": "W4A16_NVFP4", "group_size": 16}),
            );
    })
    .unwrap();
    match PACK.compile_plan(&config) {
        Err(PlanCompileError::UnsupportedSemantics { value, .. }) => assert!(
            value.contains("model.layers.0.mlp.gate_proj is W4A16_NVFP4"),
            "{value}"
        ),
        other => panic!("{other:?}"),
    }
}

/// Source readers check the artifact's `hf_quant_config.json` against the config's map: it
/// must be present, strictly valid JSON and the same map.
#[test]
fn hf_quant_config_must_match_the_config_map() {
    let config = config();
    let map = config.per_layer_quant_map().unwrap();
    map.check_hf_quant_config(Some(HF_QUANT_CONFIG)).unwrap();
    assert!(
        map.check_hf_quant_config(None)
            .unwrap_err()
            .contains("absent")
    );
    let mut file: Value = serde_json::from_str(HF_QUANT_CONFIG).unwrap();
    file["quantization"]["quantized_layers"]["model.layers.1.self_attn.o_proj"] =
        serde_json::json!({"quant_algo": "W4A16_NVFP4", "group_size": 16});
    let regrouped = file.to_string();
    let error = map.check_hf_quant_config(Some(&regrouped)).unwrap_err();
    assert!(error.contains("model.layers.1.self_attn.o_proj"), "{error}");
    let duplicated = HF_QUANT_CONFIG.replacen(
        r#""quant_algo": "MIXED_PRECISION","#,
        r#""quant_algo": "MIXED_PRECISION", "quant_algo": "MIXED_PRECISION","#,
        1,
    );
    assert_ne!(duplicated, HF_QUANT_CONFIG, "fixture spelling");
    assert!(map.check_hf_quant_config(Some(&duplicated)).is_err());
}

/// `o_lora_rank: null` means "no output LoRA" only for hy_v4_flash; another family that reads
/// the key keeps refusing a null as a parse error.
#[test]
fn a_null_low_rank_is_read_as_absent_only_here() {
    let mut json: Value = serde_json::from_str(CONFIG).unwrap();
    assert!(json["text_config"]["o_lora_rank"].is_null());
    assert!(json["text_config"]["gate_lora_rank"].is_null());
    config();
    text(&mut json).insert("o_lora_rank".into(), Value::from(64));
    let error = HfConfig::try_parse(&json.to_string()).unwrap_err();
    assert!(error.contains("o_lora_rank"), "{error}");
    let mut v41: Value =
        serde_json::from_str(include_str!("../deepseek_v41/fixtures/config.json")).unwrap();
    v41["o_lora_rank"] = Value::Null;
    let error = HfConfig::try_parse(&v41.to_string()).unwrap_err();
    assert!(error.contains("o_lora_rank must be a non-null"), "{error}");
}
