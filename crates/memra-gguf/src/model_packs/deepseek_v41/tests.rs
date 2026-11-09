use super::*;
use crate::config::HfConfig;
use crate::model_plan::OperationKind;
use crate::safetensors::StInfo;
use crate::source::census_from_safetensors_headers;
use crate::tensor_contract::StorageLayout;

/// `config.json` of deepseek-ai/DeepSeek-V4.1-Flash-1017 at 1bae0e1088993b9b82de08f59de5be5a34e42170,
/// byte-identical to the pinned revision (sha256 8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879).
const CONFIG_1017: &str = include_str!("fixtures/config.json");
/// Every header tensor of the same revision, grouped by name class
/// (research/modelplan-onboarding-dsv41-20261109/census_classes.py, lossless by construction).
const CENSUS_1017: &str = include_str!("fixtures/census-classes.tsv");

fn config_1017() -> ModelConfig {
    ModelConfig::from_hf(&HfConfig::parse(CONFIG_1017))
}

#[test]
fn parses_the_pinned_1017_config_and_derives_its_geometry() {
    assert_eq!(Arch::from_hf_model_type("deepseek_v41"), Arch::DeepSeekV41);
    assert_eq!(
        Arch::from_hf_model_type("deepseek_v41_text"),
        Arch::DeepSeekV41
    );
    assert!(!Arch::DeepSeekV41.is_dsv4());
    let config = config_1017();
    assert_eq!(config.arch, Arch::DeepSeekV41);
    assert!(config.dsv4.is_none(), "V4.1 never fills the DSv4 config");
    assert!(config.vision.is_none(), "no generic tower is fabricated");
    assert!(config.multimodal.is_none());
    assert_eq!(config.n_layer, 43);
    assert_eq!(config.nextn_predict_layers, 3);
    assert_eq!(config.n_embd, 5120);
    assert_eq!(config.n_head, 64);
    assert_eq!(config.n_head_kv, 1);
    assert_eq!(config.head_dim_k, 512);
    assert_eq!(config.n_vocab, 129_280);
    assert_eq!(config.context_length, 1_048_576);
    assert_eq!(config.rms_eps.to_bits(), 1e-20f32.to_bits());
    assert_eq!(config.tie_word_embeddings, Some(false));
    let moe = config.moe.as_ref().unwrap();
    assert_eq!(
        (
            moe.expert_count,
            moe.expert_used_count,
            moe.expert_ff_length
        ),
        (384, 6, 2304)
    );

    let v41 = config.dsv41.as_ref().unwrap();
    assert_eq!(v41.trunk_layers(), 40);
    assert_eq!(
        (v41.q_lora_rank, v41.head_dim, v41.qk_rope_head_dim),
        (1280, 512, 64)
    );
    assert_eq!(
        (v41.o_lora_rank, v41.o_groups, v41.sliding_window),
        (1024, 8, 128)
    );
    assert_eq!(
        (v41.index_n_heads, v41.index_head_dim, v41.index_topk),
        (32, 128, 512)
    );
    assert_eq!(
        (
            v41.candidate_source_layer_id,
            v41.candidate_topk_blocks,
            v41.candidate_block_size
        ),
        (20, 2048, 8)
    );
    assert_eq!((v41.hc_mult, v41.hc_sinkhorn_iters), (4, 20));
    assert_eq!(v41.hc_eps.to_bits(), 1e-6f32.to_bits());
    assert_eq!(v41.scoring_func, "sqrtsoftplus");
    assert_eq!(v41.routed_scaling_factor, 1.5);
    assert_eq!(v41.swiglu_limit, 10.0);
    assert_eq!(v41.compress_rope_theta, 160_000.0);
    assert_eq!(
        (
            v41.rope_yarn_factor,
            v41.rope_yarn_orig_ctx,
            v41.rope_yarn_beta_fast,
            v41.rope_yarn_beta_slow
        ),
        (16.0, 65_536, 32.0, 1.0)
    );
    assert_eq!(v41.quant_weight_block_size, [32, 32]);
    assert_eq!(v41.quant_expert_dtype, "fp4");
    assert_eq!(v41.image_token_id, 129_264);

    // compress_ratios: [0,0, 2 x18, 1 x20, 0,0,0]; KV sources [2,8,14,20]; index sources add
    // 24..36 step 4; everything else reuses.
    let mut expected_ratios = vec![0, 0];
    expected_ratios.extend([2; 18]);
    expected_ratios.extend([1; 20]);
    expected_ratios.extend([0; 3]);
    assert_eq!(v41.compress_ratios, expected_ratios);
    for il in 0..43 {
        let mode = v41.layer_mode(il);
        let expected = match il {
            0 | 1 | 40.. => Csa2LayerMode::Window,
            2 | 8 | 14 | 20 => Csa2LayerMode::Full,
            24 | 28 | 32 | 36 => Csa2LayerMode::Reindex { kv_source: 20 },
            3..=7 => Csa2LayerMode::Reuse {
                kv_source: 2,
                index_source: 2,
            },
            9..=13 => Csa2LayerMode::Reuse {
                kv_source: 8,
                index_source: 8,
            },
            15..=19 => Csa2LayerMode::Reuse {
                kv_source: 14,
                index_source: 14,
            },
            _ => Csa2LayerMode::Reuse {
                kv_source: 20,
                index_source: (il / 4) * 4,
            },
        };
        assert_eq!(mode, expected, "layer {il}");
    }

    // Engram at 1 and 14; the declared table rows are exactly the prime bucket sums.
    assert_eq!(v41.engram_layer_ids, [1, 14]);
    assert_eq!(v41.engram_slot(14), Some(1));
    assert_eq!(v41.engram_hash_columns(), 24);
    assert_eq!(
        v41.derived_engram_table_rows(),
        [384_006_168u64, 384_016_682]
    );

    // DSpark: 3 blocks, 128 experts top-3, targets 37..39.
    assert_eq!(v41.num_nextn_predict_layers, 3);
    assert_eq!(v41.dspark_target_layer_ids, [37, 38, 39]);
    assert_eq!(
        (v41.dspark_n_routed_experts, v41.dspark_num_experts_per_tok),
        (128, 3)
    );
    assert_eq!(
        (
            v41.dspark_block_size,
            v41.dspark_noise_token_id,
            v41.dspark_markov_rank
        ),
        (5, 128_799, 256)
    );

    let vision = &v41.vision;
    assert_eq!(
        (
            vision.num_hidden_layers,
            vision.hidden_size,
            vision.num_attention_heads,
            vision.intermediate_size,
            vision.patch_size,
            vision.downsample_ratio
        ),
        (32, 1024, 16, 2816, 14, 3)
    );
}

#[test]
fn every_topology_invariant_refuses_by_name() {
    let base = config_1017();
    let v41 = base.dsv41.as_ref().unwrap();
    let check = |mutate: &dyn Fn(&mut DeepSeekV41Config), needle: &str| {
        let mut broken = v41.clone();
        mutate(&mut broken);
        let error = broken.validate(40, 64, 129_280).unwrap_err();
        assert!(error.contains(needle), "{needle:?} not in {error:?}");
    };
    v41.validate(40, 64, 129_280).unwrap();
    check(
        &|c| c.index_source_layer_ids.retain(|&l| l != 8),
        "kv source layer 8 is not an index source",
    );
    check(
        &|c| c.candidate_source_layer_id = 24,
        "candidate_source_layer_id 24 is not a kv source",
    );
    check(
        &|c| c.compress_ratios[21] = 2,
        "layer 21 (ratio 2) has no kv source at or before it with the same ratio",
    );
    check(&|c| c.compress_ratios[2] = 0, "index source layer 2");
    check(
        &|c| c.engram_layer_ids = vec![1, 40],
        "engram layer 40 is not a trunk layer",
    );
    check(
        &|c| c.engram_num_embeddings = vec![384_006_168],
        "engram_num_embeddings has 1 entries for 2 engram layers",
    );
    check(&|c| c.engram_num_embeddings[1] += 1, "prime bucket sums");
    check(
        &|c| c.num_nextn_predict_layers = 2,
        "compress_ratios has 43 entries",
    );
    check(
        &|c| c.compress_ratios[41] = 1,
        "DSpark block 1 has compress ratio 1",
    );
    check(
        &|c| c.quant_weight_block_size = vec![128, 128],
        "weight_block_size",
    );
    check(&|c| c.scoring_func = "sigmoid".into(), "scoring_func");
    check(
        &|c| c.dspark_target_layer_ids = vec![40],
        "dspark_target_layer_ids",
    );
}

#[test]
#[should_panic(expected = "deepseek_v41 config.json missing required field kv_source_layer_ids")]
fn a_missing_vendor_field_refuses_by_name() {
    let json = CONFIG_1017.replace("\"kv_source_layer_ids\"", "\"kv_source_layer_ids_absent\"");
    ModelConfig::from_hf(&HfConfig::parse(&json));
}

#[test]
#[should_panic(
    expected = "deepseek_v41 config.json missing required field quantization_config.scale_fmt"
)]
fn the_checkpoint_numerics_are_required() {
    let json = CONFIG_1017.replace("\"scale_fmt\"", "\"scale_format\"");
    ModelConfig::from_hf(&HfConfig::parse(&json));
}

#[test]
fn plan_is_typed_csa2_with_single_pass_mhc_engram_and_dspark() {
    let config = config_1017();
    assert_eq!(for_config(&config).unwrap().family, "deepseek_v41");
    let plan = super::super::compile_for_load(&config).unwrap();
    assert_eq!(plan, PACK.compile_plan(&config).unwrap());
    // The canonical compiler and the DSv4 packs refuse the family instead of compiling it.
    assert!(matches!(
        ModelPlan::compile(&config),
        Err(PlanCompileError::UnsupportedSemantics {
            field: "deepseek_v41",
            ..
        })
    ));
    for alias in ["deepseek_v4", "deepseek_v4_dspark"] {
        assert!(matches!(
            by_alias(alias).unwrap().compile_plan(&config),
            Err(PlanCompileError::ModelPackMismatch { .. })
        ));
    }

    assert_eq!(plan.layers.len(), 40);
    assert_eq!(plan.hidden_size, 5120);
    assert!(plan.mtp_blocks.is_empty());
    assert!(plan.vision.is_none() && plan.multimodal.is_none());
    assert_eq!(plan.partition_boundaries, (1..40).collect::<Vec<_>>());
    let numerics = plan.deepseek_v41.unwrap();
    assert_eq!(numerics.dense_weight_block, 32);
    assert_eq!(numerics.linear_activation, BlockQuant::FP8_32_UE8M0);

    let csa2 = |layer: &LayerPlan| match &layer.attention {
        AttentionPlan::Csa2(attention) => *attention,
        other => panic!("layer {} is {other:?}", layer.index),
    };
    let window = csa2(&plan.layers[0]);
    assert_eq!(window.mode, Csa2Mode::Window);
    assert_eq!(window.rope.base, 10_000.0);
    assert_eq!(window.rope.factors, RopeFactors::None);
    assert_eq!(
        (
            window.query_heads,
            window.q_lora_rank,
            window.head_dim,
            window.rope_head_dim,
            window.output_lora_rank,
            window.output_groups,
            window.window
        ),
        (64, 1280, 512, 64, 1024, 8, 128)
    );
    let full = csa2(&plan.layers[2]);
    assert_eq!(full.rope.base, 160_000.0);
    assert_eq!(
        full.rope.factors,
        RopeFactors::Yarn {
            factor: 16.0,
            original_context: 65_536,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    );
    let Csa2Mode::Full {
        ratio: 2,
        compressor,
        indexer,
    } = full.mode
    else {
        panic!("layer 2 is {:?}", full.mode);
    };
    assert_eq!(compressor.pooling, Csa2Pooling::SoftmaxGated);
    assert_eq!(compressor.latent_quant, BlockQuant::FP4_16_E4M3);
    assert_eq!(indexer.candidates, Csa2Candidates::None);
    assert_eq!(
        (indexer.heads, indexer.head_dim, indexer.top_k),
        (32, 128, 512)
    );
    let Csa2Mode::Full {
        ratio: 1,
        compressor,
        indexer,
    } = csa2(&plan.layers[20]).mode
    else {
        panic!("layer 20 is not Full");
    };
    assert_eq!(compressor.pooling, Csa2Pooling::Projection);
    assert_eq!(
        indexer.candidates,
        Csa2Candidates::Produce {
            top_blocks: 2048,
            block_size: 8
        }
    );
    let Csa2Mode::Reindex {
        ratio: 1,
        kv_source: 20,
        indexer,
    } = csa2(&plan.layers[28]).mode
    else {
        panic!("layer 28 is not Reindex");
    };
    assert_eq!(
        indexer.candidates,
        Csa2Candidates::Restrict { source_layer: 20 }
    );
    assert_eq!(
        csa2(&plan.layers[31]).mode,
        Csa2Mode::Reuse {
            ratio: 1,
            kv_source: 20,
            index_source: 28
        }
    );
    assert_eq!(
        plan.layers[2].state,
        StatePlan::Csa2(Csa2StatePlan {
            window: 128,
            head_dim: 512,
            compressed: Csa2CompressedState::Own {
                ratio: 2,
                index_head_dim: 128,
                partial_group: true
            }
        })
    );
    assert_eq!(
        plan.layers[5].state,
        StatePlan::Csa2(Csa2StatePlan {
            window: 128,
            head_dim: 512,
            compressed: Csa2CompressedState::Shared {
                ratio: 2,
                source_layer: 2
            }
        })
    );

    for (il, layer) in plan.layers.iter().enumerate() {
        let ResidualTopology::SinglePassHyperConnections(hc) = layer.residual else {
            panic!("layer {il} residual is {:?}", layer.residual);
        };
        assert_eq!((hc.streams, hc.sinkhorn_iterations), (4, 20));
        assert_eq!(hc.mix_norm_epsilon.to_bits(), 1e-20f32.to_bits());
        assert_eq!(hc.engram.is_some(), il == 1 || il == 14, "layer {il}");
        let MlpPlan::Moe(moe) = &layer.mlp else {
            panic!("layer {il} is dense");
        };
        assert_eq!((moe.expert_count, moe.experts_per_token), (384, 6));
        assert_eq!(
            moe.activation,
            ActivationPlan::SwiGluPreClamped { limit: 10.0 }
        );
    }
    let ResidualTopology::SinglePassHyperConnections(SinglePassHcPlan {
        engram: Some(engram),
        ..
    }) = plan.layers[14].residual
    else {
        unreachable!()
    };
    assert_eq!(
        (engram.table_index, engram.table_rows, engram.hash_columns),
        (1, 384_016_682, 24)
    );

    let Some(DrafterPlan::Dspark(dspark)) = plan.drafter.as_ref() else {
        panic!("no DSpark drafter");
    };
    assert_eq!(dspark.target_layer_ids, [37, 38, 39]);
    assert_eq!(
        (dspark.block_size, dspark.noise_token_id, dspark.markov_rank),
        (5, 128_799, 256)
    );
    assert_eq!(
        dspark.blocks.iter().map(|b| b.index).collect::<Vec<_>>(),
        [40, 41, 42]
    );
    for block in &dspark.blocks {
        assert_eq!(csa2(block).mode, Csa2Mode::Window);
        let MlpPlan::Moe(moe) = &block.mlp else {
            panic!("DSpark block is dense");
        };
        assert_eq!((moe.expert_count, moe.experts_per_token), (128, 3));
    }

    let operations = plan.operations();
    for operation in [
        OperationKind::Fp8Block32Linear,
        OperationKind::Csa2Attention,
        OperationKind::Csa2Compressor,
        OperationKind::Csa2Indexer,
        OperationKind::Csa2CandidatePool,
        OperationKind::Csa2KvReuse,
        OperationKind::Csa2IndexReuse,
        OperationKind::Csa2State,
        OperationKind::SinglePassHyperConnections,
        OperationKind::EngramLookup,
        OperationKind::SqrtSoftplusRouter,
        OperationKind::SwiGluPreClampedActivation,
        OperationKind::DsparkFusion,
        OperationKind::DsparkMarkovHead,
        OperationKind::DsparkConfidenceHead,
    ] {
        assert!(operations.contains(&operation), "missing {operation:?}");
    }
    // None of the DSv4 programs is claimed for V4.1.
    for operation in [
        OperationKind::CompressedMlaAttention,
        OperationKind::KvCompressor,
        OperationKind::SparseIndex,
        OperationKind::SharedSparseIndex,
        OperationKind::HyperConnections,
        OperationKind::TokenHashRouter,
        OperationKind::CompressedAttentionState,
    ] {
        assert!(!operations.contains(&operation), "claims {operation:?}");
    }
    assert_eq!(
        operations
            .iter()
            .filter(|&&op| op == OperationKind::EngramLookup)
            .count(),
        2
    );
    assert_eq!(
        operations
            .iter()
            .filter(|&&op| op == OperationKind::Csa2Compressor)
            .count(),
        4
    );
    // Nothing executes it: every tuned capability is blocked.
    for manifest in [
        crate::execution_manifest::NATIVE_EAGER,
        crate::execution_manifest::DECODE_BATCH,
        crate::execution_manifest::DECODE_GRAPH,
        crate::execution_manifest::PIPELINE,
    ] {
        let capabilities = manifest.capabilities(&plan);
        assert!(!capabilities.batch.supported);
        assert!(
            capabilities
                .batch
                .blockers
                .contains(&OperationKind::Csa2Attention)
        );
    }
}

/// `{name: (dtype, shape)}` of every header tensor.
type HeaderCensus = BTreeMap<String, (String, Vec<u64>)>;

/// Expand the class fixture into the header census plus its recorded tensor and byte totals.
fn pinned_census() -> (HeaderCensus, usize, u64) {
    let mut names = BTreeMap::new();
    let (mut tensors, mut bytes) = (0, 0);
    let parse_ranges = |text: &str| -> Vec<u32> {
        text.split(',')
            .flat_map(|part| {
                let (a, b) = part.split_once('-').unwrap_or((part, part));
                a.parse::<u32>().unwrap()..=b.parse::<u32>().unwrap()
            })
            .collect()
    };
    for line in CENSUS_1017.lines() {
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
        let [pattern, dtype, shape, indices, experts] = cols[..] else {
            panic!("malformed census fixture line {line:?}");
        };
        let shape: Vec<u64> = shape.split(',').map(|d| d.parse().unwrap()).collect();
        let layers: Vec<Option<u32>> = if indices == "-" {
            vec![None]
        } else {
            parse_ranges(indices).into_iter().map(Some).collect()
        };
        let experts: Vec<Option<u32>> = if experts == "-" {
            vec![None]
        } else {
            (0..experts.parse::<u32>().unwrap()).map(Some).collect()
        };
        for layer in &layers {
            let name = match layer {
                Some(layer) => pattern.replace("{L}", &layer.to_string()),
                None => pattern.to_string(),
            };
            for expert in &experts {
                let name = match expert {
                    Some(expert) => name.replace("{E}", &expert.to_string()),
                    None => name.clone(),
                };
                assert!(
                    names
                        .insert(name.clone(), (dtype.to_string(), shape.clone()))
                        .is_none(),
                    "duplicate {name}"
                );
            }
        }
    }
    (names, tensors, bytes)
}

fn dtype_bytes(dtype: &str) -> u64 {
    match dtype {
        "BF16" => 2,
        "F32" => 4,
        "F8_E4M3" | "F8_E8M0" | "I8" => 1,
        other => panic!("unexpected dtype {other}"),
    }
}

/// Census gate: the config-derived schema reproduces all 96,085 header tensors of the pinned
/// artifact (names, dtypes, shapes), and the generated contract binds the census the header
/// reader builds from them with nothing missing, extra, ambiguous or reshaped.
#[test]
fn expected_census_reproduces_the_pinned_artifact_and_the_contract_binds_it() {
    let (artifact, tensors, bytes) = pinned_census();
    assert_eq!(artifact.len(), 96_085);
    assert_eq!(tensors, 96_085);
    let config = config_1017();
    let expected = expected_census(&config);
    let missing: Vec<_> = expected
        .keys()
        .filter(|name| !artifact.contains_key(*name))
        .collect();
    let extra: Vec<_> = artifact
        .keys()
        .filter(|name| !expected.contains_key(*name))
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

    // The header reader's census, from synthesized headers with exact byte extents.
    let mut offset = 0usize;
    let headers: BTreeMap<String, StInfo> = artifact
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
    assert_eq!(offset as u64, bytes, "fixture byte total");
    let census = census_from_safetensors_headers(&headers).unwrap();
    let entries: Vec<_> = census.tensors.iter().map(|row| row.entry.clone()).collect();
    assert_eq!(
        entries.iter().map(|e| e.physical_bytes).sum::<u64>(),
        bytes,
        "every header byte lands in exactly one census row"
    );
    for entry in &entries {
        if let StorageLayout::Quantized(layout) = &entry.storage
            && layout.format == "FP8_E4M3"
        {
            let expect: &[u32] = if entry.name.ends_with(".engram.embed.weight") {
                &[1, 32]
            } else {
                &[32, 32]
            };
            assert_eq!(layout.block_shape, expect, "{}", entry.name);
        }
    }

    let plan = PACK.compile_plan(&config).unwrap();
    let contract = PACK
        .compile_tensor_contract(
            &config,
            &plan,
            CheckpointDialect::HfSafetensors,
            ContractOptions::default(),
        )
        .unwrap();
    let ids: std::collections::BTreeSet<_> =
        contract.requirements.iter().map(|r| r.id.clone()).collect();
    assert_eq!(
        ids.len(),
        contract.requirements.len(),
        "semantic ids are unique"
    );
    let bound = contract.bind(&entries).unwrap();
    assert_eq!(bound.tensors.len(), entries.len());
    assert_eq!(bound.tensors.len(), contract.requirements.len());

    // The whole loader boundary (head ownership included) accepts it too.
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
    let wq_b = requirement("layers.0.attn.wq_b.weight");
    assert_eq!(wq_b.quant, QuantConstraint::Fp8Block32);
    assert_eq!(
        wq_b.auxiliaries.as_deref(),
        Some(["layers.0.attn.wq_b.scale".to_string()].as_slice())
    );
    let expert = requirement("mtp.1.ffn.experts.127.w2.weight");
    assert_eq!(expert.quant, QuantConstraint::Mxfp4);
    assert_eq!(expert.shape, [5120, 2304]);
    assert_eq!(
        expert.id,
        TensorId::Expert {
            layer: 41,
            expert: 127,
            tensor: ExpertTensor::Down
        }
    );
    assert_eq!(expert.owner, TensorOwner::Mtp(1));
    assert_eq!(
        requirement("layers.14.engram.embed.weight").quant,
        QuantConstraint::Fp8Row32
    );
    assert_eq!(
        requirement("mtp.2.markov_head.embed.weight").id,
        TensorId::Dspark(DsparkTensor::MarkovEmbedding)
    );
    assert_eq!(
        requirement("mtp.0.main_proj.weight").id,
        TensorId::Dspark(DsparkTensor::MainProjection)
    );
    assert_eq!(
        requirement("layers.20.attn.compressor.wkv.weight").id,
        TensorId::Layer {
            index: 20,
            tensor: LayerTensor::KvCompressorKeyValue
        }
    );
    assert_eq!(
        requirement("vision.blocks.31.mlp.w1.weight").owner,
        TensorOwner::Vision(Some(31))
    );
    for name in ["aligner.w1.weight", "image_newline"] {
        assert_eq!(requirement(name).owner, TensorOwner::Vision(None), "{name}");
    }
    assert_eq!(requirement("head.weight").id, TensorId::OutputProjection);

    // A missing or extra tensor fails the census.
    let mut short = entries.clone();
    short.retain(|entry| entry.name != "layers.8.attn.compressor.wgate.weight");
    assert!(matches!(
        contract.bind(&short),
        Err(TensorContractError::Missing { .. })
    ));
    let mut long = entries.clone();
    let mut stray = long
        .iter()
        .find(|entry| entry.name == "layers.20.attn.compressor.wkv.weight")
        .unwrap()
        .clone();
    stray.name = "layers.20.attn.compressor.wgate.weight".into();
    long.push(stray);
    assert!(matches!(
        contract.bind(&long),
        Err(TensorContractError::Extra { .. })
    ));
}
