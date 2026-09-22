//! Real-file composite loading compared with Memra's independent deterministic reference fixture.
use memra_gguf::{
    bound_source::composite::BoundCompositeSource,
    config::{HfConfig, ModelConfig},
    model_packs,
    model_plan::MlpPlan,
    source::Hy3RepackSource,
    surface_catalog::LoadScope,
    tensor_contract::{CheckpointDialect, LayerTensor, TensorId, TensorMatch},
};
use memra_reference::{ReferenceTensor, ReferenceWeights, deterministic_fixture, execute};
use std::path::{Path, PathBuf};

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn write_artifact(directory: &Path, rows: Vec<(String, Vec<u64>, Vec<f32>)>, overlay: bool) {
    std::fs::create_dir(directory).unwrap();
    let mut bytes = Vec::new();
    let mut entries = Vec::new();
    for (name, shape, values) in rows {
        let offset = bytes.len();
        for value in values {
            bytes.extend(value.to_le_bytes());
        }
        entries.push(format!("{name:?}:{{\"file\":\"weights.bin\",\"offset\":{offset},\"qtype\":\"F32\",\"ne\":{shape:?},\"bytes\":{}}}",bytes.len()-offset));
    }
    std::fs::write(directory.join("weights.bin"), bytes).unwrap();
    let prefix = if overlay {
        r#""format":"memra-expert-overlay-v2","source_dir":"../base","pruned_experts":{"0":[0,2]},"#
    } else {
        r#""format":"memra-repack-v1","#
    };
    std::fs::write(
        directory.join("manifest.json"),
        format!("{{{prefix}\"tensors\":{{{}}}}}", entries.join(",")),
    )
    .unwrap();
}

#[test]
fn retained_composite_load_matches_reference_weights_and_logits_bit_for_bit() {
    let raw = r#"{"model_type":"qwen3_moe","tie_word_embeddings":true,"num_hidden_layers":1,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"intermediate_size":16,"vocab_size":32,"max_position_embeddings":32,"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":8}"#;
    let config = ModelConfig::from_hf(&HfConfig::try_parse(raw).unwrap());
    let pack = model_packs::for_config(&config).unwrap();
    let original_plan = pack.compile_plan(&config).unwrap();
    let original = deterministic_fixture(&original_plan).unwrap();
    let mut plan = original_plan.clone();
    let MlpPlan::Moe(moe) = &mut plan.layers[0].mlp else {
        unreachable!()
    };
    moe.retained_experts = Some(vec![1, 3]);
    let expected = deterministic_fixture(&plan).unwrap();
    let contract = pack
        .compile_tensor_contract(
            &config,
            &original_plan,
            CheckpointDialect::Gguf,
            pack.contract_options(&config),
        )
        .unwrap();
    let directory =
        std::env::temp_dir().join(format!("memra-composite-reference-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let scratch = Scratch(directory);
    let mut base_rows = Vec::new();
    for requirement in contract
        .requirements
        .iter()
        .filter(|r| r.required || original.weights.contains_key(&r.id))
    {
        assert_eq!(requirement.match_mode, TensorMatch::OneOf);
        let tensor = original
            .weights
            .get(&requirement.id)
            .unwrap_or_else(|| panic!("reference fixture lacks {:?}", requirement.id));
        assert!(tensor.ints.is_none());
        assert_eq!(
            requirement
                .shape
                .iter()
                .rev()
                .map(|&d| d as usize)
                .collect::<Vec<_>>(),
            tensor.shape
        );
        base_rows.push((
            requirement.names[0].clone(),
            requirement.shape.clone(),
            tensor.data.clone(),
        ));
    }
    write_artifact(&scratch.0.join("base"), base_rows, false);
    std::fs::write(scratch.0.join("base/config.json"), raw).unwrap();
    let mut overlay = Vec::new();
    for (tensor, suffix) in [
        (LayerTensor::MoeExpertGateBank, "gate"),
        (LayerTensor::MoeExpertUpBank, "up"),
        (LayerTensor::MoeExpertDownBank, "down"),
    ] {
        let id = TensorId::Layer { index: 0, tensor };
        let values = &expected.weights[&id];
        let count: usize = values.shape[1..].iter().product();
        for (row, original) in [1, 3].into_iter().enumerate() {
            overlay.push((
                format!("blk.0.ffn_{suffix}_exps.{original}.weight"),
                values.shape[1..].iter().rev().map(|&d| d as u64).collect(),
                values.data[row * count..(row + 1) * count].to_vec(),
            ));
        }
    }
    write_artifact(&scratch.0.join("overlay"), overlay, true);
    let source = Hy3RepackSource::open(&scratch.0.join("overlay")).unwrap();
    let bound = BoundCompositeSource::compile(&source, LoadScope::Full).unwrap();
    assert_eq!(bound.plan(), &plan);
    let mut actual = ReferenceWeights::new();
    for (id, expected_tensor) in &expected.weights {
        let selection = bound
            .catalog()
            .selected()
            .get(id)
            .unwrap_or_else(|| panic!("composite fixture lacks {id:?}"));
        let values = if let Some(ids) = &selection.member_ids {
            ids.iter()
                .flat_map(|&original| {
                    let tensor = bound.member_tensor(id, original).unwrap();
                    assert_eq!(
                        tensor
                            .ne
                            .iter()
                            .rev()
                            .map(|&d| d as usize)
                            .collect::<Vec<_>>(),
                        expected_tensor.shape[1..]
                    );
                    tensor
                        .bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect::<Vec<_>>()
                })
                .collect()
        } else {
            let tensor = bound.tensor(id).unwrap();
            assert_eq!(
                tensor
                    .ne
                    .iter()
                    .rev()
                    .map(|&d| d as usize)
                    .collect::<Vec<_>>(),
                expected_tensor.shape
            );
            tensor
                .bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        };
        actual.insert(
            id.clone(),
            ReferenceTensor::new(expected_tensor.shape.clone(), values).unwrap(),
        );
    }
    assert_eq!(actual, expected.weights);
    let reference = execute(&plan, &expected.weights, &expected.token_ids).unwrap();
    let loaded = execute(bound.plan(), &actual, &expected.token_ids).unwrap();
    assert_eq!(
        loaded
            .logits
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        reference
            .logits
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );

    // Exercise the same preparation and ggml-name adapter now used by engine root entrypoints.
    let prepared = memra_gguf::bound_source::PreparedModelSource::text(&source).unwrap();
    prepared
        .with_runtime(|runtime| {
            let (config, root_plan) = model_packs::compile_for_source(runtime).unwrap();
            assert_eq!(root_plan, plan);
            let contract = pack
                .compile_tensor_contract(
                    &config,
                    &root_plan,
                    CheckpointDialect::Gguf,
                    pack.contract_options(&config),
                )
                .unwrap();
            let mut root_weights = ReferenceWeights::new();
            for (id, expected_tensor) in &expected.weights {
                let requirement = contract.requirements.iter().find(|r| &r.id == id).unwrap();
                let names = if requirement.match_mode == TensorMatch::All {
                    &requirement.names[..]
                } else {
                    &requirement.names[..1]
                };
                let mut values = Vec::new();
                for name in names {
                    let tensor = runtime.try_find(name).unwrap().unwrap();
                    values.extend(
                        tensor
                            .bytes
                            .chunks_exact(4)
                            .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
                    );
                }
                root_weights.insert(
                    id.clone(),
                    ReferenceTensor::new(expected_tensor.shape.clone(), values).unwrap(),
                );
            }
            assert_eq!(root_weights, expected.weights);
            let output = execute(&root_plan, &root_weights, &expected.token_ids).unwrap();
            assert_eq!(
                output
                    .logits
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                reference
                    .logits
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            for name in [
                "exp_probs_b.bias",
                "ffn_gate_shexp.weight",
                "ffn_up_shexp.weight",
                "ffn_down_shexp.weight",
                "ffn_gate_inp_shexp.weight",
            ] {
                assert!(!runtime.try_has(&format!("blk.0.{name}")).unwrap());
            }
        })
        .unwrap();
}
