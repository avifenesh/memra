//! Pure source/placement accounting shared by automatic placement and CPU integration gates.
use memra_gguf::config::ModelConfig;
use memra_gguf::model_plan::{MlpPlan, ModelPlan};
use memra_gguf::placement::LayerPlacementCost;
use memra_gguf::source::TensorSource;
use memra_gguf::tensor_contract::{LayerTensor, TensorId, TensorOwner};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyError {
    message: String,
}

impl TopologyError {
    pub(super) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for TopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}

impl std::error::Error for TopologyError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AutoArtifactCosts {
    pub(super) layers: Vec<LayerPlacementCost>,
    pub(super) first_fixed_bytes: u64,
    pub(super) last_fixed_bytes: u64,
    pub(super) trunk_expert_bytes: u64,
    pub(super) non_distributed_bytes: u64,
}

fn placement_first_stage_tensor(id: &TensorId) -> bool {
    match id {
        TensorId::TokenEmbedding | TensorId::RopeFactors | TensorId::Vision { .. } => true,
        TensorId::QuantAux { tensor, .. } => placement_first_stage_tensor(tensor),
        _ => false,
    }
}

fn routed_expert_tensor(id: &TensorId) -> bool {
    match id {
        TensorId::Expert { .. } => true,
        TensorId::Layer {
            tensor:
                LayerTensor::MoeExpertGateUpBank
                | LayerTensor::MoeExpertGateBank
                | LayerTensor::MoeExpertUpBank
                | LayerTensor::MoeExpertDownBank
                | LayerTensor::MoeExpertOutputScale,
            ..
        } => true,
        TensorId::QuantAux { tensor, .. } => routed_expert_tensor(tensor),
        _ => false,
    }
}

fn checked_add_bytes(total: &mut u64, bytes: u64, label: &str) -> Result<(), TopologyError> {
    *total = total
        .checked_add(bytes)
        .ok_or_else(|| TopologyError::new(format!("{label} byte total overflows u64")))?;
    Ok(())
}

pub(super) fn artifact_costs(
    src: &dyn TensorSource,
    cfg: &ModelConfig,
    plan: &ModelPlan,
) -> Result<AutoArtifactCosts, TopologyError> {
    let charges = match src.bound_tensor_charges().map_err(TopologyError::new)? {
        Some(charges) => charges,
        None => {
            let binding = memra_gguf::checkpoint_binding::bind_source(src, cfg, plan)
                .map_err(|error| {
                    TopologyError::new(format!(
                        "automatic parallel placement cannot bind the tensor contract: {error}"
                    ))
                })?
                .bound;

            binding
                .tensors
                .into_iter()
                .map(|(id, tensor)| memra_gguf::source::BoundTensorCharge {
                    id,
                    owner: tensor.owner,
                    physical_bytes: tensor.physical_bytes,
                    execution_selected: true,
                })
                .collect()
        }
    };

    let mut layers = vec![LayerPlacementCost::default(); plan.layers.len()];
    let mut first_fixed_bytes = 0u64;
    let mut last_fixed_bytes = 0u64;
    let mut trunk_expert_bytes = 0u64;
    let mut total_bytes = 0u64;
    for tensor in &charges {
        let id = &tensor.id;
        checked_add_bytes(
            &mut total_bytes,
            tensor.physical_bytes,
            "automatic placement checkpoint",
        )?;
        match tensor.owner {
            TensorOwner::Layer(layer) if (layer as usize) < layers.len() => {
                checked_add_bytes(
                    &mut layers[layer as usize].weight_bytes,
                    tensor.physical_bytes,
                    "automatic placement layer",
                )?;
            }
            // Some legacy contracts retain the physical MTP index rather than rewriting the
            // owner to TensorOwner::Mtp. It executes with the tail/head stage either way.
            TensorOwner::Layer(_) => checked_add_bytes(
                &mut last_fixed_bytes,
                tensor.physical_bytes,
                "automatic placement head stage",
            )?,
            TensorOwner::Vision(_) => checked_add_bytes(
                &mut first_fixed_bytes,
                tensor.physical_bytes,
                "automatic placement first stage",
            )?,
            TensorOwner::Global if placement_first_stage_tensor(id) => checked_add_bytes(
                &mut first_fixed_bytes,
                tensor.physical_bytes,
                "automatic placement first stage",
            )?,
            TensorOwner::Global | TensorOwner::Mtp(_) => checked_add_bytes(
                &mut last_fixed_bytes,
                tensor.physical_bytes,
                "automatic placement head stage",
            )?,
        }

        let trunk_expert = routed_expert_tensor(id)
            && matches!(
                tensor.owner,
                TensorOwner::Layer(layer)
                    if (layer as usize) < plan.layers.len()
                        && matches!(plan.layers[layer as usize].mlp, MlpPlan::Moe(_))
            );
        if trunk_expert {
            checked_add_bytes(
                &mut trunk_expert_bytes,
                tensor.physical_bytes,
                "automatic placement trunk experts",
            )?;
        }
    }
    let non_distributed_bytes = total_bytes
        .checked_sub(trunk_expert_bytes)
        .ok_or_else(|| TopologyError::new("automatic placement expert bytes exceed total bytes"))?;
    Ok(AutoArtifactCosts {
        layers,
        first_fixed_bytes,
        last_fixed_bytes,
        trunk_expert_bytes,
        non_distributed_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::{
        bound_source::PreparedModelSource,
        config::HfConfig,
        source::{GgufSource, Hy3RepackSource, SafetensorsSource},
        tensor_contract::{CheckpointDialect, TensorMatch},
    };
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Fixture(PathBuf);
    impl Fixture {
        fn new(raw: &str) -> Self {
            static SEQ: AtomicUsize = AtomicUsize::new(0);
            let root = std::env::temp_dir().join(format!(
                "memra-root-placement-{}-{}",
                std::process::id(),
                SEQ.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&root).unwrap();
            let cfg = ModelConfig::from_hf(&HfConfig::try_parse(raw).unwrap());
            let pack = memra_gguf::model_packs::for_config(&cfg).unwrap();
            let plan = pack.compile_plan(&cfg).unwrap();
            let mut contract = pack
                .compile_tensor_contract(
                    &cfg,
                    &plan,
                    CheckpointDialect::HfSafetensors,
                    pack.contract_options(&cfg),
                )
                .unwrap();
            for inventory in pack
                .additional_inventory(&cfg, CheckpointDialect::HfSafetensors, Some(raw))
                .unwrap()
            {
                contract.requirements.extend(inventory.requirements);
            }
            let mut rows = BTreeMap::new();
            for r in contract.requirements.iter().filter(|r| r.required) {
                let names = if r.match_mode == TensorMatch::All {
                    &r.names[..]
                } else {
                    &r.names[..1]
                };
                for name in names {
                    rows.insert(
                        name.clone(),
                        (
                            "F32",
                            r.shape.clone(),
                            1.0f32
                                .to_le_bytes()
                                .repeat(r.shape.iter().product::<u64>() as usize),
                        ),
                    );
                }
            }
            // A real folded FP8 scale verifies that placement includes auxiliary storage.
            if rows.contains_key("model.layers.0.self_attn.q_proj.weight") {
                let shape = rows["model.layers.0.self_attn.q_proj.weight"].1.clone();
                rows.insert(
                    "model.layers.0.self_attn.q_proj.weight".into(),
                    (
                        "F8_E4M3",
                        shape.clone(),
                        vec![0x38; shape.iter().product::<u64>() as usize],
                    ),
                );
                rows.insert(
                    "model.layers.0.self_attn.q_proj.weight_scale".into(),
                    ("F32", vec![1], 1.0f32.to_le_bytes().to_vec()),
                );
            }
            let mut header = Vec::new();
            let mut bytes = Vec::new();
            for (name, (dtype, shape, data)) in rows {
                let offset = bytes.len();
                bytes.extend(data);
                header.push(format!("{name:?}:{{\"dtype\":{dtype:?},\"shape\":{shape:?},\"data_offsets\":[{offset},{}]}}",bytes.len()));
            }
            let header = format!("{{{}}}", header.join(","));
            let mut data = (header.len() as u64).to_le_bytes().to_vec();
            data.extend(header.as_bytes());
            data.extend(bytes);
            std::fs::write(root.join("model.safetensors"), data).unwrap();
            std::fs::write(root.join("config.json"), raw).unwrap();
            Self(root)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
    const QWEN: &str = r#"{"model_type":"qwen3_moe","num_hidden_layers":1,"hidden_size":32,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,"intermediate_size":64,"vocab_size":32,"max_position_embeddings":128,"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":32}"#;
    const GEMMA: &str = r#"{"model_type":"gemma4","image_token_id":31,"vision_soft_tokens_per_image":1,"text_config":{"model_type":"gemma4_text","num_hidden_layers":2,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"num_global_key_value_heads":1,"head_dim":4,"global_head_dim":4,"intermediate_size":16,"vocab_size":32,"max_position_embeddings":64,"rms_norm_eps":0.000001,"sliding_window":8,"layer_types":["sliding_attention","full_attention"],"rope_parameters":{"full_attention":{"rope_theta":10000,"partial_rotary_factor":0.5},"sliding_attention":{"rope_theta":10000}}},"vision_config":{"hidden_size":8,"intermediate_size":16,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"max_position_embeddings":64,"patch_size":2,"position_embedding_size":16,"pooling_kernel_size":2,"rms_norm_eps":0.000001,"standardize":true,"use_clipped_linears":false,"hidden_activation":"gelu_pytorch_tanh","rope_parameters":{"rope_theta":100}}}"#;

    fn overlay(path: &Path, hidden: u64, mask: bool) {
        std::fs::create_dir(path).unwrap();
        std::fs::write(
            path.join("norm.bin"),
            1.0f32.to_le_bytes().repeat(hidden as usize),
        )
        .unwrap();
        let mask = if mask {
            r#""pruned_experts":{"0":[0,2]},"#
        } else {
            ""
        };
        std::fs::write(path.join("manifest.json"),format!("{{\"format\":\"memra-expert-overlay-v2\",\"source_dir\":\"..\",{mask}\"tensors\":{{\"output_norm.weight\":{{\"file\":\"norm.bin\",\"qtype\":\"F32\",\"ne\":[{hidden}],\"bytes\":{}}}}}}}",hidden*4)).unwrap();
    }

    #[test]
    fn prepared_ordinary_costs_match_legacy_tied_and_separate_head_accounting() {
        for tied in [false, true] {
            let dense = QWEN
                .replace("qwen3_moe", "qwen3")
                .replace(",\"num_experts\":4,\"num_experts_per_tok\":2", "")
                .replace(",\"moe_intermediate_size\":32", "")
                .replace("\"intermediate_size\":64", "\"intermediate_size\":32");
            let raw = dense.replacen('{', &format!("{{\"tie_word_embeddings\":{tied},"), 1);
            let f = Fixture::new(&raw);
            let source = SafetensorsSource::open(&f.0).unwrap();
            let (cfg, plan) = memra_gguf::model_packs::compile_for_source(&source).unwrap();
            let legacy = artifact_costs(&source, &cfg, &plan).unwrap();
            PreparedModelSource::text(&source)
                .unwrap()
                .with_runtime(|runtime| {
                    let (cfg, plan) = memra_gguf::model_packs::compile_for_source(runtime).unwrap();
                    let actual = artifact_costs(runtime, &cfg, &plan).unwrap();
                    assert_eq!(actual, legacy);
                    assert_eq!(actual.first_fixed_bytes, 32 * 32 * 4);
                    assert_eq!(
                        actual.last_fixed_bytes,
                        32 * 4 + if tied { 0 } else { 32 * 32 * 4 }
                    );
                    let charges = runtime.bound_tensor_charges().unwrap().unwrap();
                    let query = charges
                        .iter()
                        .find(|c| {
                            c.id == TensorId::Layer {
                                index: 0,
                                tensor: LayerTensor::Query,
                            }
                        })
                        .unwrap();
                    assert_eq!(query.physical_bytes, 32 * 32 + 4);
                    assert!(charges.iter().all(|c| c.execution_selected));
                })
                .unwrap();
        }
    }

    #[test]
    fn prepared_gguf_costs_match_the_legacy_production_census_path() {
        let f = Fixture::new(QWEN);
        let path = f.0.join("model.gguf");
        memra_gguf::micro_gguf::write_glm_dsa_micro(&path, 541).unwrap();
        let file = memra_gguf::GgufFile::open(&path).unwrap();
        let source = GgufSource(&file);
        let (cfg, plan) = memra_gguf::model_packs::compile_for_source(&source).unwrap();
        let legacy = artifact_costs(&source, &cfg, &plan).unwrap();
        PreparedModelSource::text(&source)
            .unwrap()
            .with_runtime(|runtime| {
                let (cfg, plan) = memra_gguf::model_packs::compile_for_source(runtime).unwrap();
                assert_eq!(artifact_costs(runtime, &cfg, &plan).unwrap(), legacy);
            })
            .unwrap();
    }

    #[test]
    fn composite_costs_keep_original_experts_and_exclude_shadowed_banks() {
        let f = Fixture::new(QWEN);
        let raw = SafetensorsSource::open(&f.0).unwrap();
        let (cfg, plan) = memra_gguf::model_packs::compile_for_source(&raw).unwrap();
        let legacy = artifact_costs(&raw, &cfg, &plan).unwrap();
        let path = f.0.join("overlay");
        overlay(&path, 32, true);
        let source = Hy3RepackSource::open(&path).unwrap();
        PreparedModelSource::text(&source)
            .unwrap()
            .with_runtime(|runtime| {
                let (cfg, plan) = memra_gguf::model_packs::compile_for_source(runtime).unwrap();
                let actual = artifact_costs(runtime, &cfg, &plan).unwrap();
                assert_eq!(
                    runtime.active_experts(0),
                    Some(&[false, true, false, true][..])
                );
                assert_eq!(actual.trunk_expert_bytes, legacy.trunk_expert_bytes / 2);
                assert_eq!(actual.non_distributed_bytes, legacy.non_distributed_bytes);
                assert_eq!(
                    runtime
                        .physical_tensor_inventory()
                        .unwrap()
                        .components
                        .len(),
                    2
                );
                assert!(runtime.try_find("blk.0.ffn_gate_exps.0.weight").is_err());
            })
            .unwrap();
    }

    #[test]
    fn composite_text_vision_reservation_is_separate_from_execution_bytes() {
        let f = Fixture::new(GEMMA);
        let raw = SafetensorsSource::open(&f.0).unwrap();
        let (cfg, plan) = memra_gguf::model_packs::compile_for_source(&raw).unwrap();
        let legacy = artifact_costs(&raw, &cfg, &plan).unwrap();
        let path = f.0.join("overlay");
        overlay(&path, 8, false);
        let source = Hy3RepackSource::open(&path).unwrap();
        PreparedModelSource::text(&source).unwrap().with_runtime(|runtime| {
            let(cfg,plan)=memra_gguf::model_packs::compile_for_source(runtime).unwrap();assert!(plan.vision.is_none());assert!(cfg.vision.is_some());
            let charges=runtime.bound_tensor_charges().unwrap().unwrap();let reservation:u64=charges.iter().filter(|c|!c.execution_selected).map(|c|c.physical_bytes).sum();
            let selected:u64=charges.iter().filter(|c|c.execution_selected).map(|c|c.physical_bytes).sum();assert!(reservation>0 && selected>0);
            assert!(charges.iter().filter(|c|!c.execution_selected).all(|c|matches!(c.owner,TensorOwner::Vision(_))));
            let actual=artifact_costs(runtime,&cfg,&plan).unwrap();assert_eq!(actual,legacy);assert!(actual.first_fixed_bytes>=reservation);
            assert_eq!(actual.non_distributed_bytes,selected+reservation);
            assert!(runtime.try_find("vision_model.patch_embedder.input_proj.weight").is_err());
            eprintln!("selected_execution_bytes={selected} inventory_vision_reservation_bytes={reservation}");
        }).unwrap();
    }
}
