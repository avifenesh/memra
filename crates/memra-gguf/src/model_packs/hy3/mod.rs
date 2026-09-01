use super::*;
use crate::config::HfConfig;
use crate::model_plan::SamplingDefaultsPlan;
use crate::tensor_contract::{LayerTensor, QuantConstraint, TensorId};

pub static PACK: ModelPack = ModelPack {
    family: "hy3",
    aliases: &["hy_v3", "hunyuan3", "hunyuan-3"],
    config_layout: ConfigLayout::Flat,
    tokenizer_sources: &[
        TokenizerSource::TokenizerJson,
        TokenizerSource::GgufMetadata,
    ],
    template: TemplateContract::ArtifactRequired,
    support: Some(NativeSupport::NativeReference),
    gates: &[
        Gate::Config,
        Gate::TokenizerTemplate,
        Gate::TensorCensus,
        Gate::TinyParity,
        Gate::CheckpointParity,
        Gate::RewriteParity,
        Gate::Serve,
    ],
    checkpoint_parity: Some(CheckpointParityGate {
        max_abs: 0.005,
        max_rel: 0.005,
        require_argmax: true,
    }),
    matches_config: |config| matches!(config.arch, Arch::Hy3),
    plan_builder,
    tensor_schema: canonical_tensor_schema,
    tiny_plan: Some(tiny_plan),
};

pub static NVFP4_PACK: ModelPack = ModelPack {
    family: "hy3_nvfp4",
    aliases: &["hy3-nvfp4", "hy_v3_nvfp4"],
    config_layout: ConfigLayout::Flat,
    tokenizer_sources: &[TokenizerSource::TokenizerJson],
    template: TemplateContract::ArtifactRequired,
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
    // The frozen same-artifact ModelOpt-vs-Memra gate was declared before capture: finite
    // logits, equal argmax, top-20 overlap >=18, cosine >=0.999, RMSE <=0.25, MAE <=0.10,
    // and max absolute error <=1.0. The pack-level scalar gate records the elementwise part;
    // the aggregate conditions remain in the sealed qualification receipt.
    checkpoint_parity: Some(CheckpointParityGate {
        max_abs: 1.0,
        max_rel: 0.0,
        require_argmax: true,
    }),
    matches_config: |config| matches!(config.arch, Arch::Hy3),
    plan_builder,
    tensor_schema: nvfp4_tensor_schema,
    tiny_plan: None,
};

fn plan_builder(config: &ModelConfig) -> Result<ModelPlan, PlanCompileError> {
    let mut plan = canonical_plan(config)?;
    plan.sampling_defaults = Some(SamplingDefaultsPlan {
        temperature: 0.9,
        top_p: 1.0,
    });
    Ok(plan)
}

fn tiny_plan() -> Result<ModelPlan, PlanCompileError> {
    plan_builder(&ModelConfig::from_hf(&HfConfig::parse(
        r#"{"model_type":"hy_v3","num_hidden_layers":2,
        "num_nextn_predict_layers":1,"hidden_size":8,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
        "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32,
        "rms_norm_eps":0.00001,"rope_parameters":{"rope_theta":11158840.0,
        "rope_type":"default"},"first_k_dense_replace":1,"num_experts":4,
        "num_experts_per_tok":2,"moe_intermediate_size":8,"num_shared_experts":1,
        "moe_router_use_sigmoid":true,"moe_router_enable_expert_bias":true,
        "route_norm":true,"router_scaling_factor":2.826,"qk_norm":true,
        "hidden_act":"silu"}"#,
    )))
}

#[allow(clippy::result_large_err)] // allow: the fat error type is the diagnostic contract here; boxing it would change the error surface
fn nvfp4_tensor_schema(
    config: &ModelConfig,
    plan: &ModelPlan,
    dialect: CheckpointDialect,
    options: ContractOptions,
) -> Result<TensorContract, TensorContractError> {
    let _ = config;
    if dialect != CheckpointDialect::HfSafetensors {
        return Err(TensorContractError::UnsupportedPlanOperation {
            operation: "hy3 NVFP4 non-safetensors schema",
        });
    }
    let mut contract = TensorContract::for_plan(plan, dialect, options)?;
    for requirement in &mut contract.requirements {
        if requirement.quant != QuantConstraint::Weight {
            continue;
        }
        let expert_layer = match requirement.id {
            TensorId::Layer {
                index,
                tensor:
                    LayerTensor::MoeExpertGateBank
                    | LayerTensor::MoeExpertUpBank
                    | LayerTensor::MoeExpertDownBank,
            } => Some(index),
            _ => None,
        };
        requirement.quant = match expert_layer {
            Some(_) => QuantConstraint::Nvfp4,
            _ => QuantConstraint::FloatOnly,
        };
    }
    Ok(contract)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AttentionGateKind;
    use crate::model_plan::{MlpPlan, RouterPlan, TensorPresence};
    use crate::tensor_contract::{
        FloatType, OutputHead, QuantLayout, StorageLayout, TensorCensusEntry, TensorMatch,
    };

    #[test]
    fn tiny_plan_covers_hy3_dense_moe_and_embedded_mtp_semantics() {
        let plan = tiny_plan().unwrap();
        assert_eq!(plan.arch, Arch::Hy3);
        assert_eq!(plan.layers.len(), 2);
        assert_eq!(plan.mtp_blocks.len(), 1);
        assert!(matches!(plan.layers[0].mlp, MlpPlan::Dense(_)));

        for layer in [&plan.layers[1], &plan.mtp_blocks[0].layer] {
            let MlpPlan::Moe(moe) = &layer.mlp else {
                panic!("Hy3 routed layer compiled as dense")
            };
            assert!(matches!(
                moe.router,
                RouterPlan::Sigmoid {
                    normalize_selected: true,
                    scaling_factor: 2.826,
                    selection_bias: true,
                }
            ));
            assert_eq!(moe.shared.as_ref().unwrap().intermediate_size, 8);
            let crate::model_plan::AttentionPlan::Full(attention) = &layer.attention else {
                panic!("Hy3 layer did not compile as full attention")
            };
            assert_eq!(attention.qk_norm, TensorPresence::Required);
            assert_eq!(attention.output_gate, AttentionGateKind::None);
        }

        assert_eq!(
            plan.sampling_defaults,
            Some(SamplingDefaultsPlan {
                temperature: 0.9,
                top_p: 1.0,
            })
        );
    }

    fn census_for(contract: &TensorContract) -> Vec<TensorCensusEntry> {
        let mut census = Vec::new();
        for requirement in &contract.requirements {
            if !requirement.required || requirement.names.is_empty() {
                continue;
            }
            let names = match requirement.match_mode {
                TensorMatch::OneOf => &requirement.names[..1],
                TensorMatch::All => requirement.names.as_slice(),
            };
            let storage = match requirement.quant {
                QuantConstraint::FloatOnly | QuantConstraint::ExactFloat(_) => {
                    StorageLayout::Float(FloatType::Bf16)
                }
                QuantConstraint::Nvfp4 => StorageLayout::Quantized(QuantLayout {
                    format: "NVFP4".to_string(),
                    block_shape: vec![16],
                    auxiliaries: Vec::new(),
                }),
                other => panic!("unexpected Hy3 NVFP4 contract class {other:?}"),
            };
            for name in names {
                census.push(TensorCensusEntry {
                    name: name.clone(),
                    shape: requirement.shape.clone(),
                    storage: storage.clone(),
                    physical_bytes: 1,
                });
            }
        }
        census
    }

    #[test]
    fn nvfp4_profile_quantizes_every_routed_expert_including_mtp() {
        assert_eq!(NVFP4_PACK.support, Some(NativeSupport::NativeQualified));
        assert_eq!(
            NVFP4_PACK.checkpoint_parity,
            Some(CheckpointParityGate {
                max_abs: 1.0,
                max_rel: 0.0,
                require_argmax: true,
            })
        );
        let plan = tiny_plan().unwrap();
        let all_experts = nvfp4_tensor_schema(
            &ModelConfig::from_hf(&HfConfig::parse(
                r#"{"model_type":"hy_v3","num_hidden_layers":2,
                "num_nextn_predict_layers":1,"hidden_size":8,
                "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
                "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32,
                "first_k_dense_replace":1,"num_experts":4,"num_experts_per_tok":2,
                "moe_intermediate_size":8,"num_shared_experts":1,
                "moe_router_use_sigmoid":true,"moe_router_enable_expert_bias":true,
                "route_norm":true,"router_scaling_factor":2.826,"qk_norm":true}"#,
            )),
            &plan,
            CheckpointDialect::HfSafetensors,
            ContractOptions {
                output_head: OutputHead::Separate,
            },
        )
        .unwrap();
        let mut census = census_for(&all_experts);
        all_experts.bind(&census).unwrap();

        let expert = census
            .iter_mut()
            .position(|entry| entry.name == "model.layers.1.mlp.experts.0.gate_proj.weight")
            .unwrap();
        census[expert].storage = StorageLayout::Float(FloatType::Bf16);
        assert!(all_experts.bind(&census).is_err());
        census[expert].storage = StorageLayout::Quantized(QuantLayout {
            format: "NVFP4".to_string(),
            block_shape: vec![16],
            auxiliaries: Vec::new(),
        });

        let attention = census
            .iter_mut()
            .position(|entry| entry.name == "model.layers.1.self_attn.q_proj.weight")
            .unwrap();
        census[attention].storage = StorageLayout::Quantized(QuantLayout {
            format: "NVFP4".to_string(),
            block_shape: vec![16],
            auxiliaries: Vec::new(),
        });
        assert!(all_experts.bind(&census).is_err());
        let mtp_experts = all_experts
            .requirements
            .iter()
            .find(|requirement| {
                matches!(
                    requirement.id,
                    TensorId::Layer {
                        index: 2,
                        tensor: LayerTensor::MoeExpertGateBank
                    }
                )
            })
            .unwrap();
        assert_eq!(mtp_experts.quant, QuantConstraint::Nvfp4);
    }
}
