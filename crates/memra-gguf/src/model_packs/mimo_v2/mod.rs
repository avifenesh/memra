//! MiMo V2.6 tensor schema slices. Only the explicit inspection profile is
//! registered; automatic serving selection remains closed.

pub(crate) mod audio;
pub(crate) mod mint_headers;
pub(crate) mod mtp;
pub(crate) mod vision;

use crate::config::{HfConfig, ModelConfig};
use crate::model_plan::{ModelPlan, MoeMlpPlan, PlanCompileError};
use crate::safetensors::StInfo;
use crate::source::census_from_safetensors_headers;
use crate::tensor_contract::{
    BoundTensorContract, CheckpointDialect, ContractOptions, ExpertTensor, LayerTensor,
    QuantConstraint, TensorContract, TensorContractError, TensorId, TensorMatch, TensorOwner,
    TensorRequirement, TensorTransform,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

use super::{
    ConfigLayout, Gate, ModelPack, OutputHeadContract, TemplateContract, TensorConsumption,
    TokenizerSource, canonical_plan,
};

const PINNED_MINT_HEADER_DIGEST: &str =
    "00403ccacf38567327ec41b4cb0cbd9eade5b830094191dce7e90b24b8efe45a";
const PINNED_SOURCE_HEADER_DIGEST: &str =
    "5ebbdd27e45716b805fc2bdf115c8345b4bfc03c6012b860f76b6b222c758aee";

/// Explicit inspection profile. It never participates in automatic family selection.
pub static MINT_PROFILE: ModelPack = ModelPack {
    family: "mimo_v2_mint",
    output_head: OutputHeadContract::SeparateHead,
    tensor_consumption: TensorConsumption::Refuse,
    aliases: &["mimo_v2_mint", "mimo-v2.6-nvfp4"],
    config_layout: ConfigLayout::Flat,
    tokenizer_sources: &[TokenizerSource::TokenizerJson],
    template: TemplateContract::ArtifactRequired,
    support: None,
    gates: &[
        Gate::Config,
        Gate::TokenizerTemplate,
        Gate::TensorCensus,
        Gate::TinyParity,
        Gate::CheckpointParity,
        Gate::RewriteParity,
        Gate::Serve,
    ],
    checkpoint_parity: None,
    matches_config: |config| config.arch == crate::config::Arch::MiMoV2 && config.mimo.is_some(),
    plan_builder: canonical_plan,
    tensor_schema: mint_tensor_schema,
    tiny_plan: None,
};

/// Text-only reference fixture for the two MiMo attention programs. This does
/// not cover the checkpoint's vision, audio, or separate MTP artifacts and
/// does not change the source profile's unsupported serving state.
pub fn tiny_text_plan() -> Result<ModelPlan, PlanCompileError> {
    canonical_plan(&ModelConfig::from_hf(&HfConfig::parse(
        r#"{"model_type":"mimo_v2","num_hidden_layers":2,
        "hidden_size":32,"vocab_size":32,"max_position_embeddings":32,
        "num_attention_heads":4,"num_key_value_heads":1,
        "head_dim":12,"v_head_dim":8,"partial_rotary_factor":0.33333334,
        "swa_num_attention_heads":4,"swa_num_key_value_heads":2,
        "swa_head_dim":12,"swa_v_head_dim":8,
        "hybrid_layer_pattern":[0,1],"moe_layer_freq":[0,1],
        "intermediate_size":64,"moe_intermediate_size":16,
        "n_routed_experts":4,"num_experts_per_tok":2,
        "scoring_func":"sigmoid","topk_method":"noaux_tc",
        "norm_topk_prob":true,"n_group":1,"topk_group":1,
        "moe_router_dtype":"bfloat16","num_nextn_predict_layers":0,
        "attention_projection_layout":"fused_qkv","attention_value_scale":0.707,
        "add_full_attention_sink_bias":false,
        "add_swa_attention_sink_bias":true,
        "attention_chunk_size":4,"sliding_window":4,
        "rope_theta":10000000.0,"swa_rope_theta":10000.0,
        "layernorm_epsilon":0.000001,"hidden_act":"silu"}"#,
    )))
}

/// The quality-tested Xiaomi MXFP4 checkpoint is an explicit inspection profile.
/// It remains outside automatic serving selection until native parity passes.
pub static SOURCE_PROFILE: ModelPack = ModelPack {
    family: "mimo_v2_source",
    output_head: OutputHeadContract::SeparateHead,
    tensor_consumption: TensorConsumption::Refuse,
    aliases: &["mimo_v2_source", "mimo-v2.6-mxfp4"],
    config_layout: ConfigLayout::Flat,
    tokenizer_sources: &[TokenizerSource::TokenizerJson],
    template: TemplateContract::ArtifactRequired,
    support: None,
    gates: &[
        Gate::Config,
        Gate::TokenizerTemplate,
        Gate::TensorCensus,
        Gate::TinyParity,
        Gate::CheckpointParity,
        Gate::RewriteParity,
        Gate::Serve,
    ],
    checkpoint_parity: None,
    matches_config: |config| config.arch == crate::config::Arch::MiMoV2 && config.mimo.is_some(),
    plan_builder: canonical_plan,
    tensor_schema: source_tensor_schema,
    tiny_plan: None,
};

/// GGUF-style request names for the HF source's text trunk. This is a loader
/// address map, not an accepted GGUF tensor contract. Modalities and separate
/// MTP remain unavailable to this text loader and return None.
pub fn source_ggml_name(id: &TensorId) -> Option<String> {
    match id {
        TensorId::TokenEmbedding => Some("token_embd.weight".into()),
        TensorId::OutputNorm => Some("output_norm.weight".into()),
        TensorId::OutputProjection => Some("output.weight".into()),
        TensorId::Layer { index, tensor } => {
            let suffix = match tensor {
                LayerTensor::PreAttentionNorm => "attn_norm.weight",
                LayerTensor::FusedQkv => "attn_qkv.weight",
                LayerTensor::AttentionOutput => "attn_output.weight",
                LayerTensor::AttentionSink => "attn_sink.bias",
                LayerTensor::PreMlpNorm => "ffn_norm.weight",
                LayerTensor::MlpGate => "ffn_gate.weight",
                LayerTensor::MlpUp => "ffn_up.weight",
                LayerTensor::MlpDown => "ffn_down.weight",
                LayerTensor::MoeRouter => "ffn_gate_inp.weight",
                LayerTensor::MoeRouterBias => "exp_probs_b.bias",
                _ => return None,
            };
            Some(format!("blk.{index}.{suffix}"))
        }
        TensorId::Expert {
            layer,
            expert,
            tensor,
        } => {
            let projection = match tensor {
                ExpertTensor::Gate => "gate",
                ExpertTensor::Up => "up",
                ExpertTensor::Down => "down",
            };
            Some(format!("blk.{layer}.ffn_{projection}_exps.{expert}.weight"))
        }
        _ => None,
    }
}

#[allow(clippy::result_large_err)] // the contract error names the exact rejected tensor
fn mint_tensor_schema(
    config: &ModelConfig,
    plan: &ModelPlan,
    dialect: CheckpointDialect,
    options: ContractOptions,
) -> Result<TensorContract, TensorContractError> {
    if config.arch != crate::config::Arch::MiMoV2
        || plan.arch != crate::config::Arch::MiMoV2
        || dialect != CheckpointDialect::HfSafetensors
    {
        return Err(TensorContractError::UnsupportedPlanOperation {
            operation: "MiMo mint inspection requires a safetensors MiMo plan",
        });
    }
    let mimo = config
        .mimo
        .as_ref()
        .ok_or(TensorContractError::UnsupportedPlanOperation {
            operation: "MiMo config has no family geometry",
        })?;
    let audio =
        mimo.audio_config
            .as_ref()
            .ok_or(TensorContractError::UnsupportedPlanOperation {
                operation: "MiMo config has no audio geometry",
            })?;
    let vision =
        mimo.vision_config
            .as_ref()
            .ok_or(TensorContractError::UnsupportedPlanOperation {
                operation: "MiMo config has no vision geometry",
            })?;
    let mut contract = TensorContract::for_plan(plan, dialect, options)?;
    contract
        .requirements
        .extend(mtp::mint_mtp_requirements(config)?);
    contract
        .requirements
        .extend(audio::pinned_audio_requirements(audio, config.n_embd)?);
    contract
        .requirements
        .extend(vision::pinned_vision_requirements(vision, config.n_embd)?);
    let mut ids = BTreeSet::new();
    for row in &contract.requirements {
        if !ids.insert(&row.id) {
            return Err(TensorContractError::DuplicateTensorId { id: row.id.clone() });
        }
    }
    Ok(contract)
}

#[allow(clippy::result_large_err)] // the contract error names the exact rejected tensor
fn source_tensor_schema(
    config: &ModelConfig,
    plan: &ModelPlan,
    dialect: CheckpointDialect,
    options: ContractOptions,
) -> Result<TensorContract, TensorContractError> {
    let mut contract = mint_tensor_schema(config, plan, dialect, options)?;
    for requirement in &mut contract.requirements {
        if matches!(&requirement.id, TensorId::Expert { .. }) {
            requirement.quant = QuantConstraint::Mxfp4;
            let stem = requirement.names[0].strip_suffix(".weight").unwrap();
            requirement.auxiliaries = Some(vec![format!("{stem}.weight_scale")]);
        }
    }
    Ok(contract)
}

fn header_digest(headers: &BTreeMap<String, StInfo>) -> String {
    let mut digest = Sha256::new();
    for (name, info) in headers {
        digest.update(name.as_bytes());
        digest.update([0]);
        digest.update(info.dtype.as_bytes());
        digest.update([0]);
        for (index, dim) in info.shape.iter().enumerate() {
            if index != 0 {
                digest.update(b",");
            }
            digest.update(dim.to_string().as_bytes());
        }
        digest.update(b"\n");
    }
    format!("{:x}", digest.finalize())
}

/// Bind the full pinned mint's tensor headers without admitting native execution.
///
/// The result proves names, shapes and storage classes at the header boundary.
/// It says nothing about weight values, numerical parity, throughput or serving.
pub fn inspect_pinned_mint_headers(
    config: &ModelConfig,
    headers: &BTreeMap<String, StInfo>,
) -> Result<BoundTensorContract, String> {
    let plan = ModelPlan::compile(config).map_err(|error| format!("{error:?}"))?;
    let contract = mint_tensor_schema(
        config,
        &plan,
        CheckpointDialect::HfSafetensors,
        ContractOptions::default(),
    )
    .map_err(|error| format!("{error:?}"))?;
    mint_headers::verify_mint_expert_headers(&plan, headers)?;
    let census = census_from_safetensors_headers(headers)?;
    let entries = census
        .tensors
        .into_iter()
        .map(|record| record.entry)
        .collect::<Vec<_>>();
    let bound = contract
        .bind(&entries)
        .map_err(|error| format!("{error:?}"))?;
    let digest = header_digest(headers);
    if digest != PINNED_MINT_HEADER_DIGEST {
        return Err(format!(
            "MiMo mint header digest changed: got {digest}, expected {PINNED_MINT_HEADER_DIGEST}"
        ));
    }
    Ok(bound)
}

/// Bind the quality-tested Xiaomi MXFP4 source headers without admitting load.
pub fn inspect_pinned_source_headers(
    config: &ModelConfig,
    headers: &BTreeMap<String, StInfo>,
) -> Result<BoundTensorContract, String> {
    let plan = ModelPlan::compile(config).map_err(|error| format!("{error:?}"))?;
    let contract = source_tensor_schema(
        config,
        &plan,
        CheckpointDialect::HfSafetensors,
        ContractOptions::default(),
    )
    .map_err(|error| format!("{error:?}"))?;
    let census = census_from_safetensors_headers(headers)?;
    let entries = census
        .tensors
        .into_iter()
        .map(|record| record.entry)
        .collect::<Vec<_>>();
    let bound = contract
        .bind(&entries)
        .map_err(|error| format!("{error:?}"))?;
    let digest = header_digest(headers);
    if digest != PINNED_SOURCE_HEADER_DIGEST {
        return Err(format!(
            "MiMo source header digest changed: got {digest}, expected {PINNED_SOURCE_HEADER_DIGEST}"
        ));
    }
    Ok(bound)
}

pub(crate) fn mint_expert_requirements(
    plan: &ModelPlan,
    layer: u32,
    moe: &MoeMlpPlan,
) -> Vec<TensorRequirement> {
    let mut requirements = Vec::with_capacity(moe.expert_count as usize * 3);
    for expert in 0..moe.expert_count {
        for (tensor, projection, output, input) in [
            (
                ExpertTensor::Gate,
                "gate",
                moe.expert_intermediate_size,
                plan.hidden_size,
            ),
            (
                ExpertTensor::Up,
                "up",
                moe.expert_intermediate_size,
                plan.hidden_size,
            ),
            (
                ExpertTensor::Down,
                "down",
                plan.hidden_size,
                moe.expert_intermediate_size,
            ),
        ] {
            let name = crate::hf_mapping::hf_expert_name(layer, expert, projection, &plan.arch);
            let stem = name.strip_suffix(".weight").unwrap();
            let auxiliaries = vec![
                format!("{stem}.weight_scale"),
                format!("{stem}.weight_scale_2"),
                format!("{stem}.input_scale"),
            ];
            requirements.push(TensorRequirement {
                id: TensorId::Expert {
                    layer,
                    expert,
                    tensor,
                },
                names: vec![name],
                match_mode: TensorMatch::OneOf,
                shape: vec![output as u64, input as u64],
                owner: TensorOwner::Layer(layer),
                transform: TensorTransform::Identity,
                quant: QuantConstraint::Nvfp4,
                auxiliaries: Some(auxiliaries),
                required: true,
            });
        }
    }
    requirements
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HfConfig, ModelConfig};
    use crate::execution_manifest::{RewriteSurface, execution_rewrites};
    use crate::hf_mapping::{HfTarget, resolve_ggml};
    use crate::model_plan::{AttentionPlan, MlpPlan, OperationKind, TensorPresence};
    use crate::tensor_contract::{
        CheckpointDialect, ContractOptions, QuantLayout, StorageLayout, TensorCensusEntry,
        TensorContract, TensorContractError,
    };

    #[test]
    fn source_tiny_text_plan_covers_both_attention_programs() {
        let plan = tiny_text_plan().unwrap();
        assert_eq!(plan.arch, crate::config::Arch::MiMoV2);
        assert_eq!(plan.layers.len(), 2);
        assert!(matches!(plan.layers[0].attention, AttentionPlan::Full(_)));
        assert!(matches!(
            plan.layers[1].attention,
            AttentionPlan::SlidingWindow { window: 4, .. }
        ));
        assert!(matches!(plan.layers[0].mlp, MlpPlan::Dense(_)));
        assert!(matches!(plan.layers[1].mlp, MlpPlan::Moe(_)));
        let AttentionPlan::SlidingWindow { attention, .. } = &plan.layers[1].attention else {
            unreachable!()
        };
        let mimo = attention.mimo_math.unwrap();
        assert_eq!(mimo.value_scale_before_cache.to_bits(), 0.707f32.to_bits());
        assert_eq!(mimo.sink, TensorPresence::Required);
        assert!(plan.vision.is_none());
        assert!(plan.speech.is_none());
        assert!(plan.mtp_blocks.is_empty());
        assert!(SOURCE_PROFILE.support.is_none());
        assert!(SOURCE_PROFILE.compile_tiny_plan().is_err());
    }

    #[test]
    fn mimo_attention_math_blocks_generic_tuned_rewrites() {
        assert!(crate::op_registry::surfaces(OperationKind::MiMoAttentionMath).is_none());
        let full_config =
            ModelConfig::from_hf(&HfConfig::parse(include_str!("fixtures/config.json")));
        for plan in [
            tiny_text_plan().unwrap(),
            SOURCE_PROFILE.compile_plan(&full_config).unwrap(),
        ] {
            assert!(
                plan.trunk_operations()
                    .contains(&OperationKind::MiMoAttentionMath)
            );
            let rewrites = execution_rewrites(&plan);
            for surface in [
                RewriteSurface::DecodeEager,
                RewriteSurface::DecodeBatch,
                RewriteSurface::DecodeGraph,
                RewriteSurface::Pipeline,
            ] {
                let rewrite = rewrites
                    .iter()
                    .find(|rewrite| rewrite.surface == surface)
                    .unwrap();
                assert!(
                    rewrite.blockers.contains(&OperationKind::MiMoAttentionMath),
                    "{surface:?} lost the MiMo math blocker"
                );
            }
        }
    }

    #[test]
    fn source_text_loader_addresses_match_pinned_hf_contract() {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!("fixtures/config.json")));
        let plan = SOURCE_PROFILE.compile_plan(&config).unwrap();
        let contract = SOURCE_PROFILE
            .compile_tensor_contract(
                &config,
                &plan,
                CheckpointDialect::HfSafetensors,
                ContractOptions::default(),
            )
            .unwrap();
        let mut checked = 0usize;
        for requirement in &contract.requirements {
            if !matches!(
                requirement.id,
                TensorId::TokenEmbedding
                    | TensorId::OutputNorm
                    | TensorId::OutputProjection
                    | TensorId::Layer { .. }
                    | TensorId::Expert { .. }
            ) {
                continue;
            }
            let ggml = source_ggml_name(&requirement.id)
                .unwrap_or_else(|| panic!("{:?} has no source loader address", requirement.id));
            let Some(HfTarget::Plain(hf)) = resolve_ggml(&ggml, &config) else {
                panic!("{ggml} has no plain HF source address");
            };
            assert!(
                requirement.names.contains(&hf),
                "{:?}: {ggml} resolved to {hf}, outside {:?}",
                requirement.id,
                requirement.names
            );
            checked += 1;
        }
        assert!(checked > 36_000, "source text census unexpectedly small");
        assert!(
            SOURCE_PROFILE
                .compile_tensor_contract(
                    &config,
                    &plan,
                    CheckpointDialect::Gguf,
                    ContractOptions::default()
                )
                .is_err()
        );
    }

    #[test]
    fn mint_profile_is_explicit_and_cannot_admit_native_load() {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!("fixtures/config.json")));
        let profile = crate::model_packs::by_alias("mimo_v2_mint").unwrap();
        assert_eq!(profile.family, "mimo_v2_mint");
        assert!(profile.support.is_none());
        assert!(profile.matches_config(&config));
        assert!(crate::model_packs::for_config(&config).is_none());
        assert!(crate::model_packs::compile_for_load(&config).is_err());
        let plan = profile.compile_plan(&config).unwrap();
        let contract = profile
            .compile_tensor_contract(
                &config,
                &plan,
                CheckpointDialect::HfSafetensors,
                ContractOptions::default(),
            )
            .unwrap();
        assert!(contract.requirements.len() > 36_922);
        assert!(
            profile
                .compile_tensor_contract(
                    &config,
                    &plan,
                    CheckpointDialect::Gguf,
                    ContractOptions::default(),
                )
                .is_err()
        );
    }

    #[test]
    fn source_profile_uses_mxfp4_and_keeps_native_load_closed() {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!("fixtures/config.json")));
        let profile = crate::model_packs::by_alias("mimo_v2_source").unwrap();
        assert_eq!(profile.family, "mimo_v2_source");
        assert!(profile.support.is_none());
        assert!(profile.matches_config(&config));
        assert!(crate::model_packs::for_config(&config).is_none());
        assert!(crate::model_packs::compile_for_load(&config).is_err());
        let plan = profile.compile_plan(&config).unwrap();
        let contract = profile
            .compile_tensor_contract(
                &config,
                &plan,
                CheckpointDialect::HfSafetensors,
                ContractOptions::default(),
            )
            .unwrap();
        let expert = contract
            .requirements
            .iter()
            .find(|row| {
                row.id
                    == TensorId::Expert {
                        layer: 1,
                        expert: 0,
                        tensor: ExpertTensor::Gate,
                    }
            })
            .unwrap();
        assert_eq!(expert.quant, QuantConstraint::Mxfp4);
        assert_eq!(
            expert.auxiliaries,
            Some(vec![
                "model.layers.1.mlp.experts.0.gate_proj.weight_scale".to_owned()
            ])
        );
        assert!(
            profile
                .compile_tensor_contract(
                    &config,
                    &plan,
                    CheckpointDialect::Gguf,
                    ContractOptions::default(),
                )
                .is_err()
        );
    }

    #[test]
    fn pinned_mint_owns_every_routed_expert_projection_separately() {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!("fixtures/config.json")));
        let plan = ModelPlan::compile(&config).unwrap();
        let contract = TensorContract::for_plan(
            &plan,
            CheckpointDialect::HfSafetensors,
            ContractOptions::default(),
        )
        .unwrap();
        let experts: Vec<_> = contract
            .requirements
            .iter()
            .filter(|requirement| matches!(&requirement.id, TensorId::Expert { .. }))
            .collect();
        assert_eq!(experts.len(), 47 * 256 * 3);
        for (tensor, projection, shape) in [
            (ExpertTensor::Gate, "gate", vec![2_048, 4_096]),
            (ExpertTensor::Up, "up", vec![2_048, 4_096]),
            (ExpertTensor::Down, "down", vec![4_096, 2_048]),
        ] {
            let requirement = experts
                .iter()
                .copied()
                .find(|requirement| {
                    requirement.id
                        == TensorId::Expert {
                            layer: 1,
                            expert: 0,
                            tensor,
                        }
                })
                .unwrap();
            let stem = format!("model.layers.1.mlp.experts.0.{projection}_proj");
            assert_eq!(requirement.names, vec![format!("{stem}.weight")]);
            assert_eq!(requirement.shape, shape);
            assert_eq!(requirement.quant, QuantConstraint::Nvfp4);
            assert_eq!(
                requirement.auxiliaries,
                Some(vec![
                    format!("{stem}.weight_scale"),
                    format!("{stem}.weight_scale_2"),
                    format!("{stem}.input_scale"),
                ])
            );
            assert!(requirement.required);
        }
    }

    #[test]
    fn mint_expert_refuses_source_format_and_missing_input_scale() {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!("fixtures/config.json")));
        let plan = ModelPlan::compile(&config).unwrap();
        let requirement = mint_expert_requirements(
            &plan,
            1,
            match &plan.layers[1].mlp {
                crate::model_plan::MlpPlan::Moe(moe) => moe,
                _ => panic!("layer 1 must be MoE"),
            },
        )
        .into_iter()
        .next()
        .unwrap();
        let contract = TensorContract {
            dialect: CheckpointDialect::HfSafetensors,
            requirements: vec![requirement.clone()],
        };
        let mut entry = TensorCensusEntry {
            name: requirement.names[0].clone(),
            shape: requirement.shape.clone(),
            storage: StorageLayout::Quantized(QuantLayout {
                format: "MXFP4".to_owned(),
                block_shape: vec![32],
                auxiliaries: vec![requirement.auxiliaries.as_ref().unwrap()[0].clone()],
            }),
            physical_bytes: 1,
        };
        assert!(matches!(
            contract.bind(&[entry.clone()]),
            Err(TensorContractError::QuantLayoutMismatch { .. })
        ));
        if let StorageLayout::Quantized(layout) = &mut entry.storage {
            layout.format = "NVFP4".to_owned();
            layout.block_shape = vec![16];
        }
        assert!(matches!(
            contract.bind(&[entry.clone()]),
            Err(TensorContractError::AuxiliaryLayoutMismatch { .. })
        ));
        if let StorageLayout::Quantized(layout) = &mut entry.storage {
            layout.auxiliaries = requirement.auxiliaries.unwrap();
        }
        contract.bind(&[entry]).unwrap();
    }
}
