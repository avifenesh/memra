use super::*;
use crate::dsv4::{TensorSpec, expected_census};
use crate::model_plan::{DrafterPlan, DsparkPlan};
use crate::tensor_contract::{
    DsparkTensor, ExpertTensor, FloatType, LayerTensor, QuantConstraint, TensorId, TensorMatch,
    TensorOwner, TensorRequirement, TensorTransform,
};

pub static PACK: ModelPack = ModelPack {
    family: "deepseek_v4",
    aliases: &["deepseek_v4", "deepseek-v4", "deepseek_v4_preview"],
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
    matches_config: |config| {
        matches!(config.arch, Arch::DeepSeekV4)
            && config.dsv4.is_some()
            && config.nextn_predict_layers != 3
    },
    plan_builder,
    tensor_schema,
    tiny_plan: None,
};

pub static DSPARK_PACK: ModelPack = ModelPack {
    family: "deepseek_v4_dspark",
    aliases: &["deepseek_v4_dspark", "deepseek-v4-dspark"],
    config_layout: ConfigLayout::Flat,
    tokenizer_sources: &[TokenizerSource::TokenizerJson],
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
    matches_config: |config| {
        matches!(config.arch, Arch::DeepSeekV4)
            && config.dsv4.is_some()
            && config.nextn_predict_layers == 3
    },
    plan_builder,
    tensor_schema,
    tiny_plan: Some(tiny_plan),
};

fn tiny_plan() -> Result<ModelPlan, PlanCompileError> {
    let config = ModelConfig::from_hf(&crate::config::HfConfig::parse(
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
        "index_topk":16,"compress_ratios":[0,4,0,0,0],
        "compress_rope_theta":160000,"sliding_window":128,"swiglu_limit":10.0,
        "rope_scaling":{"factor":4,"beta_fast":32,"beta_slow":1,
        "original_max_position_embeddings":1024}}"#,
    ));
    let mut plan = plan_builder(&config)?;
    let Some(DrafterPlan::Dspark(dspark)) = plan.drafter.as_mut() else {
        return Err(PlanCompileError::MissingTinyFixture {
            pack: DSPARK_PACK.family,
        });
    };
    dspark.block_size = 3;
    dspark.noise_token_id = 31;
    dspark.target_layer_ids = vec![1];
    dspark.markov_rank = 8;
    Ok(plan)
}

fn plan_builder(config: &ModelConfig) -> Result<ModelPlan, PlanCompileError> {
    let mut plan = canonical_plan(config)?;
    if config.nextn_predict_layers == 3 {
        let blocks = plan.mtp_blocks.drain(..).map(|block| block.layer).collect();
        plan.drafter = Some(DrafterPlan::Dspark(DsparkPlan {
            block_size: 5,
            noise_token_id: 128_799,
            target_layer_ids: vec![40, 41, 42],
            markov_rank: 256,
            blocks,
        }));
    }
    Ok(plan)
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
            operation: "deepseek_v4 non-safetensors schema",
        });
    }
    let expected = expected_for_pack(config);
    let mut requirements = Vec::new();
    for (name, spec) in &expected {
        if quant_primary(name, &expected).is_some() {
            continue;
        }
        let stem = name.strip_suffix(".weight");
        let auxiliaries: Vec<String> = stem
            .map(|stem| {
                [
                    format!("{stem}.weight_scale"),
                    format!("{stem}.weight_scale_2"),
                    format!("{stem}.input_scale"),
                    format!("{stem}.scale"),
                ]
                .into_iter()
                .filter(|candidate| expected.contains_key(candidate))
                .collect()
            })
            .unwrap_or_default();
        let mut shape = spec.shape.clone();
        let quant = match spec.dtype {
            "BF16" => QuantConstraint::ExactFloat(FloatType::Bf16),
            "F32" => QuantConstraint::ExactFloat(FloatType::F32),
            "I64" => QuantConstraint::I64,
            "U8" => {
                *shape.last_mut().expect("NVFP4 weight rank") *= 2;
                QuantConstraint::Nvfp4
            }
            "I8" => {
                *shape.last_mut().expect("MXFP4 weight rank") *= 2;
                QuantConstraint::Mxfp4
            }
            "F8_E4M3" => QuantConstraint::Fp8Block128,
            other => {
                return Err(TensorContractError::UnsupportedPlanOperation {
                    operation: match other {
                        "F8_E8M0" => "orphan deepseek_v4 E8M0 scale",
                        _ => "unknown deepseek_v4 dtype",
                    },
                });
            }
        };
        let id = match name.as_str() {
            "embed.weight" => TensorId::TokenEmbedding,
            "hc_head_base" => TensorId::HyperHeadBase,
            "hc_head_fn" => TensorId::HyperHeadFunction,
            "hc_head_scale" => TensorId::HyperHeadScale,
            "head.weight" => TensorId::OutputProjection,
            "norm.weight" => TensorId::OutputNorm,
            _ => semantic_tensor_id(name, plan),
        };
        requirements.push(TensorRequirement {
            id,
            names: vec![name.clone()],
            match_mode: TensorMatch::OneOf,
            shape,
            owner: tensor_owner(name),
            transform: TensorTransform::Identity,
            quant,
            auxiliaries: Some(auxiliaries),
            required: true,
        });
    }
    Ok(TensorContract {
        dialect,
        requirements,
    })
}

fn expected_for_pack(config: &ModelConfig) -> std::collections::BTreeMap<String, TensorSpec> {
    let mut expected = expected_census(config);
    if config.nextn_predict_layers != 3 {
        return expected;
    }
    // The 0731 serving artifact is the source-exact house mint: only the 43 trunk expert
    // banks were losslessly recoded to NVFP4, `input_scale` was deliberately omitted because
    // the native engine quantizes activations dynamically, and all three DSpark expert banks
    // remain in DeepSeek's MXFP4 container. Keep this contract byte-faithful to that artifact;
    // NVIDIA's later 0731 mint adds calibrated trunk input_scale tensors but makes the same
    // `mtp.*` exclusion.
    expected.retain(|name, _| {
        let trunk_input_scale = name.starts_with("layers.")
            && name.contains(".ffn.experts.")
            && name.ends_with(".input_scale");
        let mtp_expert = name.starts_with("mtp.") && name.contains(".ffn.experts.");
        let old_glue = name.starts_with("mtp.")
            && (name.contains(".e_proj")
                || name.contains(".h_proj")
                || name.ends_with(".enorm.weight")
                || name.ends_with(".hnorm.weight")
                || name.ends_with(".norm.weight")
                || name.contains(".hc_head_"));
        !(trunk_input_scale || mtp_expert || old_glue)
    });
    let dsv4 = config.dsv4.as_ref().expect("DSV4 config");
    let moe = config.moe.as_ref().expect("DSV4 MoE config");
    let hidden = config.n_embd as u64;
    let vocab = config.n_vocab as u64;
    let experts = moe.expert_count as u64;
    let expert_ff = moe.expert_ff_length as u64;
    let fp8 = |map: &mut std::collections::BTreeMap<String, TensorSpec>,
               stem: String,
               output: u64,
               input: u64| {
        map.insert(
            format!("{stem}.weight"),
            TensorSpec {
                dtype: "F8_E4M3",
                shape: vec![output, input],
            },
        );
        map.insert(
            format!("{stem}.scale"),
            TensorSpec {
                dtype: "F8_E8M0",
                shape: vec![output / 128, input / 128],
            },
        );
    };
    // Rebuild the DSpark expert schema in its source MXFP4 layout. The generic DSV4 census
    // describes the older one-block NextN program, so the three-block DSpark pack replaces it.
    for block in 0..3u32 {
        for expert in 0..experts {
            for (projection, output, input) in [
                ("w1", expert_ff, hidden),
                ("w2", hidden, expert_ff),
                ("w3", expert_ff, hidden),
            ] {
                let stem = format!("mtp.{block}.ffn.experts.{expert}.{projection}");
                expected.insert(
                    format!("{stem}.weight"),
                    TensorSpec {
                        dtype: "I8",
                        shape: vec![output, input / 2],
                    },
                );
                expected.insert(
                    format!("{stem}.scale"),
                    TensorSpec {
                        dtype: "F8_E8M0",
                        shape: vec![output, input / 32],
                    },
                );
            }
        }
    }
    fp8(&mut expected, "mtp.0.main_proj".into(), hidden, 3 * hidden);
    expected.insert(
        "mtp.0.main_norm.weight".into(),
        TensorSpec {
            dtype: "BF16",
            shape: vec![hidden],
        },
    );
    let last = 2;
    expected.insert(
        format!("mtp.{last}.norm.weight"),
        TensorSpec {
            dtype: "BF16",
            shape: vec![hidden],
        },
    );
    expected.insert(
        format!("mtp.{last}.markov_head.markov_w1.weight"),
        TensorSpec {
            dtype: "BF16",
            shape: vec![vocab, 256],
        },
    );
    expected.insert(
        format!("mtp.{last}.markov_head.markov_w2.weight"),
        TensorSpec {
            dtype: "BF16",
            shape: vec![vocab, 256],
        },
    );
    expected.insert(
        format!("mtp.{last}.confidence_head.proj.weight"),
        TensorSpec {
            dtype: "BF16",
            shape: vec![1, hidden + 256],
        },
    );
    let hc_width = dsv4.hc_mult as u64 * hidden;
    for (name, shape) in [
        ("hc_head_base", vec![dsv4.hc_mult as u64]),
        ("hc_head_fn", vec![dsv4.hc_mult as u64, hc_width]),
        ("hc_head_scale", vec![1]),
    ] {
        expected.insert(
            format!("mtp.{last}.{name}"),
            TensorSpec {
                dtype: "F32",
                shape,
            },
        );
    }
    expected
}

fn quant_primary(
    name: &str,
    expected: &std::collections::BTreeMap<String, crate::dsv4::TensorSpec>,
) -> Option<String> {
    for suffix in [".weight_scale", ".weight_scale_2", ".input_scale", ".scale"] {
        if let Some(stem) = name.strip_suffix(suffix) {
            let primary = format!("{stem}.weight");
            if expected.contains_key(&primary) {
                return Some(primary);
            }
        }
    }
    None
}

fn tensor_owner(name: &str) -> TensorOwner {
    let mut parts = name.split('.');
    match (parts.next(), parts.next()) {
        (Some("layers"), Some(index)) => index
            .parse()
            .map(TensorOwner::Layer)
            .unwrap_or(TensorOwner::Global),
        (Some("mtp"), Some(index)) => index
            .parse()
            .map(TensorOwner::Mtp)
            .unwrap_or(TensorOwner::Global),
        _ => TensorOwner::Global,
    }
}

fn semantic_key(name: &str) -> String {
    if let Some(suffix) = name.strip_prefix("layers.") {
        format!("trunk.{suffix}")
    } else {
        name.to_string()
    }
}

fn semantic_tensor_id(name: &str, plan: &ModelPlan) -> TensorId {
    if let Some(rest) = name.strip_prefix("layers.")
        && let Some((index, suffix)) = rest.split_once('.')
        && let Ok(index) = index.parse()
    {
        if let Some((expert, tensor)) = expert_tensor_for_suffix(suffix) {
            return TensorId::Expert {
                layer: index,
                expert,
                tensor,
            };
        }
        if let Some(tensor) = layer_tensor_for_suffix(suffix) {
            return TensorId::Layer { index, tensor };
        }
    }
    if let Some(rest) = name.strip_prefix("mtp.")
        && let Some((stage, suffix)) = rest.split_once('.')
        && let Ok(stage) = stage.parse::<usize>()
    {
        let Some(DrafterPlan::Dspark(dspark)) = plan.drafter.as_ref() else {
            return TensorId::Family {
                family: "deepseek_v4",
                key: semantic_key(name),
            };
        };
        let last = dspark.blocks.len().saturating_sub(1);
        let special = match (stage, suffix) {
            (0, "main_proj.weight") => Some(DsparkTensor::MainProjection),
            (0, "main_norm.weight") => Some(DsparkTensor::MainNorm),
            (stage, "norm.weight") if stage == last => Some(DsparkTensor::OutputNorm),
            (stage, "markov_head.markov_w1.weight") if stage == last => {
                Some(DsparkTensor::MarkovEmbedding)
            }
            (stage, "markov_head.markov_w2.weight") if stage == last => {
                Some(DsparkTensor::MarkovOutput)
            }
            (stage, "confidence_head.proj.weight") if stage == last => {
                Some(DsparkTensor::ConfidenceProjection)
            }
            (stage, "hc_head_base") if stage == last => Some(DsparkTensor::HeadHyperBase),
            (stage, "hc_head_fn") if stage == last => Some(DsparkTensor::HeadHyperFunction),
            (stage, "hc_head_scale") if stage == last => Some(DsparkTensor::HeadHyperScale),
            _ => None,
        };
        if let Some(tensor) = special {
            return TensorId::Dspark(tensor);
        }
        if let (Some(layer), Some((expert, tensor))) =
            (dspark.blocks.get(stage), expert_tensor_for_suffix(suffix))
        {
            return TensorId::Expert {
                layer: layer.index,
                expert,
                tensor,
            };
        }
        if let (Some(layer), Some(tensor)) =
            (dspark.blocks.get(stage), layer_tensor_for_suffix(suffix))
        {
            return TensorId::Layer {
                index: layer.index,
                tensor,
            };
        }
    }
    TensorId::Family {
        family: "deepseek_v4",
        key: semantic_key(name),
    }
}

fn expert_tensor_for_suffix(suffix: &str) -> Option<(u32, ExpertTensor)> {
    let rest = suffix.strip_prefix("ffn.experts.")?;
    let (expert, projection) = rest.split_once('.')?;
    let expert = expert.parse().ok()?;
    let tensor = match projection {
        "w1.weight" => ExpertTensor::Gate,
        "w2.weight" => ExpertTensor::Down,
        "w3.weight" => ExpertTensor::Up,
        _ => return None,
    };
    Some((expert, tensor))
}

fn layer_tensor_for_suffix(suffix: &str) -> Option<LayerTensor> {
    match suffix {
        "attn_norm.weight" => Some(LayerTensor::PreAttentionNorm),
        "ffn_norm.weight" => Some(LayerTensor::PreMlpNorm),
        "attn.wq_a.weight" => Some(LayerTensor::MlaQueryDown),
        "attn.q_norm.weight" => Some(LayerTensor::MlaQueryDownNorm),
        "attn.wq_b.weight" => Some(LayerTensor::MlaQueryUp),
        "attn.wkv.weight" => Some(LayerTensor::MlaKvDown),
        "attn.kv_norm.weight" => Some(LayerTensor::MlaKvDownNorm),
        "attn.wo_a.weight" => Some(LayerTensor::MlaOutputDown),
        "attn.wo_b.weight" => Some(LayerTensor::MlaOutput),
        "attn.attn_sink" => Some(LayerTensor::AttentionSink),
        "attn.compressor.wkv.weight" => Some(LayerTensor::KvCompressorKeyValue),
        "attn.compressor.wgate.weight" => Some(LayerTensor::KvCompressorGate),
        "attn.compressor.norm.weight" => Some(LayerTensor::KvCompressorNorm),
        "attn.compressor.ape" => Some(LayerTensor::KvCompressorPosition),
        "attn.indexer.wq_b.weight" => Some(LayerTensor::SparseQuery),
        "attn.indexer.weights_proj.weight" => Some(LayerTensor::SparseProjection),
        "attn.indexer.compressor.wkv.weight" => Some(LayerTensor::SparseCompressorKeyValue),
        "attn.indexer.compressor.wgate.weight" => Some(LayerTensor::SparseCompressorGate),
        "attn.indexer.compressor.norm.weight" => Some(LayerTensor::SparseCompressorNorm),
        "attn.indexer.compressor.ape" => Some(LayerTensor::SparseCompressorPosition),
        "hc_attn_base" => Some(LayerTensor::HyperAttentionBase),
        "hc_attn_fn" => Some(LayerTensor::HyperAttentionFunction),
        "hc_attn_scale" => Some(LayerTensor::HyperAttentionScale),
        "hc_ffn_base" => Some(LayerTensor::HyperMlpBase),
        "hc_ffn_fn" => Some(LayerTensor::HyperMlpFunction),
        "hc_ffn_scale" => Some(LayerTensor::HyperMlpScale),
        "ffn.gate.weight" => Some(LayerTensor::MoeRouter),
        "ffn.gate.bias" => Some(LayerTensor::MoeRouterBias),
        "ffn.gate.tid2eid" => Some(LayerTensor::MoeTokenToExpert),
        "ffn.shared_experts.w1.weight" => Some(LayerTensor::SharedMlpGate),
        "ffn.shared_experts.w2.weight" => Some(LayerTensor::SharedMlpDown),
        "ffn.shared_experts.w3.weight" => Some(LayerTensor::SharedMlpUp),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HfConfig;
    use crate::tensor_contract::{IntegerType, QuantLayout, StorageLayout, TensorCensusEntry};

    fn config() -> ModelConfig {
        ModelConfig::from_hf(&HfConfig::parse(
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
        ))
    }

    fn dspark_config() -> ModelConfig {
        ModelConfig::from_hf(&HfConfig::parse(
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
            "index_topk":16,"compress_ratios":[0,4,0,0,0],
            "compress_rope_theta":160000,"sliding_window":128,"swiglu_limit":10.0,
            "rope_scaling":{"factor":4,"beta_fast":32,"beta_slow":1,
            "original_max_position_embeddings":1024}}"#,
        ))
    }

    #[test]
    fn generated_schema_binds_all_three_quant_recipes_and_auxiliaries() {
        let config = config();
        let plan = PACK.compile_plan(&config).unwrap();
        let contract = PACK
            .compile_tensor_contract(
                &config,
                &plan,
                CheckpointDialect::HfSafetensors,
                ContractOptions::default(),
            )
            .unwrap();
        let mut census: Vec<_> = contract
            .requirements
            .iter()
            .map(|requirement| TensorCensusEntry {
                name: requirement.names[0].clone(),
                shape: requirement.shape.clone(),
                storage: match requirement.quant {
                    QuantConstraint::ExactFloat(float) => StorageLayout::Float(float),
                    QuantConstraint::I64 => StorageLayout::Integer(IntegerType::I64),
                    QuantConstraint::Nvfp4 => StorageLayout::Quantized(QuantLayout {
                        format: "NVFP4".into(),
                        block_shape: vec![16],
                        auxiliaries: requirement.auxiliaries.clone().unwrap(),
                    }),
                    QuantConstraint::Mxfp4 => StorageLayout::Quantized(QuantLayout {
                        format: "MXFP4".into(),
                        block_shape: vec![32],
                        auxiliaries: requirement.auxiliaries.clone().unwrap(),
                    }),
                    QuantConstraint::Fp8Block128 => StorageLayout::Quantized(QuantLayout {
                        format: "FP8_E4M3".into(),
                        block_shape: vec![128, 128],
                        auxiliaries: requirement.auxiliaries.clone().unwrap(),
                    }),
                    other => panic!("unexpected DSV4 constraint {other:?}"),
                },
                physical_bytes: 1,
            })
            .collect();
        let bound = contract.bind(&census).unwrap();
        assert_eq!(bound.tensors.len(), contract.requirements.len());
        assert!(contract.requirements.iter().any(|requirement| {
            requirement.quant == QuantConstraint::Nvfp4
                && requirement.owner == TensorOwner::Layer(0)
        }));
        assert!(contract.requirements.iter().any(|requirement| {
            requirement.names == ["layers.0.ffn.experts.2.w3.weight"]
                && requirement.id
                    == TensorId::Expert {
                        layer: 0,
                        expert: 2,
                        tensor: ExpertTensor::Up,
                    }
        }));
        assert!(contract.requirements.iter().any(|requirement| {
            requirement.names == ["layers.0.ffn.gate.tid2eid"]
                && requirement.id
                    == (TensorId::Layer {
                        index: 0,
                        tensor: LayerTensor::MoeTokenToExpert,
                    })
                && requirement.quant == QuantConstraint::I64
        }));
        for (name, id) in [
            ("hc_head_fn", TensorId::HyperHeadFunction),
            (
                "layers.0.attn.wq_a.weight",
                TensorId::Layer {
                    index: 0,
                    tensor: LayerTensor::MlaQueryDown,
                },
            ),
            (
                "layers.0.attn.attn_sink",
                TensorId::Layer {
                    index: 0,
                    tensor: LayerTensor::AttentionSink,
                },
            ),
            (
                "layers.0.hc_attn_fn",
                TensorId::Layer {
                    index: 0,
                    tensor: LayerTensor::HyperAttentionFunction,
                },
            ),
            (
                "layers.0.hc_ffn_fn",
                TensorId::Layer {
                    index: 0,
                    tensor: LayerTensor::HyperMlpFunction,
                },
            ),
        ] {
            assert!(
                contract
                    .requirements
                    .iter()
                    .any(|requirement| requirement.names == [name] && requirement.id == id)
            );
        }
        assert!(contract.requirements.iter().any(|requirement| {
            requirement.quant == QuantConstraint::Mxfp4 && requirement.owner == TensorOwner::Mtp(0)
        }));
        assert!(
            contract
                .requirements
                .iter()
                .any(|requirement| requirement.quant == QuantConstraint::Fp8Block128)
        );

        let row = census
            .iter_mut()
            .find(|entry| {
                matches!(
                    entry.storage,
                    StorageLayout::Quantized(QuantLayout { ref auxiliaries, .. })
                        if !auxiliaries.is_empty()
                )
            })
            .unwrap();
        let StorageLayout::Quantized(layout) = &mut row.storage else {
            unreachable!()
        };
        layout.auxiliaries.pop();
        assert!(matches!(
            contract.bind(&census),
            Err(TensorContractError::AuxiliaryLayoutMismatch { .. })
        ));
    }

    #[test]
    fn three_block_config_selects_live_dspark_schema_and_plan() {
        use crate::model_plan::{
            AttentionPlan, DrafterPlan, MlaAttentionPlan, RopeFactors, StatePlan,
        };

        let config = dspark_config();
        assert_eq!(config.nextn_predict_layers, 3);
        let plan = DSPARK_PACK.compile_plan(&config).unwrap();
        assert!(plan.mtp_blocks.is_empty());
        let Some(DrafterPlan::Dspark(dspark)) = plan.drafter.as_ref() else {
            panic!("expected DSpark plan");
        };
        assert_eq!(dspark.blocks.len(), 3);
        assert_eq!(dspark.target_layer_ids, vec![40, 41, 42]);
        assert!(matches!(
            plan.layers[1].attention,
            AttentionPlan::Mla(MlaAttentionPlan::CompressedKv {
                output_lora_rank: 128,
                output_groups: 1,
                window: 128,
                compressor: Some(crate::model_plan::KvCompressorPlan {
                    ratio: 4,
                    latent_dim: 256,
                }),
                ..
            })
        ));
        let AttentionPlan::Mla(MlaAttentionPlan::CompressedKv { rope, .. }) =
            &plan.layers[1].attention
        else {
            unreachable!()
        };
        assert_eq!(rope.base, 160_000.0);
        assert!(matches!(
            rope.factors,
            RopeFactors::Yarn {
                factor: 4.0,
                original_context: 1024,
                beta_fast: 32.0,
                beta_slow: 1.0,
            }
        ));
        let operations = plan.operations();
        for operation in [
            crate::model_plan::OperationKind::CompressedMlaAttention,
            crate::model_plan::OperationKind::KvCompressor,
            crate::model_plan::OperationKind::SparseIndex,
            crate::model_plan::OperationKind::HyperConnections,
            crate::model_plan::OperationKind::TokenHashRouter,
            crate::model_plan::OperationKind::SqrtSoftplusRouter,
            crate::model_plan::OperationKind::DsparkFusion,
            crate::model_plan::OperationKind::DsparkMarkovHead,
            crate::model_plan::OperationKind::DsparkConfidenceHead,
        ] {
            assert!(operations.contains(&operation), "missing {operation:?}");
        }
        assert!(matches!(
            plan.layers[1].state,
            StatePlan::CompressedAttention {
                compressor_ratio: Some(4),
                sparse_top_k: Some(16),
                ..
            }
        ));
        let contract = DSPARK_PACK
            .compile_tensor_contract(
                &config,
                &plan,
                CheckpointDialect::HfSafetensors,
                ContractOptions::default(),
            )
            .unwrap();
        assert!(contract.requirements.iter().any(|requirement| {
            requirement.names == ["mtp.0.main_proj.weight"]
                && requirement.id == TensorId::Dspark(DsparkTensor::MainProjection)
                && requirement.quant == QuantConstraint::Fp8Block128
        }));
        assert!(
            !contract
                .requirements
                .iter()
                .any(|requirement| requirement.names == ["mtp.0.e_proj.weight"])
        );
        assert!(contract.requirements.iter().any(|requirement| {
            requirement.names == ["mtp.0.ffn.experts.0.w1.weight"]
                && requirement.quant == QuantConstraint::Mxfp4
        }));
        let trunk = contract
            .requirements
            .iter()
            .find(|requirement| requirement.names == ["layers.0.ffn.experts.0.w1.weight"])
            .unwrap();
        assert_eq!(trunk.quant, QuantConstraint::Nvfp4);
        assert_eq!(
            trunk.auxiliaries.as_deref(),
            Some(
                [
                    "layers.0.ffn.experts.0.w1.weight_scale".to_string(),
                    "layers.0.ffn.experts.0.w1.weight_scale_2".to_string(),
                ]
                .as_slice()
            )
        );
        let draft = contract
            .requirements
            .iter()
            .find(|requirement| requirement.names == ["mtp.0.ffn.experts.0.w1.weight"])
            .unwrap();
        assert_eq!(
            draft.auxiliaries.as_deref(),
            Some(["mtp.0.ffn.experts.0.w1.scale".to_string()].as_slice())
        );
        assert!(contract.requirements.iter().any(|requirement| {
            requirement.names == ["mtp.2.confidence_head.proj.weight"]
                && requirement.id == TensorId::Dspark(DsparkTensor::ConfidenceProjection)
        }));
        assert!(contract.requirements.iter().any(|requirement| {
            requirement.names == ["mtp.1.attn.wq_a.weight"]
                && requirement.id
                    == TensorId::Layer {
                        index: dspark.blocks[1].index,
                        tensor: LayerTensor::MlaQueryDown,
                    }
        }));
        assert!(contract.requirements.iter().any(|requirement| {
            requirement.names == ["mtp.1.ffn.experts.2.w3.weight"]
                && requirement.id
                    == TensorId::Expert {
                        layer: dspark.blocks[1].index,
                        expert: 2,
                        tensor: ExpertTensor::Up,
                    }
        }));
        assert!(
            !contract
                .requirements
                .iter()
                .any(|requirement| matches!(requirement.id, TensorId::Family { .. })),
            "live DSpark contract must not leak checkpoint names into semantic IDs"
        );
    }
}
