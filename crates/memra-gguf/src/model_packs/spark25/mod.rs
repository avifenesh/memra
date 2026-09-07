use super::*;
use crate::config::{HfConfig, Step35Variant};
use crate::tensor_contract::{
    LayerTensor, TensorId, TensorMatch, TensorRequirement, TensorTransform,
};

/// XHToken Spark-X2.5 (HF `spark2_5`, `Spark2_5ForCausalLM`): the DENSE member of the `step35`
/// execution family. Same block as Step-3.5: sliding-window attention (512) on a 3:1 pattern,
/// rope base 5e6 on the full-attention layers and 1e4 on the sliding ones, partial rotary
/// (0.25 of head_dim) on the full layers only, and one sigmoid gate scalar per head projected
/// by a separate `g_proj` and multiplied into the attention output before `out_proj`.
///
/// What is its own, and why this is a separate pack rather than a step35 alias (no
/// generic-support law): no MoE anywhere (a plain gate/up/down FFN on every layer), no QK-norm,
/// RMSNorm weights used verbatim (no `+1` fold), exact-erf GELU on the FFN gate (the vendor
/// modeling raises on any other `hidden_act`), ONE fused `self_attn.q_k_v_proj.weight` (rows
/// q | k | v), `self_attn.out_proj`, `model.embedding.weight`, and a tied output head.
///
/// Brought up on `XHToken/Spark-X2.5-4B` (census 290 tensors, 2026-09-07): 36 layers, hidden
/// 2560, ffn 10240, 16 heads over 4 KV heads, head_dim 256, vocab 131072, 1M positions.
pub static PACK: ModelPack = ModelPack {
    family: "spark25",
    aliases: &["spark2_5", "spark-x2.5", "spark_x25"],
    config_layout: ConfigLayout::Flat,
    tokenizer_sources: &[TokenizerSource::TokenizerJson],
    template: TemplateContract::ArtifactRequired,
    // Inspect-only until the tiny parity fixture runs on the reference executor; flipped by the
    // bring-up lane with its receipt, never by default.
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
    // Set together with `support` once the tiny fixture has run on the reference executor
    // (the pack invariant: no parity gate without a native support state). The threshold to
    // use then is llama_dense's (0.005 abs/rel, argmax required): the same bf16-from-HF class.
    checkpoint_parity: None,
    matches_config: |config| {
        config
            .step35
            .as_ref()
            .is_some_and(|step| step.variant == Step35Variant::SparkX25)
            && !config.moe.as_ref().is_some_and(|moe| moe.expert_count > 0)
    },
    plan_builder: canonical_plan,
    tensor_schema: spark_tensor_schema,
    tiny_plan: Some(tiny_plan),
};

/// The canonical contract with Spark's four spellings applied on the HF dialect. The GGUF
/// dialect keeps the canonical names: a future Spark GGUF mint would be written with separate
/// `attn_q/k/v` like every other GGUF, and the census would say so.
#[allow(clippy::result_large_err)] // allow: the fat error type is the diagnostic contract here; boxing it would change the error surface
fn spark_tensor_schema(
    config: &ModelConfig,
    plan: &ModelPlan,
    dialect: CheckpointDialect,
    options: ContractOptions,
) -> Result<TensorContract, TensorContractError> {
    let mut contract = TensorContract::for_plan(plan, dialect, options)?;
    if dialect != CheckpointDialect::HfSafetensors {
        return Ok(contract);
    }
    let (q, k, v) = crate::hf_mapping::spark_qkv_bands(config);
    let fused_rows = (q + k + v) as u64;
    let hidden = plan.hidden_size as u64;
    // Spark's spellings, applied to the weight requirement AND to its optional quant
    // auxiliaries (`<stem>.pre_quant_scale` etc.), which the builder derives from the
    // canonical name.
    let respell = |name: &str| -> String {
        name.replace("model.embed_tokens.weight", "model.embedding.weight")
            .replace("self_attn.o_proj.weight", "self_attn.out_proj.weight")
            .replace("self_attn.gate_proj.weight", "self_attn.g_proj.weight")
            .replace("self_attn.o_proj.", "self_attn.out_proj.")
            .replace("self_attn.gate_proj.", "self_attn.g_proj.")
            .replace("model.embed_tokens.", "model.embedding.")
            .replace("self_attn.q_proj.", "self_attn.q_k_v_proj.")
    };
    let is_kv_plane = |id: &TensorId| -> bool {
        match id {
            TensorId::Layer { tensor, .. } => {
                matches!(tensor, LayerTensor::Key | LayerTensor::Value)
            }
            TensorId::QuantAux { tensor, .. } => match tensor.as_ref() {
                TensorId::Layer { tensor, .. } => {
                    matches!(tensor, LayerTensor::Key | LayerTensor::Value)
                }
                _ => false,
            },
            _ => false,
        }
    };
    let mut out: Vec<TensorRequirement> = Vec::with_capacity(contract.requirements.len());
    for requirement in contract.requirements.drain(..) {
        // The k and v planes have no tensor of their own: they are slices of the fused one.
        if is_kv_plane(&requirement.id) {
            continue;
        }
        let names: Vec<String> = requirement.names.iter().map(|n| respell(n)).collect();
        match &requirement.id {
            // ONE requirement for the fused tensor, keyed on its own id: `bind` refuses a
            // name claimed by two ids, and the loader's map slices all three planes out of
            // it (`TransformKind::SparkQkv*`).
            TensorId::Layer {
                index,
                tensor: LayerTensor::Query,
            } => out.push(TensorRequirement {
                id: TensorId::Layer {
                    index: *index,
                    tensor: LayerTensor::QkvSource,
                },
                names,
                match_mode: TensorMatch::OneOf,
                shape: vec![fused_rows, hidden],
                transform: TensorTransform::SplitQkvRows,
                ..requirement
            }),
            TensorId::QuantAux { tensor, kind } => match tensor.as_ref() {
                TensorId::Layer {
                    index,
                    tensor: LayerTensor::Query,
                } => out.push(TensorRequirement {
                    id: TensorId::QuantAux {
                        tensor: Box::new(TensorId::Layer {
                            index: *index,
                            tensor: LayerTensor::QkvSource,
                        }),
                        kind: *kind,
                    },
                    names,
                    ..requirement
                }),
                _ => out.push(TensorRequirement {
                    names,
                    ..requirement
                }),
            },
            _ => out.push(TensorRequirement {
                names,
                ..requirement
            }),
        }
    }
    contract.requirements = out;
    Ok(contract)
}

/// Two layers (one sliding, one full), the real family's field spellings, tiny widths.
fn tiny_plan() -> Result<ModelPlan, PlanCompileError> {
    canonical_plan(&ModelConfig::from_hf(&HfConfig::parse(
        r#"{"model_type":"spark2_5","num_hidden_layers":2,"hidden_size":8,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
        "intermediate_size":16,"vocab_size":32,"max_position_embeddings":64,
        "rms_norm_eps":0.000001,"hidden_act":"gelu","tie_word_embeddings":true,
        "sliding_window":4,"layer_types":["sliding_attention","full_attention"],
        "rope_parameters":{"full_attention":{"partial_rotary_factor":0.5,"rope_theta":5000000},
        "sliding_attention":{"partial_rotary_factor":1.0,"rope_theta":10000}},
        "headwise_attn_output_gate":true,"gate_attn_act_mode":"sigmoid"}"#,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Arch, AttentionGateKind};
    use crate::hf_mapping::{HfTarget, TransformKind, resolve_ggml};
    use crate::model_plan::{
        ActivationPlan, AttentionPlan, MlpPlan, ResidualTopology, RopeFactors, TensorPresence,
    };
    use crate::tensor_contract::{CheckpointDialect, ContractOptions, OutputHead};

    /// `XHToken/Spark-X2.5-4B` config.json, verbatim except the 36-entry `layer_types` array,
    /// which is generated below from the checkpoint's pattern (full attention at il % 4 == 3).
    fn spark_4b_json() -> String {
        let layer_types: Vec<String> = (0..36)
            .map(|il| {
                if il % 4 == 3 {
                    "\"full_attention\"".to_string()
                } else {
                    "\"sliding_attention\"".to_string()
                }
            })
            .collect();
        format!(
            r#"{{"architectures":["Spark2_5ForCausalLM"],"attention_bias":false,
            "attention_dropout":0.0,"auto_map":{{"AutoConfig":"configuration_spark.Spark2_5Config",
            "AutoModel":"modeling_spark.Spark2_5Model",
            "AutoModelForCausalLM":"modeling_spark.Spark2_5ForCausalLM"}},
            "bos_token_id":0,"dtype":"bfloat16","eos_token_id":1,"gate_attn_act_mode":"sigmoid",
            "head_dim":256,"headwise_attn_output_gate":true,"hidden_act":"gelu","hidden_size":2560,
            "initializer_range":0.01976,"intermediate_size":10240,"layer_types":[{}],
            "max_position_embeddings":1048576,"mlp_bias":false,"model_type":"spark2_5",
            "num_attention_heads":16,"num_hidden_layers":36,"num_key_value_heads":4,
            "pad_token_id":2,"rms_norm_eps":0.000001,
            "rope_parameters":{{"full_attention":{{"partial_rotary_factor":0.25,"rope_theta":5000000}},
            "sliding_attention":{{"partial_rotary_factor":1.0,"rope_theta":10000}}}},
            "sliding_window":512,"tie_word_embeddings":true,"transformers_version":"4.57.1",
            "use_cache":true,"vocab_size":131072}}"#,
            layer_types.join(",")
        )
    }

    fn spark_config() -> ModelConfig {
        ModelConfig::from_hf(&HfConfig::parse(&spark_4b_json()))
    }

    /// The checkpoint's 290 tensor names, generated from the census pattern
    /// (2 globals + 36 layers x 8).
    fn spark_4b_tensor_names() -> Vec<String> {
        let mut names = vec![
            "model.embedding.weight".to_string(),
            "model.norm.weight".to_string(),
        ];
        for il in 0..36 {
            for suffix in [
                "input_layernorm.weight",
                "mlp.down_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "post_attention_layernorm.weight",
                "self_attn.g_proj.weight",
                "self_attn.out_proj.weight",
                "self_attn.q_k_v_proj.weight",
            ] {
                names.push(format!("model.layers.{il}.{suffix}"));
            }
        }
        assert_eq!(names.len(), 290);
        names
    }

    #[test]
    fn spark2_5_model_type_is_the_step35_program_in_its_dense_variant() {
        let config = spark_config();
        assert_eq!(config.arch, Arch::Step35);
        let step = config
            .step35
            .as_ref()
            .expect("spark2_5 must build a Step35Config");
        assert_eq!(step.variant, Step35Variant::SparkX25);
        assert!(step.is_spark());
        assert!(!step.norm_plus_one, "Spark2_5RMSNorm is plain w * normed");
        assert!(!step.qk_norm, "no q_norm/k_norm tensors in the checkpoint");
        assert!(!step.rope_factors, "plain rope on every layer");
        assert!(step.fused_qkv);
        assert_eq!(step.hidden_act, "gelu");
        assert!(config.moe.is_none(), "dense everywhere");
        assert!(config.dense_act_gelu_erf());
        assert_eq!(super::super::for_config(&config).unwrap().family, "spark25");
        assert!(
            !super::super::step35::PACK.matches_config(&config),
            "the MoE Step pack must not claim the dense sibling"
        );
    }

    #[test]
    fn spark_4b_geometry_from_the_real_config() {
        let config = spark_config();
        let step = config.step35.as_ref().unwrap();
        assert_eq!(config.n_layer, 36);
        assert_eq!(
            (config.n_embd, config.n_ff, config.n_vocab),
            (2560, 10240, 131072)
        );
        assert_eq!((config.n_head, config.n_head_kv), (16, 4));
        assert_eq!((config.head_dim_k, config.head_dim_v), (256, 256));
        assert_eq!(config.context_length, 1_048_576);
        assert_eq!(config.rms_eps, 1e-6);
        assert_eq!(step.sliding_window, 512);
        assert_eq!((step.rope_base_global, step.rope_base_swa), (5e6, 1e4));
        assert_eq!((step.rope_dims_full, step.rope_dims_swa), (64, 256));
        assert_eq!(step.first_k_dense_replace, 36);
        assert_eq!(step.n_full_attn(36), 9);
        for il in 0..36 {
            assert_eq!(step.is_swa(il), il % 4 != 3, "layer {il}");
            assert_eq!(step.n_head(il), 16);
            assert_eq!(step.n_head_kv(il), 4);
            assert_eq!(step.n_rot(il), if il % 4 == 3 { 64 } else { 256 });
            assert_eq!(step.rope_base(il), if il % 4 == 3 { 5e6 } else { 1e4 });
            assert_eq!(step.clamp_exp(il), None);
            assert_eq!(step.clamp_shexp(il), None);
        }
        assert_eq!(config.validate_attention_gate_layout().map(|_| ()), Ok(()));
    }

    #[test]
    fn spark_4b_compiles_the_dense_gated_swa_stack() {
        let config = spark_config();
        let plan = PACK.compile_plan(&config).expect("plan must compile");
        assert_eq!(plan.layers.len(), 36);
        assert!(plan.mtp_blocks.is_empty());
        assert!(plan.vision.is_none());
        for (il, layer) in plan.layers.iter().enumerate() {
            let (attn, window) = match &layer.attention {
                AttentionPlan::Full(attn) => (attn, None),
                AttentionPlan::SlidingWindow { attention, window } => (attention, Some(*window)),
                other => panic!("layer {il}: unexpected attention {other:?}"),
            };
            assert_eq!(
                window,
                if il % 4 == 3 { None } else { Some(512) },
                "layer {il}"
            );
            assert_eq!((attn.query_heads, attn.kv_heads), (16, 4));
            assert_eq!((attn.key_head_dim, attn.value_head_dim), (256, 256));
            assert_eq!(attn.rope.dimensions, if il % 4 == 3 { 64 } else { 256 });
            assert_eq!(attn.rope.base, if il % 4 == 3 { 5e6 } else { 1e4 });
            assert_eq!(attn.rope.factors, RopeFactors::None, "layer {il}");
            assert_eq!(attn.qk_norm, TensorPresence::Absent, "layer {il}");
            assert_eq!(attn.output_gate, AttentionGateKind::SeparateHead);
            assert_eq!(layer.residual, ResidualTopology::Serial);
            let MlpPlan::Dense(mlp) = &layer.mlp else {
                panic!("layer {il}: Spark is dense, not MoE");
            };
            assert_eq!(mlp.intermediate_size, 10240);
            assert_eq!(mlp.activation, ActivationPlan::GeluErf);
        }
    }

    #[test]
    fn hf_contract_binds_the_real_checkpoint_names_and_every_name_resolves() {
        let config = spark_config();
        let plan = PACK.compile_plan(&config).unwrap();
        let contract = PACK
            .compile_tensor_contract(
                &config,
                &plan,
                CheckpointDialect::HfSafetensors,
                ContractOptions {
                    output_head: OutputHead::TiedToEmbedding,
                },
            )
            .expect("HF contract must compile");
        let names = spark_4b_tensor_names();
        // Every REQUIRED contract name is a real checkpoint name, and every checkpoint name is
        // claimed exactly once. The optional quant auxiliaries (`.pre_quant_scale`) are named
        // off the same stems and must not carry the canonical spellings either.
        let mut claimed = std::collections::BTreeSet::new();
        let mut required = 0;
        for requirement in &contract.requirements {
            if requirement.names.is_empty() {
                // GGUF-only auxiliaries (weight/input scales) carry no HF name.
                assert!(!requirement.required, "{:?}", requirement.id);
                continue;
            }
            assert_eq!(requirement.names.len(), 1, "{:?}", requirement.id);
            let name = &requirement.names[0];
            for canonical in [
                "model.embed_tokens.",
                "self_attn.o_proj.",
                "self_attn.gate_proj.",
                "self_attn.q_proj.",
                "self_attn.k_proj.",
                "self_attn.v_proj.",
            ] {
                assert!(
                    !name.contains(canonical),
                    "{name} keeps a canonical spelling the checkpoint does not use"
                );
            }
            if !requirement.required {
                continue;
            }
            required += 1;
            assert!(
                names.contains(name),
                "contract names {name} which the checkpoint lacks"
            );
            assert!(claimed.insert(name.clone()), "{name} claimed twice");
        }
        for name in &names {
            assert!(
                claimed.contains(name),
                "checkpoint tensor {name} is not in the contract"
            );
        }
        assert_eq!(required, 290);
        // The fused projection carries the split transform and the fused shape.
        let fused = contract
            .requirements
            .iter()
            .find(|r| {
                r.names
                    .first()
                    .is_some_and(|n| n == "model.layers.3.self_attn.q_k_v_proj.weight")
                    && r.required
            })
            .unwrap();
        assert_eq!(fused.transform, TensorTransform::SplitQkvRows);
        assert_eq!(fused.shape, vec![(16 + 4 + 4) * 256, 2560]);
        assert_eq!(
            fused.id,
            TensorId::Layer {
                index: 3,
                tensor: LayerTensor::QkvSource
            }
        );
        let gate = contract
            .requirements
            .iter()
            .find(|r| {
                r.names
                    .first()
                    .is_some_and(|n| n == "model.layers.3.self_attn.g_proj.weight")
                    && r.required
            })
            .unwrap();
        assert_eq!(gate.shape, vec![16, 2560]);
        // The engine asks for ggml names; each resolves to the checkpoint's own spelling.
        for (ggml, hf, kind) in [
            ("token_embd.weight", "model.embedding.weight", None),
            ("output_norm.weight", "model.norm.weight", None),
            (
                "blk.3.attn_q.weight",
                "model.layers.3.self_attn.q_k_v_proj.weight",
                Some(TransformKind::SparkQkvQuery),
            ),
            (
                "blk.3.attn_k.weight",
                "model.layers.3.self_attn.q_k_v_proj.weight",
                Some(TransformKind::SparkQkvKey),
            ),
            (
                "blk.3.attn_v.weight",
                "model.layers.3.self_attn.q_k_v_proj.weight",
                Some(TransformKind::SparkQkvValue),
            ),
            (
                "blk.3.attn_output.weight",
                "model.layers.3.self_attn.out_proj.weight",
                None,
            ),
            (
                "blk.3.attn_gate.weight",
                "model.layers.3.self_attn.g_proj.weight",
                None,
            ),
            (
                "blk.3.attn_norm.weight",
                "model.layers.3.input_layernorm.weight",
                None,
            ),
            (
                "blk.3.ffn_norm.weight",
                "model.layers.3.post_attention_layernorm.weight",
                None,
            ),
            (
                "blk.3.ffn_gate.weight",
                "model.layers.3.mlp.gate_proj.weight",
                None,
            ),
            (
                "blk.3.ffn_up.weight",
                "model.layers.3.mlp.up_proj.weight",
                None,
            ),
            (
                "blk.3.ffn_down.weight",
                "model.layers.3.mlp.down_proj.weight",
                None,
            ),
        ] {
            match resolve_ggml(ggml, &config) {
                Some(HfTarget::Plain(got)) => {
                    assert_eq!(got, hf, "{ggml}");
                    assert!(kind.is_none(), "{ggml} must be a plain rename");
                    assert!(
                        names.contains(&got),
                        "{ggml} -> {got} is not a checkpoint tensor"
                    );
                }
                Some(HfTarget::Transform {
                    hf: got,
                    kind: got_kind,
                }) => {
                    assert_eq!(got, hf, "{ggml}");
                    let want = kind.expect("{ggml} must not carry a transform");
                    assert!(
                        std::mem::discriminant(&got_kind) == std::mem::discriminant(&want),
                        "{ggml} took the wrong transform"
                    );
                }
                None => panic!("{ggml} does not resolve"),
            }
        }
        // Norms are NOT +1-folded on this sibling.
        for norm in [
            "blk.3.attn_norm.weight",
            "blk.3.ffn_norm.weight",
            "output_norm.weight",
        ] {
            assert!(
                matches!(resolve_ggml(norm, &config), Some(HfTarget::Plain(_))),
                "{norm} must load verbatim (no NormPlusOne)"
            );
        }
        // The tied head resolves to a name the checkpoint does not carry: that absence IS the
        // loader's tied-output path.
        match resolve_ggml("output.weight", &config) {
            Some(HfTarget::Plain(hf)) => assert!(!names.contains(&hf)),
            other => panic!("output.weight resolved unexpectedly: {other:?}"),
        }
    }

    #[test]
    fn fused_qkv_row_slices_take_the_vendor_bands_in_order() {
        let config = spark_config();
        let (q, k, v) = crate::hf_mapping::spark_qkv_bands(&config);
        assert_eq!((q, k, v), (4096, 1024, 1024));
        // A small fused matrix in HF row-major [out=q+k+v, in]: value = row*1000 + col.
        let cfg = ModelConfig::from_hf(&HfConfig::parse(&tiny_json()));
        let (q, k, v) = crate::hf_mapping::spark_qkv_bands(&cfg);
        assert_eq!((q, k, v), (8, 4, 4));
        let (out_f, in_f) = (q + k + v, cfg.n_embd as usize);
        let data: Vec<f32> = (0..out_f * in_f)
            .map(|i| (i / in_f * 1000 + i % in_f) as f32)
            .collect();
        for (kind, start, rows) in [
            (TransformKind::SparkQkvQuery, 0, q),
            (TransformKind::SparkQkvKey, q, k),
            (TransformKind::SparkQkvValue, q + k, v),
        ] {
            let mut d = data.clone();
            let (ne, bytes) = kind.apply(&mut d, vec![in_f as u64, out_f as u64], &cfg);
            assert_eq!(ne, vec![in_f as u64, rows as u64]);
            let got: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            assert_eq!(got, data[start * in_f..(start + rows) * in_f].to_vec());
        }
    }

    fn tiny_json() -> String {
        r#"{"model_type":"spark2_5","num_hidden_layers":2,"hidden_size":8,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
        "intermediate_size":16,"vocab_size":32,"max_position_embeddings":64,
        "rms_norm_eps":0.000001,"hidden_act":"gelu","tie_word_embeddings":true,
        "sliding_window":4,"layer_types":["sliding_attention","full_attention"],
        "rope_parameters":{"full_attention":{"partial_rotary_factor":0.5,"rope_theta":5000000},
        "sliding_attention":{"partial_rotary_factor":1.0,"rope_theta":10000}},
        "headwise_attn_output_gate":true,"gate_attn_act_mode":"sigmoid"}"#
            .to_string()
    }

    #[test]
    fn tiny_fixture_compiles_on_both_dialects() {
        let plan = PACK.compile_tiny_plan().expect("tiny fixture must compile");
        let cfg = ModelConfig::from_hf(&HfConfig::parse(&tiny_json()));
        for dialect in [CheckpointDialect::HfSafetensors, CheckpointDialect::Gguf] {
            PACK.compile_tensor_contract(
                &cfg,
                &plan,
                dialect,
                ContractOptions {
                    output_head: OutputHead::TiedToEmbedding,
                },
            )
            .unwrap_or_else(|e| panic!("tensor contract must compile for {dialect:?}: {e:?}"));
        }
    }

    #[test]
    #[should_panic(expected = "headwise_attn_output_gate must be true")]
    fn a_config_without_the_head_gate_is_refused() {
        let json = tiny_json().replace(
            "\"headwise_attn_output_gate\":true",
            "\"headwise_attn_output_gate\":false",
        );
        let _ = ModelConfig::from_hf(&HfConfig::parse(&json));
    }

    #[test]
    #[should_panic(expected = "gate_attn_act_mode must be")]
    fn a_config_with_another_gate_activation_is_refused() {
        let json = tiny_json().replace(
            "\"gate_attn_act_mode\":\"sigmoid\"",
            "\"gate_attn_act_mode\":\"silu\"",
        );
        let _ = ModelConfig::from_hf(&HfConfig::parse(&json));
    }

    /// A step3p5-shaped config (the MoE Step-3.7 fixture) still lands in the step35 pack and
    /// keeps its own behaviours: this pack never claims it.
    #[test]
    fn the_moe_step_config_keeps_its_own_pack() {
        let step = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"step3p5","num_hidden_layers":3,"num_nextn_predict_layers":1,
            "hidden_size":16,"intermediate_size":32,"num_attention_heads":2,
            "num_attention_groups":1,"head_dim":8,"vocab_size":64,
            "max_position_embeddings":2048,"moe_num_experts":6,"moe_top_k":2,
            "moe_intermediate_size":12,"share_expert_dim":12,"moe_layers_enum":"1,2",
            "moe_router_activation":"sigmoid","layer_types":["full_attention",
            "sliding_attention","sliding_attention","sliding_attention"],
            "rope_theta":[5000000,10000,10000,10000],
            "partial_rotary_factors":[0.5,1,1,1],"sliding_window":512,
            "attention_other_setting":{"num_attention_heads":4,"num_attention_groups":1},
            "swiglu_limits":[0,0,0,0],"swiglu_limits_shared":[0,0,0,0]}"#,
        ));
        let s = step.step35.as_ref().unwrap();
        assert_eq!(s.variant, Step35Variant::StepFlash);
        assert!(s.norm_plus_one && s.qk_norm && s.rope_factors && !s.fused_qkv);
        assert_eq!(s.hidden_act, "silu");
        assert!(!PACK.matches_config(&step));
        assert_eq!(super::super::for_config(&step).unwrap().family, "step35");
    }
}
