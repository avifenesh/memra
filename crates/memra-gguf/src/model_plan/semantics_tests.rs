//! Load-time semantic regressions for #537. These tests require no weights or GPU.
use crate::config::{HfConfig, ModelConfig};
use crate::model_packs::{compile_for_load, compile_for_source, for_config};
use crate::model_plan::{
    ActivationPlan, AttentionPlan, MlpPlan, ModelPlan, PlanCompileError, RopeFactors, RouterPlan,
    WeightTransform,
};

struct ConfigOnlySource<'a> {
    config: &'a ModelConfig,
    rope_tensor: bool,
}

#[test]
#[allow(clippy::result_large_err)] // allow: assert the shared tensor-contract diagnostic unchanged
fn step_rope_source_shapes_bind_consistently_before_and_after_preflight() {
    use crate::source::{GgufSource, TensorSource};
    use crate::tensor_contract::{ContractOptions, TensorContractError, TensorId};
    for (case, shape, accepted) in [
        ("compact", vec![32], true),
        ("full", vec![64], true),
        ("short", vec![1], false),
        ("intermediate", vec![48], false),
        ("oversized", vec![65], false),
        ("matrix", vec![2, 16], false),
        ("compact_rank2", vec![32, 1], false),
        ("full_rank2", vec![64, 1], false),
    ] {
        let path = std::env::temp_dir().join(format!(
            "memra-537-binding-{case}-{}.gguf",
            std::process::id()
        ));
        crate::micro_gguf::write_step35_rope_contract_fixture(&path, &shape).unwrap();
        let gguf = crate::GgufFile::open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let source = GgufSource(&gguf);
        let raw_config = source.config();
        assert_eq!(
            raw_config
                .step35
                .as_ref()
                .unwrap()
                .rope_freq_shape
                .as_deref(),
            Some(shape.as_slice()),
            "{case}"
        );
        let plan = compile_for_load(&raw_config).unwrap();
        let census = source.tensor_census().unwrap();
        let entries = census
            .tensors
            .iter()
            .map(|row| row.entry.clone())
            .collect::<Vec<_>>();
        let bind = |config| {
            for_config(config)
                .unwrap()
                .compile_tensor_contract(config, &plan, census.dialect, ContractOptions::default())
                .and_then(|contract| contract.bind(&entries))
        };
        let raw_binding = bind(&raw_config);
        let loaded = compile_for_source(&source);
        if accepted {
            let (loaded_config, loaded_plan) = loaded.unwrap();
            assert_eq!(loaded_plan, plan);
            let raw_binding = raw_binding
                .unwrap_or_else(|error| panic!("{case}: raw config bind failed: {error}"));
            let loaded_binding = bind(&loaded_config).unwrap();
            assert_eq!(raw_binding, loaded_binding, "{case}");
            let factors = &raw_binding.tensors[&TensorId::RopeFactors];
            assert_eq!(factors.shapes, vec![shape.clone()], "{case}");
            assert_eq!(factors.physical_bytes, shape[0] * 4, "{case}");
            let step = loaded_config.step35.unwrap();
            assert_eq!(
                step.rope_freq_shape.as_deref(),
                Some(shape.as_slice()),
                "{case}"
            );
            assert_eq!(step.rope_freq_factors.unwrap().len(), shape[0] as usize);
        } else {
            assert!(loaded.is_err(), "{case}: malformed source passed preflight");
            assert!(
                matches!(
                    raw_binding,
                    Err(TensorContractError::ShapeMismatch {
                        id: TensorId::RopeFactors,
                        ..
                    })
                ),
                "{case}: {raw_binding:?}"
            );
        }
    }
}

impl crate::source::TensorSource for ConfigOnlySource<'_> {
    fn config(&self) -> ModelConfig {
        self.config.clone()
    }
    fn find(&self, name: &str) -> Option<crate::source::TensorView<'_>> {
        assert!(
            self.config.step35.is_some(),
            "semantic rejection must precede tensor lookup"
        );
        assert_eq!(
            name, "rope_freqs.weight",
            "preflight must not read model weights"
        );
        self.rope_tensor.then(|| crate::source::TensorView {
            bytes: std::borrow::Cow::Owned(
                [1.0f32; 32].iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
            ggml_type: crate::GgmlType::F32,
            ne: vec![32],
        })
    }
    fn has(&self, name: &str) -> bool {
        assert_eq!(name, "rope_freqs.weight");
        self.rope_tensor
    }
}

#[test]
fn step_gguf_rejects_short_or_invalid_rope_factors_before_upload() {
    use crate::micro_gguf::{GgufWriter, MetaW};
    for (name, factors, accepted) in [
        ("short", vec![1.0], false),
        ("valid", vec![2.0; 32], true),
        // Official Step IQ4_XS and Q8_0 MTP headers store the full-head [64]
        // vector; partial RoPE consumes its first 32 entries. Preserve every byte.
        ("official_full_head", vec![2.0; 64], true),
        ("undeclared_width", vec![2.0; 48], false),
        ("zero", vec![0.0; 32], false),
        ("nan", vec![f32::NAN; 32], false),
    ] {
        let mut writer = GgufWriter::new();
        writer.kv("general.architecture", MetaW::Str("step35"));
        for (key, value) in [
            ("block_count", 2),
            ("embedding_length", 256),
            ("feed_forward_length", 512),
            ("attention.key_length", 128),
            ("attention.value_length", 128),
            ("attention.sliding_window", 512),
            ("vocab_size", 32),
        ] {
            writer.kv(&format!("step35.{key}"), MetaW::U32(value));
        }
        writer.kv("step35.attention.head_count", MetaW::ArrU32(vec![2, 3]));
        writer.kv("step35.attention.head_count_kv", MetaW::ArrU32(vec![1, 1]));
        writer.kv(
            "step35.attention.sliding_window_pattern",
            MetaW::ArrBool(vec![false, true]),
        );
        writer.kv("step35.rope.scaling.type", MetaW::Str("llama3"));
        writer.tensor_f32("rope_freqs.weight", &[factors.len() as u64], &factors);
        let path = std::env::temp_dir().join(format!(
            "memra-537-step-factors-{name}-{}.gguf",
            std::process::id()
        ));
        writer.write(&path).unwrap();
        let file = crate::GgufFile::open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let result = compile_for_source(&crate::source::GgufSource(&file));
        if accepted {
            let (cfg, _) = result.unwrap();
            if name == "official_full_head" {
                let plan = compile_for_load(&cfg).unwrap();
                let contract = for_config(&cfg)
                    .unwrap()
                    .compile_tensor_contract(
                        &cfg,
                        &plan,
                        crate::tensor_contract::CheckpointDialect::Gguf,
                        crate::tensor_contract::ContractOptions::default(),
                    )
                    .unwrap();
                let stored = contract
                    .requirements
                    .iter()
                    .find(|tensor| tensor.id == crate::tensor_contract::TensorId::RopeFactors)
                    .unwrap();
                assert_eq!(stored.shape, vec![64]);
            }
            assert_eq!(cfg.step35.unwrap().rope_freq_factors.unwrap(), factors);
        } else {
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("rope_freqs.weight")
            );
        }
    }
}

fn config(model_type: &str, extra: &str) -> ModelConfig {
    ModelConfig::from_hf(&HfConfig::parse(&format!(
        r#"{{"model_type":"{model_type}","num_hidden_layers":2,"hidden_size":8,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
        "intermediate_size":16,"vocab_size":32,"max_position_embeddings":8192,
        {extra}}}"#
    )))
}

fn refuses(cfg: &ModelConfig, field: &str) {
    let error = ModelPlan::compile(cfg).expect_err("must not substitute a different program");
    assert!(error.to_string().contains(field), "{error}");
    // Both dense and hybrid loaders use this entry point before accessing weights.
    assert_eq!(compile_for_load(cfg).unwrap_err(), error);
    assert_eq!(
        compile_for_source(&ConfigOnlySource {
            config: cfg,
            rope_tensor: false
        })
        .unwrap_err()
        .to_string(),
        error.to_string()
    );
}

#[test]
fn canonical_compiler_refuses_unrepresented_sliding_window() {
    for family in ["mistral", "llama", "qwen3", "qwen3_5"] {
        refuses(
            &config(family, r#""sliding_window":4096"#),
            "sliding_window",
        );
    }
}

#[test]
fn canonical_compiler_refuses_unrepresented_rope_scaling() {
    for family in ["mistral", "llama", "qwen3", "qwen3_5"] {
        for key in ["rope_scaling", "rope_parameters"] {
            refuses(
                &config(
                    family,
                    &format!(
                        r#""{key}":{{"rope_type":"llama3","factor":8.0,
                    "low_freq_factor":1.0,"high_freq_factor":4.0,
                    "original_max_position_embeddings":8192}}"#
                    ),
                ),
                "rope_scaling",
            );
        }
    }
}

#[test]
fn canonical_compiler_refuses_unrepresented_activation() {
    for family in ["mistral", "llama", "qwen3", "qwen3_5", "hy3"] {
        refuses(&config(family, r#""hidden_act":"gelu""#), "hidden_act");
    }
}

#[test]
fn gguf_rope_declarations_reach_the_compiler() {
    use crate::micro_gguf::{GgufWriter, MetaW};
    let mut control = None;
    for kind in [None, Some("none"), Some("yarn")] {
        let mut writer = GgufWriter::new();
        writer.kv("general.architecture", MetaW::Str("qwen3"));
        for (name, value) in [
            ("block_count", 2),
            ("embedding_length", 8),
            ("attention.head_count", 2),
            ("attention.head_count_kv", 1),
            ("feed_forward_length", 16),
            ("vocab_size", 32),
        ] {
            writer.kv(&format!("qwen3.{name}"), MetaW::U32(value));
        }
        if let Some(kind) = kind {
            writer.kv("qwen3.rope.scaling.type", MetaW::Str(kind));
            writer.kv("qwen3.rope.scaling.factor", MetaW::F32(2.0));
        }
        let path = std::env::temp_dir().join(format!("memra-537-rope-{}.gguf", std::process::id()));
        writer.write(&path).unwrap();
        let file = crate::GgufFile::open(&path).unwrap();
        let cfg = ModelConfig::from_gguf(&file);
        std::fs::remove_file(&path).unwrap();
        if kind == Some("yarn") {
            assert_eq!(cfg.rope_scaling_hint.as_deref(), Some("yarn"));
            refuses(&cfg, "rope_scaling");
        } else {
            let plan = compile_for_load(&cfg).unwrap();
            if let Some(control) = &control {
                assert_eq!(&plan, control);
            }
            control = Some(plan);
        }
    }
}

#[test]
fn gguf_factor_tensors_cannot_be_ignored_by_checkpoint_free_plans() {
    use crate::micro_gguf::{GgufWriter, MetaW};
    use crate::source::{GgufSource, TensorSource};
    for family in ["llama", "qwen3"] {
        for has_factors in [false, true] {
            let mut writer = GgufWriter::new();
            writer.kv("general.architecture", MetaW::Str(family));
            for (name, value) in [
                ("block_count", 2),
                ("embedding_length", 8),
                ("attention.head_count", 2),
                ("attention.head_count_kv", 1),
                ("feed_forward_length", 16),
                ("vocab_size", 32),
            ] {
                writer.kv(&format!("{family}.{name}"), MetaW::U32(value));
            }
            // Converted Llama3 GGUFs declare scaling through this tensor even when
            // their headers have no rope.scaling.type metadata.
            if has_factors {
                writer.tensor_f32("rope_freqs.weight", &[2], &[2.0, 8.0]);
            }
            let path = std::env::temp_dir().join(format!(
                "memra-537-unrepresented-factors-{family}-{has_factors}-{}.gguf",
                std::process::id()
            ));
            writer.write(&path).unwrap();
            let file = crate::GgufFile::open(&path).unwrap();
            std::fs::remove_file(path).unwrap();
            let source = GgufSource(&file);
            let config = source.config();
            assert!(config.rope_scaling_hint.is_none());
            let plan = compile_for_load(&config).unwrap();
            assert!(crate::tensor_contract::rope_factor_width(&plan).is_none());
            let loaded = compile_for_source(&source);
            if has_factors {
                let Err(error) = loaded else {
                    panic!("a source factor tensor must not be ignored");
                };
                assert!(error.to_string().contains("rope_freqs.weight"), "{error}");
            } else {
                loaded.unwrap();
            }
        }
    }
}

#[test]
fn gemma_per_attention_rope_declarations_are_not_erased() {
    let extra = r#""sliding_window":8,"layer_types":["sliding_attention","full_attention"],
        "hidden_activation":"gelu_pytorch_tanh","rope_parameters":{
        "full_attention":{"rope_type":"proportional","partial_rotary_factor":0.5},
        "sliding_attention":{"rope_type":"default"}}"#;
    let cfg = config("gemma4", extra);
    let plan = compile_for_load(&cfg).unwrap();
    assert!(matches!(
        plan.layers[0].attention,
        AttentionPlan::SlidingWindow { window: 8, .. }
    ));
    let MlpPlan::Dense(mlp) = &plan.layers[0].mlp else {
        panic!("expected dense MLP")
    };
    assert_eq!(mlp.activation, ActivationPlan::GeluTanh);
    let AttentionPlan::Full(global) = &plan.layers[1].attention else {
        panic!("expected global attention")
    };
    assert_eq!(
        global.rope.factors,
        RopeFactors::PartialRotary { factor: 0.5 }
    );
    for original in [r#""rope_type":"proportional""#, r#""rope_type":"default""#] {
        let altered = extra.replace(original, r#""rope_type":"linear","factor":2.0"#);
        refuses(&config("gemma4", &altered), "rope_scaling");
    }
    let global_default = extra.replace(
        r#""rope_type":"proportional","partial_rotary_factor":0.5"#,
        r#""rope_type":"default""#,
    );
    refuses(&config("gemma4", &global_default), "rope_scaling");
}

#[test]
fn nested_text_activation_and_alias_are_preserved() {
    for key in ["hidden_act", "hidden_activation"] {
        let cfg = config(
            "mistral",
            &format!(r#""hidden_act":"silu","text_config":{{"{key}":"gelu"}}"#),
        );
        assert_eq!(cfg.hidden_act.as_deref(), Some("gelu"));
        assert!(for_config(&cfg).is_none());
        refuses(&cfg, "hidden_act");
    }
}

#[test]
fn load_selection_does_not_bypass_pack_refusal() {
    // A llama-shaped MoE is representable by generic operations, but no registered pack
    // owns its semantic/tensor contract. Do not treat that as permission to load it.
    let cfg = config(
        "llama",
        r#""num_experts":8,"num_experts_per_tok":2,
        "moe_intermediate_size":16"#,
    );
    assert!(ModelPlan::compile(&cfg).is_ok());
    assert!(for_config(&cfg).is_none());
    assert!(matches!(
        compile_for_load(&cfg),
        Err(PlanCompileError::NoMatchingModelPack { .. })
    ));
}

#[test]
fn supported_dense_programs_are_unchanged() {
    for family in ["mistral", "llama", "qwen3"] {
        let implicit = config(family, r#""sliding_window":null"#);
        let explicit = config(
            family,
            r#""sliding_window":null,"hidden_act":"silu",
            "rope_scaling":{"rope_type":"default"}"#,
        );
        let plan = compile_for_load(&explicit).unwrap();
        assert_eq!(plan, compile_for_load(&implicit).unwrap());
        for layer in &plan.layers {
            let AttentionPlan::Full(attention) = &layer.attention else {
                panic!("full attention expected");
            };
            assert_eq!(attention.rope.factors, RopeFactors::None);
            let MlpPlan::Dense(mlp) = &layer.mlp else {
                panic!("dense MLP expected");
            };
            assert_eq!(mlp.activation, ActivationPlan::Silu);
        }
    }
}

#[test]
fn established_olmoe_and_minimax_programs_keep_explicit_packs() {
    for (family, extra, expected_activation) in [
        (
            "olmoe",
            r#""num_experts":8,"num_experts_per_tok":2,"hidden_act":"silu""#,
            ActivationPlan::Silu,
        ),
        (
            "minimax_m3",
            r#""num_local_experts":8,"num_experts_per_tok":2,
            "swiglu_alpha":1.702,"swiglu_limit":7.0,"hidden_act":"swigluoai""#,
            ActivationPlan::SwiGluOai {
                alpha: 1.702,
                limit: 7.0,
            },
        ),
    ] {
        let cfg = config(family, extra);
        let pack = for_config(&cfg).expect("established program needs explicit pack ownership");
        assert_eq!(pack.family, family);
        assert!(
            pack.support.is_none(),
            "registration must not promote qualification"
        );
        let plan = compile_for_load(&cfg).unwrap();
        assert_eq!(plan, ModelPlan::compile(&cfg).unwrap());
        for layer in &plan.layers {
            assert!(matches!(&layer.attention, AttentionPlan::Full(_)));
            let MlpPlan::Moe(moe) = &layer.mlp else {
                panic!("expected MoE")
            };
            assert_eq!(moe.activation, expected_activation);
            if family == "olmoe" {
                assert_eq!(moe.router, RouterPlan::Softmax);
                assert!(moe.shared.is_none());
            }
        }
        for bad in [
            r#""sliding_window":4096"#,
            r#""rope_scaling":{"rope_type":"llama3"}"#,
            r#""hidden_act":"gelu""#,
        ] {
            // Replace the activation for the last case instead of introducing duplicate keys.
            let mut altered = cfg.clone();
            let declaration = config(family, bad);
            altered.window_hint = declaration.window_hint;
            altered.rope_scaling_hint = declaration.rope_scaling_hint;
            altered.hidden_act = declaration.hidden_act.or_else(|| cfg.hidden_act.clone());
            assert!(compile_for_load(&altered).is_err());
        }
    }
}

#[test]
fn minimax_wrapper_keeps_dense_routed_and_shared_programs() {
    let cfg = config(
        "minimax_m3_vl",
        r#""text_config":{"model_type":"minimax_m3_text",
        "num_local_experts":8,"num_experts_per_tok":2,"dense_intermediate_size":32,
        "shared_intermediate_size":8,"moe_layer_freq":[0,1],"use_gemma_norm":true,
        "scoring_func":"sigmoid","use_routing_bias":true,"routed_scaling_factor":2.0,
        "rotary_dim":2,"swiglu_alpha":1.702,"swiglu_limit":7.0}"#,
    );
    let plan = compile_for_load(&cfg).unwrap();
    let MlpPlan::Dense(dense) = &plan.layers[0].mlp else {
        panic!("expected dense layer")
    };
    assert_eq!(dense.intermediate_size, 32);
    assert_eq!(
        dense.activation,
        ActivationPlan::SwiGluOai {
            alpha: 1.702,
            limit: 7.0
        }
    );
    let MlpPlan::Moe(moe) = &plan.layers[1].mlp else {
        panic!("expected routed layer")
    };
    assert_eq!(moe.expert_count, 8);
    assert_eq!(moe.shared.as_ref().unwrap().intermediate_size, 8);
    assert_eq!(
        moe.router,
        RouterPlan::Sigmoid {
            normalize_selected: true,
            scaling_factor: 2.0,
            selection_bias: true
        }
    );
    assert_eq!(
        plan.layers[0].pre_attention_norm.weight_transform,
        WeightTransform::AddOne
    );
    let AttentionPlan::Full(attention) = &plan.layers[0].attention else {
        panic!("expected full attention")
    };
    assert_eq!(attention.rope.dimensions, 2);
}

#[test]
fn step_gguf_checkpoint_factors_and_mtp_survive_pack_selection() {
    for (name, writer) in [
        (
            "trunk",
            crate::micro_gguf::write_step35_meta_only
                as fn(&std::path::Path) -> std::io::Result<()>,
        ),
        ("mtp", crate::micro_gguf::write_step35_mtp_meta_only),
    ] {
        let path =
            std::env::temp_dir().join(format!("memra-537-step-{name}-{}.gguf", std::process::id()));
        writer(&path).unwrap();
        let mut file = crate::GgufFile::open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let implicit_config = ModelConfig::from_gguf(&file);
        assert!(implicit_config.rope_scaling_hint.is_none());
        let implicit = compile_for_load(&implicit_config).unwrap();
        let Err(missing) = compile_for_source(&crate::source::GgufSource(&file)) else {
            panic!("the actual Step header requires factors without a scaling-type key");
        };
        assert!(missing.to_string().contains("requires rope_freqs.weight"));
        file.metadata.insert(
            "step35.rope.scaling.type".into(),
            crate::MetaValue::String("llama3".into()),
        );
        let cfg = ModelConfig::from_gguf(&file);
        assert!(cfg.step35.as_ref().unwrap().rope_freq_factors.is_none());
        let plan = compile_for_load(&cfg).unwrap();
        assert_eq!(plan, implicit);
        // No auto-placement/census is involved: the same source preflight used by
        // both eager loaders must reject the missing declared factor tensor.
        let missing = compile_for_source(&crate::source::GgufSource(&file)).unwrap_err();
        assert!(missing.to_string().contains("requires rope_freqs.weight"));
        let (_, present) = compile_for_source(&ConfigOnlySource {
            config: &cfg,
            rope_tensor: true,
        })
        .unwrap();
        assert_eq!(present, plan);
        let mut hf_factors = cfg.clone();
        hf_factors.step35.as_mut().unwrap().rope_freq_factors = Some(vec![2.0; 32]);
        let (_, normalized) = compile_for_source(&ConfigOnlySource {
            config: &hf_factors,
            rope_tensor: false,
        })
        .unwrap();
        assert_eq!(normalized, plan);
        for layer in plan
            .layers
            .iter()
            .chain(plan.mtp_blocks.iter().map(|block| &block.layer))
        {
            match &layer.attention {
                AttentionPlan::Full(attention) => {
                    assert_eq!(attention.rope.factors, RopeFactors::Checkpoint);
                    assert_eq!(attention.rope.dimensions, 64);
                    assert_eq!(attention.rope.base, 5_000_000.0);
                }
                AttentionPlan::SlidingWindow { attention, window } => {
                    assert_eq!(*window, 512);
                    assert_eq!(attention.rope.factors, RopeFactors::None);
                    assert_eq!(attention.rope.dimensions, 128);
                    assert_eq!(attention.rope.base, 10_000.0);
                }
                _ => panic!("unexpected Step attention"),
            }
        }
    }
}

#[test]
fn step_plan_declares_centered_norms_and_gguf_contract_preserves_folded_weights() {
    use crate::tensor_contract::{CheckpointDialect, ContractOptions, TensorTransform};
    let path =
        std::env::temp_dir().join(format!("memra-541-step-norm-{}.gguf", std::process::id()));
    crate::micro_gguf::write_step35_mtp_meta_only(&path).unwrap();
    let file = crate::GgufFile::open(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    let cfg = ModelConfig::from_gguf(&file);
    let plan = compile_for_load(&cfg).unwrap();
    assert_eq!(plan.output_norm.weight_transform, WeightTransform::AddOne);
    for layer in plan
        .layers
        .iter()
        .chain(plan.mtp_blocks.iter().map(|block| &block.layer))
    {
        assert_eq!(
            layer.pre_attention_norm.weight_transform,
            WeightTransform::AddOne
        );
        assert_eq!(layer.pre_mlp_norm.weight_transform, WeightTransform::AddOne);
    }
    assert!(!plan.mtp_blocks.is_empty());
    for block in &plan.mtp_blocks {
        assert_eq!(
            block.input.embedding_norm.weight_transform,
            WeightTransform::AddOne
        );
        assert_eq!(
            block.input.hidden_norm.weight_transform,
            WeightTransform::AddOne
        );
    }
    let contract = for_config(&cfg)
        .unwrap()
        .compile_tensor_contract(
            &cfg,
            &plan,
            CheckpointDialect::Gguf,
            ContractOptions::default(),
        )
        .unwrap();
    assert!(
        contract
            .requirements
            .iter()
            .all(|r| r.transform == TensorTransform::Identity),
        "GGUF already stores folded weights; it must not acquire a second +1"
    );
}
