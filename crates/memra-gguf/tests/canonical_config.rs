use memra_gguf::{
    config::{HfConfig, ModelConfig},
    execution_manifest::execution_rewrites,
    model_packs,
};
const BASE: &str = r#"{"model_type":"qwen3","num_hidden_layers":2,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"intermediate_size":16,"vocab_size":32,"max_position_embeddings":8192}"#;
fn with(fields: &str) -> String {
    BASE.replacen('{', &format!("{{{fields}"), 1)
}
fn plan_hash(text: &str) -> String {
    let cfg = ModelConfig::from_hf(&HfConfig::try_parse(text).unwrap());
    execution_rewrites(&model_packs::compile_for_load(&cfg).unwrap())[0]
        .plan_sha256
        .clone()
}
#[test]
fn decoded_activation_declarations_cannot_disappear_before_pack_admission() {
    for fields in [
        r#""hidden_act":"relu","#,
        r#""hidden\u005fact":"relu","#,
        r#""hidden_act":"r\u0065lu","#,
        r#""text_config":{"hidden\u005fact":"relu"},"#,
    ] {
        let cfg = ModelConfig::from_hf(&HfConfig::try_parse(&with(fields)).unwrap());
        assert_eq!(cfg.hidden_act.as_deref(), Some("relu"));
        assert!(
            model_packs::compile_for_load(&cfg)
                .unwrap_err()
                .to_string()
                .contains("hidden_act=relu")
        );
    }
}
#[test]
fn equivalent_decoded_keys_values_and_arrays_keep_the_same_plan() {
    let literal = with(r#""hidden_act":"silu","name_or_path":"café-😀","#);
    let escaped = with(r#""hidden\u005fact":"si\u006cu","name_or_path":"caf\u00e9-\ud83d\ude00","#);
    assert_eq!(plan_hash(&literal), plan_hash(&escaped));
    let literal = with(
        r#""text_config":{"hidden_act":"silu","hidden_size":16,"head_dim":8,"intermediate_size":32},"#,
    );
    let escaped = with(
        r#""text\u005fconfig":{"hidden\u005fact":"si\u006cu","hidden\u005fsize":16,"head\u005fdim":8,"intermediate_size":32},"#,
    );
    assert_eq!(plan_hash(&literal), plan_hash(&escaped));
    let cfg = HfConfig::try_parse(&with(
        r#""layer_types":["sliding\u005fattention","full_attention"],"#,
    ))
    .unwrap();
    assert_eq!(cfg.gemma4_swa_pattern, Some(vec![true, false]));
    assert_eq!(
        cfg.layer_types,
        Some(vec!["sliding_attention".into(), "full_attention".into()])
    );
}
#[test]
fn malformed_and_duplicate_decoded_declarations_refuse_recursively() {
    for text in [
        with(r#""hidden_act":"relu","hidden_act":"silu","#),
        with(r#""hidden_act":"silu","hidden\u005fact":"relu","#),
        with(r#""text_config":{"hidden_act":"silu","hidden\u005fact":"relu"},"#),
        with(r#""unused":[{"x":1,"\u0078":2}],"#),
        with(r#""name_or_path":"\ud800","#),
        with(r#""name_or_path":"\q","#),
        format!("garbage{BASE}"),
        format!("{BASE} trailing"),
        BASE[..BASE.len() - 1].to_owned(),
        "[]".into(),
    ] {
        assert!(HfConfig::try_parse(&text).is_err(), "{text}");
    }
}
#[test]
fn present_wrong_types_and_lossy_numeric_conversions_refuse() {
    for field in [
        r#""hidden_act":17,"#,
        r#""hidden_act":null,"#,
        r#""hidden_act":[],"#,
        r#""hidden_act":"silu","hidden_activation":17,"#,
        r#""text_config":{"num_hidden_layers":1.5},"#,
        r#""text_config":{"num_attention_heads":4294967296},"#,
        r#""text_config":{"num_attention_heads":-1},"#,
        r#""rms_norm_eps":1e100,"#,
        r#""qk_norm":1,"#,
        r#""text_config":[],"#,
        r#""layer_types":["full_attention",9],"#,
        r#""partial_rotary_factors":[1.0,"bad"],"#,
        r#""eos_token_id":[1,-1],"#,
        r#""architectures":["Qwen3ForCausalLM",null],"#,
    ] {
        assert!(HfConfig::try_parse(&with(field)).is_err(), "{field}");
    }
}

#[test]
fn numeric_literals_are_not_rounded_through_f64_before_normalization() {
    for decimal in [
        "1.0000000596046448",
        "-0",
        "-0.0",
        "1e-46",
        "1.401298464324817e-45",
    ] {
        let cfg = HfConfig::try_parse(&with(&format!(
            "\"rms_norm_eps\":{decimal},\"partial_rotary_factors\":[{decimal}],"
        )))
        .unwrap();
        let expected = decimal.parse::<f32>().unwrap().to_bits();
        assert_eq!(cfg.rms_norm_eps.to_bits(), expected, "scalar {decimal}");
        assert_eq!(
            cfg.partial_rotary_factors.unwrap()[0].to_bits(),
            expected,
            "array {decimal}"
        );
    }
}

#[test]
fn glm_interval_one_preserves_the_existing_program_and_other_intervals_refuse() {
    // The pinned GLM-5.2 config's scheduling declarations, without model weights.
    let text = r#"{"model_type":"glm_moe_dsa","first_k_dense_replace":3,"moe_layer_freq": 1}"#;
    let cfg = HfConfig::try_parse(text).unwrap();
    assert_eq!(cfg.first_k_dense_replace, Some(3));
    assert_eq!(cfg.moe_layer_freq, None);
    assert!(text.contains("\"moe_layer_freq\": 1"));
    for value in [
        "0",
        "2",
        "1.0000000000000001",
        "-1",
        "null",
        "true",
        "\"1\"",
    ] {
        let altered = text.replace(
            "\"moe_layer_freq\": 1",
            &format!("\"moe_layer_freq\": {value}"),
        );
        let error = HfConfig::try_parse(&altered).unwrap_err();
        assert!(error.contains("moe_layer_freq"), "{value}: {error}");
    }
    let mask =
        HfConfig::try_parse(r#"{"model_type":"minimax_m3","moe_layer_freq":[0,1,1]}"#).unwrap();
    assert_eq!(mask.moe_layer_freq, Some(vec![0, 1, 1]));
    assert!(HfConfig::try_parse(r#"{"model_type":"minimax_m3","moe_layer_freq":1}"#).is_err());
    assert!(HfConfig::try_parse(r#"{"model_type":"wrapper","moe_layer_freq":1,"text_config":{"model_type":"glm_moe_dsa"}}"#).is_ok());
    assert!(HfConfig::try_parse(r#"{"model_type":"glm_moe_dsa","moe_layer_freq":1,"text_config":{"model_type":"minimax_m3"}}"#).is_err());
}

#[test]
fn fractional_integer_fields_cannot_round_to_a_whole_number() {
    for decimal in [
        "2.0000000000000001",
        "1.99999999999999999",
        "20.000000000000001e-1",
        "1e-1000",
    ] {
        let text = with(&format!(
            "\"text_config\":{{\"num_attention_heads\":{decimal}}},"
        ));
        assert!(HfConfig::try_parse(&text).is_err(), "{decimal}");
    }
}

#[test]
fn integer_valued_decimals_preserve_range_without_float_casts() {
    for number in ["2", "2.0", "2e0", "20e-1", "0.02e+2"] {
        let text = BASE.replace(
            "\"num_attention_heads\":2",
            &format!("\"num_attention_heads\":{number}"),
        );
        assert_eq!(plan_hash(BASE), plan_hash(&text), "{number}");
    }
    for number in [
        "18446744073709551615",
        "18446744073709551615.0",
        "184467440737095516150e-1",
    ] {
        let text = with(&format!("\"ngram_vocab_size_base\":{number},"));
        assert_eq!(
            HfConfig::try_parse(&text).unwrap().ngram_vocab_size_base,
            Some(u64::MAX)
        );
    }
    for number in [
        "18446744073709551616",
        "18446744073709551616.0",
        "1e1000",
        "-1",
    ] {
        let text = with(&format!("\"ngram_vocab_size_base\":{number},"));
        assert!(HfConfig::try_parse(&text).is_err(), "{number}");
    }
}
#[test]
fn documented_absence_null_and_scalar_vector_forms_remain_distinct_from_errors() {
    assert_eq!(
        plan_hash(BASE),
        plan_hash(&with(
            r#""sliding_window":null,"rope_scaling":null,"vision_config":null,"#
        ))
    );
    let whole = BASE.replace("\"num_attention_heads\":2", "\"num_attention_heads\":2.0");
    assert_eq!(plan_hash(BASE), plan_hash(&whole));
    let cfg = HfConfig::try_parse(&with(
        r#""eos_token_id":[7,8],"rope_theta":[10000,1000000],"#,
    ))
    .unwrap();
    assert_eq!(cfg.eos_token_id, Some(7));
    assert_eq!(cfg.rope_theta_layers, Some(vec![10000.0, 1000000.0]));
}
#[test]
fn malformed_file_config_refuses_before_checkpoint_io() {
    let root = std::env::temp_dir().join(format!("memra-config-strict-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("config.json"), with(r#""hidden_act":17,"#)).unwrap();
    let e = ModelConfig::from_config_json(&root.join("config.json")).unwrap_err();
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
    assert!(e.to_string().contains("hidden_act"));
    let e = match memra_gguf::source::SafetensorsSource::open(&root) {
        Ok(_) => panic!("malformed config accepted"),
        Err(e) => e,
    };
    assert!(e.to_string().contains("hidden_act"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn valid_literal_plan_hashes_match_frozen_pre_repair_ac3c39712() {
    // Generated against immutable ac3c39712486170ce2beb77527df50eede07070c,
    // before changing the parser. These are semantic-plan hashes, not parser snapshots.
    for line in include_str!("fixtures/canonical-config-plans.tsv").lines() {
        let mut fields = line.splitn(3, '\t');
        let name = fields.next().unwrap();
        let expected = fields.next().unwrap();
        let config = fields.next().unwrap();
        assert_eq!(
            plan_hash(config),
            expected,
            "literal program changed: {name}"
        );
    }
}

#[test]
fn explicit_config_override_does_not_hide_a_malformed_present_config_file() {
    let root = std::env::temp_dir().join(format!(
        "memra-config-override-strict-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("config.json"), with(r#""hidden_act":17,"#)).unwrap();
    let cfg = ModelConfig::from_hf(&HfConfig::try_parse(BASE).unwrap());
    let error = match memra_gguf::source::SafetensorsSource::open_with_config(&root, cfg) {
        Ok(_) => panic!("present malformed config ignored"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("hidden_act"), "{error}");
    std::fs::remove_dir_all(root).unwrap();
}
