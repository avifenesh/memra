use std::process::Command;
#[test]
fn config_cli_returns_errors_for_escaped_unsupported_and_malformed_declarations() {
    let root =
        std::env::temp_dir().join(format!("memra-cli-canonical-config-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let base = r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":32,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,"intermediate_size":64,"vocab_size":32,"max_position_embeddings":128}"#;
    for (field, expected) in [
        (r#""hidden_act":"relu","#, "hidden_act=relu"),
        (r#""hidden\u005fact":"relu","#, "hidden_act=relu"),
        (r#""hidden_act":17,"#, "hidden_act"),
        (
            r#""hidden_act":"silu","hidden\u005fact":"relu","#,
            "duplicate JSON key",
        ),
    ] {
        std::fs::write(
            root.join("config.json"),
            base.replacen('{', &format!("{{{field}"), 1),
        )
        .unwrap();
        let result = memra_cli::verify_model(memra_cli::VerifyRequest {
            stage: memra_cli::VerifyStage::Config,
            source: root.display().to_string(),
            against: "qwen3".into(),
            out_dir: None,
            oracle: None,
            native_runner: None,
        });
        let error = match result {
            Ok(_) => panic!("declaration accepted: {field}"),
            Err(error) => error,
        };
        assert!(error.to_string().contains(expected), "{error}");
    }
    std::fs::write(
        root.join("config.json"),
        base.replacen('{', "{\"hidden_act\":17,", 1),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_memra"))
        .args(["model", "inspect"])
        .arg(&root)
        .args(["--against", "qwen3", "--out"])
        .arg(root.join("out"))
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(err.contains("hidden_act"), "{err}");
    assert!(!err.contains("panicked"), "{err}");
    std::fs::remove_dir_all(root).unwrap();
}
