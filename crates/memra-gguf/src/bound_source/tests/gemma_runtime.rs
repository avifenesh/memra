use super::*;

// Same complete registered text programs as Carver's two preserved real-file repros.
const DENSE: &str = r#"{"model_type":"gemma4","text_config":{"model_type":"gemma4_text",
"num_hidden_layers":2,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,
"num_global_key_value_heads":1,"head_dim":4,"global_head_dim":4,"intermediate_size":16,
"vocab_size":32,"max_position_embeddings":64,"rms_norm_eps":0.000001,"sliding_window":8,
"layer_types":["sliding_attention","full_attention"],"rope_parameters":{
"full_attention":{"rope_theta":10000,"partial_rotary_factor":0.5},
"sliding_attention":{"rope_theta":10000}}}}"#;

fn moe_config() -> String {
    DENSE.replace("\"intermediate_size\":16,", "\"intermediate_size\":16,\"moe_intermediate_size\":8,\"num_experts\":4,\"top_k_experts\":2,")
}

struct BoundOnly<'a>(&'a SafetensorsSource);
impl TensorSource for BoundOnly<'_> {
    fn config(&self) -> ModelConfig {
        self.0.config()
    }
    fn raw_config_json(&self) -> Option<&str> {
        self.0.raw_config_json()
    }
    fn bound_interpretation(&self) -> Result<BoundSourceInterpretation, String> {
        self.0.bound_interpretation()
    }
    fn runtime_metadata(&self) -> Result<crate::source::RuntimeSourceMetadata, String> {
        self.0.runtime_metadata()
    }
    fn tensor_census(&self) -> Result<TensorCensus, String> {
        self.0.tensor_census()
    }
    fn validate_bound_metadata(&self, r: &BoundTensorRequest<'_>) -> Result<(), String> {
        self.0.validate_bound_metadata(r)
    }
    fn read_bound(&self, r: &BoundTensorRequest<'_>) -> Result<TensorView<'_>, String> {
        self.0.read_bound(r)
    }
    fn auxiliary_bound(
        &self,
        r: &BoundTensorRequest<'_>,
        kind: QuantAuxTensor,
    ) -> Result<Option<TensorView<'_>>, String> {
        self.0.auxiliary_bound(r, kind)
    }
    fn find(&self, _: &str) -> Option<TensorView<'_>> {
        panic!("Gemma root fell back to raw name lookup")
    }
}

#[test]
fn prepared_gemma_moe_enters_loader_with_semantic_scales_fused_banks_and_shared_aliases() {
    let mut expected = BTreeMap::new();
    let f = fixture(&moe_config(), |rows| {
        // Distinct values catch aliases selecting a different, same-shaped physical operand.
        for (i, (name, t)) in rows.iter_mut().enumerate() {
            t.bytes = (0..t.shape.iter().product::<u64>())
                .flat_map(|j| (i as f32 + j as f32 / 1024.0).to_le_bytes())
                .collect();
            expected.insert(name.clone(), (t.shape.clone(), t.bytes.clone()));
        }
        // A quantization macro on the router is separate from the operation-owned input scale.
        rows.insert(
            "model.layers.0.router.proj.weight_scale_2".into(),
            float(vec![1], 7.0),
        );
        rows.insert(
            "model.layers.0.self_attn.q_proj.weight_scale_2".into(),
            float(vec![1], 11.0),
        );
    });
    let raw = SafetensorsSource::open(&f.dir).unwrap();
    let (cfg, plan) = model_packs::compile_for_source(&raw).unwrap();
    assert_eq!(model_packs::for_config(&cfg).unwrap().family, "gemma4_moe");
    let guarded = BoundOnly(&raw);
    let prepared = PreparedModelSource::text(&guarded).unwrap();
    let mut called = false;
    prepared
        .with_runtime(|runtime| {
            called = true;
            assert_eq!(model_packs::compile_for_source(runtime).unwrap().1, plan);
            for index in 0..2 {
                // Exact spellings used by load_ffn and the Gemma4LayerBits loader.
                for (alias, physical) in [
                    ("attn_norm.weight", "input_layernorm.weight"),
                    (
                        "post_attention_norm.weight",
                        "post_attention_layernorm.weight",
                    ),
                    ("ffn_norm.weight", "pre_feedforward_layernorm.weight"),
                    ("post_ffw_norm.weight", "post_feedforward_layernorm.weight"),
                    ("layer_output_scale.weight", "layer_scalar"),
                    ("ffn_gate.weight", "mlp.gate_proj.weight"),
                    ("ffn_up.weight", "mlp.up_proj.weight"),
                    ("ffn_down.weight", "mlp.down_proj.weight"),
                    ("ffn_gate_shexp.weight", "mlp.gate_proj.weight"),
                    ("ffn_up_shexp.weight", "mlp.up_proj.weight"),
                    ("ffn_down_shexp.weight", "mlp.down_proj.weight"),
                    ("ffn_gate_inp.weight", "router.proj.weight"),
                    ("ffn_gate_inp.scale", "router.scale"),
                    ("ffn_down_exps.scale", "router.per_expert_scale"),
                    (
                        "post_ffw_norm_1.weight",
                        "post_feedforward_layernorm_1.weight",
                    ),
                    (
                        "pre_ffw_norm_2.weight",
                        "pre_feedforward_layernorm_2.weight",
                    ),
                    (
                        "post_ffw_norm_2.weight",
                        "post_feedforward_layernorm_2.weight",
                    ),
                    ("ffn_gate_up_exps.weight", "experts.gate_up_proj"),
                    ("ffn_down_exps.weight", "experts.down_proj"),
                ] {
                    let name = format!("blk.{index}.{alias}");
                    assert!(runtime.try_has(&name).unwrap(), "{name}");
                    let view = runtime.try_find(&name).unwrap().unwrap();
                    let (shape, bytes) = &expected[&format!("model.layers.{index}.{physical}")];
                    assert_eq!(
                        view.ne,
                        shape.iter().rev().copied().collect::<Vec<_>>(),
                        "{name}"
                    );
                    assert_eq!(view.bytes.as_ref(), bytes, "{name}");
                }
                assert!(
                    !runtime
                        .try_has(&format!("blk.{index}.ffn_gate_exps.weight"))
                        .unwrap()
                );
                assert!(
                    !runtime
                        .try_has(&format!("blk.{index}.ffn_up_exps.weight"))
                        .unwrap()
                );
                assert!(
                    runtime
                        .try_find(&format!("blk.{index}.exp_probs_b.bias"))
                        .unwrap()
                        .is_none()
                );
                assert!(
                    !runtime
                        .try_has(&format!("blk.{index}.inp_gate.weight"))
                        .unwrap()
                );
            }
            assert_eq!(
                runtime
                    .try_find("blk.0.attn_q.scale")
                    .unwrap()
                    .unwrap()
                    .bytes
                    .as_ref(),
                11.0f32.to_le_bytes()
            );
            assert_eq!(
                runtime
                    .try_find("blk.0.ffn_gate_inp.scale")
                    .unwrap()
                    .unwrap()
                    .ne,
                [8]
            );
            assert_eq!(
                runtime
                    .try_find("blk.0.ffn_down_exps.scale")
                    .unwrap()
                    .unwrap()
                    .ne,
                [4]
            );
            let fused = runtime
                .try_find("blk.0.ffn_gate_up_exps.weight")
                .unwrap()
                .unwrap();
            assert_eq!(fused.ne, [8, 16, 4]); // retained encoded gate-then-up rows; no made-up split bank
            for name in [
                "blk.0.ffn_gate_inp.scal",
                "blk.0.ffn_unknown.scale",
                "blk.99.ffn_gate_inp.scale",
                "blk.99.ffn_gate_up_exps.weight",
                "blk.00.ffn_gate_inp.scale",
                "blk.+0.ffn_down_exps.scale",
            ] {
                assert!(runtime.try_find(name).is_err(), "{name}");
            }
        })
        .unwrap();
    assert!(called);
}

#[test]
fn prepared_gemma_dense_preserves_existing_optional_loader_probes() {
    let f = fixture(DENSE, |_| {});
    let raw = SafetensorsSource::open(&f.dir).unwrap();
    assert_eq!(
        model_packs::for_config(&raw.config()).unwrap().family,
        "gemma4_dense"
    );
    assert!(raw.try_find("blk.0.ffn_gate_inp.scale").unwrap().is_none());
    let guarded = BoundOnly(&raw);
    PreparedModelSource::text(&guarded)
        .unwrap()
        .with_runtime(|runtime| {
            for index in 0..2 {
                for suffix in [
                    "ffn_gate_inp.scale",
                    "ffn_gate_inp.weight",
                    "ffn_down_exps.scale",
                    "ffn_gate_up_exps.weight",
                    "inp_gate.weight",
                ] {
                    let name = format!("blk.{index}.{suffix}");
                    assert!(runtime.try_find(&name).unwrap().is_none(), "{name}");
                    assert!(!runtime.try_has(&name).unwrap(), "{name}");
                }
                for suffix in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                    let name = format!("blk.{index}.{suffix}");
                    assert_eq!(
                        runtime.try_find(&name).unwrap().unwrap().bytes,
                        raw.try_find(&name).unwrap().unwrap().bytes
                    );
                }
            }
            for name in [
                "blk.99.ffn_gate_inp.scale",
                "blk.0.ffn_gate_inp.scales",
                "blk.0.inp_gate_typo.weight",
            ] {
                assert!(runtime.try_find(name).is_err(), "{name}");
            }
        })
        .unwrap();
}

#[test]
fn prepared_gemma_moe_keeps_physical_gguf_refusal_and_strict_complete_census() {
    let config = moe_config();
    let cfg = ModelConfig::from_hf(&HfConfig::try_parse(&config).unwrap());
    let pack = model_packs::for_config(&cfg).unwrap();
    let plan = pack.compile_plan(&cfg).unwrap();
    let physical = pack
        .compile_tensor_contract(
            &cfg,
            &plan,
            CheckpointDialect::Gguf,
            pack.contract_options(&cfg),
        )
        .unwrap_err();
    assert!(
        physical
            .to_string()
            .contains("gemma parallel MoE non-safetensors schema")
    );
    let e4b = fixture(DENSE, |_| {});
    std::fs::write(
        e4b.dir.join("config.json"),
        DENSE.replace(
            "\"hidden_size\":8",
            "\"hidden_size\":8,\"hidden_size_per_layer_input\":4",
        ),
    )
    .unwrap();
    assert!(PreparedModelSource::text(&SafetensorsSource::open(&e4b.dir).unwrap()).is_err());
    for bad in [
        "missing-router-scale",
        "missing-output-scale",
        "wrong-scale-shape",
        "unknown",
        "ambiguous",
        "undeclared-per-layer-gate",
    ] {
        let f = fixture(&config, |rows| match bad {
            "missing-router-scale" => {
                rows.remove("model.layers.0.router.scale");
            }
            "missing-output-scale" => {
                rows.remove("model.layers.0.router.per_expert_scale");
            }
            "wrong-scale-shape" => {
                rows.insert("model.layers.0.router.scale".into(), float(vec![1], 7.0));
            }
            "unknown" => {
                rows.insert(
                    "model.layers.0.router.misspelled".into(),
                    float(vec![8], 0.0),
                );
            }
            "ambiguous" => {
                rows.insert(
                    "model.language_model.layers.0.router.scale".into(),
                    float(vec![8], 0.0),
                );
            }
            "undeclared-per-layer-gate" => {
                rows.insert("model.layers.0.inp_gate.weight".into(), float(vec![8], 0.0));
            }
            _ => unreachable!(),
        });
        let raw = SafetensorsSource::open(&f.dir).unwrap();
        let spy = Spy {
            source: &raw,
            reads: AtomicUsize::new(0),
        };
        assert!(PreparedModelSource::text(&spy).is_err(), "{bad}");
        assert_eq!(spy.reads.load(Ordering::Relaxed), 0, "{bad}");
    }
}
