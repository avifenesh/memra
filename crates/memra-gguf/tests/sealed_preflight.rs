//! External-crate controls: a public source cannot replace canonical preflight with stale state.
use memra_gguf::{
    config::{HfConfig, ModelConfig},
    model_packs,
    model_plan::ModelPlan,
    source::{TensorSource, TensorView},
};

struct UnboundAdapter {
    config: ModelConfig,
    stale_plan: ModelPlan,
    unexpected_rope: bool,
}
impl UnboundAdapter {
    // A same-named caller helper is not the compiler-private trait hook.
    fn bound_program(&self) -> (&ModelConfig, &ModelPlan) {
        (&self.config, &self.stale_plan)
    }
}
impl TensorSource for UnboundAdapter {
    fn config(&self) -> ModelConfig {
        self.config.clone()
    }
    fn find(&self, name: &str) -> Option<TensorView<'_>> {
        (self.unexpected_rope && name == "rope_freqs.weight").then(|| TensorView {
            bytes: std::borrow::Cow::Borrowed(&[0; 64]),
            ggml_type: memra_gguf::GgmlType::F32,
            ne: vec![16],
        })
    }
}
fn source() -> UnboundAdapter {
    let config = ModelConfig::from_hf(&HfConfig::parse(
        r#"{
        "model_type":"qwen3", "num_hidden_layers":1, "hidden_size":32,
        "num_attention_heads":2, "num_key_value_heads":1, "head_dim":16,
        "intermediate_size":64, "vocab_size":32, "max_position_embeddings":128
    }"#,
    ));
    let stale_plan = model_packs::compile_for_load(&config).unwrap();
    UnboundAdapter {
        config,
        stale_plan,
        unexpected_rope: false,
    }
}

#[test]
fn external_adapter_cannot_bypass_unsupported_config_with_a_valid_stale_plan() {
    let mut adapter = source();
    adapter.config.hidden_act = Some("relu".into());
    assert!(model_packs::compile_for_load(adapter.bound_program().0).is_err());
    let error = model_packs::compile_for_source(&adapter)
        .unwrap_err()
        .to_string();
    assert!(error.contains("hidden_act"), "{error}");
}

#[test]
fn external_adapter_cannot_substitute_an_independently_constructed_plan() {
    let mut adapter = source();
    adapter.stale_plan.vocab_size += 1;
    let (config, actual) = model_packs::compile_for_source(&adapter).unwrap();
    assert_eq!(actual, model_packs::compile_for_load(&config).unwrap());
    assert_ne!(&actual, adapter.bound_program().1);
}

#[test]
fn external_adapter_still_runs_source_backed_rope_preflight() {
    let mut adapter = source();
    adapter.unexpected_rope = true;
    assert!(model_packs::compile_for_load(&adapter.config).is_ok());
    let error = model_packs::compile_for_source(&adapter)
        .unwrap_err()
        .to_string();
    assert!(error.contains("rope_freqs.weight"), "{error}");
}
