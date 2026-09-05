use super::*;
use crate::config::HfConfig;

/// Dense llama-family stack (`llama`, and Mistral's `mistral` spelling of the same program):
/// RMSNorm, GQA full attention with rope over the whole head, SwiGLU MLP, no QK-norm, no
/// biases, no MoE. The plainest transformer memra carries — `qk_norm_presence` already answers
/// `Absent` for `Arch::Llama`, so the canonical plan compiler is the whole execution program.
///
/// Brought up for DictaLM-3.0-24B (Mistral-Small-3.1-24B-Base continued pretraining, Hebrew +
/// English): 40 layers, hidden 5120, 32 heads over 8 KV heads, head_dim 128, ffn 32768,
/// vocab 131072, rope_theta 1e6, no sliding window, untied embeddings.
pub static PACK: ModelPack = ModelPack {
    family: "llama_dense",
    aliases: &["llama", "mistral"],
    config_layout: ConfigLayout::FlatOrTextConfig,
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
    matches_config: |config| {
        matches!(config.arch, Arch::Llama)
            && !config.moe.as_ref().is_some_and(|moe| moe.expert_count > 0)
            // Each of these is a different semantic program reachable from a llama-shaped
            // config; none of them belongs to this pack.
            && config.gemma4.is_none()
            && config.mla.is_none()
            && config.dsv4.is_none()
            && config.m3.is_none()
            && config.hy3.is_none()
            && config.step35.is_none()
            && config.geometry.is_none()
            // A window or a non-identity rope scaling makes this a different attention/RoPE
            // program than the one this pack compiles, and nothing downstream would notice:
            // `attention_window` would answer None (full attention) and the canonical plan
            // would compile `RopeFactors::None`. Mistral-7B-v0.1 (`sliding_window: 4096`) and
            // Llama-3.1 (`rope_type: llama3`) are both llama-shaped and both belong to
            // neither this pack nor any other today, so they must fail closed rather than
            // load as something they are not.
            && config.window_hint.is_none()
            && config
                .rope_scaling_hint
                .as_deref()
                .is_none_or(|kind| kind == "default")
    },
    plan_builder: canonical_plan,
    tensor_schema: canonical_tensor_schema,
    tiny_plan: Some(tiny_plan),
};

fn tiny_plan() -> Result<ModelPlan, PlanCompileError> {
    canonical_plan(&ModelConfig::from_hf(&HfConfig::parse(
        r#"{"model_type":"mistral","num_hidden_layers":2,"hidden_size":8,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
        "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32,
        "rms_norm_eps":0.00001,"rope_theta":1000000.0}"#,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AttentionGateKind;
    use crate::model_plan::{
        ActivationPlan, AttentionPlan, MlpPlan, ResidualTopology, RopeFactors, TensorPresence,
    };
    use crate::tensor_contract::{CheckpointDialect, ContractOptions};

    /// dicta-il/DictaLM-3.0-24B-Thinking `config.json`, verbatim (the Base/FP8/GGUF siblings
    /// carry the same geometry). This is the artifact the family was brought up on, so the
    /// fixture is the checkpoint's own config rather than a hand-written shape.
    const DICTALM_24B: &str = r#"{
      "architectures": ["MistralForCausalLM"],
      "attention_dropout": 0.0,
      "bos_token_id": 1,
      "eos_token_id": 2,
      "head_dim": 128,
      "hidden_act": "silu",
      "hidden_size": 5120,
      "initializer_range": 0.02,
      "intermediate_size": 32768,
      "max_position_embeddings": 65280,
      "model_type": "mistral",
      "num_attention_heads": 32,
      "num_hidden_layers": 40,
      "num_key_value_heads": 8,
      "rms_norm_eps": 1e-05,
      "rope_theta": 1000000.0,
      "sliding_window": null,
      "tie_word_embeddings": false,
      "torch_dtype": "bfloat16",
      "vocab_size": 131072
    }"#;

    fn dictalm_config() -> ModelConfig {
        ModelConfig::from_hf(&HfConfig::parse(DICTALM_24B))
    }

    #[test]
    fn mistral_model_type_is_the_llama_program_and_selects_this_pack() {
        let config = dictalm_config();
        assert_eq!(
            config.arch,
            Arch::Llama,
            "HF `mistral` must map to Arch::Llama"
        );
        let pack =
            super::super::for_config(&config).expect("a dense mistral config must select a pack");
        assert_eq!(pack.family, "llama_dense");
    }

    #[test]
    fn dictalm_24b_compiles_the_dense_gqa_stack() {
        let config = dictalm_config();
        let plan = PACK.compile_plan(&config).expect("plan must compile");

        assert_eq!(plan.hidden_size, 5120);
        assert_eq!(plan.vocab_size, 131072);
        assert_eq!(plan.context_length, 65280);
        assert_eq!(plan.layers.len(), 40);
        assert!(plan.mtp_blocks.is_empty());
        assert!(plan.vision.is_none());

        for layer in &plan.layers {
            let AttentionPlan::Full(attn) = &layer.attention else {
                panic!("every DictaLM layer is full attention (no sliding window in the config)");
            };
            assert_eq!(attn.query_heads, 32);
            assert_eq!(attn.kv_heads, 8);
            assert_eq!(attn.key_head_dim, 128);
            assert_eq!(attn.value_head_dim, 128);
            assert_eq!(attn.rope.dimensions, 128, "rope covers the whole head");
            assert_eq!(attn.rope.base, 1_000_000.0);
            assert_eq!(attn.rope.factors, RopeFactors::None);
            assert_eq!(
                attn.qk_norm,
                TensorPresence::Absent,
                "the llama/mistral stack has no QK-norm — an Optional here would let a \
                 qwen3-shaped checkpoint load as this family"
            );
            assert_eq!(attn.output_gate, AttentionGateKind::None);
            assert_eq!(layer.residual, ResidualTopology::Serial);

            let MlpPlan::Dense(mlp) = &layer.mlp else {
                panic!("DictaLM-3.0-24B is dense, not MoE");
            };
            assert_eq!(mlp.intermediate_size, 32768);
            assert_eq!(mlp.activation, ActivationPlan::Silu);
        }
    }

    #[test]
    fn tensor_contract_compiles_on_both_dialects() {
        let config = dictalm_config();
        let plan = PACK.compile_plan(&config).unwrap();
        for dialect in [CheckpointDialect::HfSafetensors, CheckpointDialect::Gguf] {
            PACK.compile_tensor_contract(&config, &plan, dialect, ContractOptions::default())
                .unwrap_or_else(|e| panic!("tensor contract must compile for {dialect:?}: {e:?}"));
        }
    }

    #[test]
    fn tiny_fixture_compiles() {
        PACK.compile_tiny_plan().expect("tiny fixture must compile");
    }

    #[test]
    fn qwen3_shaped_config_does_not_land_in_this_pack() {
        let qwen = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3","num_hidden_layers":2,"hidden_size":8,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32,
            "rms_norm_eps":0.000001}"#,
        ));
        assert!(!PACK.matches_config(&qwen));
        assert_eq!(super::super::for_config(&qwen).unwrap().family, "qwen3");
    }

    /// Mistral-7B-v0.1 is `model_type: mistral` with `sliding_window: 4096`. Its attention is
    /// a different program from the one this pack compiles, and the failure is silent: this
    /// pack's plan answers `AttentionPlan::Full` and the window is never applied. Before this
    /// pack existed, HF `mistral` was `Arch::Other` and the undeclared-gate check refused it;
    /// the pack must keep that door shut rather than inherit the config on family shape.
    #[test]
    fn a_dense_mistral_with_a_sliding_window_is_refused() {
        let windowed = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"mistral","num_hidden_layers":2,"hidden_size":8,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32,
            "rms_norm_eps":0.00001,"rope_theta":1000000.0,"sliding_window":4096}"#,
        ));
        assert_eq!(windowed.arch, Arch::Llama);
        assert_eq!(windowed.window_hint, Some(4096));
        assert!(
            !PACK.matches_config(&windowed),
            "a windowed dense mistral must not load as the full-attention program"
        );
        assert!(super::super::for_config(&windowed).is_none());
    }

    /// Llama-3.1 is llama-shaped with `rope_scaling.rope_type: llama3`, a per-frequency
    /// divisor program the canonical plan does not compile (it emits `RopeFactors::None`).
    /// Same silent substitution, same refusal.
    #[test]
    fn a_llama3_rope_scaled_config_is_refused() {
        let scaled = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"llama","num_hidden_layers":2,"hidden_size":8,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32,
            "rms_norm_eps":0.00001,"rope_theta":500000.0,
            "rope_scaling":{"rope_type":"llama3","factor":8.0,
            "low_freq_factor":1.0,"high_freq_factor":4.0,
            "original_max_position_embeddings":8192}}"#,
        ));
        assert_eq!(scaled.arch, Arch::Llama);
        assert_eq!(scaled.rope_scaling_hint.as_deref(), Some("llama3"));
        assert!(
            !PACK.matches_config(&scaled),
            "llama3 rope scaling must not compile as identity rope"
        );
        assert!(super::super::for_config(&scaled).is_none());
    }

    /// The happy path stays open: DictaLM has no window and no rope scaling.
    #[test]
    fn the_dictalm_config_still_matches() {
        let dicta = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"mistral","num_hidden_layers":40,"hidden_size":5120,
            "num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,
            "intermediate_size":32768,"vocab_size":131072,
            "max_position_embeddings":131072,"rms_norm_eps":0.00001,
            "rope_theta":1000000.0,"sliding_window":null}"#,
        ));
        assert_eq!(dicta.window_hint, None);
        assert_eq!(dicta.rope_scaling_hint, None);
        assert!(PACK.matches_config(&dicta));
        assert_eq!(
            super::super::for_config(&dicta).unwrap().family,
            "llama_dense"
        );
    }
}
