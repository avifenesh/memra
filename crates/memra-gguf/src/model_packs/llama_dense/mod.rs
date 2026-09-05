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
    // Loader lane until the family's own gates run on a real checkpoint: the plan and the
    // tensor contract compile, which is not the same claim as native execution being right.
    // This flips to NativeReference (with a parity gate beside it) on the DictaLM-3.0-24B
    // NVFP4 checkpoint-parity receipt, not before — LAW:no-generic-support.
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
}
