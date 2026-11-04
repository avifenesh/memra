//! Gemma-4 E4B: the dense gemma4 program plus per-layer input embeddings (PLE) and KV sharing.
//!
//! Both extra pieces are typed plan state, not loader conventions: `ModelPlan::per_layer_input`
//! carries the PLE prologue and per-layer tail, `StatePlan::SharedKvCache` marks the trailing
//! layers that own no K/V and read an earlier layer's cache. E4B global layers also ship their
//! own V projection (`attention_k_eq_v = false`), unlike the 26B/31B globals.

use super::*;
use crate::config::HfConfig;

const E4B_LAYERS: u32 = 42;
const E4B_SHARED_KV_LAYERS: u32 = 18;
const E4B_PER_LAYER_WIDTH: u32 = 256;

pub static PACK: ModelPack = ModelPack {
    family: "gemma4_e4b",
    output_head: OutputHeadContract::TiedHeadAllowed,
    tensor_consumption: TensorConsumption::Report,
    aliases: &["gemma4_e4b"],
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
    // The support record was measured on E4B only, so the match is pinned to its geometry: 42
    // trunk layers, the last 18 KV-shared, per-layer inputs 256 wide. Another checkpoint with
    // per-layer inputs and KV sharing gets no pack until it is qualified and recorded.
    matches_config: |config| {
        matches!(config.arch, Arch::Gemma4)
            && !config.moe.as_ref().is_some_and(|moe| moe.expert_count > 0)
            && config.n_layer == E4B_LAYERS
            && config.nextn_predict_layers == 0
            && config.gemma4.as_ref().is_some_and(|gemma| {
                gemma.n_embd_per_layer == E4B_PER_LAYER_WIDTH
                    && gemma.shared_kv_layers == E4B_SHARED_KV_LAYERS
            })
    },
    plan_builder: canonical_plan,
    tensor_schema: super::gemma4_dense::tensor_schema,
    tiny_plan: Some(tiny_plan),
};

/// Four layers cover every E4B layer class: own-KV sliding (0), own-KV global (1), and the
/// shared sliding (2 reads 0) and shared global (3 reads 1) layers.
fn tiny_plan() -> Result<ModelPlan, PlanCompileError> {
    canonical_plan(&ModelConfig::from_hf(&HfConfig::parse(
        r#"{"model_type":"gemma4","text_config":{"model_type":"gemma4_text",
        "num_hidden_layers":4,"hidden_size":8,"num_attention_heads":2,
        "num_key_value_heads":1,"num_global_key_value_heads":1,"head_dim":4,
        "global_head_dim":8,"intermediate_size":16,"vocab_size":32,
        "max_position_embeddings":64,"rms_norm_eps":0.000001,"sliding_window":8,
        "hidden_size_per_layer_input":4,"num_kv_shared_layers":2,"attention_k_eq_v":false,
        "layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"],
        "rope_parameters":{"full_attention":{"rope_theta":10000,
        "partial_rotary_factor":0.5},"sliding_attention":{"rope_theta":10000}}}}"#,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_plan::{AttentionPlan, OperationKind, StatePlan, ValueProjection};
    use crate::tensor_contract::{LayerTensor, OutputHead, TensorContractError, TensorId};

    /// `google/gemma-4-E4B-it` text_config geometry (config.json, transformers 5.5.0.dev0).
    fn official_e4b() -> ModelConfig {
        let layer_types: Vec<&str> = (0..42)
            .map(|layer| {
                if layer % 6 == 5 {
                    "\"full_attention\""
                } else {
                    "\"sliding_attention\""
                }
            })
            .collect();
        ModelConfig::from_hf(&HfConfig::parse(&format!(
            r#"{{"model_type":"gemma4","text_config":{{"model_type":"gemma4_text",
            "num_hidden_layers":42,"hidden_size":2560,"num_attention_heads":8,
            "num_key_value_heads":2,"head_dim":256,"global_head_dim":512,
            "intermediate_size":10240,"vocab_size":262144,"max_position_embeddings":131072,
            "rms_norm_eps":0.000001,"sliding_window":512,"final_logit_softcapping":30,
            "hidden_size_per_layer_input":256,"num_kv_shared_layers":18,
            "attention_k_eq_v":false,"layer_types":[{}],
            "rope_parameters":{{"full_attention":{{"rope_theta":1000000,
            "partial_rotary_factor":0.25,"rope_type":"proportional"}},
            "sliding_attention":{{"rope_theta":10000,"rope_type":"default"}}}}}}}}"#,
            layer_types.join(",")
        )))
    }

    #[test]
    fn e4b_lands_on_this_pack_and_only_this_pack() {
        let config = official_e4b();
        assert_eq!(for_config(&config).unwrap().family, "gemma4_e4b");
        assert!(!super::super::gemma4_dense::PACK.matches_config(&config));
        assert!(!super::super::gemma4_moe::PACK.matches_config(&config));
        assert!(compile_for_load(&config).is_ok());
    }

    /// A sibling with per-layer inputs and KV sharing but another geometry (35 layers, 20
    /// shared, the E2B shape) binds to no pack: the record covers E4B only.
    #[test]
    fn another_per_layer_input_geometry_gets_no_pack() {
        let mut config = official_e4b();
        config.n_layer = 35;
        let gemma = config.gemma4.as_mut().unwrap();
        gemma.shared_kv_layers = 20;
        gemma.swa_pattern.truncate(35);
        gemma.head_count_kv.truncate(35);
        assert!(for_config(&config).is_none());
        assert!(matches!(
            compile_for_load(&config),
            Err(PlanCompileError::NoMatchingModelPack { .. })
        ));
        let mut width = official_e4b();
        width.gemma4.as_mut().unwrap().n_embd_per_layer = 128;
        assert!(for_config(&width).is_none());
    }

    #[test]
    fn official_geometry_compiles_per_layer_input_and_shared_kv() {
        let config = official_e4b();
        let plan = PACK.compile_plan(&config).unwrap();
        let per_layer = plan.per_layer_input.as_ref().unwrap();
        assert_eq!((per_layer.width, per_layer.layers), (256, 42));
        assert_eq!(per_layer.token_scale, 16.0);
        assert_eq!(per_layer.projection_scale, 1.0 / 2560f32.sqrt());
        for layer in &plan.layers[..24] {
            assert!(
                !matches!(layer.state, StatePlan::SharedKvCache { .. }),
                "layer {} owns its cache",
                layer.index
            );
        }
        for layer in &plan.layers[24..] {
            let expected = if layer.index % 6 == 5 { 23 } else { 22 };
            assert_eq!(
                layer.state,
                StatePlan::SharedKvCache {
                    source_layer: expected
                },
                "layer {}",
                layer.index
            );
        }
        let AttentionPlan::Full(global) = &plan.layers[5].attention else {
            panic!("layer 5 is global attention");
        };
        assert_eq!(global.value_projection, ValueProjection::Separate);
        assert_eq!((global.kv_heads, global.key_head_dim), (2, 512));
        let operations = plan.operations();
        for operation in [
            OperationKind::PerLayerInputEmbedding,
            OperationKind::PerLayerInputGate,
            OperationKind::SharedKvState,
            OperationKind::GemmaResidual,
        ] {
            assert!(operations.contains(&operation), "{operation:?}");
        }
    }

    #[test]
    fn gguf_contract_drops_shared_kv_tensors_and_adds_the_per_layer_tail() {
        let config = official_e4b();
        let plan = PACK.compile_plan(&config).unwrap();
        let contract = PACK
            .compile_tensor_contract(
                &config,
                &plan,
                CheckpointDialect::Gguf,
                ContractOptions {
                    output_head: OutputHead::TiedToEmbedding,
                },
            )
            .unwrap();
        let find = |id: &TensorId| {
            contract
                .requirements
                .iter()
                .find(|requirement| &requirement.id == id)
        };
        let layer = |index, tensor| TensorId::Layer { index, tensor };
        assert_eq!(
            find(&TensorId::PerLayerTokenEmbedding).unwrap().shape,
            [10_752, 262_144]
        );
        assert_eq!(
            find(&TensorId::PerLayerModelProjection).unwrap().shape,
            [2560, 10_752]
        );
        assert_eq!(
            find(&TensorId::PerLayerProjectionNorm).unwrap().shape,
            [256]
        );
        for index in [0, 5, 23] {
            for tensor in [LayerTensor::Key, LayerTensor::Value, LayerTensor::KeyNorm] {
                assert!(find(&layer(index, tensor)).is_some(), "{index} {tensor:?}");
            }
        }
        assert_eq!(
            find(&layer(5, LayerTensor::Key)).unwrap().shape,
            [2560, 1024]
        );
        for index in [24, 29, 41] {
            for tensor in [LayerTensor::Key, LayerTensor::Value, LayerTensor::KeyNorm] {
                assert!(find(&layer(index, tensor)).is_none(), "{index} {tensor:?}");
            }
            assert!(find(&layer(index, LayerTensor::Query)).is_some());
            assert!(find(&layer(index, LayerTensor::QueryNorm)).is_some());
        }
        for index in [0, 41] {
            assert_eq!(
                find(&layer(index, LayerTensor::PerLayerInputGate))
                    .unwrap()
                    .shape,
                [2560, 256]
            );
            assert_eq!(
                find(&layer(index, LayerTensor::PerLayerProjection))
                    .unwrap()
                    .shape,
                [256, 2560]
            );
            assert_eq!(
                find(&layer(index, LayerTensor::PerLayerPostNorm))
                    .unwrap()
                    .names,
                [format!("blk.{index}.post_norm.weight")]
            );
        }
    }

    /// No safetensors loader binds the PLE names, so the HF schema refuses instead of guessing.
    #[test]
    fn safetensors_schema_refuses_per_layer_input() {
        let config = official_e4b();
        let plan = PACK.compile_plan(&config).unwrap();
        assert!(matches!(
            PACK.compile_tensor_contract(
                &config,
                &plan,
                CheckpointDialect::HfSafetensors,
                ContractOptions {
                    output_head: OutputHead::TiedToEmbedding,
                },
            ),
            Err(TensorContractError::UnsupportedPlanOperation { .. })
        ));
    }

    #[test]
    fn tiny_plan_covers_every_layer_class() {
        let plan = PACK.compile_tiny_plan().unwrap();
        assert!(plan.vision.is_none());
        assert_eq!(plan.per_layer_input.as_ref().unwrap().width, 4);
        let states: Vec<_> = plan.layers.iter().map(|layer| layer.state).collect();
        assert!(matches!(states[0], StatePlan::SlidingKvCache { .. }));
        assert!(matches!(states[1], StatePlan::KvCache { .. }));
        assert_eq!(states[2], StatePlan::SharedKvCache { source_layer: 0 });
        assert_eq!(states[3], StatePlan::SharedKvCache { source_layer: 1 });
    }
}
