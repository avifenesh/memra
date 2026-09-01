use super::*;
use crate::config::HfConfig;

pub static PACK: ModelPack = ModelPack {
    family: "gemma4_moe",
    aliases: &["gemma4_moe", "gemma4_a4b", "gemma4-26b-a4b"],
    config_layout: ConfigLayout::FlatOrTextConfig,
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
        matches!(config.arch, Arch::Gemma4)
            && config.moe.as_ref().is_some_and(|moe| moe.expert_count > 0)
            && config
                .gemma4
                .as_ref()
                .is_some_and(|gemma| gemma.n_embd_per_layer == 0 && gemma.shared_kv_layers == 0)
    },
    plan_builder: canonical_plan,
    tensor_schema: super::gemma4_dense::tensor_schema,
    tiny_plan: Some(tiny_plan),
};

fn tiny_plan() -> Result<ModelPlan, PlanCompileError> {
    canonical_plan(&ModelConfig::from_hf(&HfConfig::parse(
        r#"{"model_type":"gemma4","image_token_id":31,"vision_soft_tokens_per_image":1,
        "text_config":{"model_type":"gemma4_text","num_hidden_layers":2,
        "hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,
        "num_global_key_value_heads":1,"head_dim":4,"global_head_dim":4,
        "intermediate_size":16,"moe_intermediate_size":8,"num_experts":4,
        "top_k_experts":2,"vocab_size":32,"max_position_embeddings":64,
        "rms_norm_eps":0.000001,"sliding_window":8,
        "layer_types":["sliding_attention","full_attention"],
        "rope_parameters":{"full_attention":{"rope_theta":10000,
        "partial_rotary_factor":0.5},"sliding_attention":{"rope_theta":10000}}},
        "vision_config":{"hidden_size":8,"intermediate_size":16,"num_hidden_layers":2,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
        "max_position_embeddings":64,"patch_size":2,"position_embedding_size":16,
        "pooling_kernel_size":2,"rms_norm_eps":0.000001,"standardize":true,
        "use_clipped_linears":false,"hidden_activation":"gelu_pytorch_tanh",
        "rope_parameters":{"rope_theta":100}}}"#,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HfConfig;
    use crate::tensor_contract::{LayerTensor, OutputHead, TensorId};

    #[test]
    fn official_a4b_geometry_compiles_parallel_moe_tensor_contract() {
        let config = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"gemma4","text_config":{"model_type":"gemma4_text",
            "num_hidden_layers":30,"hidden_size":2816,"num_attention_heads":16,
            "num_key_value_heads":8,"num_global_key_value_heads":2,"head_dim":256,
            "global_head_dim":512,"intermediate_size":2112,"moe_intermediate_size":704,
            "num_experts":128,"top_k_experts":8,"vocab_size":262144,
            "max_position_embeddings":262144,"rms_norm_eps":0.000001,
            "sliding_window":1024,"final_logit_softcapping":30,
            "layer_types":["sliding_attention","sliding_attention","sliding_attention",
            "sliding_attention","sliding_attention","full_attention","sliding_attention",
            "sliding_attention","sliding_attention","sliding_attention","sliding_attention",
            "full_attention","sliding_attention","sliding_attention","sliding_attention",
            "sliding_attention","sliding_attention","full_attention","sliding_attention",
            "sliding_attention","sliding_attention","sliding_attention","sliding_attention",
            "full_attention","sliding_attention","sliding_attention","sliding_attention",
            "sliding_attention","sliding_attention","full_attention"],
            "rope_parameters":{"full_attention":{"rope_theta":1000000,
            "partial_rotary_factor":0.25},"sliding_attention":{"rope_theta":10000}}}}"#,
        ));
        assert_eq!(config.moe.as_ref().unwrap().expert_used_count, 8);
        assert_eq!(config.moe.as_ref().unwrap().expert_shared_ff_length, 2112);
        assert_eq!(for_config(&config).unwrap().family, "gemma4_moe");
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
            .unwrap();
        let requirement = |tensor| {
            contract
                .requirements
                .iter()
                .find(|requirement| requirement.id == TensorId::Layer { index: 0, tensor })
                .unwrap()
        };
        assert_eq!(
            requirement(LayerTensor::MoeExpertGateUpBank).shape,
            [128, 1408, 2816]
        );
        assert_eq!(
            requirement(LayerTensor::MoeExpertDownBank).shape,
            [128, 2816, 704]
        );
        assert_eq!(
            requirement(LayerTensor::MoeRouterScale).names,
            ["model.layers.0.router.scale"]
        );
        assert_eq!(
            requirement(LayerTensor::PostSharedMlpNorm).names,
            ["model.layers.0.post_feedforward_layernorm_1.weight"]
        );
    }
}
