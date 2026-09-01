use super::*;
use crate::config::HfConfig;
use crate::tensor_contract::{
    QuantConstraint, TensorId, TensorMatch, TensorOwner, TensorRequirement, TensorTransform,
    VisionTensor,
};

pub static PACK: ModelPack = ModelPack {
    family: "gemma4_dense",
    aliases: &["gemma4", "gemma4_text"],
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
        matches!(config.arch, Arch::Gemma4)
            && !config.moe.as_ref().is_some_and(|moe| moe.expert_count > 0)
            && config
                .gemma4
                .as_ref()
                .is_some_and(|gemma| gemma.n_embd_per_layer == 0 && gemma.shared_kv_layers == 0)
    },
    plan_builder: canonical_plan,
    tensor_schema,
    tiny_plan: Some(tiny_plan),
};

fn tiny_plan() -> Result<ModelPlan, PlanCompileError> {
    canonical_plan(&ModelConfig::from_hf(&HfConfig::parse(
        r#"{"model_type":"gemma4","image_token_id":31,"vision_soft_tokens_per_image":1,
        "text_config":{"model_type":"gemma4_text","num_hidden_layers":2,
        "hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,
        "num_global_key_value_heads":1,"head_dim":4,"global_head_dim":4,
        "intermediate_size":16,"vocab_size":32,"max_position_embeddings":64,
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

#[allow(clippy::result_large_err)] // allow: the fat error type is the diagnostic contract here; boxing it would change the error surface
pub(super) fn tensor_schema(
    config: &ModelConfig,
    plan: &ModelPlan,
    dialect: CheckpointDialect,
    options: ContractOptions,
) -> Result<TensorContract, TensorContractError> {
    let mut contract = canonical_tensor_schema(config, plan, dialect, options)?;
    let Some(crate::model_plan::VisionPlan::Factored(vision)) = plan.vision.as_ref() else {
        return Ok(contract);
    };
    if dialect != CheckpointDialect::HfSafetensors {
        return Err(TensorContractError::UnsupportedPlanOperation {
            operation: "gemma4 vision non-safetensors schema",
        });
    }
    let mut add = |layer: Option<u32>, tensor: VisionTensor, name: String, shape: Vec<u64>| {
        contract.requirements.push(TensorRequirement {
            id: TensorId::Vision { layer, tensor },
            names: vec![name],
            match_mode: TensorMatch::OneOf,
            shape,
            owner: TensorOwner::Vision(layer),
            transform: TensorTransform::Identity,
            quant: QuantConstraint::FloatOnly,
            auxiliaries: None,
            required: true,
        });
    };
    let hidden = vision.hidden_size as u64;
    let patch_input =
        (vision.patch.channels * vision.patch.patch_size * vision.patch.patch_size) as u64;
    add(
        None,
        VisionTensor::PatchProjection,
        "model.vision_tower.patch_embedder.input_proj.weight".into(),
        vec![hidden, patch_input],
    );
    add(
        None,
        VisionTensor::PositionEmbedding,
        "model.vision_tower.patch_embedder.position_embedding_table".into(),
        vec![
            vision.patch.position_axes as u64,
            vision.patch.position_embedding_size as u64,
            hidden,
        ],
    );
    if vision.standardize {
        add(
            None,
            VisionTensor::StandardizeBias,
            "model.vision_tower.std_bias".into(),
            vec![hidden],
        );
        add(
            None,
            VisionTensor::StandardizeScale,
            "model.vision_tower.std_scale".into(),
            vec![hidden],
        );
    }
    add(
        None,
        VisionTensor::OutputProjection,
        "model.embed_vision.embedding_projection.weight".into(),
        vec![vision.projection_output_size as u64, hidden],
    );
    for layer in &vision.layers {
        let prefix = format!("model.vision_tower.encoder.layers.{}", layer.index);
        for (tensor, suffix) in [
            (VisionTensor::InputNorm, "input_layernorm.weight"),
            (
                VisionTensor::PostAttentionNorm,
                "post_attention_layernorm.weight",
            ),
            (VisionTensor::PreMlpNorm, "pre_feedforward_layernorm.weight"),
            (
                VisionTensor::PostMlpNorm,
                "post_feedforward_layernorm.weight",
            ),
        ] {
            add(
                Some(layer.index),
                tensor,
                format!("{prefix}.{suffix}"),
                vec![hidden],
            );
        }
        let query_width = (layer.attention.query_heads * layer.attention.head_dim) as u64;
        let kv_width = (layer.attention.kv_heads * layer.attention.head_dim) as u64;
        for (tensor, suffix, shape) in [
            (
                VisionTensor::Query,
                "self_attn.q_proj.linear.weight",
                vec![query_width, hidden],
            ),
            (
                VisionTensor::Key,
                "self_attn.k_proj.linear.weight",
                vec![kv_width, hidden],
            ),
            (
                VisionTensor::Value,
                "self_attn.v_proj.linear.weight",
                vec![kv_width, hidden],
            ),
            (
                VisionTensor::AttentionOutput,
                "self_attn.o_proj.linear.weight",
                vec![hidden, query_width],
            ),
            (
                VisionTensor::QueryNorm,
                "self_attn.q_norm.weight",
                vec![layer.attention.head_dim as u64],
            ),
            (
                VisionTensor::KeyNorm,
                "self_attn.k_norm.weight",
                vec![layer.attention.head_dim as u64],
            ),
            (
                VisionTensor::MlpGate,
                "mlp.gate_proj.linear.weight",
                vec![layer.mlp.intermediate_size as u64, hidden],
            ),
            (
                VisionTensor::MlpUp,
                "mlp.up_proj.linear.weight",
                vec![layer.mlp.intermediate_size as u64, hidden],
            ),
            (
                VisionTensor::MlpDown,
                "mlp.down_proj.linear.weight",
                vec![hidden, layer.mlp.intermediate_size as u64],
            ),
        ] {
            add(
                Some(layer.index),
                tensor,
                format!("{prefix}.{suffix}"),
                shape,
            );
        }
    }
    Ok(contract)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HfConfig;
    use crate::model_plan::OperationKind;

    fn config() -> ModelConfig {
        ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"gemma4","text_config":{"model_type":"gemma4_text",
            "num_hidden_layers":4,"hidden_size":8,"num_attention_heads":8,
            "num_key_value_heads":4,"num_global_key_value_heads":4,"head_dim":32,
            "global_head_dim":64,"intermediate_size":64,"vocab_size":262144,
            "max_position_embeddings":262144,"rms_norm_eps":0.000001,
            "sliding_window":1024,"final_logit_softcapping":30,
            "layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"],
            "rope_parameters":{"full_attention":{"rope_theta":1000000,
            "partial_rotary_factor":0.25},"sliding_attention":{"rope_theta":10000}}},
            "vision_config":{"hidden_size":8,"intermediate_size":64,
            "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":4,
            "head_dim":32,"max_position_embeddings":131072,"patch_size":16,
            "position_embedding_size":10240,"pooling_kernel_size":3,
            "rms_norm_eps":0.000001,"standardize":true,"use_clipped_linears":false,
            "hidden_activation":"gelu_pytorch_tanh","rope_parameters":{"rope_theta":100}}}"#,
        ))
    }

    #[test]
    fn vision_config_compiles_to_typed_plan_and_exact_auxiliary_schema() {
        let config = config();
        let plan = PACK.compile_plan(&config).unwrap();
        let Some(crate::model_plan::VisionPlan::Factored(vision)) = plan.vision.as_ref() else {
            panic!("gemma4 vision plan must be the factored program");
        };
        assert_eq!(vision.layers.len(), 2);
        assert_eq!(vision.patch.position_axes, 2);
        assert_eq!(vision.patch.position_embedding_size, 10_240);
        assert_eq!(vision.projection_output_size, 8);
        let operations = plan.operations();
        for operation in [
            OperationKind::VisionPatchEmbedding,
            OperationKind::VisionBidirectionalAttention,
            OperationKind::VisionMlp,
            OperationKind::VisionStandardize,
            OperationKind::VisionProjection,
        ] {
            assert!(operations.contains(&operation));
        }

        let contract = PACK
            .compile_tensor_contract(
                &config,
                &plan,
                CheckpointDialect::HfSafetensors,
                ContractOptions {
                    output_head: crate::tensor_contract::OutputHead::TiedToEmbedding,
                },
            )
            .unwrap();
        let vision_requirements: Vec<_> = contract
            .requirements
            .iter()
            .filter(|requirement| matches!(requirement.id, TensorId::Vision { .. }))
            .collect();
        assert_eq!(vision_requirements.len(), 31);
        assert!(vision_requirements.iter().any(|requirement| {
            requirement.names == ["model.vision_tower.patch_embedder.position_embedding_table"]
                && requirement.shape == [2, 10_240, 8]
        }));
        assert!(vision_requirements.iter().any(|requirement| {
            requirement.names
                == ["model.vision_tower.encoder.layers.1.self_attn.q_proj.linear.weight"]
                && requirement.shape == [128, 8]
        }));
    }
}
