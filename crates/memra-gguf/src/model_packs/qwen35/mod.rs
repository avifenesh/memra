use super::*;
use crate::config::HfConfig;

pub mod activation;

pub static PACK: ModelPack = ModelPack {
    family: "qwen35",
    output_head: OutputHeadContract::TiedHeadAllowed,
    // Refuse is a qualification state: the 9B NVFP4 MTP GGUF loads with zero unconsumed bound
    // tensors once the by-design-unread `.input_scale` planes are excluded
    // (research/loader-census-20260922/raw/serve-9b-boot-refuse.log).
    tensor_consumption: TensorConsumption::Refuse,
    aliases: &[
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_next",
        "qwen35",
        "qwen3next",
    ],
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
        matches!(config.arch, Arch::Qwen35)
            && !config.moe.as_ref().is_some_and(|moe| moe.expert_count > 0)
    },
    plan_builder: canonical_plan,
    tensor_schema: qwen35_tensor_schema,
    tiny_plan: Some(tiny_plan),
};

/// The canonical trunk contract plus the qwen3_5 ViT when the trunk shards bundle it.
///
/// The tower is side-loaded at serving (`MEMRA_VISION_DIR`, `memra_engine::vision`), never part
/// of the plan. Most quantized trunks strip it, but some checkpoints ship the BF16
/// `model.visual.*` tensors in the trunk shards: NVIDIA's NVFP4 export of Qwen3.8-27B (333
/// tensors, `vision_config.model_type` `qwen3_5_vision`) and the official Qwen3.5 checkpoints
/// (297 on the 4B, which write the trunk's `qwen3_5` there, memra #1166). They are censused
/// here as vision-owned with their exact geometry from `vision_config`, so the trunk loader's
/// consumption audit leaves them to the vision owner instead of the bind refusing them as
/// extras. They are optional because a stripped trunk with the same config carries none; a
/// partial tower still refuses when the vision side-load reads it.
#[allow(clippy::result_large_err)] // allow: the fat error type is the diagnostic contract here; boxing it would change the error surface
fn qwen35_tensor_schema(
    config: &ModelConfig,
    plan: &ModelPlan,
    dialect: CheckpointDialect,
    options: ContractOptions,
) -> Result<TensorContract, TensorContractError> {
    use crate::tensor_contract::{
        QuantConstraint, TensorId, TensorMatch, TensorOwner, TensorRequirement, TensorTransform,
    };
    let mut contract = canonical_tensor_schema(config, plan, dialect, options)?;
    let (CheckpointDialect::HfSafetensors, Some(vt)) = (dialect, config.qwen35_vision.as_ref())
    else {
        return Ok(contract);
    };
    let vh = u64::from(vt.hidden_size);
    let vff = u64::from(vt.intermediate_size);
    let mi = u64::from(vt.merger_in());
    let out = u64::from(vt.out_hidden_size);
    let mut push = |block: Option<u32>, key: String, shape: Vec<u64>| {
        contract.requirements.push(TensorRequirement {
            id: TensorId::Family {
                family: "qwen35_vision",
                key: key.clone(),
            },
            names: vec![format!("model.visual.{key}")],
            match_mode: TensorMatch::OneOf,
            shape,
            owner: TensorOwner::Vision(block),
            transform: TensorTransform::Identity,
            quant: QuantConstraint::FloatOnly,
            auxiliaries: None,
            required: false,
        });
    };
    for block in 0..vt.depth {
        let b = Some(block);
        let n = |s: &str| format!("blocks.{block}.{s}");
        push(b, n("attn.qkv.weight"), vec![3 * vh, vh]);
        push(b, n("attn.qkv.bias"), vec![3 * vh]);
        push(b, n("attn.proj.weight"), vec![vh, vh]);
        push(b, n("attn.proj.bias"), vec![vh]);
        push(b, n("mlp.linear_fc1.weight"), vec![vff, vh]);
        push(b, n("mlp.linear_fc1.bias"), vec![vff]);
        push(b, n("mlp.linear_fc2.weight"), vec![vh, vff]);
        push(b, n("mlp.linear_fc2.bias"), vec![vh]);
        for norm in ["norm1", "norm2"] {
            push(b, n(&format!("{norm}.weight")), vec![vh]);
            push(b, n(&format!("{norm}.bias")), vec![vh]);
        }
    }
    push(None, "merger.linear_fc1.weight".into(), vec![mi, mi]);
    push(None, "merger.linear_fc1.bias".into(), vec![mi]);
    push(None, "merger.linear_fc2.weight".into(), vec![out, mi]);
    push(None, "merger.linear_fc2.bias".into(), vec![out]);
    push(None, "merger.norm.weight".into(), vec![vh]);
    push(None, "merger.norm.bias".into(), vec![vh]);
    let p = u64::from(vt.patch_size);
    push(
        None,
        "patch_embed.proj.weight".into(),
        vec![
            vh,
            u64::from(vt.in_channels),
            u64::from(vt.temporal_patch_size),
            p,
            p,
        ],
    );
    push(None, "patch_embed.proj.bias".into(), vec![vh]);
    push(
        None,
        "pos_embed.weight".into(),
        vec![u64::from(vt.num_position_embeddings), vh],
    );
    Ok(contract)
}

fn tiny_plan() -> Result<ModelPlan, PlanCompileError> {
    canonical_plan(&ModelConfig::from_hf(&HfConfig::parse(
        r#"{"model_type":"qwen3_5","num_hidden_layers":4,
        "num_nextn_predict_layers":1,"hidden_size":8,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
        "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32,
        "rms_norm_eps":0.000001,"full_attention_interval":2,
        "linear_conv_kernel_dim":3,"linear_key_head_dim":4,
        "linear_value_head_dim":4,"linear_num_key_heads":1,
        "linear_num_value_heads":2}"#,
    )))
}

#[cfg(test)]
mod vision_census_tests {
    use super::*;
    use crate::tensor_contract::{FloatType, StorageLayout, TensorCensusEntry, TensorOwner};

    const VISION: &str = r#""vision_config":{"model_type":"qwen3_5_vision","depth":2,
        "hidden_size":8,"intermediate_size":12,"num_heads":2,"num_position_embeddings":16,
        "out_hidden_size":8,"patch_size":2,"spatial_merge_size":2,"temporal_patch_size":2,
        "in_channels":3}"#;

    fn config(with_vision: bool) -> ModelConfig {
        config_with(with_vision.then_some(VISION))
    }

    /// A tiny qwen3_5 trunk config with the given `vision_config` member, or none.
    fn config_with(vision: Option<&str>) -> ModelConfig {
        let vision = vision.map(|v| format!(",{v}")).unwrap_or_default();
        ModelConfig::from_hf(&HfConfig::parse(&format!(
            r#"{{"model_type":"qwen3_5","num_hidden_layers":4,
            "num_nextn_predict_layers":1,"hidden_size":8,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "intermediate_size":16,"vocab_size":32,"max_position_embeddings":32,
            "rms_norm_eps":0.000001,"full_attention_interval":2,
            "linear_conv_kernel_dim":3,"linear_key_head_dim":4,
            "linear_value_head_dim":4,"linear_num_key_heads":1,
            "linear_num_value_heads":2{vision}}}"#
        )))
    }

    fn census(contract: &TensorContract, tower: bool) -> Vec<TensorCensusEntry> {
        contract
            .requirements
            .iter()
            .filter(|r| r.required || (tower && matches!(r.owner, TensorOwner::Vision(_))))
            .flat_map(|r| {
                let names = match r.match_mode {
                    crate::tensor_contract::TensorMatch::All => r.names.clone(),
                    crate::tensor_contract::TensorMatch::OneOf => vec![r.names[0].clone()],
                };
                names.into_iter().map(|name| TensorCensusEntry {
                    name,
                    shape: r.shape.clone(),
                    storage: StorageLayout::Float(FloatType::Bf16),
                    physical_bytes: 0,
                })
            })
            .collect()
    }

    #[test]
    fn bundled_qwen35_tower_is_censused_as_vision_owned() {
        let cfg = config(true);
        let vt = cfg
            .qwen35_vision
            .clone()
            .expect("qwen3_5_vision geometry parsed");
        assert_eq!((vt.depth, vt.hidden_size, vt.merger_in()), (2, 8, 32));
        let plan = canonical_plan(&cfg).unwrap();
        let contract = qwen35_tensor_schema(
            &cfg,
            &plan,
            CheckpointDialect::HfSafetensors,
            ContractOptions::default(),
        )
        .unwrap();
        let vision: Vec<_> = contract
            .requirements
            .iter()
            .filter(|r| matches!(r.owner, TensorOwner::Vision(_)))
            .collect();
        // 12 per block, 6 merger, patch weight + bias, pos table: 333 at depth 27.
        assert_eq!(vision.len(), 12 * 2 + 9);
        assert!(
            vision
                .iter()
                .all(|r| !r.required && r.names[0].starts_with("model.visual."))
        );
        let patch = vision
            .iter()
            .find(|r| r.names[0] == "model.visual.patch_embed.proj.weight")
            .unwrap();
        assert_eq!(patch.shape, vec![8, 3, 2, 2, 2]);

        // The NVIDIA layout (tower in the trunk shards) and a stripped trunk both bind.
        contract.bind(&census(&contract, true)).unwrap();
        contract.bind(&census(&contract, false)).unwrap();
        // A tower tensor with the wrong geometry still refuses.
        let mut bad = census(&contract, true);
        let row = bad
            .iter_mut()
            .find(|e| e.name == "model.visual.pos_embed.weight")
            .unwrap();
        row.shape = vec![15, 8];
        assert!(contract.bind(&bad).is_err());

        // GGUF trunks never census a tower, and a config without one adds nothing.
        let gguf = qwen35_tensor_schema(
            &cfg,
            &plan,
            CheckpointDialect::Gguf,
            ContractOptions::default(),
        )
        .unwrap();
        assert!(
            gguf.requirements
                .iter()
                .all(|r| !matches!(r.owner, TensorOwner::Vision(_)))
        );
        let text = config(false);
        assert!(text.qwen35_vision.is_none());
        let text_contract = qwen35_tensor_schema(
            &text,
            &canonical_plan(&text).unwrap(),
            CheckpointDialect::HfSafetensors,
            ContractOptions::default(),
        )
        .unwrap();
        assert_eq!(
            text_contract.requirements.len(),
            contract.requirements.len() - vision.len()
        );
    }

    /// memra #1166: the official Qwen3.5 checkpoints (4B, 4B-Base, 9B) write the trunk's
    /// model_type, `qwen3_5`, inside `vision_config`, where the Qwen3.8-27B exports write
    /// `qwen3_5_vision`. The geometry keys are the same. A checkpoint that bundles the tower
    /// under that spelling must bind exactly as one under the other: before the fix the
    /// contract declared no tower and this bind refused with "extra checkpoint tensors", the
    /// refusal `run-gen` and the server printed for the official Qwen3.5-4B directory.
    #[test]
    fn the_qwen3_5_vision_model_type_binds_a_bundled_tower() {
        let schema = |cfg: &ModelConfig| {
            qwen35_tensor_schema(
                cfg,
                &canonical_plan(cfg).unwrap(),
                CheckpointDialect::HfSafetensors,
                ContractOptions::default(),
            )
            .unwrap()
        };
        let reference = config(true);
        let official = config_with(Some(&VISION.replace("qwen3_5_vision", "qwen3_5")));
        // The checkpoint: the trunk plus every tower tensor, as the reference spelling censuses it.
        let checkpoint = census(&schema(&reference), true);
        let tower = checkpoint
            .iter()
            .filter(|e| e.name.starts_with("model.visual."))
            .count();
        assert_eq!(tower, 12 * 2 + 9);

        let contract = schema(&official);
        if let Err(err) = contract.bind(&checkpoint) {
            panic!("a tower bundled under vision model_type qwen3_5 must bind: {err}");
        }
        assert_eq!(official.qwen35_vision, reference.qwen35_vision);
        // The stripped trunk of the same config still binds, and a wrong tower shape still refuses.
        contract.bind(&census(&contract, false)).unwrap();
        let mut bad = census(&schema(&reference), true);
        bad.iter_mut()
            .find(|e| e.name == "model.visual.pos_embed.weight")
            .unwrap()
            .shape = vec![15, 8];
        assert!(contract.bind(&bad).is_err());
    }

    /// memra #1166: the official Qwen3.5-4B tower geometry, sent through the schema. The
    /// checkpoint's shards carry 297 `model.visual.*` tensors (12 a block at depth 24, 6 merger,
    /// the patch weight and bias, the position table) with these shapes; the contract must
    /// declare exactly that, or the bind still refuses them.
    #[test]
    fn the_qwen35_4b_tower_geometry_declares_its_297_tensors() {
        // The checkpoint's vision_config, verbatim, on the tiny trunk.
        let cfg = config_with(Some(
            r#""vision_config":{"deepstack_visual_indexes":[],"depth":24,
            "hidden_act":"gelu_pytorch_tanh","hidden_size":1024,"in_channels":3,
            "initializer_range":0.02,"intermediate_size":4096,"model_type":"qwen3_5",
            "num_heads":16,"num_position_embeddings":2304,"out_hidden_size":2560,
            "patch_size":16,"spatial_merge_size":2,"temporal_patch_size":2}"#,
        ));
        let contract = qwen35_tensor_schema(
            &cfg,
            &canonical_plan(&cfg).unwrap(),
            CheckpointDialect::HfSafetensors,
            ContractOptions::default(),
        )
        .unwrap();
        let vision: Vec<_> = contract
            .requirements
            .iter()
            .filter(|r| matches!(r.owner, TensorOwner::Vision(_)))
            .collect();
        assert_eq!(vision.len(), 297);
        let shape = |name: &str| {
            vision
                .iter()
                .find(|r| r.names[0] == format!("model.visual.{name}"))
                .unwrap_or_else(|| panic!("{name} is not declared"))
                .shape
                .clone()
        };
        assert_eq!(shape("patch_embed.proj.weight"), vec![1024, 3, 2, 16, 16]);
        assert_eq!(shape("pos_embed.weight"), vec![2304, 1024]);
        assert_eq!(shape("merger.linear_fc1.weight"), vec![4096, 4096]);
        assert_eq!(shape("merger.linear_fc2.weight"), vec![2560, 4096]);
        assert_eq!(shape("blocks.23.attn.qkv.weight"), vec![3072, 1024]);
        assert_eq!(shape("blocks.23.mlp.linear_fc1.weight"), vec![4096, 1024]);
    }
}
