//! Qwen3.8-Flash-Next (`qwen4_exp`) loader-lane pack — census-gated safetensors ingest
//! (research/qwen4exp-bringup-20260829; geometry ARCH.md, semantics SEMANTICS.md).
//!
//! Checkpoint namespaces (census, 1658 tensors, all BF16 except three I64 index tables):
//!   lm_head.weight / model.language_model.*   trunk (48 layers, `layers.N.` prefix)
//!   model.visual.*                            ViT tower (side-loaded at serving)
//!   mtp.* / mtp.layers.0.*                    1-block NextN draft (QSA + MoE, no PLE)
//! There is NO `model.language_model.norm` — the global `hyper_connection_mixer` is the exit
//! downmix (SEMANTICS.md §Layer stack), so this schema deliberately emits no OutputNorm row.

use std::collections::BTreeMap;

use super::*;
use crate::config::{HfConfig, LayerKind, Qwen4ExpConfig};
use crate::dsv4::TensorSpec;
use crate::tensor_contract::{
    ExpertTensor, FloatType, LayerTensor, MtpTensor, QuantConstraint, TensorId, TensorMatch,
    TensorOwner, TensorRequirement, TensorTransform,
};

pub static PACK: ModelPack = ModelPack {
    family: "qwen4_exp",
    aliases: &["qwen4_exp", "qwen4exp", "qwen4_exp_text"],
    config_layout: ConfigLayout::FlatOrTextConfig,
    tokenizer_sources: &[TokenizerSource::TokenizerJson],
    template: TemplateContract::ArtifactRequired,
    // Loader lane: inspection/census only, native plan execution unsupported.
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
    matches_config: |config| matches!(config.arch, Arch::Qwen4Exp) && config.qwen4exp.is_some(),
    plan_builder: canonical_plan,
    tensor_schema,
    tiny_plan: Some(tiny_plan),
};

/// 4-layer miniature carrying every family key scaled down. It exists to COMPILE — it
/// protects the from_hf parse and the plan arms, not any numeric behavior.
fn tiny_plan() -> Result<ModelPlan, PlanCompileError> {
    canonical_plan(&ModelConfig::from_hf(&HfConfig::parse(
        r#"{"model_type":"qwen4_exp","text_config":{
        "model_type":"qwen4_exp_text","eos_token_id":63,
        "num_hidden_layers":4,"hidden_size":16,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,
        "intermediate_size":32,"vocab_size":64,"max_position_embeddings":64,
        "rms_norm_eps":0.000001,"full_attention_interval":4,
        "rope_parameters":{"rope_theta":10000,"partial_rotary_factor":0.25,
        "mrope_section":[1,1,2],"mrope_interleaved":true},
        "linear_conv_kernel_dim":4,"linear_key_head_dim":4,"linear_value_head_dim":4,
        "linear_num_key_heads":1,"linear_num_value_heads":2,
        "num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":8,
        "shared_expert_intermediate_size":8,
        "indexer_n_heads":1,"indexer_kv_heads":1,"indexer_head_dim":4,
        "indexer_compress_ratio":4,"indexer_budget":8,
        "hc_count":2,"hc_lowrank":4,
        "ngram_size":3,"heads_per_ngram":2,"ngram_vocab_size_base":64,
        "make_ngram_vocab_size_divisible_by":128,"split_ngram_parts":2,
        "ple_layer_ids":[2],"ple_embed_dim":16,"ple_conv_kernel_size":4,
        "output_gate_type":"sigmoid",
        "mtp":{"num_hidden_layers":1,"rope_theta":10000},"mtp_num_hidden_layers":1}}"#,
    )))
}

/// Which routed-expert layout the artifact carries. Two real dialects exist
/// (2026-08-29): the BF16 export ships fused 3D module tensors
/// (`mlp.experts.gate_up_proj` / `down_proj`); the modelopt NVFP4 mint
/// (Avifenesh/Qwen3.8-Flash-Next-NVFP4) UN-FUSES to per-expert 2D projections with
/// gate/up SPLIT and the modelopt sibling set
/// (`experts.E.{gate,up,down}_proj.{weight,weight_scale,weight_scale_2,input_scale}`).
/// Everything outside the trunk routed experts is BF16 in both (mtp.* rides a BF16
/// graft shard in the mint).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertDialect {
    FusedBanks,
    PerExpertModelopt,
}

/// The tensor contract for an explicit expert dialect. `ModelPack::compile_tensor_contract`
/// keeps the FusedBanks (BF16-census) default; loaders that probed the artifact call this.
pub fn tensor_contract_for(
    config: &ModelConfig,
    plan: &ModelPlan,
    experts: ExpertDialect,
) -> Result<TensorContract, TensorContractError> {
    schema_impl(config, plan, CheckpointDialect::HfSafetensors, experts)
}

fn tensor_schema(
    config: &ModelConfig,
    plan: &ModelPlan,
    dialect: CheckpointDialect,
    _options: ContractOptions,
) -> Result<TensorContract, TensorContractError> {
    schema_impl(config, plan, dialect, ExpertDialect::FusedBanks)
}

/// Trunk layer / expert / projection of a per-expert row (`model.language_model.layers.N.
/// mlp.experts.E.{gate,up,down}_proj.weight`). MTP names never match (fused BF16 graft).
fn per_expert_weight(name: &str) -> Option<(u32, u32, ExpertTensor)> {
    let rest = name.strip_prefix("model.language_model.layers.")?;
    let (layer, rest) = rest.split_once('.')?;
    let rest = rest.strip_prefix("mlp.experts.")?;
    let (expert, rest) = rest.split_once('.')?;
    let tensor = match rest {
        "gate_proj.weight" => ExpertTensor::Gate,
        "up_proj.weight" => ExpertTensor::Up,
        "down_proj.weight" => ExpertTensor::Down,
        _ => return None,
    };
    Some((layer.parse().ok()?, expert.parse().ok()?, tensor))
}

/// The dsv4 `quant_primary` twin: a modelopt scale sibling folds into its `.weight`
/// primary's auxiliaries instead of becoming its own requirement.
fn sibling_primary(name: &str, expected: &BTreeMap<String, TensorSpec>) -> Option<String> {
    for suffix in [".weight_scale", ".weight_scale_2", ".input_scale"] {
        if let Some(stem) = name.strip_suffix(suffix) {
            let primary = format!("{stem}.weight");
            if expected.contains_key(&primary) {
                return Some(primary);
            }
        }
    }
    None
}

fn schema_impl(
    config: &ModelConfig,
    plan: &ModelPlan,
    dialect: CheckpointDialect,
    experts: ExpertDialect,
) -> Result<TensorContract, TensorContractError> {
    if dialect != CheckpointDialect::HfSafetensors {
        return Err(TensorContractError::UnsupportedPlanOperation {
            operation: "qwen4_exp non-safetensors schema",
        });
    }
    let n_trunk = plan.layers.len() as u32;
    let expected = expected_census_for(config, experts);
    let mut requirements = Vec::new();
    // The 128 ngram shards are ONE semantic bank (concat on dim 0, shard order = index
    // order — SEMANTICS.md §Loading notes); collect them into a single All-mode group.
    let mut shard_groups: BTreeMap<String, Vec<(u32, String, TensorSpec)>> = BTreeMap::new();
    for (name, spec) in &expected {
        if let Some((stem, rest)) = name.split_once(".ngram_embedding.shard_") {
            let index: u32 = rest
                .strip_suffix(".weight")
                .and_then(|s| s.parse().ok())
                .ok_or(TensorContractError::UnsupportedPlanOperation {
                    operation: "qwen4_exp ngram shard name",
                })?;
            shard_groups.entry(stem.to_string()).or_default().push((
                index,
                name.clone(),
                spec.clone(),
            ));
            continue;
        }
        // modelopt scale siblings ride their primary's auxiliaries (dsv4 precedent).
        if sibling_primary(name, &expected).is_some() {
            continue;
        }
        // The mint's UNSHARDED n-gram table: same semantic id as the fused dialect's
        // shard bank (the `.weight`-suffixed name strips to the bank key), so loaders
        // key the host table identically across dialects.
        if let Some(stem) = name.strip_suffix(".weight") {
            if stem.ends_with(".ple_embedding.ngram_embedding") {
                requirements.push(TensorRequirement {
                    id: TensorId::Family {
                        family: "qwen4_exp",
                        key: semantic_key(stem),
                    },
                    names: vec![name.clone()],
                    match_mode: TensorMatch::OneOf,
                    shape: spec.shape.clone(),
                    owner: tensor_owner(name),
                    transform: TensorTransform::Identity,
                    quant: QuantConstraint::ExactFloat(FloatType::Bf16),
                    auxiliaries: None,
                    required: true,
                });
                continue;
            }
        }
        // Per-expert projections (PerExpertModelopt): TensorId::Expert rows, the dsv4
        // per-expert-requirement pattern (per-row auxiliaries cannot ride an All-group).
        if let Some((layer, expert, tensor)) = per_expert_weight(name) {
            let stem = name
                .strip_suffix(".weight")
                .expect("per-expert weight name");
            let auxiliaries: Vec<String> = [
                format!("{stem}.weight_scale"),
                format!("{stem}.weight_scale_2"),
                format!("{stem}.input_scale"),
            ]
            .into_iter()
            .filter(|candidate| expected.contains_key(candidate))
            .collect();
            let mut shape = spec.shape.clone();
            let quant = match spec.dtype {
                // U8 packs two e2m1 codes per byte; the requirement carries the LOGICAL
                // [out, in] (dsv4 convention).
                "U8" => {
                    *shape.last_mut().expect("expert weight rank") *= 2;
                    QuantConstraint::Nvfp4
                }
                "BF16" => QuantConstraint::ExactFloat(FloatType::Bf16),
                _ => {
                    return Err(TensorContractError::UnsupportedPlanOperation {
                        operation: "unknown qwen4_exp expert dtype",
                    });
                }
            };
            requirements.push(TensorRequirement {
                id: TensorId::Expert {
                    layer,
                    expert,
                    tensor,
                },
                names: vec![name.clone()],
                match_mode: TensorMatch::OneOf,
                shape,
                owner: TensorOwner::Layer(layer),
                transform: TensorTransform::Identity,
                quant,
                auxiliaries: Some(auxiliaries),
                required: true,
            });
            continue;
        }
        let quant = match spec.dtype {
            "BF16" => QuantConstraint::ExactFloat(FloatType::Bf16),
            "I64" => QuantConstraint::I64,
            _ => {
                return Err(TensorContractError::UnsupportedPlanOperation {
                    operation: "unknown qwen4_exp dtype",
                });
            }
        };
        requirements.push(TensorRequirement {
            id: semantic_tensor_id(name, n_trunk),
            names: vec![name.clone()],
            match_mode: TensorMatch::OneOf,
            shape: spec.shape.clone(),
            owner: tensor_owner(name),
            transform: transform_for(name),
            quant,
            auxiliaries: None,
            required: true,
        });
    }
    for (stem, mut shards) in shard_groups {
        shards.sort_by_key(|(index, _, _)| *index);
        let shape = shards[0].2.shape.clone();
        let bank = format!("{stem}.ngram_embedding");
        requirements.push(TensorRequirement {
            id: TensorId::Family {
                family: "qwen4_exp",
                key: semantic_key(&bank),
            },
            names: shards.into_iter().map(|(_, name, _)| name).collect(),
            match_mode: TensorMatch::All,
            shape,
            owner: tensor_owner(&bank),
            // Concatenate on dim 0 in shard order at load; HF skips device placement for
            // this table entirely (host-resident / TP-colwise is the loader's call).
            transform: TensorTransform::Identity,
            quant: QuantConstraint::ExactFloat(FloatType::Bf16),
            auxiliaries: None,
            required: true,
        });
    }
    Ok(TensorContract {
        dialect,
        requirements,
    })
}

/// Complete expected tensor map for the BF16 artifact (FusedBanks dialect).
pub(crate) fn expected_census(config: &ModelConfig) -> BTreeMap<String, TensorSpec> {
    expected_census_for(config, ExpertDialect::FusedBanks)
}

/// Complete expected tensor map, derived from config math (the dsv4 pattern). Every shape
/// is safetensors row-major `[out, in]`; receipts are the ARCH.md geometry table and the
/// banked census (raw/q48fn-census.tsv.gz) for FusedBanks; the PerExpertModelopt map keys
/// against the NVFP4 mint census (raw/nvfp4-census-names.tsv fixture when banked). Only
/// the TRUNK routed experts differ between dialects — the MTP block keeps its fused BF16
/// module tensors in both (the mint grafts mtp.* as a BF16 shard).
pub fn expected_census_for(
    config: &ModelConfig,
    experts: ExpertDialect,
) -> BTreeMap<String, TensorSpec> {
    let q = config
        .qwen4exp
        .as_ref()
        .expect("qwen4_exp census requires a qwen4_exp ModelConfig");
    let moe = config
        .moe
        .as_ref()
        .expect("qwen4_exp is MoE; moe block missing");
    let ssm = config
        .ssm
        .as_ref()
        .expect("qwen4_exp is a GDN hybrid; ssm block missing");

    let h = config.n_embd as u64; // 2560
    let v = config.n_vocab as u64; // 248320
    let hc = q.hc_count as u64; // 4
    let wide = hc * h; // 10240
    let rank = q.hc_lowrank as u64; // 320
    // QSA: fused [q|gate] doubles q_proj out; o_proj reads n_head * head_dim_v.
    let q_out = 2 * config.n_head as u64 * config.head_dim_k as u64; // 12288
    let kv_out = config.n_head_kv as u64 * config.head_dim_k as u64; // 512
    let o_in = config.n_head as u64 * config.head_dim_v as u64; // 6144
    let qk_norm = config.head_dim_k as u64; // 256
    let idx_out = (q.indexer_n_heads + q.indexer_kv_heads) as u64 * q.indexer_head_dim as u64; // 640
    let idx_norm = q.indexer_head_dim as u64; // 128
    // GDN: fused qkv = 2 * QK heads * key dim + V heads * value dim.
    let gdn_v_heads = ssm.time_step_rank as u64; // 48
    let gdn_v_dim = (ssm.inner_size / ssm.time_step_rank) as u64; // 128
    let gdn_v = gdn_v_heads * gdn_v_dim; // 6144
    let gdn_qkv = 2 * ssm.group_count as u64 * ssm.state_size as u64 + gdn_v; // 10240
    let gdn_conv = ssm.conv_kernel as u64; // 4
    // MoE: fused 3D experts (Qwen4ExpTextExperts module, NOT nn.Linear — no .weight suffix).
    let ne = moe.expert_count as u64; // 512
    let mff = moe.expert_ff_length as u64; // 640
    let sff = moe.expert_shared_ff_length as u64; // 640

    let mut map: BTreeMap<String, TensorSpec> = BTreeMap::new();
    let t = |m: &mut BTreeMap<String, TensorSpec>,
             name: String,
             dtype: &'static str,
             shape: Vec<u64>| {
        let prev = m.insert(name.clone(), TensorSpec { dtype, shape });
        assert!(
            prev.is_none(),
            "qwen4_exp census: duplicate spec for {name}"
        );
    };
    // One gated-residual read/write gate set (attn_/mlp_hyper_connection.<p>).
    let hyper = |m: &mut BTreeMap<String, TensorSpec>, p: String| {
        t(
            m,
            format!("{p}block_inject_weight.weight"),
            "BF16",
            vec![hc, wide],
        );
        t(m, format!("{p}hc_norm.weight"), "BF16", vec![wide]);
        t(
            m,
            format!("{p}input_mix_weight_down.weight"),
            "BF16",
            vec![rank, wide],
        );
        t(
            m,
            format!("{p}input_mix_weight_up.weight"),
            "BF16",
            vec![wide, rank],
        );
    };
    // The exit mixer carries the read half only (no block_inject — census).
    let mixer = |m: &mut BTreeMap<String, TensorSpec>, p: String| {
        t(m, format!("{p}hc_norm.weight"), "BF16", vec![wide]);
        t(
            m,
            format!("{p}input_mix_weight_down.weight"),
            "BF16",
            vec![rank, wide],
        );
        t(
            m,
            format!("{p}input_mix_weight_up.weight"),
            "BF16",
            vec![wide, rank],
        );
    };
    let qsa = |m: &mut BTreeMap<String, TensorSpec>, p: &str| {
        t(
            m,
            format!("{p}self_attn.q_proj.weight"),
            "BF16",
            vec![q_out, h],
        );
        t(
            m,
            format!("{p}self_attn.k_proj.weight"),
            "BF16",
            vec![kv_out, h],
        );
        t(
            m,
            format!("{p}self_attn.v_proj.weight"),
            "BF16",
            vec![kv_out, h],
        );
        t(
            m,
            format!("{p}self_attn.o_proj.weight"),
            "BF16",
            vec![h, o_in],
        );
        t(
            m,
            format!("{p}self_attn.q_norm.weight"),
            "BF16",
            vec![qk_norm],
        );
        t(
            m,
            format!("{p}self_attn.k_norm.weight"),
            "BF16",
            vec![qk_norm],
        );
        t(
            m,
            format!("{p}self_attn.indexer.index_qk_proj.weight"),
            "BF16",
            vec![idx_out, h],
        );
        t(
            m,
            format!("{p}self_attn.indexer.q_layernorm.weight"),
            "BF16",
            vec![idx_norm],
        );
        t(
            m,
            format!("{p}self_attn.indexer.k_layernorm.weight"),
            "BF16",
            vec![idx_norm],
        );
    };
    let gdn = |m: &mut BTreeMap<String, TensorSpec>, p: &str| {
        t(
            m,
            format!("{p}linear_attn.in_proj_qkv.weight"),
            "BF16",
            vec![gdn_qkv, h],
        );
        t(
            m,
            format!("{p}linear_attn.in_proj_z.weight"),
            "BF16",
            vec![gdn_v, h],
        );
        t(
            m,
            format!("{p}linear_attn.in_proj_a.weight"),
            "BF16",
            vec![gdn_v_heads, h],
        );
        t(
            m,
            format!("{p}linear_attn.in_proj_b.weight"),
            "BF16",
            vec![gdn_v_heads, h],
        );
        t(
            m,
            format!("{p}linear_attn.conv1d.weight"),
            "BF16",
            vec![gdn_qkv, 1, gdn_conv],
        );
        t(
            m,
            format!("{p}linear_attn.A_log"),
            "BF16",
            vec![gdn_v_heads],
        );
        t(
            m,
            format!("{p}linear_attn.dt_bias"),
            "BF16",
            vec![gdn_v_heads],
        );
        t(
            m,
            format!("{p}linear_attn.norm.weight"),
            "BF16",
            vec![gdn_v_dim],
        );
        t(
            m,
            format!("{p}linear_attn.out_proj.weight"),
            "BF16",
            vec![h, gdn_v],
        );
    };
    // The modelopt NVFP4 mint un-fuses the trunk experts: per expert per projection, the
    // U8 packed weight plus the modelopt sibling set (fixture receipt:
    // raw/nvfp4-census-names.tsv — weight U8 [out, in/2], weight_scale F8_E4M3
    // [out, in/16], weight_scale_2 F32 [], input_scale F32 []). A projection whose in-dim
    // lacks per-16 groups falls back to a plain BF16 row — geometry, not policy (the
    // artifact's ff=640 and h=2560 both qualify; only tiny fixtures hit the fallback).
    let experts_per_expert = |m: &mut BTreeMap<String, TensorSpec>, p: &str| {
        for expert in 0..ne {
            for (proj, out, input) in [
                ("gate_proj", mff, h),
                ("up_proj", mff, h),
                ("down_proj", h, mff),
            ] {
                let stem = format!("{p}mlp.experts.{expert}.{proj}");
                if input % 16 == 0 {
                    t(m, format!("{stem}.weight"), "U8", vec![out, input / 2]);
                    t(
                        m,
                        format!("{stem}.weight_scale"),
                        "F8_E4M3",
                        vec![out, input / 16],
                    );
                    t(m, format!("{stem}.weight_scale_2"), "F32", vec![]);
                    t(m, format!("{stem}.input_scale"), "F32", vec![]);
                } else {
                    t(m, format!("{stem}.weight"), "BF16", vec![out, input]);
                }
            }
        }
    };
    let experts_fused = |m: &mut BTreeMap<String, TensorSpec>, p: &str| {
        t(
            m,
            format!("{p}mlp.experts.gate_up_proj"),
            "BF16",
            vec![ne, 2 * mff, h],
        );
        t(
            m,
            format!("{p}mlp.experts.down_proj"),
            "BF16",
            vec![ne, h, mff],
        );
    };
    // Router + shared expert (BF16 in both dialects).
    let moe_common = |m: &mut BTreeMap<String, TensorSpec>, p: &str| {
        t(m, format!("{p}mlp.gate.weight"), "BF16", vec![ne, h]);
        t(
            m,
            format!("{p}mlp.shared_expert.gate_proj.weight"),
            "BF16",
            vec![sff, h],
        );
        t(
            m,
            format!("{p}mlp.shared_expert.up_proj.weight"),
            "BF16",
            vec![sff, h],
        );
        t(
            m,
            format!("{p}mlp.shared_expert.down_proj.weight"),
            "BF16",
            vec![h, sff],
        );
        t(
            m,
            format!("{p}mlp.shared_expert_gate.weight"),
            "BF16",
            vec![1, h],
        );
    };
    let ple = |m: &mut BTreeMap<String, TensorSpec>, p: &str| {
        let ple_k = q.ple_conv_kernel_size as u64;
        let emb = q.ple_embed_dim as u64; // 2560
        let heads = q.ngram_heads() as u64; // 16
        let head_dim = q.ngram_head_embed_dim() as u64; // 160
        let shards = q.split_ngram_parts as u64; // 128
        t(
            m,
            format!("{p}ple.conv1d.weight"),
            "BF16",
            vec![wide, 1, ple_k],
        );
        t(m, format!("{p}ple.key_proj.weight"), "BF16", vec![wide, h]);
        t(m, format!("{p}ple.norm_conv.weight"), "BF16", vec![wide]);
        t(m, format!("{p}ple.norm_key.weight"), "BF16", vec![wide]);
        t(m, format!("{p}ple.norm_query.weight"), "BF16", vec![wide]);
        t(
            m,
            format!("{p}ple.value_proj.weight"),
            "BF16",
            vec![emb, emb],
        );
        t(
            m,
            format!("{p}ple.ple_embedding.layer_multipliers"),
            "I64",
            vec![q.ngram_size as u64],
        );
        t(
            m,
            format!("{p}ple.ple_embedding.ngram_heads_offsets"),
            "I64",
            vec![heads],
        );
        t(
            m,
            format!("{p}ple.ple_embedding.ngram_heads_vocab_sizes"),
            "I64",
            vec![heads],
        );
        let total = ngram_table_rows(q);
        // The BF16 export shards the table (config split_ngram_parts); the NVFP4 mint
        // re-exports it as ONE tensor with a `.weight` suffix (fixture receipt:
        // ngram_embedding.weight BF16 [320001536, 160]).
        match experts {
            ExpertDialect::FusedBanks => {
                assert!(
                    total % shards == 0,
                    "qwen4_exp ngram table rows {total} not divisible into {shards} shards"
                );
                let rows = total / shards;
                for shard in 0..shards {
                    t(
                        m,
                        format!("{p}ple.ple_embedding.ngram_embedding.shard_{shard}.weight"),
                        "BF16",
                        vec![rows, head_dim],
                    );
                }
            }
            ExpertDialect::PerExpertModelopt => {
                t(
                    m,
                    format!("{p}ple.ple_embedding.ngram_embedding.weight"),
                    "BF16",
                    vec![total, head_dim],
                );
            }
        }
    };

    // ---- globals ----
    t(&mut map, "lm_head.weight".into(), "BF16", vec![v, h]);
    t(
        &mut map,
        "model.language_model.embed_tokens.weight".into(),
        "BF16",
        vec![v, h],
    );
    // NO model.language_model.norm — the mixer is the exit downmix (SEMANTICS.md).
    mixer(
        &mut map,
        "model.language_model.hyper_connection_mixer.".into(),
    );

    // ---- trunk ----
    let n_trunk = config.n_layer - config.nextn_predict_layers;
    for il in 0..n_trunk {
        let p = format!("model.language_model.layers.{il}.");
        hyper(&mut map, format!("{p}attn_hyper_connection."));
        hyper(&mut map, format!("{p}mlp_hyper_connection."));
        if config.layer_kind(il) == LayerKind::FullAttention {
            qsa(&mut map, &p);
        } else {
            gdn(&mut map, &p);
        }
        moe_common(&mut map, &p);
        match experts {
            ExpertDialect::FusedBanks => experts_fused(&mut map, &p),
            ExpertDialect::PerExpertModelopt => experts_per_expert(&mut map, &p),
        }
        if q.has_ple(il) {
            ple(&mut map, &p);
        }
    }

    // ---- MTP (full-attention QSA block, own indexer, NO PLE; SEMANTICS.md §MTP) ----
    if config.nextn_predict_layers > 0 {
        t(
            &mut map,
            "mtp.fc_embedding.weight".into(),
            "BF16",
            vec![h, h],
        );
        t(&mut map, "mtp.fc_hidden.weight".into(), "BF16", vec![h, h]);
        t(
            &mut map,
            "mtp.pre_fc_norm_embedding.weight".into(),
            "BF16",
            vec![h],
        );
        // hidden-side norm covers the WIDE stream (census [10240]).
        t(
            &mut map,
            "mtp.pre_fc_norm_hidden.weight".into(),
            "BF16",
            vec![wide],
        );
        mixer(&mut map, "mtp.hyper_connection_mixer.".into());
        for k in 0..config.nextn_predict_layers {
            let p = format!("mtp.layers.{k}.");
            hyper(&mut map, format!("{p}attn_hyper_connection."));
            hyper(&mut map, format!("{p}mlp_hyper_connection."));
            qsa(&mut map, &p);
            // The mint grafts mtp.* as a BF16 shard — fused experts in BOTH dialects.
            moe_common(&mut map, &p);
            experts_fused(&mut map, &p);
        }
    }

    // ---- ViT tower (side-loaded at serving; censused so the artifact binds exactly) ----
    if let Some(vt) = q.vision.as_ref() {
        let vh = vt.hidden_size as u64; // 1152
        let vff = vt.intermediate_size as u64; // 4304
        for block in 0..vt.depth {
            let p = format!("model.visual.blocks.{block}.");
            t(
                &mut map,
                format!("{p}attn.qkv.weight"),
                "BF16",
                vec![3 * vh, vh],
            );
            t(&mut map, format!("{p}attn.qkv.bias"), "BF16", vec![3 * vh]);
            t(
                &mut map,
                format!("{p}attn.proj.weight"),
                "BF16",
                vec![vh, vh],
            );
            t(&mut map, format!("{p}attn.proj.bias"), "BF16", vec![vh]);
            t(
                &mut map,
                format!("{p}mlp.linear_fc1.weight"),
                "BF16",
                vec![vff, vh],
            );
            t(
                &mut map,
                format!("{p}mlp.linear_fc1.bias"),
                "BF16",
                vec![vff],
            );
            t(
                &mut map,
                format!("{p}mlp.linear_fc2.weight"),
                "BF16",
                vec![vh, vff],
            );
            t(
                &mut map,
                format!("{p}mlp.linear_fc2.bias"),
                "BF16",
                vec![vh],
            );
            // norm1/norm2 carry weight AND bias (LayerNorm, not RMSNorm) — census.
            for norm in ["norm1", "norm2"] {
                t(&mut map, format!("{p}{norm}.weight"), "BF16", vec![vh]);
                t(&mut map, format!("{p}{norm}.bias"), "BF16", vec![vh]);
            }
        }
        let mi = vt.merger_in() as u64; // 4608
        let out = vt.out_hidden_size as u64; // 2560
        t(
            &mut map,
            "model.visual.merger.linear_fc1.weight".into(),
            "BF16",
            vec![mi, mi],
        );
        t(
            &mut map,
            "model.visual.merger.linear_fc1.bias".into(),
            "BF16",
            vec![mi],
        );
        t(
            &mut map,
            "model.visual.merger.linear_fc2.weight".into(),
            "BF16",
            vec![out, mi],
        );
        t(
            &mut map,
            "model.visual.merger.linear_fc2.bias".into(),
            "BF16",
            vec![out],
        );
        t(
            &mut map,
            "model.visual.merger.norm.weight".into(),
            "BF16",
            vec![vh],
        );
        t(
            &mut map,
            "model.visual.merger.norm.bias".into(),
            "BF16",
            vec![vh],
        );
        t(
            &mut map,
            "model.visual.patch_embed.proj.weight".into(),
            "BF16",
            vec![
                vh,
                vt.in_channels as u64,
                vt.temporal_patch_size as u64,
                vt.patch_size as u64,
                vt.patch_size as u64,
            ],
        );
        t(
            &mut map,
            "model.visual.patch_embed.proj.bias".into(),
            "BF16",
            vec![vh],
        );
        t(
            &mut map,
            "model.visual.pos_embed.weight".into(),
            "BF16",
            vec![vt.num_position_embeddings as u64, vh],
        );
    }
    map
}

/// Total n-gram embedding table rows, from config math, pinned against the artifact:
/// per-head vocab sizes are the first `ngram_heads` CONSECUTIVE PRIMES >= ngram_vocab_size_base
/// (SEMANTICS.md §PLE — the runtime values ship as checkpoint I64 buffers and are LOADED,
/// never re-derived; only the census SHAPE uses this derivation), summed and rounded UP to
/// make_ngram_vocab_size_divisible_by: Σ(first 16 primes >= 2e7) = 320,001,446 -> 320,001,536
/// = 128 shards x 2,500,012 rows (census receipt: ngram_embedding.shard_S [2500012, 160]).
// VERIFY: where the 90 pad rows land (per-head vs table-end) is single-artifact evidence and
// irrelevant to shapes; a sibling whose shard shapes disagree refuses at the census gate.
fn ngram_table_rows(q: &Qwen4ExpConfig) -> u64 {
    let mut sum = 0u64;
    let mut candidate = q.ngram_vocab_size_base.max(2);
    let mut found = 0u32;
    while found < q.ngram_heads() {
        if is_prime(candidate) {
            sum += candidate;
            found += 1;
        }
        candidate += 1;
    }
    let div = q.make_ngram_vocab_size_divisible_by as u64;
    assert!(div > 0, "qwen4_exp make_ngram_vocab_size_divisible_by is 0");
    sum.div_ceil(div) * div
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut d = 3;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

fn tensor_owner(name: &str) -> TensorOwner {
    if let Some(rest) = name.strip_prefix("model.language_model.layers.") {
        if let Some(index) = rest.split('.').next().and_then(|s| s.parse().ok()) {
            return TensorOwner::Layer(index);
        }
    }
    if let Some(rest) = name.strip_prefix("mtp.layers.") {
        if let Some(depth) = rest.split('.').next().and_then(|s| s.parse().ok()) {
            return TensorOwner::Mtp(depth);
        }
    }
    if name.starts_with("mtp.") {
        return TensorOwner::Mtp(0);
    }
    if let Some(rest) = name.strip_prefix("model.visual.blocks.") {
        if let Some(block) = rest.split('.').next().and_then(|s| s.parse().ok()) {
            return TensorOwner::Vision(Some(block));
        }
    }
    if name.starts_with("model.visual.") {
        return TensorOwner::Vision(None);
    }
    TensorOwner::Global
}

/// Compact family key: trunk names lose the wrapper prefix, vision names keep their module
/// path. Uniqueness comes from the checkpoint namespace itself.
fn semantic_key(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("model.language_model.") {
        format!("trunk.{rest}")
    } else if let Some(rest) = name.strip_prefix("model.visual.") {
        format!("visual.{rest}")
    } else {
        name.to_string()
    }
}

fn semantic_tensor_id(name: &str, n_trunk: u32) -> TensorId {
    match name {
        "model.language_model.embed_tokens.weight" => return TensorId::TokenEmbedding,
        "lm_head.weight" => return TensorId::OutputProjection,
        // qwen4_exp MTP glue: TWO separate projections (never the concat FusionProjection),
        // hidden norm over the WIDE stream.
        "mtp.fc_embedding.weight" => {
            return TensorId::Mtp {
                depth: 0,
                tensor: MtpTensor::EmbeddingProjection,
            };
        }
        "mtp.fc_hidden.weight" => {
            return TensorId::Mtp {
                depth: 0,
                tensor: MtpTensor::HiddenProjection,
            };
        }
        "mtp.pre_fc_norm_embedding.weight" => {
            return TensorId::Mtp {
                depth: 0,
                tensor: MtpTensor::EmbeddingNorm,
            };
        }
        "mtp.pre_fc_norm_hidden.weight" => {
            return TensorId::Mtp {
                depth: 0,
                tensor: MtpTensor::HiddenNorm,
            };
        }
        _ => {}
    }
    if let Some(rest) = name.strip_prefix("model.language_model.layers.") {
        if let Some((index, suffix)) = rest.split_once('.') {
            if let Ok(index) = index.parse() {
                if let Some((tensor, _)) = layer_tensor_for_suffix(suffix) {
                    return TensorId::Layer { index, tensor };
                }
            }
        }
    }
    // The MTP decoder layer reuses the trunk block schema at global index n_trunk + depth
    // (the plan's MtpBlockPlan.layer.index), like the dsv4 dspark mapping.
    if let Some(rest) = name.strip_prefix("mtp.layers.") {
        if let Some((depth, suffix)) = rest.split_once('.') {
            if let Ok(depth) = depth.parse::<u32>() {
                if let Some((tensor, _)) = layer_tensor_for_suffix(suffix) {
                    return TensorId::Layer {
                        index: n_trunk + depth,
                        tensor,
                    };
                }
            }
        }
    }
    TensorId::Family {
        family: "qwen4_exp",
        key: semantic_key(name),
    }
}

fn transform_for(name: &str) -> TensorTransform {
    let suffix = name
        .strip_prefix("model.language_model.layers.")
        .or_else(|| name.strip_prefix("mtp.layers."))
        .and_then(|rest| rest.split_once('.'))
        .map(|(_, suffix)| suffix);
    match suffix.and_then(layer_tensor_for_suffix) {
        Some((_, transform)) => transform,
        None => TensorTransform::Identity,
    }
}

/// Typed roles + load transforms for the per-layer suffixes shared with existing programs.
/// GDN reorders carry over from the canonical qwen35 HF schema because
/// Qwen4ExpTextGatedDeltaNet SUBCLASSES Qwen3_5GatedDeltaNet (SEMANTICS.md §GDN — identical
/// projection layout; the sigmoid z-gate divergence is an activation, not a layout).
/// Family-specific rows (hyper-connections, indexer, PLE) stay Identity: fold/reorder
/// decisions for those are reference-executor work, and raw bytes fail closed.
fn layer_tensor_for_suffix(suffix: &str) -> Option<(LayerTensor, TensorTransform)> {
    use TensorTransform as T;
    Some(match suffix {
        "linear_attn.in_proj_qkv.weight" => (LayerTensor::GdnQkv, T::QkvVReorderRows),
        "linear_attn.in_proj_z.weight" => (LayerTensor::GdnGate, T::ZReorderRows),
        "linear_attn.in_proj_b.weight" => (LayerTensor::GdnBeta, T::AbReorderRows),
        "linear_attn.in_proj_a.weight" => (LayerTensor::GdnAlpha, T::AbReorderRows),
        "linear_attn.out_proj.weight" => (LayerTensor::GdnOutput, T::OutReorderColumns),
        "linear_attn.A_log" => (LayerTensor::GdnA, T::NegExpReorderHeads),
        "linear_attn.dt_bias" => (LayerTensor::GdnDtBias, T::ReorderHeads),
        "linear_attn.norm.weight" => (LayerTensor::GdnNorm, T::Identity),
        "linear_attn.conv1d.weight" => (LayerTensor::GdnConv1d, T::Conv1dSqueezeReorder),
        // QSA: fused [q|gate] q_proj splits at execution (q_gate_split), like qwen35.
        "self_attn.q_proj.weight" => (LayerTensor::Query, T::Identity),
        "self_attn.k_proj.weight" => (LayerTensor::Key, T::Identity),
        "self_attn.v_proj.weight" => (LayerTensor::Value, T::Identity),
        "self_attn.o_proj.weight" => (LayerTensor::AttentionOutput, T::Identity),
        "self_attn.q_norm.weight" => (LayerTensor::QueryNorm, T::Identity),
        "self_attn.k_norm.weight" => (LayerTensor::KeyNorm, T::Identity),
        // MoE: fused 3D banks; gate_up splits like the gemma4 precedent.
        "mlp.gate.weight" => (LayerTensor::MoeRouter, T::Identity),
        "mlp.experts.gate_up_proj" => (LayerTensor::MoeExpertGateUpBank, T::SplitExpertGateUp),
        "mlp.experts.down_proj" => (LayerTensor::MoeExpertDownBank, T::Identity),
        "mlp.shared_expert.gate_proj.weight" => (LayerTensor::SharedMlpGate, T::Identity),
        "mlp.shared_expert.up_proj.weight" => (LayerTensor::SharedMlpUp, T::Identity),
        "mlp.shared_expert.down_proj.weight" => (LayerTensor::SharedMlpDown, T::Identity),
        "mlp.shared_expert_gate.weight" => (LayerTensor::SharedMlpInputGate, T::Identity),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_plan::{
        AttentionPlan, GdnGateActivation, MtpFusionPlan, ResidualTopology, RouterPlan,
    };
    use crate::tensor_contract::{IntegerType, StorageLayout, TensorCensusEntry};

    fn artifact_config() -> ModelConfig {
        ModelConfig::from_hf(&HfConfig::parse(
            &crate::config::hf_tests::qwen4exp_config_json(),
        ))
    }

    #[test]
    fn tiny_plan_compiles_the_whole_family_program() {
        let plan = PACK.compile_tiny_plan().unwrap();
        assert_eq!(plan.layers.len(), 4);
        assert_eq!(plan.mtp_blocks.len(), 1);
        assert!(matches!(
            plan.layers[0].attention,
            AttentionPlan::GatedDeltaNet(gdn) if gdn.gate_activation == GdnGateActivation::Sigmoid
        ));
        assert!(matches!(plan.layers[3].attention, AttentionPlan::Full(_)));
        assert!(plan.layers[3].sparse_overlay.is_some());
        assert!(
            plan.layers[1].ple.is_some(),
            "ple_layer_ids [2] is one-indexed"
        );
        assert!(matches!(
            plan.layers[0].residual,
            ResidualTopology::GatedResidual {
                streams: 2,
                bottleneck_rank: 4
            }
        ));
        assert!(plan.exit_mixer.is_some());
        assert_eq!(
            plan.mtp_blocks[0].input.fusion,
            MtpFusionPlan::SeparateProjections
        );
        let crate::model_plan::MlpPlan::Moe(moe) = &plan.layers[0].mlp else {
            panic!("tiny qwen4_exp is MoE");
        };
        assert_eq!(moe.router, RouterPlan::Softmax);
        assert!(moe.shared.as_ref().unwrap().gated);
    }

    #[test]
    fn pack_resolves_by_alias_and_matches_the_artifact_config() {
        let config = artifact_config();
        for alias in PACK.aliases {
            assert_eq!(by_alias(alias).unwrap().family, "qwen4_exp");
        }
        assert_eq!(for_config(&config).unwrap().family, "qwen4_exp");
        PACK.compile_plan(&config).unwrap();
    }

    /// Census gate: the config-derived expected map must reproduce the banked artifact
    /// census EXACTLY — names, dtypes, shapes — and the generated contract must BIND it.
    /// Fixture: research/qwen4exp-bringup-20260829/raw/census-names.tsv (derived from
    /// raw/q48fn-census.tsv.gz, name/dtype/shape columns).
    #[test]
    fn expected_census_matches_the_banked_artifact_and_the_contract_binds_it() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../research/qwen4exp-bringup-20260829/raw/census-names.tsv"
        );
        let Ok(text) = std::fs::read_to_string(path) else {
            eprintln!("SKIP qwen4exp census gate: no fixture at {path}");
            return;
        };
        let mut artifact: BTreeMap<String, (String, Vec<u64>)> = BTreeMap::new();
        for line in text.lines().skip(1) {
            let mut cols = line.split('\t');
            let (Some(name), Some(dtype), Some(shape)) = (cols.next(), cols.next(), cols.next())
            else {
                panic!("malformed census fixture line: {line}");
            };
            let shape: Vec<u64> = shape
                .trim_start_matches('[')
                .trim_end_matches(']')
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().parse().expect("census shape"))
                .collect();
            artifact.insert(name.to_string(), (dtype.to_string(), shape));
        }
        assert_eq!(artifact.len(), 1658, "banked census tensor count");

        let config = artifact_config();
        let expected = expected_census(&config);
        let missing: Vec<_> = expected
            .keys()
            .filter(|k| !artifact.contains_key(*k))
            .collect();
        let extra: Vec<_> = artifact
            .keys()
            .filter(|k| !expected.contains_key(*k))
            .collect();
        assert!(missing.is_empty(), "derived-not-in-artifact: {missing:?}");
        assert!(extra.is_empty(), "artifact-not-derived: {extra:?}");
        for (name, spec) in &expected {
            let (dtype, shape) = &artifact[name];
            assert_eq!(spec.dtype, dtype, "{name} dtype");
            assert_eq!(&spec.shape, shape, "{name} shape");
        }

        // The contract binds the real census: every tensor claimed exactly once, nothing
        // missing, nothing extra, shard bank grouped as ONE requirement of 128 names.
        let plan = PACK.compile_plan(&config).unwrap();
        let contract = PACK
            .compile_tensor_contract(
                &config,
                &plan,
                CheckpointDialect::HfSafetensors,
                ContractOptions::default(),
            )
            .unwrap();
        let census: Vec<TensorCensusEntry> = artifact
            .iter()
            .map(|(name, (dtype, shape))| TensorCensusEntry {
                name: name.clone(),
                shape: shape.clone(),
                storage: match dtype.as_str() {
                    "BF16" => StorageLayout::Float(crate::tensor_contract::FloatType::Bf16),
                    "I64" => StorageLayout::Integer(IntegerType::I64),
                    other => panic!("unexpected artifact dtype {other}"),
                },
                // The fixture carries names/dtypes/shapes only; bytes follow exactly
                // from them for the unquantized dtypes this census contains.
                physical_bytes: shape.iter().product::<u64>()
                    * match dtype.as_str() {
                        "BF16" => 2,
                        "I64" => 8,
                        other => panic!("unexpected artifact dtype {other}"),
                    },
            })
            .collect();
        let bound = contract.bind(&census).unwrap();
        assert_eq!(bound.tensors.len(), contract.requirements.len());

        let shard_bank = contract
            .requirements
            .iter()
            .find(|requirement| requirement.match_mode == TensorMatch::All)
            .expect("ngram shard bank");
        assert_eq!(shard_bank.names.len(), 128);
        assert_eq!(shard_bank.names[0].contains("shard_0."), true);
        assert_eq!(shard_bank.names[127].contains("shard_127."), true);
        assert_eq!(shard_bank.shape, vec![2_500_012, 160]);

        // Typed spot checks against the semantic namespace.
        let has = |id: &TensorId| contract.requirements.iter().any(|r| &r.id == id);
        assert!(has(&TensorId::Layer {
            index: 3,
            tensor: LayerTensor::Query
        }));
        assert!(has(&TensorId::Layer {
            index: 0,
            tensor: LayerTensor::GdnQkv
        }));
        assert!(has(&TensorId::Layer {
            index: 48,
            tensor: LayerTensor::MoeExpertGateUpBank
        }));
        assert!(has(&TensorId::Mtp {
            depth: 0,
            tensor: MtpTensor::EmbeddingProjection
        }));
        assert!(
            !has(&TensorId::OutputNorm),
            "qwen4_exp has NO final norm module — the exit mixer replaces it"
        );
        let gate_up = contract
            .requirements
            .iter()
            .find(|r| {
                r.id == TensorId::Layer {
                    index: 0,
                    tensor: LayerTensor::MoeExpertGateUpBank,
                }
            })
            .unwrap();
        assert_eq!(gate_up.shape, vec![512, 1280, 2560]);
        assert_eq!(gate_up.transform, TensorTransform::SplitExpertGateUp);
        assert_eq!(
            bound.tensors[&TensorId::Layer {
                index: 48,
                tensor: LayerTensor::Query
            }]
                .checkpoint_names,
            vec!["mtp.layers.0.self_attn.q_proj.weight"]
        );
    }

    /// The census math itself is exercised without the fixture too (totals + spot shapes),
    /// so a checkout without the research dir still guards the derivation.
    #[test]
    fn expected_census_totals_and_spot_shapes() {
        let config = artifact_config();
        let census = expected_census(&config);
        assert_eq!(census.len(), 1658);
        let s = &census["model.language_model.layers.3.self_attn.q_proj.weight"];
        assert_eq!(
            (s.dtype, s.shape.as_slice()),
            ("BF16", &[12288u64, 2560][..])
        );
        let s = &census["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"];
        assert_eq!(
            (s.dtype, s.shape.as_slice()),
            ("BF16", &[10240u64, 2560][..])
        );
        let s = &census["model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight"];
        assert_eq!(
            (s.dtype, s.shape.as_slice()),
            ("BF16", &[2_500_012u64, 160][..])
        );
        let s = &census["model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets"];
        assert_eq!((s.dtype, s.shape.as_slice()), ("I64", &[16u64][..]));
        let s = &census["model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight"];
        assert_eq!((s.dtype, s.shape.as_slice()), ("BF16", &[640u64, 2560][..]));
        let s = &census["mtp.pre_fc_norm_hidden.weight"];
        assert_eq!((s.dtype, s.shape.as_slice()), ("BF16", &[10240u64][..]));
        let s = &census["model.visual.patch_embed.proj.weight"];
        assert_eq!(
            (s.dtype, s.shape.as_slice()),
            ("BF16", &[1152u64, 3, 2, 16, 16][..])
        );
        // absent-by-derivation: no final norm, no PLE on layer 2, no GDN on a QSA layer.
        assert!(!census.contains_key("model.language_model.norm.weight"));
        assert!(!census.contains_key("model.language_model.layers.2.ple.conv1d.weight"));
        assert!(!census.contains_key("model.language_model.layers.3.linear_attn.norm.weight"));
        assert!(!census.contains_key("mtp.layers.0.ple.conv1d.weight"));
    }

    /// NVFP4-mint census gate: the PerExpertModelopt derivation must reproduce the banked
    /// artifact census EXACTLY (names, dtypes, shapes) and the dialect contract must BIND
    /// it. Fixture: research/qwen4exp-bringup-20260829/raw/nvfp4-census-names.tsv
    /// (Avifenesh/Qwen3.8-Flash-Next-NVFP4 safetensors headers, incl. the BF16 mtp graft
    /// shard — 296,347 tensors).
    #[test]
    fn nvfp4_census_matches_the_banked_mint_and_the_dialect_contract_binds_it() {
        use crate::tensor_contract::{IntegerType, QuantLayout, StorageLayout, TensorCensusEntry};
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../research/qwen4exp-bringup-20260829/raw/nvfp4-census-names.tsv"
        );
        let Ok(text) = std::fs::read_to_string(path) else {
            eprintln!("SKIP qwen4exp nvfp4 census gate: no fixture at {path}");
            return;
        };
        let mut artifact: BTreeMap<String, (String, Vec<u64>)> = BTreeMap::new();
        for line in text.lines().skip(1) {
            let mut cols = line.split('\t');
            let (Some(name), Some(dtype), Some(shape)) = (cols.next(), cols.next(), cols.next())
            else {
                panic!("malformed census fixture line: {line}");
            };
            let shape: Vec<u64> = shape
                .trim_start_matches('[')
                .trim_end_matches(']')
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().parse().expect("census shape"))
                .collect();
            artifact.insert(name.to_string(), (dtype.to_string(), shape));
        }
        assert_eq!(artifact.len(), 296_347, "banked NVFP4 census tensor count");

        let config = artifact_config();
        let expected = expected_census_for(&config, ExpertDialect::PerExpertModelopt);
        let missing: Vec<_> = expected
            .keys()
            .filter(|k| !artifact.contains_key(*k))
            .take(20)
            .collect();
        let extra: Vec<_> = artifact
            .keys()
            .filter(|k| !expected.contains_key(*k))
            .take(20)
            .collect();
        assert!(missing.is_empty(), "derived-not-in-artifact: {missing:?}");
        assert!(extra.is_empty(), "artifact-not-derived: {extra:?}");
        for (name, spec) in &expected {
            let (dtype, shape) = &artifact[name];
            assert_eq!(spec.dtype, dtype, "{name} dtype");
            assert_eq!(&spec.shape, shape, "{name} shape");
        }

        // The dialect contract binds the real census. Census normalization: modelopt
        // scale siblings fold into their U8 primary as QuantLayout auxiliaries and the
        // packed shape presents LOGICALLY ([out, in], last dim x2) — the dsv4 convention.
        let plan = PACK.compile_plan(&config).unwrap();
        let contract =
            tensor_contract_for(&config, &plan, ExpertDialect::PerExpertModelopt).unwrap();
        let census: Vec<TensorCensusEntry> = artifact
            .iter()
            .filter(|(name, _)| sibling_primary(name, &expected).is_none())
            .map(|(name, (dtype, shape))| {
                // Bytes follow exactly from the fixture's dtypes/shapes; a quantized
                // primary carries its recognized scale-plane siblings per the census
                // contract (`physical_bytes` doc), so fold those in here too.
                let dtype_bytes = |d: &str| -> u64 {
                    match d {
                        "BF16" => 2,
                        "I64" | "F64" => 8,
                        "F32" => 4,
                        "U8" | "F8_E4M3" => 1,
                        other => panic!("unexpected artifact dtype {other}"),
                    }
                };
                let self_bytes = shape.iter().product::<u64>() * dtype_bytes(dtype);
                let mut physical_bytes = self_bytes;
                let storage = match dtype.as_str() {
                    "BF16" => StorageLayout::Float(crate::tensor_contract::FloatType::Bf16),
                    "I64" => StorageLayout::Integer(IntegerType::I64),
                    "U8" => StorageLayout::Quantized(QuantLayout {
                        format: "NVFP4".into(),
                        block_shape: vec![16],
                        auxiliaries: {
                            let stem = name.strip_suffix(".weight").unwrap();
                            [
                                format!("{stem}.weight_scale"),
                                format!("{stem}.weight_scale_2"),
                                format!("{stem}.input_scale"),
                            ]
                            .into_iter()
                            .filter(|s| artifact.contains_key(s))
                            .inspect(|s| {
                                let (aux_dtype, aux_shape) = &artifact[s];
                                physical_bytes +=
                                    aux_shape.iter().product::<u64>() * dtype_bytes(aux_dtype);
                            })
                            .collect()
                        },
                    }),
                    other => panic!("unexpected NVFP4 artifact dtype {other}"),
                };
                TensorCensusEntry {
                    name: name.clone(),
                    shape: shape.clone(),
                    storage,
                    physical_bytes,
                }
            })
            .map(|mut entry| {
                if matches!(entry.storage, StorageLayout::Quantized(_)) {
                    *entry.shape.last_mut().unwrap() *= 2;
                }
                entry
            })
            .collect();
        let bound = contract.bind(&census).unwrap();
        assert_eq!(bound.tensors.len(), contract.requirements.len());

        // Typed spot checks.
        let gate0 = contract
            .requirements
            .iter()
            .find(|r| {
                r.id == TensorId::Expert {
                    layer: 0,
                    expert: 0,
                    tensor: ExpertTensor::Gate,
                }
            })
            .expect("per-expert gate row");
        assert_eq!(gate0.shape, vec![640, 2560]);
        assert_eq!(gate0.quant, QuantConstraint::Nvfp4);
        assert_eq!(
            gate0.auxiliaries.as_deref().unwrap(),
            [
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale",
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_2",
                "model.language_model.layers.0.mlp.experts.0.gate_proj.input_scale",
            ]
        );
        let down47 = contract
            .requirements
            .iter()
            .find(|r| {
                r.id == TensorId::Expert {
                    layer: 47,
                    expert: 511,
                    tensor: ExpertTensor::Down,
                }
            })
            .expect("last per-expert down row");
        assert_eq!(down47.shape, vec![2560, 640]);
        // The unsharded ngram table binds under the SAME bank id as the fused dialect.
        let table = contract
            .requirements
            .iter()
            .find(|r| {
                r.id == TensorId::Family {
                    family: "qwen4_exp",
                    key: "trunk.layers.1.ple.ple_embedding.ngram_embedding".into(),
                }
            })
            .expect("unsharded ngram table row");
        assert_eq!(
            table.names,
            ["model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight"]
        );
        assert_eq!(table.shape, vec![320_001_536, 160]);
        // The mtp graft keeps FUSED BF16 experts.
        assert!(
            contract
                .requirements
                .iter()
                .any(|r| r.names == ["mtp.layers.0.mlp.experts.gate_up_proj"]
                    && r.quant == QuantConstraint::ExactFloat(FloatType::Bf16))
        );
        // No fused trunk banks in this dialect.
        assert!(!expected.contains_key("model.language_model.layers.0.mlp.experts.gate_up_proj"));
    }

    /// Fixture-free structural guard for the NVFP4 derivation (a checkout without the
    /// research dir still protects the math): totals + spot shapes on the artifact config.
    #[test]
    fn nvfp4_expected_census_totals_and_spot_shapes() {
        let config = artifact_config();
        let census = expected_census_for(&config, ExpertDialect::PerExpertModelopt);
        // 1658 BF16 rows - 96 fused bank rows - 128 ngram shards + 1 unsharded table
        // + 48 layers x 512 experts x 3 projections x 4 modelopt siblings = 296,347.
        assert_eq!(census.len(), 296_347);
        let s = &census["model.language_model.layers.0.mlp.experts.0.gate_proj.weight"];
        assert_eq!((s.dtype, s.shape.as_slice()), ("U8", &[640u64, 1280][..]));
        let s = &census["model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale"];
        assert_eq!(
            (s.dtype, s.shape.as_slice()),
            ("F8_E4M3", &[640u64, 160][..])
        );
        let s = &census["model.language_model.layers.47.mlp.experts.511.down_proj.weight"];
        assert_eq!((s.dtype, s.shape.as_slice()), ("U8", &[2560u64, 320][..]));
        let s = &census["model.language_model.layers.47.mlp.experts.511.down_proj.input_scale"];
        assert_eq!((s.dtype, s.shape.as_slice()), ("F32", &[][..]));
        let s = &census["model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight"];
        assert_eq!(
            (s.dtype, s.shape.as_slice()),
            ("BF16", &[320_001_536u64, 160][..])
        );
        // mtp stays fused BF16; the fused trunk banks and ngram shards are absent.
        let s = &census["mtp.layers.0.mlp.experts.gate_up_proj"];
        assert_eq!(
            (s.dtype, s.shape.as_slice()),
            ("BF16", &[512u64, 1280, 2560][..])
        );
        assert!(!census.contains_key("model.language_model.layers.0.mlp.experts.gate_up_proj"));
        assert!(!census.contains_key(
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
        ));
    }
}
