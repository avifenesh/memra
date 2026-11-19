//! Hy4-flash (`hy_v4_flash`) pack: config normalization, the typed plan and the config-derived
//! tensor schema of the ModelOpt NVFP4 artifact (`Avifenesh/Hy4-flash-NVFP4@f9eac06cf022`,
//! research/modelplan-onboarding-hy4flash-20261119/). No executor runs the plan yet: every
//! Hy4-flash operation is a registry row with no surface, and `ModelPlan::compile` and the
//! reference executor refuse the family.
//!
//! Checkpoint namespaces (2,429 header tensors):
//!   model.embed_tokens / lm_head / model.norm / model.hc_head   embedding, F32-computed head,
//!                                                              final norm, exit stream collapse
//!   model.layers.N.*              42 trunk layers (12 indexed, 30 sliding; layer 0 dense)
//!   model.layers.N.engram_layer.* Engram on the 0-based layers id - 1 of `engram_layer_ids`
//!   model.dfly_layers.*           DFly2 drafter (5 blocks plus fusion, correction, confidence)
//!   vit.*                         vision tower and its perceiver projection
//! The text plan does not execute the drafter or the tower (vision is out of scope, DFly2 is a
//! later step); the contract still owns every one of their tensors, so a missing, extra or
//! reshaped one fails the census like a trunk tensor.
//!
//! Quant layouts come from the per-layer ModelOpt map (`quantization_config.quantized_layers`,
//! the same map as `hf_quant_config.json`): `W4A16_NVFP4` modules are U8 e2m1 pairs with an
//! F8_E4M3 `weight_scale` per 16 inputs and an F32 `weight_scale_2` (`[512]` on stacked expert
//! banks, `[]` otherwise); `FP8_PB_WO` modules are F8_E4M3 with an F32 `weight_scale_inv` per
//! 128x128 block. The Engram tables are F8_E4M3 rows with one F32 `weight_scale` per row, which
//! no ModelOpt algorithm names. Everything else is BF16, except the F32 hyper-connection
//! parameters, router biases, trunk sinks and indexer pooling weights, and the I32 lookup tables.

use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::config::{HyV4FlashConfig, HyV4LayerKind};
use crate::dsv4::TensorSpec;
use crate::model_plan::{
    ActivationPlan, AttentionPlan, DenseMlpPlan, DraftSourcePlan, HyV4AttentionKind,
    HyV4AttentionPlan, HyV4EngramMultiplierBound, HyV4EngramNormEpsilon, HyV4EngramPlan,
    HyV4FlashNumerics, HyV4ResidualPlan, HyV4StatePlan, HyV4WeightFormat, IndexPoolPlan,
    LatentMoePlan, LayerPlan, MlpPlan, MoeMlpPlan, NormKind, NormPlan, ResidualTopology,
    RopeFactors, RopePlan, RouterPlan, SharedMlpPlan, StatePlan, TensorPresence, WeightTransform,
};
use crate::modelopt_quant::{LayerQuant, ModeloptAlgo};
use crate::tensor_contract::{
    FloatType, LayerTensor, QuantConstraint, TensorId, TensorMatch, TensorOwner, TensorRequirement,
    TensorTransform,
};

const FAMILY: &str = "hy_v4_flash";

/// Width of the DFly2 dense MLPs. The config does not declare it (`dfly_intermediate_size` is
/// absent); the vendor runtime takes it from the checkpoint (vLLM `hy_v4/nvidia/dfly.py`
/// `materialize_ffn`). The census gate pins it against the artifact's headers.
pub const DFLY_FFN_WIDTH: u64 = 9216;

pub static PACK: ModelPack = ModelPack {
    family: FAMILY,
    output_head: OutputHeadContract::SeparateHead,
    tensor_consumption: TensorConsumption::Report,
    aliases: &["hy_v4_flash", "hy-v4-flash", "hy4-flash"],
    config_layout: ConfigLayout::FlatOrTextConfig,
    tokenizer_sources: &[TokenizerSource::TokenizerJson],
    // The artifact ships `chat_template.jinja`; the hy_v4 dialect and tool parser are a later
    // step of the lane.
    template: TemplateContract::ArtifactRequired,
    // Nothing runs the plan: no reference executor, no tuned surface.
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
    checkpoint_parity_classes: &[],
    checkpoint_probe: None,
    matches_config: |config| matches!(config.arch, Arch::HyV4Flash) && config.hy_v4_flash.is_some(),
    plan_builder,
    tensor_schema,
    tiny_plan: None,
};

fn hy4(config: &ModelConfig) -> Result<&HyV4FlashConfig, PlanCompileError> {
    config
        .hy_v4_flash
        .as_ref()
        .ok_or_else(|| PlanCompileError::UnsupportedSemantics {
            field: "hy_v4_flash",
            value: "config carries no hy_v4_flash block".into(),
        })
}

fn rms(epsilon: f32) -> NormPlan {
    NormPlan {
        kind: NormKind::Rms,
        epsilon,
        weight_transform: WeightTransform::Identity,
    }
}

fn plan_builder(config: &ModelConfig) -> Result<ModelPlan, PlanCompileError> {
    config.validate_plan_semantics()?;
    let hy = hy4(config)?;
    // The quant map must describe exactly the modules the census quantizes.
    expected_census(hy).map_err(|value| PlanCompileError::UnsupportedSemantics {
        field: "quantization_config.quantized_layers",
        value,
    })?;
    let norm = rms(hy.rms_norm_eps);
    let layers = (0..hy.trunk_layers())
        .map(|il| layer_plan(hy, il, norm))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ModelPlan {
        speech: None,
        arch: config.arch.clone(),
        prefill_activation: None,
        hidden_size: hy.hidden_size,
        vocab_size: hy.vocab_size,
        context_length: hy.max_position_embeddings,
        embedding_scale: 1.0,
        // The tower and the DFly2 drafter are contract-owned and not part of the text plan.
        vision: None,
        mimo_audio_patch: None,
        multimodal: None,
        per_layer_input: None,
        layers,
        output_norm: norm,
        exit_mixer: None,
        logits: Vec::new(),
        mtp_blocks: Vec::new(),
        drafter: None,
        draft_source: DraftSourcePlan::None,
        sampling_defaults: None,
        partition_boundaries: (1..hy.trunk_layers() as usize).collect(),
        deepseek_v41: None,
        hy_v4_flash: Some(HyV4FlashNumerics {
            streams: hy.hc_mult,
            exit_epsilon: hy.hc_eps,
            output_projection_f32: true,
        }),
    })
}

fn weight_format(quant: Option<LayerQuant>) -> HyV4WeightFormat {
    match quant {
        None => HyV4WeightFormat::Bf16,
        Some(LayerQuant {
            algo: ModeloptAlgo::Fp8PbWo,
            group_size,
        }) => HyV4WeightFormat::Fp8Block { block: group_size },
        Some(LayerQuant {
            algo: ModeloptAlgo::W4a16Nvfp4,
            group_size,
        }) => HyV4WeightFormat::Nvfp4W4A16 { group: group_size },
    }
}

/// One format for a set of modules, or a refusal naming the disagreement.
fn shared_format(
    hy: &HyV4FlashConfig,
    modules: &[String],
) -> Result<HyV4WeightFormat, PlanCompileError> {
    let formats: BTreeSet<_> = modules
        .iter()
        .map(|module| format!("{:?}", weight_format(hy.quant.get(module))))
        .collect();
    if formats.len() != 1 {
        return Err(PlanCompileError::UnsupportedSemantics {
            field: "quantization_config.quantized_layers",
            value: format!("{modules:?} carry different formats {formats:?}; the plan runs one"),
        });
    }
    Ok(weight_format(hy.quant.get(&modules[0])))
}

fn layer_plan(
    hy: &HyV4FlashConfig,
    il: u32,
    norm: NormPlan,
) -> Result<LayerPlan, PlanCompileError> {
    let kind = hy.layer_kinds[il as usize];
    let geometry = hy.head_geometry(il);
    let p = format!("model.layers.{il}.");
    let attention_modules: Vec<String> = ["q_b_proj", "o_proj", "linear_gate"]
        .iter()
        .map(|m| format!("{p}self_attn.{m}"))
        .collect();
    let attention = HyV4AttentionPlan {
        kind: match kind {
            HyV4LayerKind::Indexed => HyV4AttentionKind::Indexed(IndexPoolPlan {
                heads: hy.index_n_heads,
                head_dim: hy.index_head_dim,
                rope_head_dim: geometry.qk_rope_head_dim,
                pool: hy.index_pool,
                top_k: hy.index_topk,
            }),
            HyV4LayerKind::Sliding => HyV4AttentionKind::Sliding {
                window: hy.sliding_window,
            },
        },
        query_heads: geometry.heads,
        q_lora_rank: hy.q_lora_rank,
        kv_lora_rank: hy.kv_lora_rank,
        qk_nope_head_dim: geometry.qk_nope_head_dim,
        qk_rope_head_dim: geometry.qk_rope_head_dim,
        value_head_dim: geometry.v_head_dim,
        rope: RopePlan {
            dimensions: geometry.qk_rope_head_dim,
            base: geometry.rope_theta,
            factors: RopeFactors::None,
        },
        sink: TensorPresence::Required,
        post_projection_norm: norm,
        projection_weights: shared_format(hy, &attention_modules)?,
    };
    let activation = ActivationPlan::SwiGluPreClamped {
        limit: hy.swiglu_limit,
    };
    let mlp = if hy.dense_layers[il as usize] {
        // The dense MLP is plain SwiGLU (transformers `HYV4FlashMLP`, no clamp).
        MlpPlan::Dense(DenseMlpPlan {
            intermediate_size: hy.intermediate_size,
            activation: ActivationPlan::Silu,
        })
    } else {
        let experts: Vec<String> = ["gate_proj", "up_proj", "down_proj"]
            .iter()
            .map(|m| format!("{p}mlp.experts.{m}"))
            .collect();
        MlpPlan::LatentMoe(LatentMoePlan {
            latent_size: hy.routed_expert_hidden_size,
            expert_weights: shared_format(hy, &experts)?,
            moe: MoeMlpPlan {
                expert_count: hy.n_routed_experts,
                experts_per_token: hy.num_experts_per_tok,
                expert_intermediate_size: hy.moe_intermediate_size,
                // transformers `HYV4FlashTopkRouter`: F32 sigmoid scores, the correction bias
                // steers selection only, the selected raw scores are renormalized (+1e-20),
                // then scaled.
                router: RouterPlan::Sigmoid {
                    normalize_selected: hy.norm_topk_prob,
                    scaling_factor: hy.routed_scaling_factor,
                    selection_bias: true,
                },
                // The shared expert is plain SwiGLU on the full width.
                shared: Some(SharedMlpPlan {
                    intermediate_size: hy.n_shared_experts * hy.moe_intermediate_size,
                    gated: false,
                }),
                // transformers `HYV4FlashExperts._apply_gate`: gate clamped from above, up to
                // +-limit, then silu(gate) * up.
                activation,
            },
            // transformers `HYV4FlashMoE.shared_experts` is the plain `HYV4FlashMLP`: no clamp.
            shared_activation: ActivationPlan::Silu,
        })
    };
    let engram = hy.engram_slot(il).map(|slot| HyV4EngramPlan {
        layer_id: hy.engram_layer_ids[slot],
        hash_seed: hy.engram_hash_seed(slot),
        max_ngram: hy.engram_max_ngram_size,
        heads: hy.engram_heads(),
        head_dim: hy.engram_head_dim(),
        table_rows: hy.engram_table_rows(slot),
        lookup_table_size: hy.engram_lookup_table_size,
        multiplier_bound: HyV4EngramMultiplierBound::LookupTableMaxPlusOne,
        norm_epsilon: HyV4EngramNormEpsilon::InputDtype,
        pad_token_id: hy.engram_pad_id,
        table_fp8_per_row: true,
    });
    Ok(LayerPlan {
        index: il,
        pre_attention_norm: norm,
        attention: AttentionPlan::HyV4(attention),
        pre_mlp_norm: norm,
        mlp,
        residual: ResidualTopology::HyV4(HyV4ResidualPlan {
            streams: hy.hc_mult,
            epsilon: hy.hc_eps,
            post_magnitude: hy.hc_magnitude,
            mix_norm_epsilon: hy.rms_norm_eps,
            mlp_output_norm: norm,
            engram,
        }),
        state: StatePlan::HyV4(HyV4StatePlan {
            latent_width: hy.kv_lora_rank,
            rope_width: geometry.qk_rope_head_dim,
            window: match kind {
                HyV4LayerKind::Indexed => None,
                HyV4LayerKind::Sliding => Some(hy.sliding_window),
            },
            index_key_width: match kind {
                HyV4LayerKind::Indexed => Some(hy.index_head_dim),
                HyV4LayerKind::Sliding => None,
            },
        }),
        sparse_overlay: None,
        ple: None,
    })
}

/// The complete expected header census, derived from config math and the per-layer quant map.
/// Shapes are safetensors row-major; packed U8 weights carry `in / 2` bytes per row. Refuses a
/// quant map entry no derived module consumes, a quantizable module the map neither quantizes
/// nor excludes, and an algorithm on a module class that cannot carry it.
pub fn expected_census(hy: &HyV4FlashConfig) -> Result<BTreeMap<String, TensorSpec>, String> {
    let mut census = Census {
        hy,
        map: BTreeMap::new(),
        consumed: BTreeSet::new(),
    };
    let h = u64::from(hy.hidden_size);
    let v = u64::from(hy.vocab_size);
    let hc = u64::from(hy.hc_mult);
    let q_lora = u64::from(hy.q_lora_rank);
    let kv_lora = u64::from(hy.kv_lora_rank);

    census.put("model.embed_tokens.weight", "BF16", vec![v, h]);
    census.put("lm_head.weight", "BF16", vec![v, h]);
    census.put("model.norm.weight", "BF16", vec![h]);
    census.put("model.hc_head.hc_head_fn", "F32", vec![hc, hc * h]);
    census.put("model.hc_head.hc_head_base", "F32", vec![hc]);
    census.put("model.hc_head.hc_head_scale", "F32", vec![1]);

    for il in 0..hy.trunk_layers() {
        let p = format!("model.layers.{il}.");
        let g = hy.head_geometry(il);
        let heads = u64::from(g.heads);
        let qk = u64::from(g.qk_nope_head_dim + g.qk_rope_head_dim);
        let value = u64::from(g.v_head_dim);
        let rope = u64::from(g.qk_rope_head_dim);
        let nope = u64::from(g.qk_nope_head_dim);
        census.put(&format!("{p}input_layernorm.weight"), "BF16", vec![h]);
        census.put(
            &format!("{p}post_attention_layernorm.weight"),
            "BF16",
            vec![h],
        );
        for sub in ["hc_attn_layer", "hc_mlp_layer"] {
            census.put(
                &format!("{p}{sub}.hc_pre.hc_fn"),
                "F32",
                vec![2 * hc, hc * h],
            );
            census.put(&format!("{p}{sub}.hc_pre.hc_base"), "F32", vec![2 * hc]);
            census.put(&format!("{p}{sub}.hc_pre.hc_scale"), "F32", vec![2]);
        }
        let a = format!("{p}self_attn.");
        census.bf16_linear(&format!("{a}q_a_proj"), q_lora, h)?;
        census.put(&format!("{a}q_a_layernorm.weight"), "BF16", vec![q_lora]);
        census.linear(&format!("{a}q_b_proj"), heads * qk, q_lora)?;
        census.bf16_linear(&format!("{a}kv_a_proj_with_mqa"), kv_lora + rope, h)?;
        census.put(&format!("{a}kv_a_layernorm.weight"), "BF16", vec![kv_lora]);
        census.bf16_linear(&format!("{a}kv_b_proj"), heads * (nope + value), kv_lora)?;
        census.linear(&format!("{a}linear_gate"), heads * value, h)?;
        census.linear(&format!("{a}o_proj"), h, heads * value)?;
        census.put(&format!("{a}post_proj_norm.weight"), "BF16", vec![h]);
        census.put(&format!("{a}learnable_sink_param"), "F32", vec![heads]);
        if hy.layer_kinds[il as usize] == HyV4LayerKind::Indexed {
            let i = format!("{a}compressed_indexer.");
            let ih = u64::from(hy.index_n_heads);
            let id = u64::from(hy.index_head_dim);
            census.bf16_linear(&format!("{i}query_projection"), ih * id, q_lora)?;
            census.bf16_linear(&format!("{i}head_weight_projection"), ih, h)?;
            census.bf16_linear(&format!("{i}token_key_projection"), id, h)?;
            census.put(&format!("{i}token_key_norm.weight"), "BF16", vec![id]);
            // The pooling runs in F32 and its weights ship in F32.
            census.put(
                &format!("{i}key_value_projection.weight"),
                "F32",
                vec![id, id],
            );
            census.put(
                &format!("{i}key_gate_projection.weight"),
                "F32",
                vec![id, id],
            );
            census.put(&format!("{i}key_norm.weight"), "F32", vec![id]);
            census.put(
                &format!("{i}intra_group_ape"),
                "F32",
                vec![u64::from(hy.index_pool), id],
            );
        }
        let m = format!("{p}mlp.");
        if hy.dense_layers[il as usize] {
            let inter = u64::from(hy.intermediate_size);
            // The plan's dense MLP carries no weight format: it is BF16, and a map that
            // quantizes it refuses.
            census.bf16_linear(&format!("{m}gate_proj"), inter, h)?;
            census.bf16_linear(&format!("{m}up_proj"), inter, h)?;
            census.bf16_linear(&format!("{m}down_proj"), h, inter)?;
            census.put(&format!("{m}post_fc2_norm.weight"), "BF16", vec![h]);
        } else {
            let experts = u64::from(hy.n_routed_experts);
            let latent = u64::from(hy.routed_expert_hidden_size);
            let inter = u64::from(hy.moe_intermediate_size);
            let shared = u64::from(hy.n_shared_experts) * inter;
            census.put(&format!("{m}gate.weight"), "BF16", vec![experts, h]);
            census.put(
                &format!("{m}gate.e_score_correction_bias"),
                "F32",
                vec![experts],
            );
            census.bf16_linear(&format!("{m}routed_expert_down_proj"), latent, h)?;
            census.bf16_linear(&format!("{m}routed_expert_up_proj"), h, latent)?;
            // gate and up are separate stacked banks, each with its own per-expert
            // second-level scale; nothing here is a fused gate_up bank.
            census.expert_bank(&format!("{m}experts.gate_proj"), experts, inter, latent)?;
            census.expert_bank(&format!("{m}experts.up_proj"), experts, inter, latent)?;
            census.expert_bank(&format!("{m}experts.down_proj"), experts, latent, inter)?;
            census.bf16_linear(&format!("{m}shared_experts.gate_proj"), shared, h)?;
            census.bf16_linear(&format!("{m}shared_experts.up_proj"), shared, h)?;
            census.bf16_linear(&format!("{m}shared_experts.down_proj"), h, shared)?;
            census.put(&format!("{m}combine_output_norm.weight"), "BF16", vec![h]);
        }
        if let Some(slot) = hy.engram_slot(il) {
            let e = format!("{p}engram_layer.");
            let width =
                u64::from(hy.engram_max_ngram_size - 1) * u64::from(hy.engram_n_embed_per_ngram);
            let rows = hy.engram_table_rows(slot);
            let dim = u64::from(hy.engram_head_dim());
            let table = format!("{e}multi_head_embedding.embedding");
            if hy.quant.get(&table).is_some() || hy.quant.is_excluded(&table) {
                return Err(format!(
                    "{table}: the engram table is FP8 per row, which no ModelOpt algorithm \
                     names; the map must neither quantize nor exclude it"
                ));
            }
            census.put(&format!("{table}.weight"), "F8_E4M3", vec![rows, dim]);
            census.put(&format!("{table}.weight_scale"), "F32", vec![rows]);
            census.put(
                &format!("{e}lookup_table"),
                "I32",
                vec![u64::from(hy.engram_lookup_table_size)],
            );
            for head in 0..hy.engram_heads() {
                census.put(
                    &format!("{e}lookup_per_head_norms.{head}.weight"),
                    "BF16",
                    vec![dim],
                );
            }
            for stream in 0..hy.hc_mult {
                census.put(
                    &format!("{e}key_projs.{stream}.weight"),
                    "BF16",
                    vec![h, width],
                );
                census.put(&format!("{e}key_projs.{stream}.bias"), "BF16", vec![h]);
                census.put(&format!("{e}norm_key.{stream}.weight"), "BF16", vec![h]);
                census.put(&format!("{e}norm_query.{stream}.weight"), "BF16", vec![h]);
                census.put(
                    &format!("{e}unified_per_hc_scales.{stream}"),
                    "BF16",
                    vec![h],
                );
            }
            census.put(&format!("{e}value_proj.weight"), "BF16", vec![h, width]);
            census.put(&format!("{e}value_proj.bias"), "BF16", vec![h]);
            census.put(&format!("{e}unified_pre_gamma"), "BF16", vec![h]);
        }
    }

    // DFly2 drafter: indexed-geometry MLA blocks with dense MLPs, no hyper-connections, no
    // indexer, a BF16 sink; then the slot-0 fusion, the hidden correction and the confidence
    // head. Contract-owned; the text plan does not run it.
    let g = hy.indexed;
    let heads = u64::from(g.heads);
    let qk = u64::from(g.qk_nope_head_dim + g.qk_rope_head_dim);
    let value = u64::from(g.v_head_dim);
    let rope = u64::from(g.qk_rope_head_dim);
    let nope = u64::from(g.qk_nope_head_dim);
    for block in 0..hy.dfly.num_hidden_layers {
        let p = format!("model.dfly_layers.{block}.");
        census.put(&format!("{p}input_layernorm.weight"), "BF16", vec![h]);
        census.put(
            &format!("{p}post_attention_layernorm.weight"),
            "BF16",
            vec![h],
        );
        let a = format!("{p}self_attn.");
        census.bf16_linear(&format!("{a}q_a_proj"), q_lora, h)?;
        census.put(&format!("{a}q_a_layernorm.weight"), "BF16", vec![q_lora]);
        census.linear(&format!("{a}q_b_proj"), heads * qk, q_lora)?;
        census.bf16_linear(&format!("{a}kv_a_proj_with_mqa"), kv_lora + rope, h)?;
        census.put(&format!("{a}kv_a_layernorm.weight"), "BF16", vec![kv_lora]);
        census.bf16_linear(&format!("{a}kv_b_proj"), heads * (nope + value), kv_lora)?;
        census.linear(&format!("{a}linear_gate"), heads * value, h)?;
        census.linear(&format!("{a}o_proj"), h, heads * value)?;
        census.put(&format!("{a}post_proj_norm.weight"), "BF16", vec![h]);
        census.put(&format!("{a}learnable_sink_param"), "BF16", vec![heads]);
        census.linear(&format!("{p}mlp.gate_proj"), DFLY_FFN_WIDTH, h)?;
        census.linear(&format!("{p}mlp.up_proj"), DFLY_FFN_WIDTH, h)?;
        census.linear(&format!("{p}mlp.down_proj"), h, DFLY_FFN_WIDTH)?;
    }
    let d = "model.dfly_layers.";
    census.put(&format!("{d}final_norm.weight"), "BF16", vec![h]);
    census.put(
        &format!("{d}confidence_head.proj.weight"),
        "BF16",
        vec![1, h],
    );
    census.put(&format!("{d}confidence_head.proj.bias"), "BF16", vec![1]);
    census.bf16_linear(&format!("{d}mtp_fusion.eh_proj"), h, 2 * h)?;
    census.put(&format!("{d}mtp_fusion.enorm.weight"), "BF16", vec![h]);
    census.put(&format!("{d}mtp_fusion.hnorm.weight"), "BF16", vec![h]);
    census.put(&format!("{d}mtp_fusion.gate"), "BF16", vec![h]);
    if hy.dfly.hidden_correction {
        let c = format!("{d}hidden_correction.");
        census.linear(&format!("{c}gate_proj"), h, 2 * h)?;
        census.linear(&format!("{c}up_proj"), h, 2 * h)?;
        census.linear(&format!("{c}down_proj"), h, h)?;
        census.put(&format!("{c}embed_norm.weight"), "BF16", vec![h]);
        census.put(&format!("{c}hidden_norm.weight"), "BF16", vec![h]);
    }

    // Vision tower (contract-owned; vision is out of scope for the text plan).
    let vis = hy.vision;
    let vd = u64::from(vis.hidden_size);
    let vi = u64::from(vis.intermediate_size);
    let patch = u64::from(vis.patch_size);
    let merge = u64::from(vis.spatial_merge_size);
    census.put(
        "vit.embeddings.patch_embedding.weight",
        "BF16",
        vec![vd, u64::from(vis.num_channels), patch, patch],
    );
    census.put("vit.embeddings.patch_embedding.bias", "BF16", vec![vd]);
    census.put(
        "vit.embeddings.position_embedding.weight",
        "BF16",
        vec![u64::from(vis.max_vit_seq_len) + 1, vd],
    );
    for layer in 0..vis.num_hidden_layers {
        let p = format!("vit.layers.{layer}.");
        for (name, shape) in [
            ("input_layernorm", vec![vd]),
            ("post_attention_layernorm", vec![vd]),
        ] {
            census.put(&format!("{p}{name}.weight"), "BF16", shape.clone());
            census.put(&format!("{p}{name}.bias"), "BF16", shape);
        }
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            census.put(&format!("{p}self_attn.{proj}.weight"), "BF16", vec![vd, vd]);
            census.put(&format!("{p}self_attn.{proj}.bias"), "BF16", vec![vd]);
        }
        census.put(
            &format!("{p}mlp.dense_h_to_4h.weight"),
            "BF16",
            vec![vi, vd],
        );
        census.put(&format!("{p}mlp.dense_h_to_4h.bias"), "BF16", vec![vi]);
        census.put(
            &format!("{p}mlp.dense_4h_to_h.weight"),
            "BF16",
            vec![vd, vi],
        );
        census.put(&format!("{p}mlp.dense_4h_to_h.bias"), "BF16", vec![vd]);
    }
    let merged = vd * merge * merge;
    let out = u64::from(vis.out_hidden_size);
    census.put("vit.perceive.norm.weight", "BF16", vec![vd]);
    census.put(
        "vit.perceive.proj.0.weight",
        "BF16",
        vec![merged, vd, merge, merge],
    );
    census.put("vit.perceive.proj.0.bias", "BF16", vec![merged]);
    census.put(
        "vit.perceive.proj.2.weight",
        "BF16",
        vec![out, merged, 1, 1],
    );
    census.put("vit.perceive.proj.2.bias", "BF16", vec![out]);
    census.put("vit.perceive.image_newline", "BF16", vec![out]);

    if let Some(unused) = hy
        .quant
        .layers
        .keys()
        .find(|module| !census.consumed.contains(*module))
    {
        return Err(format!(
            "quantized_layers names {unused}, which is not a quantizable module of this program"
        ));
    }
    Ok(census.map)
}

struct Census<'a> {
    hy: &'a HyV4FlashConfig,
    map: BTreeMap<String, TensorSpec>,
    consumed: BTreeSet<String>,
}

impl Census<'_> {
    fn put(&mut self, name: &str, dtype: &'static str, shape: Vec<u64>) {
        assert!(
            self.map
                .insert(name.to_string(), TensorSpec { dtype, shape })
                .is_none(),
            "hy_v4_flash census derives {name} twice"
        );
    }

    /// A linear the program keeps in BF16: the map must not quantize it.
    fn bf16_linear(&mut self, module: &str, out: u64, input: u64) -> Result<(), String> {
        if let Some(quant) = self.hy.quant.get(module) {
            return Err(format!(
                "{module} is {} in quantized_layers; the program keeps it BF16",
                quant.algo.name()
            ));
        }
        self.put(&format!("{module}.weight"), "BF16", vec![out, input]);
        Ok(())
    }

    /// A 2-D linear in whatever format the map gives it.
    fn linear(&mut self, module: &str, out: u64, input: u64) -> Result<(), String> {
        match self.quant(module)? {
            None => self.put(&format!("{module}.weight"), "BF16", vec![out, input]),
            Some(LayerQuant {
                algo: ModeloptAlgo::W4a16Nvfp4,
                group_size,
            }) => self.nvfp4(module, &[], out, input, group_size)?,
            Some(LayerQuant {
                algo: ModeloptAlgo::Fp8PbWo,
                group_size,
            }) => {
                // The census labels every `weight_scale_inv` grid as 128x128 blocks, and the
                // contract binds it as `Fp8Block128`; another group size is another program.
                if group_size != 128 {
                    return Err(format!(
                        "{module}: FP8_PB_WO group_size {group_size}; this program's FP8 \
                         blocks are 128x128"
                    ));
                }
                let g = u64::from(group_size);
                self.put(&format!("{module}.weight"), "F8_E4M3", vec![out, input]);
                self.put(
                    &format!("{module}.weight_scale_inv"),
                    "F32",
                    vec![out.div_ceil(g), input.div_ceil(g)],
                );
            }
        }
        Ok(())
    }

    /// A stacked routed-expert bank: NVFP4 is the only format the plan runs there.
    fn expert_bank(
        &mut self,
        module: &str,
        experts: u64,
        out: u64,
        input: u64,
    ) -> Result<(), String> {
        match self.quant(module)? {
            Some(LayerQuant {
                algo: ModeloptAlgo::W4a16Nvfp4,
                group_size,
            }) => self.nvfp4(module, &[experts], out, input, group_size),
            other => Err(format!(
                "{module}: routed expert banks are W4A16_NVFP4 in this artifact, the map says \
                 {other:?}"
            )),
        }
    }

    fn nvfp4(
        &mut self,
        module: &str,
        lead: &[u64],
        out: u64,
        input: u64,
        group: u32,
    ) -> Result<(), String> {
        let g = u64::from(group);
        if group != 16 || !input.is_multiple_of(g) {
            return Err(format!(
                "{module}: W4A16_NVFP4 group_size {group} over {input} inputs; NVFP4 scales 16"
            ));
        }
        let with = |tail: &[u64]| lead.iter().copied().chain(tail.iter().copied()).collect();
        self.put(&format!("{module}.weight"), "U8", with(&[out, input / 2]));
        self.put(
            &format!("{module}.weight_scale"),
            "F8_E4M3",
            with(&[out, input / g]),
        );
        self.put(&format!("{module}.weight_scale_2"), "F32", lead.to_vec());
        Ok(())
    }

    /// The map's entry for a quantizable module. A module the map neither quantizes nor
    /// excludes is an incomplete map, not an implicit BF16.
    fn quant(&mut self, module: &str) -> Result<Option<LayerQuant>, String> {
        match self.hy.quant.get(module) {
            Some(quant) => {
                self.consumed.insert(module.to_string());
                Ok(Some(quant))
            }
            None if self.hy.quant.is_excluded(module) => Ok(None),
            None => Err(format!(
                "{module} is neither in quantized_layers nor matched by the exclusion list"
            )),
        }
    }
}

#[allow(clippy::result_large_err)] // allow: the fat error type is the diagnostic contract here; boxing it would change the error surface
fn tensor_schema(
    config: &ModelConfig,
    _plan: &ModelPlan,
    dialect: CheckpointDialect,
    _options: ContractOptions,
) -> Result<TensorContract, TensorContractError> {
    if dialect != CheckpointDialect::HfSafetensors {
        return Err(TensorContractError::UnsupportedPlanOperation {
            operation: "hy_v4_flash non-safetensors schema",
        });
    }
    let hy = config
        .hy_v4_flash
        .as_ref()
        .ok_or(TensorContractError::UnsupportedPlanOperation {
            operation: "hy_v4_flash schema without its config",
        })?;
    let expected =
        expected_census(hy).map_err(|_| TensorContractError::UnsupportedPlanOperation {
            operation: "hy_v4_flash quant map does not match the program",
        })?;
    let auxiliary = |name: &str| {
        ["weight_scale", "weight_scale_inv", "weight_scale_2"]
            .iter()
            .any(|suffix| {
                name.strip_suffix(suffix)
                    .and_then(|stem| stem.strip_suffix('.'))
                    .is_some_and(|stem| expected.contains_key(&format!("{stem}.weight")))
            })
    };
    let mut requirements = Vec::new();
    for (name, spec) in &expected {
        if auxiliary(name) {
            continue;
        }
        let stem = name.strip_suffix(".weight");
        // Census auxiliary order: weight_scale, weight_scale_inv, weight_scale_2.
        let auxiliaries: Vec<String> = stem
            .map(|stem| {
                ["weight_scale", "weight_scale_inv", "weight_scale_2"]
                    .iter()
                    .map(|suffix| format!("{stem}.{suffix}"))
                    .filter(|aux| expected.contains_key(aux))
                    .collect()
            })
            .unwrap_or_default();
        let mut shape = spec.shape.clone();
        let quant = match (spec.dtype, auxiliaries.len()) {
            ("BF16", 0) => QuantConstraint::ExactFloat(FloatType::Bf16),
            ("F32", 0) => QuantConstraint::ExactFloat(FloatType::F32),
            ("I32", 0) => QuantConstraint::I32,
            ("U8", 2) => {
                // e2m1 pairs: the requirement carries the logical shape.
                *shape.last_mut().expect("NVFP4 weight rank") *= 2;
                QuantConstraint::Nvfp4
            }
            ("F8_E4M3", 1) if auxiliaries[0].ends_with(".weight_scale_inv") => {
                QuantConstraint::Fp8Block128
            }
            ("F8_E4M3", 1) if auxiliaries[0].ends_with(".weight_scale") => {
                QuantConstraint::Fp8RowF32
            }
            _ => {
                return Err(TensorContractError::UnsupportedPlanOperation {
                    operation: "unknown hy_v4_flash dtype or scale pairing",
                });
            }
        };
        requirements.push(TensorRequirement {
            id: semantic_tensor_id(name),
            names: vec![name.clone()],
            match_mode: TensorMatch::OneOf,
            shape,
            owner: tensor_owner(name),
            transform: TensorTransform::Identity,
            quant,
            auxiliaries: Some(auxiliaries),
            required: true,
        });
    }
    Ok(TensorContract {
        dialect,
        requirements,
    })
}

fn tensor_owner(name: &str) -> TensorOwner {
    let mut parts = name.split('.');
    match (parts.next(), parts.next(), parts.next()) {
        (Some("model"), Some("layers"), Some(index)) => index
            .parse()
            .map(TensorOwner::Layer)
            .unwrap_or(TensorOwner::Global),
        (Some("model"), Some("dfly_layers"), Some(index)) => index
            .parse()
            .map(TensorOwner::Mtp)
            .unwrap_or(TensorOwner::Global),
        (Some("vit"), Some("layers"), Some(layer)) => TensorOwner::Vision(layer.parse().ok()),
        (Some("vit"), _, _) => TensorOwner::Vision(None),
        _ => TensorOwner::Global,
    }
}

/// Typed ids where an existing role is the same tensor role; family keys for Hy4-flash-only
/// tensors (output gate, post-projection norm, the combined `kv_b_proj`, the indexer, the latent
/// projections, Engram, DFly2 and the tower).
pub fn semantic_tensor_id(name: &str) -> TensorId {
    match name {
        "model.embed_tokens.weight" => return TensorId::TokenEmbedding,
        "lm_head.weight" => return TensorId::OutputProjection,
        "model.norm.weight" => return TensorId::OutputNorm,
        "model.hc_head.hc_head_fn" => return TensorId::HyperHeadFunction,
        "model.hc_head.hc_head_base" => return TensorId::HyperHeadBase,
        "model.hc_head.hc_head_scale" => return TensorId::HyperHeadScale,
        _ => {}
    }
    let family = || TensorId::Family {
        family: FAMILY,
        key: name.strip_prefix("model.").unwrap_or(name).to_string(),
    };
    let Some((index, suffix)) = name
        .strip_prefix("model.layers.")
        .and_then(|rest| rest.split_once('.'))
        .and_then(|(index, suffix)| Some((index.parse::<u32>().ok()?, suffix)))
    else {
        return family();
    };
    match layer_tensor_for_suffix(suffix) {
        Some(tensor) => TensorId::Layer { index, tensor },
        None => family(),
    }
}

fn layer_tensor_for_suffix(suffix: &str) -> Option<LayerTensor> {
    Some(match suffix {
        "input_layernorm.weight" => LayerTensor::PreAttentionNorm,
        "post_attention_layernorm.weight" => LayerTensor::PreMlpNorm,
        "hc_attn_layer.hc_pre.hc_fn" => LayerTensor::HyperAttentionFunction,
        "hc_attn_layer.hc_pre.hc_base" => LayerTensor::HyperAttentionBase,
        "hc_attn_layer.hc_pre.hc_scale" => LayerTensor::HyperAttentionScale,
        "hc_mlp_layer.hc_pre.hc_fn" => LayerTensor::HyperMlpFunction,
        "hc_mlp_layer.hc_pre.hc_base" => LayerTensor::HyperMlpBase,
        "hc_mlp_layer.hc_pre.hc_scale" => LayerTensor::HyperMlpScale,
        "self_attn.q_a_proj.weight" => LayerTensor::MlaQueryDown,
        "self_attn.q_a_layernorm.weight" => LayerTensor::MlaQueryDownNorm,
        "self_attn.q_b_proj.weight" => LayerTensor::MlaQueryUp,
        "self_attn.kv_a_proj_with_mqa.weight" => LayerTensor::MlaKvDown,
        "self_attn.kv_a_layernorm.weight" => LayerTensor::MlaKvDownNorm,
        "self_attn.o_proj.weight" => LayerTensor::MlaOutput,
        "self_attn.learnable_sink_param" => LayerTensor::AttentionSink,
        "mlp.gate_proj.weight" => LayerTensor::MlpGate,
        "mlp.up_proj.weight" => LayerTensor::MlpUp,
        "mlp.down_proj.weight" => LayerTensor::MlpDown,
        "mlp.post_fc2_norm.weight" | "mlp.combine_output_norm.weight" => LayerTensor::PostMlpNorm,
        "mlp.gate.weight" => LayerTensor::MoeRouter,
        "mlp.gate.e_score_correction_bias" => LayerTensor::MoeRouterBias,
        "mlp.experts.gate_proj.weight" => LayerTensor::MoeExpertGateBank,
        "mlp.experts.up_proj.weight" => LayerTensor::MoeExpertUpBank,
        "mlp.experts.down_proj.weight" => LayerTensor::MoeExpertDownBank,
        "mlp.shared_experts.gate_proj.weight" => LayerTensor::SharedMlpGate,
        "mlp.shared_experts.up_proj.weight" => LayerTensor::SharedMlpUp,
        "mlp.shared_experts.down_proj.weight" => LayerTensor::SharedMlpDown,
        _ => return None,
    })
}

#[cfg(test)]
mod tests;
