//! DeepSeek-V4.1-Flash (`deepseek_v41`) pack: config normalization, the typed CSA2 plan and the
//! config-derived tensor schema. The reference executor (`memra-reference`) runs the plan
//! (`NativeReference`, tiny gate only); no tuned surface implements it.
//!
//! Checkpoint namespaces (1017 artifact, 96,085 header tensors, names already in the vendor
//! reference spelling):
//!   embed / head / norm                 token embedding, separate output head, final norm
//!   layers.N.*                          40 trunk blocks (20 causal-encoder + 20 decoder layers)
//!   mtp.K.*                             3 DSpark blocks; their embed and head are tied to the trunk
//!   vision.* / aligner.* / image_*      DeepSeek-ViT tower, 3x3 aligner and image-span embeddings
//! The text plan does not execute the tower, the aligner, the image-span embeddings or the
//! image-token routing bias (`ffn.gate.bias_vl`); the contract still owns every one of them, so a
//! missing, extra or reshaped tower tensor fails the census like a trunk tensor.
//!
//! Quant layouts (vendor `Linear`): dense weights are FP8 e4m3 with an e8m0 `.scale` per 32x32
//! block, routed experts are e2m1 pairs packed in I8 with an e8m0 `.scale` per 32 inputs, Engram
//! tables are FP8 rows with an e8m0 scale per 32 elements. Embedding, head, norms, router weight,
//! compressor, index-key projection, `weights_proj` and the tower are BF16; hyper-connection
//! parameters, sinks and router biases are F32.

use std::collections::BTreeMap;

use super::*;
use crate::config::{Csa2LayerMode, DeepSeekV41Config};
use crate::dsv4::TensorSpec;
use crate::model_plan::{
    ActivationPlan, AttentionPlan, BlockQuant, Csa2AttentionPlan, Csa2Candidates,
    Csa2CompressedState, Csa2CompressorPlan, Csa2IndexerPlan, Csa2Mode, Csa2Pooling, Csa2StatePlan,
    DeepSeekV41Numerics, DraftSourcePlan, DrafterPlan, DsparkPlan, EngramPlan, LayerPlan, MlpPlan,
    MoeMlpPlan, NormKind, NormPlan, ResidualTopology, RopeFactors, RopePlan, RouterPlan,
    SharedMlpPlan, SinglePassHcPlan, StatePlan, TensorPresence, WeightTransform,
};
use crate::tensor_contract::{
    DsparkTensor, ExpertTensor, FloatType, LayerTensor, QuantConstraint, TensorId, TensorMatch,
    TensorOwner, TensorRequirement, TensorTransform,
};

const FAMILY: &str = "deepseek_v41";

pub static PACK: ModelPack = ModelPack {
    family: "deepseek_v41",
    output_head: OutputHeadContract::SeparateHead,
    tensor_consumption: TensorConsumption::Report,
    aliases: &["deepseek_v41", "deepseek-v41", "deepseek_v41_text"],
    config_layout: ConfigLayout::FlatOrTextConfig,
    tokenizer_sources: &[TokenizerSource::TokenizerJson],
    // The 1017 artifact ships its dialect as code (`encoding/encoding.py`, "DeepSeek-V4.1 Text and
    // Vision Encoding": DSML tag names, numeric reasoning effort, mid-conversation system turns,
    // image blocks) and no template. It is not the DSv4 renderer: memra-tokenizer's
    // `chat::EngineDialect::Dsv41`, gated against that file in
    // research/modelplan-onboarding-dsv41-20261109/encoding/.
    template: TemplateContract::EngineRenderer("encoding_dsv41"),
    // The reference executor runs the plan and the tiny gate passes against an oracle the
    // vendor model.py produced (record `deepseek_v41-reference`). Checkpoint, rewrite and serve
    // gates are pending.
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
    // The reference and the vendor model.py (with correctly rounded primitives on both sides)
    // agree to the bit on the tiny fixture: 6,549 oracle rows, max_abs 0
    // (research/modelplan-onboarding-dsv41-20261109/tiny/). The bound only absorbs a last-ulp
    // libm difference between machines; anything that moves a bf16, FP8 or FP4 rounding or a
    // top-k selection moves later rows far past it.
    checkpoint_parity: Some(CheckpointParityGate {
        max_abs: 1e-5,
        max_rel: 1e-5,
        require_argmax: true,
    }),
    matches_config: |config| matches!(config.arch, Arch::DeepSeekV41) && config.dsv41.is_some(),
    plan_builder,
    tensor_schema,
    tiny_plan: Some(tiny_plan),
};

/// The tiny-gate config: every structural feature of the 1017 program at hidden 64. Ten trunk
/// layers (window-only, two ratio-2 Full groups with a Reuse layer each, a ratio-1 Full layer
/// that produces the candidate pool, two restricted Reindex layers and Reuse layers after each),
/// Engram at two layers with small prime tables, single-pass mHC over 4 streams, FP8 32x32
/// dense weights, FP4 experts (top-2 of 6 renormalized; the DSpark blocks top-1 of 4, which the
/// vendor does not renormalize) and two DSpark blocks. Window 64 with 6 compressed picks makes
/// every query cross the 64-entry block of the sparse-attention kernel.
pub const TINY_CONFIG: &str = include_str!("fixtures/tiny-config.json");

fn tiny_plan() -> Result<ModelPlan, PlanCompileError> {
    plan_builder(&ModelConfig::from_hf(&crate::config::HfConfig::parse(
        TINY_CONFIG,
    )))
}

/// [`TINY_CONFIG`] at the 1017 window, 128 (and room for its longer token stream): the window
/// spans two 64-entry blocks of the sparse-attention kernel, which the main tiny config's window
/// of 64 never does. The tiny gate runs it as a variant pinned to its own vendor oracle.
pub const TINY_WINDOW128_CONFIG: &str = include_str!("fixtures/tiny-config-window128.json");

pub fn tiny_window128_plan() -> Result<ModelPlan, PlanCompileError> {
    plan_builder(&ModelConfig::from_hf(&crate::config::HfConfig::parse(
        TINY_WINDOW128_CONFIG,
    )))
}

fn dsv41(config: &ModelConfig) -> Result<&DeepSeekV41Config, PlanCompileError> {
    config
        .dsv41
        .as_ref()
        .ok_or_else(|| PlanCompileError::UnsupportedSemantics {
            field: "deepseek_v41",
            value: "config carries no deepseek_v41 block".into(),
        })
}

fn plan_builder(config: &ModelConfig) -> Result<ModelPlan, PlanCompileError> {
    config.validate_plan_semantics()?;
    let v41 = dsv41(config)?;
    let trunk = v41.trunk_layers();
    if config.nextn_predict_layers != v41.num_nextn_predict_layers
        || config.n_layer != trunk + v41.num_nextn_predict_layers
    {
        return Err(PlanCompileError::UnsupportedSemantics {
            field: "num_nextn_predict_layers",
            value: format!(
                "{} drafter blocks but n_layer {} for {trunk} trunk layers",
                config.nextn_predict_layers, config.n_layer
            ),
        });
    }
    if trunk == 0 {
        return Err(PlanCompileError::EmptyModel);
    }
    let norm = NormPlan {
        kind: NormKind::Rms,
        epsilon: config.rms_eps,
        weight_transform: WeightTransform::Identity,
    };
    let layers = (0..trunk)
        .map(|il| layer_plan(config, v41, il, norm))
        .collect();
    let blocks = (trunk..trunk + v41.num_nextn_predict_layers)
        .map(|il| layer_plan(config, v41, il, norm))
        .collect();
    Ok(ModelPlan {
        speech: None,
        arch: config.arch.clone(),
        prefill_activation: None,
        hidden_size: config.n_embd,
        vocab_size: config.n_vocab,
        context_length: config.context_length,
        embedding_scale: 1.0,
        // The DeepSeek-ViT tower, aligner and image-span splice are contract-owned but not part
        // of the text plan (the qwen4_exp convention).
        vision: None,
        mimo_audio_patch: None,
        multimodal: None,
        per_layer_input: None,
        layers,
        output_norm: norm,
        exit_mixer: None,
        logits: Vec::new(),
        mtp_blocks: Vec::new(),
        drafter: Some(DrafterPlan::Dspark(DsparkPlan {
            block_size: v41.dspark_block_size,
            noise_token_id: v41.dspark_noise_token_id,
            target_layer_ids: v41.dspark_target_layer_ids.clone(),
            markov_rank: v41.dspark_markov_rank,
            blocks,
        })),
        draft_source: DraftSourcePlan::Embedded,
        sampling_defaults: None,
        partition_boundaries: (1..trunk as usize).collect(),
        deepseek_v41: Some(DeepSeekV41Numerics {
            dense_weight_block: v41.quant_weight_block_size[0],
            expert_weight: BlockQuant::FP4_32_UE8M0,
            linear_activation: BlockQuant::FP8_32_UE8M0,
            routed_weight_before_down: true,
            output_lora_a_dequantized_bf16: true,
        }),
    })
}

fn layer_plan(config: &ModelConfig, v41: &DeepSeekV41Config, il: u32, norm: NormPlan) -> LayerPlan {
    let drafter = il >= v41.trunk_layers();
    let ratio = v41.compress_ratio(il);
    // Vendor `Attention.__init__`: compressed layers rotate at compress_rope_theta with YaRN,
    // window-only layers (DSpark blocks included) at rope_theta with none.
    let rope = RopePlan {
        dimensions: v41.qk_rope_head_dim,
        base: if ratio > 0 {
            v41.compress_rope_theta
        } else {
            config.rope_freq_base
        },
        factors: if ratio > 0 {
            RopeFactors::Yarn {
                factor: v41.rope_yarn_factor,
                original_context: v41.rope_yarn_orig_ctx,
                beta_fast: v41.rope_yarn_beta_fast,
                beta_slow: v41.rope_yarn_beta_slow,
            }
        } else {
            RopeFactors::None
        },
    };
    let indexer = Csa2IndexerPlan {
        heads: v41.index_n_heads,
        head_dim: v41.index_head_dim,
        top_k: v41.index_topk,
        query_quant: BlockQuant::FP4_32_UE8M0,
        key_quant: BlockQuant::FP4_32_UE8M0,
        // Vendor `Indexer`: the source produces the pool, every later indexer is restricted.
        candidates: if il == v41.candidate_source_layer_id {
            Csa2Candidates::Produce {
                top_blocks: v41.candidate_topk_blocks,
                block_size: v41.candidate_block_size,
            }
        } else if il > v41.candidate_source_layer_id {
            Csa2Candidates::Restrict {
                source_layer: v41.candidate_source_layer_id,
            }
        } else {
            Csa2Candidates::None
        },
    };
    let (mode, compressed) = match v41.layer_mode(il) {
        Csa2LayerMode::Window => (Csa2Mode::Window, Csa2CompressedState::None),
        Csa2LayerMode::Full => (
            Csa2Mode::Full {
                ratio,
                compressor: Csa2CompressorPlan {
                    ratio,
                    pooling: if ratio > 1 {
                        Csa2Pooling::SoftmaxGated
                    } else {
                        Csa2Pooling::Projection
                    },
                    latent_quant: BlockQuant::FP4_16_E4M3,
                },
                indexer,
            },
            Csa2CompressedState::Own {
                ratio,
                index_head_dim: v41.index_head_dim,
                partial_group: ratio > 1,
            },
        ),
        Csa2LayerMode::Reindex { kv_source } => (
            Csa2Mode::Reindex {
                ratio,
                kv_source,
                indexer,
            },
            Csa2CompressedState::Shared {
                ratio,
                source_layer: kv_source,
            },
        ),
        Csa2LayerMode::Reuse {
            kv_source,
            index_source,
        } => (
            Csa2Mode::Reuse {
                ratio,
                kv_source,
                index_source,
            },
            Csa2CompressedState::Shared {
                ratio,
                source_layer: kv_source,
            },
        ),
    };
    let (experts, top_k) = if drafter {
        (v41.dspark_n_routed_experts, v41.dspark_num_experts_per_tok)
    } else {
        (v41.n_routed_experts, v41.num_experts_per_tok)
    };
    let engram = v41
        .engram_slot(il)
        .filter(|_| !drafter)
        .map(|slot| EngramPlan {
            table_index: slot as u32,
            table_rows: v41.engram_num_embeddings[slot].into(),
            head_dim: v41.engram_head_dim,
            heads: v41.engram_n_heads,
            max_ngram: v41.engram_max_ngram_size,
            hash_columns: v41.engram_hash_columns(),
            bucket_base: v41.engram_vocab_size,
            compressed_vocab_size: v41.engram_compressed_vocab_size,
            pad_token_id: v41.engram_pad_token_id,
            table_quant: BlockQuant::FP8_32_UE8M0,
            gate_clamp: 1e-6,
        });
    LayerPlan {
        index: il,
        pre_attention_norm: norm,
        attention: AttentionPlan::Csa2(Csa2AttentionPlan {
            query_heads: config.n_head,
            q_lora_rank: v41.q_lora_rank,
            head_dim: v41.head_dim,
            rope_head_dim: v41.qk_rope_head_dim,
            output_lora_rank: v41.o_lora_rank,
            output_groups: v41.o_groups,
            window: v41.sliding_window,
            rope,
            sink: TensorPresence::Required,
            mode,
            window_kv_quant: BlockQuant::FP8_32_UE8M0,
            output_quant: BlockQuant::FP8_32_UE8M0,
        }),
        pre_mlp_norm: norm,
        mlp: MlpPlan::Moe(MoeMlpPlan {
            expert_count: experts,
            experts_per_token: top_k,
            expert_intermediate_size: v41.moe_intermediate_size,
            // Vendor `Gate`: sqrt(softplus) scores, the correction bias steers selection only,
            // the selected raw scores are renormalized when top-k > 1, then scaled.
            router: RouterPlan::SqrtSoftplus {
                normalize_selected: v41.norm_topk_prob && top_k > 1,
                scaling_factor: v41.routed_scaling_factor,
                selection_bias: true,
            },
            shared: Some(SharedMlpPlan {
                intermediate_size: v41.n_shared_experts * v41.moe_intermediate_size,
                gated: false,
            }),
            // Vendor `Expert`: up clamped to +-limit, gate clamped from above, then silu.
            activation: ActivationPlan::SwiGluPreClamped {
                limit: v41.swiglu_limit,
            },
        }),
        residual: ResidualTopology::SinglePassHyperConnections(SinglePassHcPlan {
            streams: v41.hc_mult,
            epsilon: v41.hc_eps,
            sinkhorn_iterations: v41.hc_sinkhorn_iters,
            mix_norm_epsilon: config.rms_eps,
            engram,
        }),
        state: StatePlan::Csa2(Csa2StatePlan {
            window: v41.sliding_window,
            head_dim: v41.head_dim,
            compressed,
        }),
        sparse_overlay: None,
        ple: None,
    }
}

/// The complete expected header census, derived from config math. Shapes are safetensors
/// row-major `[out, in]`; packed I8 experts carry `in / 2` bytes per row.
pub fn expected_census(config: &ModelConfig) -> BTreeMap<String, TensorSpec> {
    let v41 = config
        .dsv41
        .as_ref()
        .expect("deepseek_v41 census needs a deepseek_v41 config");
    let hidden = u64::from(config.n_embd);
    let vocab = u64::from(config.n_vocab);
    let heads = u64::from(config.n_head);
    let head_dim = u64::from(v41.head_dim);
    let q_lora = u64::from(v41.q_lora_rank);
    let o_groups = u64::from(v41.o_groups);
    let o_lora = u64::from(v41.o_lora_rank);
    let inter = u64::from(v41.moe_intermediate_size);
    let hc = u64::from(v41.hc_mult);
    let index_heads = u64::from(v41.index_n_heads);
    let index_dim = u64::from(v41.index_head_dim);
    let block = u64::from(v41.quant_weight_block_size[0]);
    let mut map = BTreeMap::new();
    let mut put = |name: String, dtype: &'static str, shape: Vec<u64>| {
        assert!(
            map.insert(name.clone(), TensorSpec { dtype, shape })
                .is_none(),
            "deepseek_v41 census derives {name} twice"
        );
    };
    let fp8 = |put: &mut dyn FnMut(String, &'static str, Vec<u64>),
               stem: String,
               out: u64,
               input: u64| {
        put(format!("{stem}.weight"), "F8_E4M3", vec![out, input]);
        put(
            format!("{stem}.scale"),
            "F8_E8M0",
            vec![out.div_ceil(block), input.div_ceil(block)],
        );
    };
    let fp4 = |put: &mut dyn FnMut(String, &'static str, Vec<u64>),
               stem: String,
               out: u64,
               input: u64| {
        put(format!("{stem}.weight"), "I8", vec![out, input / 2]);
        put(format!("{stem}.scale"), "F8_E8M0", vec![out, input / 32]);
    };

    put("embed.weight".into(), "BF16", vec![vocab, hidden]);
    put("head.weight".into(), "BF16", vec![vocab, hidden]);
    put("norm.weight".into(), "BF16", vec![hidden]);

    let trunk = v41.trunk_layers();
    for il in 0..trunk + v41.num_nextn_predict_layers {
        let drafter = il >= trunk;
        let p = if drafter {
            format!("mtp.{}.", il - trunk)
        } else {
            format!("layers.{il}.")
        };
        put(format!("{p}attn_norm.weight"), "BF16", vec![hidden]);
        put(format!("{p}ffn_norm.weight"), "BF16", vec![hidden]);
        let mix = (2 + hc) * hc;
        for sub in ["attn", "ffn"] {
            put(format!("{p}hc_{sub}_fn"), "F32", vec![mix, hc * hidden]);
            put(format!("{p}hc_{sub}_base"), "F32", vec![mix]);
            put(format!("{p}hc_{sub}_scale"), "F32", vec![3]);
        }
        // CSA2 attention core.
        put(format!("{p}attn.attn_sink"), "F32", vec![heads]);
        fp8(&mut put, format!("{p}attn.wq_a"), q_lora, hidden);
        put(format!("{p}attn.q_norm.weight"), "BF16", vec![q_lora]);
        fp8(&mut put, format!("{p}attn.wq_b"), heads * head_dim, q_lora);
        fp8(&mut put, format!("{p}attn.wkv"), head_dim, hidden);
        put(format!("{p}attn.kv_norm.weight"), "BF16", vec![head_dim]);
        fp8(
            &mut put,
            format!("{p}attn.wo_a"),
            o_groups * o_lora,
            heads * head_dim / o_groups,
        );
        fp8(&mut put, format!("{p}attn.wo_b"), hidden, o_groups * o_lora);
        if !drafter && v41.is_kv_source(il) {
            let ratio = v41.compress_ratio(il);
            put(
                format!("{p}attn.compressor.wkv.weight"),
                "BF16",
                vec![head_dim, hidden],
            );
            if ratio > 1 {
                put(
                    format!("{p}attn.compressor.wgate.weight"),
                    "BF16",
                    vec![head_dim, hidden],
                );
            }
            put(
                format!("{p}attn.compressor.norm.weight"),
                "BF16",
                vec![head_dim],
            );
            put(
                format!("{p}attn.indexer.wk.weight"),
                "BF16",
                vec![index_dim, head_dim],
            );
            put(
                format!("{p}attn.indexer.k_norm.weight"),
                "BF16",
                vec![index_dim],
            );
        }
        if !drafter && v41.is_index_source(il) {
            fp8(
                &mut put,
                format!("{p}attn.indexer.wq_b"),
                index_heads * index_dim,
                q_lora,
            );
            put(
                format!("{p}attn.indexer.weights_proj.weight"),
                "BF16",
                vec![index_heads, hidden],
            );
        }
        // MoE: router, image-token router bias (the tower exists), FP4 experts, FP8 shared.
        let experts = u64::from(if drafter {
            v41.dspark_n_routed_experts
        } else {
            v41.n_routed_experts
        });
        put(format!("{p}ffn.gate.weight"), "BF16", vec![experts, hidden]);
        put(format!("{p}ffn.gate.bias"), "F32", vec![experts]);
        put(format!("{p}ffn.gate.bias_vl"), "F32", vec![experts]);
        for expert in 0..experts {
            let e = format!("{p}ffn.experts.{expert}.");
            fp4(&mut put, format!("{e}w1"), inter, hidden);
            fp4(&mut put, format!("{e}w2"), hidden, inter);
            fp4(&mut put, format!("{e}w3"), inter, hidden);
        }
        let shared = u64::from(v41.n_shared_experts) * inter;
        fp8(
            &mut put,
            format!("{p}ffn.shared_experts.w1"),
            shared,
            hidden,
        );
        fp8(
            &mut put,
            format!("{p}ffn.shared_experts.w2"),
            hidden,
            shared,
        );
        fp8(
            &mut put,
            format!("{p}ffn.shared_experts.w3"),
            shared,
            hidden,
        );
        // Engram.
        if let Some(slot) = v41.engram_slot(il).filter(|_| !drafter) {
            let rows = u64::from(v41.engram_num_embeddings[slot]);
            let dim = u64::from(v41.engram_head_dim);
            put(
                format!("{p}engram.embed.weight"),
                "F8_E4M3",
                vec![rows, dim],
            );
            put(
                format!("{p}engram.embed.scale"),
                "F8_E8M0",
                vec![rows, dim / 32],
            );
            fp8(
                &mut put,
                format!("{p}engram.wkv"),
                hidden * (hc + 1),
                u64::from(v41.engram_hash_columns()) * dim,
            );
            put(format!("{p}engram.q_weight"), "BF16", vec![hc, hidden]);
            put(format!("{p}engram.k_weight"), "BF16", vec![hc, hidden]);
        }
        // DSpark glue: fusion on the first block, the Markov and confidence heads on the last.
        if drafter && il == trunk {
            let targets = v41.dspark_target_layer_ids.len() as u64;
            fp8(&mut put, format!("{p}main_proj"), hidden, targets * hidden);
            put(format!("{p}main_norm.weight"), "BF16", vec![hidden]);
        }
        if drafter && il == trunk + v41.num_nextn_predict_layers - 1 {
            let rank = u64::from(v41.dspark_markov_rank);
            put(format!("{p}norm.weight"), "BF16", vec![hidden]);
            put(
                format!("{p}markov_head.embed.weight"),
                "BF16",
                vec![vocab, rank],
            );
            put(
                format!("{p}markov_head.head.weight"),
                "BF16",
                vec![vocab, rank],
            );
            put(
                format!("{p}confidence_head.proj.weight"),
                "BF16",
                vec![1, hidden + rank],
            );
        }
    }

    // DeepSeek-ViT, aligner and image-span embeddings (contract-owned, not in the text plan).
    let vision = &v41.vision;
    let vd = u64::from(vision.hidden_size);
    let vi = u64::from(vision.intermediate_size);
    let patch = u64::from(vision.patch_size);
    let ratio = u64::from(vision.downsample_ratio);
    put(
        "vision.patch_embed.proj.weight".into(),
        "BF16",
        vec![vd, 3 * patch * patch],
    );
    put("vision.patch_embed.proj.bias".into(), "BF16", vec![vd]);
    for b in 0..vision.num_hidden_layers {
        let p = format!("vision.blocks.{b}.");
        put(format!("{p}attn.wqkv.weight"), "BF16", vec![3 * vd, vd]);
        put(format!("{p}attn.wqkv.bias"), "BF16", vec![3 * vd]);
        put(format!("{p}attn.wo.weight"), "BF16", vec![vd, vd]);
        put(format!("{p}attn.wo.bias"), "BF16", vec![vd]);
        put(format!("{p}mlp.w1.weight"), "BF16", vec![2 * vi, vd]);
        put(format!("{p}mlp.w2.weight"), "BF16", vec![vd, vi]);
        put(format!("{p}norm1.weight"), "BF16", vec![vd]);
        put(format!("{p}norm2.weight"), "BF16", vec![vd]);
    }
    put("vision.norm.weight".into(), "BF16", vec![vd]);
    put(
        "aligner.w1.weight".into(),
        "BF16",
        vec![hidden, vd * ratio * ratio],
    );
    put("aligner.w1.bias".into(), "BF16", vec![hidden]);
    put("aligner.w2.weight".into(), "BF16", vec![hidden, hidden]);
    put("aligner.w2.bias".into(), "BF16", vec![hidden]);
    for name in ["image_start", "image_end", "image_newline"] {
        put(name.into(), "BF16", vec![hidden]);
    }
    map
}

#[allow(clippy::result_large_err)] // allow: the fat error type is the diagnostic contract here; boxing it would change the error surface
fn tensor_schema(
    config: &ModelConfig,
    plan: &ModelPlan,
    dialect: CheckpointDialect,
    _options: ContractOptions,
) -> Result<TensorContract, TensorContractError> {
    if dialect != CheckpointDialect::HfSafetensors {
        return Err(TensorContractError::UnsupportedPlanOperation {
            operation: "deepseek_v41 non-safetensors schema",
        });
    }
    let Some(DrafterPlan::Dspark(dspark)) = plan.drafter.as_ref() else {
        return Err(TensorContractError::UnsupportedPlanOperation {
            operation: "deepseek_v41 schema without its DSpark plan",
        });
    };
    let expected = expected_census(config);
    let mut requirements = Vec::new();
    for (name, spec) in &expected {
        if spec.dtype == "F8_E8M0" {
            // Every e8m0 plane is the `.scale` of a weight in the same map.
            let primary = name
                .strip_suffix(".scale")
                .map(|stem| format!("{stem}.weight"))
                .filter(|primary| expected.contains_key(primary));
            if primary.is_none() {
                return Err(TensorContractError::UnsupportedPlanOperation {
                    operation: "orphan deepseek_v41 E8M0 scale",
                });
            }
            continue;
        }
        let scale = name
            .strip_suffix(".weight")
            .map(|stem| format!("{stem}.scale"))
            .filter(|scale| expected.contains_key(scale));
        let mut shape = spec.shape.clone();
        let quant = match (spec.dtype, scale.is_some()) {
            ("BF16", false) => QuantConstraint::ExactFloat(FloatType::Bf16),
            ("F32", false) => QuantConstraint::ExactFloat(FloatType::F32),
            ("I8", true) => {
                // e2m1 pairs: the requirement carries the logical [out, in].
                *shape.last_mut().expect("expert weight rank") *= 2;
                QuantConstraint::Mxfp4
            }
            ("F8_E4M3", true) if name.ends_with(".engram.embed.weight") => {
                QuantConstraint::Fp8Row32
            }
            ("F8_E4M3", true) => QuantConstraint::Fp8Block32,
            _ => {
                return Err(TensorContractError::UnsupportedPlanOperation {
                    operation: "unknown deepseek_v41 dtype or scale pairing",
                });
            }
        };
        requirements.push(TensorRequirement {
            id: semantic_tensor_id(name, dspark),
            names: vec![name.clone()],
            match_mode: TensorMatch::OneOf,
            shape,
            owner: tensor_owner(name),
            transform: TensorTransform::Identity,
            quant,
            auxiliaries: Some(scale.into_iter().collect()),
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
        (Some("layers"), Some(index), _) => index
            .parse()
            .map(TensorOwner::Layer)
            .unwrap_or(TensorOwner::Global),
        (Some("mtp"), Some(index), _) => index
            .parse()
            .map(TensorOwner::Mtp)
            .unwrap_or(TensorOwner::Global),
        (Some("vision"), Some("blocks"), Some(block)) => TensorOwner::Vision(block.parse().ok()),
        (Some("vision" | "aligner"), _, _) => TensorOwner::Vision(None),
        _ if name.starts_with("image_") => TensorOwner::Vision(None),
        _ => TensorOwner::Global,
    }
}

/// Typed ids where an existing role is the same tensor role; family keys for V4.1-only tensors
/// (index-key projection, Engram, the image-token router bias, the tower). The reference
/// executor reads checkpoint tensors through this same mapping.
pub fn semantic_tensor_id(name: &str, dspark: &DsparkPlan) -> TensorId {
    match name {
        "embed.weight" => return TensorId::TokenEmbedding,
        "head.weight" => return TensorId::OutputProjection,
        "norm.weight" => return TensorId::OutputNorm,
        _ => {}
    }
    let family = || TensorId::Family {
        family: FAMILY,
        key: name
            .strip_prefix("layers.")
            .map(|rest| format!("trunk.{rest}"))
            .unwrap_or_else(|| name.to_string()),
    };
    let layer_and_suffix = if let Some(rest) = name.strip_prefix("layers.") {
        rest.split_once('.')
            .and_then(|(index, suffix)| Some((index.parse::<u32>().ok()?, suffix)))
    } else if let Some(rest) = name.strip_prefix("mtp.") {
        let Some((stage, suffix)) = rest
            .split_once('.')
            .and_then(|(stage, suffix)| Some((stage.parse::<usize>().ok()?, suffix)))
        else {
            return family();
        };
        let last = dspark.blocks.len().saturating_sub(1);
        let special = match (stage, suffix) {
            (0, "main_proj.weight") => Some(DsparkTensor::MainProjection),
            (0, "main_norm.weight") => Some(DsparkTensor::MainNorm),
            (stage, "norm.weight") if stage == last => Some(DsparkTensor::OutputNorm),
            (stage, "markov_head.embed.weight") if stage == last => {
                Some(DsparkTensor::MarkovEmbedding)
            }
            (stage, "markov_head.head.weight") if stage == last => Some(DsparkTensor::MarkovOutput),
            (stage, "confidence_head.proj.weight") if stage == last => {
                Some(DsparkTensor::ConfidenceProjection)
            }
            _ => None,
        };
        if let Some(tensor) = special {
            return TensorId::Dspark(tensor);
        }
        dspark.blocks.get(stage).map(|block| (block.index, suffix))
    } else {
        None
    };
    let Some((index, suffix)) = layer_and_suffix else {
        return family();
    };
    if let Some((expert, tensor)) = expert_tensor_for_suffix(suffix) {
        return TensorId::Expert {
            layer: index,
            expert,
            tensor,
        };
    }
    match layer_tensor_for_suffix(suffix) {
        Some(tensor) => TensorId::Layer { index, tensor },
        None => family(),
    }
}

fn expert_tensor_for_suffix(suffix: &str) -> Option<(u32, ExpertTensor)> {
    let rest = suffix.strip_prefix("ffn.experts.")?;
    let (expert, projection) = rest.split_once('.')?;
    let tensor = match projection {
        "w1.weight" => ExpertTensor::Gate,
        "w2.weight" => ExpertTensor::Down,
        "w3.weight" => ExpertTensor::Up,
        _ => return None,
    };
    Some((expert.parse().ok()?, tensor))
}

fn layer_tensor_for_suffix(suffix: &str) -> Option<LayerTensor> {
    Some(match suffix {
        "attn_norm.weight" => LayerTensor::PreAttentionNorm,
        "ffn_norm.weight" => LayerTensor::PreMlpNorm,
        "attn.wq_a.weight" => LayerTensor::MlaQueryDown,
        "attn.q_norm.weight" => LayerTensor::MlaQueryDownNorm,
        "attn.wq_b.weight" => LayerTensor::MlaQueryUp,
        "attn.wkv.weight" => LayerTensor::MlaKvDown,
        "attn.kv_norm.weight" => LayerTensor::MlaKvDownNorm,
        "attn.wo_a.weight" => LayerTensor::MlaOutputDown,
        "attn.wo_b.weight" => LayerTensor::MlaOutput,
        "attn.attn_sink" => LayerTensor::AttentionSink,
        "attn.compressor.wkv.weight" => LayerTensor::KvCompressorKeyValue,
        "attn.compressor.wgate.weight" => LayerTensor::KvCompressorGate,
        "attn.compressor.norm.weight" => LayerTensor::KvCompressorNorm,
        "attn.indexer.wq_b.weight" => LayerTensor::SparseQuery,
        "attn.indexer.weights_proj.weight" => LayerTensor::SparseProjection,
        "hc_attn_base" => LayerTensor::HyperAttentionBase,
        "hc_attn_fn" => LayerTensor::HyperAttentionFunction,
        "hc_attn_scale" => LayerTensor::HyperAttentionScale,
        "hc_ffn_base" => LayerTensor::HyperMlpBase,
        "hc_ffn_fn" => LayerTensor::HyperMlpFunction,
        "hc_ffn_scale" => LayerTensor::HyperMlpScale,
        "ffn.gate.weight" => LayerTensor::MoeRouter,
        "ffn.gate.bias" => LayerTensor::MoeRouterBias,
        "ffn.shared_experts.w1.weight" => LayerTensor::SharedMlpGate,
        "ffn.shared_experts.w2.weight" => LayerTensor::SharedMlpDown,
        "ffn.shared_experts.w3.weight" => LayerTensor::SharedMlpUp,
        _ => return None,
    })
}

#[cfg(test)]
mod tests;
