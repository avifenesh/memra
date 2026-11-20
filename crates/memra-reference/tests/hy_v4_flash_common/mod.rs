//! Shared helpers of the Hy4-flash reference gates: the oracle record of a float64 run of the
//! pinned transformers code, the tiny config as the pack's config type, and the weights in the
//! checkpoint's roles.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::Path;

use memra_gguf::config::{
    HyV4DflyConfig, HyV4FlashConfig, HyV4HeadGeometry, HyV4LayerKind, HyV4VisionConfig,
};
use memra_gguf::model_packs::hy_v4_flash::layer_plan;
use memra_gguf::model_plan::{LayerPlan, NormKind, NormPlan, WeightTransform};
use memra_gguf::modelopt_quant::LayerQuantMap;
use memra_reference::hy_v4_flash::{
    AttentionWeights, HyperConnectionWeights, IndexerWeights, LatentMoeWeights, LayerWeights,
    MlpWeights,
};
use serde_json::Value;

/// The tiny config's indexer geometry.
#[derive(Clone, Copy)]
pub struct Index {
    pub heads: u32,
    pub head_dim: u32,
    pub top_k: u32,
    pub pool: u32,
}

/// modules/gen_modules.py: 7 tokens, under the budget of 8.
pub const MODULES: Index = Index {
    heads: 2,
    head_dim: 8,
    top_k: 8,
    pool: 4,
};

/// indexpool/gen_indexpool.py: 41 tokens, over the budget of 12 (two groups plus the query's).
pub const INDEXPOOL: Index = Index {
    heads: 2,
    head_dim: 12,
    top_k: 12,
    pool: 4,
};

pub const BOUND: f64 = 1e-12;
/// The tiny config's epsilons, 2^-17 and 2^-20: exact in f32, which the typed plan carries.
pub const RMS_EPS: f32 = 1.0 / 131_072.0;
pub const HC_EPS: f32 = 1.0 / 1_048_576.0;

pub struct Oracle {
    pub record: Value,
    pub weights: BTreeMap<String, Vec<f64>>,
}

pub fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// The oracle record at `fixture`, relative to research/modelplan-onboarding-hy4flash-20261119/.
pub fn oracle(fixture: &str) -> Oracle {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../research/modelplan-onboarding-hy4flash-20261119")
        .join(fixture);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let record: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        record["transformers_modeling_blob"],
        "2d796211a4f867bd4c46d4a952cfee0375929f72"
    );
    assert_eq!(record["dtype"], "float64");
    let weights = record["weights"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), floats(&v["data"])))
        .collect();
    Oracle { record, weights }
}

impl Oracle {
    pub fn w(&self, name: &str) -> &[f64] {
        self.weights
            .get(name)
            .unwrap_or_else(|| panic!("no weight {name}"))
    }
    pub fn module(&self, name: &str) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let m = &self.record["modules"][name];
        let all = |k: &str| m[k].as_array().unwrap().iter().map(floats).collect();
        (all("in"), all("out"))
    }
    pub fn positions(&self) -> Vec<u32> {
        self.record["positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_u64().unwrap() as u32)
            .collect()
    }
}

/// The tiny config of the oracle scripts as the pack's config type (no quant map: BF16
/// everywhere). The indexer geometry is the fixture's.
pub fn config(index: Index) -> HyV4FlashConfig {
    let geometry = |heads, rope_theta| HyV4HeadGeometry {
        heads,
        qk_nope_head_dim: 6,
        qk_rope_head_dim: 8,
        v_head_dim: 5,
        rope_theta,
    };
    HyV4FlashConfig {
        num_hidden_layers: 4,
        hidden_size: 16,
        vocab_size: 64,
        rms_norm_eps: RMS_EPS,
        max_position_embeddings: 64,
        layer_kinds: vec![
            HyV4LayerKind::Indexed,
            HyV4LayerKind::Indexed,
            HyV4LayerKind::Sliding,
            HyV4LayerKind::Sliding,
        ],
        dense_layers: vec![true, false, false, false],
        indexed: geometry(2, 1e7),
        sliding: geometry(4, 1e4),
        sliding_window: 3,
        q_lora_rank: 12,
        kv_lora_rank: 8,
        index_n_heads: index.heads,
        index_head_dim: index.head_dim,
        index_topk: index.top_k,
        index_pool: index.pool,
        hc_mult: 4,
        hc_eps: HC_EPS,
        hc_magnitude: 2.0,
        intermediate_size: 24,
        n_routed_experts: 6,
        num_experts_per_tok: 2,
        moe_intermediate_size: 6,
        routed_expert_hidden_size: 8,
        n_shared_experts: 1,
        routed_scaling_factor: 1.75,
        norm_topk_prob: true,
        swiglu_limit: 1.5,
        engram_layer_ids: Vec::new(),
        engram_max_ngram_size: 3,
        engram_n_head_per_ngram: 8,
        engram_n_embed_per_ngram: 768,
        engram_head_vocab_sizes: Vec::new(),
        engram_pad_id: 0,
        engram_seed: 0,
        engram_fp8_embedding: true,
        engram_lookup_table_size: 1,
        tie_word_embeddings: false,
        quant: LayerQuantMap {
            layers: BTreeMap::new(),
            excluded: Vec::new(),
        },
        vision: HyV4VisionConfig {
            num_hidden_layers: 0,
            hidden_size: 0,
            intermediate_size: 0,
            num_channels: 0,
            patch_size: 0,
            max_vit_seq_len: 0,
            spatial_merge_size: 0,
            out_hidden_size: 16,
        },
        dfly: HyV4DflyConfig {
            num_hidden_layers: 0,
            mask_token_id: 0,
            target_kv_layers: Vec::new(),
            target_index_layers: Vec::new(),
            hidden_correction: false,
        },
    }
}

pub fn numerics() -> memra_gguf::model_plan::HyV4FlashNumerics {
    memra_gguf::model_plan::HyV4FlashNumerics {
        streams: 4,
        exit_epsilon: HC_EPS,
        output_projection_f32: true,
    }
}

pub fn head_weights(o: &Oracle) -> HyperConnectionWeights<'_> {
    HyperConnectionWeights {
        function: o.w("hc_head.hc_fn"),
        base: o.w("hc_head.hc_base"),
        scale: o.w("hc_head.hc_scale"),
    }
}

pub fn plan(index: Index, il: u32) -> LayerPlan {
    let hy = config(index);
    let norm = NormPlan {
        kind: NormKind::Rms,
        epsilon: hy.rms_norm_eps,
        weight_transform: WeightTransform::Identity,
    };
    layer_plan(&hy, il, norm).unwrap()
}

/// The transformers module's tensors in the checkpoint's roles. The routed banks are split
/// from the module's fused `gate_up_proj` (rows `[0, I)` gate, `[I, 2I)` up), as the artifact
/// stores them.
pub struct Owned {
    pub gate_bank: Vec<f64>,
    pub up_bank: Vec<f64>,
}

pub fn split_gate_up(o: &Oracle, il: u32) -> Owned {
    let fused = o.w(&format!("layers.{il}.mlp.experts.gate_up_proj"));
    let (experts, inter, latent) = (6, 6, 8);
    let mut gate_bank = Vec::new();
    let mut up_bank = Vec::new();
    for e in 0..experts {
        let base = e * 2 * inter * latent;
        gate_bank.extend_from_slice(&fused[base..base + inter * latent]);
        up_bank.extend_from_slice(&fused[base + inter * latent..base + 2 * inter * latent]);
    }
    Owned { gate_bank, up_bank }
}

/// The indexer's weights on an indexed layer (the tiny configs index layers 0 and 1).
pub fn indexer_weights<'a>(o: &'a Oracle, il: u32) -> Option<IndexerWeights<'a>> {
    let i = |n: &str| o.w(&format!("layers.{il}.self_attn.compressed_indexer.{n}"));
    (il < 2).then(|| IndexerWeights {
        query: i("query_projection.weight"),
        head_weight: i("head_weight_projection.weight"),
        token_key: i("token_key_projection.weight"),
        token_key_norm: i("token_key_norm.weight"),
        key_value: i("key_value_projection.weight"),
        key_gate: i("key_gate_projection.weight"),
        ape: i("intra_group_ape"),
        key_norm: i("key_norm.weight"),
    })
}

pub fn attention_weights<'a>(o: &'a Oracle, il: u32) -> AttentionWeights<'a> {
    let a = |n: &str| o.w(&format!("layers.{il}.self_attn.{n}"));
    AttentionWeights {
        q_a: a("q_a_proj.weight"),
        q_a_norm: a("q_a_layernorm.weight"),
        q_b: a("q_b_proj.weight"),
        kv_a: a("kv_a_proj_with_mqa.weight"),
        kv_a_norm: a("kv_a_layernorm.weight"),
        kv_b: a("kv_b_proj.weight"),
        gate: a("gate_proj.weight"),
        o: a("o_proj.weight"),
        post_norm: a("post_proj_norm.weight"),
        sinks: a("sinks"),
        indexer: indexer_weights(o, il),
    }
}

pub fn hc<'a>(o: &'a Oracle, il: u32, sub: &str) -> HyperConnectionWeights<'a> {
    HyperConnectionWeights {
        function: o.w(&format!("layers.{il}.{sub}.fn")),
        base: o.w(&format!("layers.{il}.{sub}.base")),
        scale: o.w(&format!("layers.{il}.{sub}.scale")),
    }
}

pub fn moe_weights<'a>(o: &'a Oracle, il: u32, banks: &'a Owned) -> LatentMoeWeights<'a> {
    let m = |n: &str| o.w(&format!("layers.{il}.mlp.{n}"));
    LatentMoeWeights {
        router: m("gate.weight"),
        router_bias: m("gate.e_score_correction_bias"),
        latent_down: m("pre_expert_proj.weight"),
        latent_up: m("post_expert_proj.weight"),
        expert_gate: &banks.gate_bank,
        expert_up: &banks.up_bank,
        expert_down: m("experts.down_proj"),
        shared_gate: m("shared_experts.gate_proj.weight"),
        shared_up: m("shared_experts.up_proj.weight"),
        shared_down: m("shared_experts.down_proj.weight"),
        output_norm: m("combine_output_norm.weight"),
    }
}

pub fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[track_caller]
pub fn close(got: &[f64], want: &[f64], what: &str) {
    let err = max_abs(got, want);
    let scale = want.iter().map(|v| v.abs()).fold(0.0, f64::max);
    assert!(
        scale > 1e-3,
        "{what}: the oracle output is degenerate ({scale})"
    );
    // Relative to the output's magnitude: both sides run float64, in different summation orders.
    assert!(
        err <= BOUND * scale.max(1.0),
        "{what}: max_abs {err:e} over the bound {BOUND:e} x {scale:.3}"
    );
    eprintln!("{what}: max_abs {err:e} (scale {scale:.3})");
}

pub fn layer_weights<'a>(o: &'a Oracle, il: u32, banks: Option<&'a Owned>) -> LayerWeights<'a> {
    LayerWeights {
        input_norm: o.w(&format!("layers.{il}.input_layernorm.weight")),
        mlp_input_norm: o.w(&format!("layers.{il}.post_attention_layernorm.weight")),
        attention: attention_weights(o, il),
        attention_hc: hc(o, il, "attn_hc"),
        mlp_hc: hc(o, il, "ffn_hc"),
        mlp: match banks {
            None => MlpWeights::Dense {
                gate: o.w(&format!("layers.{il}.mlp.gate_proj.weight")),
                up: o.w(&format!("layers.{il}.mlp.up_proj.weight")),
                down: o.w(&format!("layers.{il}.mlp.down_proj.weight")),
                output_norm: o.w(&format!("layers.{il}.post_mlp_layernorm.weight")),
                intermediate: 24,
            },
            Some(banks) => MlpWeights::Latent(moe_weights(o, il, banks)),
        },
    }
}
