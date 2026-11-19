//! Module-level gate of the Hy4-flash reference operations against the pinned transformers code.
//!
//! research/modelplan-onboarding-hy4flash-20261119/modules/gen_modules.py runs transformers PR
//! #50903 at b865850d0532 on a tiny config in float64 (its float32 casts mapped to float64) and
//! records every module's inputs and outputs. Each test feeds a recorded input through the
//! `memra_reference::hy_v4_flash` operation, compiled from the pack's typed layer plan, and
//! requires the recorded output to 1e-12. The red arms show the bound tells programs apart.

use std::collections::BTreeMap;
use std::path::Path;

use memra_gguf::config::{
    HyV4DflyConfig, HyV4FlashConfig, HyV4HeadGeometry, HyV4LayerKind, HyV4VisionConfig,
};
use memra_gguf::model_packs::hy_v4_flash::layer_plan;
use memra_gguf::model_plan::{LayerPlan, NormKind, NormPlan, WeightTransform};
use memra_gguf::modelopt_quant::LayerQuantMap;
use memra_reference::hy_v4_flash::{
    AttentionWeights, HyperConnectionWeights, LatentMoeWeights, LayerWeights, MlpWeights,
    decoder_layer, gated_mla_attention, hyper_connection, hyper_head, latent_moe, rms_norm,
};
use serde_json::Value;

const BOUND: f64 = 1e-12;
/// The tiny config's epsilons, 2^-17 and 2^-20: exact in f32, which the typed plan carries.
const RMS_EPS: f32 = 1.0 / 131_072.0;
const HC_EPS: f32 = 1.0 / 1_048_576.0;

struct Oracle {
    record: Value,
    weights: BTreeMap<String, Vec<f64>>,
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn oracle() -> Oracle {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../research/modelplan-onboarding-hy4flash-20261119/modules/modules.json");
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
    fn w(&self, name: &str) -> &[f64] {
        self.weights
            .get(name)
            .unwrap_or_else(|| panic!("no weight {name}"))
    }
    fn module(&self, name: &str) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let m = &self.record["modules"][name];
        let all = |k: &str| m[k].as_array().unwrap().iter().map(floats).collect();
        (all("in"), all("out"))
    }
    fn positions(&self) -> Vec<u32> {
        self.record["positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_u64().unwrap() as u32)
            .collect()
    }
}

/// The tiny config of gen_modules.py as the pack's config type (no quant map: BF16 everywhere).
fn config() -> HyV4FlashConfig {
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
        index_n_heads: 2,
        index_head_dim: 8,
        index_topk: 8,
        index_pool: 4,
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

fn numerics() -> memra_gguf::model_plan::HyV4FlashNumerics {
    memra_gguf::model_plan::HyV4FlashNumerics {
        streams: 4,
        exit_epsilon: HC_EPS,
        output_projection_f32: true,
    }
}

fn head_weights(o: &Oracle) -> HyperConnectionWeights<'_> {
    HyperConnectionWeights {
        function: o.w("hc_head.hc_fn"),
        base: o.w("hc_head.hc_base"),
        scale: o.w("hc_head.hc_scale"),
    }
}

fn plan(il: u32) -> LayerPlan {
    let hy = config();
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
struct Owned {
    gate_bank: Vec<f64>,
    up_bank: Vec<f64>,
}

fn split_gate_up(o: &Oracle, il: u32) -> Owned {
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

fn attention_weights<'a>(o: &'a Oracle, il: u32) -> AttentionWeights<'a> {
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
    }
}

fn hc<'a>(o: &'a Oracle, il: u32, sub: &str) -> HyperConnectionWeights<'a> {
    HyperConnectionWeights {
        function: o.w(&format!("layers.{il}.{sub}.fn")),
        base: o.w(&format!("layers.{il}.{sub}.base")),
        scale: o.w(&format!("layers.{il}.{sub}.scale")),
    }
}

fn moe_weights<'a>(o: &'a Oracle, il: u32, banks: &'a Owned) -> LatentMoeWeights<'a> {
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

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[track_caller]
fn close(got: &[f64], want: &[f64], what: &str) {
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

#[test]
fn attention_matches_on_indexed_and_sliding_layers() {
    let o = oracle();
    let positions = o.positions();
    for il in 0..4 {
        let layer = plan(il);
        let memra_gguf::model_plan::AttentionPlan::HyV4(attention) = &layer.attention else {
            panic!()
        };
        let (input, output) = o.module(&format!("layers.{il}.self_attn"));
        let got = gated_mla_attention(
            attention,
            layer.pre_attention_norm,
            &attention_weights(&o, il),
            &input[0],
            &positions,
            16,
        );
        close(&got, &output[0], &format!("layer {il} attention"));
    }
}

#[test]
fn hyper_connections_and_norms_match() {
    let o = oracle();
    for il in 0..4 {
        let layer = plan(il);
        let memra_gguf::model_plan::ResidualTopology::HyV4(residual) = &layer.residual else {
            panic!()
        };
        for sub in ["attn_hc", "ffn_hc"] {
            let (input, output) = o.module(&format!("layers.{il}.{sub}"));
            let (post, collapsed) = hyper_connection(residual, &hc(&o, il, sub), &input[0], 16);
            close(&post, &output[0], &format!("layer {il} {sub} post"));
            close(&collapsed, &output[1], &format!("layer {il} {sub} read"));
        }
        for (sub, norm) in [
            ("input_layernorm", layer.pre_attention_norm),
            ("post_attention_layernorm", layer.pre_mlp_norm),
        ] {
            let (input, output) = o.module(&format!("layers.{il}.{sub}"));
            let got = rms_norm(&input[0], o.w(&format!("layers.{il}.{sub}.weight")), norm);
            close(&got, &output[0], &format!("layer {il} {sub}"));
        }
    }
    let (input, output) = o.module("hc_head");
    let got = hyper_head(&numerics(), RMS_EPS, &head_weights(&o), &input[0], 16);
    close(&got, &output[0], "hc_head");
}

#[test]
fn latent_moe_matches() {
    let o = oracle();
    for il in 1..4 {
        let layer = plan(il);
        let memra_gguf::model_plan::MlpPlan::LatentMoe(moe) = &layer.mlp else {
            panic!()
        };
        let memra_gguf::model_plan::ResidualTopology::HyV4(residual) = &layer.residual else {
            panic!()
        };
        let banks = split_gate_up(&o, il);
        let (input, output) = o.module(&format!("layers.{il}.mlp"));
        let got = latent_moe(
            moe,
            residual.mlp_output_norm,
            &moe_weights(&o, il, &banks),
            &input[0],
            16,
        );
        close(&got, &output[0], &format!("layer {il} MoE"));
    }
}

fn layer_weights<'a>(o: &'a Oracle, il: u32, banks: Option<&'a Owned>) -> LayerWeights<'a> {
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

/// Every decoder layer on its recorded input, then the whole 4-layer stack from the embedding
/// streams through `hc_head` and the final norm to the model's last hidden state.
#[test]
fn decoder_layers_and_the_stack_match() {
    let o = oracle();
    let positions = o.positions();
    let banks: Vec<Option<Owned>> = (0..4)
        .map(|il| (il > 0).then(|| split_gate_up(&o, il)))
        .collect();
    let (first_input, _) = o.module("layers.0");
    let mut streams = first_input[0].clone();
    for il in 0..4u32 {
        let layer = plan(il);
        let w = layer_weights(&o, il, banks[il as usize].as_ref());
        let (input, output) = o.module(&format!("layers.{il}"));
        close(
            &decoder_layer(&layer, &w, &input[0], &positions, 16),
            &output[0],
            &format!("layer {il}"),
        );
        streams = decoder_layer(&layer, &w, &streams, &positions, 16);
    }
    let collapsed = hyper_head(&numerics(), RMS_EPS, &head_weights(&o), &streams, 16);
    let norm = plan(0).pre_attention_norm;
    let last = rms_norm(&collapsed, o.w("norm.weight"), norm);
    close(
        &last,
        &floats(&o.record["last_hidden_state"]),
        "last hidden state",
    );
}

/// Red arms: a sink dropped, the shared expert clamped like the routed experts, the
/// hyper-connection without its write-back magnitude, and a sliding layer read as full
/// attention each move the output far past the bound.
#[test]
fn the_bound_tells_programs_apart() {
    let o = oracle();
    let positions = o.positions();
    let red = |got: &[f64], want: &[f64], what: &str| {
        let err = max_abs(got, want);
        assert!(err > 1e-6, "{what}: red arm within {err:e}");
    };

    let layer = plan(2);
    let memra_gguf::model_plan::AttentionPlan::HyV4(attention) = &layer.attention else {
        panic!()
    };
    let (input, output) = o.module("layers.2.self_attn");
    let mut w = attention_weights(&o, 2);
    let no_sink = vec![-1e300; 4];
    w.sinks = &no_sink;
    red(
        &gated_mla_attention(
            attention,
            layer.pre_attention_norm,
            &w,
            &input[0],
            &positions,
            16,
        ),
        &output[0],
        "sink dropped",
    );
    let mut full = *attention;
    full.kind = plan(1).attention_kind_for_test();
    red(
        &gated_mla_attention(
            &full,
            layer.pre_attention_norm,
            &attention_weights(&o, 2),
            &input[0],
            &positions,
            16,
        ),
        &output[0],
        "sliding read as full",
    );

    let memra_gguf::model_plan::ResidualTopology::HyV4(residual) = &layer.residual else {
        panic!()
    };
    let (input, output) = o.module("layers.2.attn_hc");
    let mut unit = *residual;
    unit.post_magnitude = 1.0;
    red(
        &hyper_connection(&unit, &hc(&o, 2, "attn_hc"), &input[0], 16).0,
        &output[0],
        "write-back magnitude 1",
    );

    let memra_gguf::model_plan::MlpPlan::LatentMoe(moe) = &layer.mlp else {
        panic!()
    };
    let banks = split_gate_up(&o, 2);
    let (input, output) = o.module("layers.2.mlp");
    let mut swapped = moe.clone();
    // gate and up exchanged: the bank split is part of the program.
    let w = LatentMoeWeights {
        expert_gate: &banks.up_bank,
        expert_up: &banks.gate_bank,
        ..moe_weights(&o, 2, &banks)
    };
    red(
        &latent_moe(&swapped, residual.mlp_output_norm, &w, &input[0], 16),
        &output[0],
        "gate and up swapped",
    );
    swapped.moe.router = memra_gguf::model_plan::RouterPlan::Sigmoid {
        normalize_selected: true,
        scaling_factor: 1.75,
        selection_bias: false,
    };
    red(
        &latent_moe(
            &swapped,
            residual.mlp_output_norm,
            &moe_weights(&o, 2, &banks),
            &input[0],
            16,
        ),
        &output[0],
        "correction bias ignored",
    );
}

trait AttentionKindForTest {
    fn attention_kind_for_test(&self) -> memra_gguf::model_plan::HyV4AttentionKind;
}

impl AttentionKindForTest for LayerPlan {
    fn attention_kind_for_test(&self) -> memra_gguf::model_plan::HyV4AttentionKind {
        let memra_gguf::model_plan::AttentionPlan::HyV4(attention) = &self.attention else {
            panic!()
        };
        attention.kind
    }
}
