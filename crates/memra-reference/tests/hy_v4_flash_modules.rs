//! Module-level gate of the Hy4-flash reference operations against the pinned transformers code.
//!
//! research/modelplan-onboarding-hy4flash-20261119/modules/gen_modules.py runs transformers PR
//! #50903 at b865850d0532 on a tiny config in float64 (its float32 casts mapped to float64) and
//! records every module's inputs and outputs. Each test feeds a recorded input through the
//! `memra_reference::hy_v4_flash` operation, compiled from the pack's typed layer plan, and
//! requires the recorded output to 1e-12. The red arms show the bound tells programs apart.

mod hy_v4_flash_common;

use hy_v4_flash_common::{
    MODULES, Owned, attention_weights, close, floats, hc, head_weights, layer_weights, max_abs,
    moe_weights, numerics, split_gate_up,
};
use memra_gguf::model_plan::LayerPlan;
use memra_reference::hy_v4_flash::{
    LatentMoeWeights, decoder_layer, gated_mla_attention, hyper_connection, hyper_head, latent_moe,
    rms_norm,
};

fn oracle() -> hy_v4_flash_common::Oracle {
    hy_v4_flash_common::oracle("modules/modules.json")
}

fn plan(il: u32) -> LayerPlan {
    hy_v4_flash_common::plan(MODULES, il)
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
    let got = hyper_head(
        &numerics(),
        plan(0).pre_attention_norm,
        &head_weights(&o),
        &input[0],
        16,
    );
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
    let collapsed = hyper_head(
        &numerics(),
        plan(0).pre_attention_norm,
        &head_weights(&o),
        &streams,
        16,
    );
    let norm = plan(0).pre_attention_norm;
    let last = rms_norm(&collapsed, o.w("norm.weight"), norm);
    close(
        &last,
        &floats(&o.record["last_hidden_state"]),
        "last hidden state",
    );
}

/// Red arms: a sink dropped, a sliding layer read as full attention, the hyper-connection
/// without its write-back magnitude, gate and up swapped, and the router's correction bias
/// ignored each move the output far past the bound.
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
    // Below the budget the indexer does not run; an indexed kind still carries its weights.
    let as_indexed = memra_reference::hy_v4_flash::AttentionWeights {
        indexer: hy_v4_flash_common::indexer_weights(&o, 1),
        ..attention_weights(&o, 2)
    };
    red(
        &gated_mla_attention(
            &full,
            layer.pre_attention_norm,
            &as_indexed,
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
