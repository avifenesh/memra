//! Gate of the Hy4-flash pooled indexer (`HYV4FlashCompressedIndexer`) above its budget.
//!
//! research/modelplan-onboarding-hy4flash-20261119/indexpool/gen_indexpool.py runs transformers
//! PR #50903 at b865850d0532 on a tiny config in float64 with 41 tokens, past the indexer budget
//! of 12, so each query of the two indexed layers keeps 2 of up to 10 closed groups plus its own.
//! Its `topk` follows the CUDA rule (ties by ascending index), and the record carries the token
//! keys, pooled group keys, group logits, selected groups and visible keys of each indexed layer.
//! The reference must give the same selection exactly and every output to 1e-12; the red arms
//! show that a wrong pool size, a missing `intra_group_ape`, the flipped tie order and dense
//! attention past the budget each fail.

mod hy_v4_flash_common;

use hy_v4_flash_common::{
    INDEXPOOL, Oracle, attention_weights, close, floats, head_weights, indexer_weights,
    layer_weights, max_abs, numerics, split_gate_up,
};
use memra_gguf::model_plan::{
    AttentionPlan, GroupTieOrder, HyV4AttentionKind, HyV4AttentionPlan, IndexPoolPlan, LayerPlan,
};
use memra_reference::hy_v4_flash::{
    IndexerWeights, decoder_layer, gated_mla_attention, gated_mla_attention_recalled, group_top_k,
    hyper_head, index_pool_groups, index_pool_logits, index_pool_recall, index_token_keys,
    rms_norm,
};
use serde_json::Value;

const HIDDEN: usize = 16;
const Q_LORA: usize = 12;
const TOKENS: usize = 41;

fn oracle() -> Oracle {
    let o = hy_v4_flash_common::oracle("indexpool/indexpool.json");
    assert_eq!(o.positions().len(), TOKENS);
    o
}

fn plan(il: u32) -> LayerPlan {
    hy_v4_flash_common::plan(INDEXPOOL, il)
}

fn attention(layer: &LayerPlan) -> HyV4AttentionPlan {
    let AttentionPlan::HyV4(attention) = &layer.attention else {
        panic!("layer {} is not Hy4-flash", layer.index)
    };
    *attention
}

fn index_plan(layer: &LayerPlan) -> IndexPoolPlan {
    let HyV4AttentionKind::Indexed(index) = attention(layer).kind else {
        panic!("layer {} is not indexed", layer.index)
    };
    index
}

/// One indexed layer's record from the oracle.
struct Indexer<'a>(&'a Value);

impl Indexer<'_> {
    fn of(o: &Oracle, il: u32) -> Indexer<'_> {
        Indexer(&o.record["indexer"][il.to_string()])
    }
    fn f(&self, key: &str) -> Vec<f64> {
        floats(&self.0[key])
    }
    /// `[tokens, groups]`, `-inf` where the oracle recorded a group that has not closed.
    fn logits(&self) -> Vec<f64> {
        self.0["logits"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter())
            .map(|v| v.as_f64().unwrap_or(f64::NEG_INFINITY))
            .collect()
    }
    fn selected(&self) -> Vec<Vec<usize>> {
        self.0["selected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|g| g.as_u64().unwrap() as usize)
                    .collect()
            })
            .collect()
    }
    fn visible(&self) -> Vec<bool> {
        self.0["visible"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter())
            .map(|v| v.as_u64().unwrap() == 1)
            .collect()
    }
}

fn logits_of(o: &Oracle, il: u32, index: &IndexPoolPlan, w: &IndexerWeights) -> Vec<f64> {
    let layer = plan(il);
    let rec = Indexer::of(o, il);
    index_pool_logits(
        index,
        attention(&layer).rope.base as f64,
        layer.pre_attention_norm,
        w,
        &rec.f("x"),
        &rec.f("q_resid"),
        &o.positions(),
        HIDDEN,
        Q_LORA,
    )
}

#[test]
fn token_keys_pooled_keys_and_logits_match() {
    let o = oracle();
    for il in 0..2 {
        let layer = plan(il);
        let index = index_plan(&layer);
        assert_eq!(index.ties, GroupTieOrder::LowerIndexFirst);
        let w = indexer_weights(&o, il).unwrap();
        let rec = Indexer::of(&o, il);
        assert_eq!(rec.0["kv_length"], TOKENS);
        let keys = index_token_keys(&index, layer.pre_attention_norm, &w, &rec.f("x"), HIDDEN);
        close(
            &keys,
            &rec.f("token_keys"),
            &format!("layer {il} token keys"),
        );
        let groups = index_pool_groups(&index, layer.pre_attention_norm, &w, &keys);
        close(
            &groups,
            &rec.f("group_keys"),
            &format!("layer {il} group keys"),
        );
        assert_eq!(groups.len(), 10 * 12, "41 tokens close 10 groups of 4");

        let got = logits_of(&o, il, &index, &w);
        let want = rec.logits();
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.is_finite(), w.is_finite(), "layer {il} logit {i} masking");
        }
        let finite =
            |v: &[f64]| -> Vec<f64> { v.iter().copied().filter(|x| x.is_finite()).collect() };
        close(
            &finite(&got),
            &finite(&want),
            &format!("layer {il} group logits"),
        );
    }
}

/// The selection is exact: the same groups per query and the same visible keys, ties included.
#[test]
fn recall_matches_with_the_cuda_tie_order() {
    let o = oracle();
    for il in 0..2 {
        let index = index_plan(&plan(il));
        let rec = Indexer::of(&o, il);
        let logits = rec.logits();
        let groups = logits.len() / TOKENS;
        let keep = (index.top_k / index.pool) as usize - 1;
        let mut ties = 0;
        for (t, want) in rec.selected().iter().enumerate() {
            let row = &logits[t * groups..(t + 1) * groups];
            let mut got = group_top_k(row, keep, index.ties);
            got.sort_unstable();
            assert_eq!(&got, want, "layer {il} query {t} groups");
            let kth = want.iter().map(|g| row[*g]).fold(f64::INFINITY, f64::min);
            if kth.is_finite() && row.iter().filter(|v| **v == kth).count() > 1 {
                ties += 1;
            }
        }
        assert!(
            ties > 0,
            "layer {il}: no query has a tie at its last kept group"
        );
        let visible = index_pool_recall(
            &index,
            &logits_of(&o, il, &index, &indexer_weights(&o, il).unwrap()),
            TOKENS,
        )
        .expect("past the budget");
        assert_eq!(visible, rec.visible(), "layer {il} visible keys");
        // Past the budget the selection narrows the causal set.
        let causal = TOKENS * (TOKENS + 1) / 2;
        let kept = visible.iter().filter(|v| **v).count();
        assert!(
            kept < causal,
            "layer {il}: {kept} of {causal} causal keys kept"
        );
        eprintln!("layer {il}: {kept} of {causal} causal keys kept, {ties} queries tie at the cut");
    }
}

#[test]
fn attention_matches_above_the_budget() {
    let o = oracle();
    let positions = o.positions();
    for il in 0..4 {
        let layer = plan(il);
        let (input, output) = o.module(&format!("layers.{il}.self_attn"));
        let got = gated_mla_attention(
            &attention(&layer),
            layer.pre_attention_norm,
            &attention_weights(&o, il),
            &input[0],
            &positions,
            HIDDEN,
        );
        close(&got, &output[0], &format!("layer {il} attention"));
    }
}

#[test]
fn decoder_layers_and_the_stack_match_above_the_budget() {
    let o = oracle();
    let positions = o.positions();
    let banks: Vec<_> = (0..4)
        .map(|il| (il > 0).then(|| split_gate_up(&o, il)))
        .collect();
    let (first_input, _) = o.module("layers.0");
    let mut streams = first_input[0].clone();
    for il in 0..4u32 {
        let layer = plan(il);
        let w = layer_weights(&o, il, banks[il as usize].as_ref());
        let (input, output) = o.module(&format!("layers.{il}"));
        close(
            &decoder_layer(&layer, &w, &input[0], &positions, HIDDEN, None),
            &output[0],
            &format!("layer {il}"),
        );
        streams = decoder_layer(&layer, &w, &streams, &positions, HIDDEN, None);
    }
    let norm = plan(0).pre_attention_norm;
    let collapsed = hyper_head(&numerics(), norm, &head_weights(&o), &streams, HIDDEN);
    close(
        &rms_norm(&collapsed, o.w("norm.weight"), norm),
        &floats(&o.record["last_hidden_state"]),
        "last hidden state",
    );
}

/// The visible keys with the flipped tie order: higher index first among equal logits.
fn recall_flipped(index: &IndexPoolPlan, logits: &[f64]) -> Vec<bool> {
    let pool = index.pool as usize;
    let groups = logits.len() / TOKENS;
    let keep = ((index.top_k / index.pool) as usize - 1).min(groups);
    let mut visible = vec![false; TOKENS * TOKENS];
    for t in 0..TOKENS {
        let row = &logits[t * groups..(t + 1) * groups];
        let mut order: Vec<usize> = (0..groups).collect();
        order.sort_by(|a, b| row[*b].total_cmp(&row[*a]).then(b.cmp(a)));
        order.truncate(keep);
        order.push(t / pool);
        for g in order {
            for s in g * pool..(g + 1) * pool {
                let s = s.min(TOKENS - 1);
                if s <= t {
                    visible[t * TOKENS + s] = true;
                }
            }
        }
    }
    visible
}

/// Red arms on the indexed layers: each wrong program moves the attention output past 1e-6.
#[test]
fn wrong_indexer_programs_fail() {
    let o = oracle();
    let positions = o.positions();
    let red = |got: &[f64], want: &[f64], what: &str| {
        let err = max_abs(got, want);
        assert!(err > 1e-6, "{what}: red arm within {err:e}");
        eprintln!("{what}: max_abs {err:e}");
    };
    for il in 0..2u32 {
        let layer = plan(il);
        let plan_attention = attention(&layer);
        let index = index_plan(&layer);
        let (input, output) = o.module(&format!("layers.{il}.self_attn"));
        let want = &output[0];
        let run = |attention: &HyV4AttentionPlan,
                   w: &memra_reference::hy_v4_flash::AttentionWeights| {
            gated_mla_attention(
                attention,
                layer.pre_attention_norm,
                w,
                &input[0],
                &positions,
                HIDDEN,
            )
        };

        // Pool size 2: the program pools pairs and keeps five of them.
        let mut pairs = plan_attention;
        pairs.kind = HyV4AttentionKind::Indexed(IndexPoolPlan { pool: 2, ..index });
        let w = attention_weights(&o, il);
        let ape = &w.indexer.as_ref().unwrap().ape[..2 * 12];
        let pair_weights = memra_reference::hy_v4_flash::AttentionWeights {
            indexer: Some(IndexerWeights {
                ape,
                ..indexer_weights(&o, il).unwrap()
            }),
            ..attention_weights(&o, il)
        };
        red(
            &run(&pairs, &pair_weights),
            want,
            &format!("layer {il} pool size 2"),
        );

        // intra_group_ape omitted.
        let zeros = vec![0.0; 4 * 12];
        let no_ape = memra_reference::hy_v4_flash::AttentionWeights {
            indexer: Some(IndexerWeights {
                ape: &zeros,
                ..indexer_weights(&o, il).unwrap()
            }),
            ..attention_weights(&o, il)
        };
        red(
            &run(&plan_attention, &no_ape),
            want,
            &format!("layer {il} ape omitted"),
        );

        // The flipped tie order selects other groups for some queries.
        let logits = logits_of(&o, il, &index, &indexer_weights(&o, il).unwrap());
        let flipped = recall_flipped(&index, &logits);
        assert_ne!(
            flipped,
            Indexer::of(&o, il).visible(),
            "layer {il}: ties decide nothing"
        );
        let recalled = |visible: &[bool]| {
            gated_mla_attention_recalled(
                &plan_attention,
                layer.pre_attention_norm,
                &attention_weights(&o, il),
                &input[0],
                &positions,
                HIDDEN,
                Some(visible),
            )
        };
        red(
            &recalled(&flipped),
            want,
            &format!("layer {il} tie order flipped"),
        );

        // Dense causal attention past the budget, as below it.
        let dense: Vec<bool> = (0..TOKENS * TOKENS)
            .map(|i| i % TOKENS <= i / TOKENS)
            .collect();
        red(
            &recalled(&dense),
            want,
            &format!("layer {il} dense past the budget"),
        );
    }
}
