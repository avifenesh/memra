//! Gate of the Hy4-flash Engram memory (`HYV4FlashEngramLayer`).
//!
//! research/modelplan-onboarding-hy4flash-20261119/engram/gen_engram.py records two runs of
//! transformers PR #50903 at b865850d0532:
//!
//! - `artifact_hash`: the integer program on the artifact's own `config.json` and lookup table.
//!   The pack compiles the four Engram plans from the same config; memra_gguf's hashing must give
//!   the pinned NumPy multipliers and the pinned hash ids bit for bit, with the table entries the
//!   tokens read (the 14 placeholders past the table, the pad id and a packed second document
//!   included).
//! - `tiny`: the step 4 tiny model with one Engram layer, float64. The reference must give the
//!   module's output, every decoder layer and the last hidden state to 1e-12.
//!
//! The red arms show that a 0-based seed, a multiplier bound from the table length, placeholders
//! read as themselves (vLLM), swapped order moduli, the 2^-7 epsilon, one key projection for every
//! stream, the per-stream scales dropped and the mask ignored each fail.

mod hy_v4_flash_common;

use hy_v4_flash_common::{
    INDEXPOOL, Oracle, close, floats, head_weights, layer_weights, max_abs, numerics, split_gate_up,
};
use memra_gguf::config::{HfConfig, HyV4FlashConfig, ModelConfig};
use memra_gguf::hy_v4_flash_engram::{
    compressed_vocab, engram_mask, hash_ids, hash_multipliers, head_offsets, ngram_tokens,
};
use memra_gguf::model_packs::hy_v4_flash::layer_plan;
use memra_gguf::model_plan::{
    HyV4EngramNormEpsilon, HyV4EngramPlan, LayerPlan, NormKind, NormPlan, ResidualTopology,
    WeightTransform,
};
use memra_reference::hy_v4_flash::{
    EngramInputs, EngramTable, EngramWeights, decoder_layer, engram, engram_at_epsilon, hyper_head,
    rms_norm,
};
use serde_json::Value;

const HIDDEN: usize = 16;
const STREAMS: usize = 4;

fn fixture() -> Value {
    let o = hy_v4_flash_common::oracle_part("engram/engram.json", Some("artifact_hash"));
    o.record
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

fn u32s(v: &Value) -> Vec<u32> {
    ints(v).into_iter().map(|x| x as u32).collect()
}

fn u64s(v: &Value) -> Vec<u64> {
    ints(v).into_iter().map(|x| x as u64).collect()
}

fn norm_of(hy: &HyV4FlashConfig) -> NormPlan {
    NormPlan {
        kind: NormKind::Rms,
        epsilon: hy.rms_norm_eps,
        weight_transform: WeightTransform::Identity,
    }
}

fn engram_plan(layer: &LayerPlan) -> HyV4EngramPlan {
    let ResidualTopology::HyV4(residual) = &layer.residual else {
        panic!("layer {} is not Hy4-flash", layer.index)
    };
    residual
        .engram
        .unwrap_or_else(|| panic!("layer {} has no Engram", layer.index))
}

/// The artifact's lookup table, as far as the recorded tokens read it; every other entry is a
/// sentinel no correct hash reads.
fn sparse_table(rec: &Value, size: usize) -> Vec<i32> {
    let mut table = vec![-1i32; size];
    for (id, value) in rec["lookup_entries"].as_object().unwrap() {
        table[id.parse::<usize>().unwrap()] = value.as_i64().unwrap() as i32;
    }
    table
}

#[test]
fn artifact_hash_ids_match_the_pinned_numpy_draw() {
    let rec = fixture();
    let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(
        "../../memra-gguf/src/model_packs/hy_v4_flash/fixtures/config.json"
    )));
    let hy = config.hy_v4_flash.expect("hy_v4_flash config");
    assert_eq!(rec["lookup_table_size"], hy.engram_lookup_table_size);
    assert_eq!(rec["engram_seed"], hy.engram_seed);
    let size = hy.engram_lookup_table_size as usize;
    let table = sparse_table(&rec, size);
    let vocab = rec["compressed_vocab"].as_u64().unwrap();
    assert_eq!(vocab, 92_620, "max(lookup_table) + 1 on the artifact");
    let ids = u32s(&rec["ids"]);
    let positions = u32s(&rec["positions"]);
    assert!(
        (120_818..120_832).all(|p| ids.contains(&p)),
        "all 14 placeholders"
    );
    assert!(
        positions.iter().filter(|p| **p == 0).count() == 2,
        "two documents"
    );

    let mut checked = 0;
    for il in 0..hy.num_hidden_layers {
        let layer = layer_plan(&hy, il, norm_of(&hy)).unwrap();
        let ResidualTopology::HyV4(residual) = &layer.residual else {
            panic!()
        };
        let Some(plan) = residual.engram else {
            continue;
        };
        assert_eq!(plan.layer_id, il + 1, "the 1-based id sits in layer id - 1");
        assert_eq!(plan.norm_epsilon, HyV4EngramNormEpsilon::OpMathF32);
        assert_eq!(plan.norm_epsilon.value(), f32::EPSILON);
        assert_eq!((plan.max_ngram, plan.heads, plan.head_dim), (3, 16, 96));
        let want = &rec["layers"][plan.layer_id.to_string()];
        let slot = hy.engram_slot(il).unwrap();
        let moduli = &hy.engram_head_vocab_sizes[slot];
        assert_eq!(moduli, &u64s(&want["moduli"]), "layer {il} moduli");

        let multipliers = hash_multipliers(&plan, vocab).unwrap();
        assert_eq!(
            multipliers,
            ints(&want["multipliers"]),
            "layer {il} multipliers"
        );
        let ngrams = ngram_tokens(&plan, &ids, &positions);
        assert_eq!(ngrams, u32s(&rec["ngram_tokens"]), "n-gram tokens");
        let mask: Vec<i64> = engram_mask(&plan, &ids).iter().map(|m| *m as i64).collect();
        assert_eq!(mask, ints(&rec["mask"]), "mask");
        let got = hash_ids(&plan, &table, &multipliers, moduli, &ngrams).unwrap();
        assert_eq!(got, u64s(&want["hash_ids"]), "layer {il} hash ids");

        // Red arms: each changes the hash ids.
        let wrong = |what: &str, got: Vec<u64>| {
            assert_ne!(
                got,
                u64s(&want["hash_ids"]),
                "layer {il}: {what} hashes the same"
            );
        };
        let by_length = hash_multipliers(&plan, size as u64).unwrap();
        assert_eq!(
            by_length,
            ints(&want["multipliers_table_length_bound"]),
            "the table-length bound, as the pinned draw gives it"
        );
        wrong(
            "the table-length bound",
            hash_ids(&plan, &table, &by_length, moduli, &ngrams).unwrap(),
        );
        let zero_based = HyV4EngramPlan {
            hash_seed: hy.engram_seed + 10_007 * u64::from(plan.layer_id - 1),
            ..plan
        };
        let zero_based_multipliers = hash_multipliers(&zero_based, vocab).unwrap();
        wrong(
            "the 0-based seed",
            hash_ids(&plan, &table, &zero_based_multipliers, moduli, &ngrams).unwrap(),
        );
        // vLLM extends the map with the identity for the placeholders past it.
        let mut extended = table.clone();
        extended.extend((size as i32)..(size as i32 + 14));
        let vllm = HyV4EngramPlan {
            lookup_table_size: extended.len() as u32,
            ..plan
        };
        wrong(
            "placeholders as themselves",
            hash_ids(&vllm, &extended, &multipliers, moduli, &ngrams).unwrap(),
        );
        let mut swapped = moduli[8..].to_vec();
        swapped.extend_from_slice(&moduli[..8]);
        wrong(
            "order moduli swapped",
            hash_ids(&plan, &table, &multipliers, &swapped, &ngrams).unwrap(),
        );
        checked += 1;
    }
    assert_eq!(checked, 4, "four Engram layers");
}

// ---- the tiny model ----------------------------------------------------------------------

fn tiny() -> Oracle {
    let o = hy_v4_flash_common::oracle_part("engram/engram.json", Some("tiny"));
    assert_eq!(o.record["op_math_eps"].as_f64().unwrap(), 2f64.powi(-23));
    o
}

/// The step 4 tiny config with the oracle's Engram layer.
fn tiny_config() -> HyV4FlashConfig {
    let mut hy = hy_v4_flash_common::config(INDEXPOOL);
    hy.engram_layer_ids = vec![2];
    hy.engram_max_ngram_size = 3;
    hy.engram_n_head_per_ngram = 2;
    hy.engram_n_embed_per_ngram = 8;
    hy.engram_head_vocab_sizes = vec![vec![29, 31, 37, 41]];
    hy.engram_pad_id = 5;
    hy.engram_seed = 3;
    hy.engram_lookup_table_size = 60;
    hy.vocab_size = 64;
    hy
}

fn tiny_plan(il: u32) -> LayerPlan {
    let hy = tiny_config();
    layer_plan(&hy, il, norm_of(&hy)).unwrap()
}

struct TinyInputs {
    rows: Vec<u64>,
    mask: Vec<bool>,
}

fn tiny_inputs(o: &Oracle, plan: &HyV4EngramPlan) -> TinyInputs {
    let rec = &o.record;
    let table: Vec<i32> = ints(&rec["lookup_table"])
        .into_iter()
        .map(|v| v as i32)
        .collect();
    let vocab = compressed_vocab(plan, &table).unwrap();
    assert_eq!(vocab, 40, "a map with fewer compressed ids than entries");
    let multipliers = hash_multipliers(plan, vocab).unwrap();
    assert_eq!(multipliers, ints(&rec["multipliers"]), "tiny multipliers");
    let ids = u32s(&rec["input_ids"]);
    let ngrams = ngram_tokens(plan, &ids, &u32s(&rec["positions"]));
    assert_eq!(ngrams, u32s(&rec["ngram_tokens"]), "tiny n-gram tokens");
    let moduli = u64s(&rec["config"]["moduli"]);
    let hashes = hash_ids(plan, &table, &multipliers, &moduli, &ngrams).unwrap();
    assert_eq!(hashes, u64s(&rec["hash_ids"]), "tiny hash ids");
    let offsets = head_offsets(&moduli);
    let heads = plan.heads as usize;
    let rows = hashes
        .iter()
        .enumerate()
        .map(|(i, h)| h + offsets[i % heads])
        .collect();
    let mask = engram_mask(plan, &ids);
    assert_eq!(
        mask.iter().map(|m| *m as i64).collect::<Vec<_>>(),
        ints(&rec["mask"])
    );
    assert!(mask.iter().any(|m| !m), "the pad id appears");
    TinyInputs { rows, mask }
}

struct TinyTable {
    codes: Vec<u8>,
    scales: Vec<f64>,
}

fn tiny_table(o: &Oracle) -> TinyTable {
    TinyTable {
        codes: ints(&o.record["table_codes"])
            .into_iter()
            .map(|c| c as u8)
            .collect(),
        scales: floats(&o.record["table_scales"]),
    }
}

fn engram_weights<'a>(o: &'a Oracle, table: &'a TinyTable) -> EngramWeights<'a> {
    let e = |n: &str| o.w(&format!("layers.1.engram_layer.{n}"));
    let per_stream = |f: &dyn Fn(usize) -> String| (0..STREAMS).map(|s| e(&f(s))).collect();
    EngramWeights {
        table: EngramTable {
            codes: &table.codes,
            scales: &table.scales,
        },
        head_norms: (0..4)
            .map(|h| e(&format!("lookup_per_head_norms.{h}.weight")))
            .collect(),
        value: e("value_proj.weight"),
        value_bias: e("value_proj.bias"),
        keys: per_stream(&|s| format!("key_projs.{s}.weight")),
        key_biases: per_stream(&|s| format!("key_projs.{s}.bias")),
        norm_key: per_stream(&|s| format!("norm_key.{s}.weight")),
        norm_query: per_stream(&|s| format!("norm_query.{s}.weight")),
        pre_gamma: e("unified_pre_gamma"),
        stream_scales: per_stream(&|s| format!("unified_per_hc_scales.{s}")),
    }
}

#[test]
fn engram_module_matches() {
    let o = tiny();
    let plan = engram_plan(&tiny_plan(1));
    assert_eq!((plan.layer_id, plan.hash_seed), (2, 3 + 10_007 * 2));
    for il in [0, 2, 3] {
        let ResidualTopology::HyV4(residual) = &tiny_plan(il).residual else {
            panic!()
        };
        assert!(residual.engram.is_none(), "layer {il}");
    }
    let inputs = tiny_inputs(&o, &plan);
    let table = tiny_table(&o);
    let w = engram_weights(&o, &table);
    let (input, output) = o.module("layers.1.engram_layer");
    let got = engram(
        &plan,
        &w,
        &EngramInputs {
            rows: &inputs.rows,
            mask: &inputs.mask,
        },
        &input[0],
        STREAMS,
        HIDDEN,
    );
    close(&got, &output[0], "Engram residual");
    // Masked positions add nothing.
    for (t, m) in inputs.mask.iter().enumerate() {
        if !m {
            let row = &got[t * STREAMS * HIDDEN..(t + 1) * STREAMS * HIDDEN];
            assert!(row.iter().all(|v| *v == 0.0), "token {t}");
        }
    }
}

#[test]
fn decoder_layers_and_the_stack_match_with_engram() {
    let o = tiny();
    let positions = u32s(&o.record["positions"]);
    let plan = engram_plan(&tiny_plan(1));
    let inputs = tiny_inputs(&o, &plan);
    let table = tiny_table(&o);
    let engram_inputs = EngramInputs {
        rows: &inputs.rows,
        mask: &inputs.mask,
    };
    let banks: Vec<_> = (0..4)
        .map(|il| (il > 0).then(|| split_gate_up(&o, il)))
        .collect();
    let (first_input, _) = o.module("layers.0");
    let mut streams = first_input[0].clone();
    for il in 0..4u32 {
        let layer = tiny_plan(il);
        let mut w = layer_weights(&o, il, banks[il as usize].as_ref());
        if il == 1 {
            w.engram = Some(engram_weights(&o, &table));
        }
        let (input, output) = o.module(&format!("layers.{il}"));
        close(
            &decoder_layer(
                &layer,
                &w,
                &input[0],
                &positions,
                HIDDEN,
                Some(&engram_inputs),
            ),
            &output[0],
            &format!("layer {il}"),
        );
        streams = decoder_layer(
            &layer,
            &w,
            &streams,
            &positions,
            HIDDEN,
            Some(&engram_inputs),
        );
    }
    let norm = tiny_plan(0).pre_attention_norm;
    let collapsed = hyper_head(&numerics(), norm, &head_weights(&o), &streams, HIDDEN);
    close(
        &rms_norm(&collapsed, o.w("norm.weight"), norm),
        &floats(&o.record["last_hidden_state"]),
        "last hidden state",
    );
}

#[test]
fn wrong_engram_programs_fail() {
    let o = tiny();
    let plan = engram_plan(&tiny_plan(1));
    let inputs = tiny_inputs(&o, &plan);
    let table = tiny_table(&o);
    let (input, output) = o.module("layers.1.engram_layer");
    let want = &output[0];
    let red = |got: &[f64], what: &str| {
        let err = max_abs(got, want);
        assert!(err > 1e-6, "{what}: red arm within {err:e}");
        eprintln!("{what}: max_abs {err:e}");
    };
    let run = |w: &EngramWeights, mask: &[bool], epsilon: f64| {
        engram_at_epsilon(
            &plan,
            epsilon,
            w,
            &EngramInputs {
                rows: &inputs.rows,
                mask,
            },
            &input[0],
            STREAMS,
            HIDDEN,
        )
    };
    let eps = f64::from(plan.norm_epsilon.value());
    // The reading of `finfo(input dtype).eps` on BF16 activations.
    red(
        &run(&engram_weights(&o, &table), &inputs.mask, 2f64.powi(-7)),
        "epsilon 2^-7",
    );
    let mut shared = engram_weights(&o, &table);
    shared.keys = vec![shared.keys[0]; STREAMS];
    shared.key_biases = vec![shared.key_biases[0]; STREAMS];
    red(
        &run(&shared, &inputs.mask, eps),
        "stream 0's key for every stream",
    );
    let zeros = [0.0; HIDDEN];
    let mut unscaled = engram_weights(&o, &table);
    unscaled.stream_scales = vec![&zeros[..]; STREAMS];
    red(
        &run(&unscaled, &inputs.mask, eps),
        "per-stream scales dropped",
    );
    let mut one_norm = engram_weights(&o, &table);
    one_norm.head_norms = vec![one_norm.head_norms[0]; 4];
    red(
        &run(&one_norm, &inputs.mask, eps),
        "head 0's norm for every head",
    );
    let all = vec![true; inputs.mask.len()];
    red(&run(&engram_weights(&o, &table), &all, eps), "mask ignored");
}
