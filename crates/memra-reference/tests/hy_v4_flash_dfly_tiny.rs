//! The Hy4-flash DFly2 drafter (`memra_reference::hy_v4_flash_dfly`) against the pinned vLLM
//! program on the tiny fixture: the reference trunk's state of the main request, then a draft
//! round from each of the oracle's 120 anchors (prefill positions on both sides of the indexer
//! budget, the decode steps). The oracle
//! (research/modelplan-onboarding-hy4flash-20261119/dfly/make_dfly_oracle.py, float64, over the
//! target state the tiny trunk oracle dumped) is pinned in `dfly/tiny/oracle/`. Each wrong
//! drafter program fails the band; a geometry whose MLP width is the trunk's (the vendor
//! config's default) reads no weight.

use std::path::Path;

use memra_reference::hy_v4_flash_dfly::{DflyRed, tiny_rounds};
use serde_json::Value;

/// The tiny gate's band (the pack's `checkpoint_parity` tolerance of step 6).
const ATOL: f64 = 1e-5;

fn oracle() -> (Value, Vec<f32>) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../research/modelplan-onboarding-hy4flash-20261119/dfly/tiny/oracle");
    let tape: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("tape.json")).unwrap()).unwrap();
    assert_eq!(tape["format"], "memra-hy4f-dfly-tape-v1");
    assert_eq!(tape["backend_dtype"], "f64");
    assert!(tape["red_arm"].is_null() && tape["forced_drafts"] == false);
    assert_eq!(
        tape["vllm_pins"]["vllm/models/hy_v4/nvidia/dfly.py"],
        "c7956b89e1b72cbb6a55143ba4123855369fd0de"
    );
    let logits = std::fs::read(dir.join("logits.f32"))
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (tape, logits)
}

fn anchors(tape: &Value) -> Vec<usize> {
    tape["anchors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}

/// The largest logit move against the oracle and the steps whose argmax differs.
fn moved(red: DflyRed) -> (f64, usize, Vec<Vec<u32>>) {
    let (tape, want) = oracle();
    let (tokens, k, _, rounds) = tiny_rounds(&anchors(&tape), red).unwrap();
    assert_eq!(
        tokens,
        tape["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>()
    );
    assert_eq!(k as u64, tape["k"].as_u64().unwrap());
    let vocab = tape["vocab"].as_u64().unwrap() as usize;
    let got: Vec<f32> = rounds.iter().flat_map(|r| r.logits.clone()).collect();
    assert_eq!(got.len(), want.len());
    let max = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .fold(0.0, f64::max);
    let argmax = |row: &[f32]| memra_reference::hy_v4_flash_dfly::argmax(row);
    let differ = got
        .chunks(vocab)
        .zip(want.chunks(vocab))
        .filter(|(a, b)| argmax(a) != argmax(b))
        .count();
    (max, differ, rounds.into_iter().map(|r| r.drafts).collect())
}

#[test]
fn hy_v4_flash_dfly_tiny_matches_the_pinned_vllm_program() {
    let (tape, _) = oracle();
    let (max, differ, drafts) = moved(DflyRed::None);
    assert!(max <= ATOL, "the drafter moves a logit by {max}");
    assert_eq!(differ, 0);
    let want: Vec<Vec<u32>> = serde_json::from_value(tape["drafts"].clone()).unwrap();
    assert_eq!(drafts, want, "the drafts are the oracle's");
}

#[test]
fn hy_v4_flash_dfly_tiny_refuses_wrong_drafter_programs() {
    for red in [
        DflyRed::Layers,
        DflyRed::Width,
        DflyRed::NoCorrection,
        DflyRed::TopK,
    ] {
        let (max, differ, _) = moved(red);
        assert!(
            max > 1000.0 * ATOL && differ > 0,
            "{red:?} stays within the band: {max}, {differ} argmax moves"
        );
    }
}

#[test]
fn a_drafter_mlp_at_the_trunks_width_reads_no_weight() {
    use memra_gguf::model_packs::hy_v4_flash as pack;
    use memra_reference::hy_v4_flash_dfly::{DflyGeometry, DflyWeights};
    let fixture = memra_reference::hy_v4_flash_trunk::dfly_fixture().unwrap();
    let hy = &fixture.hy;
    let read = DflyWeights::read(hy.dfly.num_hidden_layers as usize, |name| {
        fixture
            .weights
            .get(&pack::semantic_tensor_id(name))
            .map(|t| t.data.iter().map(|v| f64::from(*v)).collect())
            .ok_or_else(|| name.to_string())
    })
    .unwrap();
    let width = pack::dfly_ffn_width(hy) as usize;
    assert_eq!(width, 32, "the tiny config declares its drafter's width");
    read.check(&DflyGeometry::of(hy, width)).unwrap();
    // the vendor derivation's default when nothing is declared: the trunk's intermediate_size
    let err = read.check(&DflyGeometry::of(hy, 48)).unwrap_err();
    assert!(err.contains("mlp.gate_proj"), "{err}");
}
