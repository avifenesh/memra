//! Hy4-flash whole-trunk selections against the pinned transformers program, compared before any
//! logit: per forward call (the prefill, then each decode step), every indexed layer's recalled
//! groups per query and every Engram layer's table rows per token must equal the oracle's
//! exactly. The oracle (research/modelplan-onboarding-hy4flash-20261119/tiny/make_tiny_oracle.py)
//! records them as FNV-1a 64 digests of the same text this test builds. The logits are the CLI
//! tiny gate's (`memra model verify tiny --against hy_v4_flash`).

use std::path::Path;

use memra_reference::hy_v4_flash_trunk::{self, TraceCall};
use serde_json::Value;

fn fnv1a(text: &str) -> String {
    let hash = text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    });
    format!("{hash:016x}")
}

fn record(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../research/modelplan-onboarding-hy4flash-20261119/tiny")
        .join(format!("trace-{name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let record: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        record["transformers_modeling_blob"],
        "2d796211a4f867bd4c46d4a952cfee0375929f72"
    );
    assert_eq!(record["dtype"], "float64");
    assert_eq!(record["raw_constants"], false);
    record
}

/// `(layer, digest)` pairs.
type Digests = Vec<(String, String)>;

/// The digests of one call (recalled groups, Engram rows), keyed by layer, as the oracle writes
/// them.
fn digests(call: &TraceCall) -> (Digests, Digests) {
    let recalled = call
        .recalled
        .iter()
        .map(|(layer, lists)| {
            let text: Vec<String> = lists
                .iter()
                .map(|groups| {
                    let mut groups = groups.clone();
                    groups.sort_unstable();
                    groups
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .collect();
            (layer.to_string(), fnv1a(&text.join("\n")))
        })
        .collect();
    let rows = call
        .engram_rows
        .iter()
        .map(|(layer, rows)| {
            let text: Vec<String> = rows.iter().map(u64::to_string).collect();
            (layer.to_string(), fnv1a(&text.join(",")))
        })
        .collect();
    (recalled, rows)
}

fn sorted(object: &Value) -> Digests {
    let mut out: Digests = object
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
        .collect();
    out.sort();
    out
}

#[test]
fn indexer_selections_and_engram_rows_equal_the_transformers_program_exactly() {
    let pack = memra_gguf::model_packs::by_alias("hy_v4_flash").unwrap();
    let plan = pack.compile_tiny_plan().unwrap();
    let mut runs = vec![(
        "main".to_string(),
        memra_reference::deterministic_fixture(&plan).unwrap(),
    )];
    for variant in memra_reference::tiny_variants(&plan).unwrap() {
        assert_eq!(variant.plan, plan, "the variants split the same program");
        runs.push((variant.name.to_string(), variant.fixture));
    }
    let mut indexed_calls = 0;
    for (name, fixture) in &runs {
        let oracle = record(name);
        let ids: Vec<u32> = oracle["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(
            ids, fixture.token_ids,
            "{name}: the oracle ran these tokens"
        );
        assert_eq!(
            oracle["decode_from"].as_u64().map(|v| v as usize),
            fixture.decode_from
        );
        let (_, trace) = hy_v4_flash_trunk::execute_traced(
            &plan,
            &fixture.weights,
            &fixture.token_ids,
            fixture.decode_from,
        )
        .unwrap();
        let calls = oracle["calls"].as_array().unwrap();
        assert_eq!(
            calls.len(),
            trace.len(),
            "{name}: one prefill, then one call per step"
        );
        for (call, theirs) in trace.iter().zip(calls) {
            assert_eq!(theirs["start"].as_u64().unwrap() as usize, call.start);
            assert_eq!(theirs["tokens"].as_u64().unwrap() as usize, call.tokens);
            let (mut recalled, mut rows) = digests(call);
            recalled.sort();
            rows.sort();
            assert_eq!(
                recalled,
                sorted(&theirs["recalled"]),
                "{name}: recalled groups of the call at {}",
                call.start
            );
            assert_eq!(
                rows,
                sorted(&theirs["engram_rows"]),
                "{name}: Engram rows of the call at {}",
                call.start
            );
            indexed_calls += recalled.len();
        }
    }
    // The main prefill and its 8 steps (3 indexed layers each), split250's 24 steps past the
    // budget, split500's prefill and 30 steps.
    assert_eq!(indexed_calls, 27 + 72 + 93);
}

/// The digests see a selection off by one group: a budget one group short (252 instead of 256)
/// recalls other groups on every indexed layer of the prefill.
#[test]
fn a_budget_one_group_short_changes_the_selection_digests() {
    let pack = memra_gguf::model_packs::by_alias("hy_v4_flash").unwrap();
    let plan = pack.compile_tiny_plan().unwrap();
    let fixture = memra_reference::deterministic_fixture(&plan).unwrap();
    let oracle = record("main");
    let prefill = &oracle["calls"][0]["recalled"];
    let mut fewer = plan.clone();
    for layer in &mut fewer.layers {
        if let memra_gguf::model_plan::AttentionPlan::HyV4(a) = &mut layer.attention
            && let memra_gguf::model_plan::HyV4AttentionKind::Indexed(index) = &mut a.kind
        {
            index.top_k -= index.pool;
        }
    }
    let (_, trace) = hy_v4_flash_trunk::execute_traced(
        &fewer,
        &fixture.weights,
        &fixture.token_ids,
        fixture.decode_from,
    )
    .unwrap();
    let (recalled, _) = digests(&trace[0]);
    for (layer, digest) in recalled {
        assert_ne!(
            prefill[&layer].as_str().unwrap(),
            digest,
            "layer {layer}: one group fewer selects the same groups"
        );
    }
}
