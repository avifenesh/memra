use memra_gguf::model_packs;
use memra_reference::{deterministic_fixture, execute};

#[test]
fn mimo_source_text_fixture_executes_deterministically() {
    let pack = model_packs::by_alias("mimo_v2_source").unwrap();
    assert!(pack.support.is_none());
    assert!(pack.compile_tiny_plan().is_err());
    let plan = model_packs::mimo_v2::tiny_text_plan().unwrap();
    let fixture = deterministic_fixture(&plan).unwrap();
    let first = execute(&plan, &fixture.weights, &fixture.token_ids).unwrap();
    let second = execute(&plan, &fixture.weights, &fixture.token_ids).unwrap();
    assert_eq!(first, second);
    assert!(first.logits.iter().all(|value| value.is_finite()));
}
