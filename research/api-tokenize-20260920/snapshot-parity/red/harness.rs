#[path = "/Users/avifen/.codex/worktrees/pr557-tokenizer-snapshot/memra/crates/memra-server/src/worker/tokenizers/fixture.rs"] mod fixture;
#[test]
fn independent_openings_must_have_one_prompt_interpretation() {
    let fixture = fixture::Fixture::new();
    let http = memra_tokenizer::Tokenizer::from_hf_dir(&fixture.0).unwrap();
    fixture.replace_config(false);
    let worker = memra_tokenizer::Tokenizer::from_hf_dir(&fixture.0).unwrap();
    assert_eq!(http.encode("hello", true), worker.encode("hello", true),
               "endpoint and worker token ids must match");
}
