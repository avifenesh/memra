//! CPU-only native-tokenizer preparation for reproducible raw-token HTTP gates.
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::path::Path;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        4,
        "usage: dsv4_http_prompt_pack <model-dir> <source.txt> <tokens>"
    );
    let count: usize = args[3].parse().expect("token count");
    assert!((1..=1_048_448).contains(&count));
    let tokenizer = Tokenizer::from_hf_dir(Path::new(&args[1])).expect("native tokenizer");
    let source = std::fs::read_to_string(&args[2]).expect("real source");
    let mut ids = tokenizer.encode(
        &format!("Review this inference engine source:\n\n{source}"),
        true,
    );
    assert!(
        ids.len() >= count,
        "source too short; no repetition/padding allowed"
    );
    ids.truncate(count);
    let mut hash = Sha256::new();
    for id in &ids {
        hash.update(id.to_le_bytes());
    }
    println!(
        "{{\"source_sha256\":\"{:x}\",\"token_sha256\":\"{:x}\",\"tokens\":{ids:?}}}",
        Sha256::digest(source.as_bytes()),
        hash.finalize()
    );
}
