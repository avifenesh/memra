//! Pinned MiMo MTP3 draft step over synthetic target hidden rows. This
//! exercises the two-card source weights, not verification or serving.

use std::path::Path;
use std::sync::Arc;

use memra_engine::Engine;
use memra_engine::mimo_mtp_weights::Mtp3Weights;
use memra_engine::mimo_text_weights::MiMoTextWeights;
use memra_gguf::source::SafetensorsSource;
use sha2::{Digest, Sha256};

type Fail = Box<dyn std::error::Error>;
const HIDDEN: usize = 4_096;
const VOCAB: usize = 152_576;

fn digest(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn run() -> Result<(), Fail> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or("usage: mimo_mtp3_source_probe <source_dir>")?;
    if args.next().is_some() {
        return Err("usage: mimo_mtp3_source_probe <source_dir>".into());
    }
    let source = Arc::new(SafetensorsSource::open(Path::new(&dir))?);
    let cards = [Engine::new(0)?, Engine::new(1)?];
    let text = MiMoTextWeights::load([&cards[0], &cards[1]], source.clone())?;
    let mtp = Mtp3Weights::load(&cards[0], source)?;
    let mut draft = mtp.draft_forward(&text, &cards[0], &cards[1])?;
    let target_hidden = (0..HIDDEN)
        .map(|index| ((index * 7 % 67) as f32 - 33.0) / 64.0)
        .collect::<Vec<_>>();

    println!("format\tmemra-mimo-mtp3-source-probe-v1");
    println!("draft_gpu\t0");
    println!("shared_head_gpu\t1");
    println!("input\tforced_token_42_synthetic_hidden");
    for depth in 0..3 {
        let step = draft.token(depth, 0, 42, &target_hidden)?;
        if step.depth != depth
            || step.position != 0
            || step.hidden_before_norm.len() != HIDDEN
            || step.logits.len() != VOCAB
            || draft.position(depth)? != 1
        {
            return Err("MiMo MTP3 source draft returned incomplete rows or position".into());
        }
        let argmax = step
            .logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .ok_or("MiMo MTP3 draft has no logits")?;
        println!(
            "depth\t{depth}\targmax\t{argmax}\thidden_sha256\t{}\tlogits_sha256\t{}",
            digest(&step.hidden_before_norm),
            digest(&step.logits)
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_mtp3_source_probe: {error}");
        std::process::exit(1);
    }
}
