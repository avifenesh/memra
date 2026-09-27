//! One pinned MiMo text token followed by three greedy MTP3 draft depths.
//! This observes target-to-draft hidden handoff, not speculative acceptance.

use std::path::Path;
use std::sync::Arc;

use memra_engine::Engine;
use memra_engine::mimo_mtp_weights::Mtp3Weights;
use memra_engine::mimo_text_weights::MiMoTextWeights;
use memra_gguf::source::SafetensorsSource;
use sha2::{Digest, Sha256};

type Fail = Box<dyn std::error::Error>;
const FOUR_GIB: usize = 4 * 1024 * 1024 * 1024;
const VOCAB: usize = 152_576;

fn digest(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn argmax(values: &[f32]) -> Result<usize, &'static str> {
    if values.len() != VOCAB || values.iter().any(|value| !value.is_finite()) {
        return Err("MiMo target or draft logits are incomplete");
    }
    let mut best = 0;
    for index in 1..values.len() {
        if values[index].total_cmp(&values[best]).is_gt() {
            best = index;
        }
    }
    Ok(best)
}

fn run() -> Result<(), Fail> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or("usage: mimo_target_mtp3_probe <source_dir>")?;
    if args.next().is_some() {
        return Err("usage: mimo_target_mtp3_probe <source_dir>".into());
    }
    let source = Arc::new(SafetensorsSource::open(Path::new(&dir))?);
    let cards = [Engine::new(0)?, Engine::new(1)?];
    let text = MiMoTextWeights::load([&cards[0], &cards[1]], source.clone())?;
    let mtp = Mtp3Weights::load(&cards[0], source)?;
    let mut target = text.compressed_text_forward([&cards[0], &cards[1]], 256, [FOUR_GIB; 2])?;
    let target_step = target.token_with_hidden(42)?;
    if target_step.position != 0
        || target.position() != 1
        || target_step.hidden_before_norm.len() != 4_096
    {
        return Err("MiMo target hidden handoff is incomplete".into());
    }

    println!("format\tmemra-mimo-target-mtp3-probe-v1");
    println!("input\tforced_target_token_42");
    println!("target_position\t{}", target_step.position);
    println!("target_argmax\t{}", argmax(&target_step.logits)?);
    println!("target_logits_sha256\t{}", digest(&target_step.logits));
    println!(
        "target_hidden_sha256\t{}",
        digest(&target_step.hidden_before_norm)
    );

    let mut draft = mtp.draft_forward(&text, &cards[0], &cards[1])?;
    let mut input_token = 42u32;
    let mut hidden = target_step.hidden_before_norm;
    for depth in 0..3 {
        let step = draft.token(depth, 0, input_token, &hidden)?;
        if step.depth != depth
            || step.position != 0
            || step.hidden_before_norm.len() != 4_096
            || draft.position(depth)? != 1
        {
            return Err("MiMo target-to-MTP3 draft depth is incomplete".into());
        }
        let proposed = argmax(&step.logits)? as u32;
        println!(
            "depth\t{depth}\tinput_token\t{input_token}\tproposed_token\t{proposed}\thidden_sha256\t{}\tlogits_sha256\t{}",
            digest(&step.hidden_before_norm),
            digest(&step.logits)
        );
        input_token = proposed;
        hidden = step.hidden_before_norm;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_target_mtp3_probe: {error}");
        std::process::exit(1);
    }
}
