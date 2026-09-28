//! Offline raw-text comparison of the explicit S5 KV experiment.
//! The pinned F32 KV path is a numeric oracle, not a task-quality label.

use std::error::Error;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use memra_engine::Engine;
use memra_engine::mimo_text_weights::MiMoTextWeights;
use memra_gguf::source::SafetensorsSource;
use sha2::{Digest, Sha256};

type Fail = Box<dyn Error>;
const VOCAB: usize = 152_576;
const FOUR_GIB: usize = 4 * 1024 * 1024 * 1024;
const MAX_INPUT: usize = 255;

struct StepPair {
    last: Vec<f32>,
    continuation: Vec<f32>,
}

fn checked_logits(logits: &[f32]) -> Result<(), Fail> {
    if logits.len() != VOCAB || logits.iter().any(|value| !value.is_finite()) {
        return Err("MiMo quality probe returned incomplete logits".into());
    }
    Ok(())
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap()
}

fn digest(logits: &[f32]) -> String {
    let mut hash = Sha256::new();
    for value in logits {
        hash.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn compare(candidate: &[f32], control: &[f32]) -> (f64, f32, usize) {
    let mut squared_diff = 0.0f64;
    let mut squared_control = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut matching_bits = 0usize;
    for (&got, &want) in candidate.iter().zip(control) {
        let difference = (got - want).abs();
        squared_diff += f64::from(difference).powi(2);
        squared_control += f64::from(want).powi(2);
        max_abs = max_abs.max(difference);
        matching_bits += usize::from(got.to_bits() == want.to_bits());
    }
    (
        (squared_diff / squared_control).sqrt(),
        max_abs,
        matching_bits,
    )
}

fn parse_ids(text: &str) -> Result<Vec<(&str, Vec<u32>)>, Fail> {
    let mut prompts = Vec::new();
    for line in text.lines() {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 3 || !matches!(fields[0], "natural" | "code" | "multilingual") {
            return Err("MiMo pinned tokenizer receipt has changed columns or prompt".into());
        }
        let decode = |field: &str| -> Result<Vec<u32>, Fail> {
            Ok(field
                .split(',')
                .map(str::parse::<u32>)
                .collect::<Result<Vec<_>, _>>()?)
        };
        let ids = decode(fields[1])?;
        if ids != decode(fields[2])?
            || ids.is_empty()
            || ids.len() > MAX_INPUT
            || ids.iter().any(|&id| id as usize >= VOCAB)
            || prompts.iter().any(|(name, _)| *name == fields[0])
        {
            return Err("MiMo pinned tokenizer receipt IDs changed".into());
        }
        prompts.push((fields[0], ids));
    }
    if prompts.len() != 3 {
        return Err("MiMo pinned tokenizer receipt omitted a prompt".into());
    }
    prompts.push(("synthetic128", (42..170).collect()));
    Ok(prompts)
}

fn run() -> Result<(), Fail> {
    let mut args = std::env::args().skip(1);
    let root = args
        .next()
        .ok_or("usage: mimo_s5_text_quality_probe <pinned-model-dir> <tokenizer-ref-ids.tsv>")?;
    let ids_path = args
        .next()
        .ok_or("MiMo quality probe requires tokenizer reference IDs")?;
    if args.next().is_some() || std::env::var("MEMRA_BF16_MMV").as_deref() != Ok("1") {
        return Err("MiMo quality probe requires two paths and MEMRA_BF16_MMV=1".into());
    }
    let ids_bytes = fs::read(&ids_path)?;
    let ids_text = std::str::from_utf8(&ids_bytes)?;
    let prompts = parse_ids(ids_text)?;
    let source = Arc::new(SafetensorsSource::open(Path::new(&root))?);
    let cards = [Engine::new(0)?, Engine::new(1)?];
    let engines = [&cards[0], &cards[1]];
    if cards[0].stream().context().ordinal() == cards[1].stream().context().ordinal() {
        return Err("MiMo quality probe requires two distinct target GPUs".into());
    }
    let text = MiMoTextWeights::load(engines, source)?;
    println!("format\tmimo-s5-raw-text-quality-component-v1");
    println!("source\tXiaomiMiMo/MiMo-V2.6-Flash-RL@3b38d063180c3e4aed9691fdc735f3d10b266ee4");
    println!("tokenizer_ref_sha256\t{:x}", Sha256::digest(&ids_bytes));
    println!("arithmetic\tbf16_mmv=1\tf32_kv_vs_q8_nvfp4_vs_q8_s5");
    for (label, ids) in prompts {
        let context = ids.len() + 1;
        let f32 = {
            let mut sequence = text.text_forward(engines)?;
            let mut last = Vec::new();
            for &id in &ids {
                last = sequence.token(id)?;
            }
            let continuation = sequence.token(220)?;
            StepPair { last, continuation }
        };
        let nvfp4 = {
            let mut sequence = text.compressed_text_forward(engines, context, [FOUR_GIB; 2])?;
            let mut last = Vec::new();
            for &id in &ids {
                last = sequence.token(id)?;
            }
            let continuation = sequence.token(220)?;
            StepPair { last, continuation }
        };
        let s5 = {
            let mut sequence =
                text.compressed_text_forward_s5_g16(engines, context, [FOUR_GIB; 2])?;
            let mut last = Vec::new();
            for &id in &ids {
                last = sequence.token(id)?;
            }
            let continuation = sequence.token(220)?;
            StepPair { last, continuation }
        };
        println!(
            "shape\t{label}\tinput_tokens={}\tcontext={context}",
            ids.len()
        );
        for (step_name, control, nvfp4_values, s5_values) in [
            ("last", &f32.last, &nvfp4.last, &s5.last),
            (
                "continuation",
                &f32.continuation,
                &nvfp4.continuation,
                &s5.continuation,
            ),
        ] {
            for logits in [control, nvfp4_values, s5_values] {
                checked_logits(logits)?;
            }
            for (arm, logits) in [("f32", control), ("nvfp4", nvfp4_values), ("s5", s5_values)] {
                let (relative_l2, max_abs, matching_bits) = compare(logits, control);
                println!(
                    "score\t{label}\t{step_name}\t{arm}\targmax={}\trel_l2_vs_f32={relative_l2:.9e}\tmax_abs_vs_f32={max_abs:.9e}\tmatching_bits={matching_bits}\tlogits_sha256={}",
                    argmax(logits),
                    digest(logits),
                );
            }
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_s5_text_quality_probe: {error}");
        std::process::exit(1);
    }
}
