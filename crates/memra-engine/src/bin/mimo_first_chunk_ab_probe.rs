//! Paired offline first-chunk timing on the pinned two-card MiMo source.
//! Source weights stay resident; each arm allocates a fresh short KV sequence.
//! This is not an API, task-quality, million-prefill, or serving benchmark.

use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use memra_engine::Engine;
use memra_engine::mimo_text_weights::MiMoTextWeights;
use memra_gguf::source::SafetensorsSource;
use sha2::{Digest, Sha256};

type Fail = Box<dyn Error>;
const TOKENS: usize = 128;
const CONTEXT: usize = TOKENS + 1;
const VOCAB: usize = 152_576;
const FOUR_GIB: usize = 4 * 1024 * 1024 * 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Serial,
    Batch,
}

impl Arm {
    fn label(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::Batch => "batch",
        }
    }
}

struct Sample {
    arm: Arm,
    allocation_ms: f64,
    prefill_ms: f64,
    continuation_ms: f64,
    logits: Vec<f32>,
    continuation: Vec<f32>,
}

fn digest(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn argmax(logits: &[f32]) -> Result<usize, &'static str> {
    if logits.len() != VOCAB || logits.iter().any(|value| !value.is_finite()) {
        return Err("MiMo first-chunk logits are incomplete or non-finite");
    }
    Ok(logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap())
}

fn compare(candidate: &[f32], control: &[f32]) -> (f32, f64) {
    let mut max_abs = 0.0f32;
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for (&got, &want) in candidate.iter().zip(control) {
        max_abs = max_abs.max((got - want).abs());
        numerator += f64::from(got - want).powi(2);
        denominator += f64::from(want).powi(2);
    }
    (max_abs, (numerator / denominator).sqrt())
}

fn median_four(mut values: Vec<f64>) -> Result<f64, &'static str> {
    if values.len() != 4
        || values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("MiMo first-chunk AB timing arm is incomplete");
    }
    values.sort_by(f64::total_cmp);
    Ok((values[1] + values[2]) / 2.0)
}

fn run() -> Result<(), Fail> {
    let mut args = std::env::args().skip(1);
    let root = args
        .next()
        .ok_or("usage: mimo_first_chunk_ab_probe <pinned-source-dir>")?;
    if args.next().is_some() {
        return Err("usage: mimo_first_chunk_ab_probe <pinned-source-dir>".into());
    }
    let source = Arc::new(SafetensorsSource::open(Path::new(&root))?);
    let cards = [Engine::new(0)?, Engine::new(1)?];
    let engines = [&cards[0], &cards[1]];
    if cards[0].stream().context().ordinal() == cards[1].stream().context().ordinal() {
        return Err("MiMo first-chunk AB needs distinct source GPUs".into());
    }
    let text = MiMoTextWeights::load(engines, source)?;
    let token_ids = (42..42 + TOKENS as u32).collect::<Vec<_>>();
    let prepared = text.modal_embedding_gpu_chunk(&cards[0], &token_ids, &[], &[], &[])?;
    if prepared.requires_payload_identity() || prepared.token_count() != TOKENS {
        return Err("MiMo first-chunk AB text input changed".into());
    }
    let mut token_hasher = Sha256::new();
    for &id in &token_ids {
        token_hasher.update(id.to_le_bytes());
    }
    println!("format\tmemra-mimo-first-chunk-offline-ab-v1");
    println!("source\tXiaomiMiMo/MiMo-V2.6-Flash-RL@3b38d063180c3e4aed9691fdc735f3d10b266ee4");
    println!("shape\tfresh_text_tokens={TOKENS}\tmax_context={CONTEXT}\tcontinuation_token=220");
    println!("token_ids_sha256\t{:x}", token_hasher.finalize());
    println!("timing\tweights_and_embeddings_preloaded\tfresh_kv_allocation_separate");

    let order = [
        Arm::Serial,
        Arm::Batch,
        Arm::Serial,
        Arm::Batch,
        Arm::Batch,
        Arm::Serial,
        Arm::Serial,
        Arm::Batch,
        Arm::Batch,
        Arm::Serial,
    ];
    let mut samples = Vec::with_capacity(order.len());
    for (index, arm) in order.into_iter().enumerate() {
        let start = Instant::now();
        let mut sequence = text.compressed_text_forward(engines, CONTEXT, [FOUR_GIB; 2])?;
        let allocation_ms = start.elapsed().as_secs_f64() * 1e3;
        let start = Instant::now();
        let step = match arm {
            Arm::Serial => sequence.consume_embedding_chunk(&prepared)?,
            Arm::Batch => sequence.consume_embedding_chunk_batched(&prepared)?,
        };
        let prefill_ms = start.elapsed().as_secs_f64() * 1e3;
        let start = Instant::now();
        let continuation = sequence.token(220)?;
        let continuation_ms = start.elapsed().as_secs_f64() * 1e3;
        if sequence.position() != CONTEXT || step.position != TOKENS - 1 {
            return Err("MiMo first-chunk AB sequence cursor drifted".into());
        }
        println!(
            "sample\t{index}\t{}\t{}\t{allocation_ms:.6}\t{prefill_ms:.6}\t{continuation_ms:.6}\t{}\t{}\t{}\t{}",
            arm.label(),
            if index < 2 { "warmup" } else { "measure" },
            argmax(&step.logits)?,
            digest(&step.logits),
            argmax(&continuation)?,
            digest(&continuation),
        );
        samples.push(Sample {
            arm,
            allocation_ms,
            prefill_ms,
            continuation_ms,
            logits: step.logits,
            continuation,
        });
    }
    for arm in [Arm::Serial, Arm::Batch] {
        let rows = samples
            .iter()
            .filter(|sample| sample.arm == arm)
            .collect::<Vec<_>>();
        if rows.len() != 5 {
            return Err("MiMo first-chunk AB has an incomplete arm".into());
        }
        let first = (digest(&rows[0].logits), digest(&rows[0].continuation));
        if rows.iter().any(|sample| {
            digest(&sample.logits) != first.0 || digest(&sample.continuation) != first.1
        }) {
            return Err("MiMo first-chunk AB was not deterministic within an arm".into());
        }
        println!("replay\t{}\tbyte_identical", arm.label());
        let measured = &rows[1..];
        for (label, timings) in [
            (
                "allocation",
                measured
                    .iter()
                    .map(|sample| sample.allocation_ms)
                    .collect::<Vec<_>>(),
            ),
            (
                "prefill",
                measured
                    .iter()
                    .map(|sample| sample.prefill_ms)
                    .collect::<Vec<_>>(),
            ),
            (
                "continuation",
                measured
                    .iter()
                    .map(|sample| sample.continuation_ms)
                    .collect::<Vec<_>>(),
            ),
        ] {
            println!(
                "median_ms\t{}\t{label}\t{:.6}",
                arm.label(),
                median_four(timings)?
            );
        }
    }
    let control = samples
        .iter()
        .find(|sample| sample.arm == Arm::Serial)
        .unwrap();
    let candidate = samples
        .iter()
        .find(|sample| sample.arm == Arm::Batch)
        .unwrap();
    for (label, batch, serial) in [
        ("last", &candidate.logits, &control.logits),
        (
            "continuation",
            &candidate.continuation,
            &control.continuation,
        ),
    ] {
        let (max_abs, relative_l2) = compare(batch, serial);
        println!(
            "numeric\t{label}\tmax_abs={max_abs:.9e}\trelative_l2={relative_l2:.9e}\tbatch_argmax={}\tserial_argmax={}",
            argmax(batch)?,
            argmax(serial)?,
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_first_chunk_ab_probe: {error}");
        std::process::exit(1);
    }
}
