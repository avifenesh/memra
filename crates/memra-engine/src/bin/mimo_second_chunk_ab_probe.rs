//! Paired offline timing for a MiMo second chunk on a live native KV sequence.
//! Source weights stay resident; each arm builds the same 128-token cached
//! prefix, then measures a fresh 128-token second chunk and one decode step.
//! This is not a serving or long-context throughput benchmark.

use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use memra_engine::Engine;
use memra_engine::mimo_text_forward::MiMoTextStep;
use memra_engine::mimo_text_weights::MiMoTextWeights;
use memra_gguf::source::SafetensorsSource;
use sha2::{Digest, Sha256};

type Fail = Box<dyn Error>;
const FIRST: usize = 128;
const SECOND: usize = 128;
const CONTEXT: usize = FIRST + SECOND + 1;
const HIDDEN: usize = 4_096;
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
    prefix_ms: f64,
    second_ms: f64,
    continuation_ms: f64,
    step: MiMoTextStep,
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
        return Err("MiMo second-chunk logits are incomplete or non-finite");
    }
    Ok(logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap())
}

fn same_bits(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn median_four(mut values: Vec<f64>) -> Result<f64, &'static str> {
    if values.len() != 4
        || values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("MiMo second-chunk timing arm is incomplete");
    }
    values.sort_by(f64::total_cmp);
    Ok((values[1] + values[2]) / 2.0)
}

fn run() -> Result<(), Fail> {
    let mut args = std::env::args().skip(1);
    let root = args
        .next()
        .ok_or("usage: mimo_second_chunk_ab_probe <pinned-source-dir>")?;
    if args.next().is_some() {
        return Err("usage: mimo_second_chunk_ab_probe <pinned-source-dir>".into());
    }
    let source = Arc::new(SafetensorsSource::open(Path::new(&root))?);
    let cards = [Engine::new(0)?, Engine::new(1)?];
    let engines = [&cards[0], &cards[1]];
    if cards[0].stream().context().ordinal() == cards[1].stream().context().ordinal() {
        return Err("MiMo second-chunk AB needs distinct source GPUs".into());
    }
    let text = MiMoTextWeights::load(engines, source)?;
    let first_ids = (42..42 + FIRST as u32).collect::<Vec<_>>();
    let second_ids = (42 + FIRST as u32..42 + (FIRST + SECOND) as u32).collect::<Vec<_>>();
    let first = text.modal_embedding_gpu_chunk(&cards[0], &first_ids, &[], &[], &[])?;
    let second = text.modal_embedding_gpu_chunk(&cards[0], &second_ids, &[], &[], &[])?;
    let mut token_hasher = Sha256::new();
    for &id in first_ids.iter().chain(&second_ids) {
        token_hasher.update(id.to_le_bytes());
    }
    println!("format\tmemra-mimo-second-chunk-offline-ab-v1");
    println!("source\tXiaomiMiMo/MiMo-V2.6-Flash-RL@3b38d063180c3e4aed9691fdc735f3d10b266ee4");
    println!(
        "shape\tcached_first={FIRST}\tfresh_second={SECOND}\tmax_context={CONTEXT}\tcontinuation_token=220"
    );
    println!("token_ids_sha256\t{:x}", token_hasher.finalize());
    println!("timing\tweights_preloaded\tfresh_kv_allocation_and_prefix_separate");

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
        let started = Instant::now();
        let mut sequence = text.compressed_text_forward(engines, CONTEXT, [FOUR_GIB; 2])?;
        let allocation_ms = started.elapsed().as_secs_f64() * 1e3;
        let started = Instant::now();
        let prefix = sequence.consume_embedding_chunk_batched(&first)?;
        let prefix_ms = started.elapsed().as_secs_f64() * 1e3;
        if prefix.position != FIRST - 1 || sequence.position() != FIRST {
            return Err("MiMo second-chunk AB cached prefix cursor drifted".into());
        }
        let started = Instant::now();
        let step = match arm {
            Arm::Serial => sequence.consume_embedding_chunk(&second)?,
            Arm::Batch => sequence.consume_embedding_chunk_batched(&second)?,
        };
        let second_ms = started.elapsed().as_secs_f64() * 1e3;
        let started = Instant::now();
        let continuation = sequence.token(220)?;
        let continuation_ms = started.elapsed().as_secs_f64() * 1e3;
        if sequence.position() != CONTEXT
            || step.position != FIRST + SECOND - 1
            || step.hidden_before_norm.len() != HIDDEN
        {
            return Err("MiMo second-chunk AB continuing cursor or hidden drifted".into());
        }
        println!(
            "sample\t{index}\t{}\t{}\t{allocation_ms:.6}\t{prefix_ms:.6}\t{second_ms:.6}\t{continuation_ms:.6}\t{}\t{}\t{}\t{}\t{}",
            arm.label(),
            if index < 2 { "warmup" } else { "measure" },
            argmax(&step.logits)?,
            digest(&step.logits),
            digest(&step.hidden_before_norm),
            argmax(&continuation)?,
            digest(&continuation),
        );
        samples.push(Sample {
            arm,
            allocation_ms,
            prefix_ms,
            second_ms,
            continuation_ms,
            step,
            continuation,
        });
    }
    for arm in [Arm::Serial, Arm::Batch] {
        let rows = samples
            .iter()
            .filter(|sample| sample.arm == arm)
            .collect::<Vec<_>>();
        if rows.len() != 5 {
            return Err("MiMo second-chunk AB has an incomplete arm".into());
        }
        let first = rows[0];
        if rows.iter().any(|row| {
            !same_bits(&row.step.logits, &first.step.logits)
                || !same_bits(&row.step.hidden_before_norm, &first.step.hidden_before_norm)
                || !same_bits(&row.continuation, &first.continuation)
        }) {
            return Err("MiMo second-chunk AB changed on byte-identical replay".into());
        }
        println!("replay\t{}\tbyte_identical", arm.label());
        let measured = &rows[1..];
        for (label, timings) in [
            (
                "allocation",
                measured.iter().map(|sample| sample.allocation_ms).collect(),
            ),
            (
                "prefix",
                measured.iter().map(|sample| sample.prefix_ms).collect(),
            ),
            (
                "second",
                measured.iter().map(|sample| sample.second_ms).collect(),
            ),
            (
                "continuation",
                measured
                    .iter()
                    .map(|sample| sample.continuation_ms)
                    .collect(),
            ),
        ] {
            println!(
                "median_ms\t{}\t{label}\t{:.6}",
                arm.label(),
                median_four(timings)?
            );
        }
    }
    let serial = samples
        .iter()
        .find(|sample| sample.arm == Arm::Serial)
        .unwrap();
    let batch = samples
        .iter()
        .find(|sample| sample.arm == Arm::Batch)
        .unwrap();
    for (label, got, want) in [
        (
            "last",
            batch.step.logits.as_slice(),
            serial.step.logits.as_slice(),
        ),
        (
            "hidden",
            batch.step.hidden_before_norm.as_slice(),
            serial.step.hidden_before_norm.as_slice(),
        ),
        (
            "continuation",
            batch.continuation.as_slice(),
            serial.continuation.as_slice(),
        ),
    ] {
        if !same_bits(got, want) {
            return Err(format!("MiMo second-chunk {label} differs from serial").into());
        }
        println!("numeric\t{label}\tbit_identical\tf32_values={}", got.len());
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_second_chunk_ab_probe: {error}");
        std::process::exit(1);
    }
}
