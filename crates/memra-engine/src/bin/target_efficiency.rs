//! Standalone MoESD target-model efficiency study harness.
//!
//! Usage: target-efficiency <model.gguf> [--b 1,2,...] [--gamma 1,2,...] [--runs N]
//!
//! The timed region is one target-model forward over B*gamma rows. Expert-union telemetry is
//! collected by replaying the identical rows after restoring the caches, so router D2H and
//! synchronization are never charged to T_T(B,gamma). All output on stdout is JSONL; load and
//! progress diagnostics go to stderr.

use memra_engine::Engine;
use memra_engine::cache::{Cache, CacheSnapshot};
use memra_engine::forward::argmax;
use memra_engine::hybrid::{Ffn, HybridModel};
use memra_engine::moesd::{self, MoesdLayerUnion};
use memra_gguf::GgufFile;
use std::io::Write;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_BATCHES: &[usize] = &[1, 2, 4, 8, 16, 24, 32];
const DEFAULT_GAMMAS: &[usize] = &[1, 2, 3, 4, 6, 8];

fn parse_list(value: &str, name: &str) -> Result<Vec<usize>, String> {
    let parsed: Vec<usize> = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .map_err(|err| format!("invalid {name} value {part:?}: {err}"))
        })
        .collect::<Result<_, _>>()?;
    if parsed.is_empty() || parsed.contains(&0) {
        return Err(format!("{name} must contain positive integers"));
    }
    Ok(parsed)
}

fn option_value(args: &[String], name: &str) -> Result<Option<String>, String> {
    let mut found = None;
    let mut index = 0;
    while index < args.len() {
        if args[index] == name {
            let value = args
                .get(index + 1)
                .ok_or_else(|| format!("{name} requires a value"))?;
            if found.replace(value.clone()).is_some() {
                return Err(format!("{name} was supplied more than once"));
            }
            index += 2;
        } else {
            index += 1;
        }
    }
    Ok(found)
}

fn unix_ms() -> Result<u128, Box<dyn std::error::Error>> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())
}

fn accepted_sum(gamma: usize) -> Result<f64, Box<dyn std::error::Error>> {
    // Frozen Step-3.7 PP-2 acceptance proxies from DESIGN.md. These are candidate-position
    // sums, not a re-fit from this public target-timing sweep.
    match gamma {
        1 => Ok(1.0),
        2 => Ok(0.737),
        3 => Ok(0.655 + 0.388),
        4 => Ok(0.676 + 0.384 + 0.048),
        6 => Ok(0.676 + 0.384 + 0.048 * 3.0),
        8 => Ok(0.676 + 0.384 + 0.048 * 5.0),
        _ => Err(format!("gamma={gamma} has no frozen acceptance proxy").into()),
    }
}

fn worker_device() -> Result<usize, Box<dyn std::error::Error>> {
    // Match memra-server: under PP the primary follows the final/head stage.
    let Some(devices) = std::env::var("MEMRA_PP_DEVICES")
        .ok()
        .filter(|value| !value.trim().is_empty())
    else {
        return Ok(0);
    };
    devices
        .split(',')
        .next_back()
        .ok_or("MEMRA_PP_DEVICES is empty")?
        .trim()
        .parse::<usize>()
        .map_err(|err| format!("invalid final MEMRA_PP_DEVICES entry: {err}").into())
}

fn restore(
    e: &Engine,
    model: &HybridModel,
    caches: &mut [Cache],
    snapshots: &[CacheSnapshot],
    count: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    for (cache, snapshot) in caches.iter_mut().zip(snapshots).take(count) {
        memra_engine::pp::restore_cache_checkpoint(e, model, None, cache, snapshot)?;
    }
    Ok(())
}

fn forward(
    e: &Engine,
    model: &HybridModel,
    caches: &mut [Cache],
    continuations: &[Vec<u32>],
    batch: usize,
    gamma: usize,
) -> Result<cudarc::driver::CudaSlice<f32>, Box<dyn std::error::Error>> {
    let tokens: Vec<u32> = continuations[..batch]
        .iter()
        .flat_map(|tokens| tokens[..gamma].iter().copied())
        .collect();
    let mut cache_refs: Vec<&mut Cache> = caches[..batch].iter_mut().collect();
    model.moesd_target_forward(e, &tokens, batch, gamma, &mut cache_refs)
}

fn validate_layers(
    layers: &[MoesdLayerUnion],
    expected_layers: usize,
    rows: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if layers.len() != expected_layers {
        return Err(format!(
            "captured {} MoE layers, expected {expected_layers}",
            layers.len(),
        )
        .into());
    }
    for layer in layers {
        let expected_assignments = rows * layer.n_used;
        if layer.assignments != expected_assignments {
            return Err(format!(
                "layer {} captured {} assignments, expected {expected_assignments}",
                layer.id, layer.assignments,
            )
            .into());
        }
        if layer.union_size == 0 || layer.union_size > layer.n_expert {
            return Err(format!(
                "layer {} has impossible expert union {}/{}",
                layer.id, layer.union_size, layer.n_expert,
            )
            .into());
        }
    }
    Ok(())
}

fn identity_check(
    e: &Engine,
    model: &HybridModel,
    caches: &mut [Cache],
    snapshots: &[CacheSnapshot],
    continuations: &[Vec<u32>],
    max_gamma: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let token = continuations[0][0];
    let reference = {
        let mut refs = [&mut caches[0]];
        model.decode_step_batch(e, &[token], &mut refs)?.remove(0)
    };
    restore(e, model, caches, snapshots, 1)?;
    let candidate_d = forward(e, model, caches, continuations, 1, 1)?;
    let candidate = e.dtoh(&candidate_d)?;
    drop(candidate_d);
    restore(e, model, caches, snapshots, 1)?;
    if reference.len() != candidate.len()
        || reference
            .iter()
            .zip(&candidate)
            .any(|(left, right)| left.to_bits() != right.to_bits())
    {
        let max_diff = reference
            .iter()
            .zip(&candidate)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        return Err(format!(
            "MoESD B=1 gamma=1 identity check failed: ref={} candidate={} max_diff={max_diff:.6e}",
            reference.len(),
            candidate.len(),
        )
        .into());
    }
    eprintln!("[target-efficiency] B=1 gamma=1 identity: BIT-IDENTICAL");

    // Prove that the diagnostic row-to-cache mapping is causal before timing it. The
    // serving path already gates independent-session batching; this check covers the new
    // dimension: gamma consecutive rows sharing one session cache. Batched projection
    // arithmetic may differ below the token decision, so require the serving contract
    // (argmax identity) and report the largest logits delta for diagnosis.
    let mut reference_rows = Vec::with_capacity(max_gamma);
    for &token in continuations[0].iter().take(max_gamma) {
        let mut refs = [&mut caches[0]];
        reference_rows.push(model.decode_step_batch(e, &[token], &mut refs)?.remove(0));
    }
    restore(e, model, caches, snapshots, 1)?;
    let packed_d = forward(e, model, caches, continuations, 1, max_gamma)?;
    let packed = e.dtoh(&packed_d)?;
    drop(packed_d);
    restore(e, model, caches, snapshots, 1)?;
    let n_vocab = reference_rows
        .first()
        .ok_or("causal identity reference is empty")?
        .len();
    if n_vocab == 0
        || reference_rows.iter().any(|row| row.len() != n_vocab)
        || packed.len() != max_gamma * n_vocab
    {
        return Err(format!(
            "MoESD causal identity shape mismatch: gamma={max_gamma} vocab={n_vocab} packed={}",
            packed.len(),
        )
        .into());
    }
    let mut max_diff = 0.0f32;
    for (position, (reference, candidate)) in reference_rows
        .iter()
        .zip(packed.chunks_exact(n_vocab))
        .enumerate()
    {
        let reference_token = argmax(reference);
        let candidate_token = argmax(candidate);
        max_diff = reference
            .iter()
            .zip(candidate)
            .map(|(left, right)| (left - right).abs())
            .fold(max_diff, f32::max);
        if reference_token != candidate_token {
            return Err(format!(
                "MoESD causal identity failed at gamma position {}: sequential={} packed={} max_diff={max_diff:.6e}",
                position + 1,
                reference_token,
                candidate_token,
            )
            .into());
        }
    }
    eprintln!(
        "[target-efficiency] B=1 gamma={max_gamma} causal identity: ARGMAX MATCH (max_diff={max_diff:.6e})",
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
fn emit_row(
    run: usize,
    batch: usize,
    gamma: usize,
    ms_step: f64,
    started_unix_ms: u128,
    finished_unix_ms: u128,
    layers: &[MoesdLayerUnion],
    depth: usize,
    cache_cap: usize,
    primary_device: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let elapsed_s = ms_step / 1000.0;
    let effective_toks = (batch * gamma) as f64 / elapsed_s;
    let accepted_toks = batch as f64 * accepted_sum(gamma)?;
    let realistic_toks = accepted_toks / elapsed_s;
    let mut out = std::io::stdout().lock();
    write!(
        out,
        "{{\"run\":{run},\"B\":{batch},\"gamma\":{gamma},\"ms_step\":{ms_step:.6},\
         \"started_unix_ms\":{started_unix_ms},\"finished_unix_ms\":{finished_unix_ms},\
         \"rows\":{},\"layers\":[",
        batch * gamma,
    )?;
    for (index, layer) in layers.iter().enumerate() {
        if index != 0 {
            write!(out, ",")?;
        }
        write!(
            out,
            "{{\"id\":{},\"union\":{},\"n_expert\":{},\"n_used\":{},\"assignments\":{}}}",
            layer.id, layer.union_size, layer.n_expert, layer.n_used, layer.assignments,
        )?;
    }
    writeln!(
        out,
        "],\"effective_toks\":{effective_toks:.6},\"accepted_toks\":{accepted_toks:.6},\
         \"realistic_toks\":{realistic_toks:.6},\"depth\":{depth},\"cache_cap\":{cache_cap},\
         \"primary_device\":{primary_device},\"config\":\"PP2\"}}",
    )?;
    out.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut argv = std::env::args().skip(1);
    let path = argv.next().ok_or(
        "usage: target-efficiency <model.gguf> [--b 1,2,...] [--gamma 1,2,...] [--runs N]",
    )?;
    let args: Vec<String> = argv.collect();
    let known = ["--b", "--gamma", "--runs"];
    let mut index = 0;
    while index < args.len() {
        if !known.contains(&args[index].as_str()) {
            return Err(format!("unknown argument {:?}", args[index]).into());
        }
        if args.get(index + 1).is_none() {
            return Err(format!("{} requires a value", args[index]).into());
        }
        index += 2;
    }
    let batches = option_value(&args, "--b")?
        .map(|value| parse_list(&value, "--b"))
        .transpose()?
        .unwrap_or_else(|| DEFAULT_BATCHES.to_vec());
    let gammas = option_value(&args, "--gamma")?
        .map(|value| parse_list(&value, "--gamma"))
        .transpose()?
        .unwrap_or_else(|| DEFAULT_GAMMAS.to_vec());
    let runs = option_value(&args, "--runs")?
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|err| format!("invalid --runs: {err}"))?
        .unwrap_or(5);
    if runs == 0 {
        return Err("--runs must be positive".into());
    }
    let max_batch = *batches.iter().max().unwrap();
    let max_gamma = *gammas.iter().max().unwrap();
    if max_batch > 32 || max_gamma > 8 || max_batch * max_gamma > 256 {
        return Err(
            "requested matrix exceeds the frozen B<=32, gamma<=8, B*gamma<=256 envelope".into(),
        );
    }
    for &gamma in &gammas {
        accepted_sum(gamma)?;
    }
    let depth = std::env::var("MEMRA_MOESD_DEPTH")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|err| format!("invalid MEMRA_MOESD_DEPTH: {err}"))?
        .unwrap_or(128);
    if depth < 16 {
        return Err("MEMRA_MOESD_DEPTH must be at least 16 for prime_cache".into());
    }
    let cache_cap = std::env::var("MEMRA_MOESD_CACHE_CAP")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|err| format!("invalid MEMRA_MOESD_CACHE_CAP: {err}"))?
        .unwrap_or(depth + max_gamma + 32);
    if cache_cap < depth + max_gamma {
        return Err(format!(
            "MEMRA_MOESD_CACHE_CAP={cache_cap} is below required {}",
            depth + max_gamma,
        )
        .into());
    }

    let primary_device = worker_device()?;
    let e = Engine::new(primary_device)?;
    let gguf = GgufFile::open(&path)?;
    let model = HybridModel::load(&e, &gguf)?;
    if model.mtp.is_none() {
        return Err(
            "model has no MTP head; set MEMRA_MTP_DRAFT to the pinned Q8_0 artifact".into(),
        );
    }
    if model.cfg.step35.is_none() {
        return Err(
            "target-efficiency currently supports only the pinned Step-3.7 geometry".into(),
        );
    }
    let expected_moe_layers = model
        .layers
        .iter()
        .filter(|layer| matches!(&layer.ffn, Ffn::Moe(_)))
        .count();
    if expected_moe_layers == 0 {
        return Err("model has no MoE layers".into());
    }
    eprintln!(
        "[target-efficiency] loaded arch={} layers={} moe_layers={} B={batches:?} gamma={gammas:?} runs={runs} depth={depth} cap={cache_cap} primary=dev{primary_device}",
        gguf.arch().unwrap_or("?"),
        model.layers.len(),
        expected_moe_layers,
    );

    let prompts: Vec<Vec<u32>> = (0..max_batch)
        .map(|session| {
            (0..depth)
                .map(|position| 55 + ((session as u32 * 997 + position as u32 * 31) % 120_000))
                .collect()
        })
        .collect();
    let mut caches: Vec<Cache> = (0..max_batch)
        .map(|_| memra_engine::pp::new_cache(&e, &model.cfg, cache_cap))
        .collect::<Result<_, _>>()?;
    let mut prime_logits = Vec::with_capacity(max_batch);
    for (session, cache) in caches.iter_mut().enumerate() {
        let (logits, _, _) = model.prime_cache(&e, &prompts[session], cache, 0)?;
        prime_logits.push(logits);
    }
    memra_engine::pp::sync_stages_after_load(&e, model.layers.len())?;
    let snapshots: Vec<CacheSnapshot> = caches
        .iter()
        .map(|cache| cache.snapshot(&e))
        .collect::<Result<_, _>>()?;
    if snapshots
        .iter()
        .any(|snapshot| snapshot.conv.iter().any(Option::is_some))
    {
        return Err("Step-3.7 harness unexpectedly encountered recurrent cache state".into());
    }

    // Freeze one realistic greedy target continuation per independent session. This is source
    // material for every matrix cell; no public score or measured union influences the tokens.
    let mut continuations = Vec::with_capacity(max_batch);
    for session in 0..max_batch {
        let mut token = argmax(&prime_logits[session]) as u32;
        let mut tokens = Vec::with_capacity(max_gamma);
        tokens.push(token);
        for _ in 1..max_gamma {
            let (logits, _) = model.decode_step_h(&e, token, &mut caches[session])?;
            token = argmax(&logits) as u32;
            tokens.push(token);
        }
        restore(&e, &model, &mut caches, &snapshots, session + 1)?;
        continuations.push(tokens);
    }

    identity_check(
        &e,
        &model,
        &mut caches,
        &snapshots,
        &continuations,
        max_gamma,
    )?;

    // One boot warmup, excluded from output.
    let warm_batch = batches[0];
    let warm_gamma = gammas[0];
    let warm_logits = forward(
        &e,
        &model,
        &mut caches,
        &continuations,
        warm_batch,
        warm_gamma,
    )?;
    e.stream().synchronize()?;
    drop(warm_logits);
    restore(&e, &model, &mut caches, &snapshots, warm_batch)?;
    eprintln!("[target-efficiency] warmup complete");

    for run in 1..=runs {
        let mut run_batches = batches.clone();
        let mut run_gammas = gammas.clone();
        if run % 2 == 0 {
            run_batches.reverse();
            run_gammas.reverse();
        }
        for &batch in &run_batches {
            for &gamma in &run_gammas {
                e.stream().synchronize()?;
                let started_unix_ms = unix_ms()?;
                let start = Instant::now();
                let logits = forward(&e, &model, &mut caches, &continuations, batch, gamma)?;
                e.stream().synchronize()?;
                let ms_step = start.elapsed().as_secs_f64() * 1000.0;
                let finished_unix_ms = unix_ms()?;
                drop(logits);
                restore(&e, &model, &mut caches, &snapshots, batch)?;

                moesd::begin_capture()?;
                let replay = forward(&e, &model, &mut caches, &continuations, batch, gamma);
                let layers = match replay {
                    Ok(logits) => {
                        e.stream().synchronize()?;
                        drop(logits);
                        moesd::finish_capture()?
                    }
                    Err(err) => {
                        let _ = moesd::finish_capture();
                        return Err(err);
                    }
                };
                validate_layers(&layers, expected_moe_layers, batch * gamma)?;
                restore(&e, &model, &mut caches, &snapshots, batch)?;
                emit_row(
                    run,
                    batch,
                    gamma,
                    ms_step,
                    started_unix_ms,
                    finished_unix_ms,
                    &layers,
                    depth,
                    cache_cap,
                    primary_device,
                )?;
                eprintln!(
                    "[target-efficiency] run={run}/{runs} B={batch} gamma={gamma} ms={ms_step:.3}",
                );
            }
        }
    }
    Ok(())
}
