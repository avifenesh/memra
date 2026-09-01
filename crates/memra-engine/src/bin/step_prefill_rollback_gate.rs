//! Gate-only Step TP4 multi-chunk prefill and cache-rollback proof.
//!
//! Usage:
//!   step-prefill-rollback-gate MODEL SERVED_TOKENS OUTPUT_DIR clean|rollback SPLIT
//!   MEMRA_PROMPT_TOKEN_FILE may name a whitespace-delimited token-id fixture.

use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use memra_engine::Engine;
use memra_engine::cache::Cache;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_engine::memra_gguf::source::SafetensorsSource;

unsafe extern "C" {
    fn cudaProfilerStart() -> i32;
    fn cudaProfilerStop() -> i32;
}

const STEP_TRUNK_LAYERS: usize = 45;

fn strict_bool(name: &str) -> Result<bool, Box<dyn std::error::Error>> {
    Ok(match std::env::var(name).ok().as_deref() {
        None | Some("") | Some("0") => false,
        Some("1") => true,
        Some(value) => return Err(format!("{name}={value:?} is invalid; expected 0 or 1").into()),
    })
}

fn write_f32(path: &Path, values: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

fn parse_token_text(raw: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let tokens = raw
        .split(|c: char| !c.is_ascii_digit())
        .filter(|field| !field.is_empty())
        .map(str::parse)
        .collect::<Result<Vec<_>, _>>()?;
    if tokens.is_empty() {
        return Err("token fixture is empty".into());
    }
    Ok(tokens)
}

fn parse_tokens(path: &Path) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    parse_token_text(&raw)
        .map_err(|error| format!("invalid token fixture {}: {error}", path.display()).into())
}

fn validate_split(total: usize, split: usize) -> Result<(), String> {
    let minimum = memra_engine::hybrid_forward::PRIME_MIN_T;
    if split < minimum || total.saturating_sub(split) < minimum {
        return Err(format!(
            "both prefill chunks must contain at least {minimum} tokens, got {split}+{}",
            total.saturating_sub(split)
        ));
    }
    Ok(())
}

fn check_cache_lengths(
    cache: &Cache,
    expected: usize,
    require_distributed: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if cache.kv.len() < STEP_TRUNK_LAYERS || cache.tp_kv.len() < STEP_TRUNK_LAYERS {
        return Err(format!(
            "expected at least {STEP_TRUNK_LAYERS} cache slots, found local={} distributed={}",
            cache.kv.len(),
            cache.tp_kv.len()
        )
        .into());
    }

    let mut distributed_layers = 0usize;
    for layer in 0..STEP_TRUNK_LAYERS {
        let local = cache.kv[layer]
            .as_ref()
            .ok_or_else(|| format!("layer {layer} is missing its local cache"))?;
        if local.len != expected {
            return Err(format!(
                "layer {layer} local cache length {} != {expected}",
                local.len
            )
            .into());
        }

        if let Some(distributed) = &cache.tp_kv[layer] {
            distributed_layers += 1;
            if distributed.committed_len() != expected || distributed.staged_len() != expected {
                return Err(format!(
                    "layer {layer} distributed cache lengths {}/{} != {expected}",
                    distributed.committed_len(),
                    distributed.staged_len()
                )
                .into());
            }
        }
    }
    if require_distributed && distributed_layers != STEP_TRUNK_LAYERS {
        return Err(format!(
            "expected {STEP_TRUNK_LAYERS} distributed attention caches, found {distributed_layers}"
        )
        .into());
    }
    if !require_distributed && distributed_layers != 0 {
        return Err(format!(
            "root prefill unexpectedly created {distributed_layers} distributed attention caches"
        )
        .into());
    }
    if cache.pos != expected {
        return Err(format!("cache position {} != {expected}", cache.pos).into());
    }
    Ok(())
}

fn layer_lengths(
    cache: &Cache,
    layer: usize,
) -> Result<(usize, usize, usize), Box<dyn std::error::Error>> {
    let local = cache
        .kv
        .get(layer)
        .and_then(Option::as_ref)
        .ok_or_else(|| format!("layer {layer} has no local cache"))?;
    let distributed = cache
        .tp_kv
        .get(layer)
        .and_then(Option::as_ref)
        .ok_or_else(|| format!("layer {layer} has no distributed cache"))?;
    Ok((
        local.len,
        distributed.committed_len(),
        distributed.staged_len(),
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .ok_or("usage: step-prefill-rollback-gate MODEL SERVED OUTPUT clean|rollback SPLIT")?;
    let served_path = args.next().ok_or("missing served-token file")?;
    let output_path = args.next().ok_or("missing output directory")?;
    let mode = args.next().ok_or("missing clean|rollback mode")?;
    let split = args
        .next()
        .ok_or("missing prefill split")?
        .parse::<usize>()?;
    if args.next().is_some() || !matches!(mode.as_str(), "clean" | "rollback") {
        return Err("expected exactly MODEL SERVED OUTPUT clean|rollback SPLIT".into());
    }

    let rank_local = match std::env::var("MEMRA_STEP_TP_PREFILL") {
        Ok(value) if value == "0" => false,
        Ok(value) if value == "1" => true,
        Ok(value) => {
            return Err(format!("MEMRA_STEP_TP_PREFILL must be 0 or 1, got {value:?}").into());
        }
        Err(std::env::VarError::NotPresent) => false,
        Err(error) => return Err(error.into()),
    };
    if mode == "rollback" && !rank_local {
        return Err("rollback mode requires MEMRA_STEP_TP_PREFILL=1".into());
    }
    let profile = strict_bool("MEMRA_STEP_TP_PREFILL_PROFILE")?;
    if profile && mode != "clean" {
        return Err("MEMRA_STEP_TP_PREFILL_PROFILE=1 requires clean mode".into());
    }

    let output = Path::new(&output_path);
    if output.exists() && output.read_dir()?.next().is_some() {
        return Err(format!("output directory {} is not empty", output.display()).into());
    }
    std::fs::create_dir_all(output)?;

    let engine = Engine::new(0)?;
    let source = SafetensorsSource::open(Path::new(&model_path))?;
    let model = HybridModel::load_from_source_without_mtp(&engine, &source)?;
    let prompt_tokens = if let Some(path) = std::env::var_os("MEMRA_PROMPT_TOKEN_FILE") {
        parse_tokens(Path::new(&path))?
    } else {
        let tokenizer = memra_tokenizer::Tokenizer::from_hf_dir(Path::new(&model_path))?;
        let prompt = std::env::var("MEMRA_PROMPT")
            .unwrap_or_else(|_| "Explain exact distributed inference.".into());
        let rendered = tokenizer.apply_chat_template(&[("user", &prompt)], true);
        tokenizer.encode(&rendered, true)
    };
    validate_split(prompt_tokens.len(), split)?;
    let served = parse_tokens(Path::new(&served_path))?;
    let total = prompt_tokens.len();
    let mut cache = Cache::new(
        &engine,
        &model.cfg,
        total
            .checked_add(served.len())
            .and_then(|value| value.checked_add(8))
            .ok_or("cache capacity overflow")?,
    )?;

    println!(
        "PREFILL_ROLLBACK_GATE mode={mode} rank_local={rank_local} \
         prompt_tokens={total} split={split}+{} served_tokens={}",
        total - split,
        served.len()
    );
    if profile {
        let result = unsafe { cudaProfilerStart() };
        if result != 0 {
            return Err(format!("cudaProfilerStart failed with CUDA error {result}").into());
        }
        println!("PREFILL_PROFILE start scope=two-prefill-chunks rank_local={rank_local}");
    }
    let prime1_started = Instant::now();
    let (prime1, _, _) =
        model.prime_cache(&engine, &prompt_tokens[..split], &mut cache, total - split)?;
    let prime1_seconds = prime1_started.elapsed().as_secs_f64();
    write_f32(&output.join("prime1-logits.f32"), &prime1)?;
    check_cache_lengths(&cache, split, rank_local)?;

    let second = &prompt_tokens[split..];
    let prime2_started = Instant::now();
    let prime2 = if mode == "rollback" {
        let snapshot = cache.snapshot(&engine)?;
        let transaction = cache.tp_kv[43]
            .as_mut()
            .ok_or("rollback gate lost layer 43 distributed cache")?
            .begin_transaction()?;
        println!(
            "PREFILL_ROLLBACK_INJECT layer=43 generation={} base={}",
            transaction.generation(),
            transaction.base_len()
        );
        let failure = match model.prime_cache(&engine, second, &mut cache, 0) {
            Ok(_) => return Err("nested layer-43 transaction did not fail the second chunk".into()),
            Err(error) => error.to_string(),
        };
        if !failure.contains("already active") {
            return Err(format!("unexpected injected failure: {failure}").into());
        }
        let layer3 = layer_lengths(&cache, 3)?;
        let layer43 = layer_lengths(&cache, 43)?;
        if cache.pos != split || layer3 != (total, total, total) || layer43 != (split, split, split)
        {
            return Err(format!(
                "injected partial state pos={} layer3={layer3:?} layer43={layer43:?}",
                cache.pos
            )
            .into());
        }
        println!(
            "PREFILL_ROLLBACK_PARTIAL pos={} layer3={layer3:?} layer43={layer43:?}",
            cache.pos
        );
        cache.rollback(&engine, &snapshot, 0)?;
        check_cache_lengths(&cache, split, true)?;
        println!("PREFILL_ROLLBACK_RESTORED pos={}", cache.pos);
        let (logits, _, _) = model.prime_cache(&engine, second, &mut cache, 0)?;
        println!("PREFILL_ROLLBACK_RETRY exact_program=true");
        logits
    } else {
        model.prime_cache(&engine, second, &mut cache, 0)?.0
    };
    let prime2_seconds = prime2_started.elapsed().as_secs_f64();
    if profile {
        engine.stream().synchronize()?;
        let result = unsafe { cudaProfilerStop() };
        if result != 0 {
            return Err(format!("cudaProfilerStop failed with CUDA error {result}").into());
        }
        println!("PREFILL_PROFILE stop scope=two-prefill-chunks rank_local={rank_local}");
    }
    println!(
        "PREFILL_TIMING mode={mode} rank_local={rank_local} \
         chunk1_seconds={prime1_seconds:.9} chunk2_seconds={prime2_seconds:.9} \
         total_seconds={:.9} performance_claim=false",
        prime1_seconds + prime2_seconds
    );
    write_f32(&output.join("prime2-logits.f32"), &prime2)?;
    check_cache_lengths(&cache, total, rank_local)?;

    let mut logits = prime2;
    for (step, token) in served.iter().copied().enumerate() {
        logits = model.decode_step(&engine, token, &mut cache)?;
        write_f32(&output.join(format!("decode-{step}-logits.f32")), &logits)?;
    }
    check_cache_lengths(&cache, total + served.len(), true)?;
    engine.stream().synchronize()?;

    let mut summary = std::io::BufWriter::new(std::fs::File::create(output.join("summary.txt"))?);
    writeln!(summary, "mode={mode}")?;
    writeln!(summary, "rank_local={rank_local}")?;
    writeln!(summary, "prompt_tokens={total}")?;
    writeln!(summary, "split={split}+{}", total - split)?;
    writeln!(summary, "served_tokens={served:?}")?;
    writeln!(summary, "final_cache_pos={}", cache.pos)?;
    writeln!(summary, "final_argmax={}", argmax(&logits))?;
    writeln!(summary, "rollback_exercised={}", mode == "rollback")?;
    summary.flush()?;
    println!(
        "PREFILL_ROLLBACK_GATE_PASS mode={mode} final_cache_pos={} final_argmax={}",
        cache.pos,
        argmax(&logits)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{parse_token_text, validate_split};

    #[test]
    fn split_requires_two_prefill_sized_chunks() {
        assert!(validate_split(53, 24).is_ok());
        assert!(validate_split(31, 15).is_err());
        assert!(validate_split(32, 16).is_ok());
        assert!(validate_split(32, 17).is_err());
    }

    #[test]
    fn token_fixture_accepts_plain_numeric_fields() {
        assert_eq!(
            parse_token_text("11 22\n33,44").unwrap(),
            vec![11, 22, 33, 44]
        );
        assert!(parse_token_text(" \n").is_err());
    }
}
