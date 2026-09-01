//! Production-B=1 versus speculative T=1/T=2 shape bisect.
//!
//! Loads the same GGUF, safetensors, or manifest-backed source as `run-spec`, primes the cache
//! with the production target class, teacher-forces a committed prefix, then evaluates one probe
//! token three ways:
//!   1. production B=1 target;
//!   2. verify T=1;
//!   3. verify T=2 column 0.
//!
//! The T=1 and T=2 runs capture every trunk layer's residual row. The first nonzero layer names
//! the operation whose batch-width dispatch breaks speculative exactness.
//!
//! Usage:
//!   MEMRA_PROMPT_TOKENS='1 2 3' verify-shape-bisect <model> <prefix...> -- <probe> [watch...]

use memra_engine::Engine;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgufFile;

fn parse_ids(raw: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    raw.split(|ch: char| !ch.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .map(str::parse)
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn decode_b1(
    model: &HybridModel,
    e: &Engine,
    token: u32,
    cache: &mut memra_engine::cache::Cache,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let batched_class = model.uses_sliding_gated_moe_program()
        || memra_engine::plan_backend::gdn_dspark_compatible(&model.plan);
    if batched_class {
        let mut caches = [cache];
        Ok(model.decode_step_batch(e, &[token], &mut caches)?.remove(0))
    } else {
        Ok(model.decode_step_h(e, token, cache)?.0)
    }
}

fn report(name: &str, logits: &[f32], baseline: &[f32], watch: &[u32]) {
    let mut top = logits
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<(usize, f32)>>();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    let maxdiff = logits
        .iter()
        .zip(baseline)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    print!(
        "{name}: argmax={} top2=({}:{:.6}, {}:{:.6}) maxdiff_vs_b1={maxdiff:.3e}",
        argmax(logits),
        top[0].0,
        top[0].1,
        top[1].0,
        top[1].1,
    );
    for &token in watch {
        print!(" l[{token}]={:.6}", logits[token as usize]);
    }
    println!();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let path_arg = args
        .first()
        .ok_or("usage: verify-shape-bisect <model> <prefix...> -- <probe> [watch...]")?;
    let separator = args
        .iter()
        .position(|arg| arg == "--")
        .ok_or("verify-shape-bisect requires -- <probe> [watch...]")?;
    let prefix = args[1..separator]
        .iter()
        .map(|value| value.parse::<u32>())
        .collect::<Result<Vec<_>, _>>()?;
    let tail = args[separator + 1..]
        .iter()
        .map(|value| value.parse::<u32>())
        .collect::<Result<Vec<_>, _>>()?;
    let (&probe, watch) = tail
        .split_first()
        .ok_or("verify-shape-bisect requires a probe token after --")?;

    let path = memra_gguf::hf::resolve_arg(path_arg)?;
    let e = Engine::new(0)?;
    let is_dir = std::path::Path::new(&path).is_dir();
    let g = if is_dir {
        None
    } else {
        Some(GgufFile::open(&path)?)
    };
    let mut source: Option<Box<dyn memra_gguf::source::TensorSource>> = None;
    let mut tokenizer_dir = std::path::PathBuf::from(&path);
    if is_dir {
        let dir = std::path::Path::new(&path);
        if dir.join("manifest.json").exists() {
            let repack = memra_gguf::source::Hy3RepackSource::open(dir)?;
            tokenizer_dir = repack
                .source_dir()
                .filter(|source| source.join("tokenizer.json").exists())
                .unwrap_or(dir)
                .to_path_buf();
            source = Some(Box::new(repack));
        } else {
            source = Some(Box::new(memra_gguf::source::SafetensorsSource::open(dir)?));
        }
    }
    let model = match (&g, &source) {
        (Some(g), _) => HybridModel::load_without_mtp(&e, g)?,
        (None, Some(source)) => HybridModel::load_from_source_without_mtp(&e, source.as_ref())?,
        _ => unreachable!(),
    };

    let prompt = if let Ok(raw) = std::env::var("MEMRA_PROMPT_TOKENS") {
        parse_ids(&raw)?
    } else if let Ok(text) = std::env::var("MEMRA_PROMPT") {
        let tokenizer = match &g {
            Some(g) => memra_tokenizer::Tokenizer::from_gguf(g)?,
            None => memra_tokenizer::Tokenizer::from_hf_dir(&tokenizer_dir)
                .map_err(|error| format!("HF tokenizer init failed: {error}"))?,
        };
        let rendered = if std::env::var_os("MEMRA_CHAT").is_some() {
            tokenizer.apply_chat_template(&[("user", text.as_str())], true)
        } else {
            text
        };
        tokenizer.encode(&rendered, true)
    } else {
        return Err("set MEMRA_PROMPT_TOKENS or MEMRA_PROMPT".into());
    };
    if prompt.is_empty() {
        return Err("verify-shape-bisect prompt is empty".into());
    }

    println!(
        "loaded {:?}: prompt={} prefix={} probe={} watch={watch:?}",
        model.cfg.arch,
        prompt.len(),
        prefix.len(),
        probe,
    );
    let max_ctx = prompt.len() + prefix.len() + 16;
    let mut cache = memra_engine::cache::Cache::new(&e, &model.cfg, max_ctx)?;
    if prompt.len() >= memra_engine::hybrid_forward::PRIME_MIN_T
        && std::env::var_os("MEMRA_PRIME_TOKENWISE").is_none()
        && !e.frozen_cpu_experts_prefer_tokenwise_prime()
    {
        let _ = model.prime_cache(&e, &prompt, &mut cache, 0)?;
    } else {
        for &token in &prompt {
            let _ = decode_b1(&model, &e, token, &mut cache)?;
        }
    }
    for &token in &prefix {
        let _ = decode_b1(&model, &e, token, &mut cache)?;
    }
    e.stream().synchronize()?;

    let pos = cache.pos;
    let snapshot = cache.snapshot(&e)?;
    let b1 = decode_b1(&model, &e, probe, &mut cache)?;
    e.stream().synchronize()?;
    cache.rollback(&e, &snapshot, 0)?;

    let layers = (0..model.layers.len()).collect::<Vec<_>>();
    let (t1_logits, t1_aux, _) =
        model.decode_step_t_aux2(&e, &[probe], pos, &mut cache, &layers, None)?;
    e.stream().synchronize()?;
    cache.rollback(&e, &snapshot, 0)?;

    let filler = argmax(&b1) as u32;
    let (t2_logits, _, t2_col0_aux) =
        model.decode_step_t_aux2(&e, &[probe, filler], pos, &mut cache, &layers, Some(0))?;
    e.stream().synchronize()?;
    cache.rollback(&e, &snapshot, 0)?;
    let t2_col0_aux = t2_col0_aux.ok_or("T=2 pred-column auxiliary rows are missing")?;

    let vocab = model.output.out_features();
    report("production B=1", &b1, &b1, watch);
    report("verify T=1", &t1_logits[..vocab], &b1, watch);
    report("verify T=2 col0", &t2_logits[..vocab], &b1, watch);
    println!("T=2 filler token: {filler}");
    println!("per-layer residual maxdiff: verify T=1 vs verify T=2 col0");
    let mut first_nonzero = None;
    let mut worst = (0usize, 0.0f32);
    for (layer, (t1, t2)) in t1_aux.iter().zip(&t2_col0_aux).enumerate() {
        let t1 = e.dtoh(t1)?;
        let t2 = e.dtoh(t2)?;
        let maxdiff = t1
            .iter()
            .zip(&t2)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        if maxdiff != 0.0 && first_nonzero.is_none() {
            first_nonzero = Some(layer);
        }
        if maxdiff > worst.1 {
            worst = (layer, maxdiff);
        }
        if maxdiff != 0.0 || layer < 3 || layer + 1 == t1_aux.len() {
            println!("  layer {layer:2}: maxdiff={maxdiff:.3e}");
            if first_nonzero.is_some() && layer > first_nonzero.unwrap() + 8 {
                break;
            }
        }
    }
    println!(
        "shape-bisect: first_nonzero={first_nonzero:?} worst_layer={} worst_maxdiff={:.3e}",
        worst.0, worst.1
    );
    Ok(())
}
