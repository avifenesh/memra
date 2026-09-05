//! Teacher-forced quality receipt for a pinned HF checkpoint and frozen raw texts.
//! No generation, sampling, or performance score: compare per-token NLL between
//! cache formats under identical token inputs. Raw prompts never become tuning labels.
use memra_engine::{Engine, hybrid::HybridModel, pp::new_cache_for_model};
use memra_gguf::source::SafetensorsSource;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};

fn nll(logits: &[f32], target: usize) -> Result<f64, &'static str> {
    if target >= logits.len() || logits.iter().any(|x| !x.is_finite()) {
        return Err("invalid target or nonfinite logits");
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|x| (*x as f64 - max).exp()).sum();
    Ok((max - logits[target] as f64) + sum.ln())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err(
            "usage: latent-cache-quality <hf-dir> <tail-token-count> <text-file>...".into(),
        );
    }
    let window: usize = args[1].parse()?;
    if !(1..=1024).contains(&window) {
        return Err("tail token count must be in 1..=1024".into());
    }
    let dir = std::path::Path::new(&args[0]);
    let source = SafetensorsSource::open(dir)?;
    let e = Engine::new(0)?;
    let model = HybridModel::load_from_source_without_mtp(&e, &source)?;
    if model.mtp.is_some() {
        return Err("quality probe must not load native MTP".into());
    }
    if model.hyper.is_none() {
        return Err("quality probe requires the GLM hyper-connection trunk".into());
    }
    let tok = Tokenizer::from_hf_dir(dir).map_err(|e| format!("tokenizer: {e}"))?;
    let format = if memra_kv::latent_layout::nvfp4_enabled() {
        "nvfp4-row"
    } else {
        "f32"
    };
    for path in &args[2..] {
        let text = std::fs::read_to_string(path)?;
        let sha = format!("{:x}", Sha256::digest(text.as_bytes()));
        let tokens = tok.encode(&text, true);
        if tokens.len() <= window + 1 {
            return Err(format!("prompt {sha} is too short for {window} targets").into());
        }
        let prefix = tokens.len() - window;
        let mut cache = new_cache_for_model(&e, &model, tokens.len() + 8)?;
        for block in &model.plan.mtp_blocks {
            let i = block.layer.index as usize;
            if cache.latent[i].is_some() || cache.kv[i].is_some() || cache.recur[i].is_some() {
                return Err(
                    "quality probe allocated state for an unloaded native MTP block".into(),
                );
            }
        }
        let latent = cache.latent.iter().flatten().count();
        let compressed = cache
            .latent
            .iter()
            .flatten()
            .filter(|p| p.nvfp4.is_some())
            .count();
        if latent == 0
            || (format == "nvfp4-row" && compressed != latent)
            || (format == "f32" && compressed != 0)
        {
            return Err("quality probe cache format did not engage on every latent plane".into());
        }
        // Prefill returns logits predicting token `prefix`; each subsequent
        // teacher-forced step predicts the next frozen input token.
        let (mut logits, _, _) = model.prime_cache(&e, &tokens[..prefix], &mut cache, 0)?;
        let mut total = 0.0;
        for i in prefix..tokens.len() {
            let loss = nll(&logits, tokens[i] as usize)?;
            total += loss;
            println!(
                "{{\"schema\":\"memra.latent-quality.v1\",\"format\":\"{format}\",\"prompt_sha256\":\"{sha}\",\"position\":{i},\"target\":{},\"nll\":{loss:.12}}}",
                tokens[i]
            );
            if i + 1 < tokens.len() {
                logits = model.decode_step(&e, tokens[i], &mut cache)?;
            }
        }
        println!(
            "{{\"schema\":\"memra.latent-quality.summary.v1\",\"format\":\"{format}\",\"prompt_sha256\":\"{sha}\",\"context_tokens\":{},\"targets\":{window},\"mean_nll\":{:.12}}}",
            tokens.len(),
            total / window as f64
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nll_is_stable_and_shift_invariant() {
        assert!((nll(&[0., 0.], 0).unwrap() - 2f64.ln()).abs() < 1e-12);
        assert_eq!(nll(&[1000., 1001.], 1).unwrap(), nll(&[0., 1.], 1).unwrap());
        assert!(nll(&[f32::NAN], 0).is_err());
        assert!(nll(&[0.], 1).is_err());
    }
}
