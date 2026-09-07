//! rewrite-receipt-oracle: mint a RewriteParity receipt for ONE execution surface of a real
//! checkpoint by running the native GPU rewrite over the tokens of a captured HF oracle
//! (`memra-checkpoint-oracle-v1`, the bundle `verify checkpoint` consumes) and comparing the
//! last-position logits with `ExecutionRewrite::verify_logits`.
//!
//! The ModelPlan reference gate (`modelplan-reference-gate`) covers the eager surface on a
//! synthetic fixture (hidden <= 256); a real checkpoint needs the oracle route, and the
//! batched / carried-prime surfaces had no producer at all. Surfaces:
//!   decode-eager   every token through `decode_step` from an empty cache
//!   decode-batch   every token through `decode_step_batch` (B=1) from an empty cache
//!   carried-prime  all tokens in one `prime_cache` call (needs >= 16 tokens)
//! usage: rewrite-receipt-oracle <model-dir> <hf-oracle.tsv> <surface> <receipt.tsv>
//! env: MEMRA_ARTIFACT_LOCK (bundle artifact.lock), MEMRA_REWRITE_ATOL / MEMRA_REWRITE_RTOL
//! (policy, default 0.05 / 0.05; argmax always required), MEMRA_FULL_PREC as the loader takes it.
use memra_engine::Engine;
use memra_engine::cache::Cache;
use memra_engine::hybrid::HybridModel;
use memra_engine::plan_backend::{RewriteParityPolicy, RewriteSurface, execution_rewrites};
use sha2::{Digest, Sha256};

fn parse_oracle(path: &str) -> Result<(Vec<u32>, Vec<f32>), Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut tokens = Vec::new();
    let mut vocab = 0usize;
    let mut logits: Vec<(usize, f32)> = Vec::new();
    for line in text.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        match cols.as_slice() {
            ["tokens", list] => {
                tokens = list
                    .split(',')
                    .map(|t| t.trim().parse())
                    .collect::<Result<_, _>>()?
            }
            ["vocab", v] => vocab = v.parse()?,
            ["logit", i, bits] => {
                logits.push((i.parse()?, f32::from_bits(u32::from_str_radix(bits, 16)?)))
            }
            _ => {}
        }
    }
    if tokens.is_empty() || vocab == 0 || logits.len() != vocab {
        return Err(format!(
            "oracle {path}: tokens={} vocab={vocab} logits={}",
            tokens.len(),
            logits.len()
        )
        .into());
    }
    let mut out = vec![0f32; vocab];
    for (i, v) in logits {
        out[i] = v;
    }
    Ok((tokens, out))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: rewrite-receipt-oracle <model-dir> <hf-oracle.tsv> <surface> <receipt.tsv>";
    let model_dir = args.next().ok_or(usage)?;
    let oracle_path = args.next().ok_or(usage)?;
    let surface_name = args.next().ok_or(usage)?;
    let receipt_path = args.next().ok_or(usage)?;
    let surface = match surface_name.as_str() {
        "decode-eager" => RewriteSurface::DecodeEager,
        "decode-batch" => RewriteSurface::DecodeBatch,
        "carried-prime" => RewriteSurface::CarriedPrime,
        other => return Err(format!("unsupported surface {other}").into()),
    };
    let env_f32 = |k: &str, d: f32| -> f32 {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let policy = RewriteParityPolicy {
        max_abs: env_f32("MEMRA_REWRITE_ATOL", 0.05),
        max_rel: env_f32("MEMRA_REWRITE_RTOL", 0.05),
        require_argmax: true,
    };
    let (tokens, expected) = parse_oracle(&oracle_path)?;
    let e = Engine::new(0)?;
    let src: Box<dyn memra_gguf::source::TensorSource> = Box::new(
        memra_gguf::source::SafetensorsSource::open(std::path::Path::new(&model_dir))?,
    );
    let model = HybridModel::load_from_source_without_mtp(&e, src.as_ref())?;
    let rewrite = execution_rewrites(&model.plan)
        .into_iter()
        .find(|r| r.surface == surface)
        .ok_or("surface is absent from the execution manifest")?;
    let ctx = tokens.len() + 32;
    let mut cache = Cache::new(&e, &model.cfg, ctx)?;
    let candidate: Vec<f32> = match surface {
        RewriteSurface::DecodeEager => {
            let mut l = Vec::new();
            for &t in &tokens {
                l = model.decode_step(&e, t, &mut cache)?;
            }
            l
        }
        RewriteSurface::DecodeBatch => {
            let mut l = Vec::new();
            for &t in &tokens {
                let mut caches = [&mut cache];
                l = model.decode_step_batch(&e, &[t], &mut caches)?.remove(0);
            }
            l
        }
        RewriteSurface::CarriedPrime => {
            if tokens.len() < 16 {
                return Err("carried-prime needs an oracle of >= 16 tokens".into());
            }
            model.prime_cache(&e, &tokens, &mut cache, 0)?.0
        }
        _ => unreachable!(),
    };
    if candidate.len() != expected.len() {
        return Err(format!(
            "vocab mismatch: oracle {} native {}",
            expected.len(),
            candidate.len()
        )
        .into());
    }
    let executable = std::fs::read(std::env::current_exe()?)?;
    let executable_sha256 = Sha256::digest(&executable)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let receipt = rewrite.verify_logits(&executable_sha256, &expected, &candidate, policy)?;
    println!(
        "{surface_name}: tokens={} values={} max_abs={} max_rel={} argmax ref={} cand={} first_violation={:?} passed={}",
        tokens.len(),
        receipt.values,
        receipt.max_abs,
        receipt.max_rel,
        receipt.reference_argmax,
        receipt.candidate_argmax,
        receipt.first_violation,
        receipt.passed
    );
    if !receipt.passed {
        return Err("native rewrite diverged from the HF oracle under the policy".into());
    }
    let receipt = memra_engine::plan_backend::bind_rewrite_artifact(receipt)?;
    receipt.validate_for(&rewrite)?;
    std::fs::write(&receipt_path, receipt.to_tsv())?;
    println!("receipt written: {receipt_path}");
    Ok(())
}
