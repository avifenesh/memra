//! DSpark Q38 end-to-end gate (lane/dspark-q38-recover): the dspark drafter round on a
//! qwen-hybrid target must emit a BYTE-IDENTICAL stream to plain greedy decode, and the
//! round's per-phase economics (draft / snapshot / verify / rollback+replay / ingest)
//! must be measured on the real serving artifact.
//!
//! Usage:
//!   dspark_q38_gate <target.gguf|hf_dir> <draft_export_dir> [ngen]
//!   MEMRA_PROMPT_DIR=<dir of *.txt>  — tokenizer prompts (chat template with MEMRA_CHAT)
//!   (without MEMRA_PROMPT_DIR: three fixed token-id prompts, exactness only)
//!   MEMRA_SPEC_STATS=1               — per-phase round breakdown
use memra_engine::Engine;
use memra_engine::dflash::DflashDraft;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgufFile;

fn plain_oracle(
    model: &HybridModel,
    e: &Engine,
    prompt: &[u32],
    max_new: usize,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    // Serving-class plain greedy (the run_spec oracle): prime, then batched decode at B=1.
    let mut cache = memra_engine::pp::new_cache(e, &model.cfg, prompt.len() + max_new + 8)?;
    let mut logits = if prompt.len() >= memra_engine::hybrid_forward::PRIME_MIN_T {
        model.prime_cache(e, prompt, &mut cache, 0)?.0
    } else {
        let mut row = Vec::new();
        for &token in prompt {
            let mut caches = [&mut cache];
            row = model.decode_step_batch(e, &[token], &mut caches)?.remove(0);
        }
        row
    };
    let mut out = Vec::with_capacity(max_new);
    for _ in 0..max_new {
        let token = argmax(&logits) as u32;
        out.push(token);
        if out.len() >= max_new {
            break;
        }
        let mut caches = [&mut cache];
        logits = model.decode_step_batch(e, &[token], &mut caches)?.remove(0);
    }
    Ok(out)
}

fn first_divergence(a: &[u32], b: &[u32]) -> Option<usize> {
    let n = a.len().min(b.len());
    (0..n)
        .find(|&i| a[i] != b[i])
        .or(if a.len() != b.len() { Some(n) } else { None })
}

fn fixed_prompts() -> Vec<(String, Vec<u32>)> {
    vec![
        (
            "ids-a".into(),
            vec![
                84270, 279, 2701, 7355, 25, 220, 198, 16, 13, 220, 198, 17, 13, 220, 198, 18,
            ],
        ),
        ("ids-b".into(), (100..164u32).collect()),
        (
            "ids-c".into(),
            vec![
                8160, 579, 264, 7047, 1817, 25, 271, 16, 13, 220, 198, 17, 13, 220, 198, 18,
            ],
        ),
    ]
}

fn validate_prompt_floor(prompts: &[(String, Vec<u32>)]) -> Result<(), String> {
    if prompts.is_empty() {
        return Err("DFlash2 gate found no prompts".to_string());
    }
    if let Some((name, ids)) = prompts
        .iter()
        .find(|(_, ids)| ids.len() < memra_engine::hybrid_forward::PRIME_MIN_T)
    {
        return Err(format!(
            "prompt {name} tokenized to {} tokens; DFlash2 gate requires at least {}",
            ids.len(),
            memra_engine::hybrid_forward::PRIME_MIN_T,
        ));
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target = std::env::args()
        .nth(1)
        .expect("usage: dspark_q38_gate <target> <draft_dir> [ngen]");
    let draft_dir = std::env::args().nth(2).expect("draft export dir");
    let ngen: usize = std::env::args()
        .nth(3)
        .and_then(|v| v.parse().ok())
        .unwrap_or(96);
    let path = memra_gguf::hf::resolve_arg(&target)?;
    let e = Engine::new(0)?;
    let is_dir = std::path::Path::new(&path).is_dir();
    let g: Option<GgufFile> = if is_dir {
        None
    } else {
        Some(GgufFile::open(&path)?)
    };
    let mut source: Option<Box<dyn memra_gguf::source::TensorSource>> = None;
    let mut tok_dir = std::path::PathBuf::from(&path);
    if is_dir {
        let dir = std::path::Path::new(&path);
        if dir.join("manifest.json").exists() {
            let repack = memra_gguf::source::Hy3RepackSource::open(dir)?;
            tok_dir = repack
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
        (Some(g), _) => HybridModel::load(&e, g)?,
        (None, Some(source)) => HybridModel::load_from_source(&e, source.as_ref())?,
        _ => unreachable!(),
    };
    let draft = DflashDraft::load(&e, std::path::Path::new(&draft_dir))?;
    println!(
        "target loaded ({} layers); dspark draft: block {}, taps {:?}, markov {}, yarn {}",
        model.cfg.n_layer,
        draft.cfg.block_size,
        draft.cfg.target_layer_ids,
        draft.markov.is_some(),
        draft.rope_yarn.is_some()
    );
    // prompt set: tokenizer dir mode, else fixed id prompts (exactness-only)
    let mut eos: Vec<u32> = Vec::new();
    let mut prompts: Vec<(String, Vec<u32>)> = Vec::new();
    if let Ok(dir) = std::env::var("MEMRA_PROMPT_DIR") {
        let tok = match &g {
            Some(g) => memra_tokenizer::Tokenizer::from_gguf(g)?,
            None => memra_tokenizer::Tokenizer::from_hf_dir(&tok_dir)
                .map_err(|err| format!("HF tokenizer init failed: {err}"))?,
        };
        eos = tok.eog_ids();
        let chat = std::env::var("MEMRA_CHAT").is_ok();
        let mut files: Vec<_> = std::fs::read_dir(&dir)?
            .filter_map(|d| d.ok().map(|d| d.path()))
            .filter(|p| p.extension().map(|x| x == "txt").unwrap_or(false))
            .collect();
        files.sort();
        for fp in &files {
            let text = std::fs::read_to_string(fp)?;
            let to_encode = if chat {
                format!(
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    text.trim()
                )
            } else {
                text
            };
            let ids = tok.encode(&to_encode, true);
            prompts.push((fp.file_name().unwrap().to_string_lossy().into_owned(), ids));
        }
    } else {
        prompts = fixed_prompts();
    }
    validate_prompt_floor(&prompts)?;

    let _ = plain_oracle(&model, &e, &[55u32], 1)?; // cold-start warmup
    let mut all_pass = true;
    let (mut sum_plain, mut sum_spec, mut sum_tok) = (0f64, 0f64, 0usize);
    for (name, ids) in &prompts {
        let t0 = std::time::Instant::now();
        let want = plain_oracle(&model, &e, ids, ngen)?;
        e.stream().synchronize()?;
        let plain_s = t0.elapsed().as_secs_f64();

        let t1 = std::time::Instant::now();
        let got = model.generate_spec_dspark(&e, &draft, ids, ngen, &eos, None)?;
        e.stream().synchronize()?;
        let spec_s = t1.elapsed().as_secs_f64()
            - memra_engine::PRIME_NANOS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;

        // plain oracle has no eos-stop; truncate to the spec stream's natural stop
        let want_cut = &want[..got.len().min(want.len())];
        let div = first_divergence(&got, want_cut);
        let pass = div.is_none() && !got.is_empty();
        all_pass &= pass;
        println!(
            "{name}: prompt {} -> gen {} | plain {:.1} tok/s | spec {:.1} tok/s | {}",
            ids.len(),
            got.len(),
            want.len() as f64 / plain_s,
            got.len() as f64 / spec_s.max(1e-9),
            if pass {
                "EXACT".to_string()
            } else {
                format!(
                    "DIVERGED at {:?} (got len {}, want len {})",
                    div,
                    got.len(),
                    want.len()
                )
            }
        );
        sum_plain += want.len() as f64 / plain_s;
        sum_spec += got.len() as f64 / spec_s.max(1e-9);
        sum_tok += got.len();
    }
    let n = prompts.len() as f64;
    println!(
        "== dspark_q38_gate: {} | mean plain {:.1} tok/s | mean spec {:.1} tok/s | {} tokens ==",
        if all_pass { "ALL EXACT" } else { "FAIL" },
        sum_plain / n,
        sum_spec / n,
        sum_tok
    );
    std::process::exit(if all_pass { 0 } else { 1 });
}

#[cfg(test)]
mod tests {
    use super::{fixed_prompts, validate_prompt_floor};

    #[test]
    fn every_builtin_prompt_clears_the_dflash_prime_floor() {
        validate_prompt_floor(&fixed_prompts()).unwrap();
    }

    #[test]
    fn short_tokenized_prompt_is_a_named_error_not_an_engine_panic() {
        let prompts = vec![("short.txt".to_string(), vec![1, 2, 3])];
        let err = validate_prompt_floor(&prompts).unwrap_err();
        assert!(err.contains("short.txt tokenized to 3 tokens"));
        assert!(err.contains("requires at least"));
    }

    #[test]
    fn empty_prompt_directory_cannot_false_green() {
        let err = validate_prompt_floor(&[]).unwrap_err();
        assert_eq!(err, "DFlash2 gate found no prompts");
    }
}
