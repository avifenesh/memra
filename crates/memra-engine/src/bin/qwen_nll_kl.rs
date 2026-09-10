//! Quality twin for the calibrated prefill program: teacher-forced NLL, and KL against a
//! reference artifact, over the continuation of an A4-primed prompt.
//!
//! WHY THE WINDOW SITS AFTER THE PRIME. Decode is W4A8 in both artifacts, so scoring decode
//! positions is not "measuring the wrong phase": decode consumes the state the prefill built,
//! and that state is exactly what the calibrated program changes. A window of teacher-forced
//! positions immediately after an 8k or 32k prefill is the cheapest probe that carries the whole
//! prefill error into a number, which is the channel the lane design names as mandatory.
//!
//! Two passes, so full-vocabulary rows never have to exist twice on one card:
//!   --dump <dir>      run the REFERENCE artifact, write log-softmax rows per position
//!   --against <dir>   run the CANDIDATE, read those rows back, report KL(reference || candidate)
//!
//! Both passes report token-weighted NLL and exp(NLL) over the identical teacher-forced tokens.
//!
//! usage: qwen-nll-kl <artifact.gguf> <out.json> (--dump <dir>|--against <dir>) <file>:<ctx>:<window> ...
use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgufFile;
use sha2::{Digest, Sha256};

/// log-softmax in f64, returned as f32 rows: the vocabulary is 248k wide and the tail matters
/// for KL, so the reduction is not done in f32.
fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Digest of the exact token ids a slice scores. Every artifact brings its OWN tokenizer out of
/// its OWN GGUF, so "both arms scored the same text" is not the same claim as "both arms scored
/// the same tokens": a mint that rewrote a tokenizer key would shift the sequence and the
/// per-position dNLL would then measure the tokenizer, not the program. The reference pass writes
/// this digest beside its rows and every candidate REFUSES on a mismatch.
fn ids_digest(ids: &[u32]) -> String {
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|v| (*v as f64 - max).exp()).sum();
    let log_z = max + sum.ln();
    logits.iter().map(|v| (*v as f64 - log_z) as f32).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let model_path = args.remove(0);
    let out_path = args.remove(0);
    let mode = args.remove(0);
    let dir = std::path::PathBuf::from(args.remove(0));
    let dumping = match mode.as_str() {
        "--dump" => true,
        "--against" => false,
        other => panic!("mode must be --dump or --against, got {other}"),
    };
    std::fs::create_dir_all(&dir)?;

    // CONTROL (b): load the calibrated artifact but DROP its activation program, so the same file
    // scores through the served W4A8 path. If that is not bitwise equal to the served arm, the two
    // arms are not scoring the same thing (window alignment, positions, chunking, an off-by-one
    // logits row) and THAT is the finding, not a quantization delta.
    let e = Engine::new(0)?;
    let g = GgufFile::open(&model_path)?;
    let tok = memra_tokenizer::Tokenizer::from_gguf(&g)?;
    // MEMRA_A4_DISABLE is honoured inside PrefillFp4::from_gguf, i.e. BEFORE the weights are
    // stamped. Dropping the config here instead would leave every weight stamped and measure the
    // A4 arm twice.
    let model = HybridModel::load(&e, &g)?;
    println!(
        "artifact {model_path}: activation program {:?}",
        model.cfg.prefill_activation
    );

    let mut slices = Vec::new();
    for spec in &args {
        let parts: Vec<&str> = spec.split(':').collect();
        let (file, ctx, window) = (
            parts[0],
            parts[1].parse::<usize>().expect("ctx tokens"),
            parts[2].parse::<usize>().expect("window"),
        );
        let text = std::fs::read_to_string(file)?;
        let mut ids = tok.encode(&text, true);
        ids.truncate(ctx + window);
        assert!(
            ids.len() == ctx + window,
            "{file}: only {} tokens, need {}",
            ids.len(),
            ctx + window
        );
        let name = format!(
            "{}-{ctx}-{window}",
            std::path::Path::new(file)
                .file_stem()
                .unwrap()
                .to_string_lossy()
        );

        // POSITION IDENTITY. The dump pass records the ids digest; the candidate pass asserts it.
        // Without this the two arms are only assumed to share teacher-forced positions, and
        // `analyze_nll.py` pairs them by index whatever they contain.
        let ids_sha = ids_digest(&ids);
        let sha_path = dir.join(format!("{name}.ids.sha256"));
        // The dump dir also names the artifact whose rows it holds, so a candidate's receipt says
        // WHICH reference it was scored against instead of leaving it to the runner's shell.
        let ref_path = dir.join("reference-artifact.txt");
        if dumping {
            std::fs::write(&sha_path, format!("{ids_sha}\n"))?;
            std::fs::write(&ref_path, format!("{model_path}\n"))?;
        } else {
            let want = std::fs::read_to_string(&sha_path)
                .unwrap_or_else(|e| panic!("{}: {e}", sha_path.display()));
            let want = want.trim();
            assert_eq!(
                want, ids_sha,
                "{name}: the reference scored ids {want} and this artifact tokenizes the same \
                 file to {ids_sha}; the two arms are NOT on the same teacher-forced positions"
            );
        }

        // EXTERNAL-ORACLE HANDOFF: dump the exact token ids so SGLang scores the identical
        // sequence through its native input_ids path. Re-tokenizing the text on the other side
        // would make a tokenizer difference look like a model difference.
        if let Some(dir) = std::env::var_os("MEMRA_DUMP_IDS_DIR") {
            let out = std::path::Path::new(&dir).join(format!("{name}.ids"));
            std::fs::create_dir_all(&dir)?;
            let mut bytes = Vec::with_capacity(ids.len() * 4);
            for id in &ids {
                bytes.extend_from_slice(&id.to_le_bytes());
            }
            std::fs::write(&out, bytes)?;
            println!("  wrote {} ids to {}", ids.len(), out.display());
        }

        // The prompt takes the A4 prefill; the window is scored one teacher-forced step at a time.
        let mut cache = memra_engine::pp::new_cache(&e, &model.cfg, ids.len() + 8)?;
        let (mut logits, _, _) = model.prime_cache(&e, &ids[..ctx], &mut cache, 0)?;

        let rows_path = dir.join(format!("{name}.f32"));
        let mut writer = dumping.then(|| {
            std::io::BufWriter::new(std::fs::File::create(&rows_path).expect("reference rows"))
        });
        let reference: Option<Vec<u8>> = (!dumping).then(|| {
            std::fs::read(&rows_path).unwrap_or_else(|e| panic!("{}: {e}", rows_path.display()))
        });
        let vocab = logits.len();
        if let Some(bytes) = &reference {
            assert_eq!(
                bytes.len(),
                window * vocab * 4,
                "{name}: reference rows are {} bytes, expected {window} x {vocab} x 4",
                bytes.len()
            );
        }

        let mut nll = 0.0f64;
        let mut kl = 0.0f64;
        // PER-POSITION, because a window mean cannot carry a confidence interval and the absolute
        // perplexity of these transcripts swings by three orders of magnitude between positions.
        // The analysis (paired dNLL bootstrap, KL band, top-1 agreement, the ppl sanity filter)
        // needs the positions, not the summary.
        let mut per_nll: Vec<f64> = Vec::with_capacity(window);
        let mut per_kl: Vec<f64> = Vec::with_capacity(window);
        let mut per_top1: Vec<u32> = Vec::with_capacity(window);
        for step in 0..window {
            let truth = ids[ctx + step];
            let lsm = log_softmax(&logits);
            nll -= lsm[truth as usize] as f64;
            per_nll.push(-(lsm[truth as usize] as f64));
            per_top1.push(
                lsm.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0),
            );
            if let Some(w) = writer.as_mut() {
                use std::io::Write;
                for v in &lsm {
                    w.write_all(&v.to_le_bytes())?;
                }
            }
            if let Some(bytes) = &reference {
                // KL(reference || candidate) = sum_v p_ref(v) * (logp_ref(v) - logp_cand(v)).
                let base = step * vocab * 4;
                let mut sum = 0.0f64;
                for (v, lp_cand) in lsm.iter().enumerate() {
                    let o = base + v * 4;
                    let lp_ref =
                        f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
                            as f64;
                    sum += lp_ref.exp() * (lp_ref - *lp_cand as f64);
                }
                kl += sum;
                per_kl.push(sum);
            }
            if reference.is_none() {
                per_kl.push(0.0);
            }
            let mut caches = [&mut cache];
            logits = model
                .decode_step_batch(&e, &[truth], &mut caches)?
                .remove(0);
        }
        if let Some(mut w) = writer {
            use std::io::Write;
            w.flush()?;
        }
        let mean_nll = nll / window as f64;
        let mean_kl = kl / window as f64;
        // CONTROL (c): does the PRIME path agree with the DECODE path on this same artifact?
        // The teacher-forced argmax over the first positions of the window is produced by the
        // prime + decode chain here; a free greedy continuation from the identical prefix uses
        // the serving generate path. Disagreement between them is an artifact-internal
        // inconsistency, independent of any comparison to the served mint.
        if std::env::var_os("MEMRA_A4_PRIME_VS_DECODE").is_some() {
            let probe = 64.min(window);
            let mut fresh = memra_engine::pp::new_cache(&e, &model.cfg, ctx + probe + 8)?;
            model.prime_cache(&e, &ids[..ctx], &mut fresh, 0)?;
            let generated = model.generate(&e, &ids[..ctx], probe)?;
            let forced: Vec<u32> = (0..probe).map(|i| ids[ctx + i]).collect();
            let same = generated
                .iter()
                .zip(forced.iter())
                .take_while(|(a, b)| a == b)
                .count();
            println!(
                "  prime-vs-decode: greedy continuation matches the teacher-forced tokens for \
                 {same} of {probe} positions (a low number is expected when the text is not what \
                 the model would have written; the point is that it is the SAME on both arms)"
            );
        }
        println!(
            "{name}: ctx {ctx} window {window} | NLL {mean_nll:.6} | ppl {:.4}{}",
            mean_nll.exp(),
            if dumping {
                String::new()
            } else {
                format!(" | KL(ref||cand) {mean_kl:.6e}")
            }
        );
        let join = |v: &[f64]| {
            v.iter()
                .map(|x| format!("{x:.6e}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        slices.push(format!(
            "    {{\"slice\": \"{name}\", \"file\": \"{}\", \"ctx\": {ctx}, \"window\": {window}, \
             \"ids_sha256\": \"{ids_sha}\", \
             \"nll\": {mean_nll:.8}, \"ppl\": {:.6}, \"kl\": {}, \
             \"per_nll\": [{}], \"per_kl\": [{}], \"per_top1\": [{}]}}",
            escape(file),
            mean_nll.exp(),
            if dumping {
                "null".into()
            } else {
                format!("{mean_kl:.8e}")
            },
            join(&per_nll),
            join(&per_kl),
            per_top1
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        ));
    }

    let reference_artifact = std::fs::read_to_string(dir.join("reference-artifact.txt"))
        .map(|v| v.trim().to_string())
        .unwrap_or_else(|_| model_path.clone());
    std::fs::write(
        &out_path,
        format!(
            "{{\n  \"artifact\": \"{model_path}\",\n  \"reference_artifact\": \"{}\",\n  \"role\": \"{}\",\n  \"slices\": [\n{}\n  ]\n}}\n",
            escape(&reference_artifact),
            if dumping { "reference" } else { "candidate" },
            slices.join(",\n")
        ),
    )?;
    println!("wrote {out_path}");
    Ok(())
}
