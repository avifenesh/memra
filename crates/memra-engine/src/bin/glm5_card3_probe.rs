//! card3-lane acceptance probe (lane/glm5-card3-acceptance-probe, 2026-08-30).
//!
//! ENGINE-LEVEL acceptance harness for the glm5_next native MTP + T-parallel verify loop
//! on the REAL artifact — the fallback path the co-tenancy lane spec names: the worker's
//! `mtp_spec_capable` deliberately reports the glm5 plan unsupported (fail-closed manifest
//! stance, tparallel-verify LANE.md), so the serving path never routes spec and acceptance
//! is measured here through `HybridModel::generate_spec` (the same MEMRA_GLM5_SPEC door,
//! same loop, no server).
//!
//! COUNT-BASED ONLY: this binary prints acceptance counters and byte-identity verdicts.
//! It deliberately prints NO timing numbers (co-tenant lane on a shared box; timing rows
//! belong to the timed windows).
//!
//! Greedy is the instrument: the glm5 loop's accept rule is greedy longest-matching-prefix;
//! the sampled accept rule is the verify lane's stated follow-up and DOES NOT EXIST yet —
//! there is no sampled twin at this level, which is itself a banked finding, not a gap to
//! paper over.
//!
//! Usage:
//!   MEMRA_GLM5_MTP=1 MEMRA_GLM5_SPEC=1 [MEMRA_FRSPEC_TRIM=<ranks.txt>] \
//!   MEMRA_ST_PINNED=1 MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=<n> NVIDIA_TF32_OVERRIDE=0 \
//!   glm5-card3-probe <model_dir> <prompts_dir> <out_dir>
//!
//!   CARD3_KS=1,2,3,4,5,6,7   draft depths (verify rows K+1 must stay inside the knee)
//!   CARD3_MAX_NEW=128        greedy cap per row (loop-law: bounded max tokens)
//!   CARD3_PLAIN=1            also run the plain decode oracle + tape-identity compare
//!
//! Per (prompt, K) row appended to <out_dir>/results.tsv (flushed immediately — the box
//! can die on owner order): prompt, k, rounds, drafted, accepted, out_len, tape_identical,
//! accept_rate, acc_per_cycle (= accepted/rounds), tok_per_cycle (= (accepted+rounds)/rounds,
//! the +1 verify bonus per round). Decoded texts land next to it for loop-law screening.

use memra_engine::Engine;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_gguf::source::SafetensorsSource;
use std::io::Write;

/// Co-tenancy hold: while the file named by CARD3_HOLD_FILE exists, another lane is
/// running TIMED cells on this host — pause between runs (never mid-run; the protocol
/// is finish the in-flight request, then hold).
fn hold_if_marked() {
    let Ok(marker) = std::env::var("CARD3_HOLD_FILE") else {
        return;
    };
    let mut held = false;
    while std::path::Path::new(&marker).exists() {
        if !held {
            eprintln!("[card3-probe] HOLD: {marker} present, pausing between runs");
            held = true;
        }
        std::thread::sleep(std::time::Duration::from_secs(10));
    }
    if held {
        eprintln!("[card3-probe] HOLD released, resuming");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .expect("usage: glm5-card3-probe <model_dir> <prompts_dir> <out_dir>");
    let prompts_dir = args.next().expect("prompts_dir");
    let out_dir = args.next().expect("out_dir");
    std::fs::create_dir_all(&out_dir)?;

    let ks: Vec<usize> = std::env::var("CARD3_KS")
        .unwrap_or_else(|_| "1,2,3,4,5,6,7".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let max_new: usize = std::env::var("CARD3_MAX_NEW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let run_plain = std::env::var("CARD3_PLAIN").as_deref() != Ok("0");

    eprintln!(
        "[card3-probe] model={model_dir} prompts={prompts_dir} out={out_dir} ks={ks:?} \
         max_new={max_new} plain={run_plain} MEMRA_GLM5_MTP={:?} MEMRA_GLM5_SPEC={:?} \
         MEMRA_FRSPEC_TRIM={:?} MEMRA_MOE_SLOTS={:?}",
        std::env::var("MEMRA_GLM5_MTP").ok(),
        std::env::var("MEMRA_GLM5_SPEC").ok(),
        std::env::var("MEMRA_FRSPEC_TRIM").ok(),
        std::env::var("MEMRA_MOE_SLOTS").ok(),
    );

    let e = Engine::new(0)?;
    eprintln!("[card3-probe] GPU: {}", e.ctx().name()?);
    let src = SafetensorsSource::open(std::path::Path::new(&model_dir))?;
    let model = HybridModel::load_from_source(&e, &src)?;
    eprintln!(
        "[card3-probe] loaded: n_layer={} n_embd={} n_vocab={} mtp_loaded={}",
        model.cfg.n_layer,
        model.cfg.n_embd,
        model.cfg.n_vocab,
        model.mtp.is_some(),
    );
    let (free, total) = e.ctx().mem_get_info()?;
    eprintln!(
        "[card3-probe] vram post-load: free={:.2} GiB / total={:.2} GiB",
        free as f64 / (1 << 30) as f64,
        total as f64 / (1 << 30) as f64
    );
    let tok = memra_tokenizer::Tokenizer::from_hf_dir(std::path::Path::new(&model_dir))?;

    // Prompt pool: one .txt file per real prompt, lexicographic order.
    let mut prompt_files: Vec<std::path::PathBuf> = std::fs::read_dir(&prompts_dir)?
        .filter_map(|d| d.ok().map(|d| d.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    prompt_files.sort();
    if prompt_files.is_empty() {
        return Err("no .txt prompts in prompts_dir".into());
    }

    let results_path = std::path::Path::new(&out_dir).join("results.tsv");
    let mut results = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&results_path)?;
    writeln!(
        results,
        "prompt\tk\trounds\tdrafted\taccepted\tout_len\ttape_identical\taccept_rate\tacc_per_cycle\ttok_per_cycle"
    )?;
    results.flush()?;

    for pf in &prompt_files {
        let name = pf.file_stem().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(pf)?;
        let rendered = tok.apply_chat_template(&[("user", text.trim())], true);
        let ids = tok.encode(&rendered, true);
        eprintln!(
            "[card3-probe] prompt {name}: {} chars -> {} tokens",
            text.len(),
            ids.len()
        );

        // Plain greedy oracle: prime + decode_step chain on a planned cache — the exact
        // program class the verify walk is decode-exact against (gate 5's plain_tape).
        let mut plain: Option<Vec<u32>> = None;
        if run_plain {
            hold_if_marked();
            let max_ctx = ids.len() + max_new + 16;
            let mut cache =
                memra_engine::cache::Cache::new_planned(&e, &model.cfg, &model.plan, max_ctx)?;
            let (logits0, _seed, _hiddens) = model.prime_cache(&e, &ids, &mut cache, 0)?;
            let mut tape = Vec::with_capacity(max_new);
            tape.push(argmax(&logits0) as u32);
            while tape.len() < max_new {
                let ll = model.decode_step(&e, *tape.last().unwrap(), &mut cache)?;
                tape.push(argmax(&ll) as u32);
            }
            let dec = tok.decode(&tape);
            std::fs::write(
                std::path::Path::new(&out_dir).join(format!("{name}-plain.txt")),
                &dec,
            )?;
            std::fs::write(
                std::path::Path::new(&out_dir).join(format!("{name}-plain.ids")),
                tape.iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
            )?;
            eprintln!("[card3-probe] {name} plain tape: {} tokens", tape.len());
            plain = Some(tape);
        }

        for &k in &ks {
            hold_if_marked();
            match model.generate_spec(&e, &ids, max_new, k) {
                Ok((out, drafted, accepted)) => {
                    let rounds = drafted.checked_div(k).unwrap_or(0);
                    let tape_id = match &plain {
                        Some(p) => {
                            if out == *p {
                                "yes"
                            } else {
                                "NO"
                            }
                        }
                        None => "-",
                    };
                    let acc_rate = accepted as f64 / drafted.max(1) as f64;
                    let acc_cyc = accepted as f64 / rounds.max(1) as f64;
                    let tok_cyc = (accepted + rounds) as f64 / rounds.max(1) as f64;
                    writeln!(
                        results,
                        "{name}\t{k}\t{rounds}\t{drafted}\t{accepted}\t{}\t{tape_id}\t{acc_rate:.4}\t{acc_cyc:.3}\t{tok_cyc:.3}",
                        out.len()
                    )?;
                    results.flush()?;
                    let dec = tok.decode(&out);
                    std::fs::write(
                        std::path::Path::new(&out_dir).join(format!("{name}-k{k}.txt")),
                        &dec,
                    )?;
                    eprintln!(
                        "[card3-probe] {name} K={k}: rounds={rounds} drafted={drafted} \
                         accepted={accepted} tape_identical={tape_id} acc/cycle={acc_cyc:.3} \
                         tok/cycle={tok_cyc:.3}"
                    );
                    if tape_id == "NO" {
                        let p = plain.as_ref().unwrap();
                        let div = out
                            .iter()
                            .zip(p.iter())
                            .position(|(a, b)| a != b)
                            .unwrap_or(out.len().min(p.len()));
                        eprintln!(
                            "[card3-probe] {name} K={k}: TAPE DIVERGENCE at token {div} \
                             (spec={:?} plain={:?})",
                            out.get(div),
                            p.get(div)
                        );
                    }
                }
                Err(err) => {
                    writeln!(results, "{name}\t{k}\tERR\t-\t-\t-\t-\t-\t-\t{err}")?;
                    results.flush()?;
                    eprintln!("[card3-probe] {name} K={k}: ERROR: {err}");
                }
            }
        }
    }
    eprintln!("[card3-probe] done: {}", results_path.display());
    Ok(())
}
