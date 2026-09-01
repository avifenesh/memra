//! Speculative-target versus live serving numeric-class localizer.
//!
//! For each requested T, replay identical token chunks through (a) T consecutive live B=1
//! batched serving steps and (b) one T-column speculative verify step. The comparison is full
//! logit bit identity per column. This is deliberately target-only: no drafter or accept walk can
//! hide whether the verifier itself matches the serving oracle.

use memra_engine::Engine;
use memra_engine::decode_batch::DevSamp;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: spec-serve-gate <model.gguf> [--steps N] [--ts 1,2]");
    let rest: Vec<String> = args.collect();
    let steps = rest
        .iter()
        .position(|a| a == "--steps")
        .and_then(|i| rest.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(2usize);
    let ts: Vec<usize> = rest
        .iter()
        .position(|a| a == "--ts")
        .and_then(|i| rest.get(i + 1))
        .map(|v| v.split(',').filter_map(|p| p.trim().parse().ok()).collect())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![1, 2]);
    let spec_k = rest
        .iter()
        .position(|a| a == "--spec-k")
        .and_then(|i| rest.get(i + 1))
        .and_then(|v| v.parse::<usize>().ok());
    let primary = std::env::var("MEMRA_PROBE_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .or_else(|| {
            std::env::var("MEMRA_PP_DEVICES")
                .ok()
                .and_then(|v| v.split(',').next().and_then(|s| s.trim().parse().ok()))
        })
        .unwrap_or(0usize);
    let e = Engine::new(primary)?;
    // GGUF file or a safetensors checkpoint directory — the gate exercises the same serving
    // paths either way (the loader maps ggml names from either container).
    let is_dir = std::path::Path::new(&path).is_dir();
    let (model, prompt) = if is_dir {
        let src = memra_gguf::source::SafetensorsSource::open(std::path::Path::new(&path))?;
        let prompt = if let Ok(prompt_path) = std::env::var("MEMRA_PROMPT_FILE") {
            let text = std::fs::read_to_string(prompt_path)?;
            let text = text.strip_suffix('\n').unwrap_or(&text);
            let tok = memra_tokenizer::Tokenizer::from_hf_dir(std::path::Path::new(&path))
                .map_err(|err| format!("HF tokenizer: {err}"))?;
            tok.encode(&tok.apply_chat_template(&[("user", text)], true), true)
        } else {
            (0..24u32).map(|j| 55 + j * 31).collect()
        };
        let model = if spec_k.is_some() {
            HybridModel::load_from_source(&e, &src)?
        } else {
            HybridModel::load_from_source_without_mtp(&e, &src)?
        };
        (model, prompt)
    } else {
        let gguf = GgufFile::open(&path)?;
        let prompt = if let Ok(prompt_path) = std::env::var("MEMRA_PROMPT_FILE") {
            let text = std::fs::read_to_string(prompt_path)?;
            let text = text.strip_suffix('\n').unwrap_or(&text);
            let tok = memra_tokenizer::Tokenizer::from_gguf(&gguf)?;
            tok.encode(&tok.apply_chat_template(&[("user", text)], true), true)
        } else {
            (0..24u32).map(|j| 55 + j * 31).collect()
        };
        let model = if spec_k.is_some() {
            HybridModel::load(&e, &gguf)?
        } else {
            HybridModel::load_without_mtp(&e, &gguf)?
        };
        (model, prompt)
    };
    let mut failed = 0usize;
    for t in ts {
        if t == 0 {
            return Err("verify width T must be positive".into());
        }
        let ctx = prompt.len() + steps * t + 64;
        let mut serving = memra_engine::pp::new_cache(&e, &model.cfg, ctx)?;
        let mut verify = memra_engine::pp::new_cache(&e, &model.cfg, ctx)?;
        let _ = model.prime_cache(&e, &prompt, &mut serving, 0)?;
        let _ = model.prime_cache(&e, &prompt, &mut verify, 0)?;
        let mut chunk = vec![*prompt.last().unwrap(); t];
        let mut diffs = 0usize;
        let mut max_abs = 0.0f32;
        let mut argmax_diffs = 0usize;
        let mut device_argmax_diffs = 0usize;
        for round in 0..steps {
            let pos0 = serving.pos;
            assert_eq!(
                verify.pos, pos0,
                "cache position drift before round {round}"
            );
            let mut reference = Vec::with_capacity(t);
            for &token in &chunk {
                let rows = {
                    let mut caches = [&mut serving];
                    model.decode_step_batch(&e, &[token], &mut caches)?
                };
                reference.push(rows.into_iter().next().unwrap());
            }
            let (got_d, _) = model.decode_step_t_h_emb_dev(&e, &chunk, pos0, &mut verify, None)?;
            let got = e.dtoh(&got_d)?;
            let mut got_pred_d = e.alloc_u32_zeroed(t)?;
            for col in 0..t {
                e.argmax_token_device_col(&got_d, col, reference[0].len(), &mut got_pred_d, col)?;
            }
            let got_pred = e.dtoh_u32(&got_pred_d)?;
            let n_vocab = reference[0].len();
            for col in 0..t {
                let actual = &got[col * n_vocab..(col + 1) * n_vocab];
                let expected = &reference[col];
                diffs += actual
                    .iter()
                    .zip(expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                max_abs = actual
                    .iter()
                    .zip(expected)
                    .fold(max_abs, |m, (a, b)| m.max((a - b).abs()));
                argmax_diffs += usize::from(argmax(actual) != argmax(expected));
                device_argmax_diffs += usize::from(got_pred[col] as usize != argmax(expected));
            }
            assert_eq!(
                verify.pos, serving.pos,
                "cache position drift after round {round}"
            );
            chunk = reference.iter().map(|row| argmax(row) as u32).collect();
        }
        println!(
            "T={t} rounds={steps} differing_logits={diffs} max_abs={max_abs:.9e} \
             argmax_diffs={argmax_diffs} device_argmax_diffs={device_argmax_diffs}"
        );
        failed += usize::from(diffs != 0 || device_argmax_diffs != 0);
    }
    if let Some(k) = spec_k {
        let ngen = steps.max(1);
        let ctx = prompt.len() + ngen + k + 16;
        let mut serving = memra_engine::pp::new_cache(&e, &model.cfg, ctx)?;
        let (prime_logits, _, _) = model.prime_cache(&e, &prompt, &mut serving, 0)?;
        let mut reference = vec![argmax(&prime_logits) as u32];
        while reference.len() < ngen {
            let token = *reference.last().unwrap();
            let rows = {
                let mut caches = [&mut serving];
                model.decode_step_batch(&e, &[token], &mut caches)?
            };
            reference.push(argmax(&rows[0]) as u32);
        }
        let mut live_cache = memra_engine::pp::new_cache(&e, &model.cfg, ctx)?;
        let (live_prime, _, _) = model.prime_cache(&e, &prompt, &mut live_cache, 0)?;
        let mut live = vec![argmax(&live_prime) as u32];
        while live.len() < ngen {
            let token = *live.last().unwrap();
            let samp = [Some(DevSamp::new(
                0.0,
                3407,
                live.len() as u32,
                0,
                1.0,
                0.0,
            ))];
            let (_, next) = {
                let mut caches = [&mut live_cache];
                model.decode_step_batch_sampled_lean_masked(
                    &e,
                    &[token],
                    &mut caches,
                    &samp,
                    &[],
                    true,
                )?
            };
            live.push(next[0].expect("greedy live row must return a device token"));
        }
        let mut session = model.new_session(&e, ctx)?;
        let (spec, drafted, accepted) =
            model.generate_spec_session(&e, &mut session, &prompt, ngen, k)?;
        let public_vs_live = reference.iter().zip(&live).position(|(a, b)| a != b);
        let first_diff = live.iter().zip(&spec).position(|(a, b)| a != b);
        println!(
            "spec-session K={k} ngen={ngen} returned={} drafted={drafted} accepted={accepted} \
             public_vs_live={public_vs_live:?} first_diff={first_diff:?}",
            spec.len(),
        );
        if let Some(i) = first_diff {
            println!(
                "spec-session mismatch index={i} serving={} spec={}",
                live[i], spec[i]
            );
            let lo = i.saturating_sub(4);
            let hi = (i + 5).min(live.len()).min(spec.len());
            println!("spec-session serving[{lo}..{hi}]={:?}", &live[lo..hi]);
            println!("spec-session spec[{lo}..{hi}]={:?}", &spec[lo..hi]);
        }
        failed += usize::from(first_diff.is_some() || spec.len() < live.len());
    }
    if failed != 0 {
        return Err(format!("spec-serve-gate: {failed} width(s) failed bit identity").into());
    }
    println!("spec-serve-gate: ALL GREEN");
    Ok(())
}
