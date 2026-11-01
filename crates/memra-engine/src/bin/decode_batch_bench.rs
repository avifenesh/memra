//! decode-batch-bench: aggregate decode throughput vs batch size (ARCHITECTURE-H100.md B2').
//!
//! The number that funds the multi-tenant thesis: decode is weight-stream-bound, so
//! aggregate tok/s should scale near-linearly with B until attention/launch overheads
//! bite. Reports per-B aggregate and per-seq rates over N timed reps of `steps` batched
//! decode steps (greedy), medians over reps. Run AFTER decode-batch-gate is green.
//!
//! Usage: decode-batch-bench <model.gguf> [--steps 128] [--reps 5] [--batches 1,2,4,8]
//!
//! PP-N (pp2-batch 2026-08-06): every cache here comes from `pp::new_cache`, which is
//! `Cache::new` verbatim with the door shut and `Cache::new_ppn` with it open. Under an open
//! door this bench previously allocated STAGE-1's KV on dev0 (primary), so the split path's
//! remote stage peer-read its own cache every step — a measurement that would have understated
//! batched PP-N and, worse, looked like a stage-split cost rather than a harness bug. The door
//! must also be set BEFORE this binary loads (weight sharding is load-time), which it is:
//! the env is read inside `HybridModel::load_*`.

use memra_engine::Engine;
use memra_engine::cache::Cache;
use memra_engine::decode_batch::DevSamp;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_engine::pp::new_cache;
use memra_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect(
        "usage: decode-batch-bench <model.gguf> [--steps N] [--reps R] [--batches 1,2,4,8]",
    );
    let rest: Vec<String> = args.collect();
    let steps: usize = rest
        .iter()
        .position(|a| a == "--steps")
        .and_then(|i| rest.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let reps: usize = rest
        .iter()
        .position(|a| a == "--reps")
        .and_then(|i| rest.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let batches: Vec<usize> = rest
        .iter()
        .position(|a| a == "--batches")
        .and_then(|i| rest.get(i + 1))
        .map(|v| v.split(',').filter_map(|s| s.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8]);

    // inc3 (3a) CHUNK-SIZE SWEEP: `--seqs N --chunk C` advances N sequences per tick via
    // ceil(N/C) chunked decode_step_batch calls (the worker's group_chunks shape) and prints
    // one aggregate tok/s line — the per-tick cost of chunking policy C for an N-seq batch.
    // One chunk config per invocation (env-dependent dispatch reads once); interleave
    // invocations at the script level for the x5 medians.
    let seqs: Option<usize> = rest
        .iter()
        .position(|a| a == "--seqs")
        .and_then(|i| rest.get(i + 1))
        .and_then(|v| v.parse().ok());
    let chunk: usize = rest
        .iter()
        .position(|a| a == "--chunk")
        .and_then(|i| rest.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);

    // Primary device follows MEMRA_PP_DEVICES[0] when set (ppn-gate's convention): stage 0's
    // engine IS the primary engine, so a mismatch would make BOTH placement orders run stage 0
    // through a non-primary engine and quietly stop testing the `dev == primary && s == 0`
    // shared-engine case. Unset = device 0, byte-identical to the historic bench.
    let primary_dev: usize = std::env::var("MEMRA_PP_DEVICES")
        .ok()
        .and_then(|v| v.split(',').next().and_then(|s| s.trim().parse().ok()))
        .unwrap_or(0);
    let e = Engine::new(primary_dev)?;
    // A safetensors DIRECTORY loads through the TensorSource path (run-gen's branch): the
    // step37-class checkpoints this bench now has to measure are not GGUF files, and the
    // batch-vs-B question is exactly the one their serving config raises.
    let (model, arch) = if std::path::Path::new(&path).is_dir() {
        let dir = std::path::Path::new(&path);
        let src: Box<dyn memra_gguf::source::TensorSource> = if dir.join("manifest.json").exists() {
            Box::new(memra_gguf::source::Hy3RepackSource::open(dir)?)
        } else {
            Box::new(memra_gguf::source::SafetensorsSource::open(dir)?)
        };
        let arch = format!("{:?}", src.config().arch);
        (
            HybridModel::load_from_source_without_mtp(&e, src.as_ref())?,
            arch,
        )
    } else {
        let g = GgufFile::open(&path)?;
        let arch = g.arch().unwrap_or("?").to_string();
        (HybridModel::load_without_mtp(&e, &g)?, arch)
    };
    println!(
        "loaded {arch} ({} layers); steps={steps} reps={reps} batches={batches:?} \
              primary_dev={primary_dev}",
        model.layers.len()
    );

    // `--verify-t 1,2,..,16` (lane/batch-spec, 2026-11-01): the cost C(R) of ONE target
    // verify forward over R rows at a fixed depth (`--ctx`, default 512), the same
    // `decode_step_t` kernel chain the spec routes run, synced by a one-token device argmax
    // readback. Each rep restores the primed cache first, so every rep sees the same depth.
    // Paired with the plain `--batches` curve on one load, this prices a pooled cross-session
    // verify (R = sum of the sessions' T) against plain batching at B rows.
    let verify_t: Option<Vec<usize>> = rest
        .iter()
        .position(|a| a == "--verify-t")
        .and_then(|i| rest.get(i + 1))
        .map(|v| v.split(',').filter_map(|s| s.parse().ok()).collect());
    // `--pooled 4,4,4,4` (lane/batch-spec): one session per entry, entry = its verify rows T_i,
    // each primed to its own depth (`--ctx` + 37*i). Pins the pooled cross-session verify
    // BITWISE against the per-session verify funnel (every logit row, plus one follow-up
    // T=1 step per session that reads the state both paths wrote), then times pooled vs the
    // serial sum of per-session verifies.
    let pooled: Option<Vec<usize>> = rest
        .iter()
        .position(|a| a == "--pooled")
        .and_then(|i| rest.get(i + 1))
        .map(|v| v.split(',').filter_map(|s| s.parse().ok()).collect());
    if let Some(ts) = pooled {
        let depth: usize = rest
            .iter()
            .position(|a| a == "--ctx")
            .and_then(|i| rest.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        let vocab = model.cfg.n_vocab.max(64) - 16;
        let n_vocab = model.cfg.n_vocab as usize;
        let mut caches: Vec<Cache> = Vec::new();
        let mut snaps = Vec::new();
        let mut toks: Vec<Vec<u32>> = Vec::new();
        let mut pos0s: Vec<usize> = Vec::new();
        for (i, &t) in ts.iter().enumerate() {
            let d = depth + 37 * i;
            let prompt: Vec<u32> = (0..d as u32)
                .map(|j| (55 + i as u32 * 97 + j * 31) % vocab)
                .collect();
            let mut c = new_cache(&e, &model.cfg, d + 64)?;
            let _ = model.prime_cache(&e, &prompt, &mut c, 0)?;
            snaps.push(c.snapshot(&e)?);
            toks.push(
                (0..t as u32)
                    .map(|j| (977 + i as u32 * 61 + j * 131) % vocab)
                    .collect(),
            );
            pos0s.push(c.pos);
            caches.push(c);
        }
        let _exact = e.exact_scope(true);
        let follow = |c: &mut Cache, i: usize| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            let p = c.pos;
            model.decode_step_t(&e, &[(4242 + i as u32 * 7) % vocab], p, c)
        };
        // Reference: per-session verify, then one follow-up step.
        let mut ref_rows: Vec<Vec<f32>> = Vec::new();
        let mut ref_follow: Vec<Vec<f32>> = Vec::new();
        for i in 0..ts.len() {
            caches[i].rollback(&e, &snaps[i], 0)?;
            let (l, _) =
                model.decode_step_t_h_emb_dev(&e, &toks[i], pos0s[i], &mut caches[i], None)?;
            ref_rows.push(e.dtoh(&l)?);
            ref_follow.push(follow(&mut caches[i], i)?);
        }
        for i in 0..ts.len() {
            caches[i].rollback(&e, &snaps[i], 0)?;
        }
        let tok_refs: Vec<&[u32]> = toks.iter().map(|v| v.as_slice()).collect();
        let pooled_logits = {
            let mut refs: Vec<&mut Cache> = caches.iter_mut().collect();
            model.verify_pooled(&e, &mut refs, &tok_refs)?
        };
        let Some(pl) = pooled_logits else {
            println!("POOLED {ts:?}: NOT POOLABLE (shape refused)");
            return Ok(());
        };
        let pl = e.dtoh(&pl)?;
        let (mut row_mismatch, mut follow_mismatch, mut r0) = (0usize, 0usize, 0usize);
        for i in 0..ts.len() {
            let n = ts[i] * n_vocab;
            if pl[r0 * n_vocab..r0 * n_vocab + n] != ref_rows[i][..n] {
                row_mismatch += 1;
                let md = pl[r0 * n_vocab..r0 * n_vocab + n]
                    .iter()
                    .zip(&ref_rows[i][..n])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!("  session {i}: verify rows DIFFER (max abs {md:e})");
            }
            let f = follow(&mut caches[i], i)?;
            if f != ref_follow[i] {
                follow_mismatch += 1;
                println!("  session {i}: follow-up step after the verify DIFFERS");
            }
            r0 += ts[i];
        }
        println!(
            "POOLED {ts:?} R={r0}: verify rows {} / follow-up steps {} ({} sessions)",
            if row_mismatch == 0 {
                "BITWISE EQUAL"
            } else {
                "MISMATCH"
            },
            if follow_mismatch == 0 {
                "BITWISE EQUAL"
            } else {
                "MISMATCH"
            },
            ts.len()
        );
        // Timing: serial per-session verifies vs one pooled verify, same restored states.
        let mut serial: Vec<f64> = Vec::new();
        let mut pooled_ms: Vec<f64> = Vec::new();
        for rep in 0..=reps {
            for i in 0..ts.len() {
                caches[i].rollback(&e, &snaps[i], 0)?;
            }
            let t0 = std::time::Instant::now();
            for i in 0..ts.len() {
                let (l, _) =
                    model.decode_step_t_h_emb_dev(&e, &toks[i], pos0s[i], &mut caches[i], None)?;
                let am = e.argmax_token_device(&l, n_vocab)?;
                let _ = e.dtoh_u32_one(&am)?;
            }
            let s_ms = t0.elapsed().as_secs_f64() * 1e3;
            for i in 0..ts.len() {
                caches[i].rollback(&e, &snaps[i], 0)?;
            }
            let t0 = std::time::Instant::now();
            let l = {
                let mut refs: Vec<&mut Cache> = caches.iter_mut().collect();
                model.verify_pooled(&e, &mut refs, &tok_refs)?.unwrap()
            };
            let am = e.argmax_token_device(&l, n_vocab)?;
            let _ = e.dtoh_u32_one(&am)?;
            let p_ms = t0.elapsed().as_secs_f64() * 1e3;
            if rep > 0 {
                serial.push(s_ms);
                pooled_ms.push(p_ms);
            }
        }
        serial.sort_by(|a, b| a.partial_cmp(b).unwrap());
        pooled_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (sm, pm) = (serial[serial.len() / 2], pooled_ms[pooled_ms.len() / 2]);
        println!(
            "POOLED-TIME {ts:?} depth={depth}: serial {sm:.3} ms, pooled {pm:.3} ms, \
             {:.2}x (median of {reps})",
            sm / pm
        );
        return Ok(());
    }

    if let Some(ts) = verify_t {
        let prompt_t: usize = rest
            .iter()
            .position(|a| a == "--ctx")
            .and_then(|i| rest.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        let t_max = ts.iter().copied().max().unwrap_or(1);
        let vocab = model.cfg.n_vocab.max(64) - 16;
        let prompt: Vec<u32> = (0..prompt_t as u32)
            .map(|j| (55 + j * 31) % vocab)
            .collect();
        let mut c = new_cache(&e, &model.cfg, prompt_t + t_max + 64)?;
        let _ = model.prime_cache(&e, &prompt, &mut c, 0)?;
        let snap = c.snapshot(&e)?;
        let n_vocab = model.cfg.n_vocab as usize;
        // The exact tier (m <= 16) the batched decode and dspark verify run under; without the
        // scope, T = 16 falls into the prime GEMM class (PRIME_MIN_T).
        let _exact = e.exact_scope(true);
        let mut base_ms = None;
        for &t in &ts {
            let toks: Vec<u32> = (0..t as u32).map(|j| (977 + j * 131) % vocab).collect();
            let mut ms: Vec<f64> = Vec::new();
            for rep in 0..=reps {
                c.rollback(&e, &snap, 0)?;
                let t0 = std::time::Instant::now();
                let (logits, _h) =
                    model.decode_step_t_h_emb_dev(&e, &toks, prompt_t, &mut c, None)?;
                // column 0 is enough: the stream is ordered, so this readback waits for the
                // whole forward.
                let am = e.argmax_token_device(&logits, n_vocab)?;
                let _ = e.dtoh_u32_one(&am)?;
                if rep > 0 {
                    ms.push(t0.elapsed().as_secs_f64() * 1e3);
                }
            }
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = ms[ms.len() / 2];
            let base = *base_ms.get_or_insert(med);
            println!(
                "VERIFY T={t} depth={prompt_t}: {med:.3} ms (median of {reps}, p10 {:.3} p90 {:.3}) \
                 = {:.3}x T=first, {:.3} ms/row",
                ms[ms.len() / 10],
                ms[ms.len() * 9 / 10],
                med / base,
                med / t as f64
            );
        }
        return Ok(());
    }

    if let Some(n_seqs) = seqs {
        let prompt_t: usize = rest
            .iter()
            .position(|a| a == "--ctx")
            .and_then(|i| rest.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        let ctx = prompt_t + n_seqs * 7 + 64 + (steps + 8) * (reps + 1);
        let mut caches: Vec<Cache> = Vec::new();
        let mut toks: Vec<u32> = Vec::new();
        for i in 0..n_seqs {
            // Ids MUST stay inside the vocabulary: the historic formula reaches 136k, which is
            // past step37's vocab, and an out-of-range embedding gather yields garbage that
            // trips the non-finite activation guard before any timing happens.
            let vocab = model.cfg.n_vocab.max(64) - 16;
            let prompt: Vec<u32> = (0..(prompt_t as u32) + i as u32 * 7)
                .map(|j| (55 + i as u32 * 97 + j * 31) % vocab)
                .collect();
            let mut c = new_cache(&e, &model.cfg, ctx)?;
            let _ = model.prime_cache(&e, &prompt, &mut c, 0)?;
            toks.push(*prompt.last().unwrap());
            caches.push(c);
        }
        // MEMRA_DBB_SAMP=greedy|temp|filt: run the SERVE tick program
        // (decode_step_batch_sampled_lean_masked, lean on) with that device-sample meta on
        // every row — the sampled-serve wall isolator (lane/moebatch-q35moe). Unset = the
        // plain decode_step_batch + host argmax loop (the original bench).
        let samp_mode = std::env::var("MEMRA_DBB_SAMP").ok();
        let mut rates: Vec<f64> = Vec::new();
        // Per-seq FNV over every emitted token (all reps): two runs with identical
        // starts print identical lines iff the greedy streams are identical — the
        // door-on/door-off identity gate for batched-walk changes.
        let mut fps: Vec<u64> = vec![0xcbf29ce484222325; n_seqs];
        for rep in 0..=reps {
            let t0 = std::time::Instant::now();
            for step in 0..steps {
                let mut next: Vec<u32> = Vec::with_capacity(n_seqs);
                for (cs, ts) in caches.chunks_mut(chunk).zip(toks.chunks(chunk)) {
                    let mut refs: Vec<&mut Cache> = cs.iter_mut().collect();
                    match samp_mode.as_deref() {
                        Some(mode) => {
                            let meta = match mode {
                                "greedy" => DevSamp::new(0.0, 0, 0, 0, 1.0, 0.0),
                                "temp" => DevSamp::new(0.6, 7, step as u32, 0, 1.0, 0.0),
                                _ => DevSamp::new(0.6, 7, step as u32, 20, 0.95, 0.0),
                            };
                            let samp = vec![Some(meta); refs.len()];
                            let masks: Vec<Option<(&cudarc::driver::CudaSlice<u32>, usize)>> =
                                vec![None; refs.len()];
                            let (_rows, toks_out) = model.decode_step_batch_sampled_lean_masked(
                                &e, ts, &mut refs, &samp, &masks, true,
                            )?;
                            for t in &toks_out {
                                next.push(t.expect("device sample requested"));
                            }
                        }
                        None => {
                            let logits = model.decode_step_batch(&e, ts, &mut refs)?;
                            for l in &logits {
                                next.push(argmax(l) as u32);
                            }
                        }
                    }
                }
                for (i, t) in next.iter().enumerate() {
                    fps[i] = fps[i].wrapping_mul(1099511628211).wrapping_add(*t as u64);
                }
                toks = next;
            }
            let dt = t0.elapsed().as_secs_f64();
            if rep > 0 {
                rates.push((n_seqs * steps) as f64 / dt);
            }
        }
        rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let agg = rates[rates.len() / 2];
        println!(
            "CHUNKSWEEP seqs={n_seqs} chunk={chunk}: aggregate {agg:.1} tok/s \
                  ({:.2} ms/tick, median of {reps})",
            n_seqs as f64 / agg * 1e3
        );
        let fp_hex: Vec<String> = fps.iter().map(|f| format!("{f:016x}")).collect();
        println!("TOKFP seqs={n_seqs} chunk={chunk} [{}]", fp_hex.join(" "));
        if memra_engine::decode_batch::batch_phase_on() {
            println!("{}", memra_engine::decode_batch::batch_phase_report());
        }
        return Ok(());
    }

    let ctx_extra: usize = rest
        .iter()
        .position(|a| a == "--ctx")
        .and_then(|i| rest.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let ctx = ctx_extra + 8 * 7 + 64 + (steps + 8) * (reps + 1);
    let mut results: Vec<(usize, f64, f64)> = Vec::new();

    for &b_n in &batches {
        // Fresh caches per batch size; prompts >= 16 tokens (PRIME_MIN_T), distinct.
        let mut caches: Vec<Cache> = Vec::new();
        let mut toks: Vec<u32> = Vec::new();
        // Prompts sized to the SERVING regime (default 512 tokens; --ctx overrides): short
        // prompts under fa_vec_min_tkv silently fall to the f32 attention path and skew the
        // step profile (2026-07-26 nsys finding: fa_decode_f32 at 21% on 24-tok prompts).
        let prompt_t: usize = rest
            .iter()
            .position(|a| a == "--ctx")
            .and_then(|i| rest.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        for i in 0..b_n {
            let prompt: Vec<u32> = (0..(prompt_t as u32) + i as u32 * 7)
                .map(|j| 55 + i as u32 * 97 + j * 31)
                .collect();
            let mut c = new_cache(&e, &model.cfg, ctx)?;
            let _ = model.prime_cache(&e, &prompt, &mut c, 0)?;
            toks.push(*prompt.last().unwrap());
            caches.push(c);
        }
        // Warmup rep + timed reps.
        let mut rates: Vec<f64> = Vec::new();
        for rep in 0..=reps {
            let t0 = std::time::Instant::now();
            for _ in 0..steps {
                let mut cache_refs: Vec<&mut Cache> = caches.iter_mut().collect();
                let logits = model.decode_step_batch(&e, &toks, &mut cache_refs)?;
                for (bi, l) in logits.iter().enumerate() {
                    toks[bi] = argmax(l) as u32;
                }
            }
            let dt = t0.elapsed().as_secs_f64();
            if rep > 0 {
                rates.push((b_n * steps) as f64 / dt);
            }
        }
        rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let agg = rates[rates.len() / 2];
        let per = agg / b_n as f64;
        println!("B={b_n}: aggregate {agg:.1} tok/s, per-seq {per:.1} tok/s (median of {reps})");
        results.push((b_n, agg, per));
    }

    // Scaling summary vs B=1.
    if let Some(&(_, base, _)) = results.first() {
        for &(b_n, agg, _) in &results {
            println!("scale B={b_n}: {:.2}x aggregate vs B=1", agg / base);
        }
    }

    // MEMRA_BATCH_PHASE=1: engine tick decomposition (sync-bounded — shares rank, not walltime).
    if memra_engine::decode_batch::batch_phase_on() {
        println!("{}", memra_engine::decode_batch::batch_phase_report());
    }

    // Host sample/emit cost at the serving vocab (the worker tick's per-seq host stage): time
    // greedy argmax vs the load harness's temp-0.7 sample on REAL last-step logits, per row.
    {
        use memra_engine::sampler::{Sampler, SamplerConfig};
        let b_n = *batches.last().unwrap();
        let mut caches: Vec<Cache> = Vec::new();
        let mut toks: Vec<u32> = Vec::new();
        for i in 0..b_n {
            let prompt: Vec<u32> = (0..512u32).map(|j| 55 + i as u32 * 97 + j * 31).collect();
            let mut c = new_cache(&e, &model.cfg, 640)?;
            let _ = model.prime_cache(&e, &prompt, &mut c, 0)?;
            toks.push(*prompt.last().unwrap());
            caches.push(c);
        }
        let mut cache_refs: Vec<&mut Cache> = caches.iter_mut().collect();
        let rows = model.decode_step_batch(&e, &toks, &mut cache_refs)?;
        let reps = 50usize;
        let t0 = std::time::Instant::now();
        let mut sink = 0u32;
        for _ in 0..reps {
            for r in &rows {
                sink = sink.wrapping_add(argmax(r) as u32);
            }
        }
        let greedy_us = t0.elapsed().as_secs_f64() * 1e6 / (reps * rows.len()) as f64;
        let mut smp = Sampler::new(SamplerConfig {
            temperature: 0.7,
            seed: 7,
            ..Default::default()
        });
        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            for r in &rows {
                sink = sink.wrapping_add(smp.sample(r));
            }
        }
        let temp_us = t1.elapsed().as_secs_f64() * 1e6 / (reps * rows.len()) as f64;
        println!(
            "[host-sample] n_vocab={} greedy argmax {greedy_us:.0} us/row | temp0.7 sample \
                  {temp_us:.0} us/row | x B={b_n} rows/tick = {:.2} ms (greedy) / {:.2} ms (temp) [sink {sink}]",
            rows[0].len(),
            greedy_us * b_n as f64 / 1e3,
            temp_us * b_n as f64 / 1e3
        );
    }
    Ok(())
}
