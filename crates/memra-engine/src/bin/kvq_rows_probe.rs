//! kvq-rows-probe (research, memra#1177; branch research/kv-quant-repetitive, not for landing).
//!
//! The question: does the quantized KV cache (q8_0 K, q5_1 V) move the argmax on real
//! repetitive context? `argmax-margin-probe` answers it for a 12-row window and one prime row
//! per run, and steps the whole prompt tokenwise. This probe reads every row of the last
//! `--rows R` positions of one prompt under up to three programs, from one prefill each:
//!
//!   * `a` (prefill over f32 K/V): `forward_last`'s trunk and final norm once over the whole
//!     prompt (`HybridModel::forward_normed_rows`), then the lm_head on each window row, the
//!     same calls `forward_last` makes for its last row.
//!   * `b` (the prefill program over the quantized cache): `prime_cache` once over the whole
//!     prompt, then `prime_row_logits` on each window row of the hidden stack it returns.
//!     Every prime chunk attends over the q8_0/q5_1 cache, chunk 0 included (the grain-free
//!     chunk-invariance fix), so a row of a whole-prompt prime is the row a prime over the
//!     prefix ending there returns. `--check-row` measures that on the unit.
//!   * `c` (decode over the quantized cache), optional: `--decode eager` (`decode_step`) or
//!     `--decode batch` (the served batched program at B=1, the call `argmax-margin-probe
//!     --decode batch` makes). `--decode-from prime` primes the prompt up to the window with
//!     `prime_cache` and decodes the window rows tokenwise, the served shape of decode after a
//!     prompt. `--decode-from tokenwise` steps the whole prompt, as `argmax-margin-probe` does.
//!
//! Output: one JSON line per window row (`--out`), with each program's top-5 ids and logits,
//! the logit and rank of the prompt's next token, and per program pair the spread at the
//! contending ids (max |x - y| over both programs' top-2 ids, the gate's `delta@ids`) and the
//! largest difference over the vocabulary. A flip is margin-explained iff that spread exceeds
//! the smaller of the two top-2 margins (the gate's rule).
//!
//! `--check-row K` (absolute row) also runs `forward_last` over `prompt[..=K]` and a fresh
//! `prime_cache` over `prompt[..=K]` and prints whether their logits are bit-identical to rows
//! `a` and `b` at K. The prime's own last-row logits are always compared with row `T - 1` of
//! `b` (`PRIME-LAST`).
//!
//! usage: kvq-rows-probe <model.gguf> <prompt-file> --rows R --out FILE
//!        [--decode none|eager|batch] [--decode-from prime|tokenwise] [--check-row K]

use memra_engine::Engine;
use memra_engine::decode_batch::DevSamp;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_engine::pp::new_cache as new_model_cache;
use memra_gguf::GgufFile;
use memra_tokenizer::Tokenizer;
use std::io::Write;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Decode {
    None,
    Eager,
    Batch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Start {
    Prime,
    Tokenwise,
}

struct Args {
    model: String,
    prompt: String,
    rows: usize,
    out: String,
    decode: Decode,
    from: Start,
    check_row: Option<usize>,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let usage = "usage: kvq-rows-probe <model.gguf> <prompt-file> --rows R --out FILE \
                 [--decode none|eager|batch] [--decode-from prime|tokenwise] [--check-row K]";
    let mut pos = Vec::new();
    let (mut rows, mut out, mut decode, mut from, mut check_row) =
        (None, None, Decode::None, Start::Prime, None);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{a} needs a value; {usage}"))
        };
        match a.as_str() {
            "--rows" => rows = Some(val()?.parse::<usize>().map_err(|e| e.to_string())?),
            "--out" => out = Some(val()?),
            "--check-row" => check_row = Some(val()?.parse::<usize>().map_err(|e| e.to_string())?),
            "--decode" => {
                decode = match val()?.as_str() {
                    "none" => Decode::None,
                    "eager" => Decode::Eager,
                    "batch" => Decode::Batch,
                    o => return Err(format!("--decode {o}; {usage}")),
                }
            }
            "--decode-from" => {
                from = match val()?.as_str() {
                    "prime" => Start::Prime,
                    "tokenwise" => Start::Tokenwise,
                    o => return Err(format!("--decode-from {o}; {usage}")),
                }
            }
            s if s.starts_with("--") => return Err(format!("unknown option {s}; {usage}")),
            _ => pos.push(a.clone()),
        }
    }
    let [model, prompt] = pos.as_slice() else {
        return Err(usage.into());
    };
    let rows = rows.ok_or(usage)?;
    if rows == 0 {
        return Err("--rows must be positive".into());
    }
    Ok(Args {
        model: model.clone(),
        prompt: prompt.clone(),
        rows,
        out: out.ok_or(usage)?,
        decode,
        from,
        check_row,
    })
}

/// Top-k (id, logit), largest first; ties keep the lower id first, like `top2`'s strict `>`.
fn topk(l: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut best: Vec<(usize, f32)> = Vec::with_capacity(k + 1);
    for (i, &v) in l.iter().enumerate() {
        if best.len() == k && v <= best[k - 1].1 {
            continue;
        }
        let at = best.iter().position(|&(_, b)| v > b).unwrap_or(best.len());
        best.insert(at, (i, v));
        best.truncate(k);
    }
    best
}

/// Rank (0 = top) of id `j`: how many ids have a strictly larger logit.
fn rank_of(l: &[f32], j: usize) -> usize {
    let v = l[j];
    l.iter().filter(|&&x| x > v).count()
}

fn bits_equal(x: &[f32], y: &[f32]) -> (bool, usize, f32) {
    let mut n = 0usize;
    let mut maxd = 0f32;
    for (a, b) in x.iter().zip(y) {
        if a.to_bits() != b.to_bits() {
            n += 1;
            maxd = maxd.max((a - b).abs());
        }
    }
    (n == 0 && x.len() == y.len(), n, maxd)
}

/// The lm_head on row `k` of a normed `[R, n_embd]` stack: `forward_last`'s own tail.
fn head_row(
    e: &Engine,
    m: &HybridModel,
    normed: &cudarc::driver::CudaSlice<f32>,
    k: usize,
) -> Res<Vec<f32>> {
    let n_embd = m.cfg.n_embd as usize;
    let mut h = e.uninit(n_embd)?;
    e.copy_view_into(
        &mut h,
        0,
        &e.view(normed, (k + 1) * n_embd)
            .slice(k * n_embd..(k + 1) * n_embd),
        n_embd,
    )?;
    let logits = e.matmul(&m.output, &h, 1)?;
    e.dtoh(&logits)
}

fn pair_json(x: &[f32], y: &[f32]) -> String {
    let (x1, x2) = {
        let t = topk(x, 2);
        (t[0], t[1])
    };
    let (y1, y2) = {
        let t = topk(y, 2);
        (t[0], t[1])
    };
    let delta = [x1.0, x2.0, y1.0, y2.0]
        .iter()
        .map(|&i| (x[i] - y[i]).abs())
        .fold(0f32, f32::max);
    let maxdiff = x
        .iter()
        .zip(y)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let (mx, my) = (x1.1 - x2.1, y1.1 - y2.1);
    let flip = x1.0 != y1.0;
    format!(
        "{{\"flip\":{flip},\"delta\":{delta:e},\"maxdiff\":{maxdiff:e},\"min_margin\":{:e},\"explained\":{}}}",
        mx.min(my),
        flip && delta > mx.min(my)
    )
}

fn prog_json(l: &[f32], next: Option<u32>) -> String {
    let top: Vec<String> = topk(l, 5)
        .iter()
        .map(|(i, v)| format!("[{i},{v:e}]"))
        .collect();
    let nx = match next {
        Some(j) => format!(
            ",\"next_logit\":{:e},\"next_rank\":{}",
            l[j as usize],
            rank_of(l, j as usize)
        ),
        None => String::new(),
    };
    format!("{{\"top\":[{}]{nx}}}", top.join(","))
}

fn main() -> Res<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let a = parse_args(&argv)?;
    let e = Engine::new(0)?;
    let g = GgufFile::open(&a.model)?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    let tok = Tokenizer::from_gguf(&g).map_err(|err| format!("tokenizer: {err}"))?;
    let text = std::fs::read_to_string(&a.prompt)?;
    let prompt = tok.encode(&text, true);
    let t = prompt.len();
    if a.rows + 16 >= t {
        return Err("prompt shorter than the window plus the prime floor".into());
    }
    let first = t - a.rows;
    if let Some(k) = a.check_row
        && !(first..t).contains(&k)
    {
        return Err(format!("--check-row {k} is outside the window [{first}, {t})").into());
    }
    let n_embd = model.cfg.n_embd as usize;
    println!(
        "kvq-rows-probe: T={t} rows={} first={first} decode={:?} from={:?} sm_count={} model={}",
        a.rows,
        a.decode,
        a.from,
        e.sm_count(),
        a.model
    );
    let ids_hash = {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for id in &prompt {
            for b in id.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        }
        h
    };
    println!("prompt ids fnv64={ids_hash:016x}");
    println!(
        "env: MEMRA_FA_SPLIT={} MEMRA_FAST={} MEMRA_PRIME_CHUNK={} MEMRA_SERVE_B1FAST={}",
        std::env::var("MEMRA_FA_SPLIT").unwrap_or_else(|_| "<unset>".into()),
        std::env::var("MEMRA_FAST").unwrap_or_else(|_| "<unset>".into()),
        std::env::var("MEMRA_PRIME_CHUNK").unwrap_or_else(|_| "<unset>".into()),
        std::env::var("MEMRA_SERVE_B1FAST").unwrap_or_else(|_| "<unset>".into()),
    );

    // --- a: prefill over f32 K/V, one trunk pass ---
    let t0 = Instant::now();
    let a_rows = model.forward_normed_rows(&e, &prompt, first)?;
    e.stream().synchronize()?;
    println!(
        "phase a (forward trunk T={t}): {:.1} s",
        t0.elapsed().as_secs_f64()
    );

    // --- b: prime over the quantized cache, one pass ---
    let t0 = Instant::now();
    let b_rows = {
        let mut pc = new_model_cache(&e, &model.cfg, t + 8)?;
        let (l_bp, _, hiddens) = model.prime_cache(&e, &prompt, &mut pc, 0)?;
        let mut rows = e.uninit(a.rows * n_embd)?;
        e.copy_view_into(
            &mut rows,
            0,
            &e.view(&hiddens, t * n_embd)
                .slice(first * n_embd..t * n_embd),
            a.rows * n_embd,
        )?;
        let last = model.prime_row_logits(&e, &rows, a.rows - 1)?;
        let (same, n, maxd) = bits_equal(&l_bp, &last);
        println!(
            "PRIME-LAST: row {} of the hidden stack vs the prime's own logits: bit-identical={same} \
             differing={n} maxdiff={maxd:e}",
            t - 1
        );
        rows
    };
    e.stream().synchronize()?;
    println!("phase b (prime T={t}): {:.1} s", t0.elapsed().as_secs_f64());

    if let Some(k) = a.check_row {
        let t0 = Instant::now();
        let ref_a = model.forward_last(&e, &prompt[..=k])?;
        let got_a = head_row(&e, &model, &a_rows, k - first)?;
        let (same, n, maxd) = bits_equal(&ref_a, &got_a);
        println!(
            "CHECK-A row {k}: forward_last(prompt[..={k}]) vs window row: bit-identical={same} \
             differing={n} maxdiff={maxd:e}"
        );
        let mut pc = new_model_cache(&e, &model.cfg, k + 1 + 8)?;
        let (ref_b, _, _) = model.prime_cache(&e, &prompt[..=k], &mut pc, 0)?;
        drop(pc);
        let got_b = model.prime_row_logits(&e, &b_rows, k - first)?;
        let (same, n, maxd) = bits_equal(&ref_b, &got_b);
        println!(
            "CHECK-B row {k}: prime_cache(prompt[..={k}]) vs window row: bit-identical={same} \
             differing={n} maxdiff={maxd:e}"
        );
        println!("check phase: {:.1} s", t0.elapsed().as_secs_f64());
    }

    let mut out = std::io::BufWriter::new(std::fs::File::create(&a.out)?);
    let emit = |out: &mut std::io::BufWriter<std::fs::File>,
                k: usize,
                c: Option<&[f32]>,
                extra: &str|
     -> Res<()> {
        let i = first + k;
        let next = prompt.get(i + 1).copied();
        let la = head_row(&e, &model, &a_rows, k)?;
        let lb = model.prime_row_logits(&e, &b_rows, k)?;
        let mut line = format!(
            "{{\"row\":{i},\"tok\":{},\"next\":{},\"a\":{},\"b\":{},\"ab\":{}",
            prompt[i],
            next.map(|n| n.to_string()).unwrap_or_else(|| "null".into()),
            prog_json(&la, next),
            prog_json(&lb, next),
            pair_json(&la, &lb)
        );
        if let Some(lc) = c {
            line.push_str(&format!(
                ",\"c\":{},\"ac\":{},\"bc\":{}{extra}",
                prog_json(lc, next),
                pair_json(&la, lc),
                pair_json(&lb, lc)
            ));
        }
        line.push('}');
        writeln!(out, "{line}")?;
        Ok(())
    };

    let t0 = Instant::now();
    match a.decode {
        Decode::None => {
            for k in 0..a.rows {
                emit(&mut out, k, None, "")?;
            }
        }
        Decode::Eager | Decode::Batch => {
            let batch = a.decode == Decode::Batch && !model.served_decode_is_eager();
            if a.decode == Decode::Batch && !batch {
                println!(
                    "decode: the batched scheduler routes this model eager; stepping decode_step"
                );
            }
            let mut cache = if batch {
                memra_engine::pp::new_cache_planned(&e, &model.cfg, &model.plan, t + 8)?
            } else {
                new_model_cache(&e, &model.cfg, t + 8)?
            };
            let greedy = [Some(DevSamp::new(0.0, 0, 0, 0, 1.0, 0.0))];
            let step = |cache: &mut memra_engine::cache::Cache,
                        tk: u32,
                        want: bool|
             -> Res<Option<(Vec<f32>, bool)>> {
                if batch {
                    if want {
                        cache.last_logits_dev = None;
                    }
                    let (_, next) = model.decode_step_batch_sampled_lean_masked(
                        &e,
                        &[tk],
                        &mut [cache],
                        &greedy,
                        &[None],
                        true,
                    )?;
                    if !want {
                        return Ok(None);
                    }
                    let parked = cache
                        .last_logits_dev
                        .as_ref()
                        .ok_or("served batch step parked no logits row")?;
                    let row = e.dtoh(parked)?;
                    let token = next
                        .first()
                        .copied()
                        .flatten()
                        .ok_or("served batch step returned no device-sampled token")?;
                    let agree = token as usize == argmax(&row);
                    Ok(Some((row, agree)))
                } else {
                    let l = model.decode_step(&e, tk, cache)?;
                    Ok(if want { Some((l, true)) } else { None })
                }
            };
            match a.from {
                Start::Prime => {
                    let _ = model.prime_cache(&e, &prompt[..first], &mut cache, 0)?;
                }
                Start::Tokenwise => {
                    for &tk in &prompt[..first] {
                        step(&mut cache, tk, false)?;
                    }
                }
            }
            e.stream().synchronize()?;
            println!(
                "phase c prefix ({:?}, {first} rows): {:.1} s",
                a.from,
                t0.elapsed().as_secs_f64()
            );
            let t1 = Instant::now();
            let mut dev_mismatch = 0usize;
            for k in 0..a.rows {
                let (lc, agree) = step(&mut cache, prompt[first + k], true)?.expect("window row");
                if !agree {
                    dev_mismatch += 1;
                }
                emit(&mut out, k, Some(&lc), "")?;
                if k % 256 == 255 {
                    out.flush()?;
                    println!(
                        "  window row {}/{}: {:.1} s",
                        k + 1,
                        a.rows,
                        t1.elapsed().as_secs_f64()
                    );
                }
            }
            println!(
                "phase c window ({} rows, {}): {:.1} s; device-sampled token != host argmax at {dev_mismatch} rows",
                a.rows,
                if batch {
                    "served batch B=1"
                } else {
                    "eager decode_step"
                },
                t1.elapsed().as_secs_f64()
            );
        }
    }
    out.flush()?;
    println!("rows written: {} -> {}", a.rows, a.out);
    println!("total after load: {:.1} s", t0.elapsed().as_secs_f64());
    Ok(())
}
