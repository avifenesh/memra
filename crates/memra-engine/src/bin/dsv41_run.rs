//! DeepSeek-V4.1-Flash, the whole trunk: the native checkpoint runner for
//! `memra model verify checkpoint --against deepseek_v41 --native-runner <this binary>`, the
//! long-context runner of the lane's gates, and the decode loop (greedy generation and
//! teacher-forced decode) over `dsv41_gpu::decode`.
//!
//! The 40 trunk layers run one at a time, each layer's weights loaded from the safetensors mmap,
//! used and freed (about 7.5 GB a layer, 384 MXFP4 experts included), so the 510 GB checkpoint runs
//! on a single 96 GB card. The program is the checkpoint's own (`crate::dsv41_gpu`): bf16
//! activations, the vendor's per-32 FP8 / FP4 quantize-dequantize steps, FP8 32x32 and MXFP4
//! weights as stored, fp32 accumulation. A prefill of the given tokens, then the fp32 head; with a
//! decode mode, one token per call after it, each sequence's window rings, compressor groups,
//! compressed KV, index keys and token history carried the vendor's way.
//!
//! usage: dsv41-run CKPT_DIR [TOKEN ...] [--ids FILE] [--token-map FILE] [--chunk N]
//!                  [--last-logits FILE] [--tf DIR] [--layers N] [--devices 0,1]
//!                  [--gen N | --force FILE] [--out DIR] [--cases FILE] [--resident N]
//!                  [--steps K] [--eos ID] [--full-steps N]
//!   MEMRA_FULL_PREC=1        required: the route has one numeric program, the checkpoint's
//!   MEMRA_ORACLE_OUT=FILE    write the `memra-checkpoint-oracle-v1` tape of the last token
//!   MEMRA_DSV41_TOKEN_MAP=F  the Engram compressed token map (raw i32), when --token-map is absent
//!   --ids FILE               prompt ids as raw little-endian i32 (instead of TOKEN args)
//!   --last-logits FILE       the last token's fp32 logits, raw
//!   --tf DIR                 teacher-forced scoring of every position: lse.f32, argmax.i32,
//!                            nll.f32 (of the next prompt token), top_ids.i32 and top_vals.f32
//!                            (the 32 largest logits, the oracle tape's top-32), raw arrays plus
//!                            meta.json
//!   --devices 0,1            run on N ranks (`dsv41_gpu::tp`: heads, output groups, `wo_b` rows,
//!                            experts and head rows split; gathers only, so the logits are the
//!                            one-card bytes); ranks may repeat a device
//! Decode (per-step files in --out, see `step_files`):
//!   --gen N                  greedy generation: up to N tokens, stopping at the end-of-sequence id
//!   --force FILE             teacher-forced decode: FILE (raw i32) holds the token taken at each
//!                            step; step k > 0 feeds the token of step k - 1 as a decode call, so
//!                            step k scores what the vendor's decode step k scored. All steps run
//!                            in one layer-by-layer pass (the weights stream once).
//!   --cases FILE             several sequences in one run, one per line: `ID PROMPT_FILE
//!                            [FORCE_FILE]` (raw i32; with FORCE_FILE teacher-forced, else greedy
//!                            with --gen), each writing --out/ID; every forward runs all of them
//!                            through each layer, each sequence alone through each block
//!   --resident N             keep the first N layers' weights on the card across the greedy
//!                            decode forwards (the prompts' pass streams every layer)
//!   --steps K                stop after K greedy decode forwards; a rerun with the same --out
//!                            resumes: it replays the saved tokens layer by layer, refuses to go on
//!                            unless every replayed step's saved values come back byte for byte,
//!                            and continues (a finished sequence is only replayed and checked)
//!   --eos ID                 end of sequence (default: the config's eos_token_id)
//!   --full-steps N           full logits of the first N steps (default 8, as the tapes)
//! Residency:
//!   --experts-host           the serving route's single-card residency: every layer's own weights
//!                            and the head resident, the routed experts in pinned host memory
//!                            behind a device LRU cache (`dsv41_gpu::experts`); the bytes a kernel
//!                            reads are the streamed route's, so the outputs are too
//!   --cache-slots N          at most N expert slots per rank (default: all the free memory less
//!                            --cache-reserve-mb; never below one slice's experts)
//!   --cache-reserve-mb N     device memory left beside the cache for the forwards (default 16384)

use memra_engine::dsv41_gpu::{
    self as v, Geo,
    decode::{Call, Runner, Scored, Seq, TOPK},
};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The checkpoint program's numeric class (both the vendor oracle and this runner run it).
const NUMERIC_CLASS: &str = "deepseek-v41-source-program";

fn main() {
    if let Err(e) = run() {
        eprintln!("dsv41-run: {e}");
        std::process::exit(1);
    }
}

fn read_ids(f: &str) -> Result<Vec<u32>, String> {
    let raw = std::fs::read(f).map_err(|e| format!("{f}: {e}"))?;
    if raw.len() % 4 != 0 {
        return Err(format!("{f}: not an i32 array"));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
        .collect())
}

enum Mode {
    /// prompt only (the checkpoint and long-context runner)
    Prefill,
    /// the token taken at each step
    Forced(Vec<u32>),
    /// greedy, up to this many tokens
    Greedy(usize),
}

struct Case {
    id: String,
    prompt: Vec<u32>,
    mode: Mode,
    dir: Option<PathBuf>,
    /// greedy: the tokens chosen so far (saved and new)
    chosen: Vec<u32>,
    /// greedy: steps found on disk at start (replayed and checked)
    saved: usize,
    done: bool,
}

impl Case {
    fn steps(&self) -> usize {
        match &self.mode {
            Mode::Prefill => 0,
            Mode::Forced(f) => f.len(),
            Mode::Greedy(n) => *n,
        }
    }
}

fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let ckpt = PathBuf::from(
        a.get(1)
            .ok_or("usage: dsv41-run CKPT_DIR [TOKEN ...] [--ids FILE] ...")?,
    );
    let flag = |k: &str| {
        a.iter()
            .position(|x| x == k)
            .and_then(|i| a.get(i + 1))
            .cloned()
    };
    let num = |k: &str| -> Result<Option<usize>, String> {
        flag(k)
            .map(|x| x.parse().map_err(|_| format!("{k} {x:?}: not a count")))
            .transpose()
    };
    let mut ids: Vec<u32> = Vec::new();
    let mut i = 2;
    while i < a.len() {
        if a[i] == "--experts-host" {
            i += 1;
            continue;
        }
        if a[i].starts_with("--") {
            i += 2;
            continue;
        }
        ids.push(
            a[i].parse()
                .map_err(|_| format!("bad token id {:?}", a[i]))?,
        );
        i += 1;
    }
    if let Some(f) = flag("--ids") {
        ids = read_ids(&f)?;
    }
    if std::env::var("MEMRA_FULL_PREC").as_deref() != Ok("1") {
        return Err("set MEMRA_FULL_PREC=1: the deepseek_v41 route runs only the checkpoint's own numeric program; no serving format exists to select".into());
    }
    let tmap = flag("--token-map")
        .or_else(|| std::env::var("MEMRA_DSV41_TOKEN_MAP").ok())
        .ok_or("the Engram hash needs the compressed token map: --token-map FILE or MEMRA_DSV41_TOKEN_MAP")?;
    let chunk = num("--chunk")?.unwrap_or(8192);
    let keep = num("--resident")?.unwrap_or(0);
    let steps_run = num("--steps")?;
    let full_steps = num("--full-steps")?.unwrap_or(8);
    let gen_n = num("--gen")?;
    let out = flag("--out").map(PathBuf::from);
    let devices: Vec<usize> = match flag("--devices") {
        Some(x) => x
            .split(',')
            .map(|d| d.trim().parse().map_err(|_| format!("--devices {x:?}")))
            .collect::<Result<_, _>>()?,
        None => vec![0],
    };

    let t0 = Instant::now();
    let cfg_path = ckpt.join("config.json");
    let mc =
        memra_gguf::config::ModelConfig::from_config_json(&cfg_path).map_err(|e| e.to_string())?;
    let g = Geo::from_config(&mc)?;
    let eos: u32 = match flag("--eos") {
        Some(x) => x.parse().map_err(|_| format!("--eos {x:?}"))?,
        None => {
            let txt = std::fs::read_to_string(&cfg_path).map_err(|e| e.to_string())?;
            let j: serde_json::Value = serde_json::from_str(&txt).map_err(|e| e.to_string())?;
            j["eos_token_id"]
                .as_u64()
                .ok_or("config.json has no eos_token_id: pass --eos")? as u32
        }
    };
    let trunk =
        g.compress_ratios.len() - mc.dsv41.as_ref().unwrap().num_nextn_predict_layers as usize;
    let n_layers = num("--layers")?.unwrap_or(trunk).min(trunk);

    // the sequences
    let mut cases: Vec<Case> = Vec::new();
    let single = flag("--cases").is_none();
    if let Some(cf) = flag("--cases") {
        if !ids.is_empty() || flag("--force").is_some() {
            return Err("--cases replaces the prompt arguments and --force".into());
        }
        let root = out
            .clone()
            .ok_or("--cases writes per-sequence directories: give --out DIR")?;
        let txt = std::fs::read_to_string(&cf).map_err(|e| format!("{cf}: {e}"))?;
        for line in txt
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
        {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 2 || f.len() > 3 {
                return Err(format!("{cf}: {line:?}: want ID PROMPT_FILE [FORCE_FILE]"));
            }
            let mode = match f.get(2) {
                Some(ff) => Mode::Forced(read_ids(ff)?),
                None => Mode::Greedy(gen_n.ok_or("a greedy case needs --gen N")?),
            };
            cases.push(Case {
                id: f[0].to_string(),
                prompt: read_ids(f[1])?,
                mode,
                dir: Some(root.join(f[0])),
                chosen: Vec::new(),
                saved: 0,
                done: false,
            });
        }
    } else {
        let mode = match (flag("--force"), gen_n) {
            (Some(_), Some(_)) => return Err("--force and --gen are two modes: pick one".into()),
            (Some(ff), None) => Mode::Forced(read_ids(&ff)?),
            (None, Some(n)) => Mode::Greedy(n),
            (None, None) => Mode::Prefill,
        };
        if !matches!(mode, Mode::Prefill) && out.is_none() {
            return Err("a decode mode writes its steps to --out DIR".into());
        }
        cases.push(Case {
            id: "prompt".into(),
            prompt: ids.clone(),
            mode,
            dir: out.clone(),
            chosen: Vec::new(),
            saved: 0,
            done: false,
        });
    }
    if !single && (flag("--tf").is_some() || flag("--last-logits").is_some()) {
        return Err("--tf and --last-logits score a single prompt".into());
    }
    for c in &cases {
        if c.prompt.is_empty() {
            return Err(format!("{}: no prompt tokens", c.id));
        }
        if let Mode::Forced(f) = &c.mode
            && f.is_empty()
        {
            return Err(format!("{}: an empty forced sequence", c.id));
        }
    }
    let max_pos = cases
        .iter()
        .map(|c| c.prompt.len() + c.steps())
        .max()
        .unwrap_or(0)
        .saturating_add(16)
        .max(1024);

    let stm = memra_gguf::safetensors::StModel::open(&ckpt).map_err(|e| e.to_string())?;
    let ec =
        v::engram::EngramConsts::derive(&g, v::engram::token_map_from_file(Path::new(&tmap))?)?;
    // the prompts stream every layer; the resident layers are kept from the first decode forward
    let mut rn = Runner::new(
        &stm,
        g.clone(),
        Some(ec),
        &devices,
        max_pos,
        n_layers,
        0,
        chunk,
    )?;
    if a.iter().any(|x| x == "--experts-host") {
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
        let ctx = rn.grp.cx[0].ops.st.context().clone();
        let host = std::sync::Arc::new(v::experts::HostExperts::fill(
            &stm, &g, n_layers, &ctx, threads,
        )?);
        eprintln!(
            "dsv41-run: routed experts of {n_layers} layers in pinned host memory: {} bytes, filled and locked in {:.1}s on {threads} threads",
            host.bytes, host.fill_s
        );
        let resident = rn.experts_off_card(host)?;
        let reserve = num("--cache-reserve-mb")?.unwrap_or(16384) as u64 * (1 << 20);
        let slots = rn.grow_caches(reserve, num("--cache-slots")?.unwrap_or(usize::MAX))?;
        eprintln!(
            "dsv41-run: resident weights {resident:?} bytes per rank; expert cache {slots:?} slots per rank ({} bytes a slot), {reserve} bytes reserved",
            v::experts::Layout::of(&g).bytes
        );
    }
    eprintln!(
        "dsv41-run: {} sequence(s), longest prompt {}, {n_layers} layers, chunk {chunk}, {} rank(s) on devices {devices:?}, {keep} resident, transport {}",
        cases.len(),
        cases.iter().map(|c| c.prompt.len()).max().unwrap_or(0),
        rn.grp.n(),
        rn.grp.transport.name()
    );

    // what a decode directory already holds: a greedy case resumes from it, a forced one restarts
    for c in cases.iter_mut() {
        let Some(dir) = c.dir.as_ref() else { continue };
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        match c.mode {
            Mode::Greedy(n) => {
                let k = saved_steps(dir, true, full_steps, g.vocab);
                truncate_steps(dir, k, true, full_steps, g.vocab)?;
                c.chosen = read_ids(&dir.join("gen.i32").to_string_lossy())?;
                c.saved = k;
                c.done = k > 0 && (c.chosen[k - 1] == eos || k >= n);
            }
            Mode::Forced(_) => truncate_steps(dir, 0, false, full_steps, g.vocab)?,
            Mode::Prefill => {}
        }
    }

    // pass 0, layer by layer: every prompt, then each sequence's known tokens (forced, or saved)
    let mut seqs: Vec<Seq> = cases.iter().map(|_| rn.seq()).collect();
    let mut calls = Vec::new();
    let mut step_of = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        calls.push(Call {
            seq: i,
            tokens: c.prompt.clone(),
        });
        step_of.push((i, 0usize));
        let feeds: &[u32] = match &c.mode {
            Mode::Prefill => &[],
            Mode::Forced(f) => &f[..f.len() - 1],
            Mode::Greedy(_) => &c.chosen[..c.saved.saturating_sub(1)],
        };
        for (j, &t) in feeds.iter().enumerate() {
            calls.push(Call {
                seq: i,
                tokens: vec![t],
            });
            step_of.push((i, j + 1));
        }
    }
    let tp = Instant::now();
    let outs = rn.forward(&mut seqs, &calls)?;
    eprintln!(
        "dsv41-run: pass 0: {} calls in {:.1}s (weights {:.1}s, {} layer loads)",
        calls.len(),
        tp.elapsed().as_secs_f64(),
        rn.stats.load_s,
        rn.stats.layers_loaded
    );
    drop(calls);

    if single {
        prompt_outputs(
            &mut rn,
            &outs[0],
            &ids,
            &flag("--last-logits"),
            &flag("--tf"),
            g.vocab,
            t0,
        )?;
    }

    // score pass 0's steps
    let mut replayed = vec![0usize; cases.len()];
    for (o, &(ci, step)) in outs.iter().zip(&step_of) {
        let c = &mut cases[ci];
        let full = step < full_steps;
        match c.mode {
            Mode::Prefill => {}
            Mode::Forced(_) => {
                let sc = rn.score_last(o, full)?;
                write_step(c.dir.as_ref().unwrap(), &sc, None, full)?;
            }
            Mode::Greedy(n) => {
                let sc = rn.score_last(o, full)?;
                let dir = c.dir.as_ref().unwrap();
                if step < c.saved {
                    check_step(dir, step, &sc, full, g.vocab)?;
                    replayed[ci] += 1;
                } else {
                    write_step(dir, &sc, Some(sc.argmax), full)?;
                    c.chosen.push(sc.argmax);
                    c.done = sc.argmax == eos || c.chosen.len() >= n;
                }
            }
        }
    }
    drop(outs);
    for (c, r) in cases.iter().zip(&replayed) {
        if *r > 0 {
            eprintln!(
                "dsv41-run: {}: replayed {r} saved steps layer by layer, every one byte-identical to the saved step",
                c.id
            );
        }
    }

    // greedy steps, one token per sequence per forward
    rn.set_keep(keep);
    let mut fwd = 0usize;
    loop {
        let active: Vec<usize> = (0..cases.len())
            .filter(|&i| matches!(cases[i].mode, Mode::Greedy(_)) && !cases[i].done)
            .collect();
        if active.is_empty() {
            break;
        }
        if steps_run.is_some_and(|k| fwd >= k) {
            eprintln!(
                "dsv41-run: stopping after {fwd} decode forwards; rerun with the same --out to resume"
            );
            break;
        }
        let calls: Vec<Call> = active
            .iter()
            .map(|&i| Call {
                seq: i,
                tokens: vec![*cases[i].chosen.last().unwrap()],
            })
            .collect();
        let (l0, ts) = (rn.stats.load_s, Instant::now());
        let outs = rn.forward(&mut seqs, &calls)?;
        for (o, &i) in outs.iter().zip(&active) {
            let c = &mut cases[i];
            let step = c.chosen.len();
            let sc = rn.score_last(o, step < full_steps)?;
            write_step(
                c.dir.as_ref().unwrap(),
                &sc,
                Some(sc.argmax),
                step < full_steps,
            )?;
            c.chosen.push(sc.argmax);
            if let Mode::Greedy(n) = c.mode {
                c.done = sc.argmax == eos || c.chosen.len() >= n;
            }
        }
        fwd += 1;
        eprintln!(
            "dsv41-run: decode forward {fwd}: {} sequence(s), {:.1}s (weights {:.1}s)",
            active.len(),
            ts.elapsed().as_secs_f64(),
            rn.stats.load_s - l0
        );
    }

    for (c, r) in cases.iter().zip(&replayed) {
        let Some(dir) = c.dir.as_ref() else { continue };
        let (mode, steps, done) = match &c.mode {
            Mode::Forced(f) => ("forced", f.len(), true),
            Mode::Greedy(_) => ("greedy", c.chosen.len(), c.done),
            Mode::Prefill => continue,
        };
        let toks = c
            .chosen
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let meta = format!(
            "{{\"id\": \"{}\", \"mode\": \"{mode}\", \"prompt_len\": {}, \"steps\": {steps}, \"done\": {done}, \"eos\": {eos}, \"replayed_identical\": {r}, \"full_steps\": {}, \"layers\": {n_layers}, \"ranks\": {}, \"numeric_class\": \"{NUMERIC_CLASS}\", \"gen\": [{toks}]}}\n",
            c.id,
            c.prompt.len(),
            full_steps.min(steps),
            rn.grp.n(),
        );
        std::fs::write(dir.join("meta.json"), meta).map_err(|e| e.to_string())?;
        println!(
            "{}: {mode}, {steps} steps{}",
            c.id,
            if done { ", done" } else { "" }
        );
    }
    let st = &rn.stats;
    eprintln!(
        "dsv41-run: {} forwards, {} calls, {} layer loads ({:.1}s), blocks {:.1}s, total {:.1}s",
        st.forwards,
        st.calls,
        st.layers_loaded,
        st.load_s,
        st.run_s,
        t0.elapsed().as_secs_f64()
    );
    if rn.offloaded() {
        let (cs, slots) = rn.cache_stats();
        eprintln!(
            "dsv41-run: expert cache {slots:?} slots: {} lookups, {} hits, {} misses, {} evictions, {} bytes H2D",
            cs.lookups, cs.hits, cs.misses, cs.evictions, cs.h2d_bytes
        );
    }
    Ok(())
}

/// The single prompt's outputs: the last token's argmax, `--last-logits`, the oracle tape and the
/// teacher-forced scoring of every prompt position.
fn prompt_outputs(
    rn: &mut Runner,
    o: &v::decode::Out,
    ids: &[u32],
    last_logits: &Option<String>,
    tf: &Option<String>,
    vocab: usize,
    t0: Instant,
) -> Result<(), String> {
    let s = o.s;
    let last = rn.logits(o, s - 1, 1)?;
    if last.iter().any(|x| !x.is_finite()) {
        return Err("non-finite logits".into());
    }
    let am = v::decode::argmax(&last);
    eprintln!(
        "dsv41-run: argmax {am} logit {:.4}, {:.1}s",
        last[am],
        t0.elapsed().as_secs_f64()
    );
    println!("argmax token = {am}");
    if let Some(f) = last_logits {
        std::fs::write(f, v::f32_bytes(&last)).map_err(|e| format!("{f}: {e}"))?;
    }
    if let Some(path) = std::env::var_os("MEMRA_ORACLE_OUT") {
        let mut t = format!(
            "format\tmemra-checkpoint-oracle-v1\nengine\tmemra-native-full-precision\nnumeric_class\t{NUMERIC_CLASS}\ntokens\t{}\nvocab\t{}\n",
            ids.iter().map(u32::to_string).collect::<Vec<_>>().join(","),
            last.len()
        );
        for (i, x) in last.iter().enumerate() {
            t += &format!("logit\t{i}\t{:08x}\n", x.to_bits());
        }
        std::fs::write(&path, t).map_err(|e| format!("{}: {e}", PathBuf::from(&path).display()))?;
    }
    if let Some(dir) = tf {
        // every position's lse and argmax, and the NLL of the next prompt token
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let mut lse = Vec::with_capacity(s);
        let mut top = Vec::with_capacity(s);
        let mut nll = Vec::with_capacity(s.saturating_sub(1));
        let mut top_ids = Vec::with_capacity(s * TOPK);
        let mut top_vals = Vec::with_capacity(s * TOPK);
        let rows = 1024;
        for r0 in (0..s).step_by(rows) {
            let n = rows.min(s - r0);
            let lg = rn.logits(o, r0, n)?;
            for i in 0..n {
                let row = &lg[i * vocab..(i + 1) * vocab];
                let l = v::decode::lse(row);
                lse.push(l as f32);
                top.push(v::decode::argmax(row) as i32);
                for (id, val) in v::decode::top_k(row) {
                    top_ids.push(id as i32);
                    top_vals.push(val);
                }
                let p = r0 + i;
                if p + 1 < s {
                    nll.push((l - row[ids[p + 1] as usize] as f64) as f32);
                }
            }
        }
        let dir = PathBuf::from(dir);
        std::fs::write(dir.join("lse.f32"), v::f32_bytes(&lse)).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("argmax.i32"), v::i32_bytes(&top)).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("nll.f32"), v::f32_bytes(&nll)).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("top_ids.i32"), v::i32_bytes(&top_ids))
            .map_err(|e| e.to_string())?;
        std::fs::write(dir.join("top_vals.f32"), v::f32_bytes(&top_vals))
            .map_err(|e| e.to_string())?;
        let mean = nll.iter().map(|&x| x as f64).sum::<f64>() / nll.len().max(1) as f64;
        std::fs::write(
            dir.join("meta.json"),
            format!("{{\"tokens\": {s}, \"scored\": {}, \"mean_nll\": {mean}, \"numeric_class\": \"{NUMERIC_CLASS}\"}}\n", nll.len()),
        )
        .map_err(|e| e.to_string())?;
        eprintln!(
            "dsv41-run: teacher-forced mean NLL {mean:.5} over {} positions",
            nll.len()
        );
    }
    Ok(())
}

/// Per-step files of a decode directory, one row per step, appended as each step finishes, with
/// each file's row size:
///   top_ids.i32   the 32 largest logits' ids, largest first (lower id among ties)
///   top_vals.f32  their values
///   lse.f64       log-sum-exp of the step's logits
///   argmax.i32    the greedy token (lowest id among equal maxima)
///   gen.i32       greedy only: the token taken (= argmax)
///   full.f32      the first `--full-steps` steps only: the full fp32 logits
fn step_files(greedy: bool) -> Vec<(&'static str, usize)> {
    let mut f = vec![
        ("top_ids.i32", TOPK * 4),
        ("top_vals.f32", TOPK * 4),
        ("lse.f64", 8),
        ("argmax.i32", 4),
    ];
    if greedy {
        f.push(("gen.i32", 4));
    }
    f
}

fn file_len(p: &Path) -> u64 {
    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

/// Complete steps on disk: the fewest whole rows over the step files.
fn saved_steps(dir: &Path, greedy: bool, full: usize, vocab: usize) -> usize {
    let mut k = usize::MAX;
    for (f, row) in step_files(greedy) {
        k = k.min((file_len(&dir.join(f)) / row as u64) as usize);
    }
    let fr = (file_len(&dir.join("full.f32")) / (vocab as u64 * 4)) as usize;
    if fr < k.min(full) {
        k = fr;
    }
    k
}

fn truncate_steps(
    dir: &Path,
    k: usize,
    greedy: bool,
    full: usize,
    vocab: usize,
) -> Result<(), String> {
    let mut files = step_files(greedy);
    files.push(("full.f32", vocab * 4));
    for (f, row) in files {
        let rows = if f == "full.f32" { k.min(full) } else { k };
        let p = dir.join(f);
        let fh = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&p)
            .map_err(|e| format!("{}: {e}", p.display()))?;
        fh.set_len((rows * row) as u64)
            .map_err(|e| format!("{}: {e}", p.display()))?;
    }
    Ok(())
}

fn step_rows(sc: &Scored, chosen: Option<u32>, full: bool) -> Vec<(&'static str, Vec<u8>)> {
    let ids: Vec<i32> = sc.top.iter().map(|t| t.0 as i32).collect();
    let vals: Vec<f32> = sc.top.iter().map(|t| t.1).collect();
    let mut r = vec![
        ("top_ids.i32", v::i32_bytes(&ids)),
        ("top_vals.f32", v::f32_bytes(&vals)),
        ("lse.f64", sc.lse.to_le_bytes().to_vec()),
        ("argmax.i32", v::i32_bytes(&[sc.argmax as i32])),
    ];
    if let Some(t) = chosen {
        r.push(("gen.i32", v::i32_bytes(&[t as i32])));
    }
    if full && let Some(lg) = sc.logits.as_ref() {
        r.push(("full.f32", v::f32_bytes(lg)));
    }
    r
}

fn write_step(dir: &Path, sc: &Scored, chosen: Option<u32>, full: bool) -> Result<(), String> {
    for (f, b) in step_rows(sc, chosen, full) {
        let p = dir.join(f);
        let mut fh = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&p)
            .map_err(|e| format!("{}: {e}", p.display()))?;
        fh.write_all(&b)
            .map_err(|e| format!("{}: {e}", p.display()))?;
    }
    Ok(())
}

/// A replayed step against its saved row: every file's bytes, the full logits included.
fn check_step(
    dir: &Path,
    step: usize,
    sc: &Scored,
    full: bool,
    vocab: usize,
) -> Result<(), String> {
    for (f, b) in step_rows(sc, Some(sc.argmax), full) {
        let p = dir.join(f);
        let all = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let row = if f == "full.f32" { vocab * 4 } else { b.len() };
        let saved = all
            .get(step * row..(step + 1) * row)
            .ok_or_else(|| format!("{}: no saved step {step}", p.display()))?;
        if saved != b.as_slice() {
            return Err(format!(
                "replay of step {step} differs from the saved {f}: the layer-by-layer replay is not the program that wrote it; not continuing"
            ));
        }
    }
    Ok(())
}
