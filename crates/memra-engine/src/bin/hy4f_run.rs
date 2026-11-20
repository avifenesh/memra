//! Hy4-flash, the native checkpoint runner for `memra model verify checkpoint --against
//! hy_v4_flash --native-runner <this binary>`: the text program of the typed plan over the
//! artifact's own shards on one GPU (`memra_engine::hy4f_gpu`), every position of a request
//! scored teacher-forced in one layer-by-layer pass.
//!
//! usage: hy4f-run CKPT_DIR (--request FILE | TOKEN ...) [--tape DIR] [--red ARM]...
//!                 [--layers N] [--gen N] [--device D]
//!        hy4f-run --numeric-class   print the class (the checkpoint gate asks)
//!   MEMRA_FULL_PREC=1      required: the route has one numeric program
//!   MEMRA_ORACLE_OUT=FILE  with TOKEN args: write the `memra-checkpoint-oracle-v1` tape of the
//!                          last token (the whole prompt as the prefill)
//!   --request FILE         `{"name", "token_ids", "decode_from"}`: the prefill
//!                          `token_ids[..decode_from]`, then one teacher-forced decode call per
//!                          later token
//!   --tape DIR             write the `memra-checkpoint-tape-v2` directory: `tape.json`,
//!                          `logits.f32` (`[positions, vocab]`), the indexer's recalled groups
//!                          (`indexer_sel.i32`) and group logits (`indexer_logits.f32`) per call
//!                          and indexed layer, the router's experts (`router_sel.i32`) and
//!                          selection scores (`router_scores.f32`) per call and MoE layer, and
//!                          each call's Engram rows as FNV-1a 64 digests
//!   --red ARM              a wrong program for the gate's red arms: `engram-off`,
//!                          `ties-reversed`, `dequant-bf16`
//!   --layers N             diagnostic: the first N decoder layers only (the tape says so)
//!   --gen N                greedy: append the argmax of the last position N times (one whole
//!                          pass per token) and print the ids

use memra_engine::hy4f_gpu::{Ckpt, Gpu, NUMERIC_CLASS, RedArms, Request, Runner};
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::{AttentionPlan, HyV4AttentionKind};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("hy4f-run: {e}");
        std::process::exit(1);
    }
}

struct Args {
    ckpt: PathBuf,
    name: String,
    tokens: Vec<u32>,
    decode_from: usize,
    tape: Option<PathBuf>,
    red: RedArms,
    red_names: Vec<String>,
    layers: Option<usize>,
    gen_tokens: usize,
    device: usize,
}

fn parse() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let ckpt = PathBuf::from(it.next().ok_or("usage: hy4f-run CKPT_DIR ...")?);
    let mut a = Args {
        ckpt,
        name: "tokens".into(),
        tokens: Vec::new(),
        decode_from: 0,
        tape: None,
        red: RedArms::default(),
        red_names: Vec::new(),
        layers: None,
        gen_tokens: 0,
        device: 0,
    };
    let mut request = None;
    while let Some(arg) = it.next() {
        let mut val = || it.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--request" => request = Some(PathBuf::from(val()?)),
            "--tape" => a.tape = Some(PathBuf::from(val()?)),
            "--layers" => a.layers = Some(val()?.parse().map_err(|e| format!("--layers: {e}"))?),
            "--gen" => a.gen_tokens = val()?.parse().map_err(|e| format!("--gen: {e}"))?,
            "--device" => a.device = val()?.parse().map_err(|e| format!("--device: {e}"))?,
            "--red" => {
                let v = val()?;
                match v.as_str() {
                    "engram-off" => a.red.engram_off = true,
                    "ties-reversed" => a.red.ties_reversed = true,
                    "dequant-bf16" => a.red.dequant_bf16 = true,
                    _ => return Err(format!("unknown red arm {v}")),
                }
                a.red_names.push(v);
            }
            t if !t.starts_with("--") => a
                .tokens
                .push(t.parse().map_err(|e| format!("token {t}: {e}"))?),
            _ => return Err(format!("unknown argument {arg}")),
        }
    }
    if let Some(path) = request {
        if !a.tokens.is_empty() {
            return Err("give --request or TOKEN arguments, not both".into());
        }
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        a.name = v["name"].as_str().ok_or("request has no name")?.to_string();
        a.tokens = v["token_ids"]
            .as_array()
            .ok_or("request has no token_ids")?
            .iter()
            .map(|t| {
                t.as_u64()
                    .and_then(|t| u32::try_from(t).ok())
                    .ok_or("token ids are u32")
            })
            .collect::<Result<_, _>>()?;
        a.decode_from = v["decode_from"]
            .as_u64()
            .ok_or("request has no decode_from")? as usize;
    } else {
        a.decode_from = a.tokens.len();
    }
    if a.tokens.is_empty() || a.decode_from == 0 || a.decode_from > a.tokens.len() {
        return Err("the prefill is a non-empty prefix of the request".into());
    }
    Ok(a)
}

fn fnv1a(text: &str) -> String {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in text.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3);
    }
    format!("{h:016x}")
}

fn sha_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn argmax(row: &[f32]) -> usize {
    // the first of equal maxima, as `torch.argmax`
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if v.total_cmp(&row[best]).is_gt() {
            best = i;
        }
    }
    best
}

struct Pass {
    logits: Vec<f32>,
    /// per layer (index, record)
    records: Vec<(usize, memra_engine::hy4f_gpu::LayerRecord)>,
    layers: usize,
}

fn pass(runner: &Runner, req: &Request, layers: usize) -> Result<Pass, String> {
    let mut s = runner.embed(req.tokens)?;
    let mut records = Vec::new();
    for il in 0..layers {
        let t0 = Instant::now();
        let rec = runner.layer(il, req, &mut s)?;
        eprintln!("hy4f-run: layer {il:2} {:6.1}s", t0.elapsed().as_secs_f64());
        records.push((il, rec));
    }
    let logits = runner.head(&s)?;
    Ok(Pass {
        logits,
        records,
        layers,
    })
}

fn run() -> Result<(), String> {
    // the checkpoint gate asks a runner for its numeric class before it runs it
    if std::env::args().nth(1).as_deref() == Some("--numeric-class") {
        println!("{NUMERIC_CLASS}");
        return Ok(());
    }
    if std::env::var("MEMRA_FULL_PREC").as_deref() != Ok("1") {
        return Err("MEMRA_FULL_PREC=1 is required: this route has one numeric program".into());
    }
    let a = parse()?;
    let t_start = Instant::now();
    let cfg_text = std::fs::read_to_string(a.ckpt.join("config.json"))
        .map_err(|e| format!("config.json: {e}"))?;
    let config = ModelConfig::from_hf(&HfConfig::try_parse(&cfg_text)?);
    let pack = memra_gguf::model_packs::by_alias("hy_v4_flash").ok_or("no hy_v4_flash pack")?;
    let plan = pack
        .compile_plan(&config)
        .map_err(|e| format!("plan: {e:?}"))?;
    let hy = config
        .hy_v4_flash
        .as_ref()
        .ok_or("not a Hy4-flash config")?;
    let ck = Ckpt::open(&a.ckpt)?;
    let mut gpu = Gpu::new(a.device)?;
    gpu.red = a.red;
    let runner = Runner {
        gpu,
        ck: &ck,
        plan: &plan,
        hy,
    };
    let layers = a.layers.unwrap_or(plan.layers.len()).min(plan.layers.len());
    let vocab = plan.vocab_size as usize;

    if a.gen_tokens > 0 {
        let mut ids = a.tokens.clone();
        for _ in 0..a.gen_tokens {
            let req = Request {
                tokens: &ids,
                decode_from: ids.len(),
            };
            let p = pass(&runner, &req, layers)?;
            let next = argmax(&p.logits[(ids.len() - 1) * vocab..]) as u32;
            ids.push(next);
            println!(
                "gen {}",
                ids[a.tokens.len()..]
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }
        return Ok(());
    }

    let req = Request {
        tokens: &a.tokens,
        decode_from: a.decode_from,
    };
    let p = pass(&runner, &req, layers)?;
    let t = a.tokens.len();
    let am: Vec<usize> = (0..t)
        .map(|r| argmax(&p.logits[r * vocab..(r + 1) * vocab]))
        .collect();
    eprintln!(
        "hy4f-run: {t} positions, {:.1}s, argmax tail {:?}",
        t_start.elapsed().as_secs_f64(),
        &am[t.saturating_sub(8)..]
    );

    if let Ok(out) = std::env::var("MEMRA_ORACLE_OUT")
        && a.tape.is_none()
    {
        {
            let last = &p.logits[(t - 1) * vocab..t * vocab];
            let mut text = format!(
                "format\tmemra-checkpoint-oracle-v1\nengine\tmemra-hy4f-run\nnumeric_class\t{NUMERIC_CLASS}\ntokens\t{}\nvocab\t{vocab}\n",
                a.tokens
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            );
            for (i, v) in last.iter().enumerate() {
                text.push_str(&format!("logit\t{i}\t{:08x}\n", v.to_bits()));
            }
            std::fs::write(&out, text).map_err(|e| format!("{out}: {e}"))?;
        }
    }
    if let Some(dir) = &a.tape {
        write_tape(dir, &a, &plan, &p, &am, vocab, t_start)?;
    }
    Ok(())
}

fn write_tape(
    dir: &Path,
    a: &Args,
    plan: &memra_gguf::model_plan::ModelPlan,
    p: &Pass,
    am: &[usize],
    vocab: usize,
    t_start: Instant,
) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let w = |name: &str, bytes: &[u8]| -> Result<String, String> {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        f.write_all(bytes)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(sha_hex(bytes))
    };
    let logit_bytes: Vec<u8> = p.logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    let logits_sha = w("logits.f32", &logit_bytes)?;
    drop(logit_bytes);
    let t = a.tokens.len();
    let mut calls = vec![(0usize, a.decode_from)];
    calls.extend((a.decode_from..t).map(|q| (q, q + 1)));
    let (mut sel, mut lg) = (Vec::<u8>::new(), Vec::<u8>::new());
    let (mut so, mut lo) = (0usize, 0usize);
    let mut idx = Vec::new();
    let mut engram = Vec::new();
    for (ci, &(start, end)) in calls.iter().enumerate() {
        for (il, rec) in &p.records {
            if let Some(first) = rec.recalled.get(start).and_then(|v| v.as_ref()) {
                let (k, groups) = (first.0.len(), first.1.len());
                for q in start..end {
                    let (g, l) = rec.recalled[q]
                        .as_ref()
                        .ok_or("a call indexes every query or none")?;
                    if g.len() != k || l.len() != groups {
                        return Err("a call's queries rank the same groups".into());
                    }
                    sel.extend(g.iter().flat_map(|v| v.to_le_bytes()));
                    lg.extend(l.iter().flat_map(|v| v.to_le_bytes()));
                }
                idx.push(serde_json::json!({"call": ci, "start": start, "tokens": end - start, "layer": il,
                    "queries": end - start, "k": k, "groups": groups, "sel_offset": so, "logit_offset": lo}));
                so += (end - start) * k;
                lo += (end - start) * groups;
            }
            if let Some(rows) = &rec.engram_rows {
                let text = rows[start..end]
                    .iter()
                    .flat_map(|r| r.iter().map(u64::to_string))
                    .collect::<Vec<_>>()
                    .join(",");
                engram.push(serde_json::json!({"call": ci, "layer": il, "rows": fnv1a(&text)}));
            }
        }
    }
    let (mut rsel, mut rsc) = (Vec::<u8>::new(), Vec::<u8>::new());
    let (mut rso, mut rco) = (0usize, 0usize);
    let mut ridx = Vec::new();
    for (ci, &(start, end)) in calls.iter().enumerate() {
        for (il, rec) in &p.records {
            if let Some(routes) = &rec.routes {
                let (k, experts) = (routes[start].0.len(), routes[start].1.len());
                for (picks, scores) in &routes[start..end] {
                    rsel.extend(picks.iter().flat_map(|v| v.to_le_bytes()));
                    rsc.extend(scores.iter().flat_map(|v| v.to_le_bytes()));
                }
                ridx.push(
                    serde_json::json!({"call": ci, "layer": il, "queries": end - start, "k": k,
                    "experts": experts, "sel_offset": rso, "score_offset": rco}),
                );
                rso += (end - start) * k;
                rco += (end - start) * experts;
            }
        }
    }
    let rsel_sha = w("router_sel.i32", &rsel)?;
    let rsc_sha = w("router_scores.f32", &rsc)?;
    drop((rsel, rsc));
    let sel_sha = w("indexer_sel.i32", &sel)?;
    let lg_sha = w("indexer_logits.f32", &lg)?;
    let indexed: Vec<usize> = plan
        .layers
        .iter()
        .filter(|l| matches!(&l.attention, AttentionPlan::HyV4(x) if matches!(x.kind, HyV4AttentionKind::Indexed(_))))
        .map(|l| l.index as usize)
        .collect();
    let tape = serde_json::json!({
        "format": "memra-checkpoint-tape-v2",
        "engine": format!("memra-hy4f-run-sm_{}", memra_engine::BUILT_CUDA_TARGET),
        "numeric_class": NUMERIC_CLASS,
        "request": a.name,
        "token_ids": a.tokens,
        "decode_from": a.decode_from,
        "vocab": vocab,
        "positions": t,
        "layers": p.layers,
        "indexed_layers": indexed,
        "red_arms": a.red_names,
        "argmax": am,
        "logits": {"file": "logits.f32", "sha256": logits_sha},
        "indexer": {"sel_file": "indexer_sel.i32", "sel_sha256": sel_sha, "logit_file": "indexer_logits.f32",
                    "logit_sha256": lg_sha, "records": idx},
        "router": {"sel_file": "router_sel.i32", "sel_sha256": rsel_sha, "score_file": "router_scores.f32",
                   "score_sha256": rsc_sha, "records": ridx},
        "engram": engram,
        "seconds": (t_start.elapsed().as_secs_f64() * 10.0).round() / 10.0,
    });
    let text = serde_json::to_string_pretty(&tape).map_err(|e| e.to_string())?;
    w("tape.json", text.as_bytes())?;
    Ok(())
}
