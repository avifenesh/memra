//! Hy4-flash on one GPU, the GPU route (`memra_engine::hy4f_route`): weights resident, a carried
//! cache, numeric class `hy-v4-flash-bf16-activations`. The native checkpoint runner for
//! `memra model verify checkpoint --against hy_v4_flash --native-runner <this binary>`, and a
//! greedy runner with a throughput reading.
//!
//! usage: hy4f-gpu CKPT_DIR --request FILE [--tape DIR] [--force-selections TAPE_DIR]
//!                 [--red ARM]... [--bench N] [--gen N] [--device D]
//!        hy4f-gpu --numeric-class   print the class (the checkpoint gate asks)
//!   --request FILE           `{"name", "token_ids", "decode_from"}`: the prefill call
//!                            `token_ids[..decode_from]`, then one teacher-forced decode call per
//!                            later token
//!   --tape DIR               write the `memra-checkpoint-tape-v2` directory (`tape.json`, every
//!                            position's logits, every indexer call's recalled groups and group
//!                            logits, every router call's experts and selection scores, every
//!                            Engram call's table rows as an FNV-1a 64 digest)
//!   --force-selections DIR   apply the router's and the indexer's picks of that oracle tape
//!                            (its own picks and scores are still recorded): the gate's run of
//!                            this class, whose logit band is set under the same forcing
//!   --red ARM                a wrong program for the gate's red arms: `engram-off`,
//!                            `ties-reversed`, `dequant-bf16`, `kv-off-by-one`, `pool-early`
//!   --gen N                  greedy: prefill `token_ids[..decode_from]`, then N decode calls on
//!                            the argmax; print the ids
//!   --bench N                the throughput reading: one warm-up, then N runs of a prefill of
//!                            `token_ids[..decode_from]` and 64 greedy decode calls, timed

use memra_engine::hy4f_route::{
    Ckpt, Dev, Forced, LayerRec, NUMERIC_CLASS, RedArms, Route, indexed_layers, mem_available,
};
use memra_gguf::config::{HfConfig, ModelConfig};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("hy4f-gpu: {e}");
        std::process::exit(1);
    }
}

struct Args {
    ckpt: PathBuf,
    name: String,
    tokens: Vec<u32>,
    decode_from: usize,
    tape: Option<PathBuf>,
    force: Option<PathBuf>,
    red: RedArms,
    red_names: Vec<String>,
    gen_tokens: usize,
    bench: usize,
    device: usize,
}

fn parse() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let ckpt = PathBuf::from(
        it.next()
            .ok_or("usage: hy4f-gpu CKPT_DIR --request FILE ...")?,
    );
    let mut a = Args {
        ckpt,
        name: String::new(),
        tokens: Vec::new(),
        decode_from: 0,
        tape: None,
        force: None,
        red: RedArms::default(),
        red_names: Vec::new(),
        gen_tokens: 0,
        bench: 0,
        device: 0,
    };
    let mut request = None;
    while let Some(arg) = it.next() {
        let mut val = || it.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--request" => request = Some(PathBuf::from(val()?)),
            "--tape" => a.tape = Some(PathBuf::from(val()?)),
            "--force-selections" => a.force = Some(PathBuf::from(val()?)),
            "--gen" => a.gen_tokens = val()?.parse().map_err(|e| format!("--gen: {e}"))?,
            "--bench" => a.bench = val()?.parse().map_err(|e| format!("--bench: {e}"))?,
            "--device" => a.device = val()?.parse().map_err(|e| format!("--device: {e}"))?,
            "--red" => {
                let v = val()?;
                match v.as_str() {
                    "engram-off" => a.red.engram_off = true,
                    "ties-reversed" => a.red.ties_reversed = true,
                    "dequant-bf16" => a.red.dequant_bf16 = true,
                    "kv-off-by-one" => a.red.kv_off_by_one = true,
                    "pool-early" => a.red.pool_early = true,
                    _ => return Err(format!("unknown red arm {v}")),
                }
                a.red_names.push(v);
            }
            _ => return Err(format!("unknown argument {arg}")),
        }
    }
    let path = request.ok_or("--request FILE is required")?;
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
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

fn i32s(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The forcing tape's picks per call: (router per layer, indexer per layer), checked against
/// its own hashes and this request. Returns the per-call maps and the tape's sha256.
fn read_forcing(dir: &Path, a: &Args) -> Result<(Vec<Forced>, String), String> {
    let path = dir.join("tape.json");
    let text = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let digest = sha_hex(&text);
    let v: serde_json::Value = serde_json::from_slice(&text).map_err(|e| e.to_string())?;
    if v["format"] != "memra-checkpoint-tape-v2" {
        return Err("the forcing tape is not memra-checkpoint-tape-v2".into());
    }
    if v["forced_selections"].as_bool() == Some(true)
        || v["red_arms"].as_array().is_some_and(|r| !r.is_empty())
    {
        return Err("the forcing tape is itself forced or a red arm".into());
    }
    let ids: Vec<u32> = v["token_ids"]
        .as_array()
        .ok_or("forcing tape: no token_ids")?
        .iter()
        .map(|t| t.as_u64().unwrap_or(u64::MAX) as u32)
        .collect();
    if ids != a.tokens || v["decode_from"].as_u64() != Some(a.decode_from as u64) {
        return Err("the forcing tape scored another request".into());
    }
    let array = |sec: &str, file: &str, sha: &str| -> Result<Vec<i32>, String> {
        let name = v[sec][file]
            .as_str()
            .ok_or(format!("forcing tape: no {sec}.{file}"))?;
        let bytes = std::fs::read(dir.join(name)).map_err(|e| format!("{name}: {e}"))?;
        if v[sec][sha].as_str() != Some(sha_hex(&bytes).as_str()) {
            return Err(format!("forcing tape: {name} does not match its sha256"));
        }
        Ok(i32s(&bytes))
    };
    let isel = array("indexer", "sel_file", "sel_sha256")?;
    let rsel = array("router", "sel_file", "sel_sha256")?;
    let calls = 1 + a.tokens.len() - a.decode_from;
    let mut out: Vec<Forced> = (0..calls).map(|_| Forced::default()).collect();
    for r in v["indexer"]["records"]
        .as_array()
        .ok_or("no indexer records")?
    {
        let (c, l, q, k, o) = (
            r["call"].as_u64().unwrap_or(0) as usize,
            r["layer"].as_u64().unwrap_or(0) as usize,
            r["queries"].as_u64().unwrap_or(0) as usize,
            r["k"].as_u64().unwrap_or(0) as usize,
            r["sel_offset"].as_u64().unwrap_or(0) as usize,
        );
        let groups = r["groups"].as_u64().unwrap_or(0) as i32;
        let slice = isel
            .get(o..o + q * k)
            .ok_or("forcing tape: indexer record past its array")?;
        // a recall is sorted and names groups of its call (the own group may be the open one)
        if slice.chunks(k.max(1)).any(|row| {
            row.windows(2).any(|w| w[0] > w[1]) || row.iter().any(|g| *g < 0 || *g > groups)
        }) {
            return Err(format!(
                "forcing tape: indexer call {c} layer {l} names groups outside its call"
            ));
        }
        out.get_mut(c)
            .ok_or("forcing tape: call past the request")?
            .indexer
            .insert(l, slice.to_vec());
    }
    for r in v["router"]["records"]
        .as_array()
        .ok_or("no router records")?
    {
        let (c, l, q, k, o) = (
            r["call"].as_u64().unwrap_or(0) as usize,
            r["layer"].as_u64().unwrap_or(0) as usize,
            r["queries"].as_u64().unwrap_or(0) as usize,
            r["k"].as_u64().unwrap_or(0) as usize,
            r["sel_offset"].as_u64().unwrap_or(0) as usize,
        );
        let experts = r["experts"].as_u64().unwrap_or(0) as i32;
        let slice = rsel
            .get(o..o + q * k)
            .ok_or("forcing tape: router record past its array")?;
        // the experts are sorted, unique and in range
        if slice.chunks(k.max(1)).any(|row| {
            row.windows(2).any(|w| w[0] >= w[1]) || row.iter().any(|e| *e < 0 || *e >= experts)
        }) {
            return Err(format!(
                "forcing tape: router call {c} layer {l} names experts outside its layer"
            ));
        }
        out.get_mut(c)
            .ok_or("forcing tape: call past the request")?
            .router
            .insert(l, slice.to_vec());
    }
    Ok((out, digest))
}

fn run() -> Result<(), String> {
    // the checkpoint gate asks a runner for its numeric class before it runs it
    if std::env::args().nth(1).as_deref() == Some("--numeric-class") {
        println!("{NUMERIC_CLASS}");
        return Ok(());
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
    // the weights stay resident in unified memory: refuse unless they fit with headroom
    let need: usize = 94_000_000_000 + (16usize << 30);
    let avail = mem_available().unwrap_or(0);
    if avail < need {
        return Err(format!(
            "MemAvailable {:.1} GiB: the resident weights and headroom need {:.1} GiB",
            avail as f64 / (1u64 << 30) as f64,
            need as f64 / (1u64 << 30) as f64
        ));
    }
    let forcing = match &a.force {
        Some(dir) => Some(read_forcing(dir, &a)?),
        None => None,
    };
    let dev = Dev::new(a.device)?;
    let cap = a.tokens.len() + a.gen_tokens.max(if a.bench > 0 { 64 } else { 0 }) + 8;
    let t_load = Instant::now();
    let mut route = Route::load(dev, &ck, &plan, hy, cap, plan.layers.len())?;
    route.red = a.red;
    let load_s = t_load.elapsed().as_secs_f64();
    eprintln!(
        "hy4f-gpu: resident {:.1} GiB in {load_s:.1}s ({:.1} GB read)",
        route.dev.allocated.get() as f64 / (1u64 << 30) as f64,
        ck.bytes_read.get() as f64 / 1e9
    );
    let vocab = plan.vocab_size as usize;

    if a.bench > 0 {
        return bench(&mut route, &a, vocab, load_s);
    }
    if a.gen_tokens > 0 {
        let prompt = &a.tokens[..a.decode_from];
        let logits = route.call(prompt, false, None, None)?;
        let mut next = argmax(&logits) as u32;
        let mut ids = vec![next];
        for _ in 1..a.gen_tokens {
            let logits = route.call(&[next], false, None, None)?;
            next = argmax(&logits) as u32;
            ids.push(next);
        }
        println!(
            "gen {}",
            ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
        );
        return Ok(());
    }

    let t = a.tokens.len();
    let mut calls = vec![(0usize, a.decode_from)];
    calls.extend((a.decode_from..t).map(|q| (q, q + 1)));
    let mut logits = Vec::with_capacity(t * vocab);
    let mut records: Vec<Vec<(usize, LayerRec)>> = Vec::new();
    let mut call_s = Vec::new();
    for (ci, &(s0, s1)) in calls.iter().enumerate() {
        let tc = Instant::now();
        let mut rec = Vec::new();
        let forced = forcing.as_ref().map(|(f, _)| &f[ci]);
        logits.extend(route.call(&a.tokens[s0..s1], true, Some(&mut rec), forced)?);
        records.push(rec);
        call_s.push(tc.elapsed().as_secs_f64());
        eprintln!("hy4f-gpu: call {ci} tokens {s0}..{s1} {:.2}s", call_s[ci]);
    }
    let am: Vec<usize> = (0..t)
        .map(|r| argmax(&logits[r * vocab..(r + 1) * vocab]))
        .collect();
    eprintln!(
        "hy4f-gpu: {t} positions, {:.1}s, argmax tail {:?}",
        t_start.elapsed().as_secs_f64(),
        &am[t.saturating_sub(8)..]
    );
    if let Some(dir) = &a.tape {
        write_tape(
            dir, &a, &plan, &calls, &logits, &records, &am, vocab, &forcing, &call_s, load_s,
            t_start,
        )?;
    }
    Ok(())
}

fn bench(route: &mut Route, a: &Args, vocab: usize, load_s: f64) -> Result<(), String> {
    let prompt = &a.tokens[..a.decode_from];
    let decode = 64usize;
    let mut rows = Vec::new();
    for run in 0..=a.bench {
        route.reset();
        route.dev.sync()?;
        let tp = Instant::now();
        let logits = route.call(prompt, false, None, None)?;
        route.dev.sync()?;
        let prefill_s = tp.elapsed().as_secs_f64();
        let mut next = argmax(&logits[..vocab]) as u32;
        let mut ids = vec![next];
        let mut steps = Vec::with_capacity(decode);
        let td = Instant::now();
        for _ in 1..decode {
            let ts = Instant::now();
            let logits = route.call(&[next], false, None, None)?;
            next = argmax(&logits) as u32;
            ids.push(next);
            steps.push(ts.elapsed().as_secs_f64());
        }
        let decode_s = td.elapsed().as_secs_f64();
        let mut sorted = steps.clone();
        sorted.sort_by(f64::total_cmp);
        let row = serde_json::json!({
            "run": run,
            "warmup": run == 0,
            "prompt_tokens": prompt.len(),
            "prefill_s": prefill_s,
            "prefill_tok_s": prompt.len() as f64 / prefill_s,
            "decode_calls": steps.len(),
            "decode_s": decode_s,
            "decode_tok_s": steps.len() as f64 / decode_s,
            "decode_step_p50_ms": sorted[sorted.len() / 2] * 1e3,
            "decode_step_p99_ms": sorted[(sorted.len() * 99 / 100).min(sorted.len() - 1)] * 1e3,
            "ids_fnv1a": fnv1a(&ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")),
            "first_ids": ids.iter().take(16).collect::<Vec<_>>(),
            "steps_ms": steps.iter().map(|s| (s * 1e5).round() / 100.0).collect::<Vec<_>>(),
        });
        println!("{row}");
        rows.push(row);
    }
    eprintln!("hy4f-gpu: bench done, load {load_s:.1}s");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_tape(
    dir: &Path,
    a: &Args,
    plan: &memra_gguf::model_plan::ModelPlan,
    calls: &[(usize, usize)],
    logits: &[f32],
    records: &[Vec<(usize, LayerRec)>],
    am: &[usize],
    vocab: usize,
    forcing: &Option<(Vec<Forced>, String)>,
    call_s: &[f64],
    load_s: f64,
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
    let logit_bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    let logits_sha = w("logits.f32", &logit_bytes)?;
    drop(logit_bytes);
    let (mut sel, mut lg) = (Vec::<u8>::new(), Vec::<u8>::new());
    let (mut so, mut lo) = (0usize, 0usize);
    let (mut rsel, mut rsc) = (Vec::<u8>::new(), Vec::<u8>::new());
    let (mut rso, mut rco) = (0usize, 0usize);
    let (mut idx, mut ridx, mut engram) = (Vec::new(), Vec::new(), Vec::new());
    for (ci, &(start, end)) in calls.iter().enumerate() {
        let q = end - start;
        for (il, rec) in &records[ci] {
            if let Some((k, groups, s, l)) = &rec.recalled {
                if s.len() != q * k || l.len() != q * groups {
                    return Err("an indexer record is not its call's geometry".into());
                }
                sel.extend(s.iter().flat_map(|v| v.to_le_bytes()));
                lg.extend(l.iter().flat_map(|v| v.to_le_bytes()));
                idx.push(
                    serde_json::json!({"call": ci, "start": start, "tokens": q, "layer": il,
                    "queries": q, "k": k, "groups": groups, "sel_offset": so, "logit_offset": lo}),
                );
                so += q * k;
                lo += q * groups;
            }
            if let Some(rows) = &rec.engram_rows {
                let text = rows
                    .iter()
                    .flat_map(|r| r.iter().map(u64::to_string))
                    .collect::<Vec<_>>()
                    .join(",");
                engram.push(serde_json::json!({"call": ci, "layer": il, "rows": fnv1a(&text)}));
            }
        }
    }
    for (ci, &(start, end)) in calls.iter().enumerate() {
        let q = end - start;
        for (il, rec) in &records[ci] {
            if let Some((k, experts, p, s)) = &rec.routes {
                if p.len() != q * k || s.len() != q * experts {
                    return Err("a router record is not its call's geometry".into());
                }
                rsel.extend(p.iter().flat_map(|v| v.to_le_bytes()));
                rsc.extend(s.iter().flat_map(|v| v.to_le_bytes()));
                ridx.push(
                    serde_json::json!({"call": ci, "layer": il, "queries": q, "k": k,
                    "experts": experts, "sel_offset": rso, "score_offset": rco}),
                );
                rso += q * k;
                rco += q * experts;
            }
        }
    }
    let rsel_sha = w("router_sel.i32", &rsel)?;
    let rsc_sha = w("router_scores.f32", &rsc)?;
    let sel_sha = w("indexer_sel.i32", &sel)?;
    let lg_sha = w("indexer_logits.f32", &lg)?;
    let tape = serde_json::json!({
        "format": "memra-checkpoint-tape-v2",
        "engine": format!("memra-hy4f-gpu-sm_{}", memra_engine::BUILT_CUDA_TARGET),
        "numeric_class": NUMERIC_CLASS,
        "request": a.name,
        "token_ids": a.tokens,
        "decode_from": a.decode_from,
        "vocab": vocab,
        "positions": a.tokens.len(),
        "layers": plan.layers.len(),
        "indexed_layers": indexed_layers(plan),
        "red_arms": a.red_names,
        "argmax": am,
        "logits": {"file": "logits.f32", "sha256": logits_sha},
        "indexer": {"sel_file": "indexer_sel.i32", "sel_sha256": sel_sha, "logit_file": "indexer_logits.f32",
                    "logit_sha256": lg_sha, "records": idx},
        "router": {"sel_file": "router_sel.i32", "sel_sha256": rsel_sha, "score_file": "router_scores.f32",
                   "score_sha256": rsc_sha, "records": ridx},
        "engram": engram,
        "forced_selections": forcing.is_some(),
        "forcing_tape_sha256": forcing.as_ref().map(|f| f.1.clone()),
        "load_seconds": (load_s * 10.0).round() / 10.0,
        "call_seconds": call_s.iter().map(|s| (s * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        "seconds": (t_start.elapsed().as_secs_f64() * 10.0).round() / 10.0,
    });
    let text = serde_json::to_string_pretty(&tape).map_err(|e| e.to_string())?;
    w("tape.json", text.as_bytes())?;
    Ok(())
}
