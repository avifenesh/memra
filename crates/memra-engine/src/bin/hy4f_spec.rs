//! Hy4-flash DFly2 drafts on the GPU route (`memra_engine::hy4f_route`, its `dfly` drafter).
//!
//! usage: hy4f-spec gate CKPT_DIR REQUEST TARGET_DIR ORACLE_TAPE OUT [--red ARM]
//!        hy4f-spec gen CKPT_DIR REQUEST N [--k K] [--prefill-chunk C] [--bench R]
//!
//! - `gate`: the drafter gate's native side. The route holds only the embedding, the head and the
//!   drafter; every round drafts from the target inputs the f64 target oracle dumped
//!   (`make_checkpoint_oracle.py --dump-dfly`), rounded once to F32 (the route's cache), at the
//!   oracle tape's anchors, along the oracle's drafts (forced), and writes the
//!   `memra-hy4f-dfly-tape-v1` the lane's comparator reads (`dfly/compare_dfly.py`). Red arms:
//!   `layers`, `width`, `no-correction`, `topk`.
//! - `gate-route CKPT REQUEST FORCING_TAPE ORACLE_TAPE OUT [--red ARM]`: the composite gate. The
//!   whole route runs the request teacher-forced with the f64 oracle's router and indexer picks
//!   applied (the target class's gate run), keeps every row's final hidden row and every target
//!   index layer's history, then drafts at the drafter oracle tape's anchors (last first) from
//!   its own cache, along the oracle's drafts; the band is the composite bf16x2a backend's.
//! - `replay CKPT PROMPTS OUT [--k K] [--prefill-chunk C]`: teacher-forced draft replay. For each
//!   prompt of PROMPTS (a JSON list of `{"name", "prompt_ids", "n", "temperature", "seed",
//!   "eos"}`), the route generates the answer one token a call, sampled as the serve route samples
//!   (the position-keyed draw at the request's temperature and seed; temperature 0 is the first
//!   maximum), up to `n` tokens or the end token, and after every call drafts a round of K from
//!   that call's row twice: greedy drafts (the first maximum) and drafts drawn with the request's
//!   own chooser at their positions. Writes one JSON line a prompt: the ids, and for every anchor
//!   `i` how many of each round's drafts equal the answer's next tokens from the front
//!   (`m_greedy[i]`, `m_chosen[i]`; for a sampled prompt also `m_half[i]`, the coupled draw from
//!   the drafter's row at half the request's temperature). The DFly2 block is causal, so a round of `k < K` drafts keeps
//!   `min(m, k)`: one replay prices every depth. The answer is the target's own, so arms of another
//!   drafter (another checkpoint directory over the same target) replay the same tokens.
//! - `cost CKPT REQUEST [--k K] [--reps R] [--temperature T]`: what a step costs by its shape, after the request's
//!   prefill: the one-token call, the verify call of 2 to K + 1 rows (its ring backup and rollback
//!   included) and the draft round of 1 to K drafts, R times each, interleaved shape by shape;
//!   prints the medians in milliseconds (the draft gate's break-even).
//! - `gen`: greedy generation of N tokens after the request's prefill, twice in one process:
//!   plain (one decode call a token) and speculative (a draft round of K after every call, then
//!   one verify call scoring the anchor and the K drafts, the longest agreeing prefix kept and
//!   the rest rolled back). Prints both id lists, whether they are equal token for token, whether
//!   every emitted token's logits row is bit-identical between the two, and the acceptance. With
//!   `--bench R` it times R more runs of each after a warm-up.

use memra_engine::dsv4_gpu::{Dsv4SampleCfg, dsv4_sample_row};
use memra_engine::hy4f_route::Dev;
use memra_engine::hy4f_route::{Ckpt, DraftRed, ExternalTarget, Route, Session, mem_available};
use memra_gguf::config::{HfConfig, ModelConfig};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("hy4f-spec: {e}");
        std::process::exit(1);
    }
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn u32s(v: &Value) -> Vec<u32> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64())
                .map(|x| x as u32)
                .collect()
        })
        .unwrap_or_default()
}

fn sha_hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if v.total_cmp(&row[best]).is_gt() {
            best = i;
        }
    }
    best as u32
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("gate") => gate(&args[1..]),
        Some("gen") => generate(&args[1..]),
        Some("gate-route") => gate_route(&args[1..]),
        Some("replay") => replay(&args[1..]),
        Some("cost") => cost(&args[1..]),
        _ => Err(
            "usage: hy4f-spec gate CKPT REQUEST TARGET ORACLE_TAPE OUT [--red ARM] | \
                  gen CKPT REQUEST N [--k K] [--prefill-chunk C] [--bench R] | \
                  replay CKPT PROMPTS OUT [--k K] [--prefill-chunk C] | \
                  cost CKPT REQUEST [--k K] [--reps R]"
                .into(),
        ),
    }
}

fn plan_of(ckpt: &Path) -> Result<(ModelConfig, memra_gguf::model_plan::ModelPlan), String> {
    let cfg_text = std::fs::read_to_string(ckpt.join("config.json"))
        .map_err(|e| format!("config.json: {e}"))?;
    let config = ModelConfig::from_hf(&HfConfig::try_parse(&cfg_text)?);
    let pack = memra_gguf::model_packs::by_alias("hy_v4_flash").ok_or("no hy_v4_flash pack")?;
    let plan = pack
        .compile_plan(&config)
        .map_err(|e| format!("plan: {e:?}"))?;
    Ok((config, plan))
}

fn read_f64(path: &Path) -> Result<Vec<f64>, String> {
    Ok(std::fs::read(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn gate(args: &[String]) -> Result<(), String> {
    let [ckpt, request, target, oracle, out, ..] = args else {
        return Err("gate CKPT REQUEST TARGET ORACLE_TAPE OUT".into());
    };
    let red_name = flag(args, "--red").unwrap_or("");
    let red = DraftRed::parse(red_name).ok_or(format!("unknown red arm {red_name:?}"))?;
    let (ckpt, target, oracle, out) = (
        Path::new(ckpt),
        Path::new(target),
        Path::new(oracle),
        Path::new(out),
    );
    let (config, plan) = plan_of(ckpt)?;
    let hy = config
        .hy_v4_flash
        .as_ref()
        .ok_or("not a Hy4-flash config")?;
    let req = read_json(Path::new(request))?;
    let tokens = u32s(&req["token_ids"]);
    let ot = read_json(&oracle.join("tape.json"))?;
    if u32s(&ot["token_ids"]) != tokens
        || ot["forced_drafts"] != json!(false)
        || !ot["red_arm"].is_null()
    {
        return Err("the oracle tape is not the f64 oracle's of this request".into());
    }
    let anchors: Vec<usize> = ot["anchors"]
        .as_array()
        .ok_or("anchors")?
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let k = ot["k"].as_u64().ok_or("k")? as usize;
    let drafts: Vec<Vec<u32>> = ot["drafts"]
        .as_array()
        .ok_or("drafts")?
        .iter()
        .map(u32s)
        .collect();
    let tj = read_json(&target.join("target.json"))?;
    let h = plan.hidden_size as usize;
    let hidden: Vec<f32> = read_f64(&target.join(tj["hidden"]["file"].as_str().ok_or("hidden")?))?
        .into_iter()
        .map(|v| v as f32)
        .collect();
    let mut kv: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut hist_all: HashMap<usize, (Vec<i32>, Vec<i64>)> = HashMap::new();
    for l in tj["layers"].as_array().ok_or("layers")? {
        let layer = l["layer"].as_u64().ok_or("layer")? as usize;
        kv.insert(
            layer,
            read_f64(&target.join(l["file"].as_str().ok_or("file")?))?
                .into_iter()
                .map(|v| v as f32)
                .collect(),
        );
        let flat = std::fs::read(target.join(l["history_file"].as_str().ok_or("hf")?))
            .map_err(|e| e.to_string())?
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let offs = std::fs::read(target.join(l["history_offsets_file"].as_str().ok_or("ho")?))
            .map_err(|e| e.to_string())?
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        hist_all.insert(layer, (flat, offs));
    }
    let ck = Ckpt::open(ckpt)?;
    let dev = Dev::new(0)?;
    let t0 = Instant::now();
    let mut route = Route::load(dev, &ck, &plan, hy, 8, 0)?;
    route.load_drafter(&ck)?;
    route.drafter_mut().unwrap().red = red;
    eprintln!(
        "hy4f-spec gate: drafter and head resident in {:.1}s",
        t0.elapsed().as_secs_f64()
    );
    let vocab = plan.vocab_size as usize;
    let mut logits: Vec<u8> = Vec::with_capacity(anchors.len() * k * vocab * 4);
    let mut own = Vec::new();
    let tr = Instant::now();
    for (n, &p) in anchors.iter().enumerate() {
        let history: HashMap<usize, Vec<i32>> = hist_all
            .iter()
            .map(|(l, (flat, offs))| (*l, flat[offs[p] as usize..offs[p + 1] as usize].to_vec()))
            .collect();
        let ext = ExternalTarget {
            hidden: &hidden[p * h..(p + 1) * h],
            kv: &kv,
            history: &history,
        };
        let r = route.draft_external(&ext, p, tokens[p + 1], k, Some(&drafts[n]))?;
        for v in &r.logits {
            logits.extend_from_slice(&v.to_le_bytes());
        }
        own.push(r.ids);
    }
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    std::fs::write(out.join("logits.f32"), &logits).map_err(|e| e.to_string())?;
    let mut accepted = Vec::new();
    for (n, &p) in anchors.iter().enumerate() {
        let mut m = 0;
        while m < k && p + 2 + m < tokens.len() && own[n][m] == tokens[p + 2 + m] {
            m += 1;
        }
        accepted.push(m);
    }
    let tape = json!({
        "format": "memra-hy4f-dfly-tape-v1",
        "engine": "memra-engine-hy4f_route-dfly-gpu",
        "numeric_class": memra_engine::hy4f_route::NUMERIC_CLASS,
        "backend_dtype": "bf16x2-route",
        "red_arm": if red == DraftRed::None { Value::Null } else { json!(red_name) },
        "request": req["name"],
        "token_ids": tokens,
        "k": k,
        "vocab": vocab,
        "anchors": anchors,
        "drafts": own,
        "accepted_teacher_forced": accepted,
        "logits": {"file": "logits.f32", "sha256": sha_hex(&logits), "shape": [anchors.len(), k, vocab]},
        "forced_drafts": true,
        "seconds": tr.elapsed().as_secs_f64(),
    });
    std::fs::write(
        out.join("tape.json"),
        serde_json::to_string_pretty(&tape).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    println!(
        "{}",
        json!({"request": tape["request"], "anchors": anchors.len(), "k": k, "red": tape["red_arm"],
               "logits_sha256": tape["logits"]["sha256"], "seconds": tape["seconds"]})
    );
    Ok(())
}

/// One greedy run: plain (`k == 0`) or speculative. Returns the ids, each emitted token's logits
/// row, the rounds' accepted counts, the drafts made, and the prefill and decode seconds. A round
/// is a verify call; the last token is a plain call when the budget leaves no draft to keep.
#[allow(clippy::type_complexity)]
fn greedy(
    route: &mut Route,
    v: usize,
    sess: &mut Session,
    prompt: &[u32],
    n: usize,
    k: usize,
    chunk: usize,
) -> Result<(Vec<u32>, Vec<Vec<f32>>, Vec<usize>, usize, f64, f64), String> {
    sess.reset();
    let tp = Instant::now();
    let mut last = Vec::new();
    let c = if chunk == 0 { prompt.len() } else { chunk };
    let mut s0 = 0;
    while s0 < prompt.len() {
        let s1 = (s0 + c).min(prompt.len());
        last = route.call_in(sess, &prompt[s0..s1], false, None, None)?;
        s0 = s1;
    }
    route.dev.sync()?;
    let prefill_s = tp.elapsed().as_secs_f64();
    let mut ids = vec![argmax(&last)];
    let mut rows = vec![last];
    let mut accepted = Vec::new();
    let mut drafted = 0usize;
    let td = Instant::now();
    // the source row of the last call: its last row
    let mut src_row = (prompt.len() - 1) % c;
    while ids.len() < n {
        let next = *ids.last().unwrap();
        // a verify call of `kk` drafts emits up to `kk + 1` tokens: no more drafts than the budget
        // can still emit after the anchor's row
        let kk = k.min(n - ids.len() - 1);
        if kk == 0 {
            let row = route.call_in(sess, &[next], false, None, None)?;
            ids.push(argmax(&row));
            rows.push(row);
            // the next round's source row is this call's only row
            src_row = 0;
            continue;
        }
        drafted += kk;
        let d = route.draft_in(sess, src_row, next, kk, None, false, None)?;
        let mut verify = vec![next];
        verify.extend(&d.ids);
        let from = sess.positions();
        let backup = route.ring_backup_in(sess, from, verify.len())?;
        let out = route.call_in(sess, &verify, true, None, None)?;
        let mut m = 0;
        let mut emitted = Vec::new();
        for j in 0..verify.len() {
            let row = &out[j * v..(j + 1) * v];
            let t = argmax(row);
            emitted.push((t, row.to_vec()));
            if j < d.ids.len() && t == d.ids[j] {
                m += 1;
            } else {
                break;
            }
        }
        accepted.push(m);
        // positions kept: from .. from + m + 1 (the anchor and the m accepted drafts)
        route.rollback_in(sess, from + m + 1, &backup)?;
        src_row = m;
        for (t, row) in emitted {
            if ids.len() < n {
                ids.push(t);
                rows.push(row);
            }
        }
    }
    route.dev.sync()?;
    Ok((
        ids,
        rows,
        accepted,
        drafted,
        prefill_s,
        td.elapsed().as_secs_f64(),
    ))
}

fn generate(args: &[String]) -> Result<(), String> {
    let [ckpt, request, n, ..] = args else {
        return Err("gen CKPT REQUEST N".into());
    };
    let n: usize = n.parse().map_err(|e| format!("N: {e}"))?;
    let ckpt = Path::new(ckpt);
    let (config, plan) = plan_of(ckpt)?;
    let hy = config
        .hy_v4_flash
        .as_ref()
        .ok_or("not a Hy4-flash config")?;
    let req = read_json(Path::new(request))?;
    let tokens = u32s(&req["token_ids"]);
    let decode_from = req["decode_from"].as_u64().ok_or("decode_from")? as usize;
    let prompt = &tokens[..decode_from];
    let chunk: usize = flag(args, "--prefill-chunk")
        .map_or(Ok(0), str::parse)
        .map_err(|e| format!("{e}"))?;
    let bench: usize = flag(args, "--bench")
        .map_or(Ok(0), str::parse)
        .map_err(|e| format!("{e}"))?;
    let need: usize = 95_000_000_000 + (16usize << 30);
    if mem_available().unwrap_or(0) < need {
        return Err(
            "MemAvailable cannot hold the resident weights, the drafter and headroom".into(),
        );
    }
    let ck = Ckpt::open(ckpt)?;
    let dev = Dev::new(0)?;
    let t0 = Instant::now();
    let mut route = Route::load(dev, &ck, &plan, hy, 8, plan.layers.len())?;
    route.load_drafter(&ck)?;
    let dw = route.drafter_mut().unwrap().weight_bytes;
    let k: usize = match flag(args, "--k") {
        Some(s) => s.parse().map_err(|e| format!("--k: {e}"))?,
        None => route
            .drafter_mut()
            .unwrap()
            .k
            .ok_or("no declared k; pass --k")?,
    };
    eprintln!(
        "hy4f-spec: resident {:.1} GiB (drafter {:.2} GiB) in {:.1}s; k={k}",
        route.dev.live_bytes() as f64 / (1u64 << 30) as f64,
        dw as f64 / (1u64 << 30) as f64,
        t0.elapsed().as_secs_f64()
    );
    drop(ck);
    let mut sess = route.new_session(prompt.len() + n + k + 2)?;
    let (plain, prow, _, _, _, plain_s) = greedy(
        &mut route,
        plan.vocab_size as usize,
        &mut sess,
        prompt,
        n,
        0,
        chunk,
    )?;
    let (spec, srow, acc, drafted, _, spec_s) = greedy(
        &mut route,
        plan.vocab_size as usize,
        &mut sess,
        prompt,
        n,
        k,
        chunk,
    )?;
    let rows_equal = prow.len() == srow.len()
        && prow
            .iter()
            .zip(&srow)
            .all(|(a, b)| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
    let flat = |rows: &[Vec<f32>]| -> String {
        let mut h = Sha256::new();
        for r in rows {
            for v in r {
                h.update(v.to_le_bytes());
            }
        }
        h.finalize().iter().map(|x| format!("{x:02x}")).collect()
    };
    let rounds = acc.len();
    let accepted: usize = acc.iter().sum();
    let mut hist = vec![0usize; k + 1];
    for a in &acc {
        hist[*a] += 1;
    }
    println!(
        "{}",
        json!({
            "request": req["name"], "prompt_tokens": prompt.len(), "tokens": n, "k": k,
            "prefill_chunk": chunk,
            "ids_equal": plain == spec, "logits_rows_bit_equal": rows_equal,
            "plain_rows_sha256": flat(&prow), "spec_rows_sha256": flat(&srow),
            "plain_ids": plain, "spec_ids": spec,
            "rounds": rounds, "drafted": drafted, "accepted": accepted,
            "accepted_per_round": accepted as f64 / rounds.max(1) as f64,
            "tokens_per_round": (n - 1) as f64 / rounds.max(1) as f64,
            "acceptance_rate": accepted as f64 / drafted.max(1) as f64,
            "accepted_histogram": hist,
            "plain_decode_s": plain_s, "spec_decode_s": spec_s,
            "plain_decode_tok_s": (n - 1) as f64 / plain_s, "spec_decode_tok_s": (n - 1) as f64 / spec_s,
        })
    );
    let mut exact = plain == spec && rows_equal;
    for run in 0..bench {
        for (arm, kk) in [("plain", 0), ("spec", k)] {
            let (ids, _, acc, drafted, prefill_s, dec_s) = greedy(
                &mut route,
                plan.vocab_size as usize,
                &mut sess,
                prompt,
                n,
                kk,
                chunk,
            )?;
            println!(
                "{}",
                json!({"bench_run": run, "arm": arm, "tokens": ids.len(), "prefill_s": prefill_s,
                       "decode_s": dec_s, "decode_tok_s": (ids.len() - 1) as f64 / dec_s,
                       "rounds": acc.len(), "drafted": drafted,
                       "accepted": acc.iter().sum::<usize>(), "ids_equal_plain": ids == plain})
            );
            exact &= ids == plain;
        }
    }
    if !exact {
        return Err("drafted greedy differs from plain greedy".into());
    }
    Ok(())
}

/// The serve route's choice for a row: the first maximum, or the position-keyed draw.
fn pick_row(row: &[f32], pos: usize, sample: Option<&Dsv4SampleCfg>) -> Result<u32, String> {
    if row.iter().any(|x| !x.is_finite()) {
        return Err("non-finite logits".into());
    }
    match sample {
        None => Ok(argmax(row)),
        Some(cfg) => dsv4_sample_row(row, pos, cfg),
    }
}

/// How many of `drafts` equal `ids` from the front.
fn lead(drafts: &[u32], ids: &[u32]) -> usize {
    drafts.iter().zip(ids).take_while(|(a, b)| a == b).count()
}

fn replay(args: &[String]) -> Result<(), String> {
    let [ckpt, prompts, out, ..] = args else {
        return Err("replay CKPT PROMPTS OUT".into());
    };
    let ckpt = Path::new(ckpt);
    let (config, plan) = plan_of(ckpt)?;
    let hy = config
        .hy_v4_flash
        .as_ref()
        .ok_or("not a Hy4-flash config")?;
    let list = read_json(Path::new(prompts))?;
    let list = list.as_array().ok_or("PROMPTS is not a JSON list")?;
    let chunk: usize = flag(args, "--prefill-chunk")
        .map_or(Ok(2048), str::parse)
        .map_err(|e| format!("{e}"))?;
    let need: usize = 95_000_000_000 + (16usize << 30);
    if mem_available().unwrap_or(0) < need {
        return Err(
            "MemAvailable cannot hold the resident weights, the drafter and headroom".into(),
        );
    }
    let ck = Ckpt::open(ckpt)?;
    let dev = Dev::new(0)?;
    let t0 = Instant::now();
    let mut route = Route::load(dev, &ck, &plan, hy, 8, plan.layers.len())?;
    route.load_drafter(&ck)?;
    let k: usize = match flag(args, "--k") {
        Some(s) => s.parse().map_err(|e| format!("--k: {e}"))?,
        None => route
            .drafter_mut()
            .unwrap()
            .k
            .ok_or("no declared k; pass --k")?,
    };
    eprintln!(
        "hy4f-spec replay: resident {:.1} GiB in {:.1}s; k={k}",
        route.dev.live_bytes() as f64 / (1u64 << 30) as f64,
        t0.elapsed().as_secs_f64()
    );
    drop(ck);
    let mut w = std::fs::File::create(out).map_err(|e| format!("{out}: {e}"))?;
    for p in list {
        let name = p["name"].as_str().unwrap_or("?").to_string();
        let prompt = u32s(&p["prompt_ids"]);
        let n = p["n"].as_u64().ok_or("n")? as usize;
        let eos = p["eos"].as_u64().map(|x| x as u32);
        let temperature = p["temperature"].as_f64().unwrap_or(0.0) as f32;
        let sample = (temperature > 0.0).then(|| Dsv4SampleCfg {
            temperature,
            top_p: 1.0,
            top_k: 0,
            seed: p["seed"].as_u64().unwrap_or(0),
        });
        if prompt.is_empty() || n == 0 {
            return Err(format!("{name}: an empty prompt or budget"));
        }
        let tr = Instant::now();
        let mut sess = route.new_session(prompt.len() + n + k + 2)?;
        let mut last = Vec::new();
        let mut s0 = 0;
        while s0 < prompt.len() {
            let s1 = (s0 + chunk).min(prompt.len());
            last = route.call_in(&mut sess, &prompt[s0..s1], false, None, None)?;
            s0 = s1;
        }
        let pl = prompt.len();
        let mut src = (pl - 1) % chunk;
        let mut ids = vec![pick_row(&last, pl, sample.as_ref())?];
        let mut greedy_drafts: Vec<Vec<u32>> = Vec::new();
        let mut chosen_drafts: Vec<Vec<u32>> = Vec::new();
        let mut half_drafts: Vec<Vec<u32>> = Vec::new();
        loop {
            let t = *ids.last().unwrap();
            if ids.len() >= n || eos == Some(t) {
                break;
            }
            // the round after this call, both ways; its block rows sit past the session's
            // positions and the next call writes them again before any query reads them
            let g = route.draft_in(&mut sess, src, t, k, None, false, None)?;
            let pos0 = pl + ids.len();
            let mut pick = |s: usize, row: &[f32]| pick_row(row, pos0 + s, sample.as_ref());
            let c = route.draft_in(&mut sess, src, t, k, None, false, Some(&mut pick))?;
            greedy_drafts.push(g.ids);
            chosen_drafts.push(c.ids);
            // the same coupled draw from the drafter's row sharpened to half the temperature
            if let Some(cfg) = sample.as_ref() {
                let half = Dsv4SampleCfg {
                    temperature: cfg.temperature * 0.5,
                    ..*cfg
                };
                let mut pick = |s: usize, row: &[f32]| pick_row(row, pos0 + s, Some(&half));
                let hd = route.draft_in(&mut sess, src, t, k, None, false, Some(&mut pick))?;
                half_drafts.push(hd.ids);
            }
            let row = route.call_in(&mut sess, &[t], false, None, None)?;
            ids.push(pick_row(&row, pl + ids.len(), sample.as_ref())?);
            src = 0;
        }
        let m_greedy: Vec<usize> = greedy_drafts
            .iter()
            .enumerate()
            .map(|(i, d)| lead(d, &ids[i + 1..]))
            .collect();
        let m_chosen: Vec<usize> = chosen_drafts
            .iter()
            .enumerate()
            .map(|(i, d)| lead(d, &ids[i + 1..]))
            .collect();
        let m_half: Vec<usize> = half_drafts
            .iter()
            .enumerate()
            .map(|(i, d)| lead(d, &ids[i + 1..]))
            .collect();
        let line = json!({
            "name": name, "prompt_tokens": pl, "n": n, "k": k, "temperature": temperature,
            "seed": sample.as_ref().map(|c| c.seed), "ids": ids, "m_greedy": m_greedy,
            "m_chosen": m_chosen, "m_half": m_half, "seconds": tr.elapsed().as_secs_f64(),
        });
        use std::io::Write;
        writeln!(w, "{line}").map_err(|e| format!("{out}: {e}"))?;
        eprintln!(
            "hy4f-spec replay: {name}: {} tokens, mean m greedy {:.3}, chosen {:.3} ({:.1}s)",
            ids.len(),
            m_greedy.iter().sum::<usize>() as f64 / m_greedy.len().max(1) as f64,
            m_chosen.iter().sum::<usize>() as f64 / m_chosen.len().max(1) as f64,
            tr.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

fn cost(args: &[String]) -> Result<(), String> {
    let [ckpt, request, ..] = args else {
        return Err("cost CKPT REQUEST".into());
    };
    let ckpt = Path::new(ckpt);
    let (config, plan) = plan_of(ckpt)?;
    let hy = config
        .hy_v4_flash
        .as_ref()
        .ok_or("not a Hy4-flash config")?;
    let req = read_json(Path::new(request))?;
    let tokens = u32s(&req["token_ids"]);
    let decode_from = req["decode_from"].as_u64().ok_or("decode_from")? as usize;
    let prompt = &tokens[..decode_from];
    let reps: usize = flag(args, "--reps")
        .map_or(Ok(8), str::parse)
        .map_err(|e| format!("--reps: {e}"))?;
    let need: usize = 95_000_000_000 + (16usize << 30);
    if mem_available().unwrap_or(0) < need {
        return Err(
            "MemAvailable cannot hold the resident weights, the drafter and headroom".into(),
        );
    }
    let ck = Ckpt::open(ckpt)?;
    let dev = Dev::new(0)?;
    let mut route = Route::load(dev, &ck, &plan, hy, 8, plan.layers.len())?;
    route.load_drafter(&ck)?;
    let k: usize = match flag(args, "--k") {
        Some(s) => s.parse().map_err(|e| format!("--k: {e}"))?,
        None => route
            .drafter_mut()
            .unwrap()
            .k
            .ok_or("no declared k; pass --k")?,
    };
    drop(ck);
    let v = plan.vocab_size as usize;
    let mut sess = route.new_session(prompt.len() + reps * (k + 2) + k + 4)?;
    let mut last = Vec::new();
    for c in prompt.chunks(2048) {
        last = route.call_in(&mut sess, c, false, None, None)?;
    }
    let mut t = argmax(&last);
    // the source row of the next round: the last call's last row
    let mut src = (prompt.len() - 1) % 2048;
    // `--temperature T`: drafts and choices drawn as a sampled request's are (each verify row
    // chosen: the most a call chooses), so the prices include the host sampler
    let temperature: f32 = flag(args, "--temperature")
        .map_or(Ok(0.0), str::parse)
        .map_err(|e| format!("--temperature: {e}"))?;
    let sample = (temperature > 0.0).then_some(Dsv4SampleCfg {
        temperature,
        top_p: 1.0,
        top_k: 0,
        seed: 20261121,
    });
    let mut verify_ms: Vec<Vec<f64>> = vec![Vec::new(); k + 1];
    let mut draft_ms: Vec<Vec<f64>> = vec![Vec::new(); k];
    for _ in 0..reps {
        for d in 1..=k {
            let pos0 = sess.positions() + 1;
            let mut pick = |s: usize, row: &[f32]| pick_row(row, pos0 + s, sample.as_ref());
            let td = Instant::now();
            let out = route.draft_in(&mut sess, src, t, d, None, false, Some(&mut pick))?;
            draft_ms[d - 1].push(td.elapsed().as_secs_f64() * 1e3);
            // the verify call of the anchor and these drafts, rolled back to the anchor
            let mut call = vec![t];
            call.extend(&out.ids);
            let tv = Instant::now();
            let from = sess.positions();
            let backup = route.ring_backup_in(&sess, from, call.len())?;
            let rows = route.call_in(&mut sess, &call, true, None, None)?;
            let mut chosen = Vec::with_capacity(call.len());
            for r in 0..call.len() {
                chosen.push(pick_row(
                    &rows[r * v..(r + 1) * v],
                    from + 1 + r,
                    sample.as_ref(),
                )?);
            }
            route.rollback_in(&mut sess, from + 1, &backup)?;
            verify_ms[d].push(tv.elapsed().as_secs_f64() * 1e3);
            // the session moves one token on, as a one-token call would have
            t = chosen[0];
            src = 0;
        }
        let tc = Instant::now();
        let row = route.call_in(&mut sess, &[t], false, None, None)?;
        let next = pick_row(&row, sess.positions(), sample.as_ref())?;
        verify_ms[0].push(tc.elapsed().as_secs_f64() * 1e3);
        t = next;
        src = 0;
    }
    let med = |x: &[f64]| -> f64 {
        let mut y = x.to_vec();
        y.sort_by(f64::total_cmp);
        y[y.len() / 2]
    };
    println!(
        "{}",
        json!({
            "request": req["name"], "prompt_tokens": prompt.len(), "k": k, "reps": reps,
            "temperature": temperature,
            // index i: a call of i + 1 rows; a round of i + 1 drafts
            "verify_ms_by_rows": (0..=k).map(|r| med(&verify_ms[r])).collect::<Vec<_>>(),
            "draft_ms_by_drafts": (0..k).map(|d| med(&draft_ms[d])).collect::<Vec<_>>(),
            "verify_rows_runs": verify_ms, "draft_runs": draft_ms,
        })
    );
    Ok(())
}

fn i32s(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The f64 target oracle tape's router and indexer picks per call (as `hy4f-gpu
/// --force-selections` reads them), checked against its own hashes and the request.
fn read_forcing(
    dir: &Path,
    tokens: &[u32],
    decode_from: usize,
) -> Result<Vec<memra_engine::hy4f_route::Forced>, String> {
    let v = read_json(&dir.join("tape.json"))?;
    if v["format"] != "memra-checkpoint-tape-v2" || v["forced_selections"].as_bool() == Some(true) {
        return Err("the forcing tape is not an unforced memra-checkpoint-tape-v2".into());
    }
    if u32s(&v["token_ids"]) != tokens || v["decode_from"].as_u64() != Some(decode_from as u64) {
        return Err("the forcing tape scored another request".into());
    }
    let array = |sec: &str, file: &str, sha: &str| -> Result<Vec<i32>, String> {
        let name = v[sec][file].as_str().ok_or(format!("no {sec}.{file}"))?;
        let bytes = std::fs::read(dir.join(name)).map_err(|e| format!("{name}: {e}"))?;
        if v[sec][sha].as_str() != Some(sha_hex(&bytes).as_str()) {
            return Err(format!("{name} does not match its sha256"));
        }
        Ok(i32s(&bytes))
    };
    let isel = array("indexer", "sel_file", "sel_sha256")?;
    let rsel = array("router", "sel_file", "sel_sha256")?;
    let calls = 1 + tokens.len() - decode_from;
    let mut out: Vec<memra_engine::hy4f_route::Forced> =
        (0..calls).map(|_| Default::default()).collect();
    for (sec, flat) in [("indexer", &isel), ("router", &rsel)] {
        for r in v[sec]["records"].as_array().ok_or("no records")? {
            let g = |k: &str| r[k].as_u64().unwrap_or(0) as usize;
            let (c, l, q, k, o) = (g("call"), g("layer"), g("queries"), g("k"), g("sel_offset"));
            let slice = flat
                .get(o..o + q * k)
                .ok_or("a record past its array")?
                .to_vec();
            let call = out.get_mut(c).ok_or("a call past the request")?;
            if sec == "indexer" {
                call.indexer.insert(l, slice);
            } else {
                call.router.insert(l, slice);
            }
        }
    }
    Ok(out)
}

fn gate_route(args: &[String]) -> Result<(), String> {
    let [ckpt, request, forcing, oracle, out, ..] = args else {
        return Err("gate-route CKPT REQUEST FORCING_TAPE ORACLE_TAPE OUT".into());
    };
    let red_name = flag(args, "--red").unwrap_or("");
    let red = DraftRed::parse(red_name).ok_or(format!("unknown red arm {red_name:?}"))?;
    let (ckpt, oracle, out) = (Path::new(ckpt), Path::new(oracle), Path::new(out));
    let (config, plan) = plan_of(ckpt)?;
    let hy = config
        .hy_v4_flash
        .as_ref()
        .ok_or("not a Hy4-flash config")?;
    let req = read_json(Path::new(request))?;
    let tokens = u32s(&req["token_ids"]);
    let decode_from = req["decode_from"].as_u64().ok_or("decode_from")? as usize;
    let forced = read_forcing(Path::new(forcing), &tokens, decode_from)?;
    let ot = read_json(&oracle.join("tape.json"))?;
    if u32s(&ot["token_ids"]) != tokens || ot["forced_drafts"] != json!(false) {
        return Err("the oracle tape is not the f64 oracle's of this request".into());
    }
    let anchors: Vec<usize> = ot["anchors"]
        .as_array()
        .ok_or("anchors")?
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let k = ot["k"].as_u64().ok_or("k")? as usize;
    let drafts: Vec<Vec<u32>> = ot["drafts"]
        .as_array()
        .ok_or("drafts")?
        .iter()
        .map(u32s)
        .collect();
    if mem_available().unwrap_or(0) < 95_000_000_000 + (16usize << 30) {
        return Err(
            "MemAvailable cannot hold the resident weights, the drafter and headroom".into(),
        );
    }
    let ck = Ckpt::open(ckpt)?;
    let dev = Dev::new(0)?;
    let t0 = Instant::now();
    let mut route = Route::load(dev, &ck, &plan, hy, 8, plan.layers.len())?;
    route.load_drafter(&ck)?;
    drop(ck);
    {
        let d = route.drafter_mut().unwrap();
        d.red = red;
        d.record = Some(HashMap::new());
    }
    eprintln!(
        "hy4f-spec gate-route: resident in {:.1}s",
        t0.elapsed().as_secs_f64()
    );
    let h = plan.hidden_size as usize;
    let vocab = plan.vocab_size as usize;
    let mut sess = route.new_session(tokens.len() + k + 2)?;
    let mut calls = vec![(0, decode_from)];
    calls.extend((decode_from..tokens.len()).map(|q| (q, q + 1)));
    let mut hidden = Vec::with_capacity(tokens.len() * h);
    for (ci, &(s0, s1)) in calls.iter().enumerate() {
        route.call_in(&mut sess, &tokens[s0..s1], false, None, Some(&forced[ci]))?;
        hidden.extend(route.last_hidden(s1 - s0)?);
    }
    let record = route.drafter_mut().unwrap().record.take().unwrap();
    let mut rounds: Vec<(Vec<u32>, Vec<f32>)> = vec![Default::default(); anchors.len()];
    let tr = Instant::now();
    for n in (0..anchors.len()).rev() {
        let p = anchors[n];
        let hist: HashMap<usize, Vec<i32>> = hy
            .dfly
            .target_index_layers
            .iter()
            .map(|l| {
                record
                    .get(&(*l as usize, p))
                    .cloned()
                    .map(|v| (*l as usize, v))
                    .ok_or(format!("no history of layer {l} at {p}"))
            })
            .collect::<Result<_, _>>()?;
        let r = route.draft_over_session(
            &sess,
            &hidden[p * h..(p + 1) * h],
            &hist,
            p,
            tokens[p + 1],
            k,
            Some(&drafts[n]),
        )?;
        rounds[n] = (r.ids, r.logits);
    }
    let mut logits: Vec<u8> = Vec::with_capacity(anchors.len() * k * vocab * 4);
    for (_, l) in &rounds {
        for v in l {
            logits.extend_from_slice(&v.to_le_bytes());
        }
    }
    let own: Vec<Vec<u32>> = rounds.into_iter().map(|(i, _)| i).collect();
    let mut accepted = Vec::new();
    for (n, &p) in anchors.iter().enumerate() {
        let mut m = 0;
        while m < k && p + 2 + m < tokens.len() && own[n][m] == tokens[p + 2 + m] {
            m += 1;
        }
        accepted.push(m);
    }
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    std::fs::write(out.join("logits.f32"), &logits).map_err(|e| e.to_string())?;
    let tape = json!({
        "format": "memra-hy4f-dfly-tape-v1",
        "engine": "memra-engine-hy4f_route-dfly-gpu-composite",
        "numeric_class": memra_engine::hy4f_route::NUMERIC_CLASS,
        "backend_dtype": "bf16x2-route-composite-forced",
        "red_arm": if red == DraftRed::None { Value::Null } else { json!(red_name) },
        "request": req["name"],
        "token_ids": tokens,
        "k": k,
        "vocab": vocab,
        "anchors": anchors,
        "drafts": own,
        "accepted_teacher_forced": accepted,
        "logits": {"file": "logits.f32", "sha256": sha_hex(&logits), "shape": [anchors.len(), k, vocab]},
        "forced_drafts": true,
        "forced_selections": true,
        "seconds": tr.elapsed().as_secs_f64(),
    });
    std::fs::write(
        out.join("tape.json"),
        serde_json::to_string_pretty(&tape).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    println!(
        "{}",
        json!({"request": tape["request"], "anchors": anchors.len(), "k": k, "red": tape["red_arm"],
               "logits_sha256": tape["logits"]["sha256"], "seconds": tape["seconds"]})
    );
    Ok(())
}
