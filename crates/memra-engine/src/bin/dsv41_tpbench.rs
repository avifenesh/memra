//! DeepSeek-V4.1-Flash on N ranks with the trunk resident: the benchmark driver of the 4-card
//! route (`dsv41_gpu::decode` on a `tp::Group`). One process loads every rank's share of the
//! layers and the head once, then runs the workload cells in rounds, each round in the reverse
//! order of the last (A B C, C B A, ...), and writes one JSON row per cell run.
//!
//! Cells (each prompt is a raw little-endian i32 token file):
//!   prefill   one sequence, the whole prompt in one forward, then the head on its last row:
//!             TTFT (forward start to the scored first token) and prefill tok/s
//!   decode-cN N sequences: their prompts in one forward (each alone through each block), then
//!             `steps` greedy decode forwards, one token per sequence per forward; per-step wall
//!             times, TPOT/ITL percentiles, decode tok/s (N x steps over the decode time) and
//!             end-to-end output tok/s (over prefill + decode). The tokens are taken greedily and
//!             fed back whatever they are (an end-of-sequence id does not stop the cell: the
//!             baselines run with ignore_eos), and saved per cell and round for identity checks.
//!
//! usage: dsv41-tpbench CKPT_DIR --token-map FILE --devices 0,1,2,3 --out DIR
//!                    [--prefill FILE,FILE,...] [--decode FILE[,FILE...] --decode-steps N]
//!                    [--conc 8,32 --conc-dir DIR --conc-steps N] [--rounds R]
//!                    [--resident N] [--layers N] [--chunk N] [--wide-rows A,B,C,D,E] [--plan]
//!   MEMRA_FULL_PREC=1      required (the route's one numeric program)
//!   --resident N           layers kept on the cards from the start (default: all); the rest
//!                          stream from the checkpoint mmap per forward (the 2-card rehearsal)
//!   --plan                 print the memory plan and the cells, load nothing
//!   --timed                drain and time every collective (`Group::timed`): a breakdown run, the
//!                          rows' `comm.timed_s`; never the headline numbers
//!
//! Every row carries the per-device pool high-water mark of the cell (`peak_used`), next to the
//! estimate the load reserved for (`dsv41_gpu::decode::prefill_workspace_bytes`).

use memra_engine::dsv41_gpu::{
    self as v, Geo,
    decode::{self as dec, Call, Runner, Seq},
};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn main() {
    if let Err(e) = run() {
        eprintln!("dsv41-tpbench: {e}");
        std::process::exit(1);
    }
}

fn read_ids(f: &str) -> Result<Vec<u32>, String> {
    let raw = std::fs::read(f).map_err(|e| format!("{f}: {e}"))?;
    if raw.len() % 4 != 0 || raw.is_empty() {
        return Err(format!("{f}: not a non-empty i32 array"));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
        .collect())
}

fn sha256_hex(b: &[u8]) -> String {
    use sha2::Digest;
    let d = sha2::Sha256::digest(b);
    d.iter().map(|x| format!("{x:02x}")).collect()
}

/// The process's minor and major page faults so far.
fn page_faults() -> (i64, i64) {
    // SAFETY: getrusage fills the struct it is given
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) };
    (u.ru_minflt, u.ru_majflt)
}

fn now_unix() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn pct(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let i = ((p / 100.0) * (s.len() - 1) as f64).round() as usize;
    s[i.min(s.len() - 1)]
}

#[derive(Clone)]
enum Cell {
    Prefill {
        name: String,
        file: String,
        ids: Vec<u32>,
    },
    Decode {
        name: String,
        files: Vec<String>,
        prompts: Vec<Vec<u32>>,
        steps: usize,
    },
}

impl Cell {
    fn name(&self) -> &str {
        match self {
            Cell::Prefill { name, .. } | Cell::Decode { name, .. } => name,
        }
    }
    /// The positions the longest sequence of the cell reaches.
    fn max_len(&self) -> usize {
        match self {
            Cell::Prefill { ids, .. } => ids.len() + 1,
            Cell::Decode { prompts, steps, .. } => {
                prompts.iter().map(|p| p.len()).max().unwrap_or(0) + steps + 1
            }
        }
    }
    fn longest_prompt(&self) -> usize {
        match self {
            Cell::Prefill { ids, .. } => ids.len(),
            Cell::Decode { prompts, .. } => prompts.iter().map(|p| p.len()).max().unwrap_or(0),
        }
    }
}

fn gib(b: usize) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let ckpt = PathBuf::from(
        a.get(1)
            .ok_or("usage: dsv41-tpbench CKPT_DIR --out DIR ...")?,
    );
    let flag = |k: &str| {
        a.iter()
            .position(|x| x == k)
            .and_then(|i| a.get(i + 1))
            .cloned()
    };
    let has = |k: &str| a.iter().any(|x| x == k);
    let num = |k: &str| -> Result<Option<usize>, String> {
        flag(k)
            .map(|x| x.parse().map_err(|_| format!("{k} {x:?}: not a count")))
            .transpose()
    };
    if std::env::var("MEMRA_FULL_PREC").as_deref() != Ok("1") {
        return Err("set MEMRA_FULL_PREC=1: the deepseek_v41 route runs only the checkpoint's own numeric program".into());
    }
    let plan_only = has("--plan");
    let out = PathBuf::from(flag("--out").ok_or("--out DIR")?);
    let tmap = flag("--token-map")
        .or_else(|| std::env::var("MEMRA_DSV41_TOKEN_MAP").ok())
        .ok_or("the Engram hash needs the compressed token map: --token-map FILE")?;
    let devices: Vec<usize> = flag("--devices")
        .unwrap_or_else(|| "0".into())
        .split(',')
        .map(|d| d.trim().parse().map_err(|_| format!("--devices {d:?}")))
        .collect::<Result<_, _>>()?;
    let chunk = num("--chunk")?.unwrap_or(8192);
    let rounds = num("--rounds")?.unwrap_or(3).max(1);

    let mut cells: Vec<Cell> = Vec::new();
    if let Some(list) = flag("--prefill") {
        for f in list.split(',').filter(|x| !x.is_empty()) {
            let ids = read_ids(f)?;
            cells.push(Cell::Prefill {
                name: format!("prefill-{}", ids.len()),
                file: f.to_string(),
                ids,
            });
        }
    }
    if let Some(list) = flag("--decode") {
        let steps = num("--decode-steps")?.unwrap_or(256);
        let files: Vec<String> = list
            .split(',')
            .filter(|x| !x.is_empty())
            .map(String::from)
            .collect();
        let prompts = files
            .iter()
            .map(|f| read_ids(f))
            .collect::<Result<Vec<_>, _>>()?;
        cells.push(Cell::Decode {
            name: format!("decode-c{}", files.len()),
            files,
            prompts,
            steps,
        });
    }
    if let Some(list) = flag("--conc") {
        let dir = flag("--conc-dir").ok_or("--conc needs --conc-dir DIR of prompt files")?;
        let steps = num("--conc-steps")?.unwrap_or(64);
        let mut files: Vec<String> = std::fs::read_dir(&dir)
            .map_err(|e| format!("{dir}: {e}"))?
            .filter_map(|e| e.ok())
            .map(|e| e.path().to_string_lossy().to_string())
            .filter(|p| p.ends_with(".i32"))
            .collect();
        files.sort();
        for c in list.split(',').filter(|x| !x.is_empty()) {
            let c: usize = c.parse().map_err(|_| format!("--conc {c:?}"))?;
            if c > files.len() {
                return Err(format!(
                    "--conc {c}: {dir} has {} prompt files",
                    files.len()
                ));
            }
            let fs: Vec<String> = files[..c].to_vec();
            let prompts = fs
                .iter()
                .map(|f| read_ids(f))
                .collect::<Result<Vec<_>, _>>()?;
            cells.push(Cell::Decode {
                name: format!("decode-c{c}"),
                files: fs,
                prompts,
                steps,
            });
        }
    }
    if cells.is_empty() {
        return Err("no cells: give --prefill, --decode and/or --conc".into());
    }

    let cfg_path = ckpt.join("config.json");
    let mc =
        memra_gguf::config::ModelConfig::from_config_json(&cfg_path).map_err(|e| e.to_string())?;
    let g = Geo::from_config(&mc)?;
    let trunk =
        g.compress_ratios.len() - mc.dsv41.as_ref().unwrap().num_nextn_predict_layers as usize;
    let n_layers = num("--layers")?.unwrap_or(trunk).min(trunk);
    let keep = num("--resident")?.unwrap_or(n_layers).min(n_layers);
    let n = devices.len();
    let eos: u32 = {
        let txt = std::fs::read_to_string(&cfg_path).map_err(|e| e.to_string())?;
        let j: serde_json::Value = serde_json::from_str(&txt).map_err(|e| e.to_string())?;
        j["eos_token_id"]
            .as_u64()
            .ok_or("config.json has no eos_token_id")? as u32
    };
    let max_pos = cells
        .iter()
        .map(Cell::max_len)
        .max()
        .unwrap_or(0)
        .saturating_add(16)
        .max(1024);

    // the memory plan: the largest of (one prefill's workspace + its state) and (a decode cell's
    // prompts' streams held across the pass + one prompt's workspace + every sequence's state)
    let mut reserve = 0usize;
    let mut plan_rows = Vec::new();
    for c in &cells {
        let (ws, state) = match c {
            Cell::Prefill { ids, .. } => (
                dec::prefill_workspace_bytes(&g, n, ids.len(), chunk),
                dec::seq_state_bytes(&g, n_layers, ids.len() + 16),
            ),
            Cell::Decode { prompts, steps, .. } => {
                let streams: usize = prompts.iter().map(|p| p.len() * g.hc * g.dim * 2).sum();
                let state: usize = prompts
                    .iter()
                    .map(|p| dec::seq_state_bytes(&g, n_layers, p.len() + steps + 16))
                    .sum();
                (
                    streams + dec::prefill_workspace_bytes(&g, n, c.longest_prompt(), chunk),
                    state,
                )
            }
        };
        reserve = reserve.max(ws + state);
        plan_rows.push((c.name().to_string(), ws, state));
    }
    let state_per_tok = dec::seq_state_bytes(&g, n_layers, 1 << 20) as f64 / (1u64 << 20) as f64;
    eprintln!(
        "dsv41-tpbench: {} cells x {rounds} rounds on {n} rank(s) {devices:?}, {keep}/{n_layers} layers resident, chunk {chunk}, max_pos {max_pos}",
        cells.len()
    );
    for (nm, ws, st) in &plan_rows {
        eprintln!(
            "  plan {nm}: workspace {:.2} GiB + state {:.2} GiB a rank",
            gib(*ws),
            gib(*st)
        );
    }
    eprintln!(
        "  carried state: {state_per_tok:.0} B a position a rank (+{:.1} MiB of window rings a sequence); reserve {:.2} GiB a device-rank",
        (n_layers * g.window * g.head_dim * 2) as f64 / (1u64 << 20) as f64,
        gib(reserve)
    );
    if plan_only {
        return Ok(());
    }
    std::fs::create_dir_all(out.join("gen")).map_err(|e| format!("{}: {e}", out.display()))?;

    let t0 = Instant::now();
    let stm = memra_gguf::safetensors::StModel::open(&ckpt).map_err(|e| e.to_string())?;
    let ec =
        v::engram::EngramConsts::derive(&g, v::engram::token_map_from_file(Path::new(&tmap))?)?;
    let mut rn = Runner::new(
        &stm,
        g.clone(),
        Some(ec),
        &devices,
        max_pos,
        n_layers,
        keep,
        chunk,
    )?;
    // ranks sharing a device share its memory: the reserve is per rank
    let per_dev = rn
        .device_ranks()
        .iter()
        .map(|x| x.1.len())
        .max()
        .unwrap_or(1);
    let res = rn.load_resident(keep, reserve * per_dev)?;
    // --timed: drain and time every collective (a breakdown run; not for the headline numbers)
    rn.grp.timed = has("--timed");
    // --wide-rows F8,BF16,FFMA,MIX,TILES: the most rows of a call that keep each family's
    // small-row body in groups of 8 (the measurement behind the defaults; bytes do not move)
    if let Some(w) = flag("--wide-rows") {
        let v: Vec<usize> = w
            .split(',')
            .map(|x| {
                x.parse()
                    .map_err(|_| format!("--wide-rows {w:?}: five counts"))
            })
            .collect::<Result<_, _>>()?;
        if v.len() != 5 {
            return Err(format!("--wide-rows {w:?}: five counts"));
        }
        memra_engine::dsv41_gpu::set_wide_rows(v[0], v[1], v[2], v[3], v[4]);
    }
    let bin = std::fs::read(&a[0])
        .map(|b| sha256_hex(&b))
        .unwrap_or_default();
    // Engram's pinned host staging (`engram::Stage`), one per rank: the host memory this run pins
    let pinned = {
        let cap = (chunk.min(max_pos) * g.engram_cols()).max(4096);
        cap * g.engram_head_dim + cap * g.engram_head_dim / 32
    };
    let mut envj = serde_json::json!({
        "binary": a[0], "binary_sha256": bin, "argv": a, "devices": devices, "ranks": n,
        "transport": rn.grp.transport.name(), "issue": rn.grp.issue_name(),
        "layers": n_layers, "resident": keep, "chunk": chunk, "max_pos": max_pos,
        "rounds": rounds, "reserve_per_rank": reserve, "load_s": res.load_s,
        "start_unix": now_unix(),
        "devices_budget": res.devices.iter().map(|b| serde_json::json!({
            "device": b.device, "ranks": b.ranks, "total": b.total, "free_before": b.free_before,
            "resident": b.resident, "free_after": b.free_after,
        })).collect::<Vec<_>>(),
        "state_bytes_per_position_per_rank": state_per_tok,
        "timed_collectives": has("--timed"),
        "pinned_engram_staging_per_rank": pinned,
        "cells": cells.iter().map(|c| match c {
            Cell::Prefill { name, file, ids } => serde_json::json!({"name": name, "kind": "prefill", "file": file, "prompt_len": ids.len(), "prompt_sha256": sha256_hex(&v::i32_bytes(&ids.iter().map(|&x| x as i32).collect::<Vec<_>>()))}),
            Cell::Decode { name, files, prompts, steps } => serde_json::json!({"name": name, "kind": "decode", "files": files, "prompt_lens": prompts.iter().map(|p| p.len()).collect::<Vec<_>>(), "steps": steps}),
        }).collect::<Vec<_>>(),
    });
    // the largest batch of sequences the free memory holds at each context, beside the reserve
    let free_min = res
        .devices
        .iter()
        .map(|b| b.free_after / b.ranks.len().max(1))
        .min()
        .unwrap_or(0);
    let mut cap_rows = Vec::new();
    for ctx in [4096usize, 32768, 131072, 262144] {
        let ws = dec::prefill_workspace_bytes(&g, n, ctx, chunk);
        let st = dec::seq_state_bytes(&g, n_layers, ctx);
        let seqs = free_min.saturating_sub(ws) / st.max(1);
        eprintln!(
            "  capacity at {ctx} positions: {seqs} sequences a rank after one prefill's {:.2} GiB workspace ({:.3} GiB state each, {:.2} GiB free a rank)",
            gib(ws),
            gib(st),
            gib(free_min)
        );
        cap_rows.push(serde_json::json!({"positions": ctx, "workspace": ws, "state_per_seq": st, "max_seqs": seqs}));
    }
    envj["capacity"] = serde_json::Value::Array(cap_rows);
    envj["load_wall_s"] = serde_json::json!(t0.elapsed().as_secs_f64());
    std::fs::write(
        out.join("env.json"),
        serde_json::to_string_pretty(&envj).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| e.to_string())?;

    let mut rows = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out.join("cells.jsonl"))
        .map_err(|e| e.to_string())?;
    for round in 0..rounds {
        let order: Vec<usize> = if round % 2 == 0 {
            (0..cells.len()).collect()
        } else {
            (0..cells.len()).rev().collect()
        };
        for ci in order {
            let cell = cells[ci].clone();
            for e in &rn.grp.engines {
                e.pool_high_water_reset();
            }
            let (c0, s0) = (rn.grp.stats, rn.stats.clone());
            let f0 = page_faults();
            let w0 = v::tp::READBACK_WAIT_NS.load(std::sync::atomic::Ordering::Relaxed);
            let row = match &cell {
                Cell::Prefill { name, ids, .. } => prefill_cell(&mut rn, name, ids, round)?,
                Cell::Decode {
                    name,
                    prompts,
                    steps,
                    ..
                } => decode_cell(&mut rn, name, prompts, *steps, round, &out, eos)?,
            };
            let mut row = row;
            let peaks: Vec<usize> = rn
                .grp
                .engines
                .iter()
                .map(|e| e.pool_high_water_reset().1)
                .collect();
            row["peak_used"] = serde_json::json!(peaks);
            let c1 = rn.grp.stats;
            row["comm"] = serde_json::json!({
                "collectives": c1.collectives - c0.collectives, "copies": c1.copies - c0.copies,
                "bytes": c1.bytes - c0.bytes, "timed_s": c1.secs - c0.secs, "timed": rn.grp.timed,
            });
            row["forwards"] = serde_json::json!(rn.stats.forwards - s0.forwards);
            // the process's page faults over the cell: the Engram tables are mapped from the
            // checkpoint, and a row whose page is not mapped costs microseconds, so pages lost
            // to memory pressure show here as a count
            let f1 = page_faults();
            row["page_faults"] = serde_json::json!({"minor": f1.0 - f0.0, "major": f1.1 - f0.1});
            // host time blocked on read-backs, per rank (the rest of a step is issue or other host work)
            let w1 = v::tp::READBACK_WAIT_NS.load(std::sync::atomic::Ordering::Relaxed);
            row["readback_wait_s_per_rank"] =
                serde_json::json!((w1 - w0) as f64 * 1e-9 / rn.grp.n() as f64);
            row["layers_streamed"] = serde_json::json!(rn.stats.layers_loaded - s0.layers_loaded);
            row["stream_load_s"] = serde_json::json!(rn.stats.load_s - s0.load_s);
            row["end_unix"] = serde_json::json!(now_unix());
            let line = serde_json::to_string(&row).map_err(|e| e.to_string())?;
            writeln!(rows, "{line}").map_err(|e| e.to_string())?;
            rows.flush().map_err(|e| e.to_string())?;
            eprintln!("dsv41-tpbench: {line}");
        }
    }
    eprintln!("dsv41-tpbench: done in {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}

fn prefill_cell(
    rn: &mut Runner,
    name: &str,
    ids: &[u32],
    round: usize,
) -> Result<serde_json::Value, String> {
    let start = now_unix();
    let t = Instant::now();
    let mut seqs: Vec<Seq> = vec![rn.seq_with_capacity(ids.len() + 16)];
    let outs = rn.forward(
        &mut seqs,
        &[Call {
            seq: 0,
            tokens: ids.to_vec(),
        }],
    )?;
    let t_fwd = t.elapsed().as_secs_f64();
    let first = rn.argmax_last(&outs[0])?;
    let ttft = t.elapsed().as_secs_f64();
    drop(outs);
    drop(seqs);
    Ok(serde_json::json!({
        "cell": name, "kind": "prefill", "round": round, "start_unix": start,
        "prompt_len": ids.len(), "ttft_s": ttft, "forward_s": t_fwd,
        "prefill_tok_s": ids.len() as f64 / ttft, "argmax": first,
    }))
}

fn decode_cell(
    rn: &mut Runner,
    name: &str,
    prompts: &[Vec<u32>],
    steps: usize,
    round: usize,
    out: &Path,
    eos: u32,
) -> Result<serde_json::Value, String> {
    let c = prompts.len();
    let start = now_unix();
    let t = Instant::now();
    let mut seqs: Vec<Seq> = prompts
        .iter()
        .map(|p| rn.seq_with_capacity(p.len() + steps + 16))
        .collect();
    let calls: Vec<Call> = prompts
        .iter()
        .enumerate()
        .map(|(i, p)| Call {
            seq: i,
            tokens: p.clone(),
        })
        .collect();
    let outs = rn.forward(&mut seqs, &calls)?;
    let mut next = Vec::with_capacity(c);
    for o in &outs {
        next.push(rn.argmax_last(o)?);
    }
    drop(outs);
    let ttft = t.elapsed().as_secs_f64();
    let mut gens: Vec<Vec<u32>> = next.iter().map(|&x| vec![x]).collect();
    let mut step_s = Vec::with_capacity(steps);
    let td = Instant::now();
    for _ in 0..steps {
        let ts = Instant::now();
        let calls: Vec<Call> = (0..c)
            .map(|i| Call {
                seq: i,
                tokens: vec![*gens[i].last().unwrap()],
            })
            .collect();
        let outs = rn.forward(&mut seqs, &calls)?;
        for (i, o) in outs.iter().enumerate() {
            gens[i].push(rn.argmax_last(o)?);
        }
        step_s.push(ts.elapsed().as_secs_f64());
    }
    let decode_s = td.elapsed().as_secs_f64();
    let total = t.elapsed().as_secs_f64();
    let eos_hit = gens.iter().filter(|gs| gs.contains(&eos)).count();
    let mut digest = Vec::new();
    for (i, gs) in gens.iter().enumerate() {
        let b = v::i32_bytes(&gs.iter().map(|&x| x as i32).collect::<Vec<_>>());
        std::fs::write(
            out.join("gen").join(format!("{name}-r{round}-s{i}.i32")),
            &b,
        )
        .map_err(|e| e.to_string())?;
        digest.push(sha256_hex(&b));
    }
    let mean = step_s.iter().sum::<f64>() / step_s.len().max(1) as f64;
    Ok(serde_json::json!({
        "cell": name, "kind": "decode", "round": round, "start_unix": start, "c": c,
        "prompt_lens": prompts.iter().map(|p| p.len()).collect::<Vec<_>>(),
        "steps": steps, "ttft_s": ttft, "decode_s": decode_s, "total_s": total,
        "step_s": step_s, "step_mean_s": mean,
        "tpot_p50_s": pct(&step_s, 50.0), "tpot_p95_s": pct(&step_s, 95.0), "tpot_p99_s": pct(&step_s, 99.0),
        "decode_tok_s": (c * steps) as f64 / decode_s,
        "single_stream_tok_s": 1.0 / mean,
        "e2e_output_tok_s": (c * (steps + 1)) as f64 / total,
        "seqs_past_eos": eos_hit, "gen_sha256": digest,
    }))
}
