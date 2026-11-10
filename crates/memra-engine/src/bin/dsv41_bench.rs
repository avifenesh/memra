//! DeepSeek-V4.1 single-card speed bench: per-layer prefill and decode times on the real
//! checkpoint, GEMM shapes on fixed random inputs, and the card's own compute ceilings.
//!
//! usage:
//!   dsv41-bench layer CKPT --ids FILE --token-map FILE --layers 2,20,21 [--prefix N] [--reps R]
//!                     [--decode D] [--chunk N] [--path group|onecard] [--nvtx] [--out FILE.json]
//!   dsv41-bench gemm [--reps R] [--out FILE.json]
//!   dsv41-bench ceil [--out FILE.json]
//!
//! `layer` runs the trunk's prefill of the first N prompt tokens (raw i32 ids) one layer at a
//! time, each layer's weights loaded from the checkpoint and freed as `dsv41-run` does, up to the
//! last listed layer. A listed layer's block (attention and MoE; Engram is timed apart) runs R
//! more times on the same input after the pass that carries forward, each run between two
//! synchronizes. Every run's output stream and pre-mix are hashed: a rerun must give the same
//! bytes, and two binaries' hashes say whether a change moved the numerics. `--decode D` then
//! runs D one-token steps from position N through each run of consecutive listed layers that
//! starts at a kv source (2; 20,21), on the CSA2 state and window rings that run's prefill left;
//! step i feeds prefill row N-1-i of the run's first input, and every step is hashed; the same rows
//! then run again as whole steps (positions N+D ..) with one wait per step. The layer
//! runs through `tp::Group` at N = 1 (the decode runner's path) unless `--path onecard` selects
//! `Ctx::block_forward`. `--nvtx` opens a range per op for an nsys profile (`perf/prof_attr.py`).
//!
//! `gemm` times the block-scaled GEMMs at the trunk's shapes (dense FP8 at M = 1, 64, 4096, 8192; the
//! grouped MXFP4 experts at decode and at a 4k and a 32k prefill's mean rows per expert) on inputs from a
//! fixed seed, with each output's sha256. `ceil` measures the per-SM instruction ceilings the
//! GEMMs and attention run on (`memra_v41_peak`): e4m3 and bf16 MMA throughput, the reference
//! per-32-block schedule (one MMA from zero, then the scale epilogue), and fp32 FMA.
//!
//! Every time is a wall time between synchronizes on one stream; nothing here is a published
//! number until it is part of an interleaved A/B with its telemetry.

use memra_engine::dsv41_gpu::{
    self as v, Buf, Ctx, Geo, LayerState, LayerW, Ops, QFmt, QW, Shared, Trace, ck, ffi, tp::Group,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let r = match a.get(1).map(|s| s.as_str()) {
        Some("layer") if a.len() > 2 => layer_bench(&a),
        Some("gemm") => gemm_bench(&a),
        Some("ceil") => ceil_bench(&a),
        Some("p2p") if a.len() > 2 => p2p_bench(&a),
        _ => Err("usage: dsv41-bench layer CKPT --ids F --token-map F --layers L,.. [--prefix N] [--reps R] [--decode D] [--chunk N] [--nvtx] [--out F] | gemm [--reps R] [--out F] | ceil [--out F]".into()),
    };
    match r {
        Ok(js) => {
            if let Some(o) = flag(&a, "--out")
                && let Err(e) = std::fs::write(&o, &js)
            {
                eprintln!("write {o}: {e}");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("dsv41-bench: {e}");
            std::process::exit(1);
        }
    }
}

/// `dsv41-bench p2p CKPT --devices 0,1[,2,3] [--reps R]`: how fast the cards move bytes to each
/// other, by the copy engine and by a kernel storing into the peer, alone and in the flow sets a
/// gather makes (`Group::p2p_probe`). No weights are loaded.
fn p2p_bench(a: &[String]) -> Result<String, String> {
    let ckpt = Path::new(&a[2]);
    let mc = memra_gguf::config::ModelConfig::from_config_json(&ckpt.join("config.json"))
        .map_err(|e| e.to_string())?;
    let g = Geo::from_config(&mc)?;
    let devices: Vec<usize> = flag(a, "--devices")
        .ok_or("p2p: --devices")?
        .split(',')
        .map(|x| x.parse::<usize>().map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let reps = flag(a, "--reps")
        .map(|x| x.parse::<usize>().map_err(|e| e.to_string()))
        .transpose()?
        .unwrap_or(5)
        .max(1);
    let mut grp = Group::new(&devices, &g, 1024)?;
    let lines = grp.p2p_probe(reps)?;
    for l in &lines {
        println!("{l}");
    }
    Ok(format!(
        "{{\"devices\": {devices:?}, \"transport\": {:?}, \"lines\": {:?}}}\n",
        grp.transport.name(),
        lines
    ))
}

fn flag(a: &[String], k: &str) -> Option<String> {
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1))
        .cloned()
}

fn sha(b: &[u8]) -> String {
    let d = Sha256::digest(b);
    d.iter().take(12).map(|x| format!("{x:02x}")).collect()
}

/// A profile range name that lives for the process (NVTX takes a C string).
fn cname(s: String) -> &'static std::ffi::CStr {
    Box::leak(std::ffi::CString::new(s).unwrap().into_boxed_c_str())
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

fn fmt_list(v: &[f64]) -> String {
    let p: Vec<String> = v.iter().map(|x| format!("{x:.3}")).collect();
    format!("[{}]", p.join(", "))
}

fn clone_buf(ops: &Ops, b: &Buf) -> Result<Buf, String> {
    let o = ops.buf(b.len)?;
    v::copy_d2d(ops, b, 0, &o, 0, b.len)?;
    Ok(o)
}

fn clone_opt(ops: &Ops, b: &Option<Buf>) -> Result<Option<Buf>, String> {
    b.as_ref().map(|x| clone_buf(ops, x)).transpose()
}

fn clone_shared(ops: &Ops, s: &Shared) -> Result<Shared, String> {
    Ok(Shared {
        compress_kv: clone_opt(ops, &s.compress_kv)?,
        n_c: s.n_c,
        index_k: clone_opt(ops, &s.index_k)?,
        topk: clone_opt(ops, &s.topk)?,
        topk_w: s.topk_w,
        cand: clone_opt(ops, &s.cand)?,
        cand_ld: s.cand_ld,
        cand_row0: s.cand_row0,
        compress_cap: s.compress_cap,
        index_k_cap: s.index_k_cap,
    })
}

/// Rows `[r0, r0 + n)` of `b` (rows of `rb` bytes) as a new buffer.
fn rows_of(ops: &Ops, b: &Buf, r0: usize, n: usize, rb: usize) -> Result<Buf, String> {
    let o = ops.buf(n * rb)?;
    v::copy_d2d(ops, b, r0 * rb, &o, 0, n * rb)?;
    Ok(o)
}

/// The layer runtime under test: the N-rank group at N = 1 (the path the decode runner and the
/// multi-card route take) or the one-card `Ctx::block_forward`. One value a process.
#[allow(clippy::large_enum_variant)]
enum Rt {
    One(Ctx, memra_engine::Engine),
    Grp(Group),
}

impl Rt {
    fn cx(&self) -> &Ctx {
        match self {
            Rt::One(c, _) => c,
            Rt::Grp(g) => &g.cx[0],
        }
    }
    fn cxm(&mut self) -> &mut Ctx {
        match self {
            Rt::One(c, _) => c,
            Rt::Grp(g) => &mut g.cx[0],
        }
    }
    fn load(
        &self,
        stm: &memra_gguf::safetensors::StModel,
        g: &Geo,
        li: usize,
    ) -> Result<Vec<LayerW>, String> {
        match self {
            Rt::One(c, _) => Ok(vec![LayerW::load(&c.ops, stm, g, li, false)?]),
            Rt::Grp(grp) => grp.load_layer(stm, g, li, false),
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn fwd(
        &mut self,
        lw: &[LayerW],
        h: &Buf,
        pre: &Buf,
        sp: usize,
        s: usize,
        sh: &mut Shared,
        ls: &mut LayerState,
        tr: &mut Trace,
    ) -> Result<(Buf, Buf), String> {
        match self {
            Rt::One(c, _) => c.block_forward(&lw[0], h, pre, sp, s, sh, ls, tr),
            Rt::Grp(grp) => {
                let (mut a, mut b) = grp.block_forward(
                    lw,
                    std::slice::from_ref(h),
                    std::slice::from_ref(pre),
                    sp,
                    s,
                    std::slice::from_mut(sh),
                    std::slice::from_mut(ls),
                    std::slice::from_mut(tr),
                )?;
                Ok((a.remove(0), b.remove(0)))
            }
        }
    }
}

/// One decode run: consecutive listed layers starting at a kv source, with what their prefill left.
struct DecRun {
    layers: Vec<usize>,
    lw: Vec<Vec<LayerW>>,
    ls: Vec<LayerState>,
    sh: Option<Shared>,
    /// the first layer's last `decode` input rows (stream and pre-mix), newest first
    h_rows: Vec<Buf>,
    p_rows: Vec<Buf>,
}

fn layer_bench(a: &[String]) -> Result<String, String> {
    let ckpt = PathBuf::from(&a[2]);
    let ids_f = flag(a, "--ids").ok_or("--ids FILE (raw i32 prompt ids)")?;
    let tmap = flag(a, "--token-map").ok_or("--token-map FILE")?;
    let mut layers: Vec<usize> = flag(a, "--layers")
        .ok_or("--layers L,..")?
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect();
    layers.sort_unstable();
    layers.dedup();
    let reps: usize = flag(a, "--reps").and_then(|x| x.parse().ok()).unwrap_or(3);
    let decode: usize = flag(a, "--decode")
        .and_then(|x| x.parse().ok())
        .unwrap_or(0);
    let chunk: usize = flag(a, "--chunk")
        .and_then(|x| x.parse().ok())
        .unwrap_or(8192);
    let nvtx = a.iter().any(|x| x == "--nvtx");
    let raw = std::fs::read(&ids_f).map_err(|e| format!("{ids_f}: {e}"))?;
    let mut ids: Vec<u32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
        .collect();
    if let Some(n) = flag(a, "--prefix").and_then(|x| x.parse::<usize>().ok()) {
        if n > ids.len() {
            return Err(format!("--prefix {n}: the prompt has {} tokens", ids.len()));
        }
        ids.truncate(n);
    }
    let s = ids.len();
    if decode >= s {
        return Err("--decode must be below the prompt length".into());
    }
    let mc = memra_gguf::config::ModelConfig::from_config_json(&ckpt.join("config.json"))
        .map_err(|e| e.to_string())?;
    let g = Geo::from_config(&mc)?;
    let trunk =
        g.compress_ratios.len() - mc.dsv41.as_ref().unwrap().num_nextn_predict_layers as usize;
    let last = *layers.last().ok_or("--layers is empty")?;
    if last >= trunk {
        return Err(format!("layer {last}: the trunk has {trunk} layers"));
    }
    let stm = memra_gguf::safetensors::StModel::open(&ckpt).map_err(|e| e.to_string())?;
    let ec =
        v::engram::EngramConsts::derive(&g, v::engram::token_map_from_file(Path::new(&tmap))?)?;
    let (hc, d) = (g.hc, g.dim);

    // decode runs: maximal runs of consecutive listed layers that start at a kv source
    let mut runs: Vec<DecRun> = Vec::new();
    if decode > 0 {
        let mut i = 0;
        while i < layers.len() {
            let mut j = i;
            while j + 1 < layers.len() && layers[j + 1] == layers[j] + 1 {
                j += 1;
            }
            let run: Vec<usize> = layers[i..=j].to_vec();
            if !g.kv_sources.contains(&run[0]) {
                eprintln!("decode: run {run:?} does not start at a kv source; prefill only");
            } else if run.iter().any(|l| g.engram_layers.contains(l)) {
                eprintln!("decode: run {run:?} holds an Engram layer; prefill only");
            } else {
                runs.push(DecRun {
                    layers: run,
                    lw: vec![],
                    ls: vec![],
                    sh: None,
                    h_rows: vec![],
                    p_rows: vec![],
                });
            }
            i = j + 1;
        }
    }

    let onecard = flag(a, "--path").as_deref() == Some("onecard");
    let mut rt = if onecard {
        let e = memra_engine::Engine::new(0).map_err(|e| e.to_string())?;
        let c = Ctx::new(e.stream(), g.clone(), s + 2 * decode + 16)?;
        Rt::One(c, e)
    } else {
        Rt::Grp(Group::new(&[0], &g, s + 2 * decode + 16)?)
    };
    let dev = match &rt {
        Rt::One(_, e) => e.ctx().name().unwrap_or_default(),
        Rt::Grp(grp) => grp.engines[0].ctx().name().unwrap_or_default(),
    };
    rt.cxm().chunk = chunk;
    rt.cxm().nvtx = nvtx;
    let st = rt.cx().ops.st.clone();
    let path = if onecard { "onecard" } else { "group1" };
    eprintln!(
        "dsv41-bench: {s} tokens, layers {layers:?}, reps {reps}, decode {decode}, chunk {chunk}, path {path}, {dev}"
    );

    let (ei, eb) = stm.raw("embed.weight").ok_or("embed.weight missing")?;
    if ei.dtype != "BF16" || ei.shape != [g.vocab as u64, d as u64] {
        return Err(format!("embed.weight: {} {:?}", ei.dtype, ei.shape));
    }
    let mut hb = vec![0u8; s * hc * d * 2];
    for (t, &id) in ids.iter().enumerate() {
        let id = id as usize;
        if id >= g.vocab {
            return Err(format!("token {id} outside the vocabulary"));
        }
        let row = &eb[id * d * 2..(id + 1) * d * 2];
        for k in 0..hc {
            hb[(t * hc + k) * d * 2..(t * hc + k + 1) * d * 2].copy_from_slice(row);
        }
    }
    let mut pm = vec![0f32; s * hc];
    for t in 0..s {
        pm[t * hc] = 1.0;
    }
    let hashes = ec.hash_prefill(&ids)?;
    let engram_cols = |li: usize| -> Result<Vec<i64>, String> {
        let slot = ec
            .layer_ids
            .iter()
            .position(|&x| x == li)
            .ok_or("engram layer not in the constants")?;
        let (nl, cols) = (ec.layer_ids.len(), ec.cols());
        Ok((0..s)
            .flat_map(|t| hashes[(t * nl + slot) * cols..(t * nl + slot + 1) * cols].to_vec())
            .collect())
    };

    let mut h = Buf::from_bytes(&st, &hb)?;
    drop(hb);
    let mut pre = Buf::from_f32(&st, &pm)?;
    let mut sh = Shared::default();
    let mut js_layers = Vec::new();
    let t_all = Instant::now();
    for li in 0..=last {
        let tl = Instant::now();
        let lw = rt.load(&stm, &g, li)?;
        let load_s = tl.elapsed().as_secs_f64();
        let mut tr = Trace::default();
        let mut engram_ms = f64::NAN;
        if lw[0].engram.is_some() {
            let mine = engram_cols(li)?;
            rt.cx().ops.sync()?;
            let t0 = Instant::now();
            rt.cxm().engram(&stm, &lw[0], &h, &mine, s, &mut tr)?;
            rt.cx().ops.sync()?;
            engram_ms = t0.elapsed().as_secs_f64() * 1e3;
        }
        let listed = layers.contains(&li);
        let run_at = runs.iter().position(|r| r.layers.contains(&li));
        if let Some(ri) = run_at
            && runs[ri].layers[0] == li
        {
            // the run's decode inputs: the last `decode` prefill rows of its first layer's input
            let r = &mut runs[ri];
            for k in 0..decode {
                let row = s - 1 - k;
                r.h_rows
                    .push(rows_of(&rt.cx().ops, &h, row, 1, hc * d * 2)?);
                r.p_rows.push(rows_of(&rt.cx().ops, &pre, row, 1, hc * 4)?);
            }
        }
        let mut ls = LayerState::default();
        rt.cx().ops.sync()?;
        let t0 = Instant::now();
        let (h2, p2) = rt.fwd(&lw, &h, &pre, 0, s, &mut sh, &mut ls, &mut tr)?;
        rt.cx().ops.sync()?;
        let first_ms = t0.elapsed().as_secs_f64() * 1e3;
        if listed {
            let want = (sha(&h2.host(&st)?), sha(&p2.host(&st)?));
            let mut ms = Vec::with_capacity(reps);
            let mut same = 0;
            let rname = cname(format!("L{li}/prefill"));
            for _ in 0..reps {
                let mut tr = Trace::default();
                rt.cx().ops.sync()?;
                let t0 = Instant::now();
                rt.cx().nv(rname);
                let (hr, pr) = rt.fwd(&lw, &h, &pre, 0, s, &mut sh, &mut ls, &mut tr)?;
                rt.cx().ops.sync()?;
                rt.cx().nv_end();
                ms.push(t0.elapsed().as_secs_f64() * 1e3);
                if (sha(&hr.host(&st)?), sha(&pr.host(&st)?)) == want {
                    same += 1;
                }
            }
            let med = median(&ms);
            eprintln!(
                "layer {li}: prefill {s} tokens: first {first_ms:.3} ms, reps {} ms, median {med:.3} ms, {same}/{reps} reruns byte-identical, engram {engram_ms:.3} ms, out {} pre {}",
                fmt_list(&ms),
                want.0,
                want.1
            );
            js_layers.push(format!(
                "{{\"layer\": {li}, \"tokens\": {s}, \"first_ms\": {first_ms:.4}, \"ms\": {}, \"median_ms\": {med:.4}, \"reruns_identical\": {same}, \"engram_ms\": {}, \"out_sha\": \"{}\", \"pre_sha\": \"{}\"}}",
                fmt_list(&ms),
                if engram_ms.is_nan() { "null".to_string() } else { format!("{engram_ms:.4}") },
                want.0,
                want.1
            ));
        } else {
            eprintln!("layer {li}: load {load_s:.1} s, block {first_ms:.1} ms");
        }
        h = h2;
        pre = p2;
        if let Some(ri) = run_at {
            let r = &mut runs[ri];
            r.lw.push(lw);
            r.ls.push(ls);
            if *r.layers.last().unwrap() == li {
                r.sh = Some(clone_shared(&rt.cx().ops, &sh)?);
            }
        } else {
            drop(lw);
        }
    }
    let prefill_s = t_all.elapsed().as_secs_f64();
    drop(h);
    drop(pre);
    drop(sh);

    let mut js_dec = Vec::new();
    for r in runs.iter_mut() {
        let mut sh = r.sh.take().ok_or("decode run without its state")?;
        let mut per: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
        let mut step_ms = Vec::new();
        let mut shas = Vec::new();
        let names: Vec<&'static std::ffi::CStr> = r
            .layers
            .iter()
            .map(|li| cname(format!("L{li}/decode")))
            .collect();
        for k in 0..decode {
            let pos = s + k;
            let mut x = clone_buf(&rt.cx().ops, &r.h_rows[k])?;
            let mut p = clone_buf(&rt.cx().ops, &r.p_rows[k])?;
            let mut tot = 0.0;
            for (i, &li) in r.layers.iter().enumerate() {
                let mut tr = Trace::default();
                rt.cx().ops.sync()?;
                let t0 = Instant::now();
                rt.cx().nv(names[i]);
                let (x2, p2) = rt.fwd(&r.lw[i], &x, &p, pos, 1, &mut sh, &mut r.ls[i], &mut tr)?;
                rt.cx().ops.sync()?;
                rt.cx().nv_end();
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                per.entry(li).or_default().push(ms);
                tot += ms;
                x = x2;
                p = p2;
            }
            step_ms.push(tot);
            shas.push(format!("{}:{}", sha(&x.host(&st)?), sha(&p.host(&st)?)));
        }
        // then the same rows as whole steps with one wait per step (positions s + decode ..): what
        // a decode loop sees once a layer no longer waits on the host
        let mut whole_ms = Vec::new();
        let mut whole_shas = Vec::new();
        let wname = cname(format!(
            "L{}-{}/step",
            r.layers[0],
            r.layers.last().unwrap()
        ));
        for k in 0..decode {
            let pos = s + decode + k;
            let mut x = clone_buf(&rt.cx().ops, &r.h_rows[k])?;
            let mut p = clone_buf(&rt.cx().ops, &r.p_rows[k])?;
            rt.cx().ops.sync()?;
            let t0 = Instant::now();
            rt.cx().nv(wname);
            for i in 0..r.layers.len() {
                let mut tr = Trace::default();
                let (x2, p2) = rt.fwd(&r.lw[i], &x, &p, pos, 1, &mut sh, &mut r.ls[i], &mut tr)?;
                x = x2;
                p = p2;
            }
            rt.cx().ops.sync()?;
            rt.cx().nv_end();
            whole_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            whole_shas.push(format!("{}:{}", sha(&x.host(&st)?), sha(&p.host(&st)?)));
        }
        let wm = if whole_ms.len() > 1 {
            median(&whole_ms[1..])
        } else {
            median(&whole_ms)
        };
        eprintln!(
            "run {:?}: whole decode steps at {}..: {} ms, median (steps 2..) {wm:.3} ms",
            r.layers,
            s + decode,
            fmt_list(&whole_ms)
        );
        for (li, v) in &per {
            // the first step pays one-time allocations (first decode-shaped buffers); report both
            let warm = if v.len() > 1 { &v[1..] } else { &v[..] };
            eprintln!(
                "layer {li}: decode at {s}..{}: steps {} ms, median (steps 2..) {:.3} ms",
                s + decode,
                fmt_list(v),
                median(warm)
            );
        }
        let rows: Vec<String> = per
            .iter()
            .map(|(li, v)| {
                let warm = if v.len() > 1 { &v[1..] } else { &v[..] };
                format!(
                    "{{\"layer\": {li}, \"ms\": {}, \"median_ms_after_first\": {:.4}}}",
                    fmt_list(v),
                    median(warm)
                )
            })
            .collect();
        js_dec.push(format!(
            "{{\"run\": {:?}, \"from_pos\": {s}, \"steps\": {decode}, \"step_ms\": {}, \"per_layer\": [{}], \"step_sha\": {:?}, \"whole_step_ms\": {}, \"whole_step_median_ms_after_first\": {wm:.4}, \"whole_step_sha\": {:?}}}",
            r.layers,
            fmt_list(&step_ms),
            rows.join(", "),
            shas,
            fmt_list(&whole_ms),
            whole_shas
        ));
    }
    Ok(format!(
        "{{\"bench\": \"layer\", \"device\": {dev:?}, \"path\": \"{path}\", \"tokens\": {s}, \"chunk\": {chunk}, \"reps\": {reps}, \"nvtx\": {nvtx}, \"prefill_wall_s\": {prefill_s:.2},\n \"layers\": [\n  {}\n ],\n \"decode\": [\n  {}\n ]\n}}\n",
        js_layers.join(",\n  "),
        js_dec.join(",\n  ")
    ))
}

// ------------------------------------------------------------------------------------------------
// GEMM shapes on fixed random inputs

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// e4m3 bytes, never NaN (0x7F / 0xFF)
    fn e4m3(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| {
                let b = (self.next() & 0xFF) as u8;
                if b & 0x7F == 0x7F { b ^ 1 } else { b }
            })
            .collect()
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next() & 0xFF) as u8).collect()
    }
    /// ue8m0 codes in [lo, hi]
    fn e8m0(&mut self, n: usize, lo: u8, hi: u8) -> Vec<u8> {
        (0..n)
            .map(|_| lo + (self.next() % (hi - lo + 1) as u32) as u8)
            .collect()
    }
}

fn time_ms<F: FnMut() -> Result<(), String>>(
    ops: &Ops,
    reps: usize,
    mut f: F,
) -> Result<Vec<f64>, String> {
    f()?; // warm
    ops.sync()?;
    let mut v = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        f()?;
        ops.sync()?;
        v.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    Ok(v)
}

fn gemm_bench(a: &[String]) -> Result<String, String> {
    let reps: usize = flag(a, "--reps").and_then(|x| x.parse().ok()).unwrap_or(20);
    let e = memra_engine::Engine::new(0).map_err(|e| e.to_string())?;
    let st = e.stream();
    let dev = e.ctx().name().unwrap_or_default();
    let ops = Ops::new(st.clone());
    let mut out = Vec::new();
    // the trunk's dense FP8 linears: (name, n, k)
    let dense = [
        ("wq_a", 1280usize, 5120usize),
        ("wkv", 512, 5120),
        ("wq_b", 32768, 1280),
        ("idx_wq_b", 4096, 1280),
        ("wo_b", 5120, 8192),
        ("shared_w13", 2304, 5120),
        ("shared_w2", 5120, 2304),
    ];
    for &(name, n, k) in &dense {
        let mut rng = Lcg(0x5eed_d541 ^ ((n * 1_000_003 + k) as u64));
        let w = QW {
            w: Buf::from_bytes(&st, &rng.e4m3(n * k))?,
            s: Buf::from_bytes(&st, &rng.e8m0(n.div_ceil(32) * (k / 32), 110, 122))?,
            n,
            k,
            fmt: QFmt::Fp8Block32,
        };
        for m in [1usize, 64, 4096, 8192] {
            let mut rng = Lcg(rng.0 ^ (m as u64) << 20);
            let aq = Buf::from_bytes(&st, &rng.e4m3(m * k))?;
            let as_ = Buf::from_bytes(&st, &rng.e8m0(m * (k / 32), 118, 130))?;
            let c = ops.buf(m * n * 2)?;
            let ms = time_ms(&ops, reps, || {
                ops.gemm_q(aq.p(&st), as_.p(&st), m, &w, c.p(&st), n, false)
            })?;
            let hsh = sha(&c.host(&st)?);
            let med = median(&ms);
            let bytes = (n * k + m * k) as f64 + (m * n * 2) as f64;
            let flops = 2.0 * (m * n * k) as f64;
            eprintln!(
                "gemm fp8 {name} m {m} n {n} k {k}: median {:.4} ms ({:.0} GB/s, {:.1} TFLOPS) sha {hsh}",
                med,
                bytes / med / 1e6,
                flops / med / 1e9
            );
            out.push(format!(
                "{{\"op\": \"fp8\", \"name\": \"{name}\", \"m\": {m}, \"n\": {n}, \"k\": {k}, \"median_ms\": {med:.5}, \"ms\": {}, \"sha\": \"{hsh}\"}}",
                fmt_list(&ms)
            ));
        }
    }
    // grouped MXFP4 experts through the tile-list dispatch: w1/w3 (n 2304, k 5120) and w2 (n 5120,
    // k 2304), `ng` experts of `rows` rows, tiles as `Ctx::moe_bm` picks them for that slice
    for &(ng, rows) in &[
        (6usize, 1usize),
        (12, 1),
        (29, 1),
        (48, 1),
        (15, 2),
        (64, 64),
        (384, 64),
        (384, 512),
    ] {
        let bm = if rows <= v::SMALL_M {
            v::SMALL_M
        } else if rows >= 128 {
            128
        } else {
            64
        };
        for &(name, n, k) in &[("exp_w13", 2304usize, 5120usize), ("exp_w2", 5120, 2304)] {
            // every shape draws its own inputs, so adding a shape never changes another's bytes
            let mut rng = Lcg(0x5eed_d541 ^ ((ng * 1_000_003 + rows * 7919 + n) as u64));
            let mut wts = Vec::with_capacity(ng);
            let mut ew = Vec::with_capacity(ng * 64);
            let mut tiles = Vec::new();
            for z in 0..ng {
                let w = Buf::from_bytes(&st, &rng.bytes(n * k / 2))?;
                let s = Buf::from_bytes(&st, &rng.e8m0(n * (k / 32), 112, 124))?;
                for _ in 0..3 {
                    ew.extend_from_slice(&w.p(&st).addr().to_le_bytes());
                }
                for _ in 0..3 {
                    ew.extend_from_slice(&s.p(&st).addr().to_le_bytes());
                }
                for _ in 0..3 {
                    ew.extend_from_slice(&1.0f32.to_le_bytes());
                }
                ew.extend_from_slice(&0i32.to_le_bytes());
                for m0 in (0..rows).step_by(bm) {
                    for x in [(z * rows) as i32, rows as i32, m0 as i32, z as i32] {
                        tiles.extend_from_slice(&x.to_le_bytes());
                    }
                }
                wts.push((w, s));
            }
            let ntiles = tiles.len() / 16;
            let m = ng * rows;
            let ewd = Buf::from_bytes(&st, &ew)?;
            let tld = Buf::from_bytes(&st, &tiles)?;
            let ntd = Buf::from_i32(&st, &[ntiles as i32])?;
            let aq = Buf::from_bytes(&st, &rng.e4m3(m * k))?;
            let as_ = Buf::from_bytes(&st, &rng.e8m0(m * (k / 32), 118, 130))?;
            let c = ops.buf(m * n * 2)?;
            let ms = time_ms(&ops, reps, || {
                ck("gemm_f8_tiles", unsafe {
                    ffi::memra_v41_gemm_f8_tiles(
                        aq.p(&st).c(),
                        k as i64,
                        as_.p(&st).c(),
                        (k / 32) as i64,
                        tld.p(&st).c(),
                        ntd.p(&st).i(),
                        ntiles as i32,
                        ewd.p(&st).c(),
                        0,
                        (k / 2) as i64,
                        (k / 32) as i64,
                        1,
                        c.p(&st).m(),
                        n as i64,
                        0,
                        n as i32,
                        k as i32,
                        bm as i32,
                        ops.sv(),
                    )
                })
            })?;
            let hsh = sha(&c.host(&st)?);
            let med = median(&ms);
            let bytes = (ng * n * k / 2 + m * k + m * n * 2) as f64;
            let flops = 2.0 * (m * n * k) as f64;
            eprintln!(
                "gemm mxfp4 tiles {name} experts {ng} rows {rows} bm {bm}: median {:.4} ms ({:.0} GB/s, {:.1} TFLOPS) sha {hsh}",
                med,
                bytes / med / 1e6,
                flops / med / 1e9
            );
            out.push(format!(
                "{{\"op\": \"mxfp4_tiles\", \"name\": \"{name}\", \"experts\": {ng}, \"rows\": {rows}, \"bm\": {bm}, \"n\": {n}, \"k\": {k}, \"median_ms\": {med:.5}, \"ms\": {}, \"sha\": \"{hsh}\"}}",
                fmt_list(&ms)
            ));
        }
    }
    Ok(format!(
        "{{\"bench\": \"gemm\", \"device\": {dev:?}, \"reps\": {reps},\n \"rows\": [\n  {}\n ]\n}}\n",
        out.join(",\n  ")
    ))
}

// ------------------------------------------------------------------------------------------------
// instruction ceilings

fn ceil_bench(_a: &[String]) -> Result<String, String> {
    let e = memra_engine::Engine::new(0).map_err(|e| e.to_string())?;
    let st = e.stream();
    let dev = e.ctx().name().unwrap_or_default();
    let ops = Ops::new(st.clone());
    let sms = e.sm_count() as usize;
    let out = ops.buf(4 * 1024 * 1024)?;
    let mut rows = Vec::new();
    // (kind, name, flops per inner iteration per warp)
    let kinds: [(i32, &str, f64); 5] = [
        (0, "e4m3_mma_accumulate", 4.0 * 16.0 * 8.0 * 32.0 * 2.0),
        (
            1,
            "e4m3_mma_from_zero_then_scale_3op",
            4.0 * 16.0 * 8.0 * 32.0 * 2.0,
        ),
        (
            2,
            "e4m3_mma_from_zero_then_scale_ffma",
            4.0 * 16.0 * 8.0 * 32.0 * 2.0,
        ),
        (3, "bf16_mma_accumulate", 4.0 * 16.0 * 8.0 * 16.0 * 2.0),
        (4, "fp32_ffma", 8.0 * 32.0 * 2.0),
    ];
    for &(kind, name, fl) in &kinds {
        let mut best = 0f64;
        for &wpb in &[4usize, 8] {
            for &bps in &[1usize, 2, 4] {
                let blocks = sms * bps;
                let iters = 4096;
                let ms = time_ms(&ops, 5, || {
                    ck("peak", unsafe {
                        ffi::memra_v41_peak(
                            kind,
                            blocks as i32,
                            (wpb * 32) as i32,
                            iters,
                            out.p(&st).fm(),
                            ops.sv(),
                        )
                    })
                })?;
                let med = median(&ms);
                let tf = fl * iters as f64 * (blocks * wpb) as f64 / med / 1e9;
                eprintln!(
                    "peak {name}: {blocks} blocks x {wpb} warps: {med:.4} ms, {tf:.1} TFLOPS"
                );
                best = best.max(tf);
                rows.push(format!(
                    "{{\"kind\": \"{name}\", \"blocks\": {blocks}, \"warps_per_block\": {wpb}, \"iters\": {iters}, \"median_ms\": {med:.5}, \"tflops\": {tf:.2}}}"
                ));
            }
        }
        eprintln!("peak {name}: best {best:.1} TFLOPS");
    }
    Ok(format!(
        "{{\"bench\": \"ceil\", \"device\": {dev:?}, \"sms\": {sms},\n \"rows\": [\n  {}\n ]\n}}\n",
        rows.join(",\n  ")
    ))
}
