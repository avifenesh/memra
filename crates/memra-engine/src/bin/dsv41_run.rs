//! DeepSeek-V4.1-Flash on one card, the whole trunk: the native checkpoint runner for
//! `memra model verify checkpoint --against deepseek_v41 --native-runner <this binary>`, and the
//! long-context runner of the lane's gates.
//!
//! The 40 trunk layers run one at a time, each layer's weights loaded from the safetensors mmap,
//! used and freed (about 7.5 GB a layer, 384 MXFP4 experts included), so the 510 GB checkpoint runs
//! on a single 96 GB card. The program is the checkpoint's own (`crate::dsv41_gpu`): bf16
//! activations, the vendor's per-32 FP8 / FP4 quantize-dequantize steps, FP8 32x32 and MXFP4
//! weights as stored, fp32 accumulation. A prefill of the given tokens, then the fp32 head.
//!
//! usage: dsv41-run CKPT_DIR [TOKEN ...] [--ids FILE] [--token-map FILE] [--chunk N]
//!                  [--last-logits FILE] [--tf DIR] [--layers N]
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

use memra_engine::dsv41_gpu::{
    self as v, Buf, Ctx, Geo, LayerState, LayerW, Shared, Trace, ck, ffi, tp::Group,
};
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
    let mut ids: Vec<u32> = Vec::new();
    let mut i = 2;
    while i < a.len() {
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
        let raw = std::fs::read(&f).map_err(|e| format!("{f}: {e}"))?;
        ids = raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
            .collect();
    }
    if ids.is_empty() {
        return Err("no prompt tokens".into());
    }
    if std::env::var("MEMRA_FULL_PREC").as_deref() != Ok("1") {
        return Err("set MEMRA_FULL_PREC=1: the deepseek_v41 route runs only the checkpoint's own numeric program; no serving format exists to select".into());
    }
    let tmap = flag("--token-map")
        .or_else(|| std::env::var("MEMRA_DSV41_TOKEN_MAP").ok())
        .ok_or("the Engram hash needs the compressed token map: --token-map FILE or MEMRA_DSV41_TOKEN_MAP")?;
    let chunk: usize = flag("--chunk").and_then(|x| x.parse().ok()).unwrap_or(8192);

    let t0 = Instant::now();
    let devices: Option<Vec<usize>> =
        flag("--devices").map(|x| x.split(',').filter_map(|d| d.trim().parse().ok()).collect());
    let mc = memra_gguf::config::ModelConfig::from_config_json(&ckpt.join("config.json"))
        .map_err(|e| e.to_string())?;
    let g = Geo::from_config(&mc)?;
    let trunk =
        g.compress_ratios.len() - mc.dsv41.as_ref().unwrap().num_nextn_predict_layers as usize;
    let n_layers = flag("--layers")
        .and_then(|x| x.parse().ok())
        .unwrap_or(trunk)
        .min(trunk);
    let stm = memra_gguf::safetensors::StModel::open(&ckpt).map_err(|e| e.to_string())?;
    let ec =
        v::engram::EngramConsts::derive(&g, v::engram::token_map_from_file(Path::new(&tmap))?)?;
    let s = ids.len();
    let (hc, d) = (g.hc, g.dim);

    // embed, then hc copies of every row; the carried pre-mix starts one-hot on copy 0
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
    let hashes = if g.engram_layers.iter().any(|&l| l < n_layers) {
        Some(ec.hash_prefill(&ids)?)
    } else {
        None
    };
    // this layer's hash columns of every token
    let engram_cols = |li: usize| -> Result<Vec<i64>, String> {
        let all = hashes.as_ref().unwrap();
        let slot = ec
            .layer_ids
            .iter()
            .position(|&x| x == li)
            .ok_or("engram layer not in the constants")?;
        let (nl, cols) = (ec.layer_ids.len(), ec.cols());
        Ok((0..s)
            .flat_map(|t| all[(t * nl + slot) * cols..(t * nl + slot + 1) * cols].to_vec())
            .collect())
    };

    let mut head: Box<dyn FnMut(usize, usize) -> Result<Vec<f32>, String>>;
    // keeps the one-card engine alive behind the head closure
    let _engine;
    if let Some(devices) = devices {
        let mut grp = Group::new(&devices, &g, (s + 16).max(1024))?;
        for c in grp.cx.iter_mut() {
            c.chunk = chunk;
        }
        let n = grp.n();
        eprintln!(
            "dsv41-run: {s} tokens, {n_layers} layers, chunk {chunk}, {n} ranks on devices {devices:?}, transport {}",
            grp.transport.name()
        );
        let mut h: Vec<Buf> = Vec::with_capacity(n);
        let mut pre: Vec<Buf> = Vec::with_capacity(n);
        for c in &grp.cx {
            h.push(Buf::from_bytes(&c.ops.st, &hb)?);
            pre.push(Buf::from_f32(&c.ops.st, &pm)?);
        }
        drop(hb);
        let mut sh: Vec<Shared> = (0..n).map(|_| Shared::default()).collect();
        let mut tr: Vec<Trace> = (0..n).map(|_| Trace::default()).collect();
        for li in 0..n_layers {
            let tl = Instant::now();
            let lw = grp.load_layer(&stm, &g, li, false)?;
            let load_s = tl.elapsed().as_secs_f64();
            if lw[0].engram.is_some() {
                let mine = engram_cols(li)?;
                for r in 0..n {
                    grp.cx[r].engram(&stm, &lw[r], &h[r], &mine, s, &mut tr[r])?;
                }
            }
            let mut ls: Vec<LayerState> = (0..n).map(|_| LayerState::default()).collect();
            let (h2, p2) = grp.block_forward(&lw, &h, &pre, 0, s, &mut sh, &mut ls, &mut tr)?;
            h = h2;
            pre = p2;
            drop(lw);
            eprintln!(
                "layer {li}: load {load_s:.1}s total {:.1}s, gathers {} ({:.1} MB so far)",
                tl.elapsed().as_secs_f64(),
                grp.stats.collectives,
                grp.stats.bytes as f64 / 1e6
            );
        }
        let (nw, hw) = grp.load_head(&stm, &g)?;
        head = Box::new(move |r0, n| grp.head_logits(&h, &pre, r0, n, &nw, &hw));
        _engine = None;
    } else {
        let e = memra_engine::Engine::new(0).map_err(|e| e.to_string())?;
        let st = e.stream();
        let mut cx = Ctx::new(st.clone(), g.clone(), (s + 16).max(1024))?;
        cx.chunk = chunk;
        eprintln!(
            "dsv41-run: {s} tokens, {n_layers} layers, chunk {chunk}, {} on device",
            e.ctx().name().unwrap_or_default()
        );
        let mut h = Buf::from_bytes(&st, &hb)?;
        drop(hb);
        let mut pre = Buf::from_f32(&st, &pm)?;
        let mut sh = Shared::default();
        let mut tr = Trace::default();
        for li in 0..n_layers {
            let tl = Instant::now();
            let lw = LayerW::load(&cx.ops, &stm, &g, li, false)?;
            let load_s = tl.elapsed().as_secs_f64();
            if lw.engram.is_some() {
                let mine = engram_cols(li)?;
                cx.engram(&stm, &lw, &h, &mine, s, &mut tr)?;
            }
            let mut ls = LayerState::default();
            let (h2, p2) = cx.block_forward(&lw, &h, &pre, 0, s, &mut sh, &mut ls, &mut tr)?;
            h = h2;
            pre = p2;
            drop(lw);
            eprintln!(
                "layer {li}: load {load_s:.1}s total {:.1}s",
                tl.elapsed().as_secs_f64()
            );
        }
        // the head: collapse with the carried pre-mix, the final norm, F.linear in fp32
        let nw = v::load_vec_bf16(&cx.ops, &stm, "norm.weight", d)?;
        let hw = v::load_bf16(&cx.ops, &stm, "head.weight", g.vocab, d)?;
        let gg = g.clone();
        head = Box::new(move |r0: usize, n: usize| -> Result<Vec<f32>, String> {
            let ops = &cx.ops;
            let x = ops.buf(n * d * 2)?;
            ck("hc_pre", unsafe {
                ffi::memra_v41_hc_pre(
                    h.at(&st, r0 * hc * d * 2).m(),
                    pre.at(&st, r0 * hc * 4).f(),
                    n as i32,
                    hc as i32,
                    d as i32,
                    x.p(&st).m(),
                    d as i64,
                    ops.sv(),
                )
            })?;
            ops.rmsnorm(x.p(&st), d, &nw, x.p(&st), d, n, d, gg.norm_eps)?;
            let lg = ops.buf(n * gg.vocab * 4)?;
            ops.linear_bf16(x.p(&st), d, n, &hw, lg.p(&st), gg.vocab, true)?;
            Ok(v::bytes_f32(&lg.host(&st)?))
        });
        _engine = Some(e);
    }
    let mut head_rows = |r0: usize, n: usize| head(r0, n);
    let last = head_rows(s - 1, 1)?;
    if last.iter().any(|x| !x.is_finite()) {
        return Err("non-finite logits".into());
    }
    let am = argmax(&last);
    eprintln!(
        "dsv41-run: argmax {am} logit {:.4}, {:.1}s",
        last[am],
        t0.elapsed().as_secs_f64()
    );
    println!("argmax token = {am}");
    if let Some(f) = flag("--last-logits") {
        std::fs::write(&f, v::f32_bytes(&last)).map_err(|e| format!("{f}: {e}"))?;
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
    if let Some(dir) = flag("--tf") {
        // every position's lse and argmax, and the NLL of the next prompt token
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let mut lse = Vec::with_capacity(s);
        let mut top = Vec::with_capacity(s);
        let mut nll = Vec::with_capacity(s.saturating_sub(1));
        let mut top_ids = Vec::with_capacity(s * TOPK);
        let mut top_vals = Vec::with_capacity(s * TOPK);
        let rows = 1024;
        for r0 in (0..s).step_by(rows) {
            let n = rows.min(s - r0);
            let lg = head_rows(r0, n)?;
            for i in 0..n {
                let row = &lg[i * g.vocab..(i + 1) * g.vocab];
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let l = m + row.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln();
                lse.push(l as f32);
                top.push(argmax(row) as i32);
                for (id, val) in top_k(row) {
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

/// Logits kept per teacher-forced position, as many as the oracle tape's `top_ids`/`top_vals`.
const TOPK: usize = 32;

/// The `TOPK` largest logits of a row, largest first (lower id first among equal values).
fn top_k(x: &[f32]) -> Vec<(usize, f32)> {
    let mut best: Vec<(usize, f32)> = x.iter().take(TOPK).cloned().enumerate().collect();
    let worse = |a: &(usize, f32), b: &(usize, f32)| a.1 < b.1 || (a.1 == b.1 && a.0 > b.0);
    let mut low = (0..best.len())
        .reduce(|a, b| if worse(&best[b], &best[a]) { b } else { a })
        .unwrap_or(0);
    for (i, &v) in x.iter().enumerate().skip(TOPK) {
        if v > best[low].1 {
            best[low] = (i, v);
            low = (0..TOPK)
                .reduce(|a, b| if worse(&best[b], &best[a]) { b } else { a })
                .unwrap_or(0);
        }
    }
    best.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    best
}

fn argmax(x: &[f32]) -> usize {
    // the lowest index among equal maxima, like the oracle's argmax
    let mut b = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[b] {
            b = i;
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_is_the_sorted_largest_with_lower_ids_first_among_ties() {
        let mut x: Vec<f32> = (0..200).map(|i| ((i * 37) % 101) as f32).collect();
        x[150] = 1000.0;
        x[7] = 1000.0;
        let got = top_k(&x);
        let mut want: Vec<(usize, f32)> = x.iter().cloned().enumerate().collect();
        want.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        want.truncate(TOPK);
        assert_eq!(got, want);
        assert_eq!(got[0], (7, 1000.0));
        assert_eq!(got[0].0, argmax(&x));
    }
}
