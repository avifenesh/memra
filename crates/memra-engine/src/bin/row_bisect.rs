//! row-bisect (memra#1203): at chosen prompt rows, record the last-row hidden state of every
//! layer under the prefill reference (`forward_last` over the prefix, f32 K/V) and under the
//! eager decode (`decode_step` over the quantized cache), and the batched prime's logits, then
//! print where the two programs part.
//!
//! usage: row-bisect <model.gguf> <prompt-file> <row,row,...> [--out DIR] [--no-prime]
//!
//! A row `r` is the logits row after `prompt[..=r]` (the argmax-margin-probe's `pos`). Per row
//! and program it prints the top-2 ids and margin; per layer the relative L2 distance of the
//! mixer output, of the post-attention residual and of the residual after the layer; and for a
//! MoE layer the router's top-k computed on the host from each program's own post-attention
//! residual (rms_norm with the layer's post_attn_norm, then the f32 router rows, softmax), with
//! the score gap between the last selected and the first unselected expert. `--out` writes
//! every tap as little-endian f32 for offline analysis.

use memra_engine::Engine;
use memra_engine::hybrid::{Ffn, HybridModel, Mixer};
use memra_engine::pp::new_cache as new_model_cache;
use memra_engine::row_tap::{self, Tap};
use memra_gguf::GgufFile;
use memra_tokenizer::Tokenizer;
use std::collections::BTreeMap;
use std::io::Write as _;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn top2(l: &[f32]) -> (usize, f32, usize, f32) {
    let (mut a, mut b) = (0usize, usize::MAX);
    for i in 1..l.len() {
        if l[i] > l[a] {
            b = a;
            a = i;
        } else if b == usize::MAX || l[i] > l[b] {
            b = i;
        }
    }
    (a, l[a], b, l[b])
}

fn rel(a: &[f32], b: &[f32]) -> (f64, f64) {
    let mut num = 0f64;
    let mut den = 0f64;
    let mut mx = 0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (*x as f64) - (*y as f64);
        num += d * d;
        den += (*y as f64) * (*y as f64);
        mx = mx.max(d.abs());
    }
    ((num / den.max(1e-30)).sqrt(), mx)
}

struct Router {
    /// [n_expert][n_embd] f32 rows
    w: Vec<f32>,
    norm: Vec<f32>,
    n_expert: usize,
}

/// Host router: softmax(W . rms_norm(x1) * norm), top-k ids by score, and the gap (in router
/// logit units) between the k-th and (k+1)-th logits.
fn route(r: &Router, x1: &[f32], eps: f32, k: usize) -> (Vec<usize>, Vec<f64>, f64) {
    let n = x1.len();
    let ms: f64 = x1.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / n as f64;
    let inv = 1.0 / (ms + eps as f64).sqrt();
    let z: Vec<f64> = x1
        .iter()
        .zip(&r.norm)
        .map(|(v, g)| (*v as f64) * inv * (*g as f64))
        .collect();
    let logits: Vec<f64> = (0..r.n_expert)
        .map(|j| {
            r.w[j * n..(j + 1) * n]
                .iter()
                .zip(&z)
                .map(|(a, b)| (*a as f64) * b)
                .sum()
        })
        .collect();
    let mut idx: Vec<usize> = (0..r.n_expert).collect();
    idx.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap());
    let gap = logits[idx[k - 1]] - logits[idx[k]];
    let sel = idx[..k].to_vec();
    (sel, logits, gap)
}

fn by_key(taps: &[Tap]) -> BTreeMap<(usize, &'static str), &Vec<f32>> {
    taps.iter().map(|t| ((t.layer, t.name), &t.row)).collect()
}

fn write_taps(
    dir: &std::path::Path,
    label: &str,
    row: usize,
    taps: &[Tap],
    logits: &[f32],
) -> Res<()> {
    std::fs::create_dir_all(dir)?;
    let mut f = std::fs::File::create(dir.join(format!("{label}-r{row}.f32")))?;
    let mut idx = std::fs::File::create(dir.join(format!("{label}-r{row}.idx")))?;
    for t in taps {
        writeln!(idx, "{}\t{}\t{}", t.name, t.layer, t.row.len())?;
        for v in &t.row {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    writeln!(idx, "logits\t-1\t{}", logits.len())?;
    for v in logits {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pos = Vec::new();
    let mut out: Option<std::path::PathBuf> = None;
    let mut prime = true;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out = Some(it.next().ok_or("--out DIR")?.into()),
            "--no-prime" => prime = false,
            _ => pos.push(a.clone()),
        }
    }
    let [model_path, prompt_file, rows_arg] = pos.as_slice() else {
        return Err(
            "usage: row-bisect <model.gguf> <prompt-file> <row,row,...> [--out DIR] [--no-prime]"
                .into(),
        );
    };
    let mut rows: Vec<usize> = rows_arg
        .split(',')
        .map(|s| s.parse::<usize>())
        .collect::<Result<_, _>>()?;
    rows.sort_unstable();
    rows.dedup();

    let e = Engine::new(0)?;
    let g = GgufFile::open(model_path)?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    let tok = Tokenizer::from_gguf(&g).map_err(|err| format!("tokenizer: {err}"))?;
    let text = std::fs::read_to_string(prompt_file)?;
    let prompt = tok.encode(&text, true);
    let maxrow = *rows.last().ok_or("no rows")?;
    assert!(maxrow + 1 < prompt.len(), "row past the prompt");
    let eps = model.cfg.rms_eps;
    let n_embd = model.cfg.n_embd as usize;
    println!(
        "row-bisect: T={} rows={rows:?} sm_count={} model={model_path}",
        prompt.len(),
        e.sm_count()
    );
    let envs = [
        "MEMRA_GDN_MMA",
        "MEMRA_GDN_CHUNKED",
        "MEMRA_FAST",
        "MEMRA_FA_F16PV",
        "MEMRA_MMQ_W4A8",
        "MEMRA_GDN_K45",
    ];
    let env_line: Vec<String> = envs
        .iter()
        .map(|k| {
            format!(
                "{k}={}",
                std::env::var(k).unwrap_or_else(|_| "<unset>".into())
            )
        })
        .collect();
    println!("env: {}", env_line.join(" "));

    // Host copies of each MoE layer's router and pre-FFN norm.
    let k_used = model.cfg.moe.as_ref().map(|m| m.expert_used_count as usize);
    let mut routers: BTreeMap<usize, Router> = BTreeMap::new();
    for (il, layer) in model.layers.iter().enumerate() {
        if let Ffn::Moe(m) = &layer.ffn {
            let w = e.dtoh(m.gate_inp.float_data())?;
            let norm = e.dtoh(layer.post_attn_norm.float_data())?;
            let n_expert = w.len() / n_embd;
            routers.insert(il, Router { w, norm, n_expert });
        }
    }
    let kinds: Vec<&str> = model
        .layers
        .iter()
        .map(|l| match &l.mixer {
            Mixer::Full(_) => "attn",
            Mixer::Linear(_) => "gdn",
            Mixer::Mla(_) => "mla",
            Mixer::Kda(_) => "kda",
        })
        .collect();

    // Prefill side: one forward_last per row. `RB_PREFILL_ENV="K=V,K2=V2"` sets engine knobs
    // for the prefill forwards only (for example MEMRA_NOFA, which the decode over a quantized
    // cache refuses), and removes them before the decode pass.
    let prefill_env: Vec<(String, String)> = std::env::var("RB_PREFILL_ENV")
        .unwrap_or_default()
        .split(',')
        .filter_map(|kv| {
            kv.split_once('=')
                .map(|(k, v)| (k.to_string(), v.to_string()))
        })
        .collect();
    println!("prefill-only env: {prefill_env:?}");
    for (k, v) in &prefill_env {
        // SAFETY: single-threaded research binary; nothing else reads the environment here.
        unsafe { std::env::set_var(k, v) };
    }
    let mut pre: BTreeMap<usize, (Vec<Tap>, Vec<f32>)> = BTreeMap::new();
    for &r in &rows {
        row_tap::arm();
        let l = model.forward_last(&e, &prompt[..=r])?;
        pre.insert(r, (row_tap::take(), l));
    }
    for (k, _) in &prefill_env {
        // SAFETY: as above.
        unsafe { std::env::remove_var(k) };
    }
    // Decode side: one eager pass, armed at the chosen steps.
    let mut dec: BTreeMap<usize, (Vec<Tap>, Vec<f32>)> = BTreeMap::new();
    {
        let mut cache = new_model_cache(&e, &model.cfg, maxrow + 8)?;
        for (i, &tk) in prompt[..=maxrow].iter().enumerate() {
            let want = rows.binary_search(&i).is_ok();
            if want {
                row_tap::arm();
            }
            let l = model.decode_step(&e, tk, &mut cache)?;
            if want {
                dec.insert(i, (row_tap::take(), l));
            }
        }
    }
    // Batched prime over the quantized cache, per row (last-position logits only).
    let mut prm: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
    if prime {
        for &r in &rows {
            let mut pc = new_model_cache(&e, &model.cfg, r + 9)?;
            let (l, _, _) = model.prime_cache(&e, &prompt[..=r], &mut pc, 0)?;
            prm.insert(r, l);
        }
    }

    let text_of = |id: usize| format!("{:?}", tok.decode(&[id as u32]));
    for &r in &rows {
        let (pt, pl) = &pre[&r];
        let (dt, dl) = &dec[&r];
        println!(
            "\n=== row {r} (next token {} = {})",
            prompt[r + 1],
            text_of(prompt[r + 1] as usize)
        );
        let mut progs: Vec<(&str, &Vec<f32>)> = vec![("prefill", pl), ("decode", dl)];
        if let Some(l) = prm.get(&r) {
            progs.push(("prime", l));
        }
        for (name, l) in &progs {
            let (a, la, b, lb) = top2(l);
            println!(
                "  {name:8} top1 {a} {} {la:.4}  top2 {b} {} {lb:.4}  margin {:.4}  logit[next]={:.4}",
                text_of(a),
                text_of(b),
                la - lb,
                l[prompt[r + 1] as usize]
            );
        }
        let (pa, _, pb, _) = top2(pl);
        let (da, _, db, _) = top2(dl);
        let mut ids = vec![pa, pb, da, db];
        ids.sort_unstable();
        ids.dedup();
        for id in ids {
            let ps: Vec<String> = progs
                .iter()
                .map(|(n, l)| format!("{n}={:.4}", l[id]))
                .collect();
            println!("  id {id} {}: {}", text_of(id), ps.join(" "));
        }
        let (rl, mx) = rel(pl, dl);
        println!("  logits prefill vs decode: rel {rl:.3e} maxabs {mx:.4}");
        if let Some(l) = prm.get(&r) {
            let (a, b) = rel(l, dl);
            let (c, d) = rel(pl, l);
            println!(
                "  logits prime vs decode: rel {a:.3e} maxabs {b:.4}; prefill vs prime: rel {c:.3e} maxabs {d:.4}"
            );
        }
        let p = by_key(pt);
        let d = by_key(dt);
        println!(
            "  layer kind  mix_rel     x1_rel      out_rel     out_maxabs  router(k) prefill-vs-decode"
        );
        let mut route_lines: Vec<String> = Vec::new();
        for il in 0..model.layers.len() {
            let (Some(pm), Some(dm), Some(px1), Some(dx1), Some(pf), Some(df)) = (
                p.get(&(il, "mix")),
                d.get(&(il, "mix")),
                p.get(&(il, "x1")),
                d.get(&(il, "x1")),
                p.get(&(il, "ffn")),
                d.get(&(il, "ffn")),
            ) else {
                println!("  {il:5} missing taps");
                continue;
            };
            let (mr, _) = rel(pm, dm);
            let (xr, _) = rel(px1, dx1);
            let po: Vec<f32> = px1.iter().zip(pf.iter()).map(|(a, b)| a + b).collect();
            let dout: Vec<f32> = dx1.iter().zip(df.iter()).map(|(a, b)| a + b).collect();
            let (or, om) = rel(&po, &dout);
            let mut rline = String::new();
            if let (Some(rt), Some(k)) = (routers.get(&il), k_used) {
                let (ps, plg, pg) = route(rt, px1, eps, k);
                let (ds, dlg, dg) = route(rt, dx1, eps, k);
                let mut pss = ps.clone();
                pss.sort_unstable();
                let mut dss = ds.clone();
                dss.sort_unstable();
                let same = pss == dss;
                let maxdl = plg
                    .iter()
                    .zip(&dlg)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f64, f64::max);
                rline = format!(
                    "{} gap_p {pg:.4} gap_d {dg:.4} max|dlogit| {maxdl:.4}",
                    if same { "same" } else { "DIFF" }
                );
                let fmt_sel = |sel: &[usize], lg: &[f64]| -> String {
                    let mx = lg.iter().cloned().fold(f64::MIN, f64::max);
                    let den: f64 = lg.iter().map(|v| (v - mx).exp()).sum();
                    sel.iter()
                        .map(|x| format!("{x}:{:.4}:{:.5}", lg[*x], (lg[*x] - mx).exp() / den))
                        .collect::<Vec<_>>()
                        .join(",")
                };
                route_lines.push(format!(
                    "  route row={r} layer={il} prefill={} decode={}",
                    fmt_sel(&ps, &plg),
                    fmt_sel(&ds, &dlg)
                ));
                if !same {
                    let only_p: Vec<usize> =
                        pss.iter().filter(|x| !dss.contains(x)).copied().collect();
                    let only_d: Vec<usize> =
                        dss.iter().filter(|x| !pss.contains(x)).copied().collect();
                    rline.push_str(&format!(" only_prefill {only_p:?} only_decode {only_d:?}"));
                    for x in only_p.iter().chain(only_d.iter()) {
                        rline.push_str(&format!(" [e{x} p={:.4} d={:.4}]", plg[*x], dlg[*x]));
                    }
                    rline.push_str(&format!(
                        " kth_p {:.4} k1th_p {:.4}",
                        plg[ps[k - 1]],
                        plg[{
                            let mut idx: Vec<usize> = (0..rt.n_expert).collect();
                            idx.sort_by(|a, b| plg[*b].partial_cmp(&plg[*a]).unwrap());
                            idx[k]
                        }]
                    ));
                }
            }
            println!(
                "  {il:5} {:5} {mr:.3e}   {xr:.3e}   {or:.3e}   {om:9.4}   {rline}",
                kinds[il]
            );
        }
        for line in &route_lines {
            println!("{line}");
        }
        if let Some(dir) = &out {
            write_taps(dir, "prefill", r, pt, pl)?;
            write_taps(dir, "decode", r, dt, dl)?;
            if let Some(l) = prm.get(&r) {
                write_taps(dir, "prime", r, &[], l)?;
            }
        }
    }
    Ok(())
}
