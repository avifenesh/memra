//! WIDTH WALK: the same rows through one projection at two call widths must produce the same bits.
//!
//! memra#427 classified a prime chunk of exactly PRIME_MIN_T (16) rows as its own numeric program:
//! the same rows digest differently as a 16-row chunk than inside a wider one, cold or restored.
//! `qwen-a4-continuation-gate` sees that only at the logits, after the whole layer walk. This
//! diagnostic asks each projection on its own: for every layer and every weight the prime path
//! multiplies, build ONE deterministic activation, call the prime path's own entry
//! (`Engine::matmul_prefill`) at a reference width and at each compared width, and compare the
//! shared leading rows bitwise. The first tensor that differs names the width-keyed dispatch.
//!
//! No server, no prefix cache, no drafter, no KV, no engine code changed, no flag read. The
//! activation values are synthetic: this is a dispatch question (does the SAME row take the same
//! program at both widths), not a model-quality measurement.
//!
//! usage: qwen-a4-width-walk <model.gguf> [ref_width] [widths...]   (defaults: 17; 16 48)
use memra_engine::Engine;
use memra_engine::hybrid::{Ffn, HybridModel, Mixer};
use memra_engine::model::GpuTensor;
use memra_gguf::GgufFile;

fn digest(row: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in row {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

fn qtype_name(w: &GpuTensor) -> String {
    match w {
        GpuTensor::Quant { qtype, .. } => match *qtype {
            memra_engine::QT_Q8_0 => "Q8_0".into(),
            memra_engine::QT_Q4_K => "Q4_K".into(),
            memra_engine::QT_Q6_K => "Q6_K".into(),
            memra_engine::QT_Q5_K => "Q5_K".into(),
            memra_engine::QT_Q3_K => "Q3_K".into(),
            memra_engine::QT_IQ4_XS => "IQ4_XS".into(),
            memra_engine::QT_NVFP4 => "NVFP4".into(),
            memra_engine::QT_F8_E4M3 => "F8_E4M3".into(),
            memra_engine::QT_F8_E4M3_BLK => "F8_E4M3_BLK".into(),
            memra_engine::QT_Q4_0 => "Q4_0".into(),
            other => format!("qtype{other}"),
        },
        GpuTensor::Float { .. } => "F32".into(),
        GpuTensor::FloatBf16 { .. } => "BF16".into(),
    }
}

/// Deterministic activation rows in (-1, 1): a 64-bit LCG seeded by the input width, so every
/// tensor with the same `in_f` sees the same rows and the two widths share their leading rows by
/// construction (the wider call's buffer is a prefix-extension of the narrower one).
fn activation(rows: usize, in_f: usize) -> Vec<f32> {
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15 ^ (in_f as u64);
    (0..rows * in_f)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

struct Site<'a> {
    layer: String,
    name: &'static str,
    w: &'a GpuTensor,
}

fn sites(model: &HybridModel) -> Vec<Site<'_>> {
    let mut out = Vec::new();
    for (il, layer) in model.layers.iter().enumerate() {
        let l = format!("{il:02}");
        match &layer.mixer {
            Mixer::Full(fa) => {
                out.push(Site {
                    layer: l.clone(),
                    name: "wq",
                    w: &fa.wq,
                });
                out.push(Site {
                    layer: l.clone(),
                    name: "wk",
                    w: &fa.wk,
                });
                out.push(Site {
                    layer: l.clone(),
                    name: "wv",
                    w: &fa.wv,
                });
                out.push(Site {
                    layer: l.clone(),
                    name: "wo",
                    w: &fa.wo,
                });
                if let Some(g) = fa.attn_gate.as_ref() {
                    out.push(Site {
                        layer: l.clone(),
                        name: "attn_gate",
                        w: g,
                    });
                }
            }
            Mixer::Linear(la) => {
                out.push(Site {
                    layer: l.clone(),
                    name: "wqkv",
                    w: &la.wqkv,
                });
                out.push(Site {
                    layer: l.clone(),
                    name: "wqkv_gate",
                    w: &la.wqkv_gate,
                });
                out.push(Site {
                    layer: l.clone(),
                    name: "ssm_beta",
                    w: &la.ssm_beta,
                });
                out.push(Site {
                    layer: l.clone(),
                    name: "ssm_alpha",
                    w: &la.ssm_alpha,
                });
                out.push(Site {
                    layer: l.clone(),
                    name: "ssm_out",
                    w: &la.ssm_out,
                });
            }
            Mixer::Mla(_) | Mixer::Kda(_) => {
                eprintln!("layer {l}: mixer class outside this walk (MLA/KDA), skipped");
            }
        }
        match &layer.ffn {
            Ffn::Dense {
                ffn_gate,
                ffn_up,
                ffn_down,
                ffn_down_pqs: _,
            } => {
                out.push(Site {
                    layer: l.clone(),
                    name: "ffn_gate",
                    w: ffn_gate,
                });
                out.push(Site {
                    layer: l.clone(),
                    name: "ffn_up",
                    w: ffn_up,
                });
                out.push(Site {
                    layer: l.clone(),
                    name: "ffn_down",
                    w: ffn_down,
                });
            }
            Ffn::Moe(_) => {
                eprintln!("layer {l}: MoE experts are outside this walk (dense projections only)");
            }
        }
    }
    out.push(Site {
        layer: "head".into(),
        name: "output",
        w: &model.output,
    });
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: qwen-a4-width-walk <model.gguf> [ref_width] [widths...]");
    let reference: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(17);
    let widths: Vec<usize> = {
        let rest: Vec<usize> = args.filter_map(|v| v.parse().ok()).collect();
        if rest.is_empty() { vec![16, 48] } else { rest }
    };
    let max_rows = widths.iter().copied().max().unwrap_or(0).max(reference);

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load(&e, &g)?;
    println!(
        "{path}: activation program {}; reference width {reference}; compared widths {widths:?}",
        if model.cfg.prefill_activation.is_some() {
            "PRESENT"
        } else {
            "absent"
        }
    );

    let sites = sites(&model);
    // Two scopes per tensor. `prime` is what the prime layer walk runs (`prime_layers` arms
    // `Engine::prefill_rows_scope`, memra#427); `bare` is the same call outside any scope, the
    // decode/verify class a batched verify or the exact-16 decode tier would take at m = 16.
    for scope in ["prime", "bare"] {
        let mut differing: Vec<Vec<String>> = vec![Vec::new(); widths.len()];
        let mut checked = 0usize;
        for site in &sites {
            let in_f = site.w.in_features();
            let out_f = site.w.out_features();
            let host = activation(max_rows, in_f);
            let guard = (scope == "prime").then(|| e.prefill_rows_scope());
            let x_ref = e.htod(&host[..reference * in_f])?;
            let y_ref = e.dtoh(&e.matmul_prefill(site.w, &x_ref, reference)?)?;
            let label = format!(
                "scope={scope:<5} layer {} {:<9} qtype={:<7} in_f={:<6} out_f={:<6}",
                site.layer,
                site.name,
                qtype_name(site.w),
                in_f,
                out_f
            );
            for (wi, &m) in widths.iter().enumerate() {
                let x_m = e.htod(&host[..m * in_f])?;
                let y_m = e.dtoh(&e.matmul_prefill(site.w, &x_m, m)?)?;
                let shared = m.min(reference);
                let mut rows_differ = 0usize;
                let mut maxabs = 0.0f32;
                for r in 0..shared {
                    let a = &y_ref[r * out_f..(r + 1) * out_f];
                    let b = &y_m[r * out_f..(r + 1) * out_f];
                    if a.iter().zip(b).any(|(p, q)| p.to_bits() != q.to_bits()) {
                        rows_differ += 1;
                    }
                    for (p, q) in a.iter().zip(b) {
                        maxabs = maxabs.max((p - q).abs());
                    }
                }
                let verdict = if rows_differ == 0 { "same" } else { "DIFFERS" };
                println!(
                    "{label} width {m:>3} vs {reference}: rows_differ={rows_differ}/{shared} maxabs={maxabs:.3e} \
                 ref_sha={:016x} sha={:016x} {verdict}",
                    digest(&y_ref[..shared * out_f]),
                    digest(&y_m[..shared * out_f]),
                );
                if rows_differ != 0 {
                    differing[wi].push(format!("{}/{}", site.layer, site.name));
                }
            }
            drop(guard);
            checked += 1;
        }
        for (wi, &m) in widths.iter().enumerate() {
            let names: std::collections::BTreeSet<&str> = differing[wi]
                .iter()
                .map(|s| s.split('/').nth(1).unwrap_or(s))
                .collect();
            println!(
                "WIDTH WALK scope={scope} width {m} vs {reference}: {} of {checked} tensors differ; tensor names: {:?}; sites: {:?}",
                differing[wi].len(),
                names,
                differing[wi]
            );
        }
    }
    Ok(())
}
