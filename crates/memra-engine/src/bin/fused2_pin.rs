//! fused2-pin — BIT-IDENTITY gate for `matmul_nvfp4_fused2` vs two `matmul_pre` singles.
//!
//! The fused2 kernel's law (like fused3/fused4): per (tensor,row) the seg body is
//! `nvfp4_mmvq_multirow_rp` VERBATIM, so its outputs must equal two separate single
//! launches EXACTLY (`f32::to_bits` equality), not approximately. This bin loads a real
//! model through the real loader (rp repack included — raw file bytes are NOT the rp
//! layout, so a synthetic pin can't exercise the shipped path), finds every same-in_f
//! NVFP4 pair the dispatch actually fuses (attn wq/wk and dense ffn gate/up), and pins
//! each against the singles on a deterministic activation.
//!
//! Usage: fused2-pin <model.gguf> [max_pairs]
//! Exit 0 = every pinned pair bit-identical; nonzero = any mismatch (or no pair found).

use memra_engine::Engine;
use memra_engine::hybrid::{Ffn, HybridModel, Mixer};
use memra_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: fused2-pin <model.gguf> [max_pairs]");
    let max_pairs: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let m = HybridModel::load(&e, &g)?;

    let mut pinned = 0usize;
    let mut fails = 0usize;
    let mut skipped = 0usize;

    // deterministic pseudo-random activation per in_f (the mmvq checks' pr() recipe)
    let pr = |i: usize| -> f32 {
        let h = (i.wrapping_mul(2654435761) ^ 0x9E37_79B9) as u32;
        ((h >> 8) as f32 / (1u32 << 24) as f32) - 0.5
    };

    let mut pin_pair = |label: String,
                        w0: &memra_engine::model::GpuTensor,
                        w1: &memra_engine::model::GpuTensor|
     -> Result<(), Box<dyn std::error::Error>> {
        if pinned + skipped >= max_pairs * 2 {
            return Ok(());
        }
        let in_f = w0.in_features();
        if w1.in_features() != in_f {
            skipped += 1;
            return Ok(());
        }
        let x: Vec<f32> = (0..in_f).map(|i| pr(i + 313) * 0.1).collect();
        let xd = e.htod(&x)?;
        let (aq, ad) = e.quantize_q8_1(&xd, 1, in_f)?;
        let Some((f0, f1)) = e.matmul_nvfp4_fused2(w0, w1, &aq, &ad, 1)? else {
            // not an NVFP4 rp pair on this model (or seam off) — not a failure, not a pin
            skipped += 1;
            println!("SKIP {label}: fused2 declined (not an NVFP4 rp pair here)");
            return Ok(());
        };
        let h0 = e.zeros(0)?;
        let s0 = e.matmul_pre(w0, &aq, &ad, &h0, 1)?;
        let s1 = e.matmul_pre(w1, &aq, &ad, &h0, 1)?;
        let (f0h, f1h) = (e.dtoh(&f0)?, e.dtoh(&f1)?);
        let (s0h, s1h) = (e.dtoh(&s0)?, e.dtoh(&s1)?);
        let bit_eq = |a: &[f32], b: &[f32]| {
            a.len() == b.len()
                && a.iter()
                    .zip(b.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits())
        };
        let ok = bit_eq(&f0h, &s0h) && bit_eq(&f1h, &s1h);
        pinned += 1;
        if ok {
            println!(
                "PIN {label}: BIT-IDENTICAL ({} + {} rows)",
                f0h.len(),
                f1h.len()
            );
        } else {
            fails += 1;
            let first = f0h
                .iter()
                .zip(s0h.iter())
                .position(|(x, y)| x.to_bits() != y.to_bits())
                .map(|i| format!("y0[{i}]"))
                .or_else(|| {
                    f1h.iter()
                        .zip(s1h.iter())
                        .position(|(x, y)| x.to_bits() != y.to_bits())
                        .map(|i| format!("y1[{i}]"))
                })
                .unwrap_or_else(|| "len".into());
            println!("PIN {label}: FAIL (first diff at {first})");
        }
        Ok(())
    };

    for (il, l) in m.layers.iter().enumerate() {
        if let Mixer::Full(fa) = &l.mixer {
            pin_pair(format!("blk.{il} attn wq/wk"), &fa.wq, &fa.wk)?;
        }
        if let Ffn::Dense {
            ffn_gate, ffn_up, ..
        } = &l.ffn
        {
            pin_pair(format!("blk.{il} ffn gate/up"), ffn_gate, ffn_up)?;
        }
    }

    println!("fused2-pin: {pinned} pinned, {fails} FAIL, {skipped} skipped");
    if pinned == 0 {
        eprintln!("fused2-pin: NO pair pinned — model carries no NVFP4 rp pair (wrong artifact?)");
        std::process::exit(2);
    }
    if fails > 0 {
        std::process::exit(1);
    }
    Ok(())
}
