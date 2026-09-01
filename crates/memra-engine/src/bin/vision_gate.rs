//! Vision tower parity gate: image -> preprocessor -> ViT forward -> merger embeddings.
//!
//! Modes:
//!   vision-gate <image>                       print embedding stats (shape, norms, checksum)
//!   vision-gate <image> --dump <out.bin>      also dump f32 LE embeddings + a .json sidecar
//!   vision-gate <image> --ref <ref.bin>       compare vs an HF reference dump (same layout):
//!                                             per-token cosine, gate PASS iff min cosine > 0.999
//!
//! `MEMRA_VISION_DIR` must point at a directory carrying `outside.safetensors` with the
//! `model.visual.*` tensors. The reference dump comes from tools/hf_vision_ref.py (HF
//! transformers merger output for the same image, row-major merged-grid order).

use memra_engine::Engine;
use memra_engine::vision::VisionTower;
use memra_engine::vision_pre::prep_image_bytes;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let image_path = args
        .next()
        .expect("usage: vision-gate <image> [--dump out.bin] [--ref ref.bin]");
    let rest: Vec<String> = args.collect();
    let flag = |name: &str| {
        rest.iter()
            .position(|a| a == name)
            .and_then(|i| rest.get(i + 1))
            .cloned()
    };
    let dir =
        std::env::var("MEMRA_VISION_DIR").expect("MEMRA_VISION_DIR must point at the tower dir");
    let e = Engine::new(
        std::env::var("MEMRA_PROBE_DEVICE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
    )?;
    let tower = VisionTower::load(&e, std::path::Path::new(&dir))?;
    let bytes = std::fs::read(&image_path)?;
    let video = rest.iter().any(|a| a == "--video");
    let t0 = std::time::Instant::now();
    let (patches, groups, gh, gw) = if video {
        let vid = memra_engine::vision_pre::prep_video_gif(&bytes)?;
        let (gh, gw) = (vid.groups[0].gh, vid.groups[0].gw);
        let mut cat = Vec::new();
        for g in &vid.groups {
            cat.extend_from_slice(&g.patches);
        }
        println!(
            "video: {} groups, grid {}x{}, timestamps {:?}",
            vid.groups.len(),
            gh,
            gw,
            vid.timestamps
        );
        (cat, vid.groups.len(), gh, gw)
    } else {
        let prep = prep_image_bytes(&bytes)?;
        let (gh, gw) = (prep.gh, prep.gw);
        (prep.patches, 1, gh, gw)
    };
    let t_prep = t0.elapsed();
    let t1 = std::time::Instant::now();
    let emb_d = tower.forward_seq(&e, &patches, groups, gh, gw)?;
    let emb = e.dtoh(&emb_d)?;
    let t_fwd = t1.elapsed();
    let n_tok = groups * gh * gw / 4;
    let dim = emb.len() / n_tok;
    let mean_norm = (0..n_tok)
        .map(|t| {
            emb[t * dim..(t + 1) * dim]
                .iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum::<f64>()
                .sqrt()
        })
        .sum::<f64>()
        / n_tok as f64;
    println!(
        "vision-gate: grid {}x{} -> {} tokens x {}; prep {:.1}ms fwd {:.1}ms; mean_norm {:.4} first8 {:?}",
        gh,
        gw,
        n_tok,
        dim,
        t_prep.as_secs_f64() * 1e3,
        t_fwd.as_secs_f64() * 1e3,
        mean_norm,
        &emb[..8.min(emb.len())]
    );
    if let Some(out) = flag("--dump") {
        let raw: Vec<u8> = emb.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&out, raw)?;
        std::fs::write(
            format!("{out}.json"),
            format!(
                "{{\"gh\":{},\"gw\":{},\"tokens\":{},\"dim\":{}}}",
                gh, gw, n_tok, dim
            ),
        )?;
        println!("dumped {n_tok}x{dim} f32le -> {out}");
    }
    if let Some(refp) = flag("--ref") {
        let raw = std::fs::read(&refp)?;
        let refv: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        if refv.len() != emb.len() {
            return Err(format!(
                "reference length {} != ours {} (tokens {} dim {})",
                refv.len(),
                emb.len(),
                n_tok,
                dim
            )
            .into());
        }
        let mut min_cos = f64::INFINITY;
        let mut mean_cos = 0.0f64;
        let mut worst = 0usize;
        for t in 0..n_tok {
            let a = &emb[t * dim..(t + 1) * dim];
            let b = &refv[t * dim..(t + 1) * dim];
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for i in 0..dim {
                dot += a[i] as f64 * b[i] as f64;
                na += (a[i] as f64).powi(2);
                nb += (b[i] as f64).powi(2);
            }
            let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
            mean_cos += cos;
            if cos < min_cos {
                min_cos = cos;
                worst = t;
            }
        }
        mean_cos /= n_tok as f64;
        let pass = min_cos > 0.999;
        println!(
            "parity: mean_cos {mean_cos:.6} min_cos {min_cos:.6} (worst token {worst}) -> {}",
            if pass { "PASS" } else { "FAIL" }
        );
        if !pass {
            return Err("vision parity gate FAILED (min cosine <= 0.999)".into());
        }
    }
    Ok(())
}
