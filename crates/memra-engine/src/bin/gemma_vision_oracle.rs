//! Parity-oracle harness for the gemma-4 vision tower (lane/gemma-vision).
//!
//! Runs the memra tower on either a real image or a deterministic synthetic gradient,
//! and dumps the inputs + stage outputs for the independent NumPy reference
//! (`research/gemma-vision-20260816/gemma_vision_ref.py`) to compare per-token cosine.
//!
//! Usage:
//!   gemma_vision_oracle <mmproj.gguf> <out_dir> [image_path]
//!
//! Without an image it builds a 384x384 synthetic RGB gradient (deterministic, no
//! decode dependency); with one it runs the full prep law (smart-resize 48-aligned,
//! 40..280 token budget, bilinear). Dumps (f32 LE):
//!   patches.bin   [n, 768]  tower input rows, 2x-1 applied, (c,ky,kx) order
//!   grid.txt      "gw gh"
//!   rust_pre_blocks.bin / rust_blk0.bin / rust_post_blocks.bin / rust_pre_proj.bin /
//!   rust_projected.bin    stage dumps (MEMRA_VISION_DEBUG is set to out_dir)

use memra_engine::Engine;
use memra_engine::vision_gemma::{
    GV_OUT, GV_PATCH, GV_PATCH_IN, GemmaVisionTower, gemma_prep_image,
};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mmproj = args
        .next()
        .expect("usage: gemma_vision_oracle <mmproj.gguf> <out_dir> [image]");
    let out_dir = args.next().expect("out_dir required");
    std::fs::create_dir_all(&out_dir)?;
    // stage dumps land next to the reference inputs
    unsafe { std::env::set_var("MEMRA_VISION_DEBUG", &out_dir) };

    let (patches, gw, gh) = match args.next() {
        Some(img) => gemma_prep_image(&std::fs::read(img)?)?,
        None => {
            // deterministic 384x384 gradient: R = x/383, G = y/383, B = (x+y)/766,
            // patchified in the tower's (c, ky, kx) order with 2x-1 applied.
            let (side, gw) = (384usize, 384 / GV_PATCH);
            let gh = gw;
            let mut patches = vec![0f32; gw * gh * GV_PATCH_IN];
            for py in 0..gh {
                for px in 0..gw {
                    let dst = &mut patches
                        [(py * gw + px) * GV_PATCH_IN..(py * gw + px + 1) * GV_PATCH_IN];
                    for ky in 0..GV_PATCH {
                        for kx in 0..GV_PATCH {
                            let (x, y) = (px * GV_PATCH + kx, py * GV_PATCH + ky);
                            let rgb = [
                                x as f32 / (side - 1) as f32,
                                y as f32 / (side - 1) as f32,
                                (x + y) as f32 / (2 * (side - 1)) as f32,
                            ];
                            for (c, v) in rgb.iter().enumerate() {
                                dst[(c * GV_PATCH + ky) * GV_PATCH + kx] = v * 2.0 - 1.0;
                            }
                        }
                    }
                }
            }
            (patches, gw, gh)
        }
    };

    let raw: Vec<u8> = patches.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(format!("{out_dir}/patches.bin"), raw)?;
    std::fs::write(format!("{out_dir}/grid.txt"), format!("{gw} {gh}"))?;

    let e = Engine::new(0)?;
    let tower = GemmaVisionTower::load(&e, Path::new(&mmproj))?;
    let t0 = std::time::Instant::now();
    let out = tower.forward(&e, &patches, gw, gh)?;
    let host = e.dtoh(&out)?;
    let n_out = host.len() / GV_OUT;
    println!(
        "gemma_vision_oracle: grid {gw}x{gh} -> {n_out} tokens x {GV_OUT} in {:.2}s",
        t0.elapsed().as_secs_f32()
    );
    println!(
        "out[0][..4] = {:?}  out[last][..4] = {:?}",
        &host[..4],
        &host[(n_out - 1) * GV_OUT..(n_out - 1) * GV_OUT + 4],
    );
    Ok(())
}
