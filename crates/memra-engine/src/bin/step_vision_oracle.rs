//! Parity-oracle harness for the step37 (Step-3.7-Flash) vision tower
//! (lane/step37-vision, 2026-08-30).
//!
//! Runs the memra tower on either a real image or a deterministic synthetic gradient
//! and dumps the inputs + stage outputs for the two offline references to score
//! per-token cosine per stage:
//!   - research/step37-vision-20260830/step_vision_ref.py (independent NumPy
//!     implementation of the derived law, same safetensors weights), and
//!   - research/step37-vision-20260830/step_vision_vendor_ref.py (the vendor's own
//!     vision_encoder.py + downsamplers + projector via transformers, offline only).
//!
//! Usage:
//!   step_vision_oracle <model_dir> <out_dir> [--grid 52|36] [image]
//!
//! Without an image it builds a deterministic synthetic RGB gradient at the exact ViT
//! input size for the grid (52 -> 728px main view, 36 -> 504px crop tile), CLIP
//! mean/std applied, patchified in the tower's (c, ky, kx) order — no decode or
//! resample in the loop, so the tower is gated independently of resampling kernels.
//! With an image it runs the full vendor prep law (pad/cap/tile) and forwards the
//! MAIN view. Dumps (f32 LE):
//!   patches.bin    [n, 588] tower input rows
//!   grid.txt       "g"
//!   rust_pre_blocks.bin / rust_blk0.bin / rust_post_blocks.bin /
//!   rust_downsampled.bin / rust_projected.bin   (MEMRA_VISION_DEBUG = out_dir)
//!
//! TF32 law (gemma lane finding): run parity with NVIDIA_TF32_OVERRIDE=0; measure the
//! TF32-on arm separately before any serving decision.

use memra_engine::Engine;
use memra_engine::vision_step::{
    SV_GRID_MAIN, SV_GRID_TILE, SV_PATCH, SV_PATCH_IN, StepVisionTower, step_prep_image,
};
use std::path::Path;

/// CLIP normalization, mirrored from vision_step (private there by design: the oracle
/// bakes its own copy so a constant drift shows up as a parity failure, not a silent
/// shared change).
const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: step_vision_oracle <model_dir> <out_dir> [--grid 52|36] [image]");
        std::process::exit(2);
    }
    let model_dir = args.remove(0);
    let out_dir = args.remove(0);
    let mut grid = SV_GRID_MAIN;
    if let Some(i) = args.iter().position(|a| a == "--grid") {
        args.remove(i);
        grid = args.remove(i).parse()?;
        assert!(
            grid == SV_GRID_MAIN || grid == SV_GRID_TILE,
            "step37 ViT inputs are only ever 52 (728 main) or 36 (504 tile) grids"
        );
    }
    std::fs::create_dir_all(&out_dir)?;
    unsafe { std::env::set_var("MEMRA_VISION_DEBUG", &out_dir) };

    let (patches, g) = match args.first() {
        Some(img) => {
            let unit = step_prep_image(&std::fs::read(img)?)?;
            println!(
                "image prep: {} tile(s), newline_mask {:?}; forwarding the MAIN view",
                unit.tiles.len(),
                unit.newline_mask
            );
            (unit.main, SV_GRID_MAIN)
        }
        None => {
            // deterministic gradient at the ViT input size: R = x/(s-1), G = y/(s-1),
            // B = (x+y)/(2(s-1)); CLIP-normalized, (c, ky, kx) patch order.
            let side = grid * SV_PATCH;
            let mut patches = vec![0f32; grid * grid * SV_PATCH_IN];
            for py in 0..grid {
                for px in 0..grid {
                    let dst = &mut patches
                        [(py * grid + px) * SV_PATCH_IN..(py * grid + px + 1) * SV_PATCH_IN];
                    for ky in 0..SV_PATCH {
                        for kx in 0..SV_PATCH {
                            let (x, y) = (px * SV_PATCH + kx, py * SV_PATCH + ky);
                            let rgb = [
                                x as f32 / (side - 1) as f32,
                                y as f32 / (side - 1) as f32,
                                (x + y) as f32 / (2 * (side - 1)) as f32,
                            ];
                            for (c, v) in rgb.iter().enumerate() {
                                dst[(c * SV_PATCH + ky) * SV_PATCH + kx] = (v - MEAN[c]) / STD[c];
                            }
                        }
                    }
                }
            }
            (patches, grid)
        }
    };

    let raw: Vec<u8> = patches.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(format!("{out_dir}/patches.bin"), raw)?;
    std::fs::write(format!("{out_dir}/grid.txt"), format!("{g}"))?;

    let e = Engine::new(
        std::env::var("MEMRA_PROBE_DEVICE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
    )?;
    let tower = StepVisionTower::load(&e, Path::new(&model_dir))?;
    let t0 = std::time::Instant::now();
    let out = tower.forward(&e, &patches, g)?;
    let host = e.dtoh(&out)?;
    let w = tower.out_width();
    let n_out = host.len() / w;
    println!(
        "step_vision_oracle: grid {g}x{g} -> {n_out} rows x {w} in {:.2}s",
        t0.elapsed().as_secs_f32()
    );
    println!(
        "out[0][..4] = {:?}  out[last][..4] = {:?}",
        &host[..4],
        &host[(n_out - 1) * w..(n_out - 1) * w + 4],
    );
    Ok(())
}
