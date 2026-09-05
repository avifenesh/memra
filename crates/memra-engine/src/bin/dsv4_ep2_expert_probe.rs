//! Compute-only same-layer expert-parallel feasibility probe for DSV4.
//!
//! Usage: dsv4_ep2_expert_probe <model-dir> <fixtures.json> [layer] [reps]

use memra_engine::dsv4_gpu::Dsv4Gpu;
use memra_gguf::dsv4_forward::FixtureSpec;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: dsv4_ep2_expert_probe <model-dir> <fixtures.json> [layer] [reps]");
        std::process::exit(2);
    }
    let layer = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(3u32);
    let reps = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(11usize);
    let fixture = FixtureSpec::load(Path::new(&args[2]));
    let gpu = Dsv4Gpu::load(Path::new(&args[1]), &[0, 1], fixture.variant, 128)
        .expect("load DSV4 GPU model");
    let result = gpu
        .expert_parallel_probe(layer, reps)
        .expect("run expert-parallel probe");
    println!(
        "[dsv4-ep2-expert-id-probe] PASS layer={} selected={:?} outputs_compared={} \
         reps={} baseline_median_ms={:.6} parallel_median_ms={:.6} speedup={:.3}x",
        result.layer,
        result.selected_experts,
        result.outputs_compared,
        reps,
        result.baseline_median_s * 1e3,
        result.parallel_median_s * 1e3,
        result.speedup,
    );
    let tp = gpu
        .expert_tensor_parallel_probe(layer, reps)
        .expect("run expert tensor-parallel probe");
    println!(
        "[dsv4-ep2-expert-tp-probe] PASS layer={} selected={:?} outputs_compared={} \
         reps={} max_abs_drift={:.8} max_rel_drift={:.8} baseline_median_ms={:.6} \
         parallel_median_ms={:.6} speedup={:.3}x",
        tp.layer,
        tp.selected_experts,
        tp.outputs_compared,
        reps,
        tp.max_abs_drift,
        tp.max_rel_drift,
        tp.baseline_median_s * 1e3,
        tp.parallel_median_s * 1e3,
        tp.speedup,
    );
}
