//! DSV4 two-card native-context allocation gate.
//!
//! Loads the exact artifact at 1M model context with bundled DSpark, then allocates
//! one full-capacity compact trunk state, DSpark session state, and width-32 prefill
//! transaction workspace. This is a reachability gate, not serving or performance.

use memra_engine::dsv4_gpu::Dsv4Gpu;
use memra_gguf::dsv4_forward::FixtureSpec;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: dsv4_capacity_gate <model-dir> <fixtures.json> [dev0,dev1]");
        std::process::exit(2);
    }
    let model_dir = Path::new(&args[1]);
    let fixture = FixtureSpec::load(Path::new(&args[2]));
    let devices: Vec<usize> = args
        .get(3)
        .map(|raw| {
            raw.split(',')
                .map(|part| part.parse().expect("device index"))
                .collect()
        })
        .unwrap_or_else(|| vec![0, 1]);
    let max_seq = 1_048_576usize;
    let chunk = 32usize;
    let gpu = Dsv4Gpu::load(model_dir, &devices, fixture.variant, max_seq)
        .expect("load 1M DSV4 GPU model");
    assert!(
        gpu.dspark.is_some(),
        "capacity gate requires bundled DSpark"
    );
    let post_load = gpu.vram_report().expect("post-load VRAM report");
    let state = gpu
        .alloc_decode_state_for_transient(max_seq, chunk)
        .expect("allocate full 1M compact trunk state");
    let dstate = gpu
        .dspark_alloc_state()
        .expect("allocate DSpark session state");
    let vstate = gpu
        .alloc_prefill_state_for(max_seq, chunk)
        .expect("allocate width-32 1M prefill transaction state");
    let post_alloc = gpu.vram_report().expect("post-allocation VRAM report");
    let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
    println!(
        "[dsv4-capacity-gate] placement split_at={} post_load={} post_alloc={} cache_gib=[{:.3},{:.3}] verify_gib=[{:.3},{:.3}]",
        gpu.split_at,
        post_load
            .iter()
            .map(|(dev, free, total, _)| format!(
                "dev{dev}:{:.3}/{:.3}GiB",
                gib(total - free),
                gib(*total)
            ))
            .collect::<Vec<_>>()
            .join(","),
        post_alloc
            .iter()
            .map(|(dev, free, total, _)| format!(
                "dev{dev}:{:.3}/{:.3}GiB",
                gib(total - free),
                gib(*total)
            ))
            .collect::<Vec<_>>()
            .join(","),
        gib(state.cache_bytes[0]),
        gib(state.cache_bytes[1]),
        gib(vstate.bytes[0]),
        gib(vstate.bytes[1]),
    );
    std::hint::black_box((&state, &dstate, &vstate));
    println!("[dsv4-capacity-gate] PASS 1M compact state + DSpark + chunk32 workspace");
}
