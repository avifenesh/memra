//! Measurement-only Step B=1 serving harness.
//!
//! This deliberately uses the two production seams the legacy `decode-window-profile`
//! predates: `pp::new_cache` for stage-owned KV and the sampled lean B=1 batch path used by
//! the worker. Runtime code comes from the pinned, clean memra checkout through path deps.

use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_engine::Engine;
use memra_gguf::GgufFile;

fn step(
    model: &HybridModel,
    engine: &Engine,
    cache: &mut memra_engine::cache::Cache,
    token: u32,
    counter: u32,
) -> Result<u32, Box<dyn std::error::Error>> {
    let tokens = [token];
    let samples = [Some((0.0_f32, 0_u64, counter))];
    let mut caches = [cache];
    let (_rows, next) = model.decode_step_batch_sampled_lean(
        engine,
        &tokens,
        &mut caches,
        &samples,
        true,
    )?;
    next[0].ok_or_else(|| "device sampler returned no token".into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: ncuspike-profile <model.gguf> [depth=512] [n=32]");
    let depth: usize = std::env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(512);
    let n: usize = std::env::args()
        .nth(3)
        .and_then(|value| value.parse().ok())
        .unwrap_or(32);

    let engine = Engine::new(0)?;
    let gguf = GgufFile::open(&path)?;
    let model = HybridModel::load(&engine, &gguf)?;
    let prompt: Vec<u32> = (0..depth)
        .map(|index| (100 + (index * 7) % 900) as u32)
        .collect();
    let mut cache = memra_engine::pp::new_cache(&engine, &model.cfg, depth + n + 8)?;
    let (logits, _, _) = model.prime_cache(&engine, &prompt, &mut cache, 0)?;
    let mut token = argmax(&logits) as u32;

    for counter in 0..4 {
        token = step(&model, &engine, &mut cache, token, counter)?;
    }
    engine.stream().synchronize()?;

    unsafe {
        cudarc::driver::sys::cuProfilerStart().result()?;
    }
    let started = std::time::Instant::now();
    for offset in 0..n {
        token = step(&model, &engine, &mut cache, token, (offset + 4) as u32)?;
    }
    engine.stream().synchronize()?;
    let elapsed = started.elapsed().as_secs_f64();
    unsafe {
        cudarc::driver::sys::cuProfilerStop().result()?;
    }

    println!(
        "serving decode window @d{depth}: {n} tokens in {elapsed:.3}s = {:.1} tok/s ({:.1} us/tok) final_token={token}",
        n as f64 / elapsed,
        elapsed * 1e6 / n as f64,
    );
    Ok(())
}
