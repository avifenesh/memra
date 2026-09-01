//! LAUNCH ECONOMICS (2026-08-26): what does one kernel launch cost this box, eager versus graph
//! replay? The step37 spec round is now GPU-optimal — verify-wait is 14.1 ms for two verify
//! columns against 12.16 ms for one plain decode token — and the whole remaining gap to plain is
//! 7.9 ms/round of HOST launch submission across ~1346 launches. Capturing the verify walk is a
//! large, multi-device build, so this prices the win BEFORE paying for it: if graph replay is not
//! materially cheaper per launch than eager submit on this silicon, the build is not worth it.
//!
//! Method: N launches of a real but trivial kernel on one stream, timed as pure host submit (the
//! stream is synchronized once, after the loop, so the measurement is issue cost plus the tail),
//! then the same N captured once and replayed. Reports us/launch for both.
//!
//! usage: launch-econ [n_launches]
use memra_engine::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(1346);
    let e = Engine::new(0)?;
    // A trivial-but-real kernel: 4096-element add. Small enough that GPU time is not the story.
    let a = e.htod(&vec![1.0f32; 4096])?;
    let b = e.htod(&vec![2.0f32; 4096])?;
    let mut y = e.htod(&vec![0.0f32; 4096])?;

    // warm
    for _ in 0..64 {
        e.add(&a, &b, &mut y, 4096)?;
    }
    e.stream().synchronize()?;

    let reps = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        for _ in 0..n {
            e.add(&a, &b, &mut y, 4096)?;
        }
    }
    e.stream().synchronize()?;
    let eager_us = t0.elapsed().as_secs_f64() * 1e6 / (reps * n) as f64;

    // Same work, captured once and replayed. CAPTURE-RETAIN keeps every allocation made during
    // warmup/capture alive, which is what makes capturing a walk that allocates legal at all.
    let (graph, _keep) = e.capture_graph_retained(|eng| {
        let mut yy = y.clone();
        for _ in 0..n {
            eng.add(&a, &b, &mut yy, 4096)?;
        }
        Ok(())
    })?;
    for _ in 0..4 {
        graph.launch()?;
    }
    e.stream().synchronize()?;
    let t1 = std::time::Instant::now();
    for _ in 0..reps {
        graph.launch()?;
    }
    e.stream().synchronize()?;
    let graph_us = t1.elapsed().as_secs_f64() * 1e6 / (reps * n) as f64;

    println!(
        "launches={n} eager={eager_us:.3} us/launch  graph-replay={graph_us:.3} us/launch  \
         ratio={:.2}x",
        eager_us / graph_us.max(1e-9)
    );
    println!(
        "predicted saving on a step37 verify round (1346 launches, 7.9 ms measured host issue): \
         {:.2} ms",
        (eager_us - graph_us) * 1346.0 / 1000.0
    );
    Ok(())
}
