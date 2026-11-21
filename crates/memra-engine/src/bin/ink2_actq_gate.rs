//! ink2-actq-gate: the Inkling 2 Mini act-quant kernels against the reference and its goldens,
//! and (with `--bench`) their microbench at the model's shapes.
//!
//! Usage: `ink2-actq-gate [--bench [experts] [reps]] [--capture FILE] [--device N]`

use memra_engine::Engine;
use memra_engine::inkling2_actq::{ink2_actq_bench, ink2_actq_gate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let device = args
        .iter()
        .position(|a| a == "--device")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let e = Engine::new(device)?;
    let (lines, fails) = ink2_actq_gate(&e)?;
    for line in &lines {
        println!("{line}");
    }
    if let Some(i) = args.iter().position(|a| a == "--capture") {
        let path = args.get(i + 1).ok_or("--capture needs a path")?;
        std::fs::write(path, memra_engine::inkling2_actq::ink2_actq_capture(&e)?)?;
        println!("capture written to {path}");
    }
    if let Some(i) = args.iter().position(|a| a == "--bench") {
        let experts = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(128);
        let reps = args.get(i + 2).and_then(|v| v.parse().ok()).unwrap_or(200);
        for line in ink2_actq_bench(&e, experts, reps)? {
            println!("{line}");
        }
    }
    if fails != 0 {
        return Err(format!("{fails} ink2 act-quant cell(s) FAILED").into());
    }
    println!("ink2-actq-gate: ALL OK ({} lines)", lines.len());
    Ok(())
}
