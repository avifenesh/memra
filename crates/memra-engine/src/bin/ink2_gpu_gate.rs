//! ink2-gpu-gate: the kernels of the Inkling 2 Mini GPU route (`cu/inkling2_gpu.cu`) against the
//! f64 reference operations. The same cells run inside `kernel-check`.
//!
//! Usage: `ink2-gpu-gate [--device N]`

use memra_engine::inkling2_gpu::{Dev, ink2g_gate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let device = args
        .iter()
        .position(|a| a == "--device")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let dev = Dev::new(device)?;
    let (lines, fails) = ink2g_gate(&dev)?;
    for line in &lines {
        println!("{line}");
    }
    if fails != 0 {
        return Err(format!("{fails} ink2 GPU route cell(s) FAILED").into());
    }
    println!("ink2-gpu-gate: ALL OK ({} cells)", lines.len());
    Ok(())
}
