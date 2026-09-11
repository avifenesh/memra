//! PP-2 port gate for the three norm doors. Kernel contract only: no model, no
//! tokenizer, no server. It answers one question, which is the question the port
//! turns on: at every routed row count the served program presents, does the
//! fused arm produce the SAME BITS as the arm it replaces?
//!
//! Run it before any perf cell. A perf number from an arm that moved a bit is
//! not a perf number, it is a different model.
use memra_engine::dsv4_gpu::Dsv4Gpu;

fn main() {
    let devices: Vec<usize> = match std::env::args().nth(1) {
        Some(arg) => arg
            .split(',')
            .map(|s| s.parse().expect("device index"))
            .collect(),
        None => vec![0],
    };
    for device in devices {
        Dsv4Gpu::run_norm_pp2_port_gate_for_gate(device).unwrap();
    }
    println!("PASS dsv4-norm-pp2-port-gate");
}
