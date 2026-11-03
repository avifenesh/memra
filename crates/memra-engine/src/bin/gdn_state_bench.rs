//! gdn-state-bench (bf16-state research arm): one decode step's batched GDN scans (48 layers of
//! 48 heads x 128 x 128 state per session, a separate allocation per layer and session) with the
//! state in fp32 and in bf16, at B = 1, 8, 16, 32. Prints ms per step and effective GB/s.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = memra_engine::Engine::new(0)?;
    let (heads, layers) = (48usize, 48usize);
    for b in [1usize, 8, 16, 32] {
        let f32ms = e.gdn_state_scan_bench(heads, b, false, layers, 9)?;
        let bf16ms = e.gdn_state_scan_bench(heads, b, true, layers, 9)?;
        let bytes32 = (b * layers * heads * 128 * 128 * 4 * 2) as f64;
        println!(
            "GDNSTATE B={b}: fp32 {f32ms:.2} ms ({:.0} GB/s)  bf16 {bf16ms:.2} ms ({:.0} GB/s state-equivalent bytes)  ratio {:.2}",
            bytes32 / (f32ms / 1e3) / 1e9,
            bytes32 / 2.0 / (bf16ms / 1e3) / 1e9,
            bf16ms / f32ms
        );
    }
    Ok(())
}
