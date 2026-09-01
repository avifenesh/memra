//! CUTLASS NVFP4 GEMV-shape timing (feasibility receipt for the tensor-core expert
//! family): times m in {1, 2, 8} over the step37 expert-sweep-equivalent widths and
//! prints us/call + achieved weight bandwidth. Compare against the dp4a sel_v2 receipts
//! (gate+up 22.4 MB in ~28.8 us = 0.78 TB/s; down 445 GB/s). W4A4 numeric class — this
//! binary is a SPEED receipt only, never a serving arm (owner ruling: W4A4 hurt
//! bit-exactness; any adoption would be W4A8-class and owner-gated).
#[cfg(not(memra_cutlass))]
fn main() {
    eprintln!("build with MEMRA_CUTLASS=1 on sm_120a");
}

#[cfg(memra_cutlass)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use memra_engine::Engine;
    let eng = Engine::new(0)?;
    // (m, n, k, label): n folds the 8-expert sweep into one stacked GEMV so the
    // tensor-core streaming rate is measured at the sweep's byte count.
    let shapes: &[(usize, usize, usize, &str)] = &[
        (1, 20480, 4096, "gate+up all-8-experts m=1"),
        (1, 4096, 5120, "down all-8-experts m=1 (k=8*640)"),
        (2, 20480, 4096, "gate+up m=2 (verify T=2)"),
        (8, 20480, 4096, "gate+up m=8 (batch B=8)"),
        (1, 6144, 4096, "qkv-class m=1 (bf16 comparison shape)"),
        // PREFILL-CHUNK shapes (2026-08-25): the t-row walk gave prefill only 1.21x
        // (193-token TTFT 2.976 -> 2.459 s) because it is GEMV-shaped. These rows are
        // the arithmetic-bound ceiling a chunked-GEMM prefill would run at, per layer:
        // dense QKV + o_proj at the chunk width, and one routed-expert tile (512 tokens
        // x top-8 / 288 experts ~ 14 rows per expert, expert_ff 1280 so gate+up = 2560).
        (512, 6144, 4096, "PREFILL qkv chunk m=512"),
        (512, 4096, 4096, "PREFILL o_proj chunk m=512"),
        (16, 2560, 4096, "PREFILL expert gate+up tile m=16"),
        (16, 4096, 1280, "PREFILL expert down tile m=16"),
        (512, 2560, 4096, "PREFILL expert gate+up dense-equiv m=512"),
        // TTFT lane (2026-08-27): the sub-second-cold sizing rows. A 4,092-token prime as ONE
        // chunk per matrix class — if these sustain tensor-core-class TFLOP/s the NVFP4 grouped
        // prefill clears the bar with room; if they sag toward the GEMV numbers the dequant-f16
        // GEMM route is the build instead. Speed receipts only (W4A4 class, never a serving arm).
        (4096, 2560, 4096, "PRIME-CHUNK expert gate+up m=4096"),
        (4096, 4096, 1280, "PRIME-CHUNK expert down m=4096"),
        (4096, 5152, 4096, "PRIME-CHUNK qkv+gate m=4096"),
    ];
    for &(m, n, k, label) in shapes {
        let us = time_shape(&eng, m, n, k)?;
        let bytes = (n * k) as f64 / 2.0 + (n * k) as f64 / 16.0;
        let tflops = 2.0 * m as f64 * n as f64 * k as f64 / us / 1e6;
        println!(
            "[{label}] m={m} n={n} k={k}: {us:.1} us/call  weightBW={:.2} TB/s  {tflops:.1} TFLOP/s",
            bytes / us / 1e6
        );
    }
    Ok(())
}

#[cfg(memra_cutlass)]
fn time_shape(
    eng: &memra_engine::Engine,
    m: usize,
    n: usize,
    k: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let a_h = vec![0.25f32; m * k];
    let b_h = vec![0.5f32; n * k];
    let a_d = eng.htod(&a_h)?;
    let b_d = eng.htod(&b_h)?;
    let mut a_packed = eng.alloc_u8(m * k / 2)?;
    let mut a_sf_lin = eng.alloc_u8(m * k / 16)?;
    let mut b_packed = eng.alloc_u8(n * k / 2)?;
    let mut b_sf_lin = eng.alloc_u8(n * k / 16)?;
    eng.cutlass_nvfp4_quant_ref(&a_d, &mut a_packed, &mut a_sf_lin, m, k)?;
    eng.cutlass_nvfp4_quant_ref(&b_d, &mut b_packed, &mut b_sf_lin, n, k)?;
    let sfa_bytes = eng.cutlass_sfa_size(m, k);
    let sfb_bytes = eng.cutlass_sfb_size(n, k);
    let mut a_sf_sw = eng.alloc_u8(sfa_bytes)?;
    let mut b_sf_sw = eng.alloc_u8(sfb_bytes)?;
    eng.cutlass_repack_sfa(&a_sf_lin, &mut a_sf_sw, m, k)?;
    eng.cutlass_repack_sfb(&b_sf_lin, &mut b_sf_sw, n, k)?;
    let alpha_d = eng.htod(&[1.0f32])?;
    let ws_bytes = eng.cutlass_fp4_workspace_size(m, n, k);
    let mut workspace = eng.alloc_u8(ws_bytes.max(1))?;
    let mut d_d = eng.htod(&vec![0f32; m * n])?;
    // Warm
    for _ in 0..20 {
        eng.cutlass_fp4_gemm_raw(
            &a_packed,
            &b_packed,
            &a_sf_sw,
            &b_sf_sw,
            &alpha_d,
            &mut d_d,
            m,
            n,
            k,
            &mut workspace,
        )?;
    }
    eng.stream().synchronize()?;
    let reps = 200;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        eng.cutlass_fp4_gemm_raw(
            &a_packed,
            &b_packed,
            &a_sf_sw,
            &b_sf_sw,
            &alpha_d,
            &mut d_d,
            m,
            n,
            k,
            &mut workspace,
        )?;
    }
    eng.stream().synchronize()?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / reps as f64)
}
