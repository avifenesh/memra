//! `rms_norm_zq8_f32_v2` (MEMRA_NORM_ILP) vs the v1 kernel, BITWISE on all three outputs
//! (z f32, q i8, d f32), at several ncols (a 3-round tail, a ragged pass-2 tail, the glm5
//! 4096, a 1024-block shape) and at both block widths the launcher can take
//! (`MEMRA_RMS_BLOCK`, a process latch: run once plain and once with `MEMRA_RMS_BLOCK=1024`).
//! The claim under test is the kernel header's "moves no bits" construction; a single
//! differing bit anywhere is a defect in the twin, not a tolerance question.
use memra_engine::Engine;

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static M: std::sync::Mutex<()> = std::sync::Mutex::new(());
    M.lock().unwrap_or_else(|p| p.into_inner())
}

/// Deterministic pseudo-random f32 in [-lo..hi] with a few large outliers so the q8 blocks
/// exercise both tiny and saturating scales.
fn lcg_rows(n: usize, seed: u64, outlier_every: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 40) as f32) / ((1u64 << 24) as f32);
            let v = u * 2.0 - 1.0;
            if outlier_every > 0 && i % outlier_every == 7 {
                v * 37.0
            } else {
                v
            }
        })
        .collect()
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn zq8_ilp_twin_is_bit_identical() {
    let _g = gpu_guard();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let nrows = 3usize;
    let mut checked = 0usize;
    for &ncols in &[96usize, 1536, 2048, 4096, 8192] {
        let xh = lcg_rows(nrows * ncols, 0x5EED_0000 + ncols as u64, 61);
        let wh = lcg_rows(ncols, 0xC0FFEE + ncols as u64, 0);
        let x = e.htod(&xh).expect("x");
        let w = e.htod(&wh).expect("w");
        let mut z1 = e.htod(&vec![0.0f32; nrows * ncols]).expect("z1");
        let mut z2 = e.htod(&vec![0.0f32; nrows * ncols]).expect("z2");
        let (q1, d1) = e
            .rms_norm_zq8_f32_arm(&x, &w, &mut z1, ncols, nrows, 1e-5, false)
            .expect("v1");
        let (q2, d2) = e
            .rms_norm_zq8_f32_arm(&x, &w, &mut z2, ncols, nrows, 1e-5, true)
            .expect("v2");
        let (z1h, z2h) = (e.dtoh(&z1).expect("z1 back"), e.dtoh(&z2).expect("z2 back"));
        let (q1h, q2h) = (
            e.dtoh_i8(&q1).expect("q1 back"),
            e.dtoh_i8(&q2).expect("q2 back"),
        );
        let (d1h, d2h) = (e.dtoh(&d1).expect("d1 back"), e.dtoh(&d2).expect("d2 back"));
        let zdiff = z1h
            .iter()
            .zip(&z2h)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let qdiff = q1h.iter().zip(&q2h).filter(|(a, b)| a != b).count();
        let ddiff = d1h
            .iter()
            .zip(&d2h)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        // Non-vacuity: the outputs must be real (a zero row on both arms would also "match").
        let nz = z1h.iter().filter(|v| **v != 0.0).count();
        let qnz = q1h.iter().filter(|v| **v != 0).count();
        assert!(
            nz > nrows * ncols / 2 && qnz > nrows * ncols / 4,
            "ncols={ncols}: vacuous outputs (z nonzero {nz}, q nonzero {qnz})"
        );
        assert_eq!(
            (zdiff, qdiff, ddiff),
            (0, 0, 0),
            "ncols={ncols}: rms_norm_zq8_f32_v2 differs from v1 (z {zdiff} of {}, q {qdiff}, d {ddiff} of {})",
            nrows * ncols,
            nrows * ncols / 32
        );
        checked += nrows * ncols;
    }
    println!(
        "[norm-zq8-ilp] bit-identical z/q/d over {checked} elements x 5 shapes (block=MEMRA_RMS_BLOCK or default)"
    );
}
