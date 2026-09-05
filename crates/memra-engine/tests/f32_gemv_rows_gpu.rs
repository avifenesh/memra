//! Gate for `MEMRA_F32_GEMV_KERNEL` (lane/f32-gemv-rows-20260905): the native f32 row GEMV
//! (`gemv_f32_rows`) against cuBLASLt (`linear`) at the DSA indexer's shapes (4096 -> 128 and
//! 4096 -> 32) and the verify tier (m = 4): NUMERIC CLASS, so the contract is tolerance
//! (|a-b| <= 1e-5 * (1 + |b|)) plus DETERMINISM (two launches bitwise equal) plus the M-IDENTITY
//! (row j of an m=4 launch is bitwise the m=1 launch on that row: verify == decode by
//! construction) plus a red arm (perturbed W moves the result) plus shape refusal (in_f % 1024 != 0
//! and m > 16 return Ok(false) and launch nothing) plus the door's non-vacuity (the counter moves
//! when `linear` runs under the door).
use memra_engine::Engine;
use std::sync::atomic::Ordering;

fn vecf(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.8
        })
        .collect()
}

fn run(
    e: &Engine,
    w: &[f32],
    x: &[f32],
    m: usize,
    in_f: usize,
    out_f: usize,
    native: bool,
) -> Vec<f32> {
    let xd = e.htod(x).unwrap();
    let wd = e.htod(w).unwrap();
    let mut y = e.uninit(m * out_f).unwrap();
    if native {
        assert!(
            e.gemv_f32_rows_into(&xd, &wd, &mut y, m, in_f, out_f)
                .unwrap()
        );
    } else {
        let yy = e.linear(&xd, &wd, m, in_f, out_f).unwrap();
        e.copy_into(&mut y, 0, &yy, m * out_f).unwrap();
    }
    e.stream().synchronize().unwrap();
    e.dtoh(&y).unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn gemv_f32_rows_matches_cublas_deterministic_and_m_identical() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    for (in_f, out_f) in [(4096usize, 128usize), (4096, 32), (8192, 24)] {
        let w = vecf(out_f * in_f, 31);
        let x = vecf(4 * in_f, 32);
        // m = 1 on row 0
        let a = run(&e, &w, &x[..in_f], 1, in_f, out_f, false);
        let b = run(&e, &w, &x[..in_f], 1, in_f, out_f, true);
        let c = run(&e, &w, &x[..in_f], 1, in_f, out_f, true);
        assert_eq!(
            bits(&b),
            bits(&c),
            "two native launches differ ({in_f}->{out_f})"
        );
        let mut worst = 0.0f32;
        for (p, q) in a.iter().zip(&b) {
            let tol = 1e-5 * (1.0 + q.abs());
            worst = worst.max((p - q).abs() / tol);
            assert!(
                (p - q).abs() <= tol,
                "cublas {p} vs native {q} ({in_f}->{out_f})"
            );
        }
        // m = 4: every row bitwise the m=1 launch on that row
        let m4 = run(&e, &w, &x, 4, in_f, out_f, true);
        for j in 0..4 {
            let one = run(&e, &w, &x[j * in_f..(j + 1) * in_f], 1, in_f, out_f, true);
            assert_eq!(
                bits(&m4[j * out_f..(j + 1) * out_f]),
                bits(&one),
                "row {j} of the m=4 launch is not the m=1 launch ({in_f}->{out_f})"
            );
        }
        // NOT compared: cuBLASLt's n>=2 path is a different algorithm from its m=1 dot pair
        // (memra's own probe: m=1 vs m=2 col-0 differ in every bit, maxdiff 3.5e-3, which is why
        // `linear_decode_exact` issues per-column m=1 calls). The m=1 tolerance above plus the
        // m-identity below bound every row of the m=4 launch against the m=1 reference.
        // red arm
        let mut w2 = w.clone();
        w2[in_f / 2] += 0.25;
        let r = run(&e, &w2, &x[..in_f], 1, in_f, out_f, true);
        assert_ne!(bits(&r), bits(&b), "perturbed W did not move row 0");
        println!("gemv_f32_rows {in_f}->{out_f}: worst |cublas-native|/tol = {worst:.3}");
    }
    // shape refusal
    let w = vecf(8 * 1000, 33);
    let x = vecf(1000, 34);
    let xd = e.htod(&x).unwrap();
    let wd = e.htod(&w).unwrap();
    let mut y = e.uninit(8).unwrap();
    assert!(!e.gemv_f32_rows_into(&xd, &wd, &mut y, 1, 1000, 8).unwrap());
    let w = vecf(8 * 1024, 35);
    let x = vecf(17 * 1024, 36);
    let xd = e.htod(&x).unwrap();
    let wd = e.htod(&w).unwrap();
    let mut y = e.uninit(17 * 8).unwrap();
    assert!(!e.gemv_f32_rows_into(&xd, &wd, &mut y, 17, 1024, 8).unwrap());
    // door non-vacuity through `linear`
    let c0 = memra_engine::F32_GEMV_KERNEL_DISPATCHES.load(Ordering::Relaxed);
    unsafe {
        std::env::set_var("MEMRA_F32_GEMV_KERNEL", "1");
    }
    let w = vecf(32 * 4096, 37);
    let x = vecf(4096, 38);
    let xd = e.htod(&x).unwrap();
    let wd = e.htod(&w).unwrap();
    let _ = e.linear(&xd, &wd, 1, 4096, 32).unwrap();
    unsafe {
        std::env::set_var("MEMRA_F32_GEMV_KERNEL", "0");
    }
    let c1 = memra_engine::F32_GEMV_KERNEL_DISPATCHES.load(Ordering::Relaxed);
    assert!(
        c1 > c0,
        "VACUOUS: `linear` under the door never took the kernel ({c0} -> {c1})"
    );
}
