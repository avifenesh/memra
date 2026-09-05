//! Gate for `MEMRA_HC_MIXES_KERNEL` (lane/hc-mixes-gemv-20260905): the native hc mixes GEMV
//! against cuBLASLt (`linear_t1_into`) at the served shape (24 x 16384): NUMERIC CLASS, so the
//! contract is tolerance (|a-b| <= 1e-5 * (1 + |b|)) plus DETERMINISM (two launches bitwise
//! equal) plus a red arm (perturbed W moves the result), plus the refusal of a shape that does
//! not fit (in_f != 16384 returns Ok(false) and launches nothing).
use memra_engine::Engine;

const ROWS: usize = 24;
const IN_F: usize = 16384;

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

fn run(e: &Engine, w: &[f32], x: &[f32], native: bool) -> Vec<f32> {
    let xd = e.htod(x).unwrap();
    let wd = e.htod(w).unwrap();
    let mut y = e.uninit(ROWS).unwrap();
    let xv = xd.slice(0..IN_F);
    let wv = wd.slice(0..ROWS * IN_F);
    let mut yv = y.slice_mut(0..ROWS);
    if native {
        assert!(e.hc_mixes_gemv_into(&xv, &wv, &mut yv, IN_F, ROWS).unwrap());
    } else {
        e.linear_t1_into(&xv, &wv, &mut yv, IN_F, ROWS).unwrap();
    }
    e.stream().synchronize().unwrap();
    e.dtoh(&y).unwrap()
}

#[test]
fn hc_mixes_gemv_matches_cublas_within_tolerance_and_is_deterministic() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let w = vecf(ROWS * IN_F, 21);
    let x = vecf(IN_F, 22);
    let a = run(&e, &w, &x, false);
    let b = run(&e, &w, &x, true);
    let c = run(&e, &w, &x, true);
    assert_eq!(
        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "two native launches differ: the reduction is not deterministic"
    );
    let mut worst = 0.0f32;
    for (p, q) in a.iter().zip(&b) {
        let tol = 1e-5 * (1.0 + q.abs());
        worst = worst.max((p - q).abs() / tol);
        assert!((p - q).abs() <= tol, "cublas {p} vs native {q}");
    }
    assert!(
        a.iter().any(|v| v.abs() > 1e-3),
        "vacuous: cuBLAS output is ~zero"
    );
    eprintln!("worst |cublas - native| / tol = {worst:.3}");
    // red arm
    let mut w2 = w.clone();
    w2[5] += 1.0;
    let d = run(&e, &w2, &x, true);
    assert!(
        d[0] != b[0],
        "red arm: a perturbed W row 0 did not move y[0]"
    );
}

#[test]
fn hc_mixes_gemv_refuses_a_shape_it_cannot_schedule() {
    let Ok(e) = Engine::new(0) else {
        return;
    };
    let in_f = 4096;
    let xd = e.htod(&vecf(in_f, 1)).unwrap();
    let wd = e.htod(&vecf(ROWS * in_f, 2)).unwrap();
    let mut y = e.uninit(ROWS).unwrap();
    let xv = xd.slice(0..in_f);
    let wv = wd.slice(0..ROWS * in_f);
    let mut yv = y.slice_mut(0..ROWS);
    assert!(!e.hc_mixes_gemv_into(&xv, &wv, &mut yv, in_f, ROWS).unwrap());
}
