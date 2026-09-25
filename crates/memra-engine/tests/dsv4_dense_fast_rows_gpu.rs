//! Kernel-boundary bit gate for the multi-row dense-fast FP8 GEMV (memra #710 B-row).
//!
//! At m = 2..8 the FP8 dense projection (B-row decode, verify rounds) runs
//! `dsv4_dense_fast_fp8_kernel<2, false, M>` instead of `dsv4_gemv_fp8_m_kernel<M>`. The claim:
//! every output bit equals the old m-row kernel's (dense-fast armed off) and each row's own
//! one-row launch, at the served shapes, with contiguous and strided activations and outputs.
//! Red arm: one token row's activations moved must move that row's outputs and no other row's.
//!
//! Rig law: correctness-only, one CUDA card under the rig's lock,
//! `-- --ignored --test-threads=1`.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_engine::Engine;
use memra_engine::dsv4_ffi as k;
use std::os::raw::c_void;

unsafe extern "C" {
    fn memra_dsv4_dense_fast_set_for_gate(enabled: i32) -> i32;
}

fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed | 1;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        s >> 16
    }
}

/// Finite e4m3 codes (no NaN encodings 0x7f / 0xff).
fn codes(n: usize, seed: u64) -> Vec<u8> {
    let mut r = lcg(seed);
    (0..n)
        .map(|_| {
            let c = (r() & 0xff) as u8;
            if c & 0x7f == 0x7f { c & 0xfe } else { c }
        })
        .collect()
}

/// Power-of-two block scales, one per (128 rows, 128 cols) block.
fn scales(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut r = lcg(seed);
    (0..rows.div_ceil(128) * cols.div_ceil(128))
        .map(|_| 2f32.powi((r() % 13) as i32 - 9))
        .collect()
}

fn bf16_rows(n: usize, seed: u64) -> Vec<u16> {
    let mut r = lcg(seed);
    (0..n)
        .map(|_| {
            let exp = 110 + (r() % 30) as u16;
            let sign = ((r() & 1) as u16) << 15;
            sign | (exp << 7) | (r() & 0x7f) as u16
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run(
    e: &Engine,
    w: &CudaSlice<u8>,
    sc: &CudaSlice<f32>,
    sc_cols: usize,
    x: &CudaSlice<u16>,
    m: usize,
    n: usize,
    kk: usize,
    xstride: usize,
    ystride: usize,
) -> Vec<u32> {
    let mut y: CudaSlice<f32> = e
        .htod(&vec![f32::from_bits(0x7fc0_4321); m * ystride])
        .unwrap();
    let st = e.stream();
    let rc = unsafe {
        k::memra_dsv4_gemv_fp8_m(
            w.device_ptr(&st).0 as *const c_void,
            sc.device_ptr(&st).0 as *const f32,
            sc_cols as i32,
            x.device_ptr(&st).0 as *const c_void,
            y.device_ptr_mut(&st).0 as *mut f32,
            m as i32,
            n as i32,
            kk as i32,
            xstride as i32,
            ystride as i32,
            st.cu_stream() as *mut c_void,
        )
    };
    assert_eq!(rc, 0, "dense launch m={m} n={n} k={kk}");
    e.dtoh(&y).unwrap().iter().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "needs a CUDA device; run under the rig's GPU lock"]
fn multi_row_dense_fast_is_the_m_row_gemv_and_the_one_row_launch_bit_for_bit() {
    let e = Engine::new(0).expect("CUDA engine on device 0");
    // (n, k) of the served TP/EP half shapes and the full ones, plus ragged widths.
    let shapes = [
        (1024, 4096),
        (512, 4096),
        (4096, 4096),
        (2048, 4096),
        (4096, 2048),
        (2048, 8192),
        (1027, 1024),
        (7, 2048),
        (1, 4096),
    ];
    let mut cases = 0;
    for (si, &(n, kk)) in shapes.iter().enumerate() {
        let w = codes(n * kk, 0xD5 ^ si as u64);
        let sc_cols = kk.div_ceil(128);
        let sc = scales(n, kk, 0x3C ^ si as u64);
        let w_dev: CudaSlice<u8> = e.stream().clone_htod(&w).unwrap();
        let sc_dev: CudaSlice<f32> = e.htod(&sc).unwrap();
        for m in 2..=8usize {
            for (xs, ys) in [(kk, n), (kk + 8 * 3, n + 5)] {
                let x = bf16_rows(m * xs, 0x91 ^ ((m as u64) << 8) ^ si as u64);
                let x_dev: CudaSlice<u16> = e.stream().clone_htod(&x).unwrap();
                let fast = run(&e, &w_dev, &sc_dev, sc_cols, &x_dev, m, n, kk, xs, ys);
                assert_eq!(unsafe { memra_dsv4_dense_fast_set_for_gate(0) }, 0);
                let old = run(&e, &w_dev, &sc_dev, sc_cols, &x_dev, m, n, kk, xs, ys);
                assert_eq!(unsafe { memra_dsv4_dense_fast_set_for_gate(1) }, 0);
                for t in 0..m {
                    let row: CudaSlice<u16> =
                        e.stream().clone_htod(&x[t * xs..t * xs + kk]).unwrap();
                    let one = run(&e, &w_dev, &sc_dev, sc_cols, &row, 1, n, kk, kk, n);
                    for r in 0..n {
                        let i = t * ys + r;
                        assert_eq!(
                            fast[i], old[i],
                            "n={n} k={kk} m={m} xs={xs} ys={ys} row {t} col {r}: dense-fast vs m-row"
                        );
                        assert_eq!(
                            fast[i], one[r],
                            "n={n} k={kk} m={m} row {t} col {r}: dense-fast vs one-row"
                        );
                    }
                }
                cases += 1;
            }
        }
    }
    println!("{cases} cases bit-identical (dense-fast m-row = m-row GEMV = one-row launch)");

    // Red arm: move token row 1's activations; row 1 must change and rows 0, 2 must not.
    let (n, kk, m) = (1024usize, 4096usize, 3usize);
    let w_dev: CudaSlice<u8> = e.stream().clone_htod(&codes(n * kk, 7)).unwrap();
    let sc_dev: CudaSlice<f32> = e.htod(&scales(n, kk, 8)).unwrap();
    let mut x = bf16_rows(m * kk, 9);
    let x_dev: CudaSlice<u16> = e.stream().clone_htod(&x).unwrap();
    let base = run(
        &e,
        &w_dev,
        &sc_dev,
        kk.div_ceil(128),
        &x_dev,
        m,
        n,
        kk,
        kk,
        n,
    );
    for v in &mut x[kk..2 * kk] {
        *v ^= 0x0001;
    }
    let x_dev: CudaSlice<u16> = e.stream().clone_htod(&x).unwrap();
    let moved = run(
        &e,
        &w_dev,
        &sc_dev,
        kk.div_ceil(128),
        &x_dev,
        m,
        n,
        kk,
        kk,
        n,
    );
    assert!(
        (0..n).any(|r| base[n + r] != moved[n + r]),
        "row 1 did not move"
    );
    for t in [0usize, 2] {
        assert!(
            (0..n).all(|r| base[t * n + r] == moved[t * n + r]),
            "row {t} moved with row 1"
        );
    }
    println!("red arm: row 1 moved alone");
}
