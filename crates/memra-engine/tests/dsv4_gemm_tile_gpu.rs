//! Kernel-boundary bit gate for the prefill dense tile (memra #472, #700).
//!
//! At m > 32 the FP8 dense projection runs `dsv4_gemm_fp8_tile_kernel` instead of looping the
//! 32-row GEMV. The claim: every output bit equals the GEMV loop's, at the prefill widths and
//! shapes the served program uses, with contiguous and strided activations and outputs.
//! Red arm: one weight row's codes moved must move that output row in every token row and no
//! other row.
//!
//! Rig law: correctness-only, one CUDA card under the rig's lock,
//! `-- --ignored --test-threads=1`.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_engine::Engine;
use memra_engine::dsv4_ffi as k;
use std::os::raw::c_void;

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
fn prefill_dense_tile_is_the_gemv_loop_bit_for_bit() {
    let e = Engine::new(0).expect("CUDA engine on device 0");
    // (n, k) of the served projections: wq_a, wq_b, wkv, wo_b, the shared expert, the
    // indexer weights, the head, and ragged widths.
    let shapes = [
        (1024, 4096),
        (32768, 1024),
        (512, 4096),
        (4096, 8192),
        (2048, 4096),
        (4096, 2048),
        (64, 4096),
        (1027, 1024),
        (7, 2048),
    ];
    let widths = [33usize, 48, 64, 100, 256, 512];
    let mut cases = 0;
    for tile_shape in 0..4 {
        unsafe { k::memra_dsv4_gemm_fp8_tile_shape_set_for_gate(tile_shape) };
        for (si, &(n, kk)) in shapes.iter().enumerate() {
            let w = codes(n * kk, 0xF8 ^ si as u64);
            let sc_cols = kk.div_ceil(128);
            let sc = scales(n, kk, 0x5C ^ si as u64);
            let w_dev: CudaSlice<u8> = e.stream().clone_htod(&w).unwrap();
            let sc_dev: CudaSlice<f32> = e.htod(&sc).unwrap();
            for &m in &widths {
                if n * m > 32768 * 256 {
                    continue; // the head-sized shapes at the widest widths add time, not coverage
                }
                for (xs, ys) in [(kk, n), (kk + 8 * 3, n + 5)] {
                    let x = bf16_rows(m * xs, 0xA7 ^ ((m as u64) << 8) ^ si as u64);
                    let x_dev: CudaSlice<u16> = e.stream().clone_htod(&x).unwrap();
                    let prev = unsafe { k::memra_dsv4_gemm_fp8_tile_set_for_gate(1) };
                    let before = unsafe { k::memra_dsv4_gemm_fp8_tile_launches() };
                    let tile = run(&e, &w_dev, &sc_dev, sc_cols, &x_dev, m, n, kk, xs, ys);
                    assert!(
                        unsafe { k::memra_dsv4_gemm_fp8_tile_launches() } > before,
                        "tile did not engage at m={m}"
                    );
                    unsafe { k::memra_dsv4_gemm_fp8_tile_set_for_gate(0) };
                    let gemv = run(&e, &w_dev, &sc_dev, sc_cols, &x_dev, m, n, kk, xs, ys);
                    unsafe { k::memra_dsv4_gemm_fp8_tile_set_for_gate(prev) };
                    for t in 0..m {
                        for r in 0..n {
                            let i = t * ys + r;
                            assert_eq!(
                                tile[i], gemv[i],
                                "m={m} n={n} k={kk} xstride={xs} ystride={ys}: y[{t}][{r}] tile {:#010x} gemv {:#010x}",
                                tile[i], gemv[i]
                            );
                            assert_ne!(tile[i], 0x7fc0_4321, "unwritten y[{t}][{r}]");
                        }
                        // The strided gaps stay untouched.
                        for r in n..ys {
                            assert_eq!(tile[t * ys + r], 0x7fc0_4321, "gap y[{t}][{r}] written");
                        }
                    }
                    cases += 1;
                }
            }
        }
    }
    unsafe { k::memra_dsv4_gemm_fp8_tile_shape_set_for_gate(0) };
    // Red arm: move output row 5's codes by one step; row 5 moves in every token row, row 4 in none.
    let (n, kk, m) = (1024usize, 4096usize, 64usize);
    let mut w = codes(n * kk, 0xF8);
    let sc_cols = kk.div_ceil(128);
    let sc_dev: CudaSlice<f32> = e.htod(&scales(n, kk, 0x5C)).unwrap();
    let x_dev: CudaSlice<u16> = e.stream().clone_htod(&bf16_rows(m * kk, 0xA7)).unwrap();
    let clean_dev: CudaSlice<u8> = e.stream().clone_htod(&w).unwrap();
    let clean = run(&e, &clean_dev, &sc_dev, sc_cols, &x_dev, m, n, kk, kk, n);
    for c in &mut w[5 * kk..6 * kk] {
        *c = if *c & 0x7f == 0x7e { *c - 1 } else { *c + 1 };
    }
    let red_dev: CudaSlice<u8> = e.stream().clone_htod(&w).unwrap();
    let red = run(&e, &red_dev, &sc_dev, sc_cols, &x_dev, m, n, kk, kk, n);
    for t in 0..m {
        assert_ne!(
            red[t * n + 5],
            clean[t * n + 5],
            "red arm: row 5 did not move at t={t}"
        );
        assert_eq!(
            red[t * n + 4],
            clean[t * n + 4],
            "red arm: row 4 moved at t={t}"
        );
    }
    println!(
        "DSV4_DENSE_TILE EXACT cases={cases} tile_shapes=4 shapes={} widths={widths:?} red_arm=1",
        shapes.len()
    );
}

fn f32_rows(n: usize, seed: u64) -> Vec<f32> {
    let mut r = lcg(seed);
    (0..n)
        .map(|_| ((r() & 0xffff) as f32 / 65535.0 - 0.5) * 2f32.powi((r() % 9) as i32 - 4))
        .collect()
}

fn run_dots(
    e: &Engine,
    x: &CudaSlice<f32>,
    w: *const c_void,
    bf16: bool,
    s: usize,
    kk: usize,
    n: usize,
) -> Vec<u32> {
    let mut y: CudaSlice<f32> = e.htod(&vec![f32::from_bits(0x7fc0_4321); s * n]).unwrap();
    let st = e.stream();
    let rc = unsafe {
        k::memra_dsv4_dots_f32acc_mrow(
            x.device_ptr(&st).0 as *const f32,
            w,
            i32::from(bf16),
            y.device_ptr_mut(&st).0 as *mut f32,
            s as i32,
            kk as i32,
            n as i32,
            st.cu_stream() as *mut c_void,
        )
    };
    assert_eq!(rc, 0, "dots launch s={s} n={n} k={kk}");
    e.dtoh(&y).unwrap().iter().map(|v| v.to_bits()).collect()
}

/// The compressor projections' f32-island dots at prefill widths: the tile against the 32-row
/// loop, BF16 and f32 weight storage.
#[test]
#[ignore = "needs a CUDA device; run under the rig's GPU lock"]
fn prefill_dots_tile_is_the_mrow_loop_bit_for_bit() {
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let mut cases = 0;
    for (si, &(n, kk)) in [
        (1024usize, 4096usize),
        (512, 4096),
        (256, 4096),
        (2048, 4096),
        (37, 1024),
    ]
    .iter()
    .enumerate()
    {
        let wb = bf16_rows(n * kk, 0xB1 ^ si as u64);
        let wf: Vec<f32> = wb
            .iter()
            .map(|&b| f32::from_bits(u32::from(b) << 16))
            .collect();
        let wb_dev: CudaSlice<u16> = e.stream().clone_htod(&wb).unwrap();
        let wf_dev: CudaSlice<f32> = e.htod(&wf).unwrap();
        let st = e.stream();
        for &s in &[33usize, 64, 100, 512] {
            let x: CudaSlice<f32> = e
                .htod(&f32_rows(s * kk, 0xD0 ^ ((s as u64) << 4) ^ si as u64))
                .unwrap();
            for (bf16, w) in [
                (true, wb_dev.device_ptr(&st).0 as *const c_void),
                (false, wf_dev.device_ptr(&st).0 as *const c_void),
            ] {
                let prev = unsafe { k::memra_dsv4_gemm_fp8_tile_set_for_gate(1) };
                let tile = run_dots(&e, &x, w, bf16, s, kk, n);
                unsafe { k::memra_dsv4_gemm_fp8_tile_set_for_gate(0) };
                let loopd = run_dots(&e, &x, w, bf16, s, kk, n);
                unsafe { k::memra_dsv4_gemm_fp8_tile_set_for_gate(prev) };
                if let Some(i) = (0..tile.len()).find(|&i| tile[i] != loopd[i]) {
                    panic!(
                        "dots s={s} n={n} k={kk} bf16={bf16}: y[{}][{}] tile {:#010x} loop {:#010x}",
                        i / n,
                        i % n,
                        tile[i],
                        loopd[i]
                    );
                }
                assert!(tile.iter().all(|&b| b != 0x7fc0_4321), "unwritten output");
                cases += 1;
            }
        }
    }
    println!("DSV4_DOTS_TILE EXACT cases={cases}");
}

/// Device time of one 512-row prefill projection, tile against the GEMV loop, per shape. The output
/// is allocated once and read back never; only back-to-back launches are timed.
#[test]
#[ignore = "needs a CUDA device; timing only"]
fn prefill_dense_tile_timing() {
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let m = 512usize;
    for &(n, kk) in &[
        (1024usize, 4096usize),
        (32768, 1024),
        (4096, 8192),
        (512, 4096),
        (2048, 4096),
    ] {
        let sc_cols = kk.div_ceil(128);
        let w_dev: CudaSlice<u8> = e.stream().clone_htod(&codes(n * kk, 1)).unwrap();
        let sc_dev: CudaSlice<f32> = e.htod(&scales(n, kk, 2)).unwrap();
        let x_dev: CudaSlice<u16> = e.stream().clone_htod(&bf16_rows(m * kk, 3)).unwrap();
        let mut y: CudaSlice<f32> = e.htod(&vec![0f32; m * n]).unwrap();
        let st = e.stream();
        let launch = |y: &mut CudaSlice<f32>| {
            let rc = unsafe {
                k::memra_dsv4_gemv_fp8_m(
                    w_dev.device_ptr(&st).0 as *const c_void,
                    sc_dev.device_ptr(&st).0 as *const f32,
                    sc_cols as i32,
                    x_dev.device_ptr(&st).0 as *const c_void,
                    y.device_ptr_mut(&st).0 as *mut f32,
                    m as i32,
                    n as i32,
                    kk as i32,
                    kk as i32,
                    n as i32,
                    st.cu_stream() as *mut c_void,
                )
            };
            assert_eq!(rc, 0);
        };
        for (rep, tile_shape) in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 0)] {
            unsafe { k::memra_dsv4_gemm_fp8_tile_shape_set_for_gate(tile_shape) };
            let mut ms = [0f64; 2];
            for (arm, on) in [(0usize, 1i32), (1, 0)] {
                let prev = unsafe { k::memra_dsv4_gemm_fp8_tile_set_for_gate(on) };
                launch(&mut y);
                st.synchronize().unwrap();
                let t0 = std::time::Instant::now();
                for _ in 0..20 {
                    launch(&mut y);
                }
                st.synchronize().unwrap();
                ms[arm] = t0.elapsed().as_secs_f64() * 1e3 / 20.0;
                unsafe { k::memra_dsv4_gemm_fp8_tile_set_for_gate(prev) };
            }
            let tflops = 2.0 * (m * n * kk) as f64 / (ms[0] * 1e-3) / 1e12;
            println!(
                "TIMING dense m={m} n={n} k={kk} rep={rep} tile_shape={tile_shape} tile_ms={:.3} gemv_ms={:.3} speedup={:.2} tile_tflops={tflops:.1}",
                ms[0],
                ms[1],
                ms[1] / ms[0]
            );
        }
    }
}
