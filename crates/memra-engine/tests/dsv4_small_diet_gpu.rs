//! Kernel-boundary bit gate for the DSV4 small-kernel diet (#339), which is the code on every
//! f32x device program of the HC4 / hidden 4096 shape, PP-2 and TP/EP alike.
//!
//! The diet replaces two unfused chains on every row count (plain t=1 steps, verify rows,
//! prefill rows):
//!
//! - HC finish: `memra_dsv4_rowsq_scale_f32acc` + `memra_dsv4_hc_sinkhorn_m` +
//!   `memra_dsv4_hc_collapse` becomes `memra_dsv4_small_hc_f32_fixed_order`.
//! - Q-LoRA norm/pack: `memra_dsv4_rmsnorm_f32acc` (in place) + `memra_dsv4_cvt_bf16` becomes
//!   `memra_dsv4_small_norm_pack_f32_fixed_order`.
//!
//! The claim is bit identity, not a band: every output (scaled mixes, pre, post, comb, y;
//! normalized f32 row, bf16 pack) must compare `to_bits`-equal, and a row of a multi-row launch
//! must carry the bits of the same row launched alone, so a request that alternates plain steps
//! and verify transactions stays one numeric program. The fixtures sweep magnitudes over many decades so the rowsq tree,
//! the Sinkhorn divisions and the collapse sums see rounding in every exponent range. The live
//! checkpoint twin of this gate is `dsv4_tp_ep_sampled_perf_gate --small-kernel-components`.
//!
//! Red arms: a 2^-10 relative change to one scale or one weight must be caught by the
//! comparator on the output it feeds, so a PASS cannot come from comparing a buffer against
//! itself.
//!
//! Rig law: correctness-only, run under `flock /tmp/memra-5090.lock`, TF32 forced off,
//! `-- --ignored --test-threads=1`.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_engine::Engine;
use memra_engine::dsv4_ffi as k;
use std::os::raw::c_void;

const HC: usize = 4;
const D: usize = 4096;
const ROWS: usize = (2 + HC) * HC;
const ITERS: i32 = 20; // DeepSeek-V4-Flash hc_sinkhorn_iters
const HC_EPS: f32 = 1e-6; // DeepSeek-V4-Flash hc_eps
const RMS_EPS: f32 = 1e-6; // DeepSeek-V4-Flash rms_norm_eps

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn force_true_f32() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
            // SAFETY: no CUDA call has been made in this process yet, and call_once
            // serializes every test thread behind this write.
            unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
        }
    });
}

/// Deterministic values in [-1, 1) times a per-element power of two in 2^-lo .. 2^hi.
fn fixture(n: usize, seed: u64, lo: i32, hi: i32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = ((s >> 33) as u32 as f32) / (u32::MAX as f32 / 2.0) - 1.0;
            let e = lo + ((s >> 13) % (hi - lo + 1) as u64) as i32;
            u * 2f32.powi(e)
        })
        .collect()
}

fn dp(s: &CudaSlice<f32>, e: &Engine) -> *const f32 {
    s.device_ptr(&e.stream()).0 as *const f32
}
fn dpm(s: &mut CudaSlice<f32>, e: &Engine) -> *mut f32 {
    s.device_ptr_mut(&e.stream()).0 as *mut f32
}
fn sv(e: &Engine) -> *mut c_void {
    e.stream().cu_stream() as *mut c_void
}

fn assert_bits_eq(what: &str, a: &[u32], b: &[u32], case: &str) {
    assert_eq!(a.len(), b.len(), "{case}: {what} length");
    if let Some(i) = (0..a.len()).find(|&i| a[i] != b[i]) {
        panic!(
            "{case}: {what}[{i}] fused {:#010x} != unfused {:#010x}",
            a[i], b[i]
        );
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

struct HcCase {
    s: usize,
    x: Vec<f32>,
    mixes: Vec<f32>,
    scale: Vec<f32>,
    base: Vec<f32>,
}

/// Output order: scaled mixes, pre, post, comb, y.
type HcOut = [Vec<u32>; 5];

fn hc_unfused(e: &Engine, c: &HcCase) -> HcOut {
    let s = c.s;
    let x = e.htod(&c.x).unwrap();
    let mut mixes = e.htod(&c.mixes).unwrap();
    let scale = e.htod(&c.scale).unwrap();
    let base = e.htod(&c.base).unwrap();
    let mut pre = e.uninit(s * HC).unwrap();
    let mut post = e.uninit(s * HC).unwrap();
    let mut comb = e.uninit(s * HC * HC).unwrap();
    let mut y = e.uninit(s * D).unwrap();
    unsafe {
        let rc = k::memra_dsv4_rowsq_scale_f32acc(
            dp(&x, e),
            dpm(&mut mixes, e),
            s as i32,
            (HC * D) as i32,
            ROWS as i32,
            HC_EPS,
            sv(e),
        );
        assert_eq!(rc, 0, "rowsq_scale_f32acc");
        let rc = k::memra_dsv4_hc_sinkhorn_m(
            dp(&mixes, e),
            dp(&scale, e),
            dp(&base, e),
            dpm(&mut pre, e),
            dpm(&mut post, e),
            dpm(&mut comb, e),
            s as i32,
            HC as i32,
            ITERS,
            HC_EPS,
            sv(e),
        );
        assert_eq!(rc, 0, "hc_sinkhorn_m");
        let rc = k::memra_dsv4_hc_collapse(
            dp(&x, e),
            dp(&pre, e),
            dpm(&mut y, e),
            s as i32,
            HC as i32,
            D as i32,
            sv(e),
        );
        assert_eq!(rc, 0, "hc_collapse");
    }
    [&mixes, &pre, &post, &comb, &y].map(|b| bits(&e.dtoh(b).unwrap()))
}

fn hc_fused(e: &Engine, c: &HcCase) -> HcOut {
    let s = c.s;
    let x = e.htod(&c.x).unwrap();
    let mut mixes = e.htod(&c.mixes).unwrap();
    let scale = e.htod(&c.scale).unwrap();
    let base = e.htod(&c.base).unwrap();
    let mut pre = e.uninit(s * HC).unwrap();
    let mut post = e.uninit(s * HC).unwrap();
    let mut comb = e.uninit(s * HC * HC).unwrap();
    let mut y = e.uninit(s * D).unwrap();
    unsafe {
        let rc = k::memra_dsv4_small_hc_f32_fixed_order(
            dp(&x, e),
            dpm(&mut mixes, e),
            dp(&scale, e),
            dp(&base, e),
            dpm(&mut pre, e),
            dpm(&mut post, e),
            dpm(&mut comb, e),
            dpm(&mut y, e),
            s as i32,
            HC as i32,
            D as i32,
            ITERS,
            HC_EPS,
            sv(e),
        );
        assert_eq!(rc, 0, "small_hc_f32_fixed_order");
    }
    [&mixes, &pre, &post, &comb, &y].map(|b| bits(&e.dtoh(b).unwrap()))
}

fn hc_case(seed: u64, x_range: (i32, i32), mix_range: (i32, i32)) -> HcCase {
    HcCase {
        s: 1,
        x: fixture(HC * D, seed ^ 0x11, x_range.0, x_range.1),
        mixes: fixture(ROWS, seed ^ 0x22, mix_range.0, mix_range.1),
        scale: fixture(3, seed ^ 0x33, -1, 2),
        base: fixture(ROWS, seed ^ 0x44, -3, 1),
    }
}

/// `s` positions stacked row-major, each drawn as its own single-row case with its own
/// magnitude ranges, so one launch mixes exponent regimes across its rows.
fn hc_rows(seed: u64, s: usize) -> (HcCase, Vec<HcCase>) {
    const X: [(i32, i32); 4] = [(-12, -6), (-4, 2), (0, 6), (-10, 8)];
    const MIX: [(i32, i32); 3] = [(-6, -1), (-2, 3), (2, 6)];
    let one: Vec<HcCase> = (0..s)
        .map(|p| {
            let mut c = hc_case(seed ^ ((p as u64) << 20), X[p % 4], MIX[p % 3]);
            // The gate scale is per layer, shared by every row of the launch.
            c.scale = fixture(3, seed ^ 0x33, -1, 2);
            c.base = fixture(ROWS, seed ^ 0x44, -3, 1);
            c
        })
        .collect();
    let all = HcCase {
        s,
        x: one.iter().flat_map(|c| c.x.iter().copied()).collect(),
        mixes: one.iter().flat_map(|c| c.mixes.iter().copied()).collect(),
        scale: one[0].scale.clone(),
        base: one[0].base.clone(),
    };
    (all, one)
}

/// Output order: normalized f32 row, bf16 pack (as u16 widened to u32).
fn norm_unfused(e: &Engine, x0: &[f32], w: &[f32]) -> [Vec<u32>; 2] {
    let s = x0.len() / w.len();
    let n = w.len();
    let mut x = e.htod(x0).unwrap();
    let wd = e.htod(w).unwrap();
    let stream = e.stream();
    let mut packed = stream.alloc_zeros::<u16>(s * n).unwrap();
    unsafe {
        let xp = dpm(&mut x, e);
        let rc =
            k::memra_dsv4_rmsnorm_f32acc(xp, dp(&wd, e), xp, s as i32, n as i32, RMS_EPS, sv(e));
        assert_eq!(rc, 0, "rmsnorm_f32acc");
        let rc = k::memra_dsv4_cvt_bf16(
            dp(&x, e),
            packed.device_ptr_mut(&stream).0 as *mut c_void,
            (s * n) as i64,
            sv(e),
        );
        assert_eq!(rc, 0, "cvt_bf16");
    }
    let p = stream.clone_dtoh(&packed).unwrap();
    [
        bits(&e.dtoh(&x).unwrap()),
        p.into_iter().map(u32::from).collect(),
    ]
}

fn norm_fused(e: &Engine, x0: &[f32], w: &[f32]) -> [Vec<u32>; 2] {
    let s = x0.len() / w.len();
    let n = w.len();
    let mut x = e.htod(x0).unwrap();
    let wd = e.htod(w).unwrap();
    let stream = e.stream();
    let mut packed = stream.alloc_zeros::<u16>(s * n).unwrap();
    unsafe {
        let rc = k::memra_dsv4_small_norm_pack_f32_fixed_order(
            dpm(&mut x, e),
            dp(&wd, e),
            packed.device_ptr_mut(&stream).0 as *mut c_void,
            s as i32,
            n as i32,
            RMS_EPS,
            sv(e),
        );
        assert_eq!(rc, 0, "small_norm_pack_f32_fixed_order");
    }
    let p = stream.clone_dtoh(&packed).unwrap();
    [
        bits(&e.dtoh(&x).unwrap()),
        p.into_iter().map(u32::from).collect(),
    ]
}

#[test]
#[ignore = "needs a CUDA device; run under flock /tmp/memra-5090.lock"]
fn dsv4_small_hc_diet_is_bit_identical_to_the_unfused_chain() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let mut cases = 0;
    // Residual-stream magnitudes from tiny to large, mixes from near-zero logits to saturating.
    for (xi, &x_range) in [(-12, -6), (-4, 2), (0, 6), (-10, 8)].iter().enumerate() {
        for (mi, &mix_range) in [(-6, -1), (-2, 3), (2, 6)].iter().enumerate() {
            for rep in 0..8u64 {
                let seed = 0xD54F_0000 ^ ((xi as u64) << 16) ^ ((mi as u64) << 8) ^ rep;
                let c = hc_case(seed, x_range, mix_range);
                let case = format!("hc x2^{x_range:?} mix2^{mix_range:?} seed {seed:#x}");
                let (u, f) = (hc_unfused(&e, &c), hc_fused(&e, &c));
                for (i, what) in ["mixes", "pre", "post", "comb", "y"].iter().enumerate() {
                    assert_bits_eq(what, &f[i], &u[i], &case);
                }
                cases += 1;
            }
        }
    }
    // Red arm: a 2^-10 relative change on each gate scale must reach the outputs it feeds.
    let c = hc_case(0xBAD5, (-4, 2), (-2, 3));
    let clean = hc_unfused(&e, &c);
    for (s, fed) in [(0usize, [1usize, 4]), (1, [2, 2]), (2, [3, 3])] {
        let mut red = hc_case(0xBAD5, (-4, 2), (-2, 3));
        red.scale[s] = f32::from_bits(red.scale[s].to_bits() + (1 << 13));
        let f = hc_fused(&e, &red);
        assert!(
            fed.iter().any(|&o| f[o] != clean[o]),
            "red arm: scale[{s}] moved and every fed output stayed bit-equal"
        );
    }
    println!("DSV4_SMALL_HC_DIET EXACT cases={cases} outputs=5 red_arms=3");
}

#[test]
#[ignore = "needs a CUDA device; run under flock /tmp/memra-5090.lock"]
fn dsv4_small_norm_pack_diet_is_bit_identical_to_the_unfused_pair() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let mut cases = 0;
    // 1024 is the DeepSeek-V4-Flash Q-LoRA rank; the others cover tail loops and the 8-wide
    // unrolled body boundary on both sides.
    for &n in &[1024usize, 1536, 512, 4096, 1023, 1025, 129, 7] {
        for (ri, &range) in [(-14, -8), (-4, 3), (2, 9), (-12, 10)].iter().enumerate() {
            for rep in 0..4u64 {
                let seed = 0x0A0B_0000 ^ ((n as u64) << 12) ^ ((ri as u64) << 4) ^ rep;
                let x = fixture(n, seed ^ 0x55, range.0, range.1);
                let w = fixture(n, seed ^ 0x66, -2, 1);
                let case = format!("norm n={n} x2^{range:?} seed {seed:#x}");
                let (u, f) = (norm_unfused(&e, &x, &w), norm_fused(&e, &x, &w));
                assert_bits_eq("row", &f[0], &u[0], &case);
                assert_bits_eq("pack", &f[1], &u[1], &case);
                cases += 1;
            }
        }
    }
    // Red arm: a 2^-10 relative weight change must reach the normalized row.
    let x = fixture(1024, 0xBAD6, -4, 3);
    let w = fixture(1024, 0xBAD7, -2, 1);
    let clean = norm_unfused(&e, &x, &w);
    let mut wr = w.clone();
    wr[17] = f32::from_bits(wr[17].to_bits() + (1 << 13));
    let red = norm_fused(&e, &x, &wr);
    assert_ne!(
        red[0], clean[0],
        "red arm: w[17] moved and the row stayed bit-equal"
    );
    println!("DSV4_SMALL_NORM_PACK_DIET EXACT cases={cases} outputs=2 red_arms=1");
}

/// Row counts: DSpark verify (T=6), the 16-row prime segment, tails on both sides, and one
/// prefill-sized chunk.
const MROWS: [usize; 6] = [2, 6, 7, 16, 17, 64];

#[test]
#[ignore = "needs a CUDA device; run under flock /tmp/memra-5090.lock"]
fn dsv4_small_hc_diet_multirow_is_bit_identical_to_the_unfused_chain_and_to_each_row_alone() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let (mut cases, mut rows) = (0, 0);
    for &s in &MROWS {
        for rep in 0..4u64 {
            let seed = 0xD54F_A000 ^ ((s as u64) << 8) ^ rep;
            let (all, one) = hc_rows(seed, s);
            let case = format!("hc s={s} seed {seed:#x}");
            let (u, f) = (hc_unfused(&e, &all), hc_fused(&e, &all));
            for (i, what) in ["mixes", "pre", "post", "comb", "y"].iter().enumerate() {
                assert_bits_eq(what, &f[i], &u[i], &case);
            }
            // Each row of the launch against the same row as a plain t=1 step.
            for (p, c) in one.iter().enumerate() {
                let alone = hc_fused(&e, c);
                for (i, what) in ["mixes", "pre", "post", "comb", "y"].iter().enumerate() {
                    let w = alone[i].len();
                    assert_bits_eq(
                        what,
                        &f[i][p * w..(p + 1) * w],
                        &alone[i],
                        &format!("{case} row {p} alone"),
                    );
                }
                rows += 1;
            }
            cases += 1;
        }
    }
    // Red arm: one mix of the last row moves, and only that row's outputs may move.
    let (mut all, _) = hc_rows(0xBAD8, 6);
    let clean = hc_fused(&e, &all);
    all.mixes[5 * ROWS + 9] = f32::from_bits(all.mixes[5 * ROWS + 9].to_bits() + (1 << 13));
    let red = hc_fused(&e, &all);
    assert_ne!(
        red[3][5 * HC * HC..],
        clean[3][5 * HC * HC..],
        "red arm: row 5 comb stayed bit-equal"
    );
    assert_eq!(
        red[3][..5 * HC * HC],
        clean[3][..5 * HC * HC],
        "red arm: a row 5 change reached rows 0..5"
    );
    println!("DSV4_SMALL_HC_DIET_MROW EXACT cases={cases} rows={rows} outputs=5 red_arms=1");
}

#[test]
#[ignore = "needs a CUDA device; run under flock /tmp/memra-5090.lock"]
fn dsv4_small_norm_pack_diet_multirow_is_bit_identical_to_the_unfused_pair_and_to_each_row_alone() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let (mut cases, mut rows) = (0, 0);
    const RANGES: [(i32, i32); 4] = [(-14, -8), (-4, 3), (2, 9), (-12, 10)];
    for &n in &[1024usize, 1023, 129] {
        for &s in &MROWS {
            let seed = 0x0A0C_0000 ^ ((n as u64) << 12) ^ s as u64;
            let w = fixture(n, seed ^ 0x66, -2, 1);
            let one: Vec<Vec<f32>> = (0..s)
                .map(|p| {
                    let r = RANGES[p % 4];
                    fixture(n, seed ^ 0x55 ^ ((p as u64) << 20), r.0, r.1)
                })
                .collect();
            let x: Vec<f32> = one.iter().flatten().copied().collect();
            let case = format!("norm n={n} s={s} seed {seed:#x}");
            let (u, f) = (norm_unfused(&e, &x, &w), norm_fused(&e, &x, &w));
            assert_bits_eq("row", &f[0], &u[0], &case);
            assert_bits_eq("pack", &f[1], &u[1], &case);
            for (p, xr) in one.iter().enumerate() {
                let alone = norm_fused(&e, xr, &w);
                let pc = format!("{case} row {p} alone");
                assert_bits_eq("row", &f[0][p * n..(p + 1) * n], &alone[0], &pc);
                assert_bits_eq("pack", &f[1][p * n..(p + 1) * n], &alone[1], &pc);
                rows += 1;
            }
            cases += 1;
        }
    }
    println!("DSV4_SMALL_NORM_PACK_DIET_MROW EXACT cases={cases} rows={rows} outputs=2");
}
