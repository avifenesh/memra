//! Kernel-boundary bit gate for the DSV4 small-kernel diet (#339), which is the code on every
//! f32x device program of the HC4 / hidden 4096 shape, PP-2 and TP/EP alike.
//!
//! The diet replaces two unfused chains on the plain t=1 step:
//!
//! - HC finish: `memra_dsv4_rowsq_scale_f32acc` + `memra_dsv4_hc_sinkhorn_m` +
//!   `memra_dsv4_hc_collapse` becomes `memra_dsv4_small_hc_f32_fixed_order`.
//! - Q-LoRA norm/pack: `memra_dsv4_rmsnorm_f32acc` (in place) + `memra_dsv4_cvt_bf16` becomes
//!   `memra_dsv4_small_norm_pack_f32_fixed_order`.
//!
//! A multi-row verify or prefill row keeps the unfused chain, so the claim is bit identity, not
//! a band: every output (scaled mixes, pre, post, comb, y; normalized f32 row, bf16 pack) must
//! compare `to_bits`-equal. The fixtures sweep magnitudes over many decades so the rowsq tree,
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
    x: Vec<f32>,
    mixes: Vec<f32>,
    scale: Vec<f32>,
    base: Vec<f32>,
}

/// Output order: scaled mixes, pre, post, comb, y.
type HcOut = [Vec<u32>; 5];

fn hc_unfused(e: &Engine, c: &HcCase) -> HcOut {
    let x = e.htod(&c.x).unwrap();
    let mut mixes = e.htod(&c.mixes).unwrap();
    let scale = e.htod(&c.scale).unwrap();
    let base = e.htod(&c.base).unwrap();
    let mut pre = e.uninit(HC).unwrap();
    let mut post = e.uninit(HC).unwrap();
    let mut comb = e.uninit(HC * HC).unwrap();
    let mut y = e.uninit(D).unwrap();
    unsafe {
        let rc = k::memra_dsv4_rowsq_scale_f32acc(
            dp(&x, e),
            dpm(&mut mixes, e),
            1,
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
            1,
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
            1,
            HC as i32,
            D as i32,
            sv(e),
        );
        assert_eq!(rc, 0, "hc_collapse");
    }
    [&mixes, &pre, &post, &comb, &y].map(|b| bits(&e.dtoh(b).unwrap()))
}

fn hc_fused(e: &Engine, c: &HcCase) -> HcOut {
    let x = e.htod(&c.x).unwrap();
    let mut mixes = e.htod(&c.mixes).unwrap();
    let scale = e.htod(&c.scale).unwrap();
    let base = e.htod(&c.base).unwrap();
    let mut pre = e.uninit(HC).unwrap();
    let mut post = e.uninit(HC).unwrap();
    let mut comb = e.uninit(HC * HC).unwrap();
    let mut y = e.uninit(D).unwrap();
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
            1,
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
        x: fixture(HC * D, seed ^ 0x11, x_range.0, x_range.1),
        mixes: fixture(ROWS, seed ^ 0x22, mix_range.0, mix_range.1),
        scale: fixture(3, seed ^ 0x33, -1, 2),
        base: fixture(ROWS, seed ^ 0x44, -3, 1),
    }
}

/// Output order: normalized f32 row, bf16 pack (as u16 widened to u32).
fn norm_unfused(e: &Engine, x0: &[f32], w: &[f32]) -> [Vec<u32>; 2] {
    let n = x0.len();
    let mut x = e.htod(x0).unwrap();
    let wd = e.htod(w).unwrap();
    let stream = e.stream();
    let mut packed = stream.alloc_zeros::<u16>(n).unwrap();
    unsafe {
        let xp = dpm(&mut x, e);
        let rc = k::memra_dsv4_rmsnorm_f32acc(xp, dp(&wd, e), xp, 1, n as i32, RMS_EPS, sv(e));
        assert_eq!(rc, 0, "rmsnorm_f32acc");
        let rc = k::memra_dsv4_cvt_bf16(
            dp(&x, e),
            packed.device_ptr_mut(&stream).0 as *mut c_void,
            n as i64,
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
    let n = x0.len();
    let mut x = e.htod(x0).unwrap();
    let wd = e.htod(w).unwrap();
    let stream = e.stream();
    let mut packed = stream.alloc_zeros::<u16>(n).unwrap();
    unsafe {
        let rc = k::memra_dsv4_small_norm_pack_f32_fixed_order(
            dpm(&mut x, e),
            dp(&wd, e),
            packed.device_ptr_mut(&stream).0 as *mut c_void,
            1,
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
