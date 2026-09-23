//! Kernel-boundary bit gate for the DSV4 HC finish.
//!
//! `memra_dsv4_hc_dot_split_partial` + `memra_dsv4_hc_finish_f32_fixed_order` replace the
//! HC entry chain of every f32x device program of the HC4 / hidden 4096 shape:
//!
//! `memra_dsv4_hc_dot_split` (partial + slice sum) + `memra_dsv4_rowsq_scale_f32acc` +
//! `memra_dsv4_hc_sinkhorn_m` + `memra_dsv4_hc_collapse` + `memra_dsv4_rmsnorm_f32acc` +
//! `memra_dsv4_cvt_bf16`.
//!
//! The claim is bit identity on every output (scaled mixes, pre, post, comb, y, normalized
//! row, bf16 pack), for every split slice count and for 1 to 6 positions per launch. The
//! fixtures sweep magnitudes over many decades so every tree and division sees rounding in
//! every exponent range.
//!
//! Red arms: a 2^-10 relative change to one HC weight row, one gate scale and one norm weight must
//! each be caught on the output it feeds, so a PASS cannot come from comparing a buffer
//! against itself.
//!
//! Rig law: correctness-only, run under `flock /tmp/memra-5090.lock`, TF32 forced off,
//! `-- --ignored --test-threads=1`.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_engine::Engine;
use memra_engine::dsv4_ffi as k;
use std::os::raw::c_void;

const HC: usize = 4;
const D: usize = 4096;
const W: usize = HC * D;
const ROWS: usize = (2 + HC) * HC;
const ITERS: i32 = 20; // DeepSeek-V4-Flash hc_sinkhorn_iters
const HC_EPS: f32 = 1e-6; // DeepSeek-V4-Flash hc_eps
const RMS_EPS: f32 = 1e-6; // DeepSeek-V4-Flash rms_norm_eps
const MAX_SLICES: usize = 32;

unsafe extern "C" {
    fn memra_dsv4_hc_dot_split(
        x: *const f32,
        w: *const f32,
        partial: *mut f32,
        partial_len: i32,
        y: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_dsv4_hc_dot_split_set_for_gate(slices: i32) -> i32;
}

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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn assert_bits_eq(what: &str, a: &[u32], b: &[u32], case: &str) {
    assert_eq!(a.len(), b.len(), "{case}: {what} length");
    if let Some(i) = (0..a.len()).find(|&i| a[i] != b[i]) {
        panic!(
            "{case}: {what}[{i}] fused {:#010x} != chain {:#010x}",
            a[i], b[i]
        );
    }
}

struct Case {
    m: usize,
    x: Vec<f32>,
    fn_w: Vec<f32>,
    scale: Vec<f32>,
    base: Vec<f32>,
    norm_w: Vec<f32>,
}

fn case(m: usize, seed: u64, x_range: (i32, i32), w_range: (i32, i32)) -> Case {
    Case {
        m,
        x: fixture(m * W, seed ^ 0x11, x_range.0, x_range.1),
        fn_w: fixture(ROWS * W, seed ^ 0x22, w_range.0, w_range.1),
        scale: fixture(3, seed ^ 0x33, -1, 2),
        base: fixture(ROWS, seed ^ 0x44, -3, 1),
        norm_w: fixture(D, seed ^ 0x55, -2, 1),
    }
}

/// Output order: scaled mixes, pre, post, comb, y, normalized row, bf16 pack.
type Out = [Vec<u32>; 7];

struct Bufs {
    x: CudaSlice<f32>,
    fn_w: CudaSlice<f32>,
    scale: CudaSlice<f32>,
    base: CudaSlice<f32>,
    norm_w: CudaSlice<f32>,
    partial: CudaSlice<f32>,
    mixes: CudaSlice<f32>,
    pre: CudaSlice<f32>,
    post: CudaSlice<f32>,
    comb: CudaSlice<f32>,
    y: CudaSlice<f32>,
    out: CudaSlice<f32>,
    packed: CudaSlice<u16>,
}

fn bufs(e: &Engine, c: &Case) -> Bufs {
    let m = c.m;
    // Canary-filled outputs, so an unwritten element cannot match by accident.
    let nan = |n: usize| e.htod(&vec![f32::from_bits(0x7fc0_1234); n]).unwrap();
    Bufs {
        x: e.htod(&c.x).unwrap(),
        fn_w: e.htod(&c.fn_w).unwrap(),
        scale: e.htod(&c.scale).unwrap(),
        base: e.htod(&c.base).unwrap(),
        norm_w: e.htod(&c.norm_w).unwrap(),
        partial: nan(m * ROWS * MAX_SLICES),
        mixes: nan(m * ROWS),
        pre: nan(m * HC),
        post: nan(m * HC),
        comb: nan(m * HC * HC),
        y: nan(m * D),
        out: nan(m * D),
        packed: e.stream().clone_htod(&vec![0xdeadu16; m * D]).unwrap(),
    }
}

fn read(e: &Engine, b: &Bufs) -> Out {
    let p = e.stream().clone_dtoh(&b.packed).unwrap();
    [
        bits(&e.dtoh(&b.mixes).unwrap()),
        bits(&e.dtoh(&b.pre).unwrap()),
        bits(&e.dtoh(&b.post).unwrap()),
        bits(&e.dtoh(&b.comb).unwrap()),
        bits(&e.dtoh(&b.y).unwrap()),
        bits(&e.dtoh(&b.out).unwrap()),
        p.into_iter().map(u32::from).collect(),
    ]
}

fn chain(e: &Engine, c: &Case) -> Out {
    let mut b = bufs(e, c);
    launch_chain(e, &mut b, c.m);
    read(e, &b)
}

fn fused(e: &Engine, c: &Case, slices: i32, with_y: bool) -> Out {
    let mut b = bufs(e, c);
    launch_fused(e, &mut b, c.m, slices, with_y);
    read(e, &b)
}

/// The unfused multi-row chain: split dots, rowsq, Sinkhorn, collapse, rmsnorm, pack.
fn launch_chain(e: &Engine, b: &mut Bufs, rows: usize) {
    let m = rows as i32;
    let stream = e.stream();
    unsafe {
        let plen = b.partial.len() as i32;
        let rc = memra_dsv4_hc_dot_split(
            dp(&b.x, e),
            dp(&b.fn_w, e),
            dpm(&mut b.partial, e),
            plen,
            dpm(&mut b.mixes, e),
            m,
            ROWS as i32,
            W as i32,
            sv(e),
        );
        assert_eq!(rc, 0, "hc_dot_split");
        let rc = k::memra_dsv4_rowsq_scale_f32acc(
            dp(&b.x, e),
            dpm(&mut b.mixes, e),
            m,
            W as i32,
            ROWS as i32,
            HC_EPS,
            sv(e),
        );
        assert_eq!(rc, 0, "rowsq_scale_f32acc");
        let rc = k::memra_dsv4_hc_sinkhorn_m(
            dp(&b.mixes, e),
            dp(&b.scale, e),
            dp(&b.base, e),
            dpm(&mut b.pre, e),
            dpm(&mut b.post, e),
            dpm(&mut b.comb, e),
            m,
            HC as i32,
            ITERS,
            HC_EPS,
            sv(e),
        );
        assert_eq!(rc, 0, "hc_sinkhorn_m");
        let rc = k::memra_dsv4_hc_collapse(
            dp(&b.x, e),
            dp(&b.pre, e),
            dpm(&mut b.y, e),
            m,
            HC as i32,
            D as i32,
            sv(e),
        );
        assert_eq!(rc, 0, "hc_collapse");
        let rc = k::memra_dsv4_rmsnorm_f32acc(
            dp(&b.y, e),
            dp(&b.norm_w, e),
            dpm(&mut b.out, e),
            m,
            D as i32,
            RMS_EPS,
            sv(e),
        );
        assert_eq!(rc, 0, "rmsnorm_f32acc");
        let rc = k::memra_dsv4_cvt_bf16(
            dp(&b.out, e),
            b.packed.device_ptr_mut(&stream).0 as *mut c_void,
            (rows * D) as i64,
            sv(e),
        );
        assert_eq!(rc, 0, "cvt_bf16");
    }
}

/// The t=1 diet on main: split dots, the one-CTA small HC, rmsnorm, pack.
fn launch_diet(e: &Engine, b: &mut Bufs) {
    let stream = e.stream();
    unsafe {
        let plen = b.partial.len() as i32;
        let rc = memra_dsv4_hc_dot_split(
            dp(&b.x, e),
            dp(&b.fn_w, e),
            dpm(&mut b.partial, e),
            plen,
            dpm(&mut b.mixes, e),
            1,
            ROWS as i32,
            W as i32,
            sv(e),
        );
        assert_eq!(rc, 0, "hc_dot_split");
        let rc = k::memra_dsv4_small_hc_f32_fixed_order(
            dp(&b.x, e),
            dpm(&mut b.mixes, e),
            dp(&b.scale, e),
            dp(&b.base, e),
            dpm(&mut b.pre, e),
            dpm(&mut b.post, e),
            dpm(&mut b.comb, e),
            dpm(&mut b.y, e),
            1,
            HC as i32,
            D as i32,
            ITERS,
            HC_EPS,
            sv(e),
        );
        assert_eq!(rc, 0, "small_hc_f32_fixed_order");
        let rc = k::memra_dsv4_rmsnorm_f32acc(
            dp(&b.y, e),
            dp(&b.norm_w, e),
            dpm(&mut b.out, e),
            1,
            D as i32,
            RMS_EPS,
            sv(e),
        );
        assert_eq!(rc, 0, "rmsnorm_f32acc");
        let rc = k::memra_dsv4_cvt_bf16(
            dp(&b.out, e),
            b.packed.device_ptr_mut(&stream).0 as *mut c_void,
            D as i64,
            sv(e),
        );
        assert_eq!(rc, 0, "cvt_bf16");
    }
}

/// The fused pair: split-dot partials, then one CTA per position.
fn launch_fused(e: &Engine, b: &mut Bufs, rows: usize, slices: i32, with_y: bool) {
    let m = rows as i32;
    let stream = e.stream();
    unsafe {
        let plen = b.partial.len() as i32;
        let rc = k::memra_dsv4_hc_dot_split_partial(
            dp(&b.x, e),
            dp(&b.fn_w, e),
            dpm(&mut b.partial, e),
            plen,
            m,
            ROWS as i32,
            W as i32,
            sv(e),
        );
        assert_eq!(rc, 0, "hc_dot_split_partial");
        let y = if with_y {
            dpm(&mut b.y, e)
        } else {
            std::ptr::null_mut()
        };
        let rc = k::memra_dsv4_hc_finish_f32_fixed_order(
            dp(&b.partial, e),
            slices,
            dp(&b.x, e),
            dpm(&mut b.mixes, e),
            dp(&b.scale, e),
            dp(&b.base, e),
            dpm(&mut b.pre, e),
            dpm(&mut b.post, e),
            dpm(&mut b.comb, e),
            y,
            dp(&b.norm_w, e),
            dpm(&mut b.out, e),
            b.packed.device_ptr_mut(&stream).0 as *mut c_void,
            m,
            HC as i32,
            D as i32,
            ITERS,
            HC_EPS,
            RMS_EPS,
            sv(e),
        );
        assert_eq!(rc, 0, "hc_finish_f32_fixed_order");
    }
}

const NAMES: [&str; 7] = ["mixes", "pre", "post", "comb", "y", "row", "pack"];

#[test]
#[ignore = "needs a CUDA device; run under flock /tmp/memra-5090.lock"]
fn dsv4_hc_finish_is_bit_identical_to_the_unfused_chain() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let mut cases = 0;
    for &slices in &[16, 8, 32] {
        assert_eq!(
            unsafe { memra_dsv4_hc_dot_split_set_for_gate(slices) },
            0,
            "slice count {slices}"
        );
        // Residual-stream magnitudes from tiny to large; HC weights from near-zero logits to
        // saturating gates.
        for (xi, &x_range) in [(-12, -6), (-4, 2), (0, 6), (-10, 8)].iter().enumerate() {
            for (wi, &w_range) in [(-14, -8), (-9, -4), (-6, -1)].iter().enumerate() {
                for &m in &[1usize, 2, 5, 6] {
                    let seed = 0xC4F1_0000
                        ^ ((slices as u64) << 20)
                        ^ ((xi as u64) << 16)
                        ^ ((wi as u64) << 8)
                        ^ m as u64;
                    let c = case(m, seed, x_range, w_range);
                    let name =
                        format!("S{slices} m={m} x2^{x_range:?} w2^{w_range:?} seed {seed:#x}");
                    let (u, f) = (chain(&e, &c), fused(&e, &c, slices, true));
                    for (i, what) in NAMES.iter().enumerate() {
                        assert_bits_eq(what, &f[i], &u[i], &name);
                    }
                    cases += 1;
                }
            }
        }
    }
    // The served launch passes no y; every other output must not move.
    unsafe { memra_dsv4_hc_dot_split_set_for_gate(16) };
    let c = case(3, 0x5EED, (-4, 2), (-9, -4));
    let (with_y, without_y) = (fused(&e, &c, 16, true), fused(&e, &c, 16, false));
    for i in [0usize, 1, 2, 3, 5, 6] {
        assert_bits_eq(NAMES[i], &without_y[i], &with_y[i], "null y");
    }
    // Red arms: a 2^-10 relative change on an HC weight row, a gate scale and a norm weight
    // must reach the outputs they feed. The weight arm moves a whole row: one element of a
    // 16384-term dot moves the sum by less than its ulp and proves nothing.
    let clean = fused(&e, &c, 16, true);
    let bump = |v: &mut f32| *v = f32::from_bits(v.to_bits() + (1 << 13));
    let mut red = case(3, 0x5EED, (-4, 2), (-9, -4));
    red.fn_w[5 * W..6 * W].iter_mut().for_each(bump);
    assert_ne!(
        fused(&e, &red, 16, true)[0],
        clean[0],
        "red arm: fn_w moved, mixes did not"
    );
    let mut red = case(3, 0x5EED, (-4, 2), (-9, -4));
    bump(&mut red.scale[2]);
    assert_ne!(
        fused(&e, &red, 16, true)[3],
        clean[3],
        "red arm: scale moved, comb did not"
    );
    let mut red = case(3, 0x5EED, (-4, 2), (-9, -4));
    bump(&mut red.norm_w[1234]);
    assert_ne!(
        fused(&e, &red, 16, true)[5],
        clean[5],
        "red arm: norm_w moved, row did not"
    );
    println!("DSV4_HC_FINISH EXACT cases={cases} outputs=7 slices=8,16,32 rows=1,2,5,6 red_arms=3");
}

fn chain_us(e: &Engine, launches: usize, mut f: impl FnMut()) -> f64 {
    let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    for _ in 0..64 {
        f();
    }
    let start = e.stream().record_event(flags).unwrap();
    for _ in 0..launches {
        f();
    }
    let end = e.stream().record_event(flags).unwrap();
    e.stream().synchronize().unwrap();
    1000.0 * f64::from(start.elapsed_ms(&end).unwrap()) / launches as f64
}

/// Device time per HC entry site, as a CUDA event chain of back-to-back sites (kernels plus
/// inter-kernel gaps): the unfused chain, main's t=1 diet, and the fused pair, at the served
/// plain (1) and DSpark verify (6) row counts. Correctness is the test above.
#[test]
#[ignore = "needs a CUDA device; run under flock /tmp/memra-5090.lock"]
fn dsv4_hc_finish_timing() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    assert_eq!(unsafe { memra_dsv4_hc_dot_split_set_for_gate(16) }, 0);
    let (sites, reps) = (2000, 5);
    for rows in [1usize, 6] {
        let c = case(rows, 0x7117 ^ rows as u64, (-4, 2), (-9, -4));
        let mut b = bufs(&e, &c);
        for rep in 0..reps {
            let chain = chain_us(&e, sites, || launch_chain(&e, &mut b, rows));
            let diet = if rows == 1 {
                chain_us(&e, sites, || launch_diet(&e, &mut b))
            } else {
                f64::NAN
            };
            let fused = chain_us(&e, sites, || launch_fused(&e, &mut b, rows, 16, false));
            println!(
                "TIMING hc_entry rows={rows} rep={rep} chain_us={chain:.3} diet_us={diet:.3} fused_us={fused:.3}"
            );
        }
    }
}
