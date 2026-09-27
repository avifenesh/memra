//! Kernel-boundary bit gate for the DSV4 latency-kernel lane: the single-CTA kernels that
//! sit on every f32x t=1 layer of both topologies (PP-2 and TP/EP), where each one is a
//! chain of dependent load round trips rather than bandwidth.
//!
//! - `memra_dsv4_rmsnorm_f32acc`: rows up to 4096 columns keep x and w in registers
//!   (one load latency instead of one per 8-load batch plus one per output). Checked
//!   against an exact CPU oracle: dsv4_gpu.cu builds with `-fmad=false` and IEEE div and
//!   sqrt, so the per-thread ascending order, the 128 halving tree and the f32 mean and
//!   reciprocal square root are reproducible bit for bit on the host.
//! - `memra_dsv4_grouped_routes{,_fault,_partition}`: the expert prefix is a warp scan
//!   instead of a 256-step serial loop on one lane. Integer metadata, so the CPU oracle
//!   is exact, including the lost-slot status and the fault bit.
//! - `memra_dsv4_route_m`: warp 0 selects with a shuffle butterfly instead of a
//!   128-thread shared tree with a block barrier per level. softplus uses expf/log1pf, so
//!   the proof is a golden hash of (sel, selw bits, order) captured from the former kernel
//!   at 31dd11455 on this TU, plus structural checks that hold for any correct selection.
//!
//! Red arms: a 2^-10 relative change to one input must be caught by the comparator on
//! the output it feeds, so a PASS cannot come from comparing a buffer against itself.
//!
//! - FP8 wo_a (tested in `dsv4_gpu.rs`): the grouped t=1 launch takes a grouped dense-fast
//!   twin, so one launch replaces the eight per-group dense-fast launches. The instrument
//!   below times both at the served 8x1024x4096 shape.
//!
//! `latency_kernel_timing` is the component instrument: CUDA event chains of back-to-back
//! launches at the served shapes. Build it on the base tree and on the lane and run the
//! two binaries interleaved; a served A/B at sub-2% size is swamped by run drift.
//!
//! Rig law: run under `flock /tmp/memra-5090.lock`, TF32 forced off,
//! `-- --ignored --test-threads=1`.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_engine::Engine;
use memra_engine::dsv4_ffi as k;
use std::os::raw::c_void;

const RMS_EPS: f32 = 1e-6; // DeepSeek-V4-Flash rms_norm_eps

/// FNV-1a 64 over the route_m outputs of every `route_m_cases()` case, captured from
/// the former 128-thread tree kernel (base 31dd11455) on sm_120.
const ROUTE_M_GOLDEN: u64 = 0xa883_c1c0_5022_dc87;

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

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 17
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 8) as u32 as f32) / (1u32 << 31) as f32 - 1.0
    }
}

/// Deterministic values in [-1, 1) times a per-element power of two in 2^lo .. 2^hi.
fn fixture(n: usize, seed: u64, lo: i32, hi: i32) -> Vec<f32> {
    let mut r = Lcg(seed | 1);
    (0..n)
        .map(|_| {
            let u = r.unit();
            let e = lo + r.below((hi - lo + 1) as usize) as i32;
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
fn ip(s: &CudaSlice<i32>, e: &Engine) -> *const i32 {
    s.device_ptr(&e.stream()).0 as *const i32
}
fn ipm(s: &mut CudaSlice<i32>, e: &Engine) -> *mut i32 {
    s.device_ptr_mut(&e.stream()).0 as *mut i32
}
fn sv(e: &Engine) -> *mut c_void {
    e.stream().cu_stream() as *mut c_void
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn assert_bits_eq(what: &str, got: &[u32], want: &[u32], case: &str) {
    assert_eq!(got.len(), want.len(), "{case}: {what} length");
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        panic!(
            "{case}: {what}[{i}] gpu {:#010x} != oracle {:#010x}",
            got[i], want[i]
        );
    }
}

fn assert_i32_eq(what: &str, got: &[i32], want: &[i32], case: &str) {
    assert_eq!(got.len(), want.len(), "{case}: {what} length");
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        panic!("{case}: {what}[{i}] gpu {} != oracle {}", got[i], want[i]);
    }
}

// ---------------------------------------------------------------------------------------
// rmsnorm

/// dsv4_rmsnorm_f32acc_kernel at 128 threads: thread t sums x[j]^2 for j = t, t+128, ...
/// ascending, the 128 partials meet in a halving tree (partner t+off added to t), then
/// mean = tot / ncols, rsq = 1 / sqrt(mean + eps), dst = w * (x * rsq). Every step is
/// one IEEE f32 operation with no contraction.
fn rmsnorm_oracle(x: &[f32], w: Option<&[f32]>, rows: usize, ncols: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * ncols];
    for r in 0..rows {
        let xr = &x[r * ncols..(r + 1) * ncols];
        let mut sh = [0.0f32; 128];
        for (t, slot) in sh.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            let mut j = t;
            while j < ncols {
                acc += xr[j] * xr[j];
                j += 128;
            }
            *slot = acc;
        }
        let mut off = 64;
        while off > 0 {
            for t in 0..off {
                sh[t] += sh[t + off];
            }
            off >>= 1;
        }
        let mean = sh[0] / ncols as f32;
        let rsq = 1.0f32 / (mean + eps).sqrt();
        for j in 0..ncols {
            let wj = w.map_or(1.0f32, |w| w[j]);
            out[r * ncols + j] = wj * (xr[j] * rsq);
        }
    }
    out
}

fn rmsnorm_gpu(
    e: &Engine,
    x: &[f32],
    w: Option<&[f32]>,
    rows: usize,
    ncols: usize,
    in_place: bool,
) -> Vec<f32> {
    let mut xd = e.htod(x).unwrap();
    let wd = w.map(|w| e.htod(w).unwrap());
    let wp = wd.as_ref().map_or(std::ptr::null(), |w| dp(w, e));
    let rc = if in_place {
        let p = dpm(&mut xd, e);
        unsafe { k::memra_dsv4_rmsnorm_f32acc(p, wp, p, rows as i32, ncols as i32, RMS_EPS, sv(e)) }
    } else {
        let mut dd = e.uninit(rows * ncols).unwrap();
        let rc = unsafe {
            k::memra_dsv4_rmsnorm_f32acc(
                dp(&xd, e),
                wp,
                dpm(&mut dd, e),
                rows as i32,
                ncols as i32,
                RMS_EPS,
                sv(e),
            )
        };
        assert_eq!(rc, 0, "rmsnorm_f32acc");
        return e.dtoh(&dd).unwrap();
    };
    assert_eq!(rc, 0, "rmsnorm_f32acc in place");
    e.dtoh(&xd).unwrap()
}

const RMS_NCOLS: [usize; 15] = [
    1, 7, 127, 128, 129, 512, 1000, 1024, 2047, 4095, 4096, 4097, 7168, 8192, 16384,
];

#[test]
#[ignore]
fn rmsnorm_f32acc_matches_exact_oracle() {
    force_true_f32();
    let _g = gpu_guard();
    let e = Engine::new(0).unwrap();
    let mut cases = 0;
    for (ci, &ncols) in RMS_NCOLS.iter().enumerate() {
        for rows in [1usize, 3] {
            // Magnitudes over many decades so the per-thread sums and the tree round in
            // every exponent range; a second fixture keeps the row near unit scale.
            for (fi, (lo, hi)) in [(-18, 18), (-2, 1)].into_iter().enumerate() {
                let seed = 0x5eed_0000 + (ci * 16 + rows * 4 + fi) as u64;
                let x = fixture(rows * ncols, seed, lo, hi);
                let w = fixture(ncols, seed ^ 0xa5a5, -3, 3);
                for wopt in [Some(w.as_slice()), None] {
                    let want = bits(&rmsnorm_oracle(&x, wopt, rows, ncols, RMS_EPS));
                    for in_place in [false, true] {
                        let case = format!(
                            "ncols={ncols} rows={rows} mag=2^{lo}..2^{hi} w={} in_place={in_place}",
                            wopt.is_some()
                        );
                        let got = bits(&rmsnorm_gpu(&e, &x, wopt, rows, ncols, in_place));
                        assert_bits_eq("dst", &got, &want, &case);
                        cases += 1;
                    }
                }
            }
        }
    }
    println!("rmsnorm_f32acc exact oracle: {cases} cases bit-identical");
}

#[test]
#[ignore]
fn rmsnorm_f32acc_red_arm() {
    force_true_f32();
    let _g = gpu_guard();
    let e = Engine::new(0).unwrap();
    for ncols in [512usize, 4096, 8192] {
        let x = fixture(ncols, 0xbad0 + ncols as u64, -6, 6);
        let w = fixture(ncols, 0xbad1, -3, 3);
        let want = bits(&rmsnorm_oracle(&x, Some(&w), 1, ncols, RMS_EPS));
        // The largest element moves the sum of squares; every output then moves with rsq.
        let big = (0..ncols)
            .max_by(|&a, &b| x[a].abs().total_cmp(&x[b].abs()))
            .unwrap();
        let mut xr = x.clone();
        xr[big] *= 1.0 + 2f32.powi(-10);
        let got = bits(&rmsnorm_gpu(&e, &xr, Some(&w), 1, ncols, false));
        let moved = (0..ncols).filter(|&i| got[i] != want[i]).count();
        assert!(
            moved > 1,
            "ncols={ncols}: red arm moved {moved} outputs; comparator is blind"
        );
    }
}

// ---------------------------------------------------------------------------------------
// grouped routes

#[derive(Debug, PartialEq)]
struct Routes {
    counts: Vec<i32>,
    offsets: Vec<i32>,
    expert_ids: Vec<i32>,
    pairs: Vec<i32>,
    tokens: Vec<i32>,
    weights: Vec<u32>,
    macros: [Vec<u32>; 3],
    status: i32,
    fault: i32,
}

/// count + prefix + scatter. `first..first + groups` is the owned expert range (the whole
/// bank for the unpartitioned launch); scales stay addressed by global expert id.
#[allow(clippy::too_many_arguments)]
fn routes_oracle(
    selected: &[i32],
    weights: &[f32],
    scale2: &[f32],
    first: usize,
    groups: usize,
    global: usize,
    topk: usize,
    partition: bool,
    fault_bit: i32,
) -> Routes {
    let slots = selected.len();
    let counts: Vec<i32> = (0..groups)
        .map(|g| {
            selected
                .iter()
                .filter(|&&s| s == (first + g) as i32)
                .count() as i32
        })
        .collect();
    let mut offsets = vec![0i32; groups + 1];
    for g in 0..groups {
        offsets[g + 1] = offsets[g] + counts[g];
    }
    let mut pairs = vec![-1i32; slots];
    let mut tokens = vec![-1i32; slots];
    let mut rw = vec![0.0f32; slots];
    let mut m = [
        vec![0.0f32; slots],
        vec![0.0f32; slots],
        vec![0.0f32; slots],
    ];
    for (g, &off) in offsets.iter().take(groups).enumerate() {
        let expert = first + g;
        let mut dst = off as usize;
        for p in 0..slots {
            if selected[p] == expert as i32 {
                pairs[dst] = p as i32;
                tokens[dst] = (p / topk) as i32;
                rw[dst] = weights[p];
                for (c, mc) in m.iter_mut().enumerate() {
                    mc[dst] = scale2[expert * 3 + c];
                }
                dst += 1;
            }
        }
    }
    let status = if partition {
        i32::from(selected.iter().any(|&s| s < 0 || s >= global as i32))
    } else {
        i32::from(offsets[groups] != slots as i32)
    };
    let fault = if !partition && status != 0 {
        fault_bit
    } else {
        0
    };
    Routes {
        counts,
        offsets,
        expert_ids: (0..groups as i32).collect(),
        pairs,
        tokens,
        weights: bits(&rw),
        macros: [bits(&m[0]), bits(&m[1]), bits(&m[2])],
        status,
        fault,
    }
}

enum Launch {
    Plain,
    Fault(i32),
    Partition { first: usize, groups: usize },
}

fn routes_gpu(
    e: &Engine,
    selected: &[i32],
    weights: &[f32],
    scale2: &[f32],
    global: usize,
    topk: usize,
    launch: &Launch,
) -> Routes {
    let slots = selected.len();
    let groups = match launch {
        Launch::Partition { groups, .. } => *groups,
        _ => global,
    };
    let sel = e.htod_i32(selected).unwrap();
    let w = e.htod(weights).unwrap();
    let s2 = e.htod(scale2).unwrap();
    // Poisoned outputs: a slot the kernels failed to write shows up as a mismatch.
    let mut counts = e.htod_i32(&vec![0x7eadbeef; groups]).unwrap();
    let mut offsets = e.htod_i32(&vec![0x7eadbeef; groups + 1]).unwrap();
    let mut ids = e.htod_i32(&vec![0x7eadbeef; groups]).unwrap();
    let mut pairs = e.htod_i32(&vec![0x7eadbeef; slots]).unwrap();
    let mut tokens = e.htod_i32(&vec![0x7eadbeef; slots]).unwrap();
    let poison = vec![f32::from_bits(0x7fc0_dead); slots];
    let mut rw = e.htod(&poison).unwrap();
    let mut m1 = e.htod(&poison).unwrap();
    let mut m2 = e.htod(&poison).unwrap();
    let mut m3 = e.htod(&poison).unwrap();
    let mut status = e.htod_i32(&[0x7eadbeef]).unwrap();
    let mut fault = e.htod_i32(&[0]).unwrap();
    let rc = unsafe {
        match launch {
            Launch::Plain => k::memra_dsv4_grouped_routes(
                ip(&sel, e),
                dp(&w, e),
                dp(&s2, e),
                ipm(&mut counts, e),
                ipm(&mut offsets, e),
                ipm(&mut ids, e),
                ipm(&mut pairs, e),
                ipm(&mut tokens, e),
                dpm(&mut rw, e),
                dpm(&mut m1, e),
                dpm(&mut m2, e),
                dpm(&mut m3, e),
                ipm(&mut status, e),
                slots as i32,
                global as i32,
                topk as i32,
                sv(e),
            ),
            Launch::Fault(bit) => k::memra_dsv4_grouped_routes_fault(
                ip(&sel, e),
                dp(&w, e),
                dp(&s2, e),
                ipm(&mut counts, e),
                ipm(&mut offsets, e),
                ipm(&mut ids, e),
                ipm(&mut pairs, e),
                ipm(&mut tokens, e),
                dpm(&mut rw, e),
                dpm(&mut m1, e),
                dpm(&mut m2, e),
                dpm(&mut m3, e),
                ipm(&mut status, e),
                slots as i32,
                global as i32,
                topk as i32,
                ipm(&mut fault, e),
                *bit,
                sv(e),
            ),
            Launch::Partition { first, groups } => k::memra_dsv4_grouped_routes_partition(
                ip(&sel, e),
                dp(&w, e),
                dp(&s2, e),
                ipm(&mut counts, e),
                ipm(&mut offsets, e),
                ipm(&mut ids, e),
                ipm(&mut pairs, e),
                ipm(&mut tokens, e),
                dpm(&mut rw, e),
                dpm(&mut m1, e),
                dpm(&mut m2, e),
                dpm(&mut m3, e),
                ipm(&mut status, e),
                slots as i32,
                global as i32,
                *first as i32,
                *groups as i32,
                topk as i32,
                sv(e),
            ),
        }
    };
    assert_eq!(rc, 0, "grouped routes launch");
    Routes {
        counts: e.dtoh_i32(&counts).unwrap(),
        offsets: e.dtoh_i32(&offsets).unwrap(),
        expert_ids: e.dtoh_i32(&ids).unwrap(),
        pairs: e.dtoh_i32(&pairs).unwrap(),
        tokens: e.dtoh_i32(&tokens).unwrap(),
        weights: bits(&e.dtoh(&rw).unwrap()),
        macros: [
            bits(&e.dtoh(&m1).unwrap()),
            bits(&e.dtoh(&m2).unwrap()),
            bits(&e.dtoh(&m3).unwrap()),
        ],
        status: e.dtoh_i32(&status).unwrap()[0],
        fault: e.dtoh_i32(&fault).unwrap()[0],
    }
}

fn assert_routes_eq(got: &Routes, want: &Routes, case: &str) {
    assert_i32_eq("counts", &got.counts, &want.counts, case);
    assert_i32_eq("offsets", &got.offsets, &want.offsets, case);
    assert_i32_eq("expert_ids", &got.expert_ids, &want.expert_ids, case);
    assert_i32_eq("pairs", &got.pairs, &want.pairs, case);
    assert_i32_eq("tokens", &got.tokens, &want.tokens, case);
    assert_bits_eq("route_weights", &got.weights, &want.weights, case);
    for c in 0..3 {
        assert_bits_eq(
            &format!("macro{}", c + 1),
            &got.macros[c],
            &want.macros[c],
            case,
        );
    }
    assert_eq!(got.status, want.status, "{case}: status");
    assert_eq!(got.fault, want.fault, "{case}: fault word");
}

/// topk distinct experts per token (the router's shape), or with repeats when `repeat`.
fn selection(r: &mut Lcg, tokens: usize, topk: usize, experts: usize, repeat: bool) -> Vec<i32> {
    let mut out = Vec::with_capacity(tokens * topk);
    for _ in 0..tokens {
        let mut row: Vec<i32> = Vec::with_capacity(topk);
        while row.len() < topk {
            let x = r.below(experts) as i32;
            if repeat || !row.contains(&x) {
                row.push(x);
            }
        }
        out.extend(row);
    }
    out
}

#[test]
#[ignore]
fn grouped_routes_match_cpu_oracle() {
    force_true_f32();
    let _g = gpu_guard();
    let e = Engine::new(0).unwrap();
    let mut r = Lcg(0x6007e5);
    let mut cases = 0;
    for experts in [1usize, 7, 31, 32, 33, 64, 255, 256, 257, 511, 512] {
        let scale2 = fixture(experts * 3, 0x5ca1e + experts as u64, -8, 8);
        for topk in [1usize, 6, 8] {
            if topk > experts {
                continue;
            }
            for tokens in [1usize, 3, 17, 64] {
                for repeat in [false, true] {
                    // Corruption: 0 none, 1 a negative id, 2 an id one past the bank.
                    for bad in 0..3 {
                        let mut sel = selection(&mut r, tokens, topk, experts, repeat);
                        let slots = sel.len();
                        if bad > 0 {
                            let p = r.below(slots);
                            sel[p] = if bad == 1 { -1 } else { experts as i32 };
                        }
                        let w = fixture(slots, r.next(), -4, 4);
                        for (li, launch) in
                            [Launch::Plain, Launch::Fault(1 << 5)].iter().enumerate()
                        {
                            let bit = if li == 1 { 1 << 5 } else { 0 };
                            let want = routes_oracle(
                                &sel, &w, &scale2, 0, experts, experts, topk, false, bit,
                            );
                            let got = routes_gpu(&e, &sel, &w, &scale2, experts, topk, launch);
                            let case = format!(
                                "experts={experts} topk={topk} tokens={tokens} repeat={repeat} bad={bad} fault={}",
                                li == 1
                            );
                            assert_routes_eq(&got, &want, &case);
                            assert_eq!(want.status, i32::from(bad > 0), "{case}: oracle status");
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    // TP/EP partitions of the 256-expert bank, including a rank that owns nothing routed.
    let global = 256usize;
    let scale2 = fixture(global * 3, 0x5ca1e256, -8, 8);
    for (first, groups) in [
        (0usize, 256usize),
        (0, 128),
        (128, 128),
        (64, 56),
        (200, 56),
        (255, 1),
        (37, 1),
    ] {
        for topk in [1usize, 6, 8] {
            for tokens in [1usize, 3, 17, 64, 512] {
                for bad in 0..3 {
                    let mut sel = selection(&mut r, tokens, topk, global, false);
                    let slots = sel.len();
                    if bad > 0 {
                        let p = r.below(slots);
                        sel[p] = if bad == 1 { -1 } else { global as i32 };
                    }
                    let w = fixture(slots, r.next(), -4, 4);
                    let want =
                        routes_oracle(&sel, &w, &scale2, first, groups, global, topk, true, 0);
                    let got = routes_gpu(
                        &e,
                        &sel,
                        &w,
                        &scale2,
                        global,
                        topk,
                        &Launch::Partition { first, groups },
                    );
                    let case = format!(
                        "partition first={first} groups={groups} topk={topk} tokens={tokens} bad={bad}"
                    );
                    assert_routes_eq(&got, &want, &case);
                    cases += 1;
                }
            }
        }
    }
    println!("grouped routes exact oracle: {cases} cases identical");
}

#[test]
#[ignore]
fn grouped_routes_red_arm() {
    force_true_f32();
    let _g = gpu_guard();
    let e = Engine::new(0).unwrap();
    let mut r = Lcg(0x6007bad);
    let (experts, topk, tokens) = (256usize, 6usize, 17usize);
    let scale2 = fixture(experts * 3, 7, -8, 8);
    let sel = selection(&mut r, tokens, topk, experts, false);
    let w = fixture(sel.len(), 9, -4, 4);
    let want = routes_oracle(&sel, &w, &scale2, 0, experts, experts, topk, false, 0);
    // Move one slot to a different valid expert: counts, offsets and the scatter must move.
    let mut moved = sel.clone();
    moved[5] = (moved[5] + 1) % experts as i32;
    let got = routes_gpu(&e, &moved, &w, &scale2, experts, topk, &Launch::Plain);
    assert_ne!(
        got.counts, want.counts,
        "red arm: counts comparator is blind"
    );
    assert_ne!(
        got.offsets, want.offsets,
        "red arm: offsets comparator is blind"
    );
    assert_ne!(got.pairs, want.pairs, "red arm: pairs comparator is blind");
}

// ---------------------------------------------------------------------------------------
// route_m

struct RouteCase {
    s: usize,
    ne: usize,
    topk: usize,
    raw: Vec<f32>,
    bias: Option<Vec<f32>>,
    /// (tid2eid [vocab, topk], tok [s]) for the hash layers.
    hash: Option<(Vec<i32>, Vec<i32>)>,
    route_scale: f32,
}

fn route_m_cases() -> Vec<RouteCase> {
    let mut r = Lcg(0x7007e);
    let mut out = Vec::new();
    for ne in [8usize, 31, 32, 33, 64, 200, 255, 256] {
        for topk in [1usize, 6, 8, 32] {
            if topk > ne {
                continue;
            }
            for s in [1usize, 4, 9] {
                // 0: continuous logits and bias; 1: no bias (all biased scores tie at 0,
                // so the pick is index-ascending); 2: logits on a 4-value grid with a
                // 2-value bias (value ties broken by index); 3: tid2eid hash layer.
                for kind in 0..4 {
                    let raw: Vec<f32> = (0..s * ne)
                        .map(|_| match kind {
                            2 => (r.below(4) as f32) * 0.75 - 1.5,
                            _ => r.unit() * 12.0,
                        })
                        .collect();
                    let bias = match kind {
                        0 | 3 => Some((0..ne).map(|_| r.unit() * 0.25).collect()),
                        2 => Some((0..ne).map(|_| (r.below(2) as f32) * 0.125).collect()),
                        _ => None,
                    };
                    let hash = (kind == 3).then(|| {
                        let vocab = 37;
                        let table = selection(&mut r, vocab, topk, ne, false);
                        let tok = (0..s).map(|_| r.below(vocab) as i32).collect();
                        (table, tok)
                    });
                    out.push(RouteCase {
                        s,
                        ne,
                        topk,
                        raw,
                        bias,
                        hash,
                        route_scale: 1.5,
                    });
                }
            }
        }
    }
    out
}

fn route_m_gpu(e: &Engine, c: &RouteCase) -> (Vec<i32>, Vec<f32>, Vec<i32>) {
    let raw = e.htod(&c.raw).unwrap();
    let bias = c.bias.as_ref().map(|b| e.htod(b).unwrap());
    let hash = c
        .hash
        .as_ref()
        .map(|(t, tok)| (e.htod_i32(t).unwrap(), e.htod_i32(tok).unwrap()));
    let n = c.s * c.topk;
    let mut sel = e.htod_i32(&vec![0x7eadbeef; n]).unwrap();
    let mut selw = e.htod(&vec![f32::from_bits(0x7fc0_dead); n]).unwrap();
    let mut order = e.htod_i32(&vec![0x7eadbeef; n]).unwrap();
    let rc = unsafe {
        k::memra_dsv4_route_m(
            dp(&raw, e),
            bias.as_ref().map_or(std::ptr::null(), |b| dp(b, e)),
            hash.as_ref().map_or(std::ptr::null(), |(t, _)| ip(t, e)),
            hash.as_ref()
                .map_or(std::ptr::null(), |(_, tok)| ip(tok, e)),
            c.s as i32,
            c.ne as i32,
            c.topk as i32,
            c.route_scale,
            ipm(&mut sel, e),
            dpm(&mut selw, e),
            ipm(&mut order, e),
            sv(e),
        )
    };
    assert_eq!(rc, 0, "route_m launch");
    (
        e.dtoh_i32(&sel).unwrap(),
        e.dtoh(&selw).unwrap(),
        e.dtoh_i32(&order).unwrap(),
    )
}

fn fnv(h: &mut u64, word: u32) {
    for b in word.to_le_bytes() {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// Checks that hold for any correct kernel, independent of the golden hash: distinct
/// in-range picks, the tid2eid row verbatim on hash layers, value-descending picks on
/// clearly separated biased scores (host f64 softplus; near-ties are left to the hash),
/// weights that sum to route_scale, and order = slots sorted by ascending expert id.
fn route_m_structure(c: &RouteCase, sel: &[i32], selw: &[f32], order: &[i32], ci: usize) {
    let sp = |x: f32| -> f64 {
        let x = f64::from(x);
        if x > 20.0 { x } else { x.exp().ln_1p() }
    };
    for p in 0..c.s {
        let case = format!("case {ci} ne={} topk={} s={} pos={p}", c.ne, c.topk, c.s);
        let ps = &sel[p * c.topk..(p + 1) * c.topk];
        let pw = &selw[p * c.topk..(p + 1) * c.topk];
        let po = &order[p * c.topk..(p + 1) * c.topk];
        for (k, &x) in ps.iter().enumerate() {
            assert!(
                x >= 0 && (x as usize) < c.ne,
                "{case}: sel[{k}]={x} out of range"
            );
        }
        if let Some((table, tok)) = &c.hash {
            let row = &table[tok[p] as usize * c.topk..(tok[p] as usize + 1) * c.topk];
            assert_eq!(ps, row, "{case}: hash layer must take the tid2eid row");
        } else {
            let mut uniq = ps.to_vec();
            uniq.sort_unstable();
            uniq.dedup();
            assert_eq!(uniq.len(), c.topk, "{case}: duplicate pick {ps:?}");
            let score = |x: usize| -> f64 {
                let s = sp(c.raw[p * c.ne + x]).sqrt();
                c.bias.as_ref().map_or(0.0, |b| s + f64::from(b[x]))
            };
            let picked: Vec<f64> = ps.iter().map(|&x| score(x as usize)).collect();
            for k in 1..c.topk {
                let (a, b) = (picked[k - 1], picked[k]);
                if (a - b).abs() > 1e-4 {
                    assert!(
                        a > b,
                        "{case}: pick {k} scores {b} above pick {} {a}",
                        k - 1
                    );
                } else if a == b {
                    assert!(ps[k - 1] < ps[k], "{case}: exact tie not index-ascending");
                }
            }
            let floor = picked[c.topk - 1];
            for x in 0..c.ne {
                if !ps.contains(&(x as i32)) {
                    assert!(
                        score(x) <= floor + 1e-4,
                        "{case}: unpicked expert {x} scores {} above the last pick {floor}",
                        score(x)
                    );
                }
            }
        }
        let sum: f64 = pw.iter().map(|&w| f64::from(w)).sum();
        assert!(
            (sum - f64::from(c.route_scale)).abs() < 1e-4,
            "{case}: weights sum {sum}"
        );
        let mut want: Vec<i32> = (0..c.topk as i32).collect();
        want.sort_by_key(|&k| ps[k as usize]);
        assert_eq!(po, &want[..], "{case}: order");
    }
}

#[test]
#[ignore]
fn route_m_matches_golden_and_structure() {
    force_true_f32();
    let _g = gpu_guard();
    let e = Engine::new(0).unwrap();
    let cases = route_m_cases();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for (ci, c) in cases.iter().enumerate() {
        let (sel, selw, order) = route_m_gpu(&e, c);
        route_m_structure(c, &sel, &selw, &order, ci);
        fnv(&mut h, ci as u32);
        for i in 0..sel.len() {
            fnv(&mut h, sel[i] as u32);
            fnv(&mut h, selw[i].to_bits());
            fnv(&mut h, order[i] as u32);
        }
    }
    println!("route_m: {} cases, golden {h:#018x}", cases.len());
    assert_eq!(
        h, ROUTE_M_GOLDEN,
        "route_m outputs differ from the 31dd11455 capture: {h:#018x}"
    );
}

#[test]
#[ignore]
fn route_m_red_arm() {
    force_true_f32();
    let _g = gpu_guard();
    let e = Engine::new(0).unwrap();
    // A one-pick case normalizes its weight to route_scale whatever the logit, so the red
    // arm takes the first continuous-logit case with six picks.
    let mut c = route_m_cases()
        .into_iter()
        .find(|c| c.topk == 6 && c.bias.is_some() && c.hash.is_none())
        .expect("a six-pick biased case");
    let (sel, selw, _) = route_m_gpu(&e, &c);
    // Scale the first pick's logit by 2^-10: its weight and the normalizer move.
    let x = sel[0] as usize;
    c.raw[x] *= 1.0 + 2f32.powi(-10);
    let (_, selw2, _) = route_m_gpu(&e, &c);
    assert_ne!(
        bits(&selw),
        bits(&selw2),
        "route_m red arm: selw comparator is blind"
    );
}

// ---------------------------------------------------------------------------------------
// component instrument

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

/// Per-launch device time of each lane kernel at the served t=1 shapes, as a CUDA event
/// chain of back-to-back launches (kernel plus inter-kernel gap). Prints one TIMING row
/// per kernel and repeat so the base and lane binaries can be interleaved and compared.
#[test]
#[ignore]
fn latency_kernel_timing() {
    force_true_f32();
    let _g = gpu_guard();
    let e = Engine::new(0).unwrap();
    let launches = 4000;
    let reps = 5;
    let tree = std::env::var("DSV4_LATENCY_TREE").unwrap_or_else(|_| "unlabeled".into());
    for ncols in [4096usize, 1024, 512] {
        let x = fixture(ncols, 1, -4, 4);
        let w = fixture(ncols, 2, -2, 2);
        let xd = e.htod(&x).unwrap();
        let wd = e.htod(&w).unwrap();
        let mut dd = e.uninit(ncols).unwrap();
        let (xp, wp, op) = (dp(&xd, &e), dp(&wd, &e), dpm(&mut dd, &e));
        for rep in 0..reps {
            let us = chain_us(&e, launches, || unsafe {
                assert_eq!(
                    k::memra_dsv4_rmsnorm_f32acc(xp, wp, op, 1, ncols as i32, RMS_EPS, sv(&e)),
                    0
                );
            });
            println!(
                "TIMING tree={tree} kernel=rmsnorm_f32acc shape=1x{ncols} rep={rep} us_per_launch={us:.3}"
            );
        }
    }
    {
        let (ne, topk) = (256usize, 6usize);
        let raw = e.htod(&fixture(ne, 3, -2, 3)).unwrap();
        let bias = e.htod(&fixture(ne, 4, -6, -2)).unwrap();
        let mut sel = e.uninit_i32(topk).unwrap();
        let mut selw = e.uninit(topk).unwrap();
        let mut order = e.uninit_i32(topk).unwrap();
        let (rp, bp) = (dp(&raw, &e), dp(&bias, &e));
        let (sp, wp, op) = (ipm(&mut sel, &e), dpm(&mut selw, &e), ipm(&mut order, &e));
        for rep in 0..reps {
            let us = chain_us(&e, launches, || unsafe {
                let rc = k::memra_dsv4_route_m(
                    rp,
                    bp,
                    std::ptr::null(),
                    std::ptr::null(),
                    1,
                    ne as i32,
                    topk as i32,
                    1.5,
                    sp,
                    wp,
                    op,
                    sv(&e),
                );
                assert_eq!(rc, 0);
            });
            println!(
                "TIMING tree={tree} kernel=route_m shape=s1_ne{ne}_top{topk} rep={rep} us_per_launch={us:.3}"
            );
        }
    }
    for (label, global, first, groups) in [
        ("grouped_routes", 256usize, 0usize, 256usize),
        ("grouped_routes_partition", 256, 128, 128),
    ] {
        let topk = 6usize;
        let mut r = Lcg(5);
        let selv = selection(&mut r, 1, topk, global, false);
        let sel = e.htod_i32(&selv).unwrap();
        let w = e.htod(&fixture(topk, 6, -2, 0)).unwrap();
        let s2 = e.htod(&fixture(global * 3, 7, -4, 4)).unwrap();
        let mut counts = e.uninit_i32(groups).unwrap();
        let mut offsets = e.uninit_i32(groups + 1).unwrap();
        let mut ids = e.uninit_i32(groups).unwrap();
        let mut pairs = e.uninit_i32(topk).unwrap();
        let mut tokens = e.uninit_i32(topk).unwrap();
        let mut rw = e.uninit(topk).unwrap();
        let mut m1 = e.uninit(topk).unwrap();
        let mut m2 = e.uninit(topk).unwrap();
        let mut m3 = e.uninit(topk).unwrap();
        let mut status = e.uninit_i32(1).unwrap();
        let p = (
            ip(&sel, &e),
            dp(&w, &e),
            dp(&s2, &e),
            ipm(&mut counts, &e),
            ipm(&mut offsets, &e),
            ipm(&mut ids, &e),
            ipm(&mut pairs, &e),
            ipm(&mut tokens, &e),
            dpm(&mut rw, &e),
            dpm(&mut m1, &e),
            dpm(&mut m2, &e),
            dpm(&mut m3, &e),
            ipm(&mut status, &e),
        );
        for rep in 0..reps {
            // Three kernels per call (count, prefix, scatter).
            let us = chain_us(&e, launches, || unsafe {
                let rc = if groups == global {
                    k::memra_dsv4_grouped_routes(
                        p.0,
                        p.1,
                        p.2,
                        p.3,
                        p.4,
                        p.5,
                        p.6,
                        p.7,
                        p.8,
                        p.9,
                        p.10,
                        p.11,
                        p.12,
                        topk as i32,
                        global as i32,
                        topk as i32,
                        sv(&e),
                    )
                } else {
                    k::memra_dsv4_grouped_routes_partition(
                        p.0,
                        p.1,
                        p.2,
                        p.3,
                        p.4,
                        p.5,
                        p.6,
                        p.7,
                        p.8,
                        p.9,
                        p.10,
                        p.11,
                        p.12,
                        topk as i32,
                        global as i32,
                        first as i32,
                        groups as i32,
                        topk as i32,
                        sv(&e),
                    )
                };
                assert_eq!(rc, 0);
            });
            println!(
                "TIMING tree={tree} kernel={label} shape=slots{topk}_groups{groups} rep={rep} us_per_call={us:.3}"
            );
        }
    }
    // FP8 wo_a at the served t=1 shape: eight per-group GEMVs against the grouped launch.
    // The weight (32 MiB) exceeds L2, so every call streams it from DRAM.
    {
        let (groups, rows, kdim) = (8usize, 1024usize, 4096usize);
        let sc_cols = kdim / 128;
        let codes: Vec<u8> = (0..groups * rows * kdim)
            .map(|i| ((i * 37 + i / 4096) % 0x7e) as u8 | (((i / 3) & 1) << 7) as u8)
            .collect();
        let scales = fixture(groups * rows / 128 * sc_cols, 8, 1, 2);
        let x: Vec<u16> = (0..groups * kdim)
            .map(|i| 0x3c00 + ((i * 13) % 0x200) as u16)
            .collect();
        let st = e.stream();
        let cd = st.clone_htod(&codes).unwrap();
        let sd = e.htod(&scales).unwrap();
        let xd = st.clone_htod(&x).unwrap();
        let mut yd = e.uninit(groups * rows).unwrap();
        let cp = cd.device_ptr(&st).0 as usize;
        let spp = dp(&sd, &e) as usize;
        let xp = xd.device_ptr(&st).0 as usize;
        let yp = dpm(&mut yd, &e) as usize;
        let launches = 1000;
        for rep in 0..reps {
            let slices = chain_us(&e, launches, || unsafe {
                for g in 0..groups {
                    let rc = k::memra_dsv4_gemv_fp8_m(
                        (cp + g * rows * kdim) as *const c_void,
                        (spp + g * rows / 128 * sc_cols * 4) as *const f32,
                        sc_cols as i32,
                        (xp + g * kdim * 2) as *const c_void,
                        (yp + g * rows * 4) as *mut f32,
                        1,
                        rows as i32,
                        kdim as i32,
                        (groups * kdim) as i32,
                        (groups * rows) as i32,
                        sv(&e),
                    );
                    assert_eq!(rc, 0);
                }
            });
            let grouped = chain_us(&e, launches, || unsafe {
                let rc = k::memra_dsv4_gemv_fp8_grouped_m1(
                    cp as *const c_void,
                    spp as *const f32,
                    sc_cols as i32,
                    xp as *const c_void,
                    yp as *mut f32,
                    groups as i32,
                    rows as i32,
                    kdim as i32,
                    kdim as i32,
                    rows as i32,
                    sv(&e),
                );
                assert_eq!(rc, 0);
            });
            println!(
                "TIMING tree={tree} kernel=wo_a_slices8 shape={groups}x{rows}x{kdim} rep={rep} us_per_call={slices:.3}"
            );
            println!(
                "TIMING tree={tree} kernel=wo_a_grouped shape={groups}x{rows}x{kdim} rep={rep} us_per_call={grouped:.3}"
            );
        }
    }
}
