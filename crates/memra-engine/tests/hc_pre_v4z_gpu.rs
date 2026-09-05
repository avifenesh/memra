//! Gate for `MEMRA_HC_PRE_V4Z` (lane/hc-pre-v4z-20260905): the v4 register schedule with the
//! `rms_norm_zq8_f32_v2` replay folded in is BIT-IDENTICAL to v4 followed by the served norm
//! (`Engine::rms_norm_zq8_f32`, i.e. the kernels.cu kernel at `rms_block()`) on every output:
//! pre, post, comb, y, z, q, d. Inputs carry a wide dynamic range (2^-6..2^6) so a fused
//! multiply-add vs mul+add difference in the sum of squares would show (the narrow inputs of
//! the zq8 gate never did while the served tape forked). Red arm on the norm weights.
use cudarc::driver::{DevicePtr, DevicePtrMut};
use memra_engine::Engine;
use memra_engine::dsv4_ffi as k;
use std::os::raw::c_void;

const HC: usize = 4;
const D: usize = 4096;
const ITERS: i32 = 20;
const EPS_HC: f32 = 1e-6;
const EPS_NORM: f32 = 1e-5;

fn vecf(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let frac = (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            let kexp = ((s >> 8) % 13) as i32 - 6;
            frac * (2.0f32).powi(kexp)
        })
        .collect()
}

struct Out {
    pre: Vec<f32>,
    post: Vec<f32>,
    comb: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    q: Vec<i8>,
    d: Vec<f32>,
}

fn run(e: &Engine, fused: bool, x: &[f32], base: &[f32], norm_w: &[f32]) -> Out {
    let stream = e.stream();
    let sp = |st: &std::sync::Arc<cudarc::driver::CudaStream>| st.cu_stream() as *mut c_void;
    let rows = (2 + HC) * HC;
    let xd = e.htod(x).unwrap();
    let mixes = e.htod(&vecf(rows, 12)).unwrap();
    let scale = e.htod(&[0.7f32, 1.3, 0.9]).unwrap();
    let based = e.htod(base).unwrap();
    let nw = e.htod(norm_w).unwrap();
    let mut pre = e.uninit(HC).unwrap();
    let mut post = e.uninit(HC).unwrap();
    let mut comb = e.uninit(HC * HC).unwrap();
    let mut y = e.uninit(D).unwrap();
    let mut z = e.uninit(D).unwrap();
    let rms_bd = 256i32; // rms_block() default; the served pairing
    let (q, d) = if fused {
        let mut q = e.uninit_i8(D).unwrap();
        let mut d = e.uninit(D / 32).unwrap();
        let rc = unsafe {
            k::memra_dsv4_hc_pre_v4z(
                xd.device_ptr(&stream).0 as *const f32,
                mixes.device_ptr(&stream).0 as *const f32,
                scale.device_ptr(&stream).0 as *const f32,
                based.device_ptr(&stream).0 as *const f32,
                pre.device_ptr_mut(&stream).0 as *mut f32,
                post.device_ptr_mut(&stream).0 as *mut f32,
                comb.device_ptr_mut(&stream).0 as *mut f32,
                y.device_ptr_mut(&stream).0 as *mut f32,
                1,
                HC as i32,
                D as i32,
                ITERS,
                EPS_HC,
                std::ptr::null_mut(),
                nw.device_ptr(&stream).0 as *const f32,
                z.device_ptr_mut(&stream).0 as *mut f32,
                q.device_ptr_mut(&stream).0 as *mut i8,
                d.device_ptr_mut(&stream).0 as *mut f32,
                EPS_NORM,
                rms_bd,
                sp(&stream),
            )
        };
        assert_eq!(rc, 0, "v4z rc");
        (q, d)
    } else {
        let rc = unsafe {
            k::memra_dsv4_hc_pre_v4(
                xd.device_ptr(&stream).0 as *const f32,
                mixes.device_ptr(&stream).0 as *const f32,
                scale.device_ptr(&stream).0 as *const f32,
                based.device_ptr(&stream).0 as *const f32,
                pre.device_ptr_mut(&stream).0 as *mut f32,
                post.device_ptr_mut(&stream).0 as *mut f32,
                comb.device_ptr_mut(&stream).0 as *mut f32,
                y.device_ptr_mut(&stream).0 as *mut f32,
                1,
                HC as i32,
                D as i32,
                ITERS,
                EPS_HC,
                std::ptr::null_mut(),
                512,
                sp(&stream),
            )
        };
        assert_eq!(rc, 0, "v4 rc");
        e.rms_norm_zq8_f32(&y, &nw, &mut z, D, 1, EPS_NORM).unwrap()
    };
    stream.synchronize().unwrap();
    Out {
        pre: e.dtoh(&pre).unwrap(),
        post: e.dtoh(&post).unwrap(),
        comb: e.dtoh(&comb).unwrap(),
        y: e.dtoh(&y).unwrap(),
        z: e.dtoh(&z).unwrap(),
        q: e.dtoh_i8(&q).unwrap(),
        d: e.dtoh(&d).unwrap(),
    }
}

fn bits(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count()
}

#[test]
fn v4z_is_bit_identical_to_v4_then_rms_norm_zq8() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let rows = (2 + HC) * HC;
    let x = vecf(HC * D, 11);
    let base = vecf(rows, 13);
    let norm_w: Vec<f32> = vecf(D, 14).iter().map(|v| 1.0 + v).collect();
    let a = run(&e, false, &x, &base, &norm_w);
    let b = run(&e, true, &x, &base, &norm_w);
    let words = bits(&a.pre, &b.pre)
        + bits(&a.post, &b.post)
        + bits(&a.comb, &b.comb)
        + bits(&a.y, &b.y)
        + bits(&a.z, &b.z)
        + bits(&a.d, &b.d);
    let qbad = a.q.iter().zip(&b.q).filter(|(p, q)| p != q).count();
    assert_eq!(
        (words, qbad),
        (0, 0),
        "v4z vs v4 + rms_norm_zq8: {words} differing f32 words, {qbad} differing q8 bytes \
         (z: {}, d: {})",
        bits(&a.z, &b.z),
        bits(&a.d, &b.d)
    );
    assert!(
        a.d.iter().any(|v| *v != 0.0) && a.q.iter().any(|v| *v != 0),
        "vacuous outputs"
    );
    // red arm: perturbed norm weights must move z / q
    let mut w2 = norm_w.clone();
    w2[7] += 0.5;
    let c = run(&e, true, &x, &base, &w2);
    assert!(
        bits(&a.z, &c.z) > 0,
        "red arm: perturbed norm weights did not move z"
    );
}

#[test]
fn v4z_refuses_a_replay_width_it_cannot_run() {
    let Ok(e) = Engine::new(0) else {
        return;
    };
    let stream = e.stream();
    let sp = |st: &std::sync::Arc<cudarc::driver::CudaStream>| st.cu_stream() as *mut c_void;
    let rows = (2 + HC) * HC;
    let xd = e.htod(&vecf(HC * D, 1)).unwrap();
    let mixes = e.htod(&vecf(rows, 2)).unwrap();
    let scale = e.htod(&[0.7f32, 1.3, 0.9]).unwrap();
    let based = e.htod(&vecf(rows, 3)).unwrap();
    let nw = e.htod(&vecf(D, 4)).unwrap();
    let mut pre = e.uninit(HC).unwrap();
    let mut post = e.uninit(HC).unwrap();
    let mut comb = e.uninit(HC * HC).unwrap();
    let mut y = e.uninit(D).unwrap();
    let mut z = e.uninit(D).unwrap();
    let mut q = e.uninit_i8(D).unwrap();
    let mut d = e.uninit(D / 32).unwrap();
    let rc = unsafe {
        k::memra_dsv4_hc_pre_v4z(
            xd.device_ptr(&stream).0 as *const f32,
            mixes.device_ptr(&stream).0 as *const f32,
            scale.device_ptr(&stream).0 as *const f32,
            based.device_ptr(&stream).0 as *const f32,
            pre.device_ptr_mut(&stream).0 as *mut f32,
            post.device_ptr_mut(&stream).0 as *mut f32,
            comb.device_ptr_mut(&stream).0 as *mut f32,
            y.device_ptr_mut(&stream).0 as *mut f32,
            1,
            HC as i32,
            D as i32,
            ITERS,
            EPS_HC,
            std::ptr::null_mut(),
            nw.device_ptr(&stream).0 as *const f32,
            z.device_ptr_mut(&stream).0 as *mut f32,
            q.device_ptr_mut(&stream).0 as *mut i8,
            d.device_ptr_mut(&stream).0 as *mut f32,
            EPS_NORM,
            2048,
            sp(&stream),
        )
    };
    assert_eq!(rc, 40025);
}
