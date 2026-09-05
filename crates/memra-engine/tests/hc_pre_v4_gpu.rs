//! Gate for `MEMRA_HC_PRE_V4` (lane/hc-pre-phases-20260905): the v4 register schedule of the hc
//! pre-chain is BIT-IDENTICAL to the served v3 kernel (sink_reg=1) on every output -- `pre`,
//! `post`, `comb`, `y` -- against v3 at the served block (512) and at 1024 (v4 itself always
//! runs 1024 threads x 16 elements), on real-shaped inputs (hc=4,
//! d=4096, 20 Sinkhorn rounds). Red arm: v4 on perturbed `base` must differ from v3 on the
//! original, so a "pass" cannot come from comparing two copies of nothing.
use cudarc::driver::{DevicePtr, DevicePtrMut};
use memra_engine::Engine;
use memra_engine::dsv4_ffi as k;
use std::os::raw::c_void;

const HC: usize = 4;
const D: usize = 4096;
const ITERS: i32 = 20;
const EPS_HC: f32 = 1e-6;

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

type Out = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

fn run(e: &Engine, v4: bool, block: i32, base: &[f32], x: &[f32]) -> Out {
    let stream = e.stream();
    let sp = |st: &std::sync::Arc<cudarc::driver::CudaStream>| st.cu_stream() as *mut c_void;
    let rows = (2 + HC) * HC;
    let xd = e.htod(x).unwrap();
    let mixes = e.htod(&vecf(rows, 12)).unwrap();
    let scale = e.htod(&[0.7f32, 1.3, 0.9]).unwrap();
    let based = e.htod(base).unwrap();
    let mut pre = e.uninit(HC).unwrap();
    let mut post = e.uninit(HC).unwrap();
    let mut comb = e.uninit(HC * HC).unwrap();
    let mut y = e.uninit(D).unwrap();
    let rc = unsafe {
        if v4 {
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
                block,
                sp(&stream),
            )
        } else {
            k::memra_dsv4_hc_pre_fused_v3(
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
                block,
                1,
                0,
                sp(&stream),
            )
        }
    };
    assert_eq!(rc, 0, "rc (v4={v4} block={block})");
    stream.synchronize().unwrap();
    (
        e.dtoh(&pre).unwrap(),
        e.dtoh(&post).unwrap(),
        e.dtoh(&comb).unwrap(),
        e.dtoh(&y).unwrap(),
    )
}

fn bits(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count()
}

#[test]
fn hc_pre_v4_is_bit_identical_to_v3() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let rows = (2 + HC) * HC;
    let x = vecf(HC * D, 11);
    let base = vecf(rows, 13);
    // v4 always runs its 1024x16 schedule; v3 is served at 512 and is bit-identical across
    // widths, so v4 must match v3 at BOTH widths.
    for block in [512i32, 1024] {
        let a = run(&e, false, block, &base, &x);
        let b = run(&e, true, block, &base, &x);
        let bad = bits(&a.0, &b.0) + bits(&a.1, &b.1) + bits(&a.2, &b.2) + bits(&a.3, &b.3);
        assert_eq!(bad, 0, "v4 vs v3 at block {block}: {bad} differing words");
        assert!(
            a.3.iter().all(|v| v.is_finite()) && a.3.iter().any(|v| *v != 0.0),
            "vacuous y"
        );
    }
    // red arm: a perturbed base must move the outputs (pre depends on base[0..hc])
    let mut base2 = base.clone();
    base2[0] += 0.25;
    let a = run(&e, false, 512, &base, &x);
    let c = run(&e, true, 512, &base2, &x);
    assert!(
        bits(&a.0, &c.0) + bits(&a.3, &c.3) > 0,
        "red arm: perturbed base did not move pre/y"
    );
}

#[test]
fn hc_pre_v4_refuses_shapes_it_cannot_schedule() {
    // d not a multiple of the block: the launcher must return 40025 (fall-through code), not
    // compute something.
    let Ok(e) = Engine::new(0) else {
        return;
    };
    let stream = e.stream();
    let sp = |st: &std::sync::Arc<cudarc::driver::CudaStream>| st.cu_stream() as *mut c_void;
    let d = 4000usize;
    let rows = (2 + HC) * HC;
    let x = e.htod(&vecf(HC * d, 1)).unwrap();
    let mixes = e.htod(&vecf(rows, 2)).unwrap();
    let scale = e.htod(&[0.7f32, 1.3, 0.9]).unwrap();
    let base = e.htod(&vecf(rows, 3)).unwrap();
    let mut pre = e.uninit(HC).unwrap();
    let mut post = e.uninit(HC).unwrap();
    let mut comb = e.uninit(HC * HC).unwrap();
    let mut y = e.uninit(d).unwrap();
    let rc = unsafe {
        k::memra_dsv4_hc_pre_v4(
            x.device_ptr(&stream).0 as *const f32,
            mixes.device_ptr(&stream).0 as *const f32,
            scale.device_ptr(&stream).0 as *const f32,
            base.device_ptr(&stream).0 as *const f32,
            pre.device_ptr_mut(&stream).0 as *mut f32,
            post.device_ptr_mut(&stream).0 as *mut f32,
            comb.device_ptr_mut(&stream).0 as *mut f32,
            y.device_ptr_mut(&stream).0 as *mut f32,
            1,
            HC as i32,
            d as i32,
            ITERS,
            EPS_HC,
            std::ptr::null_mut(),
            512,
            sp(&stream),
        )
    };
    assert_eq!(rc, 40025);
}
