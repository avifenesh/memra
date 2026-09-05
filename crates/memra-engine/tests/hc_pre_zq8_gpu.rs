//! BIT-IDENTITY GATE for `dsv4_hc_pre_zq8_kernel` (lane/hcpre-zq8-fusion-20260905): the hc
//! pre-chain (v3) and the `rms_norm_zq8_f32_v2` that consumes its `y`, fused into one launch.
//!
//! The claim is exactness, not closeness: stages 1-3 are the v3 body generated verbatim, and
//! the norm is the served twin with its own block width substituted for blockDim, so every one
//! of the four outputs -- `y`, `z`, `q`, `d` -- must match the two-launch program BIT for bit.
//! The reference is the real two-launch program (`memra_dsv4_hc_pre_fused_v3` then
//! `Engine::rms_norm_zq8_f32`), not a host re-derivation. A RED arm swaps the norm weights and
//! asserts `z` and `q` move, so a fused kernel that ignored `norm_w` cannot pass. Exactness only;
//! this runs on the rig 5090.
use cudarc::driver::{DevicePtr, DevicePtrMut};
use memra_engine::Engine;
use memra_engine::dsv4_ffi as k;
use std::os::raw::c_void;

const HC: usize = 4;
const D: usize = 4096;
const ITERS: i32 = 20;
const EPS_HC: f32 = 1e-6;
const EPS_NORM: f32 = 1e-5;

/// Wide-dynamic-range inputs: a full-mantissa fraction times 2^k for k in -6..=6. The old
/// narrow inputs ((s>>40)/2^24 - 0.5)*0.8 never exposed a fused-multiply-add vs mul+add
/// difference in a sum of squares (2026-09-05: the served tape forked while this gate was
/// green); with a spread of exponents the product rounding decides the sum's last bit often.
fn vecf(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let frac = (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            let k = ((s >> 8) % 13) as i32 - 6;
            frac * (2.0f32).powi(k)
        })
        .collect()
}

#[test]
fn fused_hc_pre_zq8_is_bit_identical_to_v3_then_rms_norm_zq8() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let stream = e.stream();
    let sp = |st: &std::sync::Arc<cudarc::driver::CudaStream>| st.cu_stream() as *mut c_void;
    let rows = (2 + HC) * HC;
    let w = HC * D;

    let x = e.htod(&vecf(w, 11)).unwrap();
    let mixes = e.htod(&vecf(rows, 12)).unwrap();
    let scale = e.htod(&[0.7f32, 1.3, 0.9]).unwrap();
    let base = e.htod(&vecf(rows, 13)).unwrap();
    let norm_w = e
        .htod(&vecf(D, 14).iter().map(|v| 1.0 + v).collect::<Vec<_>>())
        .unwrap();

    let block = 512i32;
    let rms_bd = 256i32; // rms_block()'s default; the served pairing on 100a is (512, 256)

    // ---- REFERENCE: the two-launch program.
    let mut pre_r = e.uninit(HC).unwrap();
    let mut post_r = e.uninit(HC).unwrap();
    let mut comb_r = e.uninit(HC * HC).unwrap();
    let mut y_r = e.uninit(D).unwrap();
    unsafe {
        let rc = k::memra_dsv4_hc_pre_fused_v3(
            x.device_ptr(&stream).0 as *const f32,
            mixes.device_ptr(&stream).0 as *const f32,
            scale.device_ptr(&stream).0 as *const f32,
            base.device_ptr(&stream).0 as *const f32,
            pre_r.device_ptr_mut(&stream).0 as *mut f32,
            post_r.device_ptr_mut(&stream).0 as *mut f32,
            comb_r.device_ptr_mut(&stream).0 as *mut f32,
            y_r.device_ptr_mut(&stream).0 as *mut f32,
            1,
            HC as i32,
            D as i32,
            ITERS,
            EPS_HC,
            std::ptr::null_mut(),
            block,
            1,
            sp(&stream),
        );
        assert_eq!(rc, 0, "v3 rc");
    }
    let mut z_r = e.uninit(D).unwrap();
    let (q_r, d_r) = e
        .rms_norm_zq8_f32(&y_r, &norm_w, &mut z_r, D, 1, EPS_NORM)
        .expect("reference norm");
    stream.synchronize().unwrap();

    // ---- FUSED.
    let run_fused = |nw: &cudarc::driver::CudaSlice<f32>| {
        let mut pre = e.uninit(HC).unwrap();
        let mut post = e.uninit(HC).unwrap();
        let mut comb = e.uninit(HC * HC).unwrap();
        let mut y = e.uninit(D).unwrap();
        let mut z = e.uninit(D).unwrap();
        let mut q = e.uninit_i8(D).unwrap();
        let mut d = e.uninit(D / 32).unwrap();
        unsafe {
            let rc = k::memra_dsv4_hc_pre_zq8(
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
                D as i32,
                ITERS,
                EPS_HC,
                std::ptr::null_mut(),
                block,
                1,
                nw.device_ptr(&stream).0 as *const f32,
                z.device_ptr_mut(&stream).0 as *mut f32,
                q.device_ptr_mut(&stream).0 as *mut i8,
                d.device_ptr_mut(&stream).0 as *mut f32,
                rms_bd,
                EPS_NORM,
                sp(&stream),
            );
            assert_eq!(rc, 0, "fused rc");
        }
        stream.synchronize().unwrap();
        (
            e.dtoh(&pre).unwrap(),
            e.dtoh(&post).unwrap(),
            e.dtoh(&comb).unwrap(),
            e.dtoh(&y).unwrap(),
            e.dtoh(&z).unwrap(),
            stream.clone_dtoh(&q).unwrap(),
            e.dtoh(&d).unwrap(),
        )
    };
    let (pre_f, post_f, comb_f, y_f, z_f, q_f, d_f) = run_fused(&norm_w);

    let bits = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .filter(|(p, q)| p.to_bits() != q.to_bits())
            .count()
    };
    assert_eq!(bits(&e.dtoh(&pre_r).unwrap(), &pre_f), 0, "pre differs");
    assert_eq!(bits(&e.dtoh(&post_r).unwrap(), &post_f), 0, "post differs");
    assert_eq!(bits(&e.dtoh(&comb_r).unwrap(), &comb_f), 0, "comb differs");
    assert_eq!(
        bits(&e.dtoh(&y_r).unwrap(), &y_f),
        0,
        "y differs: stages 1-3 are not verbatim"
    );
    assert_eq!(
        bits(&e.dtoh(&z_r).unwrap(), &z_f),
        0,
        "z differs: the norm partition moved"
    );
    assert_eq!(
        stream.clone_dtoh(&q_r).unwrap(),
        q_f,
        "q differs: the q8 epilogue is not the served one"
    );
    assert_eq!(
        bits(&e.dtoh(&d_r).unwrap(), &d_f),
        0,
        "d (per-32 scales) differs"
    );

    // ---- RED ARM: a different norm weight must move z and q. If it does not, the fused
    // kernel never read `norm_w`, and every green assertion above was vacuous for the norm.
    let norm_w2 = e
        .htod(&vecf(D, 99).iter().map(|v| 1.0 + v).collect::<Vec<_>>())
        .unwrap();
    let (_, _, _, _, z2, q2, _) = run_fused(&norm_w2);
    assert!(
        bits(&z_f, &z2) > 0,
        "RED: swapping norm weights left z unchanged"
    );
    assert!(q_f != q2, "RED: swapping norm weights left q unchanged");
}
