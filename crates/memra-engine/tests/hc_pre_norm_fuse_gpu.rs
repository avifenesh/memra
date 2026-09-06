//! Gate for the hc-pre kernel with `rms_norm_zq8_f32`'s epilogue folded in
//! (lane/hc-pre-norm-fuse-20260906).
//!
//! WHY THE FUSION EXISTS. The decode census puts 39% of the token in kernels that move almost no
//! bytes, and ncu on the hc-pre kernel itself says why they cannot be fixed from the inside: 4
//! active warps, 0.15 eligible, 88.4% of cycles with nothing to issue. At grid 1 it cannot fill a
//! 148-SM part, and seven arms rearranging its interior proved that the expensive way. The lever
//! is to DELETE the neighbouring launch: `rms_norm_zq8_f32` runs immediately after, on the vector
//! hc-pre just wrote, and costs ~3.9 us x ~79 launches per token.
//!
//! WHAT THIS GATE PROVES. Not "close": BITWISE equality of all four outputs (`z`, the q8 codes,
//! the q8 scales, and `y` itself) between the fused kernel and the exact two-launch pair it
//! replaces, on the served geometry. The claim rests on an index coincidence rather than a
//! tolerance, so a bitwise bar is the honest one: at BLOCK 1024 and d 4096, both passes of
//! `rms_norm_zq8_f32` own exactly the four elements (tid, tid+1024, tid+2048, tid+3072) that the
//! hc-pre combine tail already holds in registers, in the same order.
//!
//! RED ARM: perturbing one element of the norm weight must move the fused output. Without it the
//! bitwise pass could be reading a norm that never ran.
//!
//! PRECONDITION, and it is load-bearing: `rms_norm_zq8_f32` launches at `rms_block()` threads, and
//! the index coincidence above holds at 1024. At any other block size the unfused reduction walks
//! a different per-thread element set and the two are NOT bit-identical. The served posture sets
//! `MEMRA_RMS_BLOCK=1024`; this gate pins it explicitly rather than inheriting it, so a change to
//! that default fails here instead of silently moving decode bits.
use memra_engine::Engine;
use memra_engine::dsv4_ffi as k;
use std::os::raw::c_void;

const HC: usize = 4;
const D: usize = 4096;
const ITERS: i32 = 20;
const EPS: f32 = 1e-6;
const NORM_EPS: f32 = 1e-5;

fn vecf(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * amp
        })
        .collect()
}

struct Site {
    x: cudarc::driver::CudaSlice<f32>,
    mixes: cudarc::driver::CudaSlice<f32>,
    scale: cudarc::driver::CudaSlice<f32>,
    base: cudarc::driver::CudaSlice<f32>,
    nw: cudarc::driver::CudaSlice<f32>,
}

fn site(e: &Engine, nw_perturb: Option<usize>) -> Site {
    let rows = (2 + HC) * HC;
    let mut nw = vecf(D, 99, 1.0);
    if let Some(i) = nw_perturb {
        nw[i] += 0.5;
    }
    Site {
        x: e.htod(&vecf(HC * D, 11, 1.0)).unwrap(),
        mixes: e.htod(&vecf(rows, 22, 1.0)).unwrap(),
        scale: e.htod(&vecf(rows, 33, 0.5)).unwrap(),
        base: e.htod(&vecf(rows, 44, 0.5)).unwrap(),
        nw: e.htod(&nw).unwrap(),
    }
}

/// Runs the fused kernel and returns (z, q, d, y).
#[allow(clippy::type_complexity)]
fn fused(e: &Engine, s: &Site) -> (Vec<f32>, Vec<i8>, Vec<f32>, Vec<f32>) {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let rows = (2 + HC) * HC;
    let nblk = D / 32;
    let mut pre = e.zeros(rows).unwrap();
    let mut post = e.zeros(rows).unwrap();
    let mut comb = e.zeros(rows).unwrap();
    let mut y = e.zeros(D).unwrap();
    let mut z = e.zeros(D).unwrap();
    let mut q = e.htod_i8(&vec![0i8; D]).unwrap();
    let mut dq = e.zeros(nblk).unwrap();
    let st = e.stream();
    // SAFETY: every buffer above is sized to the geometry the launcher audits (hc 4, d 4096).
    let rc = unsafe {
        k::memra_dsv4_hc_pre_v4_norm_zq8(
            s.x.device_ptr(&st).0 as *const f32,
            s.mixes.device_ptr(&st).0 as *const f32,
            s.scale.device_ptr(&st).0 as *const f32,
            s.base.device_ptr(&st).0 as *const f32,
            pre.device_ptr_mut(&st).0 as *mut f32,
            post.device_ptr_mut(&st).0 as *mut f32,
            comb.device_ptr_mut(&st).0 as *mut f32,
            y.device_ptr_mut(&st).0 as *mut f32,
            1,
            HC as i32,
            D as i32,
            ITERS,
            EPS,
            std::ptr::null_mut(),
            1024,
            s.nw.device_ptr(&st).0 as *const f32,
            z.device_ptr_mut(&st).0 as *mut f32,
            q.device_ptr_mut(&st).0 as *mut i8,
            dq.device_ptr_mut(&st).0 as *mut f32,
            NORM_EPS,
            st.cu_stream() as *mut c_void,
        )
    };
    assert_eq!(rc, 0, "fused rc {rc}");
    (
        e.dtoh_view(&z.slice(0..D)).unwrap(),
        e.dtoh_i8(&q).unwrap(),
        e.dtoh_view(&dq.slice(0..nblk)).unwrap(),
        e.dtoh_view(&y.slice(0..D)).unwrap(),
    )
}

/// Runs the two-launch pair the fusion replaces and returns the same four outputs.
#[allow(clippy::type_complexity)]
fn unfused(e: &Engine, s: &Site) -> (Vec<f32>, Vec<i8>, Vec<f32>, Vec<f32>) {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let rows = (2 + HC) * HC;
    let mut pre = e.zeros(rows).unwrap();
    let mut post = e.zeros(rows).unwrap();
    let mut comb = e.zeros(rows).unwrap();
    let mut y = e.zeros(D).unwrap();
    let st = e.stream();
    // SAFETY: as above, the same geometry the unfused launcher audits.
    let rc = unsafe {
        k::memra_dsv4_hc_pre_v4(
            s.x.device_ptr(&st).0 as *const f32,
            s.mixes.device_ptr(&st).0 as *const f32,
            s.scale.device_ptr(&st).0 as *const f32,
            s.base.device_ptr(&st).0 as *const f32,
            pre.device_ptr_mut(&st).0 as *mut f32,
            post.device_ptr_mut(&st).0 as *mut f32,
            comb.device_ptr_mut(&st).0 as *mut f32,
            y.device_ptr_mut(&st).0 as *mut f32,
            1,
            HC as i32,
            D as i32,
            ITERS,
            EPS,
            std::ptr::null_mut(),
            1024,
            st.cu_stream() as *mut c_void,
        )
    };
    assert_eq!(rc, 0, "unfused rc {rc}");
    let mut z = e.zeros(D).unwrap();
    let (q, dq) = e
        .rms_norm_zq8_f32(&y, &s.nw, &mut z, D, 1, NORM_EPS)
        .unwrap();
    let nblk = D / 32;
    (
        e.dtoh_view(&z.slice(0..D)).unwrap(),
        e.dtoh_i8(&q).unwrap(),
        e.dtoh_view(&dq.slice(0..nblk)).unwrap(),
        e.dtoh_view(&y.slice(0..D)).unwrap(),
    )
}

#[test]
fn fused_epilogue_is_bitwise_the_two_launch_pair() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    // SAFETY: single-threaded test setup, before any engine call that reads the value.
    unsafe { std::env::set_var("MEMRA_RMS_BLOCK", "1024") };
    let s = site(&e, None);
    let (zf, qf, df, yf) = fused(&e, &s);
    let (zu, qu, du, yu) = unfused(&e, &s);
    for i in 0..D {
        assert_eq!(yf[i].to_bits(), yu[i].to_bits(), "y differs at {i}");
        assert_eq!(zf[i].to_bits(), zu[i].to_bits(), "z differs at {i}");
        assert_eq!(qf[i], qu[i], "q8 code differs at {i}");
    }
    for b in 0..D / 32 {
        assert_eq!(
            df[b].to_bits(),
            du[b].to_bits(),
            "q8 scale differs at block {b}"
        );
    }
}

/// RED ARM: the norm weight must reach the fused output. A bitwise pass on a norm that never ran
/// would be the vacuous pass this arm exists to close.
#[test]
fn perturbing_the_norm_weight_moves_the_fused_output() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    // SAFETY: as above.
    unsafe { std::env::set_var("MEMRA_RMS_BLOCK", "1024") };
    let (z0, ..) = fused(&e, &site(&e, None));
    let (z1, ..) = fused(&e, &site(&e, Some(1234)));
    let moved = (0..D)
        .filter(|&i| z0[i].to_bits() != z1[i].to_bits())
        .count();
    assert!(
        moved > 0,
        "red arm: a perturbed norm weight changed nothing"
    );
}
