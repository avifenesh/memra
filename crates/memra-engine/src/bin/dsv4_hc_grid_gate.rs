//! dsv4-hc-grid-gate: the 65535 grid.y ceiling tooth for the dsv4 hc launchers
//! (hermes finding, fixed 2026-08-23).
//!
//! Before the fix, `memra_dsv4_hc_post` launched `dim3 grid((d+255)/256, s*hc)`: at a
//! 16384-token prefill with hc=4, grid.y = 65536 > 65535 and the launch failed
//! (cudaErrorInvalidConfiguration) — the hc_post 16k-prefill crash.
//! `memra_dsv4_hc_collapse` carried the same shape one power of hc later (s on grid.y).
//! Both launchers now CHUNK over a y0 base offset — this gate proves, at exactly the
//! crashing shape:
//!   (a) the calls SUCCEED (rc == 0) at s=16384, hc=4 (grid.y total 65536, one past the
//!       ceiling — the boundary the bug lived on) and at s=70000 for collapse
//!       (multi-chunk on grid.y = s itself);
//!   (b) the chunked outputs are BIT-IDENTICAL to a CPU f32 reference evaluated in the
//!       kernels' own accumulation order (ascending c / j loops, f32 adds) — the offset
//!       is a pure index shift, so bit equality is the contract, not a tolerance.
//!
//! Usage: dsv4-hc-grid-gate [device]        exit 0 PASS

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use memra_engine::dsv4_ffi as k;
use std::os::raw::c_void;

fn pr(i: usize) -> f32 {
    // deterministic pseudo-random in [-1, 1)
    let h = (i.wrapping_mul(2654435761) >> 8) & 0xffff;
    h as f32 / 32768.0 - 1.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let ctx = CudaContext::new(dev)?;
    let stream = ctx.default_stream();
    let sp = |s: &std::sync::Arc<cudarc::driver::CudaStream>| s.cu_stream() as *mut c_void;
    let mut fails = 0usize;

    // ---- hc_post at the exact crashing shape: s=16384, hc=4 -> grid.y total 65536 ----
    {
        let (s, hc, d) = (16384usize, 4usize, 64usize);
        let f_h: Vec<f32> = (0..s * d).map(|i| pr(i + 3)).collect();
        let res_h: Vec<f32> = (0..s * hc * d).map(|i| pr(i + 17)).collect();
        let post_h: Vec<f32> = (0..s * hc).map(|i| pr(i + 29)).collect();
        let comb_h: Vec<f32> = (0..s * hc * hc).map(|i| pr(i + 41)).collect();
        let f_d = stream.clone_htod(&f_h)?;
        let res_d = stream.clone_htod(&res_h)?;
        let post_d = stream.clone_htod(&post_h)?;
        let comb_d = stream.clone_htod(&comb_h)?;
        let mut out_d = stream.alloc_zeros::<f32>(s * hc * d)?;
        let rc = unsafe {
            k::memra_dsv4_hc_post(
                f_d.device_ptr(&stream).0 as *const f32,
                res_d.device_ptr(&stream).0 as *const f32,
                post_d.device_ptr(&stream).0 as *const f32,
                comb_d.device_ptr(&stream).0 as *const f32,
                out_d.device_ptr_mut(&stream).0 as *mut f32,
                s as i32,
                hc as i32,
                d as i32,
                sp(&stream),
            )
        };
        if rc != 0 {
            println!(
                "hc_post s={s} hc={hc} (grid.y total {}): rc={rc} FAIL",
                s * hc
            );
            fails += 1;
        } else {
            let got = stream.clone_dtoh(&out_d)?;
            // CPU reference in the kernel's own accumulation order.
            let mut bad = 0usize;
            for t in 0..s {
                for kk in 0..hc {
                    for i in 0..d {
                        let mut acc = post_h[t * hc + kk] * f_h[t * d + i];
                        for j in 0..hc {
                            acc += comb_h[(t * hc + j) * hc + kk] * res_h[(t * hc + j) * d + i];
                        }
                        if acc.to_bits() != got[(t * hc + kk) * d + i].to_bits() {
                            bad += 1;
                        }
                    }
                }
            }
            println!(
                "hc_post s={s} hc={hc} d={d} (grid.y total {} = ceiling+1): rc=0 bit-bad={bad}/{} {}",
                s * hc,
                s * hc * d,
                if bad == 0 {
                    "OK"
                } else {
                    fails += 1;
                    "FAIL"
                }
            );
        }
    }

    // ---- hc_collapse past the ceiling on s itself: s=70000 (multi-chunk) ----
    {
        let (s, hc, d) = (70000usize, 2usize, 32usize);
        let x_h: Vec<f32> = (0..s * hc * d).map(|i| pr(i + 7)).collect();
        let pre_h: Vec<f32> = (0..s * hc).map(|i| pr(i + 11)).collect();
        let x_d = stream.clone_htod(&x_h)?;
        let pre_d = stream.clone_htod(&pre_h)?;
        let mut y_d = stream.alloc_zeros::<f32>(s * d)?;
        let rc = unsafe {
            k::memra_dsv4_hc_collapse(
                x_d.device_ptr(&stream).0 as *const f32,
                pre_d.device_ptr(&stream).0 as *const f32,
                y_d.device_ptr_mut(&stream).0 as *mut f32,
                s as i32,
                hc as i32,
                d as i32,
                sp(&stream),
            )
        };
        if rc != 0 {
            println!("hc_collapse s={s} (grid.y = s > ceiling): rc={rc} FAIL");
            fails += 1;
        } else {
            let got = stream.clone_dtoh(&y_d)?;
            let mut bad = 0usize;
            for t in 0..s {
                for i in 0..d {
                    let mut acc = 0.0f32;
                    for c in 0..hc {
                        acc += pre_h[t * hc + c] * x_h[(t * hc + c) * d + i];
                    }
                    if acc.to_bits() != got[t * d + i].to_bits() {
                        bad += 1;
                    }
                }
            }
            println!(
                "hc_collapse s={s} hc={hc} d={d} (2 y-chunks): rc=0 bit-bad={bad}/{} {}",
                s * d,
                if bad == 0 {
                    "OK"
                } else {
                    fails += 1;
                    "FAIL"
                }
            );
        }
    }

    if fails == 0 {
        println!("ALL GREEN: dsv4 hc grid.y ceiling gate");
        Ok(())
    } else {
        Err(format!("{fails} arm(s) FAILED").into())
    }
}
