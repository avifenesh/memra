//! hc-fused-gate: bit-identity + N=5 per-site timing gate for the mHC decode/verify chain
//! at the GLM-5.3-Flash shape (streams=4, n_embd=4096, sinkhorn_iterations=20, eps=1e-6).
//!
//! research/b200-sinkhorn-fusion-20260902/LANE.md is the write-up this gate feeds. Short
//! version: the task that opened this lane asked for ONE launch per site covering
//! `rowsq_scale -> sinkhorn -> collapse -> hc_post`. That four-kernel single-launch fusion
//! is NOT shipped here and is not attempted — `hc_post`'s `f` operand is the SITE'S OWN
//! attention or FFN branch output (`f = branch(rms_norm(y))`, `hyper.rs` line ~18), which
//! runs as its own multi-kernel program (QKV/RoPE/flash-or-MLA-or-KDA attention, or the
//! MoE/FFN branch) strictly between the collapse write and the post read — confirmed at
//! both call sites this gate's arms are drawn from:
//!   * `hybrid_forward.rs::hyper_range_decode_ws_body` (glm5_next persistent T=1 walk):
//!     `pre_t1_ws -> rms_norm -> mixer -> post_t1_ws -> pre_t1_ws -> rms_norm -> ffn -> post_t1_ws`.
//!   * `dsv4_gpu.rs` verify-batch path: `hc_post (attn) -> hc_pre_batch_dev (ffn site) ->
//!     rmsnorm -> moe_verify_dev -> hc_post (ffn)`.
//!
//! In both, `hc_post` sits on the OTHER side of a full attention or FFN sub-layer from the
//! collapse it would need to share a kernel launch with, and the site AFTER it starts with
//! its own mixes GEMM before `rowsq_scale` runs. No same-launch fusion bridges either gap
//! without inlining attention/FFN math into this glue kernel, which is a different scope
//! (and a different lane's kernels) than the mHC pre-chain this gate exists to qualify.
//!
//! What IS fusable, and already shipped (lane/glm5-decode-diet, 2026-08-31, unmodified by
//! this lane): `rowsq_scale + hc_sinkhorn_m + hc_collapse` -> ONE `memra_dsv4_hc_pre_fused`
//! launch per site, door `MEMRA_HC_FUSED_PRE`. Per the b200-sinkhorn-fusion-20260902 nsys
//! census (2x B200 SXM, GLM-5.3-Flash NVFP4, resident PP2, plain decode, t=1, both devices
//! summed, per token): sinkhorn 130x20.3us=2.64ms, rowsq 130x4.8us=0.62ms, collapse
//! 130x1.8us=0.23ms — that pre-chain is ~3.5ms of the ~3.8ms four-kernel total (~92%);
//! hc_post is 130x2.4us=0.31ms (~8%) and is NOT reachable by this fusion for the reason
//! above. This gate re-proves that existing fusion's bit-identity at the real GLM-5.3-Flash
//! shape (the shipped gate `hc_fused_pre_gpu.rs` proves it generically across several
//! (hc,d) pairs; this one pins the production shape and adds N=5 device timings, which is
//! the box evidence `MEMRA_HC_FUSED_PRE`'s FLAGS.md row says it is still missing) and times
//! `hc_post` alone for census-completeness — clearly NOT claimed as fused with anything.
//!
//! Run under the fleet GPU lock: `MEMRA_GPU_LOCK=/tmp/memra-gpu.lock` (box1 / any 2x RTX
//! PRO 6000 pair / B200 pods — see CLAUDE.md "Lock names are a correctness surface").
//!
//! Usage: hc-fused-gate [device]   exit 0 on PASS (bit-identical fused vs unfused at every
//! tested t), prints per-arm N=5 timings (us) to stdout as both a table and one JSON line.

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use memra_engine::dsv4_ffi as k;
use std::os::raw::c_void;

/// One arm's four readbacks in gate order: `(pre, post, comb, y)`. Named so both arm closures
/// declare the same shape without tripping clippy's type-complexity bar.
type ArmOut = Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), Box<dyn std::error::Error>>;

const HC: usize = 4; // GLM-5.3-Flash mHC stream count
const D: usize = 4096; // GLM-5.3-Flash n_embd
const ITERS: i32 = 20; // GLM-5.3-Flash hc_sinkhorn_iters (hf_mapping.rs default; hyper.rs/tests)
const EPS: f32 = 1e-6; // GLM-5.3-Flash hc epsilon
const N_TIMED: usize = 5;

fn pr(i: usize, salt: u64) -> f32 {
    // deterministic pseudo-random f32 in [-1, 1) — fixture values, not statistics.
    let mut s = (i as u64).wrapping_mul(6_364_136_223_846_793_005) ^ salt;
    s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((s >> 33) as u32 as f32) / (u32::MAX as f32 / 2.0) - 1.0
}

fn vecf(n: usize, salt: u64) -> Vec<f32> {
    (0..n).map(|i| pr(i, salt)).collect()
}

struct Site {
    x: Vec<f32>,
    mixes: Vec<f32>,
    scale: Vec<f32>,
    base: Vec<f32>,
}

fn site(t: usize, salt: u64) -> Site {
    let rows = (2 + HC) * HC;
    Site {
        x: vecf(t * HC * D, salt ^ 0x11),
        mixes: vecf(t * rows, salt ^ 0x22),
        scale: vecf(3, salt ^ 0x33),
        base: vecf(rows, salt ^ 0x44),
    }
}

/// Sum of to_bits mismatches across pre/post/comb/y.
fn bit_diffs(
    a: &(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>),
    b: &(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>),
) -> usize {
    let cmp = |x: &[f32], y: &[f32]| {
        x.iter()
            .zip(y)
            .filter(|(u, v)| u.to_bits() != v.to_bits())
            .count()
    };
    cmp(&a.0, &b.0) + cmp(&a.1, &b.1) + cmp(&a.2, &b.2) + cmp(&a.3, &b.3)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
        // SAFETY: no CUDA call has been made yet in this process.
        unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
    }
    let dev: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let ctx = CudaContext::new(dev)?;
    let stream = ctx.default_stream();
    let sp = |s: &std::sync::Arc<cudarc::driver::CudaStream>| s.cu_stream() as *mut c_void;

    let rows = (2 + HC) * HC;
    let mut fails = 0usize;
    let mut timing_rows: Vec<String> = Vec::new();
    let mut json_arms: Vec<String> = Vec::new();

    for &t in &[1usize, 4, 8] {
        let s = site(t, 0xC0FFEE ^ t as u64);

        // ---- correctness: unfused chain vs fused kernel, same operand bytes ----
        let x_d = stream.clone_htod(&s.x)?;
        let scale_d = stream.clone_htod(&s.scale)?;
        let base_d = stream.clone_htod(&s.base)?;

        let run_unfused = || -> ArmOut {
            let mut mixes_d = stream.clone_htod(&s.mixes)?;
            let mut pre_d = stream.alloc_zeros::<f32>(t * HC)?;
            let mut post_d = stream.alloc_zeros::<f32>(t * HC)?;
            let mut comb_d = stream.alloc_zeros::<f32>(t * HC * HC)?;
            let mut y_d = stream.alloc_zeros::<f32>(t * D)?;
            unsafe {
                let rc = k::memra_dsv4_rowsq_scale(
                    x_d.device_ptr(&stream).0 as *const f32,
                    mixes_d.device_ptr_mut(&stream).0 as *mut f32,
                    t as i32,
                    (HC * D) as i32,
                    rows as i32,
                    EPS,
                    sp(&stream),
                );
                assert_eq!(rc, 0, "rowsq_scale rc");
                let rc = k::memra_dsv4_hc_sinkhorn_m(
                    mixes_d.device_ptr(&stream).0 as *const f32,
                    scale_d.device_ptr(&stream).0 as *const f32,
                    base_d.device_ptr(&stream).0 as *const f32,
                    pre_d.device_ptr_mut(&stream).0 as *mut f32,
                    post_d.device_ptr_mut(&stream).0 as *mut f32,
                    comb_d.device_ptr_mut(&stream).0 as *mut f32,
                    t as i32,
                    HC as i32,
                    ITERS,
                    EPS,
                    sp(&stream),
                );
                assert_eq!(rc, 0, "hc_sinkhorn_m rc");
                let rc = k::memra_dsv4_hc_collapse(
                    x_d.device_ptr(&stream).0 as *const f32,
                    pre_d.device_ptr(&stream).0 as *const f32,
                    y_d.device_ptr_mut(&stream).0 as *mut f32,
                    t as i32,
                    HC as i32,
                    D as i32,
                    sp(&stream),
                );
                assert_eq!(rc, 0, "hc_collapse rc");
            }
            stream.synchronize()?;
            Ok((
                stream.clone_dtoh(&pre_d)?,
                stream.clone_dtoh(&post_d)?,
                stream.clone_dtoh(&comb_d)?,
                stream.clone_dtoh(&y_d)?,
            ))
        };

        let run_fused = || -> ArmOut {
            let mixes_d = stream.clone_htod(&s.mixes)?; // read-only: the fused kernel applies rowsq internally
            let mut pre_d = stream.alloc_zeros::<f32>(t * HC)?;
            let mut post_d = stream.alloc_zeros::<f32>(t * HC)?;
            let mut comb_d = stream.alloc_zeros::<f32>(t * HC * HC)?;
            let mut y_d = stream.alloc_zeros::<f32>(t * D)?;
            unsafe {
                let rc = k::memra_dsv4_hc_pre_fused(
                    x_d.device_ptr(&stream).0 as *const f32,
                    mixes_d.device_ptr(&stream).0 as *const f32,
                    scale_d.device_ptr(&stream).0 as *const f32,
                    base_d.device_ptr(&stream).0 as *const f32,
                    pre_d.device_ptr_mut(&stream).0 as *mut f32,
                    post_d.device_ptr_mut(&stream).0 as *mut f32,
                    comb_d.device_ptr_mut(&stream).0 as *mut f32,
                    y_d.device_ptr_mut(&stream).0 as *mut f32,
                    t as i32,
                    HC as i32,
                    D as i32,
                    ITERS,
                    EPS,
                    std::ptr::null_mut(),
                    sp(&stream),
                );
                assert_eq!(rc, 0, "hc_pre_fused rc");
            }
            stream.synchronize()?;
            Ok((
                stream.clone_dtoh(&pre_d)?,
                stream.clone_dtoh(&post_d)?,
                stream.clone_dtoh(&comb_d)?,
                stream.clone_dtoh(&y_d)?,
            ))
        };

        let unfused_out = run_unfused()?;
        let fused_out = run_fused()?;
        let bad = bit_diffs(&unfused_out, &fused_out);
        println!(
            "[correctness] t={t} hc={HC} d={D}: fused-vs-unfused bit-bad={bad}/{} {}",
            t * HC + t * HC + t * HC * HC + t * D,
            if bad == 0 { "PASS" } else { "FAIL" }
        );
        if bad != 0 {
            fails += 1;
        }

        // ---- N=5 timing: unfused chain (3 launches) vs fused (1 launch) ----
        let mut unfused_us = Vec::with_capacity(N_TIMED);
        for _ in 0..N_TIMED {
            stream.synchronize()?;
            let t0 = std::time::Instant::now();
            let _ = run_unfused()?; // includes its own trailing sync + dtoh, matching real usage
            unfused_us.push(t0.elapsed().as_micros() as u64);
        }
        let mut fused_us = Vec::with_capacity(N_TIMED);
        for _ in 0..N_TIMED {
            stream.synchronize()?;
            let t0 = std::time::Instant::now();
            let _ = run_fused()?;
            fused_us.push(t0.elapsed().as_micros() as u64);
        }

        // ---- hc_post alone, N=5: census-context only, NOT fused with anything ----
        let f_h = vecf(t * D, 0xF00D ^ t as u64);
        let res_h = vecf(t * HC * D, 0xBEEF ^ t as u64);
        let post_h = vecf(t * HC, 0xCAFE ^ t as u64);
        let comb_h = vecf(t * HC * HC, 0xD00D ^ t as u64);
        let f_d = stream.clone_htod(&f_h)?;
        let res_d = stream.clone_htod(&res_h)?;
        let post_d = stream.clone_htod(&post_h)?;
        let comb_d = stream.clone_htod(&comb_h)?;
        let mut post_us = Vec::with_capacity(N_TIMED);
        for _ in 0..N_TIMED {
            let mut out_d = stream.alloc_zeros::<f32>(t * HC * D)?;
            stream.synchronize()?;
            let t0 = std::time::Instant::now();
            unsafe {
                let rc = k::memra_dsv4_hc_post(
                    f_d.device_ptr(&stream).0 as *const f32,
                    res_d.device_ptr(&stream).0 as *const f32,
                    post_d.device_ptr(&stream).0 as *const f32,
                    comb_d.device_ptr(&stream).0 as *const f32,
                    out_d.device_ptr_mut(&stream).0 as *mut f32,
                    t as i32,
                    HC as i32,
                    D as i32,
                    sp(&stream),
                );
                assert_eq!(rc, 0, "hc_post rc");
            }
            stream.synchronize()?;
            post_us.push(t0.elapsed().as_micros() as u64);
        }

        let mean = |v: &[u64]| v.iter().sum::<u64>() as f64 / v.len() as f64;
        timing_rows.push(format!(
            "t={t:<2} unfused(rowsq+sinkhorn+collapse)={:>7.1}us/call  fused(hc_pre_fused)={:>7.1}us/call  hc_post(unfused, not part of this fusion)={:>7.1}us/call  runs={unfused_us:?} / {fused_us:?} / {post_us:?}",
            mean(&unfused_us), mean(&fused_us), mean(&post_us)
        ));
        json_arms.push(format!(
            "{{\"t\":{t},\"hc\":{HC},\"d\":{D},\"bit_bad\":{bad},\"unfused_us\":{unfused_us:?},\"fused_us\":{fused_us:?},\"hc_post_us\":{post_us:?}}}"
        ));
    }

    println!(
        "---- timing (N={N_TIMED} device-synchronized wall time, includes dtoh — informational, not a serving-shape claim) ----"
    );
    for r in &timing_rows {
        println!("{r}");
    }
    println!("HC_FUSED_GATE_JSON [{}]", json_arms.join(","));

    if fails == 0 {
        println!(
            "hc-fused-gate: PASS ({} shapes, bit-identical fused vs unfused)",
            3
        );
        Ok(())
    } else {
        eprintln!("hc-fused-gate: FAIL ({fails} shape(s) mismatched)");
        std::process::exit(1);
    }
}
