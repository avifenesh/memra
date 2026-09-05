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
//! B200 box receipt (2x SXM, dev 0, N=5, bit-bad=0 at t=1/4/8), `=1` vs unfused: t=1
//! unfused=112.6us fused=101.0us (`hc_post` alone=13.8us); t=4 unfused=118.2us
//! fused=117.6us; t=8 unfused=140.6us fused=123.0us — matches nsys's 32.8us/launch figure
//! once host launch+sync overhead is subtracted, and is the receipt `MEMRA_HC_FUSED_PRE`'s
//! FLAGS.md row named as missing.
//!
//! `MEMRA_HC_FUSED_PRE=2` (lane/b200-sinkhorn-fusion-20260902 follow-up, same door): that
//! B200 receipt showed the `=1` kernel itself (`dsv4_hc_pre_fused_kernel`) at 32.8us/launch
//! average in serving, 15.6% of GPU time — at t=1 the real per-site math (hc=4, d=4096) is
//! tiny, so most of that time is up to 20 serial `__syncthreads()` pairs over a 128-thread
//! block synchronizing work only threads t<hc (all within warp 0) ever touch. `=2`
//! (`dsv4_hc_pre_fused_v2_kernel`) runs the Sinkhorn stage warp-0-only with `__syncwarp()`
//! in place of `__syncthreads()` when hc<=4 — a synchronization-primitive substitution
//! only (same operands, same order), so it is bit-identical to `=1` and to the unfused
//! chain by construction; the host wrapper falls back to `=1`'s kernel for hc>4. This gate
//! proves `=2`'s bit-identity against BOTH the unfused chain and `=1`, and times all three
//! arms at every tested t.
//!
//! Run under the fleet GPU lock: `MEMRA_GPU_LOCK=/tmp/memra-gpu.lock` (box1 / any 2x RTX
//! PRO 6000 pair / B200 pods — see CLAUDE.md "Lock names are a correctness surface").
//!
//! Usage: hc-fused-gate [device]   exit 0 on PASS (bit-identical unfused / fused(=1) /
//! fused(=2) at every tested t), prints per-arm N=5 timings (us) to stdout as both a table
//! and one JSON line.

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use memra_engine::dsv4_ffi as k;
use std::os::raw::c_void;

/// (pre, post, comb, y) — the fused/unfused pre-chain's four outputs.
type Hc4 = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
type Res<T> = Result<T, Box<dyn std::error::Error>>;

const HC: usize = 4; // GLM-5.3-Flash mHC stream count
const D: usize = 4096; // GLM-5.3-Flash n_embd
/// GLM-5.3-Flash `hc_sinkhorn_iters` (hf_mapping.rs default; hyper.rs/tests).
///
/// `MEMRA_HC_GATE_ITERS` overrides it FOR MEASUREMENT ONLY: timing the same kernel at several
/// iteration counts separates the Sinkhorn's serial per-iteration cost (the slope) from stages 1
/// and 3 plus launch (the intercept), which is the split that decides whether this kernel is worth
/// restructuring. It never changes what the engine runs -- only this gate reads it, and the
/// correctness arms below still run at whatever value is set, so a non-default value still has to
/// be bit-identical across all three arms.
fn iters() -> i32 {
    std::env::var("MEMRA_HC_GATE_ITERS")
        .ok()
        .and_then(|v| v.trim().parse::<i32>().ok())
        .filter(|&v| v >= 1)
        .unwrap_or(20)
}
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
fn bit_diffs(a: &Hc4, b: &Hc4) -> usize {
    let cmp = |x: &[f32], y: &[f32]| {
        x.iter()
            .zip(y)
            .filter(|(u, v)| u.to_bits() != v.to_bits())
            .count()
    };
    cmp(&a.0, &b.0) + cmp(&a.1, &b.1) + cmp(&a.2, &b.2) + cmp(&a.3, &b.3)
}

/// `MEMRA_HC_BW_PROBE=<MiB>`: measure this part's ACHIEVABLE streaming bandwidth, then exit.
///
/// WHY THIS EXISTS. Every decode matvec on the 2x B200 pair tops out near 2.2 TB/s and most sit
/// far lower, and the whole roofline argument for the 230 tok/s target was written against a
/// nominal 8 TB/s. That nominal is a spec sheet, not a measurement, and no kernel in the bench
/// had ever been compared against what the part actually delivers. `dsv4_hc_collapse_kernel` is
/// the cleanest streamer available without adding a kernel: at hc=4 it reads `4*d` floats and
/// writes `d`, one pass, fully coalesced, grid `d/256` -- a pure sequential read at any size.
fn bandwidth_probe(mib: usize) -> Res<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // d chosen so the READ side is `mib` MiB: read = 4*d*4 bytes.
    let d = (mib * 1024 * 1024) / 16;
    let x = stream.alloc_zeros::<f32>(4 * d)?;
    let pre = stream.alloc_zeros::<f32>(4)?;
    let mut y = stream.alloc_zeros::<f32>(d)?;
    let read_b = (4 * d * 4) as f64;
    let write_b = (d * 4) as f64;
    let mut best = f64::MAX;
    for _ in 0..7 {
        stream.synchronize()?;
        let t0 = std::time::Instant::now();
        unsafe {
            let rc = k::memra_dsv4_hc_collapse(
                x.device_ptr(&stream).0 as *const f32,
                pre.device_ptr(&stream).0 as *const f32,
                y.device_ptr_mut(&stream).0 as *mut f32,
                1,
                4,
                d as i32,
                stream.cu_stream() as *mut c_void,
            );
            assert_eq!(rc, 0, "hc_collapse rc");
        }
        stream.synchronize()?;
        let us = t0.elapsed().as_secs_f64() * 1e6;
        if us < best {
            best = us;
        }
    }
    println!(
        "[bw-probe] {mib} MiB read + {:.0} MiB write in {best:.1} us => {:.0} GB/s read, \
         {:.0} GB/s read+write",
        write_b / 1048576.0,
        read_b / best / 1e3,
        (read_b + write_b) / best / 1e3
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if let Ok(v) = std::env::var("MEMRA_HC_BW_PROBE") {
        let mib: usize = v.trim().parse().unwrap_or(1024);
        return bandwidth_probe(mib);
    }

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

        let run_unfused = || -> Res<Hc4> {
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
                    iters(),
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

        let run_fused = || -> Res<Hc4> {
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
                    iters(),
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

        // MEMRA_HC_FUSED_PRE=2 (lane/b200-sinkhorn-fusion-20260902 follow-up): same
        // stages as the fused kernel above, Sinkhorn warp-scoped for hc<=4. This gate
        // proves it bit-identical to BOTH the unfused chain and the =1 fused kernel.
        let run_fused_v2 = || -> Res<Hc4> {
            let mixes_d = stream.clone_htod(&s.mixes)?;
            let mut pre_d = stream.alloc_zeros::<f32>(t * HC)?;
            let mut post_d = stream.alloc_zeros::<f32>(t * HC)?;
            let mut comb_d = stream.alloc_zeros::<f32>(t * HC * HC)?;
            let mut y_d = stream.alloc_zeros::<f32>(t * D)?;
            unsafe {
                let rc = k::memra_dsv4_hc_pre_fused_v2(
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
                    iters(),
                    EPS,
                    std::ptr::null_mut(),
                    sp(&stream),
                );
                assert_eq!(rc, 0, "hc_pre_fused_v2 rc");
            }
            stream.synchronize()?;
            Ok((
                stream.clone_dtoh(&pre_d)?,
                stream.clone_dtoh(&post_d)?,
                stream.clone_dtoh(&comb_d)?,
                stream.clone_dtoh(&y_d)?,
            ))
        };

        // v3 (`memra_dsv4_hc_pre_fused_v3`) is the kernel the ENGINE actually serves -- v2's
        // stages with the block size and the Sinkhorn arm as parameters. `sink_reg` selects:
        // 0 = the shared-memory Sinkhorn, 1 = the warp-shuffle register Sinkhorn,
        // 2 = the ALL-REGISTER arm, where every lane holds the whole 4x4 matrix and the
        // shuffles and the ballot disappear. All three must be bit-identical to the unfused
        // chain, because each is a synchronization/placement change over the same addends in
        // the same order -- never a numeric class. Gating all three here is what lets the
        // served arm be chosen on speed alone.
        let run_fused_v3 = |sink_reg: i32, block: i32, split: i32| -> Res<Hc4> {
            let mixes_d = stream.clone_htod(&s.mixes)?;
            let mut pre_d = stream.alloc_zeros::<f32>(t * HC)?;
            let mut post_d = stream.alloc_zeros::<f32>(t * HC)?;
            let mut comb_d = stream.alloc_zeros::<f32>(t * HC * HC)?;
            let mut y_d = stream.alloc_zeros::<f32>(t * D)?;
            unsafe {
                let rc = k::memra_dsv4_hc_pre_fused_v3(
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
                    iters(),
                    EPS,
                    std::ptr::null_mut(),
                    block,
                    sink_reg,
                    split,
                    sp(&stream),
                );
                assert_eq!(
                    rc, 0,
                    "hc_pre_fused_v3 rc (sink_reg={sink_reg} block={block})"
                );
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
        let fused_v2_out = run_fused_v2()?;
        let bad = bit_diffs(&unfused_out, &fused_out);
        let bad_v2_vs_unfused = bit_diffs(&unfused_out, &fused_v2_out);
        let bad_v2_vs_v1 = bit_diffs(&fused_out, &fused_v2_out);
        println!(
            "[correctness] t={t} hc={HC} d={D}: fused(=1)-vs-unfused bit-bad={bad}/{tot} {} | \
             fused(=2)-vs-unfused bit-bad={bad_v2_vs_unfused}/{tot} {} | \
             fused(=2)-vs-fused(=1) bit-bad={bad_v2_vs_v1}/{tot} {}",
            if bad == 0 { "PASS" } else { "FAIL" },
            if bad_v2_vs_unfused == 0 {
                "PASS"
            } else {
                "FAIL"
            },
            if bad_v2_vs_v1 == 0 { "PASS" } else { "FAIL" },
            tot = t * HC + t * HC + t * HC * HC + t * D,
        );
        // v3's three Sinkhorn arms, at the served 512-wide block, each against the unfused chain.
        let mut v3_bad = 0usize;
        for sr in [0i32, 1] {
            for split in [0i32, 1] {
                let out = run_fused_v3(sr, 512, split)?;
                let b = bit_diffs(&unfused_out, &out);
                v3_bad += b;
                println!(
                    "[correctness] t={t} hc={HC} d={D}: \
                     v3(sink_reg={sr},block=512,split_collapse={split})-vs-unfused \
                     bit-bad={b}/{tot} {}",
                    if b == 0 { "PASS" } else { "FAIL" },
                    tot = t * HC + t * HC + t * HC * HC + t * D,
                );
            }
        }
        if bad != 0 || bad_v2_vs_unfused != 0 || bad_v2_vs_v1 != 0 || v3_bad != 0 {
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
        let mut fused_v2_us = Vec::with_capacity(N_TIMED);
        for _ in 0..N_TIMED {
            stream.synchronize()?;
            let t0 = std::time::Instant::now();
            let _ = run_fused_v2()?;
            fused_v2_us.push(t0.elapsed().as_micros() as u64);
        }

        // ---- v3's three Sinkhorn arms, N=5 each, at the served 512-wide block ----
        // These are what the engine actually runs, so this is the comparison that picks the arm.
        // Wall time includes the dtoh every arm pays equally, so the DIFFERENCE between arms is
        // the kernel difference; the absolute figures are not a serving-shape claim.
        // Sweep the BLOCK WIDTH on the served arm. Stages 1 and 3 scale with the block; the
        // Sinkhorn does not (it is warp-0-only at every width). So the width slope is what says
        // how much of the kernel is the parallel work -- which is the number that decides whether
        // giving those stages a real GRID is worth building, and the one thing a single-width
        // measurement cannot tell you.
        let mut v3_us: Vec<(i32, i32, i32, Vec<u64>)> = Vec::new();
        for sr in [0i32, 1] {
            for block in [128i32, 512] {
                for split in [0i32, 1] {
                    let mut us = Vec::with_capacity(N_TIMED);
                    for _ in 0..N_TIMED {
                        stream.synchronize()?;
                        let t0 = std::time::Instant::now();
                        let _ = run_fused_v3(sr, block, split)?;
                        us.push(t0.elapsed().as_micros() as u64);
                    }
                    us.sort_unstable();
                    println!(
                        "[v3-timing] t={t} sink_reg={sr} block={block} split={split} \
                         iters={} median={}us runs={us:?}",
                        iters(),
                        us[us.len() / 2]
                    );
                    v3_us.push((sr, block, split, us));
                }
            }
        }

        // ---- v4: bit identity against v3 (served config) and interleaved timing ----
        let run_v4 = |block: i32| -> Res<Hc4> {
            let mixes_d = stream.clone_htod(&s.mixes)?;
            let mut pre_d = stream.alloc_zeros::<f32>(t * HC)?;
            let mut post_d = stream.alloc_zeros::<f32>(t * HC)?;
            let mut comb_d = stream.alloc_zeros::<f32>(t * HC * HC)?;
            let mut y_d = stream.alloc_zeros::<f32>(t * D)?;
            unsafe {
                let rc = k::memra_dsv4_hc_pre_v4(
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
                    iters(),
                    EPS,
                    std::ptr::null_mut(),
                    block,
                    sp(&stream),
                );
                assert_eq!(rc, 0, "hc_pre_v4 rc (block={block})");
            }
            stream.synchronize()?;
            Ok((
                stream.clone_dtoh(&pre_d)?,
                stream.clone_dtoh(&post_d)?,
                stream.clone_dtoh(&comb_d)?,
                stream.clone_dtoh(&y_d)?,
            ))
        };
        for block in [512i32, 1024] {
            let v3 = run_fused_v3(1, block, 0)?;
            let v4 = run_v4(block)?;
            let bad = bit_diffs(&v3, &v4);
            let total = v3.0.len() + v3.1.len() + v3.2.len() + v3.3.len();
            println!(
                "[correctness] t={t} hc={HC} d={D}: v4(block={block})-vs-v3(sink_reg=1,block={block}) bit-bad={bad}/{total} {}",
                if bad == 0 { "PASS" } else { "FAIL" }
            );
            if bad != 0 {
                fails += 1;
            }
            // interleaved timing, N_TIMED rounds of (v3, v4)
            let mut a = Vec::with_capacity(N_TIMED);
            let mut b = Vec::with_capacity(N_TIMED);
            for _ in 0..N_TIMED {
                stream.synchronize()?;
                let t0 = std::time::Instant::now();
                let _ = run_fused_v3(1, block, 0)?;
                a.push(t0.elapsed().as_micros() as u64);
                stream.synchronize()?;
                let t0 = std::time::Instant::now();
                let _ = run_v4(block)?;
                b.push(t0.elapsed().as_micros() as u64);
            }
            a.sort_unstable();
            b.sort_unstable();
            println!(
                "[v4-timing] t={t} block={block} iters={} v3 median={}us v4 median={}us runs v3={a:?} v4={b:?}",
                iters(),
                a[a.len() / 2],
                b[b.len() / 2]
            );
        }

        // ---- kernel-only timing: 300 back-to-back launches per arm, one sync, us per launch ----
        // The wall arms above pay alloc + htod + dtoh per call (~50 us on a 5090) and cannot see a
        // 5 us kernel difference; this one can. Outputs are overwritten in place.
        if t == 1 {
            let mixes_d = stream.clone_htod(&s.mixes)?;
            let mut pre_d = stream.alloc_zeros::<f32>(t * HC)?;
            let mut post_d = stream.alloc_zeros::<f32>(t * HC)?;
            let mut comb_d = stream.alloc_zeros::<f32>(t * HC * HC)?;
            let mut y_d = stream.alloc_zeros::<f32>(t * D)?;
            const NB: usize = 300;
            for block in [512i32, 1024] {
                let mut res: Vec<(&str, f64)> = Vec::new();
                for round in 0..2 {
                    for arm in ["v3", "v4"] {
                        let mut launch = |i: usize| -> Res<()> {
                            unsafe {
                                let rc = if arm == "v3" {
                                    k::memra_dsv4_hc_pre_fused_v3(
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
                                        iters(),
                                        EPS,
                                        std::ptr::null_mut(),
                                        block,
                                        1,
                                        0,
                                        sp(&stream),
                                    )
                                } else {
                                    k::memra_dsv4_hc_pre_v4(
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
                                        iters(),
                                        EPS,
                                        std::ptr::null_mut(),
                                        block,
                                        sp(&stream),
                                    )
                                };
                                assert_eq!(rc, 0, "{arm} rc at launch {i}");
                            }
                            Ok(())
                        };
                        for i in 0..20 {
                            launch(i)?;
                        }
                        stream.synchronize()?;
                        let t0 = std::time::Instant::now();
                        for i in 0..NB {
                            launch(i)?;
                        }
                        stream.synchronize()?;
                        let us = t0.elapsed().as_secs_f64() * 1e6 / NB as f64;
                        if round == 1 {
                            res.push((arm, us));
                        }
                    }
                }
                println!(
                    "[kernel-timing] t=1 block={block} iters={} back-to-back x{NB}: v3 {:.2} us/launch, v4 {:.2} us/launch ({:+.1}%)",
                    iters(),
                    res[0].1,
                    res[1].1,
                    100.0 * (res[1].1 / res[0].1 - 1.0)
                );
            }
        }

        // ---- MEMRA_HC_PHASE_STAMPS: where v3's time goes, phase by phase, on THIS card ----
        // The stamped twin (bench-only kernel) writes %globaltimer + clock64 at six boundaries;
        // medians over N launches at the served config (sink_reg=1, block=512). This is the
        // number the rig's ncu cannot give: FP64 is 1/64 rate on the 5090 and near full rate on
        // the B200, so the stall split there does not transfer; the phase ns here do.
        if t == 1 && std::env::var("MEMRA_HC_PHASE_STAMPS").is_ok() {
            let mixes_d = stream.clone_htod(&s.mixes)?;
            let mut pre_d = stream.alloc_zeros::<f32>(t * HC)?;
            let mut post_d = stream.alloc_zeros::<f32>(t * HC)?;
            let mut comb_d = stream.alloc_zeros::<f32>(t * HC * HC)?;
            let mut y_d = stream.alloc_zeros::<f32>(t * D)?;
            let mut st_d = stream.alloc_zeros::<u64>(12)?;
            let names = [
                "P0 sumsq loads+dfma",
                "P1 block_sum(10 barriers)+rsq",
                "P2 warp0 gates+softmax+sinkhorn",
                "P3 block barrier",
                "P4 combine (x again)+store",
            ];
            let mut ns: Vec<Vec<u64>> = vec![Vec::new(); 5];
            let mut cy: Vec<Vec<u64>> = vec![Vec::new(); 5];
            let mut tot_ns: Vec<u64> = Vec::new();
            for _ in 0..60 {
                unsafe {
                    let rc = k::memra_dsv4_hc_pre_fused_v3_stamped(
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
                        iters(),
                        EPS,
                        512,
                        1,
                        st_d.device_ptr_mut(&stream).0 as *mut u64,
                        sp(&stream),
                    );
                    assert_eq!(rc, 0, "hc_pre_fused_v3_stamped rc");
                }
                stream.synchronize()?;
                let st = stream.clone_dtoh(&st_d)?;
                for i in 0..5 {
                    ns[i].push(st[i + 1].saturating_sub(st[i]));
                    cy[i].push(st[7 + i].saturating_sub(st[6 + i]));
                }
                tot_ns.push(st[5].saturating_sub(st[0]));
            }
            let med = |v: &mut Vec<u64>| {
                v.sort_unstable();
                v[v.len() / 2]
            };
            let total = med(&mut tot_ns);
            println!(
                "[v3-phases] t=1 hc={HC} d={D} sink_reg=1 block=512 iters={} N=60: total {total} ns (thread-0 view)",
                iters()
            );
            for i in 0..5 {
                let n = med(&mut ns[i]);
                let c = med(&mut cy[i]);
                println!(
                    "[v3-phases]   {:<34} {n:>6} ns  {c:>7} cyc  {:5.1}%",
                    names[i],
                    100.0 * n as f64 / total.max(1) as f64
                );
            }
        }

        // ---- MEMRA_HC_PHASE_STAMPS: the same six stamps on v4 ----
        if t == 1 && std::env::var("MEMRA_HC_PHASE_STAMPS").is_ok() {
            let mixes_d = stream.clone_htod(&s.mixes)?;
            let mut pre_d = stream.alloc_zeros::<f32>(t * HC)?;
            let mut post_d = stream.alloc_zeros::<f32>(t * HC)?;
            let mut comb_d = stream.alloc_zeros::<f32>(t * HC * HC)?;
            let mut y_d = stream.alloc_zeros::<f32>(t * D)?;
            let mut st_d = stream.alloc_zeros::<u64>(12)?;
            let names = [
                "P0 16 loads + sumsq",
                "P1 barrier1 + warp0 tree/rsq/gates",
                "P2 barrier 2",
                "P3 warp0 softmax + sinkhorn",
                "P4 combine (registers) + store",
            ];
            let mut ns: Vec<Vec<u64>> = vec![Vec::new(); 5];
            let mut cy: Vec<Vec<u64>> = vec![Vec::new(); 5];
            let mut tot_ns: Vec<u64> = Vec::new();
            for _ in 0..60 {
                unsafe {
                    let rc = k::memra_dsv4_hc_pre_v4_stamped(
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
                        iters(),
                        EPS,
                        st_d.device_ptr_mut(&stream).0 as *mut u64,
                        sp(&stream),
                    );
                    assert_eq!(rc, 0, "hc_pre_v4_stamped rc");
                }
                stream.synchronize()?;
                let st = stream.clone_dtoh(&st_d)?;
                for i in 0..5 {
                    ns[i].push(st[i + 1].saturating_sub(st[i]));
                    cy[i].push(st[7 + i].saturating_sub(st[6 + i]));
                }
                tot_ns.push(st[5].saturating_sub(st[0]));
            }
            let med = |v: &mut Vec<u64>| {
                v.sort_unstable();
                v[v.len() / 2]
            };
            let total = med(&mut tot_ns);
            println!(
                "[v4-phases] t=1 hc={HC} d={D} 1024x16 iters={} N=60: total {total} ns (thread-0 view)",
                iters()
            );
            for i in 0..5 {
                let n = med(&mut ns[i]);
                let c = med(&mut cy[i]);
                println!(
                    "[v4-phases]   {:<36} {n:>6} ns  {c:>7} cyc  {:5.1}%",
                    names[i],
                    100.0 * n as f64 / total.max(1) as f64
                );
            }
        }

        // ---- hc mixes GEMV: cuBLASLt (served) vs the native kernel, back-to-back x300 ----
        // The mixes projection (24 x 16384 f32) runs once per hc pre site, 90 per token; the
        // served path is cuBLASLt's dot+reduce pair. Kernel-only us per launch, same buffers.
        if t == 1 {
            const NB: usize = 300;
            let rows = (2 + HC) * HC;
            let eng = memra_engine::Engine::new(dev)?;
            let xe = eng.htod(&s.x[..HC * D])?;
            let we = eng.htod(&vecf(rows * HC * D, 0x31))?;
            let mut ye = eng.uninit(rows)?;
            let mut res: Vec<(&str, f64)> = Vec::new();
            for round in 0..2 {
                for arm in ["cublas", "native"] {
                    let mut launch = || -> Res<()> {
                        let xv = xe.slice(0..HC * D);
                        let wv = we.slice(0..rows * HC * D);
                        let mut yv = ye.slice_mut(0..rows);
                        if arm == "cublas" {
                            eng.linear_t1_into(&xv, &wv, &mut yv, HC * D, rows)?;
                        } else {
                            assert!(eng.hc_mixes_gemv_into(&xv, &wv, &mut yv, HC * D, rows)?);
                        }
                        Ok(())
                    };
                    for _ in 0..20 {
                        launch()?;
                    }
                    eng.stream().synchronize()?;
                    let t0 = std::time::Instant::now();
                    for _ in 0..NB {
                        launch()?;
                    }
                    eng.stream().synchronize()?;
                    let us = t0.elapsed().as_secs_f64() * 1e6 / NB as f64;
                    if round == 1 {
                        res.push((arm, us));
                    }
                }
            }
            println!(
                "[mixes-timing] t=1 rows={rows} in_f={} back-to-back x{NB}: cublas {:.2} us/launch, native {:.2} us/launch ({:+.1}%)",
                HC * D,
                res[0].1,
                res[1].1,
                100.0 * (res[1].1 / res[0].1 - 1.0)
            );
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
            "t={t:<2} unfused(rowsq+sinkhorn+collapse)={:>7.1}us/call  fused(=1,hc_pre_fused)={:>7.1}us/call  fused(=2,hc_pre_fused_v2)={:>7.1}us/call  hc_post(unfused, not part of this fusion)={:>7.1}us/call  runs={unfused_us:?} / {fused_us:?} / {fused_v2_us:?} / {post_us:?}",
            mean(&unfused_us), mean(&fused_us), mean(&fused_v2_us), mean(&post_us)
        ));
        json_arms.push(format!(
            "{{\"t\":{t},\"hc\":{HC},\"d\":{D},\"bit_bad_v1\":{bad},\"bit_bad_v2_vs_unfused\":{bad_v2_vs_unfused},\"bit_bad_v2_vs_v1\":{bad_v2_vs_v1},\"unfused_us\":{unfused_us:?},\"fused_us\":{fused_us:?},\"fused_v2_us\":{fused_v2_us:?},\"hc_post_us\":{post_us:?}}}"
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
            "hc-fused-gate: PASS ({} shapes, bit-identical unfused vs fused(=1) vs fused(=2))",
            3
        );
        Ok(())
    } else {
        eprintln!("hc-fused-gate: FAIL ({fails} shape(s) mismatched)");
        std::process::exit(1);
    }
}
