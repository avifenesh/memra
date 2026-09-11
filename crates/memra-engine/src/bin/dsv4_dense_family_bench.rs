//! dsv4-dense-family-bench: the dense family, both real kernels, one process, no model.
//!
//! WHY THIS EXISTS. The tensor-core dense path (memra #472) measures 5.25x faster than the
//! scalar kernel device-side and REGRESSES 37% on the served path, and the first explanation
//! (per-call host setup) is dead: `can_implement + initialize` prices at 0.027 us per call,
//! 0.6 ms across the whole 21,312-call family, and a census replay measures 250.0 ms
//! pre-cache against 250.4 ms cached. Something else costs about 4 seconds per 981-token
//! prefill, which is MORE than the entire dense family costs the scalar program, so the
//! question is not "how much did the kernel gain" but "does the regression reproduce outside
//! the engine at all".
//!
//! WHAT THIS IS NOT. It is not a replica. This lane has already been burned once by a
//! standalone harness that modelled the right thing for the questions it was asked and the
//! wrong thing for this one, so nothing here re-implements a kernel: it calls
//! `memra_dsv4_gemv_fp8_m`, the engine's own entry point, and selects the arm with
//! `arm_dense_cutlass_for_gate`, which is a gate arm and not a door. Both arms are the
//! shipped code, in one process, over the same buffers.
//!
//! THE CONDITION UNDER TEST: WEIGHT RESIDENCY. The standalone probe that priced the family
//! reused ONE weight buffer per shape, so weights sat in the card's 128 MiB L2 for the whole
//! replay. The engine cycles a different layer's weights on every call, about 3.4 GB of them,
//! and the CUTLASS path reads a bf16 MIRROR at 2 bytes per weight where the scalar path reads
//! e4m3 codes at 1. `--weights N` sets how many distinct weight buffers each shape cycles, so
//! the same replay runs cache-hot (1) and cache-cold (enough to blow past L2), and the two
//! arms are compared under EACH condition rather than one.
//!
//! WHAT EACH ARM KILLS:
//!   * cutlass beats scalar hot AND cold  -> the family is not where the regression lives,
//!     and the search moves to the engine's surroundings (allocator pressure, the expert
//!     path, per-request state) rather than the kernel.
//!   * cutlass beats scalar hot, loses cold -> residency is the cause, it reproduces
//!     standalone, and the 2x weight bytes of the bf16 mirror is the mechanism to attack.
//!   * cutlass loses both -> the 5.25x was an artefact of the earlier harness and the path
//!     never had the win this lane has been spending GPU time on.
//!
//! NON-VACUITY, because an arm switch that does nothing reports a clean null. The bench
//! refuses unless the ON arm's split-K call count actually rises and the OFF arm's does not,
//! and unless the binary contains the path at all (`dense_cutlass_armed_for_gate` returns
//! `None` when the CUTLASS archive was not linked, which is a DIFFERENT fact from "off").
//! This lane's first served A/B compared two byte-identical binaries and reported a clean
//! null; that is the mistake this guard exists to make impossible.
//!
//! Usage: `dsv4-dense-family-bench [device] [weights_per_shape] [reps]`
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use memra_engine::dsv4_ffi as k;
use memra_engine::dsv4_gpu::{
    arm_dense_cutlass_for_gate, dense_cutlass_armed_for_gate, dense_cutlass_counts_for_gate,
};
use std::ffi::c_void;

/// The `gemv_fp8` rows of the dense census (darklanes
/// `research/dsv4f-dense-perrow-20260910/receipts/census-r1.log`), replayed at their real
/// call counts over one 1,025-token prefill at the served chunk width. These are the rows
/// that reach this entry point; `dots_*` rows use a different one.
const CENSUS: &[(usize, usize, usize)] = &[
    // (n, k, calls)
    (1024, 4096, 12384),
    (32768, 1024, 1376),
    (4096, 8192, 1376),
    (2048, 4096, 2752),
    (4096, 2048, 1376),
    (8192, 1024, 672),
    (512, 4096, 1376),
];
/// `DSV4_TMAX`: every census call is m = 32, because the entry point re-enters itself in
/// 32-wide tiles above that, which is also why the prefill chunk width does not move the
/// census.
const M: usize = 32;
const SCALE_BLOCK: usize = 128;

fn main() {
    let arg = |i: usize, d: usize| -> usize {
        std::env::args()
            .nth(i)
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let dev = arg(1, 0);
    let weights = arg(2, 1).max(1);
    let reps = arg(3, 3).max(1);

    // The path must be PRESENT before anything is timed. Absent and off look identical in a
    // timing row and are not the same fact.
    match dense_cutlass_armed_for_gate() {
        None => {
            println!("BENCH_FAIL this binary does not contain the dense CUTLASS path");
            std::process::exit(1);
        }
        Some(armed) => println!("PRESENT armed_at_start={armed}"),
    }

    let ctx = CudaContext::new(dev).expect("ctx");
    let stream = ctx.default_stream();

    struct Buf {
        w: Vec<cudarc::driver::CudaSlice<u8>>,
        sc: cudarc::driver::CudaSlice<f32>,
        x: cudarc::driver::CudaSlice<u8>,
        y: cudarc::driver::CudaSlice<f32>,
        blocks: usize,
    }
    let mut bufs: Vec<Buf> = Vec::new();
    let mut weight_bytes = 0usize;
    for &(n, kdim, _) in CENSUS {
        let blocks = kdim / SCALE_BLOCK;
        let sc_rows = n.div_ceil(128);
        let mut w = Vec::with_capacity(weights);
        for _ in 0..weights {
            w.push(stream.alloc_zeros::<u8>(n * kdim).expect("w"));
            weight_bytes += n * kdim;
        }
        bufs.push(Buf {
            w,
            sc: stream.alloc_zeros::<f32>(sc_rows * blocks).expect("sc"),
            x: stream.alloc_zeros::<u8>(M * kdim * 2).expect("x"),
            y: stream.alloc_zeros::<f32>(M * n).expect("y"),
            blocks,
        });
    }
    let total_calls: usize = CENSUS.iter().map(|r| r.2).sum();
    println!(
        "CONFIG device={dev} weights_per_shape={weights} codes_MB={:.1} calls={total_calls} reps={reps}",
        weight_bytes as f64 / 1e6
    );

    // One pass per weight buffer per shape, so the bf16 mirror for every buffer the timed
    // region will touch is already built and no arm pays mirror construction.
    let replay = |arm_on: bool, warm_only: bool, per_shape: &mut Vec<f64>| -> f64 {
        arm_dense_cutlass_for_gate(arm_on).expect("arm");
        per_shape.clear();
        let t0 = std::time::Instant::now();
        for (s, &(n, kdim, calls)) in CENSUS.iter().enumerate() {
            let b = &bufs[s];
            let shape_t0 = std::time::Instant::now();
            let passes = if warm_only { weights } else { calls };
            for c in 0..passes {
                let wi = &b.w[c % weights];
                let rc = unsafe {
                    k::memra_dsv4_gemv_fp8_m(
                        wi.device_ptr(&stream).0 as *const c_void,
                        b.sc.device_ptr(&stream).0 as *const f32,
                        b.blocks as i32,
                        b.x.device_ptr(&stream).0 as *const c_void,
                        b.y.device_ptr(&stream).0 as *mut f32,
                        M as i32,
                        n as i32,
                        kdim as i32,
                        kdim as i32,
                        n as i32,
                        stream.cu_stream() as *mut c_void,
                    )
                };
                assert_eq!(rc, 0, "gemv_fp8_m rc={rc} at n={n} k={kdim}");
            }
            // Per shape, because the two candidate limits predict different SHAPES of row.
            // A launch-bound family costs the same per CALL at every (n, k); a byte-bound one
            // costs in proportion to n*k. One arm of rows separates them without another cell.
            stream.synchronize().expect("sync");
            per_shape.push(shape_t0.elapsed().as_secs_f64() * 1e3);
        }
        t0.elapsed().as_secs_f64() * 1e3
    };

    let mut scratch: Vec<f64> = Vec::new();
    for on in [false, true] {
        replay(on, true, &mut scratch);
    }

    // Interleaved, because this box drifts: scalar, cutlass, scalar, cutlass.
    let mut rows: Vec<(bool, f64)> = Vec::new();
    for rep in 1..=reps {
        for on in [false, true] {
            let before = dense_cutlass_counts_for_gate().expect("counts");
            let mut shape_ms: Vec<f64> = Vec::new();
            let ms = replay(on, false, &mut shape_ms);
            let after = dense_cutlass_counts_for_gate().expect("counts");
            for (s, &(n, kdim, calls)) in CENSUS.iter().enumerate() {
                // Bytes this arm moves for this shape, from the source and not from a model:
                // scalar reads n*k e4m3 codes; cutlass reads a 2 n*k bf16 mirror and then writes
                // and re-reads a split-K partial that is exactly n*k bytes ((k/128)*32*4 = k).
                let nk = (n * kdim) as f64;
                let per_call = if on {
                    4.0 * nk + (M * kdim * 2) as f64 + (M * n * 4) as f64
                } else {
                    nk + (M * kdim * 2) as f64 + (M * n * 4) as f64
                };
                let gb = per_call * calls as f64 / 1e9;
                println!(
                    "SHAPE rep={rep} arm={} n={n} k={kdim} calls={calls} ms={:.1}                      per_call_us={:.2} GB={gb:.1} GB_per_s={:.0}",
                    if on { "cutlass" } else { "scalar" },
                    shape_ms[s],
                    shape_ms[s] * 1000.0 / calls as f64,
                    gb / (shape_ms[s] / 1e3)
                );
            }
            let engaged = after.splitk - before.splitk;
            println!(
                "REPLAY rep={rep} arm={} ms={ms:.1} per_call_us={:.3} splitk_calls={engaged} \
                 declined={} mirror_MB={:.1} shapes_built={}",
                if on { "cutlass" } else { "scalar" },
                ms * 1000.0 / total_calls as f64,
                after.declined - before.declined,
                after.mirror_bytes as f64 / 1e6,
                after.shapes_built
            );
            // The guard that makes the comparison mean something.
            if on && engaged == 0 {
                println!(
                    "BENCH_FAIL the cutlass arm ran no split-K calls, so the arms are the same program"
                );
                std::process::exit(1);
            }
            if !on && engaged != 0 {
                println!(
                    "BENCH_FAIL the scalar arm ran {engaged} split-K calls, so standing the path down does nothing"
                );
                std::process::exit(1);
            }
            rows.push((on, ms));
        }
    }
    let median = |on: bool| -> f64 {
        let mut v: Vec<f64> = rows.iter().filter(|r| r.0 == on).map(|r| r.1).collect();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (scalar, cutlass) = (median(false), median(true));
    println!(
        "VERDICT weights_per_shape={weights} scalar_ms={scalar:.1} cutlass_ms={cutlass:.1} \
         cutlass_over_scalar={:.3}x",
        scalar / cutlass
    );
    println!("BENCH_OK");
}
