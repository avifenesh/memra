//! MOE SELECTED-EXPERT CENSUS (2026-08-26): prices the kernel that dominates the step37 spec
//! verify walk, at the exact per-column widths the walk runs, with no model and no walk noise.
//!
//! WHY THIS KERNEL: a corrected read of `MEMRA_SPEC_PHASE` (verify-issue is the host QUEUEING
//! under a running GPU, verify-wait is only the residual drain) puts the K=1 verify walk at
//! ~22.0 ms of GPU against a 12.16 ms plain token — two columns cost ~1.81x one, which IS the
//! measured 0.90x spec-vs-plain gap. `MEMRA_TCOL_PROF` then splits that walk at t=2 into
//! norm+qkv 0.19 ms / attn 10.08 ms / ffn 15.25 ms: the MoE is 60% of it. Attention and the
//! shared expert are already hoisted to t-row twins, so the routed sweep is the term that
//! scales with columns.
//!
//! WHAT IT DECIDES, and what it must NOT be used to claim:
//!   1. The COLUMN CURVE. n_sel = t * 8, so t=1,2,4,8 -> 8,16,32,64 pairs. If us/call is linear
//!      in n_sel the sweep is per-pair-bound and hoisting/grouping is the lever; if it flattens,
//!      the walk's t-scaling lives somewhere else and this is the wrong target.
//!   2. Whether the sweep is off its roofline. Achieved weight bandwidth per call is printed
//!      against peak. A derived 194 GB/s (11% of peak) is what motivated this bench, but that
//!      was ARITHMETIC off two segment timers, not a measurement — this replaces it.
//!   3. `MEMRA_SEL_GU_RPW=2|4`, the in-tree multirow twin that reads the activation group once
//!      and reuses it across RPW rows. Its own doc calls it bit-identical per row, and it
//!      defaults OFF, so it has never been priced on this shape. Bit-identity is NOT re-proven
//!      here — this is a SPEED row only; flipping a default needs the argmax gate + battery.
//!
//! Weights are dummy allocations: the kernel's cost is byte movement and occupancy at a given
//! geometry, neither of which depends on the values. That also makes this runnable on any single
//! card without the artifact. It is NOT an exactness gate and produces no numeric verdict.
//!
//! usage: moe-sel-census [--reps N] [--out-f 1280]
use memra_engine::Engine;

const PEAK_TBS: f64 = 1.79;
const QK: usize = 64;
const BLK: usize = 36; // NVFP4: 36 bytes per 64 weights (4 UE4M3 sub-scales + 32 code bytes)

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str, d: usize| -> usize {
        args.windows(2)
            .find(|w| w[0] == k)
            .and_then(|w| w[1].parse().ok())
            .unwrap_or(d)
    };
    let reps = arg("--reps", 200);
    // step37: n_embd 4096 in; moe_intermediate_size 1280 per expert for gate and up. Under TP2
    // the local half is 640, so both widths are priced rather than assumed.
    let in_f = arg("--in-f", 4096);
    let n_experts = arg("--experts", 288);
    let e = Engine::new(0)?;
    println!(
        "moe-sel-census reps={reps} in_f={in_f} experts={n_experts} peak={PEAK_TBS} TB/s \
         rpw={}",
        std::env::var("MEMRA_SEL_GU_RPW").unwrap_or_else(|_| "unset(1)".into())
    );
    for out_f in [1280usize, 640] {
        for t in [1usize, 2, 4, 8] {
            row(&e, reps, in_f, out_f, n_experts, t)?;
        }
    }
    Ok(())
}

fn row(
    e: &Engine,
    reps: usize,
    in_f: usize,
    out_f: usize,
    n_experts: usize,
    t: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let n_sel = t * 8; // top-8 routing, one slot set per verify column
    let row_bytes = in_f / QK * BLK;
    let expert_stride = out_f * row_bytes;
    // Two banks (gate, up). Cap the resident experts so the bench fits any card: the kernel
    // reads whichever rows `sel` names, so a smaller bank changes nothing it measures.
    let bank_experts = n_experts.min(64);
    let gate = e.alloc_u8(bank_experts * expert_stride)?;
    let up = e.alloc_u8(bank_experts * expert_stride)?;
    // DISTINCT experts per slot: the walk's real selections are only 16-41% duplicated, and a
    // sel full of one id would let the L2 serve every pair after the first and read as a win
    // that the walk never gets.
    let sel_h: Vec<i32> = (0..n_sel).map(|i| (i % bank_experts) as i32).collect();
    let sel = e.htod_i32(&sel_h)?;
    let aq = e.htod_i8(&vec![1i8; t * in_f])?;
    let ad = e.htod(&vec![0.01f32; 2 * t * in_f / 32])?;
    let mut yg = e.htod(&vec![0f32; n_sel * out_f])?;
    let mut yu = e.htod(&vec![0f32; n_sel * out_f])?;
    let mut run = || -> Result<(), Box<dyn std::error::Error>> {
        e.qmatvec_nvfp4_sel_gu_into(
            &gate,
            &up,
            &sel,
            &aq,
            &ad,
            &mut yg,
            &mut yu,
            n_sel,
            in_f,
            out_f,
            row_bytes,
            expert_stride,
        )
    };
    for _ in 0..30 {
        run()?;
    }
    e.stream().synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        run()?;
    }
    e.stream().synchronize()?;
    let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
    // gate + up both stream n_sel rows-blocks of weights.
    let bytes = 2.0 * (n_sel * expert_stride) as f64;
    let tbs = bytes / us / 1e6;
    println!(
        "[sel_gu out_f={out_f:4} t={t} n_sel={n_sel:2}] {us:8.1} us/call  {tbs:5.2} TB/s \
         ({:4.1}% of peak)  {:6.1} us/pair",
        100.0 * tbs / PEAK_TBS,
        us / n_sel as f64
    );
    Ok(())
}
