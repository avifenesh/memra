//! dsv4-dense-arm-ab: the dense tensor-core path, both arms, ONE model load, no server.
//!
//! WHY THIS EXISTS. The served A/B says the CUTLASS dense path costs 37% of the
//! prefill rate (darklanes `research/dsv4f-dense-perrow-20260910/receipts-wiring/cell-r2`
//! and `receipts/cell-rA`). Two explanations are already dead:
//!
//!   * per-call HOST setup: `can_implement` + `initialize` price at 0.027 us per call,
//!     0.6 ms across the whole 21,312-call family, and the served cell WITH the
//!     per-shape cache measured the same 37%;
//!   * the dense family itself: `dsv4-dense-family-bench` runs both real kernels over
//!     the real census call counts at three weight-residency levels and the CUTLASS arm
//!     WINS every one of them (5.93x with the family in L2, 4.55x with 3.15 GB of codes
//!     and 6.31 GB of mirror cycling past it).
//!
//! So the cost is not in the kernel and not in the family. This binary asks the next
//! question in the only order that narrows it: does it appear in the ENGINE's prefill,
//! with no HTTP, no sampler, no spec route and no second process? One model load, the
//! same prompt, the arm flipped between timed prefills.
//!
//! If prefill seconds regress here, the cause is inside `dspark_prefill_prime_chunked`
//! and an nsys run over this same binary attributes it to a kernel. If they do NOT, the
//! cause is outside prefill, and the search moves to what the server does around it.
//!
//! NON-VACUITY. The arm must be PRESENT (a binary built without the CUTLASS archive has
//! a null weak symbol and reports `None`, which is a different fact from "off"), the ON
//! arm must run split-K calls, and the OFF arm must run none. This lane's first served
//! A/B compared two byte-identical binaries and reported a clean null; that is the
//! mistake these three checks exist to make impossible.
//!
//! Usage: `dsv4-dense-arm-ab <model-dir> <real-source.txt> [reps]`
use memra_engine::dsv4_gpu::{
    Dsv4Gpu, arm_dense_cutlass_for_gate, dense_cutlass_armed_for_gate,
    dense_cutlass_counts_for_gate,
};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::Instant;

/// The prefill chunk width. memra #482 moved the served default to 512, and a cell that
/// measures 64 measures a program nobody runs any more.
fn served_width() -> usize {
    match std::env::var("MEMRA_DSV4_CENSUS_WIDTH") {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|err| panic!("MEMRA_DSV4_CENSUS_WIDTH={value}: {err}")),
        Err(std::env::VarError::NotPresent) => 512,
        Err(err) => panic!("MEMRA_DSV4_CENSUS_WIDTH: {err}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() == 3 || args.len() == 4,
        "usage: dsv4-dense-arm-ab <model-dir> <real-source.txt> [reps]"
    );
    let reps: usize = args.get(3).map_or(3, |v| v.parse().expect("reps"));
    let dir = Path::new(&args[1]);
    let source = std::fs::read_to_string(&args[2]).expect("real source");

    match dense_cutlass_armed_for_gate() {
        None => {
            println!("AB_FAIL this binary does not contain the dense CUTLASS path");
            std::process::exit(1);
        }
        Some(armed) => println!("PRESENT armed_at_start={armed}"),
    }

    // The served process runs with the drafter RESIDENT (`[dsv4-serve] ... drafter
    // RESIDENT (spec route armed)`), and the prime this cell times is the drafter's
    // own chunked prime. Without the arm the drafter is not loaded and the run dies
    // 100 seconds into the load, inside `dspark_alloc_state`, with a panic that the
    // first attempt at this cell mistook for a cell that had run. Refuse up front,
    // and name the condition being matched.
    let drafter = std::env::var("MEMRA_DSV4_DRAFTER").unwrap_or_default();
    if drafter != "dspark" {
        println!(
            "AB_FAIL MEMRA_DSV4_DRAFTER={drafter:?}: this cell measures the SERVED shape, \
             which loads the drafter (spec route armed), so it must be set to dspark"
        );
        std::process::exit(1);
    }
    println!("DRAFTER dspark (served shape)");

    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let mut prompt = tokenizer.encode(&format!("Review this engine source:\n{source}"), true);
    assert!(prompt.len() >= 1025, "do not pad/repeat source");
    prompt.truncate(1025);
    let width = served_width();
    println!(
        "SOURCE sha256={:x} tokens={} width={width}",
        Sha256::digest(source.as_bytes()),
        prompt.len()
    );

    // Served defaults: the matrix expert program (memra #482) is what loads unless a
    // gate arms the reference executor, and nothing here arms it.
    let mut gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, 4096).expect("load");

    let prime = |gpu: &mut Dsv4Gpu| -> f64 {
        let mut state = gpu
            .alloc_decode_state_for_transient(prompt.len() + 32, width)
            .expect("cache");
        let mut draft = gpu.dspark_alloc_state().expect("draft");
        let timer = Instant::now();
        gpu.dspark_prefill_prime_chunked(&prompt, &mut state, &mut draft, width)
            .expect("prime");
        timer.elapsed().as_secs_f64()
    };

    // A cold first prefill is its own regime, and the bf16 mirror is built lazily on
    // first use, so BOTH arms get a warm-up and neither pays mirror construction in a
    // timed region.
    for on in [false, true] {
        arm_dense_cutlass_for_gate(on).expect("arm");
        prime(&mut gpu);
    }
    let warm = dense_cutlass_counts_for_gate().expect("counts");
    println!(
        "WARM mirror_MB={:.1} shapes_built={} splitk={} declined={}",
        warm.mirror_bytes as f64 / 1e6,
        warm.shapes_built,
        warm.splitk,
        warm.declined
    );

    // Interleaved, because this box drifts: scalar, cutlass, scalar, cutlass.
    let mut rows: Vec<(bool, f64)> = Vec::new();
    for rep in 1..=reps {
        for on in [false, true] {
            arm_dense_cutlass_for_gate(on).expect("arm");
            let before = dense_cutlass_counts_for_gate().expect("counts");
            let seconds = prime(&mut gpu);
            let after = dense_cutlass_counts_for_gate().expect("counts");
            let engaged = after.splitk - before.splitk;
            println!(
                "PRIME rep={rep} arm={} seconds={seconds:.4} tok_per_s={:.1} splitk={engaged} \
                 declined={} mirror_MB={:.1} shapes_built={}",
                if on { "cutlass" } else { "scalar" },
                prompt.len() as f64 / seconds,
                after.declined - before.declined,
                after.mirror_bytes as f64 / 1e6,
                after.shapes_built
            );
            if on && engaged == 0 {
                println!(
                    "AB_FAIL the cutlass arm ran no split-K calls, so the arms are one program"
                );
                std::process::exit(1);
            }
            if !on && engaged != 0 {
                println!(
                    "AB_FAIL the scalar arm ran {engaged} split-K calls, so the arm does nothing"
                );
                std::process::exit(1);
            }
            rows.push((on, seconds));
        }
    }
    let median = |on: bool| -> f64 {
        let mut v: Vec<f64> = rows.iter().filter(|r| r.0 == on).map(|r| r.1).collect();
        median_of(&mut v)
    };
    let (scalar, cutlass) = (median(false), median(true));
    let (ratio, delta_pct) = verdict_ratio(scalar, cutlass);
    println!(
        "VERDICT width={width} scalar_s={scalar:.4} cutlass_s={cutlass:.4} \
         cutlass_over_scalar={ratio:.3}x delta_pct={delta_pct:.1}"
    );
    println!("AB_OK");
}

/// Middle of a sorted rep list. Pure so the verdict math stays testable without a model.
fn median_of(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Cutlass-over-scalar ratio and percent delta from the two arm medians.
fn verdict_ratio(scalar: f64, cutlass: f64) -> (f64, f64) {
    let ratio = scalar / cutlass;
    (ratio, (ratio - 1.0) * 100.0)
}

#[cfg(test)]
mod arm_ab_verdict_tests {
    use super::{median_of, verdict_ratio};

    #[test]
    fn median_picks_the_middle_rep() {
        let mut v = vec![8.1018, 8.0950, 8.0958];
        assert_eq!(median_of(&mut v), 8.0958);
    }

    #[test]
    fn verdict_reproduces_the_banked_red_row() {
        // receipts/dw-ab-gr1/ab-w512.log: 3.7078 / 8.0958 = 0.458x, -54.2%.
        let (ratio, delta) = verdict_ratio(3.7078, 8.0958);
        assert!((ratio - 0.458).abs() < 5e-4, "ratio {ratio}");
        assert!((delta - -54.2).abs() < 5e-2, "delta {delta}");
    }

    #[test]
    fn verdict_reproduces_the_banked_fixed_row() {
        // receipts/dw-ab-gr2/ab-w512.log: 3.6998 / 2.6701 = 1.386x, +38.6%.
        let (ratio, delta) = verdict_ratio(3.6998, 2.6701);
        assert!((ratio - 1.386).abs() < 5e-4, "ratio {ratio}");
        assert!((delta - 38.6).abs() < 5e-2, "delta {delta}");
    }
}
