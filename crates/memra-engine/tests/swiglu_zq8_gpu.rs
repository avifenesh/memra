//! `swiglu_preclamped_mul_scaled_q8_1_f32` (door `MEMRA_SHEXP_SWIGLU_ZQ8`) must produce the
//! SAME q8_1 bytes as `swiglu_preclamped_mul_scaled_f32` followed by `quantize_q8_1`, for every
//! width the shared expert can take, with inputs that exercise both clamps, both signs, exact
//! zeros (the `d == 0` branch) and scales other than 1. Red arms: a different limit or a
//! different `up` scale must move the bytes (proves the gate compares live output), and a width
//! that is not a multiple of 32 must be refused. Exactness only, no timing: this runs on the rig.
use memra_engine::Engine;

fn lcg(seed: u64, n: usize, amp: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / ((1u64 << 31) as f32);
            if i % 97 == 0 {
                0.0
            } else {
                (u - 0.5) * 2.0 * amp
            }
        })
        .collect()
}

#[test]
#[ignore]
fn gpu_swiglu_q8_1_matches_swiglu_then_quantize_bitwise() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let limit = 7.0f32;
    let mut cells = 0usize;
    for &n in &[32usize, 512, 2048, 4096, 2048 * 3] {
        for &(gs, us) in &[(1.0f32, 1.0f32), (0.5, 1.75)] {
            // amplitude 3x the limit so a fair share of elements sit past both clamps
            let gate = e.htod(&lcg(11 + n as u64, n, limit * 3.0)).unwrap();
            let up = e.htod(&lcg(29 + n as u64, n, limit * 3.0)).unwrap();
            // reference: the two-launch chain the door replaces
            let mut act = e.uninit(n).unwrap();
            e.swiglu_preclamped_mul_scaled(&gate, &up, gs, us, limit, &mut act, n)
                .unwrap();
            let (rq, rd) = e.quantize_q8_1(&act, 1, n).unwrap();
            let (fq, fd) = e
                .swiglu_preclamped_mul_scaled_q8_1(&gate, &up, gs, us, limit, n)
                .unwrap();
            let (rq, rd, fq, fd) = (
                e.dtoh_i8(&rq).unwrap(),
                e.dtoh(&rd).unwrap(),
                e.dtoh_i8(&fq).unwrap(),
                e.dtoh(&fd).unwrap(),
            );
            assert_eq!(rq.len(), n);
            assert_eq!(rd.len(), n / 32);
            assert_eq!(rq, fq, "q8 bytes differ at n={n} gs={gs} us={us}");
            let rdb: Vec<u32> = rd.iter().map(|x| x.to_bits()).collect();
            let fdb: Vec<u32> = fd.iter().map(|x| x.to_bits()).collect();
            assert_eq!(rdb, fdb, "block scales differ at n={n} gs={gs} us={us}");
            // non-vacuity: the row is not all zeros and not all saturated
            assert!(rq.iter().any(|&q| q != 0), "vacuous cell: all-zero q8 row");
            assert!(
                rq.iter().any(|&q| q.abs() < 127),
                "vacuous cell: every element saturated"
            );
            cells += 1;
        }
    }
    assert_eq!(cells, 10);

    // RED ARM 1: a different limit must move the bytes (the kernel applies the clamp it is given).
    let n = 2048usize;
    let gate = e.htod(&lcg(5, n, 21.0)).unwrap();
    let up = e.htod(&lcg(7, n, 21.0)).unwrap();
    let (a, _) = e
        .swiglu_preclamped_mul_scaled_q8_1(&gate, &up, 1.0, 1.0, 7.0, n)
        .unwrap();
    let (b, _) = e
        .swiglu_preclamped_mul_scaled_q8_1(&gate, &up, 1.0, 1.0, 2.0, n)
        .unwrap();
    assert_ne!(
        e.dtoh_i8(&a).unwrap(),
        e.dtoh_i8(&b).unwrap(),
        "limit is not applied"
    );
    // RED ARM 2: the up scale must reach the up operand.
    let (c, _) = e
        .swiglu_preclamped_mul_scaled_q8_1(&gate, &up, 1.0, 0.25, 7.0, n)
        .unwrap();
    assert_ne!(
        e.dtoh_i8(&a).unwrap(),
        e.dtoh_i8(&c).unwrap(),
        "us is not applied"
    );
    // RED ARM 3: a width that is not a multiple of 32 is refused, never silently truncated.
    assert!(
        e.swiglu_preclamped_mul_scaled_q8_1(&gate, &up, 1.0, 1.0, 7.0, 1000)
            .is_err()
    );
}
