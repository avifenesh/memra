//! Gate for `MEMRA_KDA_ONORM_ZQ8` (lane/kda-onorm-zq8-20260905): the fused gated-norm + q8_1
//! twin (`memra_kda_gated_rmsnorm_zq8_f32`) against `kda_gated_rmsnorm` followed by
//! `quantize_q8_1` over the token row, BITWISE on all three planes (f32 norm output, q8 bytes,
//! block scales), at the served KDA shape (32 heads x head_dim 128 = 4096 per token), t = 1 and
//! t = 3 rows; a red arm (a perturbed gate row must move the pair); and a shape refusal
//! (ncols % 32 != 0 is an error, not a silent launch).
use memra_engine::Engine;

fn vecf(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * amp
        })
        .collect()
}
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn kda_onorm_zq8_matches_norm_then_quantize_bitwise() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let (heads, hd) = (32usize, 128usize);
    let w = e.htod(&vecf(hd, 5, 2.0)).unwrap();
    for t in [1usize, 3] {
        let rows = t * heads;
        let core = e.htod(&vecf(rows * hd, 11 + t as u64, 4.0)).unwrap();
        let gate = e.htod(&vecf(rows * hd, 23 + t as u64, 6.0)).unwrap();
        // reference: plain norm, then quantize over the [t, heads*hd] token rows
        let mut ref_dst = e.uninit(rows * hd).unwrap();
        e.kda_gated_rmsnorm(&core, &w, &gate, &mut ref_dst, hd, rows, 1e-6)
            .unwrap();
        let (rq, rd) = e.quantize_q8_1(&ref_dst, t, heads * hd).unwrap();
        // fused
        let mut dst = e.uninit(rows * hd).unwrap();
        let (q, d) = e
            .kda_gated_rmsnorm_zq8(&core, &w, &gate, &mut dst, hd, rows, 1e-6)
            .unwrap();
        e.stream().synchronize().unwrap();
        let (a, b) = (e.dtoh(&ref_dst).unwrap(), e.dtoh(&dst).unwrap());
        assert_eq!(bits(&a), bits(&b), "norm output differs (t={t})");
        assert!(a.iter().all(|v| v.is_finite()));
        let (qa, qb) = (e.dtoh_i8(&rq).unwrap(), e.dtoh_i8(&q).unwrap());
        assert_eq!(qa, qb, "q8 bytes differ (t={t})");
        let (da, db) = (e.dtoh(&rd).unwrap(), e.dtoh(&d).unwrap());
        assert_eq!(bits(&da), bits(&db), "block scales differ (t={t})");
        assert_eq!(da.len(), t * heads * hd / 32);
        // red arm
        let gate2 = e.htod(&vecf(rows * hd, 99, 6.0)).unwrap();
        let mut dst2 = e.uninit(rows * hd).unwrap();
        let (q2, _) = e
            .kda_gated_rmsnorm_zq8(&core, &w, &gate2, &mut dst2, hd, rows, 1e-6)
            .unwrap();
        e.stream().synchronize().unwrap();
        assert_ne!(
            qb,
            e.dtoh_i8(&q2).unwrap(),
            "red arm: perturbed gate did not move the pair"
        );
    }
    // shape refusal
    let core = e.htod(&vecf(48, 1, 1.0)).unwrap();
    let gate = e.htod(&vecf(48, 2, 1.0)).unwrap();
    let w48 = e.htod(&vecf(48, 3, 1.0)).unwrap();
    let mut dst = e.uninit(48).unwrap();
    assert!(
        e.kda_gated_rmsnorm_zq8(&core, &w48, &gate, &mut dst, 48, 1, 1e-6)
            .is_err()
    );
    println!(
        "kda o_norm zq8: bitwise = norm + quantize_q8_1 on all planes at t=1,3; red arm bites"
    );
}
