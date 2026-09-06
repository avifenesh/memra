//! The two small KDA-core folds of lane/launch-collapse-20260906 must be byte-identical to the
//! launches they replace: `l2_norm2_f32` (q and k norms in one grid) against two `l2_norm` calls,
//! and `memra_kda_gate_beta_f32` (forget gate + beta sigmoid) against `kda_gate` then `sigmoid`.
//! Red arms: the pair kernel must write BOTH outputs (a different second input moves only the
//! second output), and the gate kernel must read its own beta operand. Exactness only.
use memra_engine::Engine;

fn lcg(seed: u64, n: usize, amp: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / ((1u64 << 31) as f32);
            (u - 0.5) * 2.0 * amp
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
#[ignore]
fn gpu_l2_norm_pair_matches_two_launches_bitwise() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let eps = 1e-6f32;
    for &(ncols, rows) in &[(128usize, 64usize), (128, 1), (96, 7), (256, 3)] {
        let q = e.htod(&lcg(3, ncols * rows, 4.0)).unwrap();
        let k = e.htod(&lcg(5, ncols * rows, 0.05)).unwrap();
        let mut rq = e.uninit(ncols * rows).unwrap();
        let mut rk = e.uninit(ncols * rows).unwrap();
        e.l2_norm(&q, &mut rq, ncols, rows, eps).unwrap();
        e.l2_norm(&k, &mut rk, ncols, rows, eps).unwrap();
        let mut pq = e.uninit(ncols * rows).unwrap();
        let mut pk = e.uninit(ncols * rows).unwrap();
        e.l2_norm_pair(&q, &mut pq, &k, &mut pk, ncols, rows, eps)
            .unwrap();
        assert_eq!(
            bits(&e.dtoh(&rq).unwrap()),
            bits(&e.dtoh(&pq).unwrap()),
            "q rows differ at {ncols}x{rows}"
        );
        assert_eq!(
            bits(&e.dtoh(&rk).unwrap()),
            bits(&e.dtoh(&pk).unwrap()),
            "k rows differ at {ncols}x{rows}"
        );
        // red arm: a different second input must move only the second output
        let k2 = e.htod(&lcg(9, ncols * rows, 0.05)).unwrap();
        let mut pq2 = e.uninit(ncols * rows).unwrap();
        let mut pk2 = e.uninit(ncols * rows).unwrap();
        e.l2_norm_pair(&q, &mut pq2, &k2, &mut pk2, ncols, rows, eps)
            .unwrap();
        assert_eq!(bits(&e.dtoh(&pq).unwrap()), bits(&e.dtoh(&pq2).unwrap()));
        assert_ne!(
            bits(&e.dtoh(&pk).unwrap()),
            bits(&e.dtoh(&pk2).unwrap()),
            "second output ignores its input"
        );
    }
}

#[test]
#[ignore]
fn gpu_kda_gate_beta_matches_gate_then_sigmoid_bitwise() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    for &(qkv, head_dim, t) in &[(8192usize, 128usize, 1usize), (8192, 128, 4), (512, 64, 3)] {
        let heads = qkv / head_dim;
        let forget = e.htod(&lcg(11, qkv * t, 6.0)).unwrap();
        let dt_bias = e.htod(&lcg(13, qkv, 1.0)).unwrap();
        let a_log = e.htod(&lcg(17, heads, 2.0)).unwrap();
        let beta_raw = e.htod(&lcg(19, heads * t, 5.0)).unwrap();
        let lb = 0.75f32;
        let mut g_ref = e.uninit(qkv * t).unwrap();
        e.kda_gate(&forget, &dt_bias, &a_log, &mut g_ref, qkv, t, head_dim, lb)
            .unwrap();
        let mut b_ref = e.uninit(heads * t).unwrap();
        e.sigmoid(&beta_raw, &mut b_ref, heads * t).unwrap();
        let mut g = e.uninit(qkv * t).unwrap();
        let mut b = e.uninit(heads * t).unwrap();
        e.kda_gate_beta(
            &forget, &dt_bias, &a_log, &mut g, &beta_raw, &mut b, qkv, t, head_dim, lb,
        )
        .unwrap();
        assert_eq!(
            bits(&e.dtoh(&g_ref).unwrap()),
            bits(&e.dtoh(&g).unwrap()),
            "g differs at {qkv}/{head_dim}/{t}"
        );
        assert_eq!(
            bits(&e.dtoh(&b_ref).unwrap()),
            bits(&e.dtoh(&b).unwrap()),
            "beta differs at {qkv}/{head_dim}/{t}"
        );
        // red arms: lower_bound reaches g; beta_raw reaches beta
        let mut g2 = e.uninit(qkv * t).unwrap();
        let mut b2 = e.uninit(heads * t).unwrap();
        let beta_raw2 = e.htod(&lcg(23, heads * t, 5.0)).unwrap();
        e.kda_gate_beta(
            &forget, &dt_bias, &a_log, &mut g2, &beta_raw2, &mut b2, qkv, t, head_dim, 0.5,
        )
        .unwrap();
        assert_ne!(
            bits(&e.dtoh(&g).unwrap()),
            bits(&e.dtoh(&g2).unwrap()),
            "lower_bound not applied"
        );
        assert_ne!(
            bits(&e.dtoh(&b).unwrap()),
            bits(&e.dtoh(&b2).unwrap()),
            "beta_raw not read"
        );
    }
    // a head_dim that does not divide qkv is refused
    let x = e.htod(&lcg(1, 100, 1.0)).unwrap();
    let mut y = e.uninit(100).unwrap();
    let mut z = e.uninit(100).unwrap();
    assert!(
        e.kda_gate_beta(&x, &x, &x, &mut y, &x, &mut z, 100, 7, 1, 1.0)
            .is_err()
    );
}
