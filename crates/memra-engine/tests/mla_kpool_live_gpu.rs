//! Gate for the k-pool live-count twins (lane/mla-kpool-live-20260905): the DSA decode scorer and
//! the single-CTA selector reading `n_pools` from a device word (capacity grid for the scorer),
//! BITWISE against the scalar launches at the same count, at the served indexer geometry (32
//! index heads, head_dim 128, pool 64, select_k 8 -> 32 rows, t_q = 1) for counts 5, 64, 300
//! under a capacity of 512; a red arm (count - 1 changes the scores' -inf tail and, with a
//! planted top pool at the end, the selection). No door: the middle-capture arc consumes them.
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
fn kpool_live_count_twins_match_scalar_launches_bitwise() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let (heads, d, pool, select_k, cap) = (32usize, 128usize, 16usize, 8usize, 512usize);
    let width = select_k * pool + pool; // rows the selector may emit (k pools + the tail)
    let q = e.htod(&vecf(heads * d, 11, 1.0)).unwrap();
    let hw = e.htod(&vecf(heads, 12, 1.0)).unwrap();
    let keys = e.htod(&vecf(cap * d, 13, 1.0)).unwrap();
    for n_pools in [5usize, 64, 300] {
        let first_pos = n_pools * pool + 3; // the query sits after every complete pool
        // scalar scorer (DSA level path) vs live
        let mut sa = e.htod(&vec![0.0f32; cap]).unwrap();
        let mut sb = e.htod(&vec![0.0f32; cap]).unwrap();
        unsafe {
            std::env::set_var("MEMRA_B200_DSA_DECODE", "1");
        }
        e.mla_kpool_score(
            &q, &keys, &hw, &mut sa, 1, heads, d, n_pools, pool, first_pos, 0.088, 1.0,
        )
        .unwrap();
        let nd = e.htod_i32(&[n_pools as i32]).unwrap();
        e.mla_kpool_score_dsa_live(
            &q, &keys, &hw, &mut sb, heads, d, &nd, cap, pool, first_pos, 0.088, 1.0,
        )
        .unwrap();
        e.stream().synchronize().unwrap();
        let (ha, hb) = (e.dtoh(&sa).unwrap(), e.dtoh(&sb).unwrap());
        assert_eq!(
            bits(&ha[..n_pools]),
            bits(&hb[..n_pools]),
            "score: live differs from scalar at n_pools {n_pools}"
        );
        // scalar single-CTA selector vs live, on the same scores
        let mut ia = e.htod_i32(&vec![-1i32; width]).unwrap();
        let mut ib = e.htod_i32(&vec![-1i32; width]).unwrap();
        unsafe {
            std::env::set_var("MEMRA_B200_DSA_SELECT", "0");
        }
        e.mla_kpool_select(
            &sa, &mut ia, 1, n_pools, pool, select_k, width, first_pos, true,
        )
        .unwrap();
        e.mla_kpool_select_live(&sa, &mut ib, 1, &nd, pool, select_k, width, first_pos, true)
            .unwrap();
        e.stream().synchronize().unwrap();
        let (ka, kb) = (e.dtoh_i32(&ia).unwrap(), e.dtoh_i32(&ib).unwrap());
        assert_eq!(
            ka, kb,
            "select: live differs from scalar at n_pools {n_pools}"
        );
        assert!(
            ka.iter().any(|&v| v >= 0),
            "selector emitted nothing at n_pools {n_pools}"
        );
        // red arm: count - 1 drops the last pool from both
        if n_pools > 1 {
            let nd1 = e.htod_i32(&[n_pools as i32 - 1]).unwrap();
            let mut sc = e.htod(&vec![0.0f32; cap]).unwrap();
            e.mla_kpool_score_dsa_live(
                &q, &keys, &hw, &mut sc, heads, d, &nd1, cap, pool, first_pos, 0.088, 1.0,
            )
            .unwrap();
            e.stream().synchronize().unwrap();
            let hc = e.dtoh(&sc).unwrap();
            assert_ne!(
                bits(&ha[..n_pools]),
                bits(&hc[..n_pools]),
                "score red arm did not bite at {n_pools}"
            );
        }
    }
    println!(
        "k-pool live-count twins: scorer and selector bitwise = scalar at counts 5/64/300 under cap 512; red arm bites"
    );
}
