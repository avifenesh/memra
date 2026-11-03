//! `MEMRA_KPOOL_SELECT_V2`: the single-CTA k-pool selector's v2 body (warp-aggregated histogram
//! atomics, warp-scan rank walk and emit prefix) against the shipped body, BITWISE on the emitted
//! index lists, for the scalar entry and the live twin, at pool counts from a handful to the
//! 61,323 the 245k-token trace ran at, with planted ties (an all-equal row is the contention
//! worst case), -inf and NaN pools, t_q 1 and 4. Engagement counted; a red arm (one selected
//! pool's score sent to -inf) must move the list THROUGH the door.
//!
//! Run: `flock /tmp/memra-5090.lock cargo test --release -p memra-engine --test mla_kpool_select_v2_gpu -- --ignored --nocapture`
use memra_engine::Engine;

fn vecf(n: usize, seed: u64, amp: f32) -> Vec<f32> {
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

fn set(key: &str, v: &str) {
    unsafe { std::env::set_var(key, v) };
}

/// One query row of pool scores with the hazards planted: every 97th pool -inf (invisible),
/// every 389th NaN (skipped), a value repeated at 13 scattered pools (ties resolve to the lower
/// pool), and for `flat` rows one value everywhere.
fn scores(n_pools: usize, seed: u64, flat: bool) -> Vec<f32> {
    let mut v = if flat {
        vec![0.75f32; n_pools]
    } else {
        vecf(n_pools, seed, 3.0)
    };
    for p in (0..n_pools).step_by(97) {
        v[p] = f32::NEG_INFINITY;
    }
    for p in (5..n_pools).step_by(389) {
        v[p] = f32::NAN;
    }
    for j in 0..13 {
        let p = (j * 1237 + 11) % n_pools;
        if v[p].is_finite() {
            v[p] = 2.5;
        }
    }
    v
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_kpool_select_v2_matches_shipped_bitwise() {
    let e = Engine::new(0).expect("CUDA engine on device 0");
    set("MEMRA_B200_DSA_SELECT", "0"); // the single-CTA path is the subject
    let (pool, select_k_cap) = (4usize, 512usize); // the served indexer geometry
    let mut cells = 0usize;
    for &(n_pools, flat) in &[
        (5usize, false),
        (64, false),
        (300, true),
        (4096, false),
        (4096, true),
        (61_323, false),
        (61_323, true),
    ] {
        for &t_q in &[1usize, 4usize] {
            let select_k = select_k_cap.min(n_pools);
            let width = select_k * pool + pool - 1;
            let first_pos = n_pools * pool - t_q; // (first_pos + t_q) / pool == n_pools
            let mut plane = Vec::with_capacity(t_q * n_pools);
            for r in 0..t_q {
                plane.extend(scores(
                    n_pools,
                    0x5E1 + (n_pools as u64) * 7 + r as u64,
                    flat,
                ));
            }
            let sc = e.htod(&plane).expect("scores");
            // scalar entry: v1 vs v2
            let mut i1 = e.htod_i32(&vec![7i32; t_q * width]).expect("i1");
            let mut i2 = e.htod_i32(&vec![9i32; t_q * width]).expect("i2");
            set("MEMRA_KPOOL_SELECT_V2", "0");
            e.mla_kpool_select(
                &sc, &mut i1, t_q, n_pools, pool, select_k, width, first_pos, true,
            )
            .expect("v1");
            let d0 = memra_engine::mla_ffi::kpool_select_v2_dispatches();
            set("MEMRA_KPOOL_SELECT_V2", "1");
            e.mla_kpool_select(
                &sc, &mut i2, t_q, n_pools, pool, select_k, width, first_pos, true,
            )
            .expect("v2");
            set("MEMRA_KPOOL_SELECT_V2", "0");
            assert!(
                memra_engine::mla_ffi::kpool_select_v2_dispatches() > d0,
                "v2 door did not engage (scalar) at n_pools={n_pools} t_q={t_q}"
            );
            e.stream().synchronize().expect("sync");
            let (h1, h2) = (e.dtoh_i32(&i1).expect("h1"), e.dtoh_i32(&i2).expect("h2"));
            let sel = h1.iter().filter(|&&x| x >= 0).count();
            assert!(sel > 0, "vacuous: nothing selected at n_pools={n_pools}");
            assert_eq!(
                h1, h2,
                "scalar v2 differs at n_pools={n_pools} t_q={t_q} flat={flat}"
            );
            // live twin: v1 vs v2 (capacity layout, positions from the device word)
            let cap = n_pools;
            let width_cap = select_k_cap * pool + pool - 1; // the capacity width the live audit wants
            let pos_d = e.htod_i32(&[first_pos as i32]).expect("pos_d");
            let mut l1 = e.htod_i32(&vec![7i32; t_q * width_cap]).expect("l1");
            let mut l2 = e.htod_i32(&vec![9i32; t_q * width_cap]).expect("l2");
            let mut w1 = e.htod_i32(&[-1]).expect("w1");
            let mut w2 = e.htod_i32(&[-1]).expect("w2");
            set("MEMRA_KPOOL_SELECT_V2", "0");
            e.mla_kpool_select_live(
                &sc,
                &mut l1,
                &mut w1,
                t_q,
                &pos_d,
                cap,
                pool,
                select_k_cap,
                width_cap,
                true,
            )
            .expect("live v1");
            set("MEMRA_KPOOL_SELECT_V2", "1");
            e.mla_kpool_select_live(
                &sc,
                &mut l2,
                &mut w2,
                t_q,
                &pos_d,
                cap,
                pool,
                select_k_cap,
                width_cap,
                true,
            )
            .expect("live v2");
            set("MEMRA_KPOOL_SELECT_V2", "0");
            e.stream().synchronize().expect("sync");
            assert_eq!(
                e.dtoh_i32(&l1).expect("l1"),
                e.dtoh_i32(&l2).expect("l2"),
                "live v2 differs at n_pools={n_pools} t_q={t_q} flat={flat}"
            );
            assert_eq!(
                e.dtoh_i32(&w1).expect("w1"),
                e.dtoh_i32(&w2).expect("w2"),
                "width word"
            );
            cells += 2;
        }
    }
    println!(
        "kpool select v2 PASS: {cells} (shape, t_q, twin) cells bitwise against the shipped body"
    );

    // RED ARM through the door: send one selected pool to -inf; the emitted list must move.
    let (n_pools, t_q) = (4096usize, 1usize);
    let select_k = select_k_cap.min(n_pools);
    let width = select_k * pool + pool - 1;
    let first_pos = n_pools * pool - t_q;
    let mut base = scores(n_pools, 0x77, false);
    let sc = e.htod(&base).expect("scores");
    let mut i2 = e.htod_i32(&vec![9i32; width]).expect("i2");
    set("MEMRA_KPOOL_SELECT_V2", "1");
    e.mla_kpool_select(
        &sc, &mut i2, t_q, n_pools, pool, select_k, width, first_pos, true,
    )
    .expect("v2 ref");
    e.stream().synchronize().expect("sync");
    let ref_list = e.dtoh_i32(&i2).expect("ref");
    let victim = ref_list[0] as usize / pool; // a selected pool
    base[victim] = f32::NEG_INFINITY;
    let sc2 = e.htod(&base).expect("scores red");
    let mut i3 = e.htod_i32(&vec![9i32; width]).expect("i3");
    e.mla_kpool_select(
        &sc2, &mut i3, t_q, n_pools, pool, select_k, width, first_pos, true,
    )
    .expect("v2 red");
    set("MEMRA_KPOOL_SELECT_V2", "0");
    e.stream().synchronize().expect("sync");
    let red = e.dtoh_i32(&i3).expect("red");
    assert_ne!(
        red, ref_list,
        "red arm: removing a selected pool left the list unchanged"
    );
    println!("kpool select v2 RED bites: pool {victim} removed moves the list");
}
