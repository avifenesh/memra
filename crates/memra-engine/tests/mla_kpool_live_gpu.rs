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
    let (heads, d, pool, select_k_cap, cap) = (32usize, 128usize, 16usize, 8usize, 512usize);
    let width_cap = select_k_cap * pool + pool - 1; // index_width at the capacity (always_tail)
    let q = e.htod(&vecf(heads * d, 11, 1.0)).unwrap();
    let hw = e.htod(&vecf(heads, 12, 1.0)).unwrap();
    let keys = e.htod(&vecf(cap * d, 13, 1.0)).unwrap();
    for n_pools in [5usize, 64, 300] {
        let pos = n_pools * pool + 3; // the query's position: n_pools complete pools before it
        let first_pos = pos;
        let select_k = select_k_cap.min(n_pools);
        let width = select_k * pool + pool - 1;
        // scalar scorer (DSA level path) vs live (pos_d)
        let mut sa = e.htod(&vec![0.0f32; cap]).unwrap();
        let mut sb = e.htod(&vec![0.0f32; cap]).unwrap();
        unsafe {
            std::env::set_var("MEMRA_B200_DSA_DECODE", "1");
        }
        e.mla_kpool_score(
            &q, &keys, &hw, &mut sa, 1, heads, d, n_pools, pool, first_pos, 0.088, 1.0,
        )
        .unwrap();
        let pos_d = e.htod_i32(&[pos as i32]).unwrap();
        e.mla_kpool_score_dsa_live(
            &q, &keys, &hw, &mut sb, 1, heads, d, &pos_d, cap, pool, 0.088, 1.0,
        )
        .unwrap();
        e.stream().synchronize().unwrap();
        let (ha, hb) = (e.dtoh(&sa).unwrap(), e.dtoh(&sb).unwrap());
        assert_eq!(
            bits(&ha[..n_pools]),
            bits(&hb[..n_pools]),
            "score: live differs from scalar at n_pools {n_pools}"
        );
        // the reference-live scorer (geometries without a head-blocked instantiation: the rig
        // fixture's heads=1, d=8) vs the shipped dispatch at level 0
        {
            let (h1, d1) = (1usize, 8usize);
            let q1 = e.htod(&vecf(h1 * d1, 41, 1.0)).unwrap();
            let hw1 = e.htod(&vecf(h1, 42, 1.0)).unwrap();
            let keys1 = e.htod(&vecf(cap * d1, 43, 1.0)).unwrap();
            let mut ra = e.htod(&vec![0.0f32; cap]).unwrap();
            let mut rb = e.htod(&vec![0.0f32; cap]).unwrap();
            unsafe {
                std::env::set_var("MEMRA_B200_DSA_DECODE", "0");
            }
            e.mla_kpool_score(
                &q1, &keys1, &hw1, &mut ra, 1, h1, d1, n_pools, pool, first_pos, 0.35, 1.0,
            )
            .unwrap();
            e.mla_kpool_score_live(
                &q1, &keys1, &hw1, &mut rb, 1, h1, d1, &pos_d, cap, pool, 0.35, 1.0,
            )
            .unwrap();
            e.stream().synchronize().unwrap();
            let (xa, xb) = (e.dtoh(&ra).unwrap(), e.dtoh(&rb).unwrap());
            assert_eq!(
                bits(&xa[..n_pools]),
                bits(&xb[..n_pools]),
                "ref-live scorer differs at n_pools {n_pools}"
            );
        }
        // scalar single-CTA selector (host-derived select_k / width) vs live (capacity layout)
        let mut ia = e.htod_i32(&vec![-1i32; width]).unwrap();
        let mut ib = e.htod_i32(&vec![7i32; width_cap]).unwrap();
        unsafe {
            std::env::set_var("MEMRA_B200_DSA_SELECT", "0");
        }
        e.mla_kpool_select(
            &sa, &mut ia, 1, n_pools, pool, select_k, width, first_pos, true,
        )
        .unwrap();
        let mut width_d = e.htod_i32(&[-1]).unwrap();
        e.mla_kpool_select_live(
            &sa,
            &mut ib,
            &mut width_d,
            1,
            &pos_d,
            cap,
            pool,
            select_k_cap,
            width_cap,
            true,
        )
        .unwrap();
        e.stream().synchronize().unwrap();
        let (ka, kb) = (e.dtoh_i32(&ia).unwrap(), e.dtoh_i32(&ib).unwrap());
        assert_eq!(
            ka[..],
            kb[..width],
            "select: live differs from scalar on the token's width at n_pools {n_pools}"
        );
        assert!(
            kb[width..].iter().all(|&v| v == -1),
            "select: capacity tail not sentinel-filled at n_pools {n_pools}"
        );
        assert!(
            ka.iter().any(|&v| v >= 0),
            "selector emitted nothing at n_pools {n_pools}"
        );
        assert_eq!(
            e.dtoh_i32(&width_d).unwrap()[0] as usize,
            width,
            "live selector published the wrong width"
        );

        // gathered attention over the capacity-layout idx at the LIVE width vs the scalar launch
        // over the exact-width idx: shipped kernel and the warp-online chunked arm, bitwise.
        let (nh, r) = (4usize, 512usize);
        let rows = pos + 1;
        let latent = e.htod(&vecf(rows * r, 21 + n_pools as u64, 1.0)).unwrap();
        let q_lat = e.htod(&vecf(nh * r, 22, 1.0)).unwrap();
        let q_pe = e.htod(&[0.0f32]).unwrap();
        let mut oa = e.htod(&vec![0.0f32; nh * r]).unwrap();
        let mut ob = e.htod(&vec![0.0f32; nh * r]).unwrap();
        unsafe {
            std::env::set_var("MEMRA_B200_DSA_DECODE", "0");
        }
        e.mla_attn_gathered(
            &q_lat, &q_pe, &latent, &ia, &mut oa, nh, r, 0, 1, width, 0.044,
        )
        .unwrap();
        e.mla_attn_gathered_live(
            &q_lat, &q_pe, &latent, &ib, &mut ob, nh, r, 0, 1, &width_d, width_cap, 0.044,
        )
        .unwrap();
        e.stream().synchronize().unwrap();
        assert_eq!(
            bits(&e.dtoh(&oa).unwrap()),
            bits(&e.dtoh(&ob).unwrap()),
            "gathered: live differs at n_pools {n_pools}"
        );
        let chunks = 32usize;
        let mut parts = (
            e.htod(&vec![0.0f32; nh * chunks]).unwrap(),
            e.htod(&vec![0.0f32; nh * chunks]).unwrap(),
            e.htod(&vec![0.0f32; nh * chunks * r]).unwrap(),
        );
        e.mla_dsa_attn_warp_online(
            &q_lat, &q_pe, &latent, &ia, &mut oa, &mut parts, nh, r, 0, 1, width, chunks, 0.044,
        )
        .unwrap();
        e.mla_dsa_attn_warp_online_live(
            &q_lat, &q_pe, &latent, &ib, &mut ob, &mut parts, nh, r, 0, 1, &width_d, width_cap,
            chunks, 0.044,
        )
        .unwrap();
        e.stream().synchronize().unwrap();
        let (wa, wb) = (e.dtoh(&oa).unwrap(), e.dtoh(&ob).unwrap());
        assert_eq!(
            bits(&wa),
            bits(&wb),
            "warp-online: live differs at n_pools {n_pools}"
        );
        // red arms for the width word, at the counts where the token's width is below the
        // capacity (select_k saturates at top_k / pool from 8 pools on, and then width == cap):
        // the slots past `width` hold VALID rows here (not the selector's sentinels), so a twin
        // that walked to the capacity width would attend them.
        if width < width_cap {
            let mut full = kb.clone();
            for (j, v) in full.iter_mut().enumerate().skip(width) {
                *v = (j % rows) as i32;
            }
            let ib_full = e.htod_i32(&full).unwrap();
            e.mla_dsa_attn_warp_online_live(
                &q_lat, &q_pe, &latent, &ib_full, &mut ob, &mut parts, nh, r, 0, 1, &width_d,
                width_cap, chunks, 0.044,
            )
            .unwrap();
            e.stream().synchronize().unwrap();
            assert_eq!(
                bits(&wa),
                bits(&e.dtoh(&ob).unwrap()),
                "warp-online live read past the width word at n_pools {n_pools}"
            );
            let cap_w = e.htod_i32(&[width_cap as i32]).unwrap();
            e.mla_dsa_attn_warp_online_live(
                &q_lat, &q_pe, &latent, &ib_full, &mut ob, &mut parts, nh, r, 0, 1, &cap_w,
                width_cap, chunks, 0.044,
            )
            .unwrap();
            e.stream().synchronize().unwrap();
            assert_ne!(
                bits(&wa),
                bits(&e.dtoh(&ob).unwrap()),
                "warp-online red arm (capacity width over valid rows) did not bite at n_pools {n_pools}"
            );
            e.mla_attn_gathered_live(
                &q_lat, &q_pe, &latent, &ib_full, &mut ob, nh, r, 0, 1, &cap_w, width_cap, 0.044,
            )
            .unwrap();
            e.stream().synchronize().unwrap();
            assert_ne!(
                bits(&wa),
                bits(&e.dtoh(&ob).unwrap()),
                "gathered red arm (capacity width over valid rows) did not bite at n_pools {n_pools}"
            );
        }
        // indexer state append + pool-key build, live vs the host incremental program
        let ring = 0usize;
        let (kn, gt) = (
            e.htod(&vecf(d, 31, 1.0)).unwrap(),
            e.htod(&vecf(d, 32, 1.0)).unwrap(),
        );
        let ape = e.htod(&vecf(pool * d, 33, 1.0)).unwrap();
        let mut plane_a = e.htod(&vecf(rows * 2 * d, 34, 1.0)).unwrap();
        let mut plane_b = e.htod(&vecf(rows * 2 * d, 34, 1.0)).unwrap();
        e.mla_index_append(&mut plane_a, &kn, &gt, 0, pos, 1, d, d, ring)
            .unwrap();
        e.mla_index_append_live(&mut plane_b, &kn, &gt, &pos_d, 1, d, d, ring)
            .unwrap();
        e.stream().synchronize().unwrap();
        assert_eq!(
            bits(&e.dtoh(&plane_a).unwrap()),
            bits(&e.dtoh(&plane_b).unwrap()),
            "index append: live differs at pos {pos}"
        );
        for extra in [0usize, pool - 4] {
            // pos + extra: `extra == pool - 4` lands on a pool boundary ((pos + 1) % pool == 0)
            let posx = pos + extra;
            let n_complete = (posx + 1) / pool;
            let posx_d = e.htod_i32(&[posx as i32]).unwrap();
            let mut keys_a = e.htod(&vec![-7.0f32; cap * d]).unwrap();
            let mut keys_b = e.htod(&vec![-7.0f32; cap * d]).unwrap();
            // the host program at t = 1 from `pools_ready = n_complete_before`
            let before = posx / pool;
            e.mla_kpool_pool_keys(
                &plane_a,
                &ape,
                &mut keys_a,
                before,
                n_complete,
                pool,
                d,
                ring,
            )
            .unwrap();
            e.mla_kpool_pool_keys_live(&plane_a, &ape, &mut keys_b, &posx_d, 1, pool, d, ring)
                .unwrap();
            e.stream().synchronize().unwrap();
            let (pa, pb) = (e.dtoh(&keys_a).unwrap(), e.dtoh(&keys_b).unwrap());
            assert_eq!(
                bits(&pa),
                bits(&pb),
                "pool keys: live differs at pos {posx} (complete={})",
                n_complete > before
            );
            if n_complete > before {
                assert!(
                    pa[before * d..n_complete * d].iter().all(|&v| v != -7.0),
                    "pool-key build wrote nothing at {posx}"
                );
            }
        }
        // red arm: pos - pool drops a pool from the live scorer's view
        if n_pools > 1 {
            let pos1 = e.htod_i32(&[(pos - pool) as i32]).unwrap();
            let mut sc = e.htod(&vec![0.0f32; cap]).unwrap();
            e.mla_kpool_score_dsa_live(
                &q, &keys, &hw, &mut sc, 1, heads, d, &pos1, cap, pool, 0.088, 1.0,
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
        "MLA middle live twins (pos_d): scorer (head-blocked + reference), capacity-width \
         selector (+width word), index append, pool-key build, gathered + warp-online \
         attention at the live width: bitwise = scalar at counts 5/64/300 under cap 512; red \
         arms bite"
    );
}

/// The verify-walk shape (t rows at consecutive positions, row 0 at pos_d[0]): every live twin
/// at t = 3 bitwise = its scalar launch at the host's (slot, t) geometry, with the score and
/// index planes at their capacity strides.
#[test]
fn kpool_live_twins_match_scalar_launches_at_t_rows() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let (heads, d, pool, select_k_cap, cap, t) =
        (32usize, 128usize, 16usize, 8usize, 512usize, 3usize);
    let width_cap = select_k_cap * pool + pool - 1;
    let q = e.htod(&vecf(t * heads * d, 51, 1.0)).unwrap();
    let hw = e.htod(&vecf(t * heads, 52, 1.0)).unwrap();
    let keys = e.htod(&vecf(cap * d, 53, 1.0)).unwrap();
    for slot in [5usize * 16 + 14, 64 * 16 + 3, 300 * 16 + 15] {
        // slot = the first row's position; the call-wide pool count covers slot + t rows
        let n_pools = (slot + t) / pool;
        let select_k = select_k_cap.min(n_pools);
        let width = select_k * pool + pool - 1;
        let pos_d = e.htod_i32(&[slot as i32]).unwrap();
        // scorer: host plane [t, n_pools] vs live plane [t, cap]
        let mut sa = e.htod(&vec![0.0f32; t * n_pools]).unwrap();
        let mut sb = e.htod(&vec![7.0f32; t * cap]).unwrap();
        unsafe {
            std::env::set_var("MEMRA_B200_DSA_DECODE", "0");
        }
        e.mla_kpool_score(
            &q, &keys, &hw, &mut sa, t, heads, d, n_pools, pool, slot, 0.088, 1.0,
        )
        .unwrap();
        e.mla_kpool_score_live(
            &q, &keys, &hw, &mut sb, t, heads, d, &pos_d, cap, pool, 0.088, 1.0,
        )
        .unwrap();
        e.stream().synchronize().unwrap();
        let (ha, hb) = (e.dtoh(&sa).unwrap(), e.dtoh(&sb).unwrap());
        for r in 0..t {
            assert_eq!(
                bits(&ha[r * n_pools..(r + 1) * n_pools]),
                bits(&hb[r * cap..r * cap + n_pools]),
                "score row {r}: live differs at slot {slot}"
            );
        }
        // selector: host rows at stride width vs live rows at stride width_cap
        let mut ia = e.htod_i32(&vec![-1i32; t * width]).unwrap();
        let mut ib = e.htod_i32(&vec![7i32; t * width_cap]).unwrap();
        let mut width_d = e.htod_i32(&[-1]).unwrap();
        unsafe {
            std::env::set_var("MEMRA_B200_DSA_SELECT", "0");
        }
        e.mla_kpool_select(&sa, &mut ia, t, n_pools, pool, select_k, width, slot, true)
            .unwrap();
        e.mla_kpool_select_live(
            &sb,
            &mut ib,
            &mut width_d,
            t,
            &pos_d,
            cap,
            pool,
            select_k_cap,
            width_cap,
            true,
        )
        .unwrap();
        e.stream().synchronize().unwrap();
        let (ka, kb) = (e.dtoh_i32(&ia).unwrap(), e.dtoh_i32(&ib).unwrap());
        assert_eq!(
            e.dtoh_i32(&width_d).unwrap()[0] as usize,
            width,
            "width word at slot {slot}"
        );
        for r in 0..t {
            assert_eq!(
                ka[r * width..(r + 1) * width],
                kb[r * width_cap..r * width_cap + width],
                "select row {r}: live differs at slot {slot}"
            );
            assert!(
                kb[r * width_cap + width..(r + 1) * width_cap]
                    .iter()
                    .all(|&v| v == -1)
            );
        }
        // gathered attention at t_q = t: idx stride width vs width_cap, live width from the word
        let (nh, rk) = (4usize, 512usize);
        let rows = slot + t;
        let latent = e.htod(&vecf(rows * rk, 61, 1.0)).unwrap();
        let q_lat = e.htod(&vecf(t * nh * rk, 62, 1.0)).unwrap();
        let q_pe = e.htod(&[0.0f32]).unwrap();
        let mut oa = e.htod(&vec![0.0f32; t * nh * rk]).unwrap();
        let mut ob = e.htod(&vec![0.0f32; t * nh * rk]).unwrap();
        e.mla_attn_gathered(
            &q_lat, &q_pe, &latent, &ia, &mut oa, nh, rk, 0, t, width, 0.044,
        )
        .unwrap();
        e.mla_attn_gathered_live(
            &q_lat, &q_pe, &latent, &ib, &mut ob, nh, rk, 0, t, &width_d, width_cap, 0.044,
        )
        .unwrap();
        e.stream().synchronize().unwrap();
        assert_eq!(
            bits(&e.dtoh(&oa).unwrap()),
            bits(&e.dtoh(&ob).unwrap()),
            "gathered t={t} differs at slot {slot}"
        );
        let chunks = 32usize;
        let mut parts = (
            e.htod(&vec![0.0f32; t * nh * chunks]).unwrap(),
            e.htod(&vec![0.0f32; t * nh * chunks]).unwrap(),
            e.htod(&vec![0.0f32; t * nh * chunks * rk]).unwrap(),
        );
        e.mla_dsa_attn_warp_online(
            &q_lat, &q_pe, &latent, &ia, &mut oa, &mut parts, nh, rk, 0, t, width, chunks, 0.044,
        )
        .unwrap();
        e.mla_dsa_attn_warp_online_live(
            &q_lat, &q_pe, &latent, &ib, &mut ob, &mut parts, nh, rk, 0, t, &width_d, width_cap,
            chunks, 0.044,
        )
        .unwrap();
        e.stream().synchronize().unwrap();
        assert_eq!(
            bits(&e.dtoh(&oa).unwrap()),
            bits(&e.dtoh(&ob).unwrap()),
            "warp-online t={t} differs at slot {slot}"
        );
        // indexer append of t rows + the pool-key build over the pools those rows complete
        let ring = 0usize;
        let (kn, gt) = (
            e.htod(&vecf(t * d, 71, 1.0)).unwrap(),
            e.htod(&vecf(t * d, 72, 1.0)).unwrap(),
        );
        let ape = e.htod(&vecf(pool * d, 73, 1.0)).unwrap();
        let mut plane_a = e.htod(&vecf(rows * 2 * d, 74, 1.0)).unwrap();
        let mut plane_b = e.htod(&vecf(rows * 2 * d, 74, 1.0)).unwrap();
        e.mla_index_append(&mut plane_a, &kn, &gt, 0, slot, t, d, d, ring)
            .unwrap();
        e.mla_index_append_live(&mut plane_b, &kn, &gt, &pos_d, t, d, d, ring)
            .unwrap();
        e.stream().synchronize().unwrap();
        assert_eq!(
            bits(&e.dtoh(&plane_a).unwrap()),
            bits(&e.dtoh(&plane_b).unwrap()),
            "append t={t} differs at slot {slot}"
        );
        let (before, ready_now) = (slot / pool, (slot + t) / pool);
        let mut keys_a = e.htod(&vec![-7.0f32; cap * d]).unwrap();
        let mut keys_b = e.htod(&vec![-7.0f32; cap * d]).unwrap();
        e.mla_kpool_pool_keys(
            &plane_a,
            &ape,
            &mut keys_a,
            before,
            ready_now,
            pool,
            d,
            ring,
        )
        .unwrap();
        e.mla_kpool_pool_keys_live(&plane_a, &ape, &mut keys_b, &pos_d, t, pool, d, ring)
            .unwrap();
        e.stream().synchronize().unwrap();
        let (pa, pb) = (e.dtoh(&keys_a).unwrap(), e.dtoh(&keys_b).unwrap());
        assert_eq!(
            bits(&pa),
            bits(&pb),
            "pool keys t={t} differ at slot {slot} (built {})",
            ready_now - before
        );
        if ready_now > before {
            assert!(
                pa[before * d..ready_now * d].iter().all(|&v| v != -7.0),
                "pool-key build wrote nothing at {slot}"
            );
        }
    }
    println!(
        "MLA live twins at t=3 rows (the verify-walk shape): scorer, selector (+width word), \
         gathered + warp-online attention, index append, pool-key build: bitwise = scalar at \
         three slots (one crossing a pool boundary) under cap 512"
    );
}
