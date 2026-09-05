//! Gate for the MLA live-length twins (lane/mla-live-len-20260905): the dense decode core and the
//! latent append reading their length / slot from a device position word instead of a launch
//! scalar, BITWISE against the scalar launches at the same length, at the served decode geometry
//! (32 heads, kv_rank 512, d_rope 64, t_q = 1) over a 300-row cache at several positions, plus a
//! 3-row append; a red arm (pos_d + 1 moves both outputs). These are the primitives the middle
//! capture rides; they carry no door of their own.
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
fn mla_live_len_twins_match_scalar_launches_bitwise() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let (nh, r, dr, cap) = (32usize, 512usize, 64usize, 300usize);
    let width = r + dr;
    let cache = e.htod(&vecf(cap * width, 1, 2.0)).unwrap();
    let q_lat = e.htod(&vecf(nh * r, 2, 1.0)).unwrap();
    let q_pe = e.htod(&vecf(nh * dr, 3, 1.0)).unwrap();
    let scale = 0.0625f32;
    for pos in [0usize, 7, 128, 299] {
        let t_kv = pos + 1;
        let mut a = e.uninit(nh * r).unwrap();
        e.mla_attn_absorbed(&q_lat, &q_pe, &cache, &mut a, nh, r, dr, 1, t_kv, scale)
            .unwrap();
        let pos_d = e.htod_i32(&[pos as i32]).unwrap();
        let mut b = e.uninit(nh * r).unwrap();
        e.mla_attn_absorbed_live(&q_lat, &q_pe, &cache, &mut b, nh, r, dr, 1, &pos_d, scale)
            .unwrap();
        e.stream().synchronize().unwrap();
        let (ha, hb) = (e.dtoh(&a).unwrap(), e.dtoh(&b).unwrap());
        assert!(ha.iter().all(|v| v.is_finite()));
        assert_eq!(
            bits(&ha),
            bits(&hb),
            "attention core: live differs from scalar at pos {pos}"
        );
    }
    // red arm: pos_d + 1 changes the attention output (one more visible row)
    let mut a = e.uninit(nh * r).unwrap();
    e.mla_attn_absorbed(&q_lat, &q_pe, &cache, &mut a, nh, r, dr, 1, 129, scale)
        .unwrap();
    let pos_d = e.htod_i32(&[129]).unwrap();
    let mut b = e.uninit(nh * r).unwrap();
    e.mla_attn_absorbed_live(&q_lat, &q_pe, &cache, &mut b, nh, r, dr, 1, &pos_d, scale)
        .unwrap();
    e.stream().synchronize().unwrap();
    assert_ne!(
        bits(&e.dtoh(&a).unwrap()),
        bits(&e.dtoh(&b).unwrap()),
        "red arm: pos_d + 1 did not bite"
    );
    // append: scalar vs live at slot 41, t = 3
    for t in [1usize, 3] {
        let c_kv = e.htod(&vecf(t * r, 5, 1.0)).unwrap();
        let k_pe = e.htod(&vecf(t * dr, 6, 1.0)).unwrap();
        let mut cache_a = e.htod(&vecf(cap * width, 7, 1.0)).unwrap();
        let mut cache_b = e.htod(&vecf(cap * width, 7, 1.0)).unwrap();
        e.mla_append_latent(&mut cache_a, &c_kv, &k_pe, 41, t, r, dr)
            .unwrap();
        let pos_d = e.htod_i32(&[41]).unwrap();
        e.mla_append_latent_live(&mut cache_b, &c_kv, &k_pe, &pos_d, t, r, dr)
            .unwrap();
        e.stream().synchronize().unwrap();
        assert_eq!(
            bits(&e.dtoh(&cache_a).unwrap()),
            bits(&e.dtoh(&cache_b).unwrap()),
            "append: live differs from scalar (t={t})"
        );
        let pos_d2 = e.htod_i32(&[42]).unwrap();
        let mut cache_c = e.htod(&vecf(cap * width, 7, 1.0)).unwrap();
        e.mla_append_latent_live(&mut cache_c, &c_kv, &k_pe, &pos_d2, t, r, dr)
            .unwrap();
        e.stream().synchronize().unwrap();
        assert_ne!(
            bits(&e.dtoh(&cache_a).unwrap()),
            bits(&e.dtoh(&cache_c).unwrap()),
            "append red arm did not bite (t={t})"
        );
    }
    println!(
        "mla live-length twins: attention core and append bitwise = scalar launches; red arms bite"
    );
}
