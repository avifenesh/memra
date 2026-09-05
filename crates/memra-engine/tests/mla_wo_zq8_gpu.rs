//! Gate for `MEMRA_MLA_WO_ZQ8` (lane/mla-wo-zq8-20260905): the coalesce-arm decompress_v launch
//! with the q8_1 epilogue against the plain launch followed by `quantize_q8_1` over the token row,
//! BITWISE on all three planes (f32 attention output, q8 bytes, block scales), for the f32 plane
//! and the BF16 plane, at the served MLA decode geometry (32 heads, d_v 128, kv_rank 512, t = 1;
//! `MEMRA_MLA_COALESCE=1 MEMRA_MLA_DECODE_SPLIT=1` puts split 4 on the launch, 32 outputs per
//! block = one q8 block); a red arm (a perturbed plane must move the pair); and the refusal when
//! the split leaves no whole q8 block per block (d_v 48 -> split 1 -> `None`, nothing launched).
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
fn bf16_bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|x| (x.to_bits() >> 16) as u16).collect()
}

#[test]
fn mla_wo_zq8_matches_decompress_then_quantize_bitwise() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    unsafe {
        std::env::set_var("MEMRA_MLA_COALESCE", "1");
        std::env::set_var("MEMRA_MLA_DECODE_SPLIT", "1");
    }
    let (nh, dv, r, t) = (32usize, 128usize, 512usize, 1usize);
    let o_lat = e.htod(&vecf(t * nh * r, 3, 2.0)).unwrap();
    let wv = vecf(nh * dv * r, 4, 0.5);
    // f32 plane
    let wv_d = e.htod(&wv).unwrap();
    let mut ref_out = e.uninit(t * nh * dv).unwrap();
    e.mla_decompress_v(&o_lat, &wv_d, &mut ref_out, t, nh, dv, r)
        .unwrap();
    let (rq, rd) = e.quantize_q8_1(&ref_out, t, nh * dv).unwrap();
    let mut out = e.uninit(t * nh * dv).unwrap();
    let (q, d) = e
        .mla_decompress_v_zq8(&o_lat, &wv_d, &mut out, t, nh, dv, r)
        .unwrap()
        .expect("the f32 zq8 arm refused the served geometry");
    e.stream().synchronize().unwrap();
    assert_eq!(
        bits(&e.dtoh(&ref_out).unwrap()),
        bits(&e.dtoh(&out).unwrap()),
        "f32: attn differs"
    );
    assert_eq!(
        e.dtoh_i8(&rq).unwrap(),
        e.dtoh_i8(&q).unwrap(),
        "f32: q8 bytes differ"
    );
    assert_eq!(
        bits(&e.dtoh(&rd).unwrap()),
        bits(&e.dtoh(&d).unwrap()),
        "f32: scales differ"
    );
    // BF16 plane (the exact-widening arm): reference = the bf16 plain kernel + quantize
    let w16 = e.htod_u16(&bf16_bits(&wv)).unwrap();
    let mut ref16 = e.uninit(t * nh * dv).unwrap();
    assert!(
        e.mla_decompress_v_bf16(&o_lat, &w16, &mut ref16, t, nh, dv, r)
            .unwrap()
    );
    let (rq16, rd16) = e.quantize_q8_1(&ref16, t, nh * dv).unwrap();
    let mut out16 = e.uninit(t * nh * dv).unwrap();
    let (q16, d16) = e
        .mla_decompress_v_bf16_zq8(&o_lat, &w16, &mut out16, t, nh, dv, r)
        .unwrap()
        .expect("the bf16 zq8 arm refused the served geometry");
    e.stream().synchronize().unwrap();
    assert_eq!(
        bits(&e.dtoh(&ref16).unwrap()),
        bits(&e.dtoh(&out16).unwrap()),
        "bf16: attn differs"
    );
    assert_eq!(
        e.dtoh_i8(&rq16).unwrap(),
        e.dtoh_i8(&q16).unwrap(),
        "bf16: q8 bytes differ"
    );
    assert_eq!(
        bits(&e.dtoh(&rd16).unwrap()),
        bits(&e.dtoh(&d16).unwrap()),
        "bf16: scales differ"
    );
    // red arm
    let wv2_d = e.htod(&vecf(nh * dv * r, 9, 0.5)).unwrap();
    let mut out2 = e.uninit(t * nh * dv).unwrap();
    let (q2, _) = e
        .mla_decompress_v_zq8(&o_lat, &wv2_d, &mut out2, t, nh, dv, r)
        .unwrap()
        .unwrap();
    e.stream().synchronize().unwrap();
    assert_ne!(
        e.dtoh_i8(&q).unwrap(),
        e.dtoh_i8(&q2).unwrap(),
        "red arm: perturbed plane did not bite"
    );
    // refusal: d_v 48 -> the decode split caps at 1 -> no fused arm
    let o48 = e.htod(&vecf(nh * r, 5, 1.0)).unwrap();
    let w48 = e.htod(&vecf(nh * 48 * r, 6, 0.5)).unwrap();
    let mut out48 = e.uninit(nh * 48).unwrap();
    assert!(
        e.mla_decompress_v_zq8(&o48, &w48, &mut out48, 1, nh, 48, r)
            .unwrap()
            .is_none()
    );
    println!(
        "mla wo zq8: bitwise = decompress + quantize_q8_1 on all planes (f32 and bf16); red arm bites"
    );
}
