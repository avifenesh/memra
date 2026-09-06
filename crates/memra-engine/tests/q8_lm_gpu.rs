//! Gate for the lane-major q8_0 twin (lane/q8-lane-major-20260907): `qmatvec_q8_0_mmvq_lm_v2`
//! on a `build_q8_lm_raw` slab against `qmatvec_q8_0_mmvq_rp_v2` on the rp slab from the SAME
//! 34 B blocks, BITWISE on every output at the two KDA W8 shapes and at t=1 and t=3. Both slabs
//! are pure byte permutations of the source blocks (checked back on the host), and the red arm
//! feeds the lane-major kernel the rp slab, which must NOT reproduce the bits.
use memra_engine::Engine;

fn synth_q8_rows(out_f: usize, in_f: usize, seed: u32) -> Vec<u8> {
    // GGUF q8_0 blocks: 2 B f16 scale + 32 int8
    let nblk = in_f / 32;
    let mut v = vec![0u8; out_f * nblk * 34];
    let mut s = seed;
    for blk in v.chunks_exact_mut(34) {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let sc: u16 = 0x2C00 | ((s >> 20) as u16 & 0x03FF); // small positive halves
        blk[0] = (sc & 0xFF) as u8;
        blk[1] = (sc >> 8) as u8;
        for q in &mut blk[2..34] {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *q = (s >> 24) as u8;
        }
    }
    v
}
fn vecf(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        })
        .collect()
}

#[test]
fn lane_major_q8_is_bitwise_rp_v2() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    for &(in_f, out_f) in &[(4096usize, 8192usize), (8192, 4096)] {
        let src = synth_q8_rows(out_f, in_f, 0x5151 + in_f as u32);
        let src_d = e.htod_bytes(&src).unwrap();
        let rp = e.build_q8_rp4_raw(&src_d, in_f, out_f).unwrap();
        let lm = e.build_q8_lm_raw(&src_d, in_f, out_f).unwrap();
        // both slabs hold exactly the source bytes, permuted
        let (h_rp, h_lm) = (e.dtoh_u8(&rp).unwrap(), e.dtoh_u8(&lm).unwrap());
        let nblk = in_f / 32;
        for o in 0..out_f.min(7) {
            for b in 0..nblk {
                let blk = &src[(o * nblk + b) * 34..(o * nblk + b) * 34 + 34];
                assert_eq!(
                    &h_rp[(o * nblk + b) * 32..(o * nblk + b) * 32 + 32],
                    &blk[2..34],
                    "rp q o={o} b={b}"
                );
                for h in 0..2 {
                    let at = o * nblk * 32 + (h * nblk + b) * 16;
                    assert_eq!(
                        &h_lm[at..at + 16],
                        &blk[2 + h * 16..2 + h * 16 + 16],
                        "lm q o={o} b={b} h={h}"
                    );
                }
                let qplane = out_f * nblk * 32;
                assert_eq!(
                    &h_lm[qplane + (o * nblk + b) * 2..qplane + (o * nblk + b) * 2 + 2],
                    &blk[0..2]
                );
            }
        }
        for &t in &[1usize, 3] {
            let x = e.htod(&vecf(t * in_f, 31 + t as u64)).unwrap();
            let (aq, ad) = e.quantize_q8_1(&x, t, in_f).unwrap();
            let mut y_rp = e.zeros(t * out_f).unwrap();
            let mut y_lm = e.zeros(t * out_f).unwrap();
            e.qmatvec_q8_0_rp_v2_raw_arm(&rp, &aq, &ad, &mut y_rp, in_f, out_f, t, false)
                .unwrap();
            e.qmatvec_q8_0_lm_v2_raw(&lm, &aq, &ad, &mut y_lm, in_f, out_f, t)
                .unwrap();
            let (a, b) = (e.dtoh(&y_rp).unwrap(), e.dtoh(&y_lm).unwrap());
            let mism = a
                .iter()
                .zip(&b)
                .filter(|(p, q)| p.to_bits() != q.to_bits())
                .count();
            assert_eq!(
                mism, 0,
                "lane-major differs from rp_v2 at {in_f}->{out_f} t={t}"
            );
            assert!(a.iter().any(|v| *v != 0.0), "vacuous");
            // red: the rp slab through the lane-major kernel
            let mut y_red = e.zeros(t * out_f).unwrap();
            e.qmatvec_q8_0_lm_v2_raw(&rp, &aq, &ad, &mut y_red, in_f, out_f, t)
                .unwrap();
            let red = e.dtoh(&y_red).unwrap();
            assert!(
                a.iter().zip(&red).any(|(p, q)| p.to_bits() != q.to_bits()),
                "red arm matched"
            );
        }
    }
}
