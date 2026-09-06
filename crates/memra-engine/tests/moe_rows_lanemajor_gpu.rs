//! Gate for the lane-major rows twins (lane/moe-rows-lanemajor-20260906): `_lm` and `_lm_w4`
//! gate/up and down against the served `_ilp` pair on the same slot-major bytes, BITWISE on
//! every output word at the served shape (4096 -> 2048, top-8, t=1) and at t=3. The activation
//! goes through `q8_to_lane_major`, which is also checked as a pure permutation (every word
//! present exactly once).
//!
//! RED ARM: feeding the lane-major kernel the UNPERMUTED activation must change the output.
//! Without it a bitwise pass could mean the two kernels read the same wrong words.
use cudarc::driver::DevicePtr;
use memra_engine::{Engine, QT_NVFP4_V2};

struct Lcg(u32);
impl Lcg {
    fn byte(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 24) as u8
    }
}
fn synth_rows(out_f: usize, in_f: usize, seed: u32) -> Vec<u8> {
    let nsb64 = in_f / 64;
    let mut w = vec![0u8; out_f * nsb64 * 36];
    let mut r = Lcg(seed);
    for chunk in w.chunks_exact_mut(36) {
        for d in &mut chunk[0..4] {
            *d = (r.byte() & 0x07) | 0x38;
        }
        for q in &mut chunk[4..36] {
            *q = r.byte();
        }
    }
    w
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
fn mism(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

#[test]
fn lane_major_rows_are_bitwise_the_served_pair() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    unsafe { std::env::set_var("MEMRA_MOE_VROWS_ILP", "1") };
    let (gin, n_ff, n_used) = (4096usize, 2048usize, 8usize);
    let stream = e.stream();
    for &t in &[1usize, 3] {
        let n_pairs = t * n_used;
        let grows = synth_rows(n_ff, gin, 0x8181);
        let drows = synth_rows(gin, n_ff, 0x4242);
        let mut ptrs = vec![0u64; 3 * n_pairs];
        let mut keep = Vec::new();
        for pr in 0..n_pairs {
            for plane in 0..3 {
                let mut d = if plane == 2 {
                    drows.clone()
                } else {
                    grows.clone()
                };
                d[5] ^= (pr * 3 + plane + 1) as u8;
                let v1 = e.htod_bytes(&d).unwrap();
                let buf = if plane == 2 {
                    e.nvfp4_expert_split_repack(&v1, 1, gin, n_ff / 64).unwrap()
                } else {
                    e.nvfp4_expert_split_repack(&v1, 1, n_ff, gin / 64).unwrap()
                };
                ptrs[plane * n_pairs + pr] = buf.device_ptr(&stream).0;
                keep.push(buf);
            }
        }
        let ptrs_d = e.htod_u64(&ptrs).unwrap();
        let scl: Vec<f32> = (0..3 * n_pairs).map(|pr| 0.5 + 0.01 * pr as f32).collect();
        let scl_d = e.htod(&scl).unwrap();
        let grb2 = (gin / 32) * 18;
        let drb2 = (n_ff / 32) * 18;
        let x_d = e.htod(&vecf(t * gin, 91)).unwrap();
        let (aq, ad) = e.quantize_q8_1(&x_d, t, gin).unwrap();
        let aq_lm = e.q8_to_lane_major(&aq, t, gin).unwrap();
        // the permutation holds every word exactly once
        {
            let h = e.dtoh_i8(&aq).unwrap();
            let hl = e.dtoh_i8(&aq_lm).unwrap();
            let nsb = gin / 32;
            for r in 0..t {
                for g in 0..nsb {
                    for w in 0..8 {
                        let src = (r * gin + g * 32 + w * 4)..(r * gin + g * 32 + w * 4 + 4);
                        let dst = (r * gin + (w * nsb + g) * 4)..(r * gin + (w * nsb + g) * 4 + 4);
                        assert_eq!(&h[src], &hl[dst], "permutation r={r} g={g} w={w}");
                    }
                }
            }
        }
        let served = e
            .moe_gate_up_preclamp8_q8_rows(
                &ptrs_d,
                &scl_d,
                &aq,
                &ad,
                7.0,
                gin,
                n_ff,
                n_used,
                n_pairs,
                QT_NVFP4_V2,
                QT_NVFP4_V2,
                grb2,
                grb2,
            )
            .unwrap();
        let lm = e
            .moe_gate_up_preclamp8_q8_rows_lm(
                &ptrs_d,
                &scl_d,
                &aq_lm,
                &ad,
                7.0,
                gin,
                n_ff,
                n_used,
                n_pairs,
                QT_NVFP4_V2,
                QT_NVFP4_V2,
                grb2,
                grb2,
                false,
            )
            .unwrap();
        let lm4 = e
            .moe_gate_up_preclamp8_q8_rows_lm(
                &ptrs_d,
                &scl_d,
                &aq_lm,
                &ad,
                7.0,
                gin,
                n_ff,
                n_used,
                n_pairs,
                QT_NVFP4_V2,
                QT_NVFP4_V2,
                grb2,
                grb2,
                true,
            )
            .unwrap();
        let (hs, hl, hl4) = (
            e.dtoh(&served).unwrap(),
            e.dtoh(&lm).unwrap(),
            e.dtoh(&lm4).unwrap(),
        );
        assert_eq!(mism(&hs, &hl), 0, "gate/up lane-major differs at t={t}");
        assert_eq!(mism(&hs, &hl4), 0, "gate/up lane-major w4 differs at t={t}");
        assert!(
            hs.iter().any(|v| *v != 0.0),
            "vacuous: served gate/up is all zeros"
        );
        // red: the unpermuted activation must NOT reproduce the served bits
        let red = e
            .moe_gate_up_preclamp8_q8_rows_lm(
                &ptrs_d,
                &scl_d,
                &aq,
                &ad,
                7.0,
                gin,
                n_ff,
                n_used,
                n_pairs,
                QT_NVFP4_V2,
                QT_NVFP4_V2,
                grb2,
                grb2,
                false,
            )
            .unwrap();
        assert!(
            mism(&hs, &e.dtoh(&red).unwrap()) > 0,
            "red arm: the unpermuted activation matched"
        );
        // down over the served activation (the gate/up output, quantized per pair)
        let (aq2, ad2) = e.quantize_q8_1(&served, n_pairs, n_ff).unwrap();
        let aq2_lm = e.q8_to_lane_major(&aq2, n_pairs, n_ff).unwrap();
        let mut ys = e.zeros(t * gin).unwrap();
        let mut yl = e.zeros(t * gin).unwrap();
        let mut yl4 = e.zeros(t * gin).unwrap();
        e.moe_down8_fma_q8_rows(
            &ptrs_d,
            &scl_d,
            &aq2,
            &ad2,
            &mut ys,
            n_ff,
            gin,
            n_used,
            n_pairs,
            QT_NVFP4_V2,
            drb2,
        )
        .unwrap();
        e.moe_down8_fma_q8_rows_lm(
            &ptrs_d,
            &scl_d,
            &aq2_lm,
            &ad2,
            &mut yl,
            n_ff,
            gin,
            n_used,
            n_pairs,
            QT_NVFP4_V2,
            drb2,
            false,
        )
        .unwrap();
        e.moe_down8_fma_q8_rows_lm(
            &ptrs_d,
            &scl_d,
            &aq2_lm,
            &ad2,
            &mut yl4,
            n_ff,
            gin,
            n_used,
            n_pairs,
            QT_NVFP4_V2,
            drb2,
            true,
        )
        .unwrap();
        let (hs, hl, hl4) = (
            e.dtoh(&ys).unwrap(),
            e.dtoh(&yl).unwrap(),
            e.dtoh(&yl4).unwrap(),
        );
        assert_eq!(mism(&hs, &hl), 0, "down lane-major differs at t={t}");
        assert_eq!(mism(&hs, &hl4), 0, "down lane-major w4 differs at t={t}");
        assert!(
            hs.iter().any(|v| *v != 0.0),
            "vacuous: served down is all zeros"
        );
    }
}

#[test]
fn lane_major_refuses_interleaved_experts() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let ptrs = e.htod_u64(&[0u64; 24]).unwrap();
    let scl = e.htod(&[1.0f32; 24]).unwrap();
    let aq = e.htod_i8(&[0i8; 4096]).unwrap();
    let ad = e.htod(&[1.0f32; 128]).unwrap();
    let aq_i8 = e.q8_to_lane_major(&aq, 1, 4096).unwrap();
    let r = e.moe_gate_up_preclamp8_q8_rows_lm(
        &ptrs,
        &scl,
        &aq_i8,
        &ad,
        7.0,
        4096,
        2048,
        8,
        8,
        memra_engine::QT_NVFP4,
        memra_engine::QT_NVFP4,
        2304,
        2304,
        false,
    );
    assert!(r.is_err(), "interleaved experts must be refused by name");
}
