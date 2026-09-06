//! Gate for the f16-class rows pair (lane/moe-rows-f16-20260906): the hardware-e2m1 f16 kernels
//! against an f64 HOST reference of the same slot-major bytes, beside the served W4A8 pair
//! measured the same way. A numeric class, not a bitwise claim: the gate asserts the f16 arm's
//! error is at or under the served arm's (mean and max, over every output of both stages at the
//! served shape 4096 -> 2048 -> 4096, top-8, t=1 and t=2), and that the two arms agree on the
//! argmax of every down row. Both accumulation variants (f16x2 and f32) are gated.
//!
//! RED ARM: the activation fed UNPERMUTED (f32 rows converted without the lane-major pairing)
//! must blow the error band. Without it a pass could be two kernels reading the same wrong words.
//!
//! Runs on sm_100a / sm_120a builds; a portable build's launcher refuses by name (checked).
use cudarc::driver::DevicePtr;
use memra_engine::{Engine, QT_NVFP4_V2};

struct Lcg(u32);
impl Lcg {
    fn byte(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 24) as u8
    }
}
/// Interleaved (V1) rows: 36 B per 64 elements = 4 ue4m3 scale bytes + 32 nibble bytes.
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
fn vecf(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * amp
        })
        .collect()
}
/// ue4m3 -> f64, the TRUE value (2^(e-7) * (1 + m/8); subnormal m * 2^-9; 0 and 0x7F -> 0).
fn ue4m3(x: u8) -> f64 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let e = ((x >> 3) & 0xF) as i32;
    let m = (x & 7) as f64;
    if e == 0 {
        m * 2f64.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    }
}
const E2M1: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
fn e2m1(n: u8) -> f64 {
    let v = E2M1[(n & 7) as usize];
    if n & 8 != 0 { -v } else { v }
}
/// Decode one slot-major V2 row (16 B per 32-group + 2 B scales at nsb*16 + 2g) to f64, in the
/// slab's nibble order: within each 16-element scale group packed as 8 bytes, byte i carries
/// element i (low nibble) and element i+8 (high nibble).
fn decode_v2_row(row: &[u8], in_f: usize) -> Vec<f64> {
    let nsb = in_f / 32;
    let mut out = vec![0f64; in_f];
    for g in 0..nsb {
        let q = &row[g * 16..g * 16 + 16];
        let sc = &row[nsb * 16 + g * 2..nsb * 16 + g * 2 + 2];
        for sb in 0..2 {
            let scale = ue4m3(sc[sb]);
            for i in 0..8 {
                let b = q[sb * 8 + i];
                out[g * 32 + sb * 16 + i] = e2m1(b & 0xF) * scale;
                out[g * 32 + sb * 16 + i + 8] = e2m1(b >> 4) * scale;
            }
        }
    }
    out
}
/// f32 -> IEEE half bits, round to nearest even (inputs here are small normals).
fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = b & 0x7F_FFFF;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    let mut h = sign | ((exp as u16) << 10) | ((mant >> 13) as u16);
    let rem = mant & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    h
}
fn dot(a: &[f64], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * (*y as f64)).sum()
}
struct Err {
    mean: f64,
    max: f64,
}
fn errs(got: &[f32], want: &[f64]) -> Err {
    let scale = want.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-9);
    let mut sum = 0.0;
    let mut max = 0f64;
    for (g, w) in got.iter().zip(want) {
        let e = ((*g as f64) - w).abs() / scale;
        sum += e;
        max = max.max(e);
    }
    Err {
        mean: sum / got.len() as f64,
        max,
    }
}

#[test]
fn f16_rows_error_is_at_or_under_the_served_pair() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    unsafe { std::env::set_var("MEMRA_MOE_VROWS_ILP", "1") };
    let (gin, n_ff, n_used) = (4096usize, 2048usize, 8usize);
    let stream = e.stream();
    for &t in &[1usize, 2] {
        let n_pairs = t * n_used;
        let grows = synth_rows(n_ff, gin, 0x8181);
        let drows = synth_rows(gin, n_ff, 0x4242);
        let mut ptrs = vec![0u64; 3 * n_pairs];
        let mut keep = Vec::new();
        let mut host_rows: Vec<Vec<u8>> = Vec::new(); // repacked V2 bytes, per (plane, pr)
        for plane in 0..3 {
            for pr in 0..n_pairs {
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
                host_rows.push(e.dtoh_u8(&buf).unwrap());
                ptrs[plane * n_pairs + pr] = buf.device_ptr(&stream).0;
                keep.push(buf);
            }
        }
        let ptrs_d = e.htod_u64(&ptrs).unwrap();
        let scl: Vec<f32> = (0..3 * n_pairs).map(|pr| 0.5 + 0.01 * pr as f32).collect();
        let scl_d = e.htod(&scl).unwrap();
        let grb2 = (gin / 32) * 18;
        let drb2 = (n_ff / 32) * 18;
        let limit = 7.0f32;
        let x = vecf(t * gin, 91, 1.0);
        let x_d = e.htod(&x).unwrap();
        // served W4A8
        let (aq, ad) = e.quantize_q8_1(&x_d, t, gin).unwrap();
        let served = e
            .moe_gate_up_preclamp8_q8_rows(
                &ptrs_d,
                &scl_d,
                &aq,
                &ad,
                limit,
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
        // f16 class, both accumulations
        let acth = e.f32_to_f16_lane_major(&x_d, t, gin).unwrap();
        let f16 = e
            .moe_gate_up_preclamp8_f16_rows(
                &ptrs_d,
                &scl_d,
                &acth,
                limit,
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
        let f16a = e
            .moe_gate_up_preclamp8_f16_rows(
                &ptrs_d,
                &scl_d,
                &acth,
                limit,
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
        // f64 reference of the gate/up stage (same epilogue)
        let mut want = vec![0f64; n_pairs * n_ff];
        for pr in 0..n_pairs {
            let tok = pr / n_used;
            let xr = &x[tok * gin..(tok + 1) * gin];
            for o in 0..n_ff {
                let g = decode_v2_row(&host_rows[pr][o * grb2..(o + 1) * grb2], gin);
                let u = decode_v2_row(&host_rows[n_pairs + pr][o * grb2..(o + 1) * grb2], gin);
                let accg = dot(&g, xr) * scl[pr] as f64;
                let accu = dot(&u, xr) * scl[n_pairs + pr] as f64;
                let uu = accu.min(limit as f64).max(-(limit as f64));
                let xx = accg.min(limit as f64);
                want[pr * n_ff + o] = (xx / (1.0 + (-xx).exp())) * uu;
            }
        }
        let (hs, h16, h16a) = (
            e.dtoh(&served).unwrap(),
            e.dtoh(&f16).unwrap(),
            e.dtoh(&f16a).unwrap(),
        );
        let (es, e16, e16a) = (errs(&hs, &want), errs(&h16, &want), errs(&h16a, &want));
        eprintln!(
            "gate/up t={t}: served mean {:.3e} max {:.3e} | f16 mean {:.3e} max {:.3e} | f16 acc32 mean {:.3e} max {:.3e}",
            es.mean, es.max, e16.mean, e16.max, e16a.mean, e16a.max
        );
        assert!(
            e16.mean <= es.mean * 1.05,
            "f16 gate/up mean error above the served arm at t={t}"
        );
        assert!(
            e16a.mean <= es.mean * 1.05,
            "f16 acc32 gate/up mean error above the served arm at t={t}"
        );
        assert!(
            e16.max <= es.max.max(1e-3) * 2.0,
            "f16 gate/up max error far above the served arm at t={t}"
        );
        // down stage: both arms consume the f64 reference act (so the stage is judged alone)
        let act_ref: Vec<f32> = want.iter().map(|v| *v as f32).collect();
        let act_d = e.htod(&act_ref).unwrap();
        let (aq2, ad2) = e.quantize_q8_1(&act_d, n_pairs, n_ff).unwrap();
        let acth2 = e.f32_to_f16_lane_major(&act_d, n_pairs, n_ff).unwrap();
        let mut ys = e.zeros(t * gin).unwrap();
        let mut y16 = e.zeros(t * gin).unwrap();
        let mut y16a = e.zeros(t * gin).unwrap();
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
        e.moe_down8_fma_f16_rows(
            &ptrs_d,
            &scl_d,
            &acth2,
            &mut y16,
            n_ff,
            gin,
            n_used,
            n_pairs,
            QT_NVFP4_V2,
            drb2,
            false,
        )
        .unwrap();
        e.moe_down8_fma_f16_rows(
            &ptrs_d,
            &scl_d,
            &acth2,
            &mut y16a,
            n_ff,
            gin,
            n_used,
            n_pairs,
            QT_NVFP4_V2,
            drb2,
            true,
        )
        .unwrap();
        let mut wantd = vec![0f64; t * gin];
        for tok in 0..t {
            for o in 0..gin {
                let mut chain = 0f64;
                for j in 0..n_used {
                    let pr = tok * n_used + j;
                    let w =
                        decode_v2_row(&host_rows[2 * n_pairs + pr][o * drb2..(o + 1) * drb2], n_ff);
                    chain += scl[2 * n_pairs + pr] as f64
                        * dot(&w, &act_ref[pr * n_ff..(pr + 1) * n_ff]);
                }
                wantd[tok * gin + o] = chain;
            }
        }
        let (hs, h16, h16a) = (
            e.dtoh(&ys).unwrap(),
            e.dtoh(&y16).unwrap(),
            e.dtoh(&y16a).unwrap(),
        );
        let (es, e16, e16a) = (errs(&hs, &wantd), errs(&h16, &wantd), errs(&h16a, &wantd));
        eprintln!(
            "down t={t}: served mean {:.3e} max {:.3e} | f16 mean {:.3e} max {:.3e} | f16 acc32 mean {:.3e} max {:.3e}",
            es.mean, es.max, e16.mean, e16.max, e16a.mean, e16a.max
        );
        assert!(
            e16.mean <= es.mean * 1.05,
            "f16 down mean error above the served arm at t={t}"
        );
        assert!(
            e16a.mean <= es.mean * 1.05,
            "f16 acc32 down mean error above the served arm at t={t}"
        );
        for tok in 0..t {
            let am = |v: &[f32]| {
                v[tok * gin..(tok + 1) * gin]
                    .iter()
                    .enumerate()
                    .fold(
                        (0usize, f32::MIN),
                        |m, (i, x)| if *x > m.1 { (i, *x) } else { m },
                    )
                    .0
            };
            assert_eq!(
                am(&hs),
                am(&h16),
                "down argmax disagrees at t={t} tok={tok}"
            );
        }
        // red arm: the UNPERMUTED activation (group-major pairs) must blow the band
        let acth_red = {
            let mut words = vec![0u32; t * gin / 2];
            for (k, w) in words.iter_mut().enumerate() {
                *w = f16_bits(x[2 * k]) as u32 | ((f16_bits(x[2 * k + 1]) as u32) << 16);
            }
            e.htod_u32_v(&words).unwrap()
        };
        let red = e
            .moe_gate_up_preclamp8_f16_rows(
                &ptrs_d,
                &scl_d,
                &acth_red,
                limit,
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
        let er = errs(&e.dtoh(&red).unwrap(), &want);
        assert!(
            er.mean > e16.mean * 10.0,
            "red arm: the unpermuted activation stayed inside the band (mean {:.3e} vs {:.3e})",
            er.mean,
            e16.mean
        );
    }
}

#[test]
fn f16_rows_refuse_interleaved_experts() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let ptrs = e.htod_u64(&[0u64; 24]).unwrap();
    let scl = e.htod(&[1.0f32; 24]).unwrap();
    let acth = e.htod_u32_v(&[0u32; 2048]).unwrap();
    let r = e.moe_gate_up_preclamp8_f16_rows(
        &ptrs,
        &scl,
        &acth,
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
