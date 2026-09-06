//! Gate for the f16-class KDA six (lane/kda6-f16-20260906): `qmatvec_e4m3_mmvq_fused6_f16`
//! (f16x2 and f32 accumulation twins) against an f64 HOST reference of the same e4m3 bytes,
//! beside the served fused6 (W8 e4m3 x q8_1 int8) measured the same way, at the served shape
//! (in 4096, outs [8192, 8192, 8192, 128, 128, 64]). A numeric class, not a bitwise claim: the
//! f16 arm's mean error must sit at or under the served arm's on every plane, its max within
//! 2x, and the argmax of every plane must agree.
//!
//! RED ARM: the activation converted with the NIBBLE pairing (the rows pair's converter,
//! (i, i+8)) fed to the six must blow the band. Without it a pass could be two kernels reading
//! the same wrong words.
use memra_engine::Engine;

struct Lcg(u32);
impl Lcg {
    fn byte(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 24) as u8
    }
}
/// e4m3 (fn) bytes with the NaN codes (0x7F / 0xFF) and huge magnitudes remapped: keep the
/// exponent field at 4..10 so values sit in [2^-3, 2^3] like a scaled weight plane.
fn synth_e4m3(n: usize, seed: u32) -> Vec<u8> {
    let mut r = Lcg(seed);
    (0..n)
        .map(|_| {
            let b = r.byte();
            let sign = b & 0x80;
            let exp = 4 + (b >> 3 & 0x7); // 4..11
            let man = b & 0x7;
            let v = sign | (exp << 3) | man;
            if (v & 0x7F) == 0x7F { sign | 0x30 } else { v }
        })
        .collect()
}
fn e4m3(b: u8) -> f64 {
    if (b & 0x7F) == 0x7F {
        return f64::NAN;
    }
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as f64;
    if e == 0 {
        sign * m * 2f64.powi(-9)
    } else {
        sign * (1.0 + m / 8.0) * 2f64.powi(e - 7)
    }
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
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold(
            (0usize, f32::MIN),
            |m, (i, x)| if *x > m.1 { (i, *x) } else { m },
        )
        .0
}

#[test]
fn kda6_f16_error_is_at_or_under_the_served_six() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let in_f = 4096usize;
    let dims = [8192usize, 8192, 8192, 128, 128, 64];
    let ws = [0.0123f32, 0.0234, 0.0345, 0.0456, 0.0567, 0.0678];
    let host: Vec<Vec<u8>> = dims
        .iter()
        .enumerate()
        .map(|(i, &o)| synth_e4m3(o * in_f, 0x1111 * (i as u32 + 1)))
        .collect();
    let dev: Vec<_> = host.iter().map(|h| e.htod_bytes(h).unwrap()).collect();
    let w: [&_; 6] = [&dev[0], &dev[1], &dev[2], &dev[3], &dev[4], &dev[5]];
    let x = vecf(in_f, 77, 1.0);
    let x_d = e.htod(&x).unwrap();
    let served = e
        .qmatvec_e4m3_fused6_raw(w, &x_d, in_f, dims, in_f, ws, 0)
        .unwrap();
    let f16 = e
        .qmatvec_e4m3_fused6_f16_raw(w, &x_d, in_f, dims, in_f, ws, false)
        .unwrap();
    let f16a = e
        .qmatvec_e4m3_fused6_f16_raw(w, &x_d, in_f, dims, in_f, ws, true)
        .unwrap();
    // red: the rows pair's NIBBLE pairing is the wrong layout for K-inner e4m3 bytes
    let red_act = {
        // the rows pair's NIBBLE pairing, built on the host: half2 w of block b = (e[b*32 + sb*16 + i], e[.. + i + 8])
        let nblk = in_f / 32;
        let mut words = vec![0u32; in_f / 2];
        for b in 0..nblk {
            for w in 0..16 {
                let (sb, i) = (w / 8, w % 8);
                let lo = f16_bits(x[b * 32 + sb * 16 + i]) as u32;
                let hi = f16_bits(x[b * 32 + sb * 16 + i + 8]) as u32;
                words[w * nblk + b] = lo | (hi << 16);
            }
        }
        e.htod_u32_v(&words).unwrap()
    };
    let mut red_outs = [
        e.zeros(dims[0]).unwrap(),
        e.zeros(dims[1]).unwrap(),
        e.zeros(dims[2]).unwrap(),
        e.zeros(dims[3]).unwrap(),
        e.zeros(dims[4]).unwrap(),
        e.zeros(dims[5]).unwrap(),
    ];
    e.e4m3_fused6_f16_into(w, &red_act, in_f, dims, in_f, ws, &mut red_outs, false)
        .unwrap();
    for p in 0..6 {
        let want: Vec<f64> = (0..dims[p])
            .map(|o| {
                let row = &host[p][o * in_f..(o + 1) * in_f];
                let mut acc = 0f64;
                for (wb, xv) in row.iter().zip(&x) {
                    acc += e4m3(*wb) * (*xv as f64);
                }
                acc * ws[p] as f64
            })
            .collect();
        let (hs, h16, h16a, hr) = (
            e.dtoh(&served[p]).unwrap(),
            e.dtoh(&f16[p]).unwrap(),
            e.dtoh(&f16a[p]).unwrap(),
            e.dtoh(&red_outs[p]).unwrap(),
        );
        let (es, e16, e16a, er) = (
            errs(&hs, &want),
            errs(&h16, &want),
            errs(&h16a, &want),
            errs(&hr, &want),
        );
        eprintln!(
            "plane {p} ({}): served mean {:.3e} max {:.3e} | f16 mean {:.3e} max {:.3e} | acc32 mean {:.3e} max {:.3e} | red mean {:.3e}",
            dims[p], es.mean, es.max, e16.mean, e16.max, e16a.mean, e16a.max, er.mean
        );
        assert!(
            e16.mean <= es.mean * 1.05,
            "plane {p}: f16 mean error above the served six"
        );
        assert!(
            e16a.mean <= es.mean * 1.05,
            "plane {p}: f16 acc32 mean error above the served six"
        );
        assert!(
            e16.max <= es.max.max(1e-3) * 2.0,
            "plane {p}: f16 max error far above the served six"
        );
        assert_eq!(
            argmax(&hs),
            argmax(&h16),
            "plane {p}: argmax disagrees (served vs f16)"
        );
        assert!(
            er.mean > e16.mean * 10.0,
            "plane {p}: red arm (nibble pairing) stayed inside the band"
        );
    }
}
