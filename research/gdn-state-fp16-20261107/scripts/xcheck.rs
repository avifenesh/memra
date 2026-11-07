/// fp16's smallest normal, 2^-14.
const F16_MIN_NORMAL: f32 = 6.103_515_6e-5;
/// fp16 subnormals are the multiples of 2^-24.
const F16_SUB_SCALE: f32 = 16_777_216.0;
pub fn round_f16(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    if x.abs() < F16_MIN_NORMAL {
        return (x * F16_SUB_SCALE).round_ties_even() / F16_SUB_SCALE;
    }
    let b = x.to_bits();
    let r = f32::from_bits((b + 0x0fff + ((b >> 13) & 1)) & !0x1fff);
    if r.abs() >= 65_536.0 {
        f32::INFINITY.copysign(x)
    } else {
        r
    }
}

/// fp32 to IEEE binary16 bits, round to nearest even, as the device's `__float2half_rn`: values
/// at or above 65520 in magnitude become infinity, the range below 2^-14 is subnormal (multiples
/// of 2^-24), magnitudes at or below 2^-25 become zero, and every NaN becomes 0x7fff.
pub fn f32_to_fp16_bits(x: f32) -> u32 {
    let b = x.to_bits();
    let sign = (b >> 16) & 0x8000;
    let u = b & 0x7fff_ffff;
    if u >= 0x7f80_0000 {
        return if u == 0x7f80_0000 { sign | 0x7c00 } else { 0x7fff };
    }
    let (r, rem) = if u > 0x477f_efff {
        (sign | 0x7bff, 0x8000_0000u32)
    } else if u >= 0x3880_0000 {
        (sign | ((u - 0x3800_0000) >> 13), u << 19)
    } else if u < 0x3300_0001 {
        (sign, u)
    } else {
        let shift = 0x7e - (u >> 23);
        let m = (u & 0x7f_ffff) | 0x80_0000;
        (sign | (m >> shift), m << (32 - shift))
    };
    if rem > 0x8000_0000 || (rem == 0x8000_0000 && r & 1 != 0) {
        r + 1
    } else {
        r
    }
}

/// IEEE binary16 bits widened to fp32 (exact).
pub fn fp16_bits_to_f32(h: u32) -> f32 {
    let sign = (h & 0x8000) << 16;
    let e = (h >> 10) & 0x1f;
    let m = h & 0x3ff;
    let v = match e {
        0 => m as f32 / 16_777_216.0,
        31 if m == 0 => f32::INFINITY,
        31 => return f32::NAN,
        _ => f32::from_bits(((e + 112) << 23) | (m << 13)),
    };
    f32::from_bits(v.to_bits() | sign)
}


fn main() {
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let mut n = 0u64;
    for _ in 0..50_000_000u64 {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        let x = f32::from_bits(s as u32);
        if !x.is_finite() { continue; }
        let a = round_f16(x);
        let b = fp16_bits_to_f32(f32_to_fp16_bits(x));
        assert!(a.to_bits() == b.to_bits() || (a == 0.0 && b == 0.0 && a.is_sign_negative() == b.is_sign_negative()), "{x:e}: {a:e} vs {b:e}");
        n += 1;
    }
    // every finite fp16 round trips
    for h in (0u32..0x7c00).chain(0x8000..0xfc00) { assert_eq!(f32_to_fp16_bits(fp16_bits_to_f32(h)), h); }
    println!("agree on {n} random finite f32 values");
}
