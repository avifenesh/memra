//! Exact synthetic gate for the production NVFP4 W4A4 MMQ path.
//!
//! The activation rows are constructed so every 16-value sub-block has amax 2688 and every value
//! is exactly E2M1-representable under a UE4M3 scale of 448. The two-level activation quantizer
//! therefore emits row scale 1, exact E2M1 codes, and the same values the host reference consumes.
//! Weight scales are powers of two and all products/sums stay below 2^24, so reduction order cannot
//! change a bit. This turns the SM100 tcgen05 path into a zero-tolerance indexing/scale/tail gate.

use memra_engine::Engine;

const BLOCK_VALUES: [f32; 15] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];
const BLOCK_CODES: [u8; 15] = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15];
const ACT_VALUES: [f32; 15] = [
    0.0, 224.0, 448.0, 672.0, 896.0, 1344.0, 1792.0, 2688.0, -224.0, -448.0, -672.0, -896.0,
    -1344.0, -1792.0, -2688.0,
];
const SCALE_CODES: [u8; 4] = [0x30, 0x38, 0x40, 0x48]; // 0.5, 1, 2, 4

fn ue4m3_to_f32(code: u8) -> f32 {
    let exp = ((code >> 3) & 0x0f) as i32;
    let mantissa = (code & 7) as f32;
    if exp == 0 {
        (mantissa / 8.0) * 2f32.powi(-6)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exp - 7)
    }
}

fn make_weights(in_f: usize, out_f: usize) -> Vec<u8> {
    assert!(in_f.is_multiple_of(64));
    let blocks_per_row = in_f / 64;
    let mut raw = vec![0u8; out_f * blocks_per_row * 36];
    for out in 0..out_f {
        for block in 0..blocks_per_row {
            let base = (out * blocks_per_row + block) * 36;
            for sub in 0..4 {
                raw[base + sub] = SCALE_CODES[(out + block + sub) % SCALE_CODES.len()];
                for lane in 0..8 {
                    let lo = BLOCK_CODES[(out * 11 + block * 7 + sub * 3 + lane) % 15];
                    let hi = BLOCK_CODES[(out * 5 + block * 13 + sub * 7 + lane + 4) % 15];
                    raw[base + 4 + sub * 8 + lane] = lo | (hi << 4);
                }
            }
        }
    }
    raw
}

fn make_activations(in_f: usize, tokens: usize) -> Vec<f32> {
    let mut x = vec![0f32; in_f * tokens];
    for token in 0..tokens {
        for sub in 0..in_f.div_ceil(16) {
            let begin = sub * 16;
            let end = (begin + 16).min(in_f);
            for k in begin..end {
                x[token * in_f + k] = ACT_VALUES[(token * 7 + sub * 11 + k - begin) % 15];
            }
            // Pin amax=2688 in every complete or tail sub-block. The quantizer's row scale is then
            // exactly 1 and its UE4M3 micro-scale exactly 448.
            x[token * in_f + begin] = if (token + sub) & 1 == 0 {
                2688.0
            } else {
                -2688.0
            };
        }
    }
    x
}

fn weight_value(raw: &[u8], in_f: usize, out: usize, k: usize) -> f32 {
    let blocks_per_row = in_f / 64;
    let block = k / 64;
    let within = k % 64;
    let sub = within / 16;
    let lane = within % 16;
    let base = (out * blocks_per_row + block) * 36;
    let packed = raw[base + 4 + sub * 8 + (lane & 7)];
    let code = if lane < 8 { packed & 0x0f } else { packed >> 4 };
    let value = BLOCK_VALUES[BLOCK_CODES.iter().position(|item| *item == code).unwrap()];
    value * ue4m3_to_f32(raw[base + sub])
}

fn host_reference(raw: &[u8], x: &[f32], in_f: usize, out_f: usize, tokens: usize) -> Vec<f32> {
    let mut y = vec![0f32; tokens * out_f];
    for token in 0..tokens {
        for out in 0..out_f {
            let mut sum = 0f32;
            for k in 0..in_f {
                sum += x[token * in_f + k] * weight_value(raw, in_f, out, k);
            }
            assert!(sum.abs() < (1u32 << 24) as f32);
            y[token * out_f + out] = sum;
        }
    }
    y
}

fn run_shape(
    engine: &Engine,
    in_f: usize,
    out_f: usize,
    tokens: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let raw = make_weights(in_f, out_f);
    let x = make_activations(in_f, tokens);
    let want = host_reference(&raw, &x, in_f, out_f, tokens);
    let weights = engine.htod_bytes(&raw)?;
    let acts = engine.htod(&x)?;
    let got = engine.dtoh(&engine.qmatvec_mmq_nvfp4_raw(&weights, &acts, tokens, in_f, out_f)?)?;
    let mismatches = got
        .iter()
        .zip(want.iter())
        .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
        .count();
    println!(
        "NVFP4-MMQ-EXACT in={in_f} out={out_f} tokens={tokens}: mismatches={mismatches}/{} {}",
        got.len(),
        if mismatches == 0 { "PASS" } else { "FAIL" }
    );
    if mismatches != 0 {
        return Err(format!("NVFP4 exact gate failed with {mismatches} mismatches").into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::new(0)?;
    for (in_f, out_f, tokens) in [
        (64usize, 128usize, 128usize),
        (128, 136, 40),
        (192, 257, 129),
    ] {
        run_shape(&engine, in_f, out_f, tokens)?;
    }
    println!("NVFP4-MMQ-EXACT ALL PASS");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_fixture_covers_codes_scales_and_exact_activation_blocks() {
        let (in_f, out_f, tokens) = (192usize, 17usize, 3usize);
        let raw = make_weights(in_f, out_f);
        let x = make_activations(in_f, tokens);

        let mut seen = [false; 16];
        for out in 0..out_f {
            for k in 0..in_f {
                let block = k / 64;
                let within = k % 64;
                let sub = within / 16;
                let lane = within % 16;
                let base = (out * (in_f / 64) + block) * 36;
                let packed = raw[base + 4 + sub * 8 + (lane & 7)];
                let code = if lane < 8 { packed & 0x0f } else { packed >> 4 };
                seen[code as usize] = true;
                assert!(SCALE_CODES.contains(&raw[base + sub]));
            }
        }
        for code in BLOCK_CODES {
            assert!(seen[code as usize], "missing E2M1 code {code}");
        }
        assert!(
            !seen[8],
            "negative zero must stay out of the semantic coverage set"
        );

        for token in 0..tokens {
            for sub in 0..(in_f / 16) {
                let values = &x[token * in_f + sub * 16..token * in_f + sub * 16 + 16];
                assert_eq!(values.iter().map(|v| v.abs()).fold(0.0, f32::max), 2688.0);
                assert!(values.iter().all(|v| ACT_VALUES.contains(v)));
            }
        }

        let reference = host_reference(&raw, &x, in_f, out_f, tokens);
        assert!(reference.iter().all(|v| v.is_finite()));
        assert!(reference.iter().all(|v| v.abs() < (1u32 << 24) as f32));
    }
}
