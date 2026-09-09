//! ARITHMETIC ORACLE for the calibrated A4 prefill program.
//!
//! The quality twin says the program changes the model far more than a W4A8 -> W4A4 step on the
//! same weights should, and in the wrong direction (NLL better, not slightly worse). Every
//! harness control is clean, so the next question is whether the kernel implements the program we
//! think it does. This answers it on REAL weights and REAL activations captured from a live prime.
//!
//! ORACLE 1, the quantizer. A CPU reference reproduces the kernel's own recipe -- per-linear
//! global multiplier, per-16 UE4M3 block scale, RN E2M1 with saturation, zero blocks -- and the
//! emitted blocks are compared BITWISE: packed nibbles and block-scale bytes. A mismatch here is
//! the bug and nothing downstream matters.
//!
//! ORACLE 2, the GEMM. Three outputs on identical inputs per projection:
//!   (i)   FP32 reference: dequantized NVFP4 weights, UNQUANTIZED activations
//!   (ii)  the served W4A8 kernel
//!   (iii) the A4 kernel
//! Reported as relative L2 against (i) and as output-norm ratios. A norm ratio off by more than a
//! few percent, or (iii)'s error piled into a few output columns, names the defect: a scale on the
//! wrong side, applied twice, per-row versus per-tensor, or a global scale fitted for the
//! pre-norm activation being applied post-norm.
//!
//! usage: qwen-a4-oracle <artifact.gguf> <prompt.txt> [ctx] [ref_rows]
use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_engine::mmq_ffi::{a4_capture_arm, a4_capture_take};
use memra_engine::model::GpuTensor;
use memra_gguf::GgufFile;

const PROBES: &[&str] = &[
    "blk.2.attn_gate.weight",
    "blk.15.attn_q.weight",
    "blk.2.ffn_up.weight",
    "blk.2.ffn_down.weight",
    "blk.2.attn_qkv.weight",
];

/// e4m3 byte -> f32, the UE4M3 (unsigned, 4-bit exponent, 3-bit mantissa) block scale.
fn ue4m3(b: u8) -> f32 {
    if b == 0 {
        return 0.0;
    }
    let e = (b >> 3) & 0x0F;
    let m = (b & 0x07) as f32;
    if e == 0 {
        (m / 8.0) * 2f32.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e as i32 - 7)
    }
}

fn f32_to_ue4m3(v: f32) -> u8 {
    // round-to-nearest-even over the representable grid, saturating at 448.
    if !(v > 0.0) {
        return 0;
    }
    let v = v.min(448.0);
    let mut best = (0u8, f32::INFINITY);
    for b in 1..=255u8 {
        let c = ue4m3(b);
        if !c.is_finite() || c > 448.0 {
            continue;
        }
        let d = (c - v).abs();
        if d < best.1 {
            best = (b, d);
        }
    }
    best.0
}

/// E2M1 codes: magnitude {0,0.5,1,1.5,2,3,4,6}, sign in the high bit.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn f32_to_e2m1(v: f32) -> u8 {
    let sign = if v.is_sign_negative() { 8u8 } else { 0u8 };
    let a = v.abs();
    let mut code = 0u8;
    let mut best = f32::INFINITY;
    for (i, m) in E2M1.iter().enumerate() {
        let d = (m - a).abs();
        // ties to even code, matching RN
        if d < best || (d == best && i % 2 == 0) {
            best = d;
            code = i as u8;
        }
    }
    sign | code
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        num += ((*x - *y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

fn norm(a: &[f32]) -> f64 {
    a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: qwen-a4-oracle <artifact.gguf> <prompt.txt> [ctx] [rows]");
    let prompt = args.next().expect("a prompt file");
    let ctx: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(2048);
    let ref_rows: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(8);

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let tok = memra_tokenizer::Tokenizer::from_gguf(&g)?;
    let model = HybridModel::load(&e, &g)?;
    let program = model
        .cfg
        .prefill_activation
        .clone()
        .expect("the artifact must declare the activation program");

    let slots: Vec<u32> = PROBES
        .iter()
        .map(|n| {
            program
                .slot(n)
                .unwrap_or_else(|| panic!("{n} is not in the program"))
        })
        .collect();
    let by_slot: std::collections::HashMap<u32, &str> =
        slots.iter().copied().zip(PROBES.iter().copied()).collect();

    let mut ids = tok.encode(&std::fs::read_to_string(&prompt)?, true);
    ids.truncate(ctx);
    a4_capture_arm(slots.clone(), 512);
    let mut cache = memra_engine::pp::new_cache(&e, &model.cfg, ids.len() + 8)?;
    model.prime_cache(&e, &ids, &mut cache, 0)?;
    let captures = a4_capture_take();
    println!(
        "captured {} of {} probed projections from a {ctx}-token prime\n",
        captures.len(),
        PROBES.len()
    );

    let mut quant_failures = 0usize;
    for cap in &captures {
        let name = by_slot[&cap.slot];
        let padded_rows = cap.m.div_ceil(128) * 128;
        let blocks_per_row = cap.in_f / 256;

        // ---------- ORACLE 1: the quantizer, bitwise ----------
        let mut bad_scales = 0usize;
        let mut bad_nibbles = 0usize;
        let mut checked = 0usize;
        for row in 0..cap.m.min(64) {
            for blk in 0..blocks_per_row {
                // block_fp4_mmq = { u32 d4[4]; i8 qs[4*32] } = 16 + 128 = 144 B, and the slab is
                // laid out [in_f/256][padded_rows] (block-major over the K groups, row-minor).
                const BLOCK: usize = 144;
                let off = (blk * padded_rows + row) * BLOCK;
                let d4 = &cap.scratch[off..off + 16];
                let qs = &cap.scratch[off + 16..off + BLOCK];
                for sub in 0..16 {
                    let base = blk * 256 + sub * 16;
                    let vals: Vec<f32> =
                        (0..16).map(|j| cap.x[row * cap.in_f + base + j]).collect();
                    let amax = vals.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let raw = amax / (6.0 * cap.input_scale);
                    let micro = f32_to_ue4m3(raw.min(448.0));
                    if micro != d4[sub] {
                        bad_scales += 1;
                    }
                    let denom = cap.input_scale * ue4m3(micro);
                    for j in 0..16 {
                        let q = if denom > 0.0 { vals[j] / denom } else { 0.0 };
                        let want = f32_to_e2m1(q);
                        // PACKING, from the kernel: four lanes own the 16 values, four each
                        // (L0=0..3, L1=4..7, L2=8..11, L3=12..15). Lane 0 writes u32 2*sub+0 with
                        // its own value in the LOW nibble and lane 2's in the HIGH; lane 1 writes
                        // u32 2*sub+1 the same way against lane 3. So the pair sharing a byte is
                        // (v, v+8) within each half, not (v, v+1).
                        let (u32_idx, byte_idx, high) = match j {
                            0..=3 => (2 * sub, j, false),
                            4..=7 => (2 * sub + 1, j - 4, false),
                            8..=11 => (2 * sub, j - 8, true),
                            _ => (2 * sub + 1, j - 12, true),
                        };
                        let byte = qs[u32_idx * 4 + byte_idx];
                        let got = if high { byte >> 4 } else { byte & 0x0F };
                        if got != want {
                            bad_nibbles += 1;
                        }
                        checked += 1;
                    }
                }
            }
        }
        if bad_scales != 0 || bad_nibbles != 0 {
            quant_failures += 1;
        }
        println!(
            "ORACLE1 {name}: rows checked {}, block scales mismatched {bad_scales}, \
             nibbles mismatched {bad_nibbles} of {checked}",
            cap.m.min(64)
        );

        // ---------- ORACLE 2: the GEMM ----------
        let v = g.find(name).expect("the weight");
        let w =
            memra_gguf::dequant::dequantize(v.ggml_type, g.tensor_data(v), cap.in_f * cap.out_f);
        let rows = ref_rows.min(cap.m);
        let mut reference = vec![0f32; rows * cap.out_f];
        for r in 0..rows {
            for o in 0..cap.out_f {
                let mut acc = 0f64;
                for k in 0..cap.in_f {
                    acc += (cap.x[r * cap.in_f + k] as f64) * (w[o * cap.in_f + k] as f64);
                }
                reference[r * cap.out_f + o] = (acc * cap.weight_scale as f64) as f32;
            }
        }
        // (ii) the served W4A8 kernel on the same activations, same weight bytes, no stamp.
        let plain = GpuTensor::load_opt(&e, &g, name)?.expect("weight");
        let x_dev = e.htod(&cap.x[..rows * cap.in_f])?;
        let served = e.dtoh(&e.matmul(&plain, &x_dev, rows)?)?;
        let a4 = &cap.y[..rows * cap.out_f];

        let (n_ref, n_ii, n_iii) = (norm(&reference), norm(&served), norm(a4));
        println!(
            "ORACLE2 {name}: relL2 served-vs-ref {:.4}%  A4-vs-ref {:.4}%  |  norm ratio \
             served/ref {:.4}  A4/ref {:.4}",
            100.0 * rel_l2(&served, &reference),
            100.0 * rel_l2(a4, &reference),
            n_ii / n_ref,
            n_iii / n_ref
        );
        // where does the A4 error live?
        let mut per_col: Vec<(usize, f64)> = (0..cap.out_f)
            .map(|o| {
                let mut num = 0.0;
                for r in 0..rows {
                    num += ((a4[r * cap.out_f + o] - reference[r * cap.out_f + o]) as f64).powi(2);
                }
                (o, num)
            })
            .collect();
        let total: f64 = per_col.iter().map(|c| c.1).sum();
        per_col.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top16: f64 = per_col.iter().take(16).map(|c| c.1).sum();
        println!(
            "         A4 error concentration: worst 16 of {} output columns carry {:.1}% of it \
             (uniform would be {:.1}%)",
            cap.out_f,
            100.0 * top16 / total.max(1e-30),
            100.0 * 16.0 / cap.out_f as f64
        );
    }
    if quant_failures != 0 {
        eprintln!("A4 ORACLE: quantizer mismatched on {quant_failures} projections");
        std::process::exit(1);
    }
    println!("\nA4 ORACLE: quantizer bitwise-clean on every probed projection");
    Ok(())
}
