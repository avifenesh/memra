//! SHORT-PREFILL gate for the calibrated A4 GEMM: the one-row tail is the shape that broke.
//!
//! The activation program is required to run on every prefill tail and restored suffix, including
//! one row. The first implementation sized the quantizer's scratch with
//! `memra_mmq_nvfp4_w4a8_act_bytes`, which describes a different block layout over an UNPADDED
//! row count. Per real row the fp4 layout is smaller, so long primes fit and every exactness gate
//! passed; at rows=1, in_f=17408 the quantizer wrote 2,368,000 B into a 38,016 B allocation. The
//! corruption surfaced far away, as NaN in a drafter logits row after hundreds of healthy
//! speculative rounds, and as `head-out NaN 5120/5120` on the MTP route.
//!
//! Two arms, because a size assertion alone would only restate the arithmetic:
//!
//!   CANARY  the scratch is over-allocated and poisoned; bytes past the declared footprint must
//!           be untouched after the GEMM. Reverting the sizing fix turns this red at every row
//!           count below 128.
//!   TAIL    the first `rows` rows of a 128-row GEMM must equal a GEMM over just those rows. Row
//!           quantization is row-independent, so this is bitwise equality, not a tolerance.
//!
//! usage: qwen-a4-tail-gate <calibrated.gguf>
use memra_engine::Engine;
use memra_engine::model::GpuTensor;
use memra_gguf::GgufFile;
use memra_gguf::model_packs::qwen35::activation::PrefillFp4;

const ROWS: &[usize] = &[1, 2, 3, 17, 33, 64, 127, 128, 129, 255, 1024];
const CANARY: u8 = 0xA4;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: qwen-a4-tail-gate <calibrated.gguf>");
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let program = PrefillFp4::from_gguf(&g)?.expect("the artifact must declare the program");

    let mut failures = 0usize;
    // Widest and narrowest stamped in_f in the program, plus a GDN projection.
    for name in [
        "blk.0.ffn_down.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.attn_qkv.weight",
    ] {
        let multiplier = *program.scales().get(name).expect("a program weight");
        let slot = program.slot(name).expect("a program slot");
        let mut weight = GpuTensor::load_opt(&e, &g, name)?.expect("the weight");
        weight.stamp_prefill_a4(multiplier, slot)?;
        let (in_f, out_f) = (weight.in_features(), weight.out_features());
        let (bytes, scale, rp) = match &weight {
            GpuTensor::Quant {
                bytes, scale, rp, ..
            } => (bytes, *scale, *rp),
            _ => panic!("{name} is not a quantized weight"),
        };

        // One 1024-row input; every arm reads a prefix of it, so rows compare directly.
        let host: Vec<f32> = (0..1024 * in_f)
            .map(|i| {
                let x = (i as f32 * 0.000_137).sin() * 3.0;
                if i % 977 == 0 { x * 40.0 } else { x }
            })
            .collect();
        let x = e.htod(&host)?;
        let reference = e.dtoh(&e.matmul_prefill(&weight, &x, 128)?)?;

        for &rows in ROWS {
            let declared = unsafe {
                memra_engine::mmq_ffi::memra_mmq_nvfp4_calibrated_prefill_act_bytes(
                    in_f as i32,
                    rows as i32,
                )
            };
            assert!(declared > 0, "{name}: no declared footprint at rows={rows}");
            let mut scratch = e.htod_bytes(&vec![CANARY; declared * 2])?;
            let y = e.qmatvec_mmq_nvfp4_calibrated_prefill_into(
                bytes,
                &x,
                rows,
                in_f,
                out_f,
                scale,
                multiplier,
                slot,
                rp,
                &mut scratch,
            )?;
            let y_host = e.dtoh(&y)?;
            let after = e.dtoh_u8(&scratch)?;
            let dirty = after[declared..].iter().filter(|b| **b != CANARY).count();
            if dirty != 0 {
                println!(
                    "FAIL canary {name} rows={rows}: {dirty} of {} bytes past the declared \
                     {declared}-byte footprint were overwritten",
                    declared
                );
                failures += 1;
            }

            let compare = rows.min(128);
            let bad = (0..compare * out_f)
                .filter(|i| y_host[*i].to_bits() != reference[*i].to_bits())
                .count();
            if bad != 0 {
                println!(
                    "FAIL tail {name} rows={rows}: {bad} of {} output values differ from the \
                     128-row GEMM over the same input rows",
                    compare * out_f
                );
                failures += 1;
            }
            if !y_host.iter().take(rows * out_f).all(|v| v.is_finite()) {
                println!("FAIL finite {name} rows={rows}: output carries non-finite values");
                failures += 1;
            }
            println!(
                "  {name} rows={rows}: declared {declared} B, {dirty} bytes past it, \
                 {bad} mismatched values"
            );
        }
    }
    if failures != 0 {
        eprintln!("A4 TAIL GATE: {failures} FAILURES");
        std::process::exit(1);
    }
    println!("A4 TAIL GATE: PASS");
    Ok(())
}
