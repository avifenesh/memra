//! GPU component gate for the official MiMo MXFP4 expert matrices.
//! Compares Memra's as-stored W4/FP8 activation GEMM with a CPU dot product
//! decoded from the GPU's actual activation codes and scales.
//! This does not exercise routing, SwiGLU, a model forward, or serving.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;

use cudarc::driver::{DevicePtr, DevicePtrMut};
use memra_engine::Engine;
use memra_engine::dsv4_ffi::{memra_dsv4_act_quant_fp8, memra_dsv4_fp4_gemm};
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::dsv4::{E2M1, e8m0_to_f32};
use memra_gguf::model_packs::mimo_v2::inspect_pinned_source_headers;
use memra_gguf::model_plan::{MlpPlan, ModelPlan};
use memra_gguf::nvfp4_repack::fp8_e4m3_to_f32;
use memra_gguf::safetensors::StModel;
use sha2::{Digest, Sha256};

type Fail = Box<dyn std::error::Error>;

const ROWS: usize = 2;
const ABS_FLOOR: f64 = 1e-5;
const L1_FACTOR: f64 = 1e-5;
// This is a declared source revision. Header checks cannot verify payload bytes.
const SOURCE_REVISION: &str = "3b38d063180c3e4aed9691fdc735f3d10b266ee4";
const CONFIG_SHA256: &str = "61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621";

fn input(rows: usize, k: usize) -> Vec<f32> {
    (0..rows * k)
        .map(|index| {
            let row = index / k;
            let col = index % k;
            let value = ((col * 37 + row * 71) % 251) as i32 - 125;
            let denominator = if row == 0 { 64.0 } else { 256.0 };
            if col.is_multiple_of(19) {
                0.0
            } else {
                value as f32 / denominator
            }
        })
        .collect()
}

fn validate_activation(codes: &[u8], scales: &[f32], rows: usize, k: usize) -> Result<(), Fail> {
    if k == 0
        || !k.is_multiple_of(128)
        || codes.len() != rows * k
        || scales.len() != rows * (k / 128)
    {
        return Err("activation code or per-128 scale grid shape changed".into());
    }
    for (index, &scale) in scales.iter().enumerate() {
        let bits = scale.to_bits();
        let exponent = (bits >> 23) & 0xff;
        if bits >> 31 != 0 || exponent == 0 || exponent == 0xff || bits & 0x7f_ffff != 0 {
            return Err(
                format!("activation scale {index} is not a finite normal power of two").into(),
            );
        }
    }
    if codes.iter().any(|code| code & 0x7f == 0x7f) {
        return Err("activation grid contains an FP8 NaN code".into());
    }
    Ok(())
}

/// f64 accumulation of the exact decoded W4 and GPU-produced FP8 grids.
/// Returns the dot and sum of absolute products for an accumulation-aware bound.
fn cpu_dot(
    codes: &[u8],
    scales: &[f32],
    weight: &[u8],
    weight_scales: &[u8],
    row: usize,
    col: usize,
    k: usize,
) -> (f64, f64) {
    let a = &codes[row * k..(row + 1) * k];
    let a_sc = &scales[row * (k / 128)..(row + 1) * (k / 128)];
    let w = &weight[col * (k / 2)..(col + 1) * (k / 2)];
    let w_sc = &weight_scales[col * (k / 32)..(col + 1) * (k / 32)];
    let mut dot = 0.0f64;
    let mut l1 = 0.0f64;
    for group in 0..k / 32 {
        let weight_scale = e8m0_to_f32(w_sc[group]) as f64;
        let activation_scale = a_sc[group / 4] as f64;
        for index in group * 32..(group + 1) * 32 {
            let byte = w[index / 2];
            let nibble = if index % 2 == 0 { byte & 15 } else { byte >> 4 };
            let product = fp8_e4m3_to_f32(a[index]) as f64
                * E2M1[nibble as usize] as f64
                * weight_scale
                * activation_scale;
            dot += product;
            l1 += product.abs();
        }
    }
    (dot, l1)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_mxfp4_gemm_gate: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Fail> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: mimo_mxfp4_gemm_gate <source_dir> <gpu_index> <report.tsv>".into());
    }
    let dir = Path::new(&args[0]);
    let gpu_index: usize = args[1].parse()?;
    let report = Path::new(&args[2]);
    if report.exists() {
        return Err(format!("refusing to overwrite {}", report.display()).into());
    }

    let config_text = std::fs::read_to_string(dir.join("config.json"))?;
    let config_hash = format!("{:x}", Sha256::digest(config_text.as_bytes()));
    if config_hash != CONFIG_SHA256 {
        return Err(format!("MiMo source config SHA256 changed: {config_hash}").into());
    }
    let config = ModelConfig::from_hf(&HfConfig::try_parse(&config_text)?);
    let plan = ModelPlan::compile(&config)?;
    if plan.layers.len() != 48 {
        return Err("pinned MiMo config has changed layer count".into());
    }
    let MlpPlan::Moe(moe) = &plan.layers[1].mlp else {
        return Err("pinned MiMo layer 1 is not MoE".into());
    };
    if config.n_embd != 4_096 || moe.expert_count != 256 || moe.expert_intermediate_size != 2_048 {
        return Err("pinned MiMo config has changed expert geometry".into());
    }
    let model = StModel::open(dir)?;
    let headers = model
        .names()
        .map(|name| (name.clone(), model.info(name).unwrap().clone()))
        .collect::<BTreeMap<_, _>>();
    let bound = inspect_pinned_source_headers(&config, &headers)?;
    eprintln!(
        "pinned source headers bound {} semantic tensors",
        bound.tensors.len()
    );
    drop(bound);
    drop(headers);

    // The full source census and local config are admitted before any GPU allocation.
    let engine = Engine::new(gpu_index)?;
    engine.gpu.ctx.bind_to_thread()?;
    let stream = engine.stream();
    let native_stream = stream.cu_stream() as *mut c_void;
    let mut text = String::from("format\tmemra-mimo-mxfp4-gemm-v1\n");
    writeln!(text, "source\tXiaomiMiMo/MiMo-V2.6-Flash-RL")?;
    writeln!(text, "declared_source_revision\t{SOURCE_REVISION}")?;
    writeln!(text, "payload_verification\texternal_hf_verify_required")?;
    writeln!(
        text,
        "numeric_class\tmemra_pow2_e4m3_activation_mxfp4_weight"
    )?;
    writeln!(text, "config_sha256\t{CONFIG_SHA256}")?;
    writeln!(text, "gpu_index\t{gpu_index}")?;
    writeln!(text, "input_rows\t{ROWS}")?;
    writeln!(
        text,
        "error_bound\t{ABS_FLOOR:.9e}+{L1_FACTOR:.9e}*sum_abs_products"
    )?;
    let mut total_violations = 0usize;

    for expert in [0u32, 255u32] {
        for projection in ["gate", "up", "down"] {
            let (n, k) = if projection == "down" {
                (4_096usize, 2_048usize)
            } else {
                (2_048usize, 4_096usize)
            };
            let stem = format!("model.layers.1.mlp.experts.{expert}.{projection}_proj");
            let (weight_info, weight) = model
                .raw(&format!("{stem}.weight"))
                .ok_or_else(|| format!("{stem}.weight missing"))?;
            let (scale_info, weight_scales) = model
                .raw(&format!("{stem}.weight_scale"))
                .ok_or_else(|| format!("{stem}.weight_scale missing"))?;
            if weight_info.dtype != "U8"
                || weight_info.shape != [n as u64, (k / 2) as u64]
                || weight.len() != n * k / 2
                || scale_info.dtype != "U8"
                || scale_info.shape != [n as u64, (k / 32) as u64]
                || weight_scales.len() != n * k / 32
                || weight_scales.contains(&0xff)
            {
                return Err(
                    format!("{stem}: changed MXFP4 shape, scale grid, or scale code").into(),
                );
            }
            if weight_scales
                .iter()
                .any(|&code| !e8m0_to_f32(code).is_finite())
            {
                return Err(format!("{stem}: non-finite decoded weight scale").into());
            }

            let values = input(ROWS, k);
            if values.iter().any(|value| !value.is_finite()) {
                return Err("non-finite deterministic input".into());
            }
            let input_gpu = engine.htod(&values)?;
            let mut codes_gpu = stream.alloc_zeros::<u8>(ROWS * k)?;
            let mut scales_gpu = stream.alloc_zeros::<f32>(ROWS * (k / 128))?;
            let mut output_gpu = stream.alloc_zeros::<f32>(ROWS * n)?;
            let weight_gpu = engine.htod_bytes(weight)?;
            let weight_scales_gpu = engine.htod_bytes(weight_scales)?;
            let rc = unsafe {
                memra_dsv4_act_quant_fp8(
                    input_gpu.device_ptr(&stream).0 as *const f32,
                    codes_gpu.device_ptr_mut(&stream).0 as *mut c_void,
                    scales_gpu.device_ptr_mut(&stream).0 as *mut f32,
                    ROWS as i32,
                    k as i32,
                    native_stream,
                )
            };
            if rc != 0 {
                return Err(format!("{stem}: act_quant_fp8 returned {rc}").into());
            }
            let rc = unsafe {
                memra_dsv4_fp4_gemm(
                    codes_gpu.device_ptr(&stream).0 as *const c_void,
                    scales_gpu.device_ptr(&stream).0 as *const f32,
                    weight_gpu.device_ptr(&stream).0 as *const c_void,
                    weight_scales_gpu.device_ptr(&stream).0 as *const c_void,
                    0.0,
                    1,
                    output_gpu.device_ptr_mut(&stream).0 as *mut f32,
                    ROWS as i32,
                    n as i32,
                    k as i32,
                    native_stream,
                )
            };
            if rc != 0 {
                return Err(format!("{stem}: fp4_gemm returned {rc}").into());
            }
            let codes = engine.dtoh_u8(&codes_gpu)?;
            let scales = engine.dtoh(&scales_gpu)?;
            let output = engine.dtoh(&output_gpu)?;
            validate_activation(&codes, &scales, ROWS, k)?;
            if output.len() != ROWS * n {
                return Err(format!("{stem}: wrong GPU output shape").into());
            }

            let mut max_abs = 0.0f64;
            let mut sum_abs = 0.0f64;
            let mut max_bound_fraction = 0.0f64;
            let mut worst_index = 0usize;
            let mut violations = 0usize;
            for (index, &observed) in output.iter().enumerate() {
                let (expected, l1) = cpu_dot(
                    &codes,
                    &scales,
                    weight,
                    weight_scales,
                    index / n,
                    index % n,
                    k,
                );
                if !observed.is_finite() || !expected.is_finite() || !l1.is_finite() {
                    return Err(format!("{stem}: non-finite output at {index}").into());
                }
                let error = (observed as f64 - expected).abs();
                let bound = ABS_FLOOR + L1_FACTOR * l1;
                if error > max_abs {
                    max_abs = error;
                    worst_index = index;
                }
                sum_abs += error;
                max_bound_fraction = max_bound_fraction.max(error / bound);
                violations += usize::from(error > bound);
            }
            total_violations += violations;
            writeln!(
                text,
                "projection\t1\t{expert}\t{projection}\t{ROWS}\t{n}\t{k}\t{max_abs:.9e}\t{:.9e}\t{max_bound_fraction:.9e}\t{worst_index}\t{violations}",
                sum_abs / output.len() as f64
            )?;
            eprintln!(
                "layer 1 expert {expert} {projection}: max_abs={max_abs:.6e}, max_bound_fraction={max_bound_fraction:.6e}, violations={violations}"
            );
        }
    }
    let staged_path = report.with_extension("tsv.writing");
    let mut staged = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staged_path)?;
    staged.write_all(text.as_bytes())?;
    staged.sync_all()?;
    drop(staged);
    std::fs::rename(staged_path, report)?;
    if total_violations != 0 {
        return Err(format!("{total_violations} GEMM outputs exceeded the error bound").into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_dot_uses_low_then_high_nibble_and_both_scale_grids() {
        let mut codes = vec![0u8; 128];
        codes[0] = 0x38;
        codes[1] = 0x38;
        let mut weight = vec![0u8; 64];
        weight[0] = 0x21;
        let mut scales = vec![1.0f32];
        validate_activation(&codes, &scales, 1, 128).unwrap();
        let mut weight_scales = vec![127u8; 4];
        let (dot, l1) = cpu_dot(&codes, &scales, &weight, &weight_scales, 0, 0, 128);
        assert_eq!((dot, l1), (1.5, 1.5));
        scales[0] = 2.0;
        weight_scales[0] = 128;
        let (dot, l1) = cpu_dot(&codes, &scales, &weight, &weight_scales, 0, 0, 128);
        assert_eq!((dot, l1), (6.0, 6.0));
        scales.push(1.0);
        assert!(validate_activation(&codes, &scales, 1, 128).is_err());
        scales.pop();
        scales[0] = 1.5;
        assert!(validate_activation(&codes, &scales, 1, 128).is_err());
        scales[0] = 1.0;
        codes[0] = 0x7f;
        assert!(validate_activation(&codes, &scales, 1, 128).is_err());
    }
}
