//! Decode the official MiMo MXFP4 expert bytes with Memra's existing CUDA
//! kernel and compare every BF16 element with the OCP scalar reference.
//! This is a component gate, not a routed MoE or serving path.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;

use cudarc::driver::{DevicePtr, DevicePtrMut};
use memra_engine::Engine;
use memra_engine::dsv4_ffi::memra_dsv4_mxfp4_deq_bf16;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::dsv4::dequant_mxfp4_expert;
use memra_gguf::model_packs::mimo_v2::inspect_pinned_source_headers;
use memra_gguf::safetensors::StModel;

type Fail = Box<dyn std::error::Error>;

fn bf16_rne_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let tie = (bits >> 16) & 1;
    (bits.wrapping_add(0x7fff + tie) >> 16) as u16
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_mxfp4_expert_gate: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Fail> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: mimo_mxfp4_expert_gate <source_dir> <gpu_index> <report.tsv>".into());
    }
    let dir = Path::new(&args[0]);
    let device: usize = args[1].parse()?;
    let report = Path::new(&args[2]);
    if report.exists() {
        return Err(format!("refusing to overwrite {}", report.display()).into());
    }
    let config_text = std::fs::read_to_string(dir.join("config.json"))?;
    let config = ModelConfig::from_hf(&HfConfig::try_parse(&config_text)?);
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

    let engine = Engine::new(device)?;
    engine.gpu.ctx.bind_to_thread()?;
    let stream = engine.stream();
    let mut text = String::from("format\tmemra-mimo-mxfp4-dequant-v1\n");
    writeln!(text, "gpu_index\t{device}")?;
    let mut mismatches = 0usize;
    for expert in [0u32, 255u32] {
        for projection in ["gate", "up", "down"] {
            let stem = format!("model.layers.1.mlp.experts.{expert}.{projection}_proj");
            let weight_name = format!("{stem}.weight");
            let scale_name = format!("{stem}.weight_scale");
            let (weight_info, weight) = model
                .raw(&weight_name)
                .ok_or_else(|| format!("{weight_name} missing"))?;
            let (scale_info, scales) = model
                .raw(&scale_name)
                .ok_or_else(|| format!("{scale_name} missing"))?;
            let (rows, cols) = if projection == "down" {
                (4_096usize, 2_048usize)
            } else {
                (2_048, 4_096)
            };
            if weight_info.dtype != "U8"
                || weight_info.shape != [rows as u64, (cols / 2) as u64]
                || scale_info.dtype != "U8"
                || scale_info.shape != [rows as u64, (cols / 32) as u64]
                || scales.contains(&0xff)
            {
                return Err(format!("{stem}: changed MXFP4 storage").into());
            }
            let expected = dequant_mxfp4_expert(weight, scales, rows, cols);
            let weight_dev = engine.htod_bytes(weight)?;
            let scale_dev = engine.htod_bytes(scales)?;
            let mut output_dev = stream.alloc_zeros::<u8>(rows * cols * 2)?;
            let native_stream = stream.cu_stream() as *mut c_void;
            let rc = unsafe {
                memra_dsv4_mxfp4_deq_bf16(
                    weight_dev.device_ptr(&stream).0 as *const c_void,
                    scale_dev.device_ptr(&stream).0 as *const c_void,
                    rows as i32,
                    cols as i32,
                    output_dev.device_ptr_mut(&stream).0 as *mut c_void,
                    native_stream,
                )
            };
            if rc != 0 {
                return Err(format!("{stem}: CUDA MXFP4 decoder returned {rc}").into());
            }
            let actual = engine.dtoh_u8(&output_dev)?;
            if actual.len() != rows * cols * 2 {
                return Err(format!("{stem}: BF16 output length mismatch").into());
            }
            let mut bad = 0usize;
            for (index, &value) in expected.iter().enumerate() {
                if !value.is_finite() {
                    return Err(format!("{stem}: non-finite source value at {index}").into());
                }
                let observed = u16::from_le_bytes([actual[2 * index], actual[2 * index + 1]]);
                if observed != bf16_rne_bits(value) {
                    bad += 1;
                }
            }
            mismatches += bad;
            writeln!(
                text,
                "projection\t1\t{expert}\t{projection}\t{rows}\t{cols}\t{bad}"
            )?;
            eprintln!(
                "layer 1 expert {expert} {projection}: {} BF16 values, {bad} mismatches",
                rows * cols
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
    if mismatches != 0 {
        return Err(format!("{mismatches} BF16 element mismatches").into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_round_to_nearest_even_handles_e2m1_power_of_two_values() {
        for (value, bits) in [
            (0.0, 0x0000),
            (0.5, 0x3f00),
            (1.0, 0x3f80),
            (1.5, 0x3fc0),
            (2.0, 0x4000),
            (-3.0, 0xc040),
        ] {
            assert_eq!(bf16_rne_bits(value), bits);
        }
    }
}
