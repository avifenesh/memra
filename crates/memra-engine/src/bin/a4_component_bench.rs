//! Issue 439: finite component comparison on actual projection shapes and weights.
//! No model forward, quality score, calibration, or production dispatch change.
use cudarc::driver::{DevicePtr, DevicePtrMut, sys::CUevent_flags};
use memra_engine::{Engine, mmq_ffi, model::GpuTensor};
use memra_gguf::{GgmlType, GgufFile};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, error::Error, fs::File, io::Write, path::Path};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const ROWS: [usize; 3] = [512, 2048, 4096];
const ROUNDS: usize = 10;
const LAUNCHES: usize = 64;
const WARMUP: usize = 20;

struct Class {
    count: usize,
    k: usize,
    n: usize,
    layer: usize,
    name: String,
    multiplier: f32,
    input_group: bool,
}

fn digest_f32(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 || args.len() > 5 {
        return Err("usage: a4-component-bench MODEL SCALES.tsv OUTPUT_DIR [--census]".into());
    }
    let census_only = args.get(4).map(String::as_str) == Some("--census");
    if args.len() == 5 && !census_only {
        return Err("unknown option".into());
    }
    let out = Path::new(&args[3]);
    std::fs::create_dir(out)?;
    let g = GgufFile::open(&args[1])?;
    let mut classes: BTreeMap<String, Class> = BTreeMap::new();
    let mut seen = std::collections::BTreeSet::new();
    for line in std::fs::read_to_string(&args[2])?.lines().skip(1) {
        let (name, scale) = line.split_once('\t').ok_or("scale TSV")?;
        if !seen.insert(name.to_owned()) {
            return Err("duplicate scale".into());
        }
        let multiplier: f32 = scale.parse()?;
        if !multiplier.is_finite() || multiplier <= 0.0 {
            return Err("invalid multiplier".into());
        }
        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() != 4 || parts[0] != "blk" || parts[3] != "weight" {
            return Err("unexpected tensor name".into());
        }
        let layer: usize = parts[1].parse()?;
        if layer >= 64 {
            return Err("non-trunk scale".into());
        }
        let class = parts[2];
        let input_group = match class {
            "attn_q" | "attn_k" | "attn_v" | "attn_qkv" | "attn_gate" | "ffn_gate" | "ffn_up" => {
                true
            }
            "ffn_down" | "ssm_out" | "attn_output" => false,
            _ => return Err("unexpected projection class".into()),
        };
        let tensor = g.find(name).ok_or("scale names missing tensor")?;
        if tensor.ne.len() != 2 || tensor.ggml_type != GgmlType::NVFP4 {
            return Err("not a 2D NVFP4 tensor".into());
        }
        let (k, n) = (tensor.ne[0] as usize, tensor.ne[1] as usize);
        let entry = classes.entry(class.to_owned()).or_insert_with(|| Class {
            count: 0,
            k,
            n,
            layer,
            name: name.to_owned(),
            multiplier,
            input_group,
        });
        if (entry.k, entry.n) != (k, n) {
            return Err("class has multiple shapes; benchmark must group them explicitly".into());
        }
        entry.count += 1;
        if layer < entry.layer {
            entry.layer = layer;
            entry.name = name.to_owned();
            entry.multiplier = multiplier;
        }
    }
    if seen.len() != 400
        || classes.len() != 10
        || classes
            .values()
            .filter(|x| x.input_group)
            .map(|x| x.count)
            .sum::<usize>()
            != 272
    {
        return Err("program census differs from 400 total / 272 input slots / 10 classes".into());
    }
    let mut census = File::create(out.join("census.tsv"))?;
    writeln!(
        census,
        "class\tslots\tk\tn\tinput_group\trepresentative\tmultiplier\tweight_sha256"
    )?;
    for (name, c) in &classes {
        let raw = g.tensor_data(g.find(&c.name).unwrap());
        writeln!(
            census,
            "{name}\t{}\t{}\t{}\t{}\t{}\t{:.9e}\t{:x}",
            c.count,
            c.k,
            c.n,
            c.input_group,
            c.name,
            c.multiplier,
            Sha256::digest(raw)
        )?;
    }
    if census_only {
        println!("CENSUS PASS: 400 slots, 272 input slots, 10 uniform-shape classes");
        return Ok(());
    }
    let e = Engine::new(0)?;
    std::fs::write(out.join("device.txt"), format!("{}\n", e.ctx().name()?))?;
    let mut measurements = File::create(out.join("measurements.tsv"))?;
    writeln!(
        measurements,
        "class\tm\tk\tn\tround\torder\tarm\tlaunches\tgpu_ms\twall_ms"
    )?;
    let mut checks = File::create(out.join("checks.tsv"))?;
    writeln!(
        checks,
        "class\tm\tinput_sha256\tint8_sha256\tfp4_sha256\trel_l2_fp4_int8\trepeat_int8\trepeat_fp4\tbad_scale_rc\trp\tint8_scratch\tfp4_scratch"
    )?;
    for (class, c) in &classes {
        let weight = GpuTensor::load_opt(&e, &g, &c.name)?.ok_or("weight missing")?;
        let GpuTensor::Quant {
            bytes,
            qtype,
            scale,
            rp,
            ..
        } = &weight
        else {
            return Err("not quantized".into());
        };
        if *qtype != memra_engine::QT_NVFP4 || !*rp {
            return Err("expected native NVFP4 split-plane layout".into());
        }
        for m in ROWS {
            let mut seed = 0x4391_0202u32;
            let x: Vec<f32> = (0..m * c.k)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((seed >> 8) as f32) * (4.0 / 16777216.0) - 2.0
                })
                .collect();
            let input_hash = digest_f32(&x);
            let x_d = e.htod(&x)?;
            drop(x);
            let mut y8 = e.zeros(m * c.n)?;
            let mut y4 = e.zeros(m * c.n)?;
            let s8 = unsafe { mmq_ffi::memra_mmq_nvfp4_w4a8_act_bytes(c.k as i32, m as i32) };
            let s4 = unsafe {
                mmq_ffi::memra_mmq_nvfp4_calibrated_prefill_act_bytes(c.k as i32, m as i32)
            };
            if s8 == 0 || s4 == 0 {
                return Err("kernel scratch unsupported".into());
            }
            let mut scratch8 = e.stream().alloc_zeros::<u8>(s8)?;
            let mut scratch4 = e.stream().alloc_zeros::<u8>(s4)?;
            let stream = e.stream();
            let mut initial8 = String::new();
            let mut initial4 = String::new();
            let mut rel_l2 = 0.0;
            for phase in 0..2 {
                {
                    let (wp, _wg) = bytes.device_ptr(&stream);
                    let (xp, _xg) = x_d.device_ptr(&stream);
                    let (p8, _y8g) = y8.device_ptr_mut(&stream);
                    let (p4, _y4g) = y4.device_ptr_mut(&stream);
                    let (a8, _a8g) = scratch8.device_ptr_mut(&stream);
                    let (a4, _a4g) = scratch4.device_ptr_mut(&stream);
                    let st = stream.cu_stream() as *mut core::ffi::c_void;
                    let launch = |fp4: bool| -> Result<()> {
                        let rc = unsafe {
                            if fp4 {
                                mmq_ffi::memra_mmq_nvfp4_calibrated_prefill(
                                    wp as _,
                                    xp as _,
                                    p4 as _,
                                    c.k as i32,
                                    c.n as i32,
                                    m as i32,
                                    a4 as _,
                                    st,
                                    *scale,
                                    c.multiplier,
                                    1,
                                    std::ptr::null_mut(),
                                )
                            } else {
                                mmq_ffi::memra_mmq_nvfp4_w4a8(
                                    wp as _, xp as _, p8 as _, c.k as i32, c.n as i32, m as i32,
                                    a8 as _, st, *scale, 1,
                                )
                            }
                        };
                        if rc != 0 {
                            return Err(format!(
                                "{} {class} m={m} rc={rc}",
                                if fp4 { "fp4" } else { "int8" }
                            )
                            .into());
                        }
                        Ok(())
                    };
                    if phase == 0 {
                        launch(false)?;
                        launch(true)?;
                    } else {
                        for _ in 0..WARMUP {
                            launch(false)?;
                            launch(true)?;
                        }
                        stream.synchronize()?;
                        for round in 0..ROUNDS {
                            let order = if round % 2 == 0 {
                                [false, true]
                            } else {
                                [true, false]
                            };
                            for fp4 in order {
                                let flags = Some(CUevent_flags::CU_EVENT_DEFAULT);
                                let start = stream.record_event(flags)?;
                                let wall = std::time::Instant::now();
                                for _ in 0..LAUNCHES {
                                    launch(fp4)?;
                                }
                                let end = stream.record_event(flags)?;
                                stream.synchronize()?;
                                let wall_ms =
                                    wall.elapsed().as_secs_f64() * 1000.0 / LAUNCHES as f64;
                                let gpu_ms = f64::from(start.elapsed_ms(&end)?) / LAUNCHES as f64;
                                writeln!(
                                    measurements,
                                    "{class}\t{m}\t{}\t{}\t{round}\t{}\t{}\t{LAUNCHES}\t{gpu_ms:.9}\t{wall_ms:.9}",
                                    c.k,
                                    c.n,
                                    if round % 2 == 0 { "AB" } else { "BA" },
                                    if fp4 { "fp4" } else { "int8" }
                                )?;
                                measurements.flush()?;
                            }
                        }
                    }
                    stream.synchronize()?;
                }
                let v8 = e.dtoh(&y8)?;
                let v4 = e.dtoh(&y4)?;
                if !v8.iter().chain(&v4).all(|x| x.is_finite()) {
                    return Err("nonfinite output".into());
                }
                let h8 = digest_f32(&v8);
                let h4 = digest_f32(&v4);
                if h8 == h4 || v8.iter().all(|x| *x == 0.0) || v4.iter().all(|x| *x == 0.0) {
                    return Err("arms did not produce distinct nonzero outputs".into());
                }
                if phase == 0 {
                    initial8 = h8;
                    initial4 = h4;
                    let denom: f64 = v8.iter().map(|x| (*x as f64).powi(2)).sum();
                    rel_l2 = (v8
                        .iter()
                        .zip(&v4)
                        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
                        .sum::<f64>()
                        / denom)
                        .sqrt();
                } else {
                    if h8 != initial8 || h4 != initial4 {
                        return Err("output changed across repetitions".into());
                    }
                    let bad = unsafe {
                        mmq_ffi::memra_mmq_nvfp4_calibrated_prefill(
                            std::ptr::null(),
                            std::ptr::null(),
                            std::ptr::null_mut(),
                            c.k as i32,
                            c.n as i32,
                            m as i32,
                            std::ptr::null_mut(),
                            stream.cu_stream() as _,
                            *scale,
                            -1.0,
                            1,
                            std::ptr::null_mut(),
                        )
                    };
                    if bad != 2902 {
                        return Err(format!("invalid-scale red arm returned {bad}").into());
                    }
                    writeln!(
                        checks,
                        "{class}\t{m}\t{input_hash}\t{h8}\t{h4}\t{rel_l2:.9}\ttrue\ttrue\t{bad}\ttrue\t{s8}\t{s4}"
                    )?;
                    checks.flush()?;
                }
            }
            println!(
                "COMPONENT PASS {class} m={m} k={} n={} slots={} rounds={ROUNDS} launches={LAUNCHES}",
                c.k, c.n, c.count
            );
            std::io::stdout().flush()?;
        }
    }
    println!("BENCH PASS: 30 class/row cells, 600 interleaved samples; component result only");
    Ok(())
}
