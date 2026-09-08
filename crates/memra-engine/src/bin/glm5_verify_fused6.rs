//! Exact real-input six-projection oracle and bounded ABBA component cell.
use cudarc::driver::CudaSlice;
use memra_engine::{Engine, QT_F8_E4M3, kda::KdaAttnLayer, model::GpuTensor};
use memra_gguf::{
    model_plan::{AttentionPlan, ModelPlan},
    source::{SafetensorsSource, TensorSource},
};
use std::{error::Error, fs, io::Write, path::Path, time::Instant};
type Res<T> = Result<T, Box<dyn Error>>;

struct Cell {
    layer: usize,
    weights: KdaAttnLayer,
    inputs: Vec<CudaSlice<f32>>,
}
fn chain(e: &Engine, cell: &Cell, wi: usize, t: usize, fused: bool) -> Res<Vec<CudaSlice<f32>>> {
    unsafe {
        std::env::set_var(
            "MEMRA_GLM5_VERIFY_E4M3_FUSED6",
            if fused { "1" } else { "0" },
        );
    }
    if fused {
        e.kda_proj_fused6(&cell.weights, &cell.inputs[wi], t)?
            .ok_or_else(|| "fused candidate refused".into())
    } else {
        [
            &cell.weights.wq,
            &cell.weights.wk,
            &cell.weights.wv,
            &cell.weights.f_a,
            &cell.weights.g_a,
            &cell.weights.b_proj,
        ]
        .into_iter()
        .map(|w| e.matmul_rows_exact(w, &cell.inputs[wi], t))
        .collect()
    }
}
fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }
    best
}
fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: model inputs out oracle|bench".into());
    }
    let input = Path::new(&args[1]);
    let out = Path::new(&args[2]);
    fs::create_dir_all(out)?;
    let bench = match args[3].as_str() {
        "oracle" => false,
        "bench" => true,
        _ => return Err("bad mode".into()),
    };
    if bench && fs::read_to_string(out.join("oracle.pass"))? != "34 layers at t2/4/7 passed\n" {
        return Err("all width oracles required before timing".into());
    }
    unsafe {
        std::env::set_var("MEMRA_KDA_FUSED_PROJ", "1");
    }
    let e = Engine::new(0)?;
    let src = SafetensorsSource::open(Path::new(&args[0]))?;
    let cfg = src.try_config().map_err(std::io::Error::other)?;
    let plan = match memra_gguf::model_packs::for_config(&cfg) {
        Some(pack) => pack.compile_plan(&cfg)?,
        None => ModelPlan::compile(&cfg)?,
    };
    let mut cells = Vec::new();
    for layer in &plan.layers {
        let AttentionPlan::KimiDeltaNet(kda) = &layer.attention else {
            continue;
        };
        let il = layer.index as usize;
        let weights = KdaAttnLayer::load(&e, &src, layer.index, kda)?;
        for w in [
            &weights.wq,
            &weights.wk,
            &weights.wv,
            &weights.f_a,
            &weights.g_a,
            &weights.b_proj,
        ] {
            if !matches!(
                w,
                GpuTensor::Quant {
                    qtype: QT_F8_E4M3,
                    ..
                }
            ) || w.in_features() != 4096
            {
                return Err("wrong E4M3 input-projection geometry".into());
            }
        }
        let mut inputs = Vec::new();
        for t in [2, 4, 7] {
            let bytes = fs::read(input.join(format!("t{t}/layer-{il:03}.f32")))?;
            if bytes.len() != t * 4096 * 4 {
                return Err("incomplete real-input capture".into());
            }
            let x: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            if x.iter().any(|x| !x.is_finite()) || x.iter().all(|x| *x == 0.) {
                return Err("invalid input".into());
            }
            inputs.push(e.htod(&x)?);
        }
        cells.push(Cell {
            layer: il,
            weights,
            inputs,
        });
    }
    if cells.len() != 34 {
        return Err("expected all 34 KDA layers".into());
    }
    if !bench {
        let mut log = fs::File::create(out.join("oracle.tsv"))?;
        writeln!(
            log,
            "t\tlayer\tprojection\trow\tbit_diffs\tcurrent_argmax\tfused_argmax\tstatus"
        )?;
        for (wi, t) in [2, 4, 7].into_iter().enumerate() {
            for cell in &cells {
                let a = chain(&e, cell, wi, t, false)?;
                let before = memra_engine::kda::GLM5_VERIFY_E4M3_FUSED6_DISPATCHES
                    .load(std::sync::atomic::Ordering::Relaxed);
                let b = chain(&e, cell, wi, t, true)?;
                let after = memra_engine::kda::GLM5_VERIFY_E4M3_FUSED6_DISPATCHES
                    .load(std::sync::atomic::Ordering::Relaxed);
                if after - before != 1 {
                    return Err("candidate engagement counter mismatch".into());
                }
                for (pi, (a, b)) in a.iter().zip(&b).enumerate() {
                    let a = e.dtoh(a)?;
                    let b = e.dtoh(b)?;
                    if a.len() != b.len() || a.len() % t != 0 {
                        return Err("output geometry mismatch".into());
                    }
                    let n = a.len() / t;
                    for row in 0..t {
                        let av = &a[row * n..(row + 1) * n];
                        let bv = &b[row * n..(row + 1) * n];
                        let bad = av
                            .iter()
                            .zip(bv)
                            .filter(|(a, b)| a.to_bits() != b.to_bits())
                            .count();
                        let finite = av.iter().chain(bv).all(|x| x.is_finite());
                        let ok = bad == 0 && finite && argmax(av) == argmax(bv);
                        writeln!(
                            log,
                            "{t}\t{}\t{pi}\t{row}\t{bad}\t{}\t{}\t{}",
                            cell.layer,
                            argmax(av),
                            argmax(bv),
                            if ok { "PASS" } else { "FAIL" }
                        )?;
                        log.flush()?;
                        if !ok {
                            return Err(format!(
                                "oracle failed t={t} layer={} projection={pi} row={row} bits={bad}",
                                cell.layer
                            )
                            .into());
                        }
                    }
                }
                eprintln!(
                    "ORACLE PASS t={t} layer={} six projections byte-exact",
                    cell.layer
                );
            }
        }
        log.sync_all()?;
        fs::write(out.join("oracle.pass"), "34 layers at t2/4/7 passed\n")?;
        return Ok(());
    }
    let mut log = fs::File::create(out.join("bench.tsv"))?;
    writeln!(log, "regime\tt\tlayer\tblock\torder\tarm\tus")?;
    for (wi, t) in [2, 4, 7].into_iter().enumerate() {
        // All layer weights stay resident. The round arm rotates all 34 weight sets
        // to avoid claiming a whole-model gain from one L2-hot layer.
        for _ in 0..40 {
            for cell in &cells {
                drop(chain(&e, cell, wi, t, false)?);
                drop(chain(&e, cell, wi, t, true)?);
            }
        }
        e.stream().synchronize()?;
        for block in 0..5 {
            for (order, fused) in [false, true, true, false].into_iter().enumerate() {
                let start = Instant::now();
                for _ in 0..100 {
                    for cell in &cells {
                        drop(chain(&e, cell, wi, t, fused)?);
                    }
                }
                e.stream().synchronize()?;
                writeln!(
                    log,
                    "rotating\t{t}\tall\t{block}\t{order}\t{}\t{:.9}",
                    if fused { "fused" } else { "current" },
                    start.elapsed().as_secs_f64() * 1e4
                )?;
                log.flush()?;
            }
        }
        for cell in &cells {
            for _ in 0..40 {
                drop(chain(&e, cell, wi, t, false)?);
                drop(chain(&e, cell, wi, t, true)?);
            }
            e.stream().synchronize()?;
            for block in 0..5 {
                for (order, fused) in [false, true, true, false].into_iter().enumerate() {
                    let start = Instant::now();
                    for _ in 0..100 {
                        drop(chain(&e, cell, wi, t, fused)?);
                    }
                    e.stream().synchronize()?;
                    writeln!(
                        log,
                        "layer-hot\t{t}\t{}\t{block}\t{order}\t{}\t{:.9}",
                        cell.layer,
                        if fused { "fused" } else { "current" },
                        start.elapsed().as_secs_f64() * 1e4
                    )?;
                    log.flush()?;
                }
            }
        }
    }
    log.sync_all()?;
    Ok(())
}
