//! b16 batched NVFP4 matvec schedule bench (batch-spec 2026-11-03). Streams every split-plane
//! NVFP4 2D tensor of the first L blocks of a GGUF through `qmatvec_mmvq_batched` at m columns
//! (default 16; mcols = the dispatch's batched_mcols(m)), one launch per tensor, so the pass is DRAM-cold like a decode step.
//! The schedule is the dispatch's auto pick, or the forced `MEMRA_BVAR` variant. Prints the
//! median pass time, effective GB/s and an FNV-1a hash of every output bit, so two schedules
//! compare bit-for-bit across processes.
//!
//! usage: mmvq-b16-bench MODEL.gguf [--layers L] [--m M] [--iters N]
use memra_engine::Engine;
use memra_engine::model::repack_nvfp4_split;
use memra_gguf::{GgmlType, GgufFile};
use std::time::Instant;

fn pr(i: usize) -> f32 {
    (((i.wrapping_mul(2654435761)) % 1000) as f32) / 500.0 - 1.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args
        .first()
        .ok_or("usage: mmvq-b16-bench MODEL.gguf [--layers L] [--m M] [--iters N]")?;
    let flag = |name: &str, def: usize| -> usize {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(def)
    };
    let layers = flag("--layers", 4);
    let m = flag("--m", 16);
    let iters = flag("--iters", 20);
    let e = Engine::new(0)?;
    let g = GgufFile::open(path)?;
    let mut ws = Vec::new();
    let mut bytes_total = 0usize;
    for t in &g.tensors {
        let in_layers = (0..layers).any(|l| t.name.starts_with(&format!("blk.{l}.")));
        if !in_layers || t.ggml_type != GgmlType::NVFP4 || t.ne.len() != 2 {
            continue;
        }
        let (in_f, out_f) = (t.ne[0] as usize, t.ne[1] as usize);
        // every variant of the sweep is admissible: whole 32-group g-iterations, smem scale plane.
        if !in_f.is_multiple_of(1024) || in_f / 64 > 272 || !out_f.is_multiple_of(16) {
            continue;
        }
        let raw = g.tensor_data(t);
        bytes_total += raw.len();
        let row_bytes = raw.len() / out_f;
        ws.push((
            t.name.clone(),
            e.htod_bytes(&repack_nvfp4_split(raw, out_f))?,
            in_f,
            out_f,
            row_bytes,
        ));
    }
    let mut acts = std::collections::BTreeMap::new();
    for w in &ws {
        if let std::collections::btree_map::Entry::Vacant(v) = acts.entry(w.2) {
            let x: Vec<f32> = (0..m * w.2).map(|i| pr(i + 211) * 0.1).collect();
            let xd = e.htod(&x)?;
            v.insert(e.quantize_q8_1(&xd, m, w.2)?);
        }
    }
    let pass = |keep: bool| -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let mut outs = Vec::new();
        for (_, wd, in_f, out_f, rb) in &ws {
            let (aq, ad) = &acts[in_f];
            let y = e.qmatvec_mmvq_batched(
                wd,
                aq,
                ad,
                m,
                *in_f,
                *out_f,
                memra_engine::QT_NVFP4,
                *rb,
                Engine::batched_mcols(m),
                1.0,
                true,
            )?;
            if keep {
                outs.push(e.dtoh(&y)?);
            }
        }
        Ok(outs)
    };
    let mut h: u64 = 0xcbf29ce484222325;
    for y in pass(true)? {
        for v in y {
            h ^= v.to_bits() as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    for _ in 0..3 {
        pass(false)?;
    }
    e.ctx().synchronize()?;
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let st = Instant::now();
        pass(false)?;
        e.ctx().synchronize()?;
        times.push(st.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    let sample = &ws[0];
    println!(
        "B16BENCH variant={} m={m} tensors={} bytes={:.1}MB median={med:.3}ms min={:.3}ms gbs={:.1} hash={h:016x} pick({}x{})={}",
        std::env::var("MEMRA_BVAR").unwrap_or_else(|_| "auto".into()),
        ws.len(),
        bytes_total as f64 / 1e6,
        times[0],
        bytes_total as f64 / (med / 1e3) / 1e9,
        sample.2,
        sample.3,
        e.batched_variant(
            m,
            sample.2,
            sample.3,
            memra_engine::QT_NVFP4,
            Engine::batched_mcols(m),
            true
        ),
    );
    Ok(())
}
