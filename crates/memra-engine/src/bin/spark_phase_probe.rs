//! spark-phase-probe: localize a prefill-exact / decode-wrong split on a dense model.
//! Runs the same prompt through four compositions and prints the argmax + top-5 of each:
//!   A  prime(P) -> eager decode_step(a0)
//!   B  prime(P) -> batched decode_step_batch([a0])
//!   C  prime(P + a0)            (pure prefill reference for step 1)
//!   D  eager decode_step over every prompt token, then decode_step(d0)
//! usage: spark-phase-probe <model-dir-or-gguf> <comma-separated token ids>
use memra_engine::Engine;
use memra_engine::cache::Cache;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgufFile;

fn top5(l: &[f32]) -> String {
    let mut ix: Vec<usize> = (0..l.len()).collect();
    ix.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap());
    ix.iter()
        .take(5)
        .map(|&i| format!("{}:{:.3}", i, l[i]))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: spark-phase-probe <model> <ids>");
    let ids: Vec<u32> = std::env::args()
        .nth(2)
        .expect("ids")
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let e = Engine::new(0)?;
    let model = if std::path::Path::new(&path).is_dir() {
        let src: Box<dyn memra_gguf::source::TensorSource> = Box::new(
            memra_gguf::source::SafetensorsSource::open(std::path::Path::new(&path))?,
        );
        HybridModel::load_from_source_without_mtp(&e, src.as_ref())?
    } else {
        let g = GgufFile::open(&path)?;
        HybridModel::load_without_mtp(&e, &g)?
    };
    let ctx = ids.len() + 64;
    let am = |l: &[f32]| memra_engine::forward::argmax(l) as u32;

    // A
    let mut ca = Cache::new(&e, &model.cfg, ctx)?;
    let (l0, _, _) = model.prime_cache(&e, &ids, &mut ca, 0)?;
    let a0 = am(&l0);
    println!("prime(P)        t0={a0}  top5 {}", top5(&l0));
    let l1 = model.decode_step(&e, a0, &mut ca)?;
    println!("A eager   t1={}  top5 {}", am(&l1), top5(&l1));
    let l2 = model.decode_step(&e, am(&l1), &mut ca)?;
    println!("A eager   t2={}  top5 {}", am(&l2), top5(&l2));

    // B
    let mut cb = Cache::new(&e, &model.cfg, ctx)?;
    let _ = model.prime_cache(&e, &ids, &mut cb, 0)?;
    let rows = {
        let mut caches = [&mut cb];
        model.decode_step_batch(&e, &[a0], &mut caches)?
    };
    println!("B batched t1={}  top5 {}", am(&rows[0]), top5(&rows[0]));

    // C
    let mut cc = Cache::new(&e, &model.cfg, ctx)?;
    let mut ext = ids.clone();
    ext.push(a0);
    let (lc, _, _) = model.prime_cache(&e, &ext, &mut cc, 0)?;
    println!("C prime(P+t0) t1={}  top5 {}", am(&lc), top5(&lc));

    // D
    let mut cd = Cache::new(&e, &model.cfg, ctx)?;
    let mut ld = Vec::new();
    for &t in &ids {
        ld = model.decode_step(&e, t, &mut cd)?;
    }
    let d0 = am(&ld);
    println!("D eager-all   t0={d0}  top5 {}", top5(&ld));
    let ld1 = model.decode_step(&e, d0, &mut cd)?;
    println!("D eager-all   t1={}  top5 {}", am(&ld1), top5(&ld1));
    Ok(())
}
