//! Numerical diagnostic: raw prime logits plus a separate layer-zero operator replay.
//! Input is `<id>\t<prompt-file>` per line. No serving/default changes.
use memra_engine::{
    Engine,
    hybrid::{HybridModel, Mixer},
};
use memra_gguf::GgufFile;
use memra_tokenizer::Tokenizer;
use std::{
    io::{BufRead, Write},
    path::Path,
};

fn save(path: &Path, values: &[f32]) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: gemma-numerical-probe <model.gguf> <output-directory>".into());
    }
    let out = Path::new(&args[2]);
    std::fs::create_dir_all(out)?;
    let e = Engine::new(0)?;
    let g = GgufFile::open(&args[1])?;
    let tok = Tokenizer::from_gguf(&g).map_err(|error| format!("{error}"))?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let (id, path) = line.split_once('\t').ok_or("expected id and prompt path")?;
        let _: usize = id.parse()?;
        let prompt = std::fs::read_to_string(path)?;
        let ids = tok.encode(&prompt, true);
        let mut cache = memra_engine::pp::new_cache(&e, &model.cfg, ids.len() + 8)?;
        let (logits, _, _) = model.prime_cache(&e, &ids, &mut cache, 0)?;
        save(&out.join(format!("{id}-logits.f32")), &logits)?;
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        println!(
            "{}\t{}\t{}",
            id,
            ids.iter().map(u32::to_string).collect::<Vec<_>>().join(","),
            order[..5]
                .iter()
                .map(|&i| format!("{}:{:.9}", i, logits[i]))
                .collect::<Vec<_>>()
                .join(",")
        );
        std::io::stdout().flush()?;
        drop(cache);
        // Recompute only the embedding, norm and Q/K/V projections after the real prime.
        // These use the same public operations and row count as the native layer-zero path.
        let n = model.cfg.n_embd as usize;
        let mut x = model.embed(&e, &ids)?;
        e.scale_inplace(&mut x, (n as f32).sqrt(), ids.len() * n)?;
        let xh = e.dtoh(&x)?;
        save(
            &out.join(format!("{id}-embedding.f32")),
            &xh[xh.len() - n..],
        )?;
        let mut h = e.zeros(ids.len() * n)?;
        let layer = &model.layers[0];
        e.rms_norm(
            &x,
            layer.attn_norm.float_data(),
            &mut h,
            n,
            ids.len(),
            model.cfg.rms_eps,
        )?;
        let hh = e.dtoh(&h)?;
        save(
            &out.join(format!("{id}-attn_norm-0.f32")),
            &hh[hh.len() - n..],
        )?;
        let Mixer::Full(attn) = &layer.mixer else {
            return Err("layer zero is not full attention".into());
        };
        e.mmq_act_begin();
        for (name, weight) in [
            ("Qcur-0", &attn.wq),
            ("Kcur-0", &attn.wk),
            ("Vcur-0", &attn.wv),
        ] {
            let values = e.matmul(weight, &h, ids.len())?;
            let values = e.dtoh(&values)?;
            let width = weight.out_features();
            save(
                &out.join(format!("{id}-{name}.f32")),
                &values[values.len() - width..],
            )?;
        }
        e.mmq_act_begin(); // clear the replay window before dropping its input
    }
    Ok(())
}
