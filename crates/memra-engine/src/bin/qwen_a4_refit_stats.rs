//! OPERAND-TRUTH CALIBRATION PASS for the Qwen prefill A4 program.
//!
//! The v2 scales were fitted off-graph on a BF16 reference model's module inputs, and the
//! ffn_down fit landed 3.6x above the operand the engine actually quantizes (the tap was
//! upstream of the engine's SiLU/mul order). A GEMM oracle cannot see that: the kernel is
//! self-consistent against whatever scale it is handed. The ground truth is the operand
//! itself, so this measures it directly.
//!
//! Mechanism: arm the A4 statistics buffer, then prime. While armed, every stamped prefill
//! projection accumulates |x| amax and a 64-bin log2 histogram of its operand and the GEMM
//! DIVERTS to the ordinary W4A8 walk, so the whole trunk runs the served arithmetic and the
//! activations being measured are the ones a served prime produces, not A4-perturbed ones.
//!
//! Output: one JSON row per program slot -- count, zeros, measured amax, fitted amax
//! (multiplier * 6 * 448), their ratio, and the histogram -- plus a printed table sorted by
//! ratio with the per-class census outside 0.9..1.1.
//!
//! usage: qwen-a4-refit-stats <artifact.gguf> <out.json> <sample.bin> [sample.bin ...]
use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect(
        "usage: qwen-a4-refit-stats <artifact.gguf> <out.json> <sample.bin> [sample.bin ...]",
    );
    let out_path = args.next().expect("an output json path");
    let samples: Vec<String> = args.collect();
    if samples.is_empty() {
        return Err("at least one sample file is required".into());
    }

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load(&e, &g)?;
    let program = model
        .cfg
        .prefill_activation
        .as_ref()
        .expect("the artifact must declare the activation program");
    let names = program.slot_names();
    let fitted: Vec<f32> = names
        .iter()
        .map(|n| *program.scales().get(*n).expect("slot name in scales"))
        .collect();

    e.a4_stats_enable()?;
    let mut rows: Vec<(String, usize)> = Vec::new();
    for sample in &samples {
        let bytes = std::fs::read(sample)?;
        if bytes.len() % 4 != 0 {
            return Err(format!("{sample}: not a u32 token stream").into());
        }
        let ids: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if ids.len() < 1024 {
            return Err(format!("{sample}: {} tokens is not a prime", ids.len()).into());
        }
        let mut cache = memra_engine::pp::new_cache(&e, &model.cfg, ids.len() + 8)?;
        let started = std::time::Instant::now();
        model.prime_cache(&e, &ids, &mut cache, 0)?;
        println!(
            "  {sample}: {} tokens primed in {:.1}s",
            ids.len(),
            started.elapsed().as_secs_f32()
        );
        rows.push((sample.clone(), ids.len()));
    }
    let stats = e
        .a4_stats_take()?
        .expect("the stats pass was armed at startup");
    let stride = memra_engine::mmq_ffi::A4_STATS_STRIDE;
    assert_eq!(stats.len(), names.len() * stride, "slot buffer shape");

    // The percentile the twin arm will need, read off the histogram by linear interpolation
    // inside the log2 bin that contains the quantile.
    let percentile = |hist: &[u64], zeros: u64, total: u64, q: f64| -> Option<f32> {
        let target = (total as f64 * q).ceil() as u64;
        let mut seen = zeros;
        for (b, count) in hist.iter().enumerate() {
            let next = seen + count;
            if next >= target && *count > 0 {
                let frac = (target.saturating_sub(seen)) as f64 / *count as f64;
                // bin b covers [2^(b-40), 2^(b-39)); interpolate inside it.
                let lo = 2f64.powi(b as i32 - 40);
                return Some((lo * 2f64.powf(frac)) as f32);
            }
            seen = next;
        }
        None
    };

    let mut out = String::from("{\n  \"artifact\": \"");
    out.push_str(&path.replace('\\', "\\\\").replace('"', "\\\""));
    out.push_str("\",\n  \"samples\": [\n");
    for (i, (file, tokens)) in rows.iter().enumerate() {
        out.push_str(&format!(
            "    {{\"file\": \"{}\", \"tokens\": {}}}{}\n",
            file.replace('\\', "\\\\").replace('"', "\\\""),
            tokens,
            if i + 1 == rows.len() { "" } else { "," }
        ));
    }
    out.push_str("  ],\n  \"projections\": [\n");
    let mut table: Vec<(f32, &str, u64, f32, f32)> = Vec::new();
    for (slot, name) in names.iter().enumerate() {
        let block = &stats[slot * stride..(slot + 1) * stride];
        let count = block[0];
        let amax = f32::from_bits(block[1] as u32);
        let zeros = block[2];
        let hist: Vec<u64> = block[8..8 + 64].to_vec();
        let fitted_amax = fitted[slot] * 6.0 * 448.0;
        let ratio = amax / fitted_amax;
        let p999 = percentile(&hist, zeros, count, 0.999);
        let p9999 = percentile(&hist, zeros, count, 0.9999);
        table.push((ratio, name, count, amax, fitted_amax));
        out.push_str(&format!(
            "    {{\"name\": \"{name}\", \"slot\": {slot}, \"values\": {count}, \"zeros\": {zeros}, \
             \"amax\": {amax:e}, \"amax_fitted\": {fitted_amax:e}, \"ratio\": {ratio:.6}, \
             \"p999\": {}, \"p9999\": {}, \"hist\": [{}]}}{}\n",
            p999.map(|v| format!("{v:e}")).unwrap_or_else(|| "null".into()),
            p9999.map(|v| format!("{v:e}")).unwrap_or_else(|| "null".into()),
            hist.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(","),
            if slot + 1 == names.len() { "" } else { "," }
        ));
    }
    out.push_str("  ]\n}\n");
    std::fs::write(&out_path, out)?;

    table.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    println!(
        "\nratio = measured amax / fitted amax, over {} projections, {} samples",
        table.len(),
        samples.len()
    );
    println!("worst 12:");
    for (ratio, name, count, amax, fitted_amax) in table.iter().take(12) {
        println!(
            "  {ratio:.4}  {name}  (measured {amax:.4}, fitted {fitted_amax:.4}, {count} values)"
        );
    }
    println!("best 4:");
    for (ratio, name, _, amax, fitted_amax) in table.iter().rev().take(4) {
        println!("  {ratio:.4}  {name}  (measured {amax:.4}, fitted {fitted_amax:.4})");
    }
    let mut classes: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
    for (ratio, name, _, _, _) in &table {
        let class = name.split('.').nth(2).unwrap_or("?").to_string();
        let entry = classes.entry(class).or_default();
        entry.0 += 1;
        if !(0.9..=1.1).contains(ratio) {
            entry.1 += 1;
        }
    }
    let outside = table
        .iter()
        .filter(|(r, _, _, _, _)| !(0.9..=1.1).contains(r))
        .count();
    println!("\noutside 0.9..1.1: {outside} of {}", table.len());
    println!("| class | total | outside |");
    println!("|---|---|---|");
    for (class, (total, bad)) in &classes {
        println!("| {class} | {total} | {bad} |");
    }
    // Any slot that never saw a value means the pass missed a projection outright.
    let silent = table
        .iter()
        .filter(|(_, _, count, _, _)| *count == 0)
        .count();
    if silent != 0 {
        eprintln!("SILENT SLOTS: {silent} projections saw no operand; the table is not complete");
        std::process::exit(1);
    }
    println!("\nwrote {out_path}");
    Ok(())
}
