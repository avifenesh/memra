//! ARTIFACT twin of the offline BF16 clipping diagnostic
//! (research/qwen-fp4-activation-mint-20260909).
//!
//! The calibration run measured, per projection, how often an activation reached the E2M1 grid
//! end, exceeded the calibrated global range, or saturated a UE4M3 block scale. It measured that
//! on BF16 activations with an offline quantizer, and its own receipt says so
//! (`native_mint_diagnostic_required: true`). Those numbers are not artifact facts until the same
//! three definitions are counted inside the kernel the engine actually runs, on the artifact it
//! actually loads.
//!
//! This primes real held-out prompts and dumps the per-projection counters as JSON.
//!
//! Usage: qwen-a4-clip-diag <calibrated.gguf> <out.json> <prompt.txt>:<tokens> ...
use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_engine::mmq_ffi::A4_CLIP_STRIDE;
use memra_gguf::GgufFile;

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: qwen-a4-clip-diag <calibrated.gguf> <out.json> <prompt.txt>:<tokens> ...");
    let out_path = args.next().expect("an output path");
    let specs: Vec<String> = args.collect();
    assert!(!specs.is_empty(), "at least one prompt is required");

    let e = Engine::new(0)?;
    let g = GgufFile::open(&model_path)?;
    let tok = memra_tokenizer::Tokenizer::from_gguf(&g)?;
    let model = HybridModel::load(&e, &g)?;
    let program = model
        .cfg
        .prefill_activation
        .clone()
        .expect("the artifact must declare the activation program");
    let names = program.slot_names();

    let mut runs: Vec<String> = Vec::new();
    for spec in &specs {
        let (file, want) = match spec.rsplit_once(':') {
            Some((f, n)) if n.parse::<usize>().is_ok() => (f, n.parse::<usize>().unwrap()),
            _ => (spec.as_str(), usize::MAX),
        };
        let text = std::fs::read_to_string(file)?;
        let mut ids = tok.encode(&text, true);
        if want != usize::MAX {
            ids.truncate(want);
        }
        let mut cache = memra_engine::pp::new_cache(&e, &model.cfg, ids.len() + 64)?;
        e.a4_clip_stats_enable()?;
        model.prime_cache(&e, &ids, &mut cache, 0)?;
        let counters = e
            .a4_clip_stats_take()?
            .expect("the diagnostic was armed before the prime");

        // A projection that quantized zero values did not run, and its row would be zeros --
        // which reads exactly like "no clipping". Refuse instead of publishing that.
        let silent: Vec<&str> = names
            .iter()
            .enumerate()
            .filter(|(slot, _)| counters[slot * A4_CLIP_STRIDE] == 0)
            .map(|(_, name)| *name)
            .collect();
        if !silent.is_empty() {
            eprintln!(
                "FAIL: {} of {} projections quantized zero values; a zero row is not a clean row",
                silent.len(),
                names.len()
            );
            for name in silent.iter().take(16) {
                eprintln!("  {name}");
            }
            std::process::exit(1);
        }

        let rate = |n: u64, d: u64| if d == 0 { 0.0 } else { n as f64 / d as f64 };
        let mut rows: Vec<String> = Vec::with_capacity(names.len());
        for (slot, name) in names.iter().enumerate() {
            let c = &counters[slot * A4_CLIP_STRIDE..slot * A4_CLIP_STRIDE + 5];
            rows.push(format!(
                "      \"{}\": {{\"values\": {}, \"blocks\": {}, \"fp4_max_fraction\": {:.6e}, \
                 \"global_clip_fraction\": {:.6e}, \"block_scale_clip_fraction\": {:.6e}}}",
                escape(name),
                c[0],
                c[4],
                rate(c[1], c[0]),
                rate(c[2], c[0]),
                rate(c[3], c[4])
            ));
        }
        println!(
            "{file}: {} tokens, {} projections counted",
            ids.len(),
            names.len()
        );
        runs.push(format!(
            "    {{\n      \"prompt\": \"{}\",\n      \"tokens\": {},\n      \"projections\": {{\n{}\n      }}\n    }}",
            escape(file),
            ids.len(),
            rows.join(",\n")
        ));
    }

    let doc = format!(
        "{{\n  \"program\": \"{}\",\n  \"artifact\": \"{}\",\n  \"source\": \"native memra \
         prefill quantizer, artifact activations\",\n  \"definition\": \"fp4_max_fraction counts \
         E2M1 magnitude code 6; global_clip_fraction counts abs(x) > 2688*s; \
         block_scale_clip_fraction counts raw UE4M3 scale > 448\",\n  \"runs\": [\n{}\n  ]\n}}\n",
        memra_gguf::model_packs::qwen35::activation::PROGRAM,
        escape(&model_path),
        runs.join(",\n")
    );
    std::fs::write(&out_path, doc)?;
    println!("wrote {out_path}");
    Ok(())
}
