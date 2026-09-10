//! WHERE does the calibrated clamp cut? Per-TOKEN-POSITION clipping for the p9999 twin.
//!
//! A microscopic clip FRACTION is not evidence that a clamp is harmless. The massive-activation
//! (attention-sink) literature says a handful of positions -- position 0 and delimiter tokens
//! above all -- carry activations orders of magnitude above every other position, concentrated in
//! early layers. Clipping exactly those elements wrecks a model while the fraction stays at 1e-4,
//! which is the shape the p9999 arm has: worst projection clips 9.7e-5 of values, and on
//! blk.0.ffn_down its clamp sits 306x BELOW the measured amax.
//!
//! So this reports, per armed projection: which row (token position) each clipped element sits on,
//! the max overshoot abs(x)/clamp, and the share of clipping that lands on the first few positions.
//! It reads the clamp from the ARTIFACT's own stamped multiplier, so running it against the
//! p9999 twin measures the p9999 clamps and running it against v3-amax measures those.
//!
//! No kernel change: it reuses the gate-harness capture, which hands back the raw f32 operand the
//! quantizer saw, so the arithmetic path the oracles certified is untouched.
//!
//! Usage: qwen-a4-sink-diag <artifact.gguf> <out.json> <prompt.txt> <tokens> <slot>[,<slot>...]
use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgufFile;

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect(
        "usage: qwen-a4-sink-diag <artifact.gguf> <out.json> <prompt.txt> <tokens> <slot,slot,...>",
    );
    let out_path = args.next().expect("an output path");
    let prompt = args.next().expect("a prompt file");
    let tokens: usize = args.next().expect("token count").parse()?;
    let slots: Vec<u32> = args
        .next()
        .expect("a comma-separated slot list")
        .split(',')
        .map(|s| s.trim().parse::<u32>().expect("slot index"))
        .collect();

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

    let text = std::fs::read_to_string(&prompt)?;
    let mut ids = tok.encode(&text, true);
    ids.truncate(tokens);
    assert!(
        ids.len() == tokens,
        "{prompt}: only {} tokens, need {tokens}",
        ids.len()
    );

    // The capture takes the FIRST launch per armed slot at or above the row threshold, so with a
    // prime that fits one chunk the captured rows ARE absolute positions 0..m-1. Refuse otherwise
    // rather than mislabel a row as a position.
    memra_engine::mmq_ffi::a4_capture_arm(slots.clone(), tokens);
    let mut cache = memra_engine::pp::new_cache(&e, &model.cfg, ids.len() + 64)?;
    model.prime_cache(&e, &ids, &mut cache, 0)?;
    let captures = memra_engine::mmq_ffi::a4_capture_take();
    assert!(
        !captures.is_empty(),
        "no slot captured: the prime chunked below {tokens} rows, so a row is not a position"
    );

    let mut rows: Vec<String> = Vec::new();
    for cap in &captures {
        let clamp = cap.input_scale * 6.0 * 448.0;
        let mut per_row_clipped = vec![0u32; cap.m];
        let mut per_row_amax = vec![0.0f32; cap.m];
        let mut clipped_total: u64 = 0;
        let mut amax = 0.0f32;
        for r in 0..cap.m {
            let row = &cap.x[r * cap.in_f..(r + 1) * cap.in_f];
            let mut n = 0u32;
            let mut mx = 0.0f32;
            for v in row {
                let a = v.abs();
                if a > mx {
                    mx = a;
                }
                if a > clamp {
                    n += 1;
                }
            }
            per_row_clipped[r] = n;
            per_row_amax[r] = mx;
            clipped_total += n as u64;
            if mx > amax {
                amax = mx;
            }
        }
        // Concentration: how much of the clipping sits on the few worst positions.
        let mut order: Vec<usize> = (0..cap.m).collect();
        order.sort_by_key(|&r| std::cmp::Reverse(per_row_clipped[r]));
        let top: Vec<usize> = order.iter().copied().take(8).collect();
        let top_share = if clipped_total == 0 {
            0.0
        } else {
            top.iter().map(|&r| per_row_clipped[r] as f64).sum::<f64>() / clipped_total as f64
        };
        let name = names[cap.slot as usize];
        println!(
            "slot {:3} {:28} clamp {:.4} amax {:.4} overshoot {:.1}x clipped {} of {} ({:.3e}) \
             top8 positions {:?} carry {:.1}%",
            cap.slot,
            name,
            clamp,
            amax,
            amax / clamp,
            clipped_total,
            cap.m * cap.in_f,
            clipped_total as f64 / (cap.m * cap.in_f) as f64,
            top,
            100.0 * top_share
        );
        // Name the positions: a sink signature is position 0 plus delimiter tokens, and an index
        // alone does not show that.
        for &r in &top {
            if per_row_clipped[r] == 0 {
                continue;
            }
            println!(
                "      pos {:5} id {:6} {:?} clipped {} amax {:.4} ({:.1}x clamp)",
                r,
                ids[r],
                tok.decode_special(&[ids[r]], true),
                per_row_clipped[r],
                per_row_amax[r],
                per_row_amax[r] / clamp
            );
        }
        rows.push(format!(
            "    {{\"slot\": {}, \"name\": \"{}\", \"rows\": {}, \"in_features\": {}, \
             \"multiplier\": {:e}, \"clamp\": {:e}, \"amax\": {:e}, \"max_overshoot\": {:.4}, \
             \"clipped\": {}, \"clip_fraction\": {:.6e}, \"top8_positions\": {:?}, \
             \"top8_share\": {:.6}, \"per_row_clipped\": {:?}, \"per_row_amax\": [{}]}}",
            cap.slot,
            escape(name),
            cap.m,
            cap.in_f,
            cap.input_scale,
            clamp,
            amax,
            amax / clamp,
            clipped_total,
            clipped_total as f64 / (cap.m * cap.in_f) as f64,
            top,
            top_share,
            per_row_clipped,
            per_row_amax
                .iter()
                .map(|v| format!("{v:e}"))
                .collect::<Vec<_>>()
                .join(","),
        ));
    }

    let doc = format!(
        "{{\n  \"artifact\": \"{}\",\n  \"prompt\": \"{}\",\n  \"tokens\": {},\n  \"ids\": {:?},\n  \"projections\": [\n{}\n  ]\n}}\n",
        escape(&model_path),
        escape(&prompt),
        tokens,
        ids,
        rows.join(",\n")
    );
    std::fs::write(&out_path, doc)?;
    println!("wrote {out_path}");
    Ok(())
}
