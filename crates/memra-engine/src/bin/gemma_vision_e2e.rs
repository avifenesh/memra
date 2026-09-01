//! gemma-4 vision end-to-end gate (lane/gemma-vision masked prefill):
//! image + prompt -> tower -> embedding overlay -> ISLAND-MASKED monolithic prime ->
//! greedy decode. The decisive can't-hallucinate probe (blue triangle law) runs against
//! the reference answers from llama-mtmd-cli on the IDENTICAL GGUF pair.
//!
//! Usage: gemma_vision_e2e <model.gguf> <mmproj.gguf> <image> <prompt> [n_gen]
//!
//! MEMRA_GV_CAUSAL=1 runs the DELIBERATELY WRONG arm (overlay without the island mask,
//! plain causal prime) — the tell that separates the mask law from the overlay splice.

use memra_engine::Engine;
use memra_engine::vision_gemma::{
    GV_MERGE, GV_TOK_BEGIN, GV_TOK_END, GV_TOK_SOFT, GemmaVisionTower, gemma_prep_image,
};
use memra_gguf::GgufFile;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let model_path = a.next().expect("model.gguf");
    let mmproj = a.next().expect("mmproj.gguf");
    let image = a.next().expect("image path");
    let prompt = a.next().expect("prompt");
    let n_gen: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(200);

    let e = Engine::new(0)?;
    let g = GgufFile::open(&model_path)?;
    let tok = memra_tokenizer::Tokenizer::from_gguf(&g)?;
    let model = memra_engine::hybrid::HybridModel::load(&e, &g)?;
    let tower = GemmaVisionTower::load(&e, Path::new(&mmproj))?;

    // tower forward
    let (patches, gw, gh) = gemma_prep_image(&std::fs::read(&image)?)?;
    let n_soft = (gw / GV_MERGE) * (gh / GV_MERGE);
    let rows = tower.forward(&e, &patches, gw, gh)?;
    eprintln!("[e2e] image grid {gw}x{gh} -> {n_soft} soft tokens");

    // Render the chat turn with a placeholder, then splice the image token ids at the
    // placeholder position: <|image> + n_soft * <|image|> + <image|>.
    // "RAW:" prefix = cross-engine alignment mode: the rest is the EXACT rendered prompt
    // (e.g. llama-server /apply-template output) with a <__media__> marker — both engines
    // then score the identical token stream and first-step logits are comparable.
    const PLACEHOLDER: &str = "\u{1f5bc}"; // frame-with-picture, single-token-safe marker
    let rendered = match prompt.strip_prefix("RAW:") {
        Some(raw) => raw.replace("<__media__>", PLACEHOLDER),
        None => {
            let content = format!("{PLACEHOLDER}{prompt}");
            tok.apply_chat_template(&[("user", &content)], true)
        }
    };
    let (pre, post) = rendered
        .split_once(PLACEHOLDER)
        .ok_or("placeholder lost in template render")?;
    let mut ids = tok.encode(pre, true);
    let span_start = ids.len() + 1; // after <|image>
    ids.push(GV_TOK_BEGIN);
    ids.extend(std::iter::repeat_n(GV_TOK_SOFT, n_soft));
    ids.push(GV_TOK_END);
    ids.extend(tok.encode(post, false));
    eprintln!(
        "[e2e] prompt {} tokens (span at {span_start}+{n_soft})",
        ids.len()
    );

    let overlay = memra_engine::vision::EmbedOverlay::new(&e, rows, vec![(span_start, 0, n_soft)]);
    let causal_wrong_arm = std::env::var("MEMRA_GV_CAUSAL").as_deref() == Ok("1");

    let mut cache = memra_engine::cache::Cache::new(&e, &model.cfg, ids.len() + n_gen + 8)?;
    let t0 = std::time::Instant::now();
    if causal_wrong_arm {
        // wrong-arm: overlay rows spliced, but plain causal mask — the ship-by-analogy bug
        unsafe { std::env::set_var("MEMRA_GV_FORCE_CAUSAL", "1") };
    }
    let (logits, _seed, _hidden) =
        model.prime_cache_overlaid(&e, &ids, &mut cache, 0, Some(&overlay))?;
    eprintln!(
        "[e2e] prime {} tokens in {:.2}s",
        ids.len(),
        t0.elapsed().as_secs_f32()
    );

    // first-step top-8 (id, logit) — the same-weights cross-engine numeric gate
    {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let top: Vec<String> = idx[..8]
            .iter()
            .map(|&i| format!("({i}, {:.4})", logits[i]))
            .collect();
        println!("TOP8: {}", top.join(" "));
    }
    let mut next = memra_engine::forward::argmax(&logits) as u32;
    let mut out_ids = Vec::with_capacity(n_gen);
    let eot = tok.id_of("<end_of_turn>");
    for _ in 0..n_gen {
        out_ids.push(next);
        if Some(next) == eot {
            break;
        }
        let ll = model.decode_step(&e, next, &mut cache)?;
        next = memra_engine::forward::argmax(&ll) as u32;
    }
    println!("ANSWER: {}", tok.decode(&out_ids));
    Ok(())
}
