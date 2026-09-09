//! Diagnostic logit projections after each trunk layer, OFF versus ON.
//! Intermediate rows use the final norm/head as a probe, not as a model quality score.
use memra_engine::{Engine, cache::Cache, forward::argmax, hybrid::HybridModel};
use memra_gguf::GgufFile;
use memra_tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let e = Engine::new(0)?;
    let g = GgufFile::open(&args[1])?;
    let mut model = HybridModel::load_without_mtp(&e, &g)?;
    let tok = Tokenizer::from_gguf(&g).map_err(|x| format!("tokenizer: {x}"))?;
    let text = std::fs::read_to_string(&args[2])?;
    let rendered = tok.apply_chat_template(&[("user", text.as_str())], true);
    let ids = tok.encode(&rendered, true);
    let layers = std::mem::take(&mut model.layers);
    for (layer, weights) in layers.into_iter().enumerate() {
        model.layers.push(weights);
        let mut arms = Vec::new();
        for flag in ["0", "1"] {
            unsafe { std::env::set_var("MEMRA_PRIME_ATTN_FA2", flag); }
            let mut cache = Cache::new(&e, &model.cfg, ids.len() + 64)?;
            let (logits, _, _) = model.prime_cache(&e, &ids, &mut cache, 0)?;
            assert!(logits.iter().all(|x| x.is_finite()));
            arms.push(logits);
        }
        let max_delta = arms[0].iter().zip(&arms[1]).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        let rms = (arms[0].iter().zip(&arms[1]).map(|(a,b)| (f64::from(*a)-f64::from(*b)).powi(2)).sum::<f64>() / arms[0].len() as f64).sqrt();
        println!("{{\"layer\":{layer},\"prompt_tokens\":{},\"max_logit_deviation\":{max_delta},\"rms_logit_deviation\":{rms},\"off_argmax\":{},\"on_argmax\":{}}}", ids.len(), argmax(&arms[0]), argmax(&arms[1]));
    }
    Ok(())
}
