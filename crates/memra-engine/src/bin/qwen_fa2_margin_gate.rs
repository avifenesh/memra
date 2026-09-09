//! Same-binary, teacher-forced OFF/ON gate for the experimental Qwen prime kernel.
use memra_engine::{Engine, cache::Cache, forward::argmax, hybrid::HybridModel};
use memra_gguf::GgufFile;
use memra_tokenizer::Tokenizer;
use std::path::Path;

fn margin(logits: &[f32]) -> f32 {
    let mut top = [f32::NEG_INFINITY; 2];
    for &x in logits {
        if x > top[0] { top[1] = top[0]; top[0] = x; }
        else if x > top[1] { top[1] = x; }
    }
    top[0] - top[1]
}

fn dump(path: &Path, values: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 { return Err("model.gguf output-dir prompt-files...".into()); }
    let out = Path::new(&args[2]);
    std::fs::create_dir_all(out)?;
    let e = Engine::new(0)?;
    let g = GgufFile::open(&args[1])?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    let tok = Tokenizer::from_gguf(&g).map_err(|x| format!("tokenizer: {x}"))?;
    let mut total_flips = 0;
    let mut max_delta = 0.0f32;
    for (prompt_index, file) in args[3..].iter().enumerate() {
        let text = std::fs::read_to_string(file)?;
        let rendered = tok.apply_chat_template(&[("user", text.as_str())], true);
        let ids = tok.encode(&rendered, true);
        assert!(ids.len() >= 128, "gate must reach FA2 prefill");
        unsafe { std::env::set_var("MEMRA_PRIME_ATTN_FA2", "0"); }
        let mut cache = Cache::new(&e, &model.cfg, ids.len() + 128)?;
        let (mut logits, _, _) = model.prime_cache(&e, &ids, &mut cache, 0)?;
        let mut gold = Vec::new();
        let mut forced = Vec::new();
        for position in 0..12 {
            assert!(logits.iter().all(|x| x.is_finite()));
            forced.push(argmax(&logits) as u32);
            dump(&out.join(format!("p{prompt_index}-s{position}-off.f32")), &logits)?;
            gold.push(logits);
            if position < 11 {
                logits = model.decode_step(&e, *forced.last().unwrap(), &mut cache)?;
            } else { logits = Vec::new(); }
        }
        drop(cache);
        unsafe { std::env::set_var("MEMRA_PRIME_ATTN_FA2", "1"); }
        let mut cache = Cache::new(&e, &model.cfg, ids.len() + 128)?;
        let (mut logits, _, _) = model.prime_cache(&e, &ids, &mut cache, 0)?;
        for position in 0..12 {
            assert!(logits.iter().all(|x| x.is_finite()));
            let delta = logits.iter().zip(&gold[position]).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
            let on = argmax(&logits) as u32;
            let flip = on != forced[position];
            total_flips += usize::from(flip);
            max_delta = max_delta.max(delta);
            dump(&out.join(format!("p{prompt_index}-s{position}-on.f32")), &logits)?;
            println!("{{\"prompt\":{prompt_index},\"prompt_tokens\":{},\"position\":{position},\"off_argmax\":{},\"on_argmax\":{on},\"flip\":{flip},\"max_logit_deviation\":{delta},\"off_margin\":{},\"on_margin\":{}}}", ids.len(), forced[position], margin(&gold[position]), margin(&logits));
            if position < 11 {
                logits = model.decode_step(&e, forced[position], &mut cache)?;
            }
        }
    }
    println!("{{\"summary\":true,\"argmax_flips\":{total_flips},\"max_logit_deviation\":{max_delta}}}");
    if total_flips != 0 { return Err(format!("FA2 margin gate failed: {total_flips} flips").into()); }
    Ok(())
}
