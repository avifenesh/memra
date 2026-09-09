// Offline-only probe, installed as a bin in a disposable source snapshot.
use memra_engine::{Engine, cache::Cache, hybrid::HybridModel};
use memra_gguf::GgufFile;
use memra_tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let e = Engine::new(0)?;
    let g = GgufFile::open(&a[1])?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    let tok = Tokenizer::from_gguf(&g).map_err(|x| format!("tokenizer: {x}"))?;
    let request: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a[2])?)?;
    let text = request["messages"][0]["content"].as_str().ok_or("missing content")?;
    let rendered = tok.apply_chat_template(&[("user", text)], true);
    let mut tokens = tok.encode(&rendered, true);
    let depth: usize = a[4].parse()?;
    assert!(tokens.len() >= depth);
    tokens.truncate(depth);
    let split = depth - 1024;
    let mut cache = Cache::new(&e, &model.cfg, depth + 1024)?;
    model.prime_cache(&e, &tokens[..split], &mut cache, 0)?;
    // This last chunk must execute the diagnostic dispatch, not replay a prefix graph.
    cache.qwen_prime_graph = None;
    unsafe {
        std::env::set_var("FA2_PROBE_ACTIVE", "1");
    }
    let (logits, _, _) = model.prime_cache(&e, &tokens[split..], &mut cache, 0)?;
    e.stream().synchronize()?;
    let raw: Vec<u8> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(format!("{}.logits.f32", a[3]), raw)?;
    let ids: Vec<u8> = tokens.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(format!("{}.tokens.u32", a[3]), ids)?;
    println!("real_chunk rows=1024 depth={depth} vocab={} argmax={}", logits.len(), memra_engine::forward::argmax(&logits));
    Ok(())
}
