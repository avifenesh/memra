//! `MEMRA_MLA_SEG_WS` — the pooled MLA PRE/POST handoff buffers vs the owned ones, BITWISE over a
//! multi-step cached decode on the glm-dsa micro fixture. The pooled path reuses one set of
//! buffers for every MLA layer and every step, so a value that survived only because its buffer
//! was fresh (an unwritten element, a stale row from the previous layer) shows up here as a
//! divergence at step 2 or later, and the door's own counter proves the arm ran.
use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgufFile;
use memra_gguf::micro_gguf::write_glm_dsa_micro;

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static M: std::sync::Mutex<()> = std::sync::Mutex::new(());
    M.lock().unwrap_or_else(|p| p.into_inner())
}

/// Prime the cache on all but the last token, then decode `steps` tokens greedily, returning the
/// exact bits of every logit vector. `door` is PINNED (`"0"` or `"1"`), never unset: the OFF arm
/// of a door read per call must say so.
fn decode_bits(
    e: &Engine,
    model: &HybridModel,
    tokens: &[u32],
    steps: usize,
    door: &str,
) -> Vec<u32> {
    // SAFETY: single-threaded test serialized behind `gpu_guard`; the door is read per call.
    unsafe { std::env::set_var("MEMRA_MLA_SEG_WS", door) };
    let mut cache = memra_kv::Cache::new(e, &model.cfg, 128).expect("latent cache allocates");
    model
        .prime_cache(e, &tokens[..tokens.len() - 1], &mut cache, 0)
        .expect("prime");
    let mut next = tokens[tokens.len() - 1];
    let mut bits = Vec::new();
    for _ in 0..steps {
        let (logits, _) = model
            .decode_step_h(e, next, &mut cache)
            .expect("decode step");
        bits.extend(logits.iter().map(|v| v.to_bits()));
        next = logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| {
                if x > b.1 { (i, x) } else { b }
            })
            .0 as u32;
    }
    bits
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn pooled_handoff_is_bit_identical_to_owned() {
    let _g = gpu_guard();
    let p = std::env::temp_dir().join(format!("memra-mla-segws-{}.gguf", std::process::id()));
    write_glm_dsa_micro(&p, 0xCACE_0904).unwrap();
    let g = GgufFile::open(&p).unwrap();
    let e = Engine::new(0).expect("CUDA device 0");
    let model = HybridModel::load(&e, &g).expect("glm-dsa micro fixture loads");
    std::fs::remove_file(&p).ok();

    let tokens: Vec<u32> = (0..20u32).map(|i| (i * 7 + 3) % 128).collect();
    let owned = decode_bits(&e, &model, &tokens, 12, "0");
    let d0 = memra_engine::hybrid_forward::mla_seg_ws_dispatches();
    let pooled = decode_bits(&e, &model, &tokens, 12, "1");
    let engaged = memra_engine::hybrid_forward::mla_seg_ws_dispatches() - d0;
    // SAFETY: same as above.
    unsafe { std::env::set_var("MEMRA_MLA_SEG_WS", "0") };

    assert!(
        engaged > 0,
        "the MEMRA_MLA_SEG_WS door did not engage on the pooled arm"
    );
    assert!(
        owned.iter().any(|b| *b != 0),
        "vacuous: the owned arm produced all-zero logits"
    );
    assert_eq!(
        owned.len(),
        pooled.len(),
        "the two arms decoded different lengths"
    );
    let diffs = owned.iter().zip(&pooled).filter(|(a, b)| a != b).count();
    assert_eq!(
        diffs,
        0,
        "the pooled MLA handoff buffers changed {diffs} of {} logit bits over 12 steps",
        owned.len()
    );
    println!(
        "[mla-seg-ws] 12 decode steps x {} logits bit-identical, pooled ({engaged} layer-calls) vs owned",
        owned.len() / 12
    );
}
