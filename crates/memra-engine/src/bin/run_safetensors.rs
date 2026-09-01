//! Load an HF safetensors checkpoint (config.json + shards) through the source-agnostic seam and
//! run a forward, printing argmax + top-5 of the last-token logits. Dispatches Model (dense /
//! dense-attn MoE like OLMoE) vs HybridModel (qwen35 linear-attn + full-attn) on the arch.
//!
//! Gate harness for ST-MOE-PLAN: `run-safetensors <hf_dir> [tok ids...]`.

use memra_engine::Engine;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_engine::model::Model;
use memra_gguf::source::{SafetensorsSource, TensorSource};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: run-safetensors <hf_dir> [tok ids...]");
    let e = Engine::new(0)?;
    let src = SafetensorsSource::open(std::path::Path::new(&path))?;
    let cfg = src.config();
    println!("GPU: {}  arch: {:?}", e.ctx().name()?, cfg.arch);

    let toks: Vec<u32> = std::env::args()
        .skip(2)
        .filter_map(|s| s.parse().ok())
        .collect();
    let toks = if toks.is_empty() {
        vec![1u32, 2, 3, 4]
    } else {
        toks
    };
    println!("tokens: {toks:?}");

    let logits = if cfg.uses_hybrid_executor() {
        let model = HybridModel::load_from_source(&e, &src)?;
        let full = model.cfg.n_full_attn_layers();
        println!(
            "loaded hybrid: n_layer={} ({} full-attn, {} linear) n_embd={} n_head={}/{} head_dim={} n_vocab={}",
            model.cfg.n_layer,
            full,
            model.cfg.n_layer - full,
            model.cfg.n_embd,
            model.cfg.n_head,
            model.cfg.n_head_kv,
            model.cfg.head_dim_k,
            model.cfg.n_vocab
        );
        // MEMRA_ST_NGEN=N: greedy continuation gate (native-vs-GGUF byte compare)
        if let Ok(nn) = std::env::var("MEMRA_ST_NGEN") {
            let n: usize = nn.parse().unwrap_or(32);
            let out = model.generate(&e, &toks, n)?;
            println!("continuation: {out:?}");
        }
        model.forward_last(&e, &toks)?
    } else {
        let model = Model::load_dense_from_source(&e, &src)?;
        let moe = model
            .cfg
            .moe
            .as_ref()
            .map(|m| format!("MoE {}x{}/tok", m.expert_count, m.expert_used_count))
            .unwrap_or_else(|| "dense".into());
        println!(
            "loaded {moe}: n_layer={} n_embd={} n_head={}/{} head_dim={} n_ff={} n_vocab={}",
            model.cfg.n_layer,
            model.cfg.n_embd,
            model.cfg.n_head,
            model.cfg.n_head_kv,
            model.cfg.head_dim_k,
            model.cfg.n_ff,
            model.cfg.n_vocab
        );
        model.forward_last(&e, &toks)?
    };

    let am = argmax(&logits);
    if let Some(path) = std::env::var_os("MEMRA_ORACLE_OUT") {
        use std::fmt::Write as _;

        let full_precision = std::env::var("MEMRA_FULL_PREC").as_deref() == Ok("1");
        let engine = if full_precision {
            "memra-native-full-precision"
        } else {
            "memra-native-serving-numeric"
        };
        let numeric_class = if full_precision {
            "source-weights-float32-accumulation"
        } else {
            "serving-weight-and-cache-formats"
        };
        let mut oracle = format!(
            "format\tmemra-checkpoint-oracle-v1\nengine\t{engine}\nnumeric_class\t{numeric_class}\n"
        );
        writeln!(
            oracle,
            "tokens\t{}",
            toks.iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        )?;
        writeln!(oracle, "vocab\t{}", logits.len())?;
        for (index, value) in logits.iter().enumerate() {
            writeln!(oracle, "logit\t{index}\t{:08x}", value.to_bits())?;
        }
        std::fs::write(path, oracle)?;
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let bad = logits.iter().filter(|v| !v.is_finite()).count();
    println!(
        "argmax token = {am}  logit = {:.4}  non-finite={bad}/{}",
        logits[am],
        logits.len()
    );
    println!(
        "top-5: {:?}",
        idx[..5].iter().map(|&i| (i, logits[i])).collect::<Vec<_>>()
    );
    assert_eq!(bad, 0, "non-finite logits — forward is broken");
    Ok(())
}
