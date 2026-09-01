//! gguf-census — per-type / per-tensor-class byte budget of a gguf artifact.
//!
//! The single-stream decode floor is bytes-read-per-token / achieved bandwidth; this
//! prints the exact numerator: every tensor's (name, type, elems, bytes) rolled up by
//! type and by decode-role class, so two artifacts' floors can be diffed analytically
//! before any kernel work is funded (gap-diagnosis arc, GAP-DIAGNOSIS.md).
//!
//! usage: gguf-census <model.gguf>

use memra_gguf::GgufFile;
use std::collections::BTreeMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: gguf-census <model.gguf>");
    let g = GgufFile::open(&path)?;

    let class = |n: &str| -> &'static str {
        if n.contains("attn_q.") || n.contains("attn_k.") {
            "attn_qk"
        } else if n.contains("attn_v.") {
            "attn_v"
        } else if n.contains("attn_output") || n.contains("attn_o.") {
            "attn_o"
        } else if n.contains("ffn_gate") {
            "ffn_gate"
        } else if n.contains("ffn_up") {
            "ffn_up"
        } else if n.contains("ffn_down") {
            "ffn_down"
        } else if n.contains("token_embd") {
            "embd"
        } else if n.contains("output.weight") {
            "lm_head"
        } else if n.contains("norm") {
            "norms"
        } else {
            "other"
        }
    };

    let mut by_type: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    let mut by_class: BTreeMap<&'static str, BTreeMap<String, u64>> = BTreeMap::new();
    let mut total: u64 = 0;
    for t in &g.tensors {
        let bytes = g.tensor_data(t).len() as u64;
        total += bytes;
        let ty = format!("{:?}", t.ggml_type);
        let e = by_type.entry(ty.clone()).or_default();
        e.0 += 1;
        e.1 += bytes;
        *by_class
            .entry(class(&t.name))
            .or_default()
            .entry(ty)
            .or_default() += bytes;
    }

    println!(
        "== {path} : {:.2} GiB total ==",
        total as f64 / (1u64 << 30) as f64
    );
    println!("-- by type --");
    for (ty, (n, b)) in &by_type {
        println!(
            "{ty:>10}  n={n:<4} {:.3} GiB",
            *b as f64 / (1u64 << 30) as f64
        );
    }
    println!("-- by class (decode-role) --");
    for (c, m) in &by_class {
        let cb: u64 = m.values().sum();
        let tys: Vec<String> = m
            .iter()
            .map(|(ty, b)| format!("{ty}={:.3}GiB", *b as f64 / (1u64 << 30) as f64))
            .collect();
        println!(
            "{c:>10}  {:.3} GiB  [{}]",
            cb as f64 / (1u64 << 30) as f64,
            tys.join(" ")
        );
    }
    Ok(())
}
