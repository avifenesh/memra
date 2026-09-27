use memra_gguf::{
    GgmlType,
    micro_gguf::{GgufWriter, MetaW},
};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

pub struct Fixture(pub PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

// Independently specified real GGUF matrices for the existing NextN loader. No contract-driven
// shape generation and no config override on an opened artifact.
pub fn make(width: u32, student: bool, scale: Option<GgmlType>, unselected: bool) -> Fixture {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "memra-draft-consumer-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&dir).unwrap();
    let d = u64::from(width);
    let hd = d / 2;
    let inner = if student { d / 2 } else { d };
    let heads = if student { 1 } else { 2 };
    let mut w = GgufWriter::new();
    w.kv("general.architecture", MetaW::Str("qwen3"));
    for (key, value) in [
        ("block_count", 3),
        ("nextn_predict_layers", 1),
        ("embedding_length", width),
        ("attention.head_count", heads),
        ("attention.head_count_kv", 1),
        ("attention.key_length", width / 2),
        ("attention.value_length", width / 2),
        ("feed_forward_length", 64),
        ("vocab_size", 16),
        ("context_length", 128),
    ] {
        w.kv(&format!("qwen3.{key}"), MetaW::U32(value));
    }
    w.kv("qwen3.rope.freq_base", MetaW::F32(10000.0));
    for (name, shape) in [
        ("nextn.enorm.weight", vec![d]),
        ("nextn.hnorm.weight", vec![d]),
        ("nextn.eh_proj.weight", vec![2 * d, inner]),
        ("nextn.shared_head_norm.weight", vec![d]),
        ("attn_norm.weight", vec![inner]),
        ("attn_q.weight", vec![inner, u64::from(heads) * hd]),
        ("attn_k.weight", vec![inner, hd]),
        ("attn_v.weight", vec![inner, hd]),
        ("attn_output.weight", vec![u64::from(heads) * hd, inner]),
        ("attn_q_norm.weight", vec![hd]),
        ("attn_k_norm.weight", vec![hd]),
        ("ffn_norm.weight", vec![inner]),
        ("ffn_gate.weight", vec![inner, 64]),
        ("ffn_up.weight", vec![inner, 64]),
        ("ffn_down.weight", vec![64, inner]),
    ] {
        w.tensor_raw(
            &format!("blk.2.{name}"),
            &shape,
            GgmlType::F32,
            0.25f32
                .to_le_bytes()
                .repeat(shape.iter().product::<u64>() as usize),
        );
    }
    let mut projection = |name: &str, input: u64, output: u64, dtype: Option<GgmlType>| {
        if let Some(dtype) = dtype {
            assert_eq!(input % 64, 0);
            w.tensor_raw(
                name,
                &[input, output],
                GgmlType::NVFP4,
                vec![0x22; (input / 64 * output * 36) as usize],
            );
            let bytes = match dtype {
                GgmlType::F32 => 2.0f32.to_le_bytes().to_vec(),
                GgmlType::F16 | GgmlType::BF16 => 0x4000u16.to_le_bytes().to_vec(),
                _ => unreachable!(),
            };
            w.tensor_raw(&name.replace(".weight", ".scale"), &[1], dtype, bytes);
        } else {
            w.tensor_raw(
                name,
                &[input, output],
                GgmlType::F32,
                0.25f32.to_le_bytes().repeat((input * output) as usize),
            );
        }
    };
    projection(
        "blk.2.nextn.shared_head_head.weight",
        d,
        16,
        if student { None } else { scale },
    );
    if student {
        projection("blk.2.nextn.out_up.weight", inner, d, scale);
    }
    if unselected {
        projection("output.weight", d, 16, Some(GgmlType::BF16));
    }
    w.write(&dir.join("draft.gguf")).unwrap();
    Fixture(dir)
}
