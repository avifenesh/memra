use memra_gguf::{
    GgmlType,
    micro_gguf::{GgufWriter, MetaW},
};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
pub struct Fixture {
    pub dir: PathBuf,
    pub heads: BTreeMap<String, Vec<u8>>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.dir).unwrap();
    }
}

pub fn make(
    dtype: GgmlType,
    width: u32,
    tied: bool,
    missing: Option<u32>,
    scale_dtype: GgmlType,
) -> Fixture {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "memra-head-trim-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&dir).unwrap();
    let mut heads = BTreeMap::new();
    let mut w = GgufWriter::new();
    w.kv("general.architecture", MetaW::Str("qwen3"));
    w.kv("qwen3.tie_word_embeddings", MetaW::Bool(tied));
    for (key, value) in [
        ("block_count", 3),
        ("nextn_predict_layers", 2),
        ("embedding_length", width),
        ("attention.head_count", 2),
        ("attention.head_count_kv", 1),
        ("attention.key_length", width / 2),
        ("attention.value_length", width / 2),
        ("feed_forward_length", 64),
        ("vocab_size", 4),
        ("context_length", 128),
    ] {
        w.kv(&format!("qwen3.{key}"), MetaW::U32(value));
    }
    w.kv("qwen3.rope.freq_base", MetaW::F32(10000.0));
    let d = u64::from(width);
    let hd = d / 2;
    let float = |w: &mut GgufWriter, name: &str, ne: &[u64]| {
        w.tensor_raw(
            name,
            ne,
            GgmlType::F32,
            0.25f32
                .to_le_bytes()
                .repeat(ne.iter().product::<u64>() as usize),
        )
    };
    float(&mut w, "output_norm.weight", &[d]);
    if !tied {
        float(&mut w, "token_embd.weight", &[d, 4]);
    }
    for index in 0..3 {
        for (name, shape) in [
            ("attn_norm.weight", vec![d]),
            ("attn_q.weight", vec![d, d]),
            ("attn_k.weight", vec![d, hd]),
            ("attn_v.weight", vec![d, hd]),
            ("attn_output.weight", vec![d, d]),
            ("attn_q_norm.weight", vec![hd]),
            ("attn_k_norm.weight", vec![hd]),
            ("ffn_norm.weight", vec![d]),
            ("ffn_gate.weight", vec![d, 64]),
            ("ffn_up.weight", vec![d, 64]),
            ("ffn_down.weight", vec![64, d]),
        ] {
            float(&mut w, &format!("blk.{index}.{name}"), &shape);
        }
        if index > 0 {
            for (name, shape) in [
                ("nextn.enorm.weight", vec![d]),
                ("nextn.hnorm.weight", vec![d]),
                ("nextn.eh_proj.weight", vec![2 * d, d]),
                ("nextn.shared_head_norm.weight", vec![d]),
            ] {
                float(&mut w, &format!("blk.{index}.{name}"), &shape);
            }
        }
    }
    for owner in 0..3u32 {
        if owner > 0 && missing == Some(owner) {
            continue;
        }
        let name = if owner == 0 {
            if tied {
                "token_embd.weight".into()
            } else {
                "output.weight".into()
            }
        } else {
            format!("blk.{owner}.nextn.shared_head_head.weight")
        };
        let (block, size) = dtype.block_and_type_size();
        let mut bytes = Vec::new();
        if d.is_multiple_of(block) {
            for row in 0..4u32 {
                match dtype {
                    GgmlType::F32 => bytes.extend(
                        (6.0f32 * 2f32.powi((row + owner) as i32))
                            .to_le_bytes()
                            .repeat(width as usize),
                    ),
                    GgmlType::BF16 => bytes.extend(
                        ((6.0f32 * 2f32.powi((row + owner) as i32)).to_bits() as u64 >> 16)
                            .to_le_bytes()[..2]
                            .repeat(width as usize),
                    ),
                    GgmlType::F16 => bytes.extend(0x3c00u16.to_le_bytes().repeat(width as usize)),
                    GgmlType::NVFP4 => {
                        for _ in 0..d / 64 {
                            bytes.extend([0x38; 4]);
                            bytes.extend([(((owner * 4 + row) % 7 + 1) * 17) as u8; 32]);
                        }
                    }
                    GgmlType::Q8_0 => {
                        for _ in 0..d / 32 {
                            bytes.extend(0x3c00u16.to_le_bytes());
                            bytes.extend([(owner * 4 + row + 1) as u8; 32]);
                        }
                    }
                    _ => bytes.extend(vec![0x11; (d / block * size) as usize]),
                }
            }
        } else {
            bytes = vec![0x11; (d * 4 / block * size) as usize];
        }
        w.tensor_raw(&name, &[d, 4], dtype, bytes.clone());
        heads.insert(name.clone(), bytes);
        if dtype == GgmlType::NVFP4 {
            let scale = [2.0f32, 3.0, 5.0][owner as usize];
            let b = if scale_dtype == GgmlType::F32 {
                scale.to_le_bytes().to_vec()
            } else {
                0x4000u16.to_le_bytes().to_vec()
            };
            w.tensor_raw(&name.replace(".weight", ".scale"), &[1], scale_dtype, b);
        }
    }
    w.write(&dir.join("model.gguf")).unwrap();
    std::fs::write(dir.join("ranks.txt"), "3\n1\n").unwrap();
    Fixture { dir, heads }
}
