use super::*;
use crate::bound_source::head_trim::{HeadChoice, PreparedHeadTrim, TrimPolicy};
use crate::bound_source::ranks::RankArtifact;

fn ranks(f: &Fixture) -> RankArtifact {
    let path = f.dir.join("ranks.txt");
    std::fs::write(&path, "3\n1\n").unwrap();
    RankArtifact::open(&path).unwrap()
}

#[test]
fn bound_head_trim_hf_output_and_tied_embedding_use_declared_roles() {
    for tied in [false, true] {
        let cfg = QWEN.replacen('{', &format!("{{\"tie_word_embeddings\":{tied},"), 1);
        let f = fixture(&cfg, |rows| {
            let name = if tied {
                "model.embed_tokens.weight"
            } else {
                "lm_head.weight"
            };
            let t = rows.get_mut(name).unwrap();
            t.bytes = (0..32)
                .flat_map(|row| std::iter::repeat_n((row as f32).to_le_bytes(), 32).flatten())
                .collect();
        });
        let ranks = ranks(&f);
        let raw = SafetensorsSource::open(&f.dir).unwrap();
        assert!(
            PreparedHeadTrim::prepare(&raw, &ranks, HeadChoice::ModelOutput, TrimPolicy::Preserve)
                .is_err()
        );
        let prepared = PreparedModelSource::text(&raw).unwrap();
        let trim = prepared
            .with_runtime(|source| {
                PreparedHeadTrim::prepare(
                    source,
                    &ranks,
                    HeadChoice::FirstMtpOrModel,
                    TrimPolicy::Preserve,
                )
            })
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(
            trim.identity().semantic_head(),
            &if tied {
                TensorId::TokenEmbedding
            } else {
                TensorId::OutputProjection
            }
        );
        assert_eq!(trim.shape(), [32, 2]);
        assert_eq!(trim.macro_scale(), 1.0);
        let values: Vec<f32> = trim
            .bytes()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(values, [vec![3.0; 32], vec![1.0; 32]].concat());
        assert!(
            prepared
                .with_runtime(|source| PreparedHeadTrim::prepare(
                    source,
                    &ranks,
                    HeadChoice::MtpBlock { index: 99 },
                    TrimPolicy::Preserve
                ))
                .unwrap()
                .is_err()
        );
    }
}

#[test]
fn bound_head_trim_composite_uses_selected_component_and_own_macro() {
    let cfg = QWEN.replace("32", "64");
    let f = fixture(&cfg, |_| {});
    let ranks = ranks(&f);
    let overlay = f.dir.join("overlay");
    std::fs::create_dir(&overlay).unwrap();
    let payload = vec![0x22; 64 * 36];
    std::fs::write(overlay.join("head.bin"), &payload).unwrap();
    std::fs::write(overlay.join("scale.bin"), 3.0f32.to_le_bytes()).unwrap();
    let manifest = format!(
        r#"{{"format":"memra-expert-overlay-v2","source_dir":{:?},"tensors":{{"output.weight":{{"file":"head.bin","qtype":"NVFP4","ne":[64,64],"bytes":{}}},"output.scale":{{"file":"scale.bin","qtype":"F32","ne":[1],"bytes":4}}}}}}"#,
        f.dir.to_str().unwrap(),
        payload.len()
    );
    std::fs::write(overlay.join("manifest.json"), manifest).unwrap();
    let raw = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let prepared = PreparedModelSource::text(&raw).unwrap();
    let trim = prepared
        .with_runtime(|source| {
            PreparedHeadTrim::prepare(
                source,
                &ranks,
                HeadChoice::ModelOutput,
                TrimPolicy::Preserve,
            )
        })
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(trim.identity().component_path().is_empty()); // root overlay owns the selected head
    assert_eq!(trim.identity().physical_name(), "output.weight");
    assert_eq!(trim.dtype(), crate::GgmlType::NVFP4);
    assert_eq!(trim.macro_scale(), 3.0);
    assert_eq!(trim.bytes(), &payload[..72]);
}

#[test]
fn bound_head_trim_receipt_covers_rank_order_and_unselected_head_rows() {
    let a = fixture(QWEN, |_| {});
    let r = ranks(&a);
    let raw = SafetensorsSource::open(&a.dir).unwrap();
    let first = PreparedModelSource::text(&raw)
        .unwrap()
        .with_runtime(|s| {
            PreparedHeadTrim::prepare(s, &r, HeadChoice::ModelOutput, TrimPolicy::Preserve)
        })
        .unwrap()
        .unwrap()
        .unwrap();
    let b = fixture(QWEN, |rows| {
        rows.get_mut("lm_head.weight").unwrap().bytes[..4].copy_from_slice(&9.0f32.to_le_bytes());
    });
    let raw = SafetensorsSource::open(&b.dir).unwrap();
    let second = PreparedModelSource::text(&raw)
        .unwrap()
        .with_runtime(|s| {
            PreparedHeadTrim::prepare(s, &r, HeadChoice::ModelOutput, TrimPolicy::Preserve)
        })
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(first.bytes(), second.bytes());
    assert_ne!(
        first.identity().source_head_sha256(),
        second.identity().source_head_sha256()
    );
    assert_ne!(
        first.identity().materialization_sha256(),
        second.identity().materialization_sha256()
    );
    let path = b.dir.join("reordered.txt");
    std::fs::write(&path, "1\n3\n").unwrap();
    let changed = RankArtifact::open(&path).unwrap();
    let third = PreparedModelSource::text(&raw)
        .unwrap()
        .with_runtime(|s| {
            PreparedHeadTrim::prepare(s, &changed, HeadChoice::ModelOutput, TrimPolicy::Preserve)
        })
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(second.bytes(), third.bytes());
    assert_ne!(
        second.identity().rank().order_sha256(),
        third.identity().rank().order_sha256()
    );
    assert_ne!(
        second.identity().materialization_sha256(),
        third.identity().materialization_sha256()
    );
}
