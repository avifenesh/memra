use super::*;
use crate::bound_source::draft_pair::{DraftProposalIdentity, PreparedDraftTarget};
use crate::bound_source::head_trim::{HeadChoice, TrimPolicy};
use crate::bound_source::ranks::RankArtifact;

#[test]
fn paired_hf_targets_preserve_tied_and_separate_head_roles() {
    for tied in [false, true] {
        let cfg = QWEN.replacen('{', &format!("{{\"tie_word_embeddings\":{tied},"), 1);
        let f = fixture(&cfg, |_| {});
        let rank_path = f.dir.join("ranks.txt");
        std::fs::write(&rank_path, "3\n1\n").unwrap();
        let ranks = RankArtifact::open(&rank_path).unwrap();
        let raw = SafetensorsSource::open(&f.dir).unwrap();
        assert!(PreparedDraftTarget::bind(&raw).is_err());
        let pair = PreparedModelSource::text(&raw)
            .unwrap()
            .with_runtime(|source| {
                PreparedDraftTarget::bind(source)
                    .unwrap()
                    .prepare_head_trim(&ranks, HeadChoice::ModelOutput, TrimPolicy::Preserve)
            })
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(
            pair.identity().target().opened_source_sha256,
            raw.artifact_sha256().unwrap()
        );
        let DraftProposalIdentity::TargetHeadTrim(trim) = pair.identity().proposal() else {
            panic!("trim role")
        };
        assert_eq!(
            trim.semantic_head(),
            &if tied {
                TensorId::TokenEmbedding
            } else {
                TensorId::OutputProjection
            }
        );
        assert_eq!(trim.as_ref(), pair.materialization().identity());
        assert_eq!(pair.materialization().ids(), &[3, 1]);
        assert!(pair.materialization().from_model_output());
    }
}

#[test]
fn paired_composite_target_keeps_complete_identity_and_selected_overlay_materialization() {
    let cfg = QWEN.replace("32", "64");
    let f = fixture(&cfg, |_| {});
    let rank_path = f.dir.join("ranks.txt");
    std::fs::write(&rank_path, "3\n1\n").unwrap();
    let ranks = RankArtifact::open(&rank_path).unwrap();
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
    let bound =
        crate::bound_source::composite::BoundCompositeSource::compile(&raw, LoadScope::Text)
            .unwrap();
    let identity = bound.artifact_identity().unwrap();
    let runtime = bound.runtime().unwrap();
    let target = PreparedDraftTarget::bind(&runtime).unwrap();
    let pair = target
        .prepare_head_trim(&ranks, HeadChoice::ModelOutput, TrimPolicy::Preserve)
        .unwrap()
        .unwrap();
    assert_eq!(pair.identity().target(), &identity);
    assert_eq!(pair.materialization().bytes(), &payload[..72]);
    assert_eq!(pair.materialization().macro_scale(), 3.0);
    assert!(
        pair.materialization()
            .identity()
            .component_path()
            .is_empty()
    );
    assert_eq!(
        pair.materialization().identity().physical_name(),
        "output.weight"
    );
    assert_ne!(
        pair.identity().target().opened_source_sha256,
        pair.materialization().identity().source_head_sha256()
    );
    // Replacement of the manifest pathname cannot switch either the opened aggregate or head.
    std::fs::rename(
        overlay.join("manifest.json"),
        overlay.join("opened-manifest.json"),
    )
    .unwrap();
    std::fs::write(overlay.join("manifest.json"), b"invalid replacement").unwrap();
    let again = target
        .prepare_head_trim(&ranks, HeadChoice::ModelOutput, TrimPolicy::Preserve)
        .unwrap()
        .unwrap();
    assert_eq!(pair.identity(), again.identity());
    println!("paired-composite-target={:?}", pair.identity());
}
