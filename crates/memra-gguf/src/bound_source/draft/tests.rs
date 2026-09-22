use super::*;
use crate::micro_gguf::{GgufWriter, MetaW};
use crate::source::GgufSource;
use crate::{GgmlType, GgufFile};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
#[derive(Clone)]
struct Row {
    shape: Vec<u64>,
    kind: GgmlType,
    bytes: Vec<u8>,
}
fn float(shape: Vec<u64>, value: f32) -> Row {
    Row {
        bytes: (0..shape.iter().product::<u64>())
            .flat_map(|_| value.to_le_bytes())
            .collect(),
        shape,
        kind: GgmlType::F32,
    }
}
fn map(values: &[i64], kind: GgmlType) -> Row {
    Row {
        shape: vec![values.len() as u64],
        kind,
        bytes: values
            .iter()
            .flat_map(|v| {
                if kind == GgmlType::I64 {
                    v.to_le_bytes().to_vec()
                } else {
                    (*v as i32).to_le_bytes().to_vec()
                }
            })
            .collect(),
    }
}
// Explicit GGUF shapes independently describe the existing concat/full-attention/dense draft
// loader. Fixture generation does not call the contract compiler under test.
fn fixture(student: bool, depths: u32, edit: impl FnOnce(&mut BTreeMap<String, Row>)) -> Fixture {
    fixture_for(student, depths, false, edit)
}
fn fixture_for(
    student: bool,
    depths: u32,
    step: bool,
    edit: impl FnOnce(&mut BTreeMap<String, Row>),
) -> Fixture {
    fixture_for_clamps(student, depths, step, None, edit)
}
fn fixture_for_clamps(
    student: bool,
    depths: u32,
    step: bool,
    clamps: Option<Vec<f32>>,
    edit: impl FnOnce(&mut BTreeMap<String, Row>),
) -> Fixture {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "memra-external-draft-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&dir).unwrap();
    let mut rows = BTreeMap::new();
    let (inner, qheads) = if student {
        (16, 1)
    } else if step {
        (32, 4)
    } else {
        (32, 2)
    };
    rows.insert("token_embd.weight".into(), float(vec![32, 16], 0.125));
    rows.insert("output.weight".into(), float(vec![32, 16], 0.25));
    rows.insert("output_norm.weight".into(), float(vec![32], 1.0));
    for depth in 0..depths {
        let n = 2 + depth;
        for (name, shape, value) in [
            ("nextn.enorm.weight", vec![32], 1.0),
            ("nextn.hnorm.weight", vec![32], 1.0),
            ("nextn.eh_proj.weight", vec![64, inner], 0.0625),
            ("nextn.shared_head_norm.weight", vec![32], 1.0),
            (
                "nextn.shared_head_head.weight",
                vec![32, 16],
                0.5 + depth as f32,
            ),
            ("attn_norm.weight", vec![inner], 1.0),
            ("attn_q.weight", vec![inner, qheads * 16], 0.125),
            ("attn_k.weight", vec![inner, 16], 0.125),
            ("attn_v.weight", vec![inner, 16], 0.125),
            ("attn_output.weight", vec![qheads * 16, inner], 0.125),
            ("attn_q_norm.weight", vec![16], 1.0),
            ("attn_k_norm.weight", vec![16], 1.0),
            ("ffn_norm.weight", vec![inner], 1.0),
            ("ffn_gate.weight", vec![inner, 64], 0.125),
            ("ffn_up.weight", vec![inner, 64], 0.125),
            ("ffn_down.weight", vec![64, inner], 0.125),
        ] {
            rows.insert(format!("blk.{n}.{name}"), float(shape, value));
        }
        if step {
            rows.insert(
                format!("blk.{n}.attn_gate.weight"),
                float(vec![32, 4], 0.25),
            );
        }
        if student {
            rows.insert(
                format!("blk.{n}.nextn.out_up.weight"),
                float(vec![inner, 32], 0.125),
            );
        }
    }
    if step {
        rows.insert("rope_freqs.weight".into(), float(vec![8], 1.0));
    }
    edit(&mut rows);
    let mut w = GgufWriter::new();
    let arch = if step { "step35" } else { "qwen3" };
    w.kv("general.architecture", MetaW::Str(arch));
    for (key, value) in [
        ("block_count", 2 + depths),
        ("nextn_predict_layers", depths),
        ("embedding_length", 32),
        ("attention.head_count", qheads as u32),
        ("attention.head_count_kv", 1),
        ("attention.key_length", 16),
        ("attention.value_length", 16),
        ("feed_forward_length", 64),
        ("vocab_size", 16),
        ("context_length", 128),
    ] {
        if step && key == "attention.head_count" {
            let mut heads = vec![2; 2 + depths as usize];
            heads[2..].fill(4);
            w.kv(&format!("{arch}.{key}"), MetaW::ArrU32(heads));
        } else {
            w.kv(&format!("{arch}.{key}"), MetaW::U32(value));
        }
    }
    if step {
        let mut swa = vec![false; 2 + depths as usize];
        swa[2..].fill(true);
        w.kv(
            "step35.attention.sliding_window_pattern",
            MetaW::ArrBool(swa),
        );
        for (key, value) in [
            ("expert_count", 4),
            ("expert_used_count", 2),
            ("expert_feed_forward_length", 16),
            ("expert_shared_feed_forward_length", 16),
            ("expert_gating_func", 2),
            ("leading_dense_block_count", 1),
            ("moe_every_n_layers", 1),
            ("attention.sliding_window", 32),
        ] {
            w.kv(&format!("step35.{key}"), MetaW::U32(value));
        }
        w.kv("step35.expert_weights_scale", MetaW::F32(1.0));
        w.kv("step35.expert_weights_norm", MetaW::Bool(true));
    }
    if let Some(clamps) = clamps {
        w.kv("step35.swiglu_clamp_shexp", MetaW::ArrF32(clamps));
    }
    w.kv(&format!("{arch}.rope.freq_base"), MetaW::F32(10000.0));
    for (name, row) in rows {
        w.tensor_raw(&name, &row.shape, row.kind, row.bytes);
    }
    w.write(&dir.join("draft.gguf")).unwrap();
    Fixture(dir)
}
fn target() -> ModelConfig {
    ModelConfig::from_hf(&crate::config::HfConfig::parse(
        r#"{"model_type":"qwen3","num_hidden_layers":2,"hidden_size":32,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,"intermediate_size":64,"vocab_size":16,"max_position_embeddings":128,"rope_theta":10000.0}"#,
    ))
}
fn err(source: &dyn TensorSource, target: &ModelConfig) -> String {
    match PreparedExternalDraftSource::compile(source, target) {
        Ok(_) => panic!("invalid draft accepted"),
        Err(e) => e,
    }
}

#[test]
fn external_draft_preserves_private_head_bytes_inventory_identity_and_root_separation() {
    let f = fixture(false, 3, |_| {});
    let file = GgufFile::open(f.0.join("draft.gguf")).unwrap();
    let source = GgufSource(&file);
    let source_identity = source.artifact_sha256().unwrap();
    assert!(
        PreparedModelSource::text(&source).is_err(),
        "sparse draft is not a complete model"
    );
    let bound = PreparedExternalDraftSource::compile(&source, &target()).unwrap();
    assert_eq!(bound.blocks().len(), 3);
    assert_eq!(bound.head_name(), "blk.2.nextn.shared_head_head.weight");
    assert_eq!(
        bound.norm_name(),
        Some("blk.2.nextn.shared_head_norm.weight")
    );
    assert_eq!(
        bound.artifact_identity().unwrap().opened_source_sha256,
        source_identity
    );
    bound.with_runtime(|runtime| {
        assert!(PreparedModelSource::text(runtime).err().unwrap().contains("draft bundle"));
        for r in bound.bound.contract.requirements.iter().filter(|r|bound.bound.scope.permits(&r.id)) {
            for name in &r.names {
                if let Some(raw)=source.find(name) {
                    let view=runtime.try_find(name).unwrap().unwrap();
                    assert_eq!(view.ne,raw.ne,"{name}");
                    assert_eq!(view.ggml_type,raw.ggml_type,"{name}");
                    assert_eq!(view.bytes,raw.bytes,"{name}");
                }
            }
        }
        assert!(runtime.try_find("output.weight").is_err(),"shadowed file head is not selected");
        assert!(runtime.try_find("blk.3.nextn.shared_head_head.weight").is_err());
        assert!(runtime.try_find("blk.2.unknown.weight").is_err());
        let charges=runtime.bound_tensor_charges().unwrap().unwrap();
        assert!(charges.iter().any(|c|c.id==TensorId::TokenEmbedding && !c.execution_selected));
        assert!(charges.iter().any(|c|c.id==mtp_id(0,MtpTensor::OutputProjection) && c.execution_selected));
    }).unwrap();
    std::fs::rename(f.0.join("draft.gguf"), f.0.join("opened.gguf")).unwrap();
    std::fs::write(f.0.join("draft.gguf"), b"replacement").unwrap();
    assert_eq!(
        bound.artifact_identity().unwrap().opened_source_sha256,
        source_identity
    );
}

#[test]
fn external_draft_file_head_and_optional_trunk_copies_follow_explicit_ownership() {
    let f = fixture(false, 1, |rows| {
        rows.remove("blk.2.nextn.shared_head_head.weight");
        rows.remove("blk.2.nextn.shared_head_norm.weight");
        rows.remove("token_embd.weight");
    });
    let file = GgufFile::open(f.0.join("draft.gguf")).unwrap();
    let source = GgufSource(&file);
    let bound = PreparedExternalDraftSource::compile(&source, &target()).unwrap();
    assert_eq!(bound.head_name(), "output.weight");
    assert_eq!(bound.norm_name(), Some("output_norm.weight"));
    bound
        .with_runtime(|runtime| {
            assert_eq!(
                runtime.try_find("output.weight").unwrap().unwrap().bytes,
                source.find("output.weight").unwrap().bytes
            )
        })
        .unwrap();
}

#[test]
fn external_draft_student_geometry_and_attached_scales_are_bound_independently() {
    let f = fixture(true, 1, |rows| {
        rows.insert("blk.2.nextn.out_up.scale".into(), float(vec![1], 1.5));
    });
    let file = GgufFile::open(f.0.join("draft.gguf")).unwrap();
    let source = GgufSource(&file);
    let bound = PreparedExternalDraftSource::compile(&source, &target()).unwrap();
    assert_eq!(
        bound.blocks()[0].geometry,
        DraftGeometry::Student {
            hidden_size: 16,
            query_heads: 1,
            kv_heads: 1
        }
    );
    bound
        .with_runtime(|runtime| {
            let fusion = runtime
                .try_find("blk.2.nextn.eh_proj.weight")
                .unwrap()
                .unwrap();
            assert_eq!(fusion.ne, [64, 16]);
            let up = runtime
                .try_find("blk.2.nextn.out_up.weight")
                .unwrap()
                .unwrap();
            assert_eq!(up.ne, [16, 32]);
            assert_eq!(
                runtime
                    .try_find("blk.2.nextn.out_up.scale")
                    .unwrap()
                    .unwrap()
                    .bytes,
                1.5f32.to_le_bytes().as_slice()
            );
            let charges = runtime.bound_tensor_charges().unwrap().unwrap();
            let charged: u64 = charges
                .iter()
                .filter(|c| base_id(&c.id) == &up_id(0) && c.execution_selected)
                .map(|c| c.physical_bytes)
                .sum();
            assert_eq!(charged, 16 * 32 * 4 + 4);
            // Independent scalar matmul chain: concat -> narrow projection -> outer carrier -> head.
            let dot = |view: &TensorView<'_>, input: &[f32]| -> Vec<f32> {
                view.bytes
                    .chunks_exact(input.len() * 4)
                    .map(|row| {
                        row.chunks_exact(4).zip(input).fold(0.0, |s, (w, x)| {
                            s + f32::from_le_bytes(w.try_into().unwrap()) * x
                        })
                    })
                    .collect()
            };
            let inner = dot(&fusion, &[1.0; 64]);
            assert_eq!(inner, vec![4.0; 16]);
            let carrier = dot(&up, &inner);
            assert_eq!(carrier, vec![8.0; 32]);
            let head = runtime.try_find(bound.head_name()).unwrap().unwrap();
            assert_eq!(dot(&head, &carrier), vec![128.0; 16]);
        })
        .unwrap();
}

#[test]
fn external_draft_trimmed_i32_and_i64_maps_keep_token_order_and_identity() {
    for kind in [GgmlType::I32, GgmlType::I64] {
        let f = fixture(false, 1, |rows| {
            rows.remove("blk.2.nextn.shared_head_head.weight");
            rows.insert("output.weight".into(), float(vec![32, 3], 0.25));
            rows.insert("d2t".into(), map(&[11, 2, 9], kind));
        });
        let file = GgufFile::open(f.0.join("draft.gguf")).unwrap();
        let source = GgufSource(&file);
        let bound = PreparedExternalDraftSource::compile(&source, &target()).unwrap();
        assert_eq!(bound.token_map(), Some(&[11, 2, 9][..]));
        assert_eq!(
            bound.vocabulary(),
            &DraftVocabulary::Trimmed {
                target_size: 16,
                token_ids: vec![11, 2, 9]
            }
        );
        bound
            .with_runtime(|runtime| {
                assert_eq!(
                    runtime.try_find(bound.head_name()).unwrap().unwrap().ne,
                    [32, 3]
                )
            })
            .unwrap();
        assert_ne!(
            bound.artifact_identity().unwrap().artifact_sha256,
            source.artifact_sha256().unwrap()
        );
    }
}

struct Spy<'a> {
    source: GgufSource<'a>,
    reads: AtomicUsize,
}
impl TensorSource for Spy<'_> {
    fn config(&self) -> ModelConfig {
        self.source.config()
    }
    fn find(&self, _: &str) -> Option<TensorView<'_>> {
        panic!("legacy name read")
    }
    fn bound_interpretation(&self) -> Result<BoundSourceInterpretation, String> {
        self.source.bound_interpretation()
    }
    fn tensor_census(&self) -> Result<TensorCensus, String> {
        self.source.tensor_census()
    }
    fn validate_bound_metadata(&self, r: &BoundTensorRequest<'_>) -> Result<(), String> {
        self.source.validate_bound_metadata(r)
    }
    fn read_bound(&self, r: &BoundTensorRequest<'_>) -> Result<TensorView<'_>, String> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.source.read_bound(r)
    }
}

#[test]
fn external_draft_bad_census_refuses_before_any_payload_read() {
    for case in [
        "missing",
        "later-missing",
        "shape",
        "integer",
        "extra",
        "alias",
        "head",
        "student-up",
    ] {
        let f = fixture(case == "student-up", 2, |rows| match case {
            "missing" => {
                rows.remove("blk.2.attn_q.weight");
            }
            "later-missing" => {
                rows.remove("blk.3.ffn_up.weight");
            }
            "shape" => {
                rows.insert("blk.2.attn_q.weight".into(), float(vec![16, 64], 0.0));
            }
            "integer" => {
                let r = rows.get_mut("blk.2.attn_q.weight").unwrap();
                r.kind = GgmlType::I32;
            }
            "extra" => {
                rows.insert("not_a_role.weight".into(), float(vec![1], 0.0));
            }
            "alias" => {
                rows.insert(
                    "blk.2.nextn.shared_head.weight".into(),
                    float(vec![32, 16], 0.0),
                );
            }
            "head" => {
                rows.remove("blk.2.nextn.shared_head_head.weight");
                rows.remove("output.weight");
            }
            "student-up" => {
                rows.insert("blk.2.nextn.out_up.weight".into(), float(vec![32, 16], 0.0));
            }
            _ => unreachable!(),
        });
        let file = GgufFile::open(f.0.join("draft.gguf")).unwrap();
        let spy = Spy {
            source: GgufSource(&file),
            reads: AtomicUsize::new(0),
        };
        assert!(!err(&spy, &target()).is_empty(), "{case}");
        assert_eq!(spy.reads.load(Ordering::Relaxed), 0, "{case}");
    }
}

#[test]
fn external_draft_map_refuses_negative_wide_duplicate_and_out_of_range_ids() {
    for values in [[-1, 2, 3], [1i64 << 32 | 1, 2, 3], [2, 2, 3], [16, 2, 3]] {
        let f = fixture(false, 1, |rows| {
            rows.remove("blk.2.nextn.shared_head_head.weight");
            rows.insert("output.weight".into(), float(vec![32, 3], 0.25));
            rows.insert("d2t".into(), map(&values, GgmlType::I64));
        });
        let file = GgufFile::open(f.0.join("draft.gguf")).unwrap();
        assert!(err(&GgufSource(&file), &target()).contains("d2t token"));
    }
}

#[test]
fn external_draft_map_shape_type_and_selected_head_must_agree() {
    for case in ["float", "rank", "count", "untrimmed-private"] {
        let f = fixture(false, 1, |rows| {
            if case != "untrimmed-private" {
                rows.remove("blk.2.nextn.shared_head_head.weight");
            }
            rows.insert("output.weight".into(), float(vec![32, 3], 0.25));
            let mut m = map(&[11, 2, 9], GgmlType::I32);
            match case {
                "float" => m.kind = GgmlType::F32,
                "rank" => m.shape = vec![1, 3],
                "count" => m = map(&[11, 2], GgmlType::I32),
                _ => {}
            }
            rows.insert("d2t".into(), m);
        });
        let file = GgufFile::open(f.0.join("draft.gguf")).unwrap();
        assert!(!err(&GgufSource(&file), &target()).is_empty());
    }
}

#[test]
fn external_draft_target_interface_and_natural_heads_are_checked() {
    let f = fixture(false, 1, |_| {});
    let file = GgufFile::open(f.0.join("draft.gguf")).unwrap();
    let source = GgufSource(&file);
    for field in [
        "hidden", "head_dim", "heads", "kv", "vocab", "rope", "epsilon",
    ] {
        let mut cfg = target();
        match field {
            "hidden" => cfg.n_embd = 64,
            "head_dim" => cfg.head_dim_v = 32,
            "heads" => cfg.n_head = 4,
            "kv" => cfg.n_head_kv = 2,
            "vocab" => cfg.n_vocab = 32,
            "rope" => cfg.rope_freq_base = 50000.0,
            "epsilon" => cfg.rms_eps = 0.01,
            _ => unreachable!(),
        }
        assert!(!err(&source, &cfg).is_empty(), "{field}");
    }
}

#[test]
fn external_draft_integer_weight_formats_do_not_gain_support() {
    for kind in [GgmlType::I8, GgmlType::I16, GgmlType::I32, GgmlType::I64] {
        let f = fixture(false, 1, |rows| {
            let r = rows.get_mut("blk.2.attn_q.weight").unwrap();
            r.kind = kind;
            r.bytes =
                vec![0; (r.shape.iter().product::<u64>() * kind.block_and_type_size().1) as usize];
        });
        let file = GgufFile::open(f.0.join("draft.gguf")).unwrap();
        assert!(err(&GgufSource(&file), &target()).contains("storage mismatch"));
    }
}

#[test]
fn external_draft_step_per_layer_geometry_and_later_heads_remain_distinct() {
    let f = fixture_for(false, 2, true, |_| {});
    let file = GgufFile::open(f.0.join("draft.gguf")).unwrap();
    let source = GgufSource(&file);
    let mut target = source.config();
    target.nextn_predict_layers = 0;
    let bound = PreparedExternalDraftSource::compile(&source, &target).unwrap();
    let AttentionPlan::SlidingWindow { attention, window } =
        &bound.blocks()[0].layer.layer.attention
    else {
        panic!("lost per-layer SWA")
    };
    assert_eq!(*window, 32);
    assert_eq!(attention.query_heads, 4);
    assert_eq!(attention.kv_heads, 1);
    assert_eq!(
        attention.output_gate,
        crate::config::AttentionGateKind::SeparateHead
    );
    bound
        .with_runtime(|runtime| {
            assert_eq!(
                runtime.try_find("blk.2.attn_q.weight").unwrap().unwrap().ne,
                [32, 64]
            );
            assert_eq!(
                runtime
                    .try_find("blk.2.attn_gate.weight")
                    .unwrap()
                    .unwrap()
                    .ne,
                [32, 4]
            );
            assert!(runtime.try_find("blk.3.attn_gate.weight").is_err());
        })
        .unwrap();
}

#[test]
fn external_draft_quantized_head_keeps_encoded_rows_and_macro_scale() {
    let f = fixture(false, 1, |rows| {
        rows.remove("blk.2.nextn.shared_head_head.weight");
        let bytes: Vec<u8> = (0..3)
            .flat_map(|i| std::iter::once(0x3cu8).chain(std::iter::repeat_n(i, 33)))
            .collect();
        rows.insert(
            "output.weight".into(),
            Row {
                shape: vec![32, 3],
                kind: GgmlType::Q8_0,
                bytes,
            },
        );
        rows.insert("output.scale".into(), float(vec![1], 2.0));
        rows.insert("d2t".into(), map(&[9, 1, 4], GgmlType::I32));
    });
    let file = GgufFile::open(f.0.join("draft.gguf")).unwrap();
    let source = GgufSource(&file);
    let bound = PreparedExternalDraftSource::compile(&source, &target()).unwrap();
    bound
        .with_runtime(|runtime| {
            let head = runtime.try_find("output.weight").unwrap().unwrap();
            assert_eq!(head.ggml_type, GgmlType::Q8_0);
            assert_eq!(head.bytes, source.find("output.weight").unwrap().bytes);
            assert_eq!(
                runtime.try_find("output.scale").unwrap().unwrap().bytes,
                2.0f32.to_le_bytes().as_slice()
            );
            let charges = runtime.bound_tensor_charges().unwrap().unwrap();
            assert_eq!(
                charges
                    .iter()
                    .filter(
                        |c| base_id(&c.id) == &TensorId::OutputProjection && c.execution_selected
                    )
                    .map(|c| c.physical_bytes)
                    .sum::<u64>(),
                3 * 34 + 4
            );
        })
        .unwrap();
}

#[test]
fn external_draft_map_order_and_unselected_bytes_both_bind_identity() {
    let build = |ids: &[i64], unused: f32| {
        fixture(false, 1, |rows| {
            rows.remove("blk.2.nextn.shared_head_head.weight");
            rows.insert("output.weight".into(), float(vec![32, 3], 0.25));
            rows.insert("token_embd.weight".into(), float(vec![32, 16], unused));
            rows.insert("d2t".into(), map(ids, GgmlType::I32));
        })
    };
    let fixtures = [
        build(&[11, 2, 9], 0.125),
        build(&[2, 11, 9], 0.125),
        build(&[11, 2, 9], 0.75),
    ];
    let identities: Vec<_> = fixtures
        .iter()
        .map(|f| {
            let file = GgufFile::open(f.0.join("draft.gguf")).unwrap();
            let source = GgufSource(&file);
            PreparedExternalDraftSource::compile(&source, &target())
                .unwrap()
                .artifact_identity()
                .unwrap()
        })
        .collect();
    assert_ne!(identities[0].artifact_sha256, identities[1].artifact_sha256);
    assert_ne!(
        identities[0].semantic_scope_sha256,
        identities[1].semantic_scope_sha256
    );
    assert_ne!(identities[0].artifact_sha256, identities[2].artifact_sha256);
    assert_eq!(
        identities[0].semantic_scope_sha256,
        identities[2].semantic_scope_sha256
    );
}

#[test]
fn external_draft_activation_refuses_target_oai_before_payload_reads() {
    let f = fixture(false, 1, |_| {});
    let file = GgufFile::open(f.0.join("draft.gguf")).unwrap();
    let source = GgufSource(&file);
    PreparedExternalDraftSource::compile(&source, &target()).unwrap();
    for (alpha, limit) in [(1.702, 7.0), (1.5, 7.0), (1.702, 6.0)] {
        let cfg = ModelConfig::from_hf(
            &crate::config::HfConfig::try_parse(&format!(
                r#"{{
            "model_type":"minimax_m3","num_hidden_layers":2,"hidden_size":32,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,
            "intermediate_size":64,"vocab_size":16,"max_position_embeddings":128,
            "rms_norm_eps":0.000001,"rope_theta":10000.0,"num_local_experts":4,
            "num_experts_per_tok":2,"dense_intermediate_size":64,"shared_intermediate_size":64,
            "moe_layer_freq":[0,0],"swiglu_alpha":{alpha},"swiglu_limit":{limit},
            "rotary_dim":16,"use_gemma_norm":false}}"#
            ))
            .unwrap(),
        );
        let spy = Spy {
            source: GgufSource(&file),
            reads: AtomicUsize::new(0),
        };
        assert!(err(&spy, &cfg).contains("FFN declares Silu"));
        assert_eq!(spy.reads.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn external_draft_step_activation_keeps_source_owned_per_layer_limits() {
    let f = fixture_for_clamps(false, 2, true, Some(vec![0.0, 0.0, 2.5, 6.5]), |_| {});
    let file = GgufFile::open(f.0.join("draft.gguf")).unwrap();
    let source = GgufSource(&file);
    let mut target = source.config();
    target.nextn_predict_layers = 0;
    target.step35.as_mut().unwrap().swiglu_clamp_shexp.fill(9.0);
    let bound = PreparedExternalDraftSource::compile(&source, &target).unwrap();
    for (block, expected) in bound.blocks().iter().zip([2.5, 6.5]) {
        let MlpPlan::Dense(dense) = &block.layer.layer.mlp else {
            panic!("expected dense Step MTP")
        };
        let clamp = step_mtp_clamp(&dense.activation).unwrap();
        assert_eq!(clamp, expected);
        FfnActivation::for_target(&target, Some(crate::config::SwigluClamp::Post(clamp)))
            .validate_declared(&dense.activation)
            .unwrap();
    }
}
